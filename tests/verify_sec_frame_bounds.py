"""Prove the SeC segmenter's frame-index mapping agrees with the loaders.

`frame_index_select` indexes the frame BATCH the node receives, and
`bat_sec_runtime.segment` RAISES on an out-of-range value rather than clamping
— so the front-end bounds the widget instead of letting the artist discover it
several seconds into a failed job. Two JS functions do the work and they have
to be exact inverses of what the loaders actually emit:

  sourceBatchLength(loaderNode, fileFrames)  -> how many frames arrive
  sourceFrameIndex(loaderNode, batchIdx)     -> which file frame that is

This replays both against the loaders' real selection semantics, transcribed
from the Python:

  Bat_VideoLoader.load()  — start/end clamped into the clip, end<0 means the
                            last frame, then range(start, end+1, stride).
  VoltLoader._load_standard/_load_exr — frames[skip::step][:cap], cap 0 = all.
  Bat_FramePicker         — exactly one frame, the one at frame_index.

An off-by-one either way is a real bug: too small a max blocks a valid pick,
too large lets the job fail at execute time, and a wrong sourceFrameIndex
previews the wrong plate while segmenting the right one (or vice versa).

There is no node binary on this box, so the two functions are pulled out of
`web/bat_sec_segmenter.js` and driven under quickjs (see the js-verification
note). Run with the RND venv python:

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_sec_frame_bounds.py
"""
import json
import os
import re
import sys

import quickjs

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
JS = os.path.join(PACK, "web", "bat_sec_segmenter.js")

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}" + (f"\n         {detail}" if detail else ""))
        FAILURES.append(label)


def eq(label, got, want):
    check(label, got == want, f"got {got!r}\n         want {want!r}")


# ── Pull the pure functions out of the extension ────────────────────────────
# They sit between `function widgetNum` and the "per-node preview state"
# banner, and depend on nothing but their arguments — no app, no api, no DOM.
src = open(JS, encoding="utf-8").read()
start = src.index("function widgetNum(")
end = src.index("// ── per-node preview state")
pure = src[start:end]
for name in ("widgetNum", "sourceFrameIndex", "sourceBatchLength"):
    if f"function {name}(" not in pure:
        sys.exit(f"could not extract {name} from {JS}")

ctx = quickjs.Context()
ctx.eval(pure + """
function _node(type, widgets) {
    return { type: type,
             widgets: Object.keys(widgets).map(function (k) {
                 return { name: k, value: widgets[k] };
             }) };
}
function batchLength(type, widgets, fileFrames) {
    return sourceBatchLength(_node(type, widgets), fileFrames);
}
function frameIndex(type, widgets, batchIdx, fileFrames) {
    return sourceFrameIndex(_node(type, widgets), batchIdx, fileFrames || 0);
}
function frameIndices(type, widgets, fileFrames) {
    var n = sourceBatchLength(_node(type, widgets), fileFrames);
    var out = [];
    for (var i = 0; i < n; i++) {
        out.push(sourceFrameIndex(_node(type, widgets), i, fileFrames));
    }
    return JSON.stringify(out);
}
""")


def batch_length(type_, widgets, file_frames):
    return ctx.eval(f"batchLength({json.dumps(type_)}, {json.dumps(widgets)}, {file_frames})")


def frame_index(type_, widgets, batch_idx, file_frames=0):
    return ctx.eval(f"frameIndex({json.dumps(type_)}, {json.dumps(widgets)}, "
                    f"{batch_idx}, {file_frames})")


def frame_indices(type_, widgets, file_frames):
    return json.loads(ctx.eval(
        f"frameIndices({json.dumps(type_)}, {json.dumps(widgets)}, {file_frames})"))


# ── Reference implementations, transcribed from the Python ──────────────────

def py_video_loader(n, start_frame=0, end_frame=-1, select_every_nth=1):
    """Bat_VideoLoader.load() — bat_video_loader.py."""
    if n <= 0:
        return []
    start = max(0, min(int(start_frame), n - 1))
    end = int(end_frame)
    if end < 0 or end >= n:
        end = n - 1
    if end < start:
        end = start
    stride = max(1, int(select_every_nth))
    return list(range(start, end + 1, stride))


def py_volt_loader(n, skip_first_frames=0, select_every_nth=1, frame_load_cap=0):
    """VoltLoader._load_standard / _load_exr — frames[skip::step][:cap]."""
    frames = list(range(n))
    selected = frames[max(0, int(skip_first_frames))::max(1, int(select_every_nth))]
    if int(frame_load_cap) > 0:
        selected = selected[:int(frame_load_cap)]
    return selected


def py_frame_picker(n, frame_index=0):
    """Bat_FramePicker — one frame."""
    return [max(0, min(int(frame_index), max(0, n - 1)))] if n else []


# ── 1. Bat_VideoLoader ──────────────────────────────────────────────────────
print("\n1. Bat_VideoLoader — length and index agree with load()")
CASES_VIDEO = [
    # (file frames, widgets)
    (100, {}),                                                     # untrimmed
    (100, {"start_frame": 12}),                                    # head trim
    (100, {"end_frame": 50}),                                      # tail trim
    (100, {"start_frame": 12, "end_frame": 50}),                   # both
    (100, {"select_every_nth": 3}),                                # stride
    (100, {"start_frame": 12, "end_frame": 50, "select_every_nth": 2}),
    (100, {"start_frame": 7,  "end_frame": 50, "select_every_nth": 7}),  # ragged tail
    (100, {"end_frame": 500}),                                     # end past the clip
    (100, {"start_frame": 500}),                                   # start past the clip
    (100, {"start_frame": 60, "end_frame": 20}),                   # inverted
    (1,   {}),                                                     # single-frame clip
    (100, {"select_every_nth": 0}),                                # bad stride
    (100, {"start_frame": -5}),                                    # negative start
]
for n, w in CASES_VIDEO:
    want = py_video_loader(n, **w)
    eq(f"length  n={n} {w}", batch_length("Bat_VideoLoader", w, n), len(want))
    eq(f"indices n={n} {w}", frame_indices("Bat_VideoLoader", w, n), want)


# ── 2. VoltLoader ───────────────────────────────────────────────────────────
print("\n2. VoltLoader — length and index agree with frames[skip::step][:cap]")
CASES_VOLT = [
    (100, {}),
    (100, {"skip_first_frames": 12}),
    (100, {"select_every_nth": 4}),
    (100, {"frame_load_cap": 30}),
    (100, {"skip_first_frames": 12, "select_every_nth": 4}),
    (100, {"skip_first_frames": 12, "select_every_nth": 4, "frame_load_cap": 10}),
    (100, {"skip_first_frames": 12, "select_every_nth": 4, "frame_load_cap": 999}),
    (100, {"skip_first_frames": 100}),          # skips the whole clip
    (100, {"skip_first_frames": 99}),           # skips all but one
    (100, {"select_every_nth": 7}),             # ragged tail
    (1,   {}),
    (100, {"frame_load_cap": 0}),               # 0 = no cap
]
for n, w in CASES_VOLT:
    want = py_volt_loader(n, **w)
    eq(f"length  n={n} {w}", batch_length("VoltLoader", w, n), len(want))
    eq(f"indices n={n} {w}", frame_indices("VoltLoader", w, n), want)


# ── 3. Bat_FramePicker ──────────────────────────────────────────────────────
print("\n3. Bat_FramePicker — always one frame, the batch index carries nothing")
for n, w in [(100, {"frame_index": 0}), (100, {"frame_index": 42}), (1, {"frame_index": 0})]:
    eq(f"length n={n} {w}", batch_length("Bat_FramePicker", w, n), 1)
    # The batch only ever has index 0, and it must resolve to the PICKED file
    # frame — using the batch index here would always preview file frame 0.
    eq(f"index  n={n} {w}", frame_index("Bat_FramePicker", w, 0),
       py_frame_picker(n, **w)[0])
check("a non-zero batch index still resolves to the picked frame",
      frame_index("Bat_FramePicker", {"frame_index": 42}, 3) == 42)


# ── 4. Unknown sources stay unbounded ───────────────────────────────────────
print("\n4. an unknown loader is passed through, not guessed at")
eq("length is the raw file count", batch_length("LoadImage", {}, 100), 100)
eq("index is the batch index", frame_index("LoadImage", {}, 7), 7)
eq("zero frames means zero", batch_length("Bat_VideoLoader", {}, 0), 0)
eq("zero frames means zero (Volt)", batch_length("VoltLoader", {}, 0), 0)


# ── 5. The invariant that actually matters ──────────────────────────────────
print("\n5. every in-bounds batch index maps to a real file frame")
# This is the whole point: if the widget's max is length-1, then every value
# the artist can now select must resolve to a frame that exists in the file.
# A violation is either a failed job or the wrong plate.
for label, kind, cases, ref in (
        ("Bat_VideoLoader", "Bat_VideoLoader", CASES_VIDEO, py_video_loader),
        ("VoltLoader", "VoltLoader", CASES_VOLT, py_volt_loader)):
    bad = []
    for n, w in cases:
        length = batch_length(kind, w, n)
        want = ref(n, **w)
        for i in range(length):
            got = frame_index(kind, w, i, n)
            if not (0 <= got < n) or got != want[i]:
                bad.append((n, w, i, got, want[i] if i < len(want) else None))
    check(f"{label}: all indices in range and correct",
          not bad, f"{len(bad)} bad mapping(s), first: {bad[0] if bad else ''}")


print("\n5b. the clip length is what lets the mapping clamp")
# sourceFrameIndex can't clamp without knowing how long the clip is, so the
# extension threads the cached count through. Prove both readings.
eq("start past the clip, length known -> last frame",
   frame_index("Bat_VideoLoader", {"start_frame": 500}, 0, 100), 99)
eq("start past the clip, length unknown -> unclamped (route 404s, we fall back)",
   frame_index("Bat_VideoLoader", {"start_frame": 500}, 0, 0), 500)
eq("in-clip start is unaffected by the length",
   frame_index("Bat_VideoLoader", {"start_frame": 12}, 3, 100),
   frame_index("Bat_VideoLoader", {"start_frame": 12}, 3, 0))


# ── 6. Clamp arithmetic ─────────────────────────────────────────────────────
print("\n6. the widget clamp only ever narrows into a valid range")
ctx.eval("""
function clamp(current, count) {
    var max = count - 1;
    if (current > max || current < 0) return Math.min(Math.max(0, current), max);
    return current;   // untouched
}
""")
for current, count, want in [
        (0, 100, 0), (99, 100, 99), (150, 100, 99), (-3, 100, 0),
        (0, 1, 0), (7, 1, 0), (11, 100, 11)]:
    eq(f"clamp({current}, count={count})", ctx.eval(f"clamp({current}, {count})"), want)


print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
