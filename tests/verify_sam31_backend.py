"""GPU checks for the SAM 3.1 Segmenter backend (bat_sam31_runtime / bat_sam31_segmenter).

Runs the real model through the node and the live-route functions, outside
ComfyUI's server. Needs sam3.1_multiplex_fp16.safetensors in models/checkpoints
and ~4 GB of free VRAM; skips otherwise.

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_sam31_backend.py [FRAMES_DIR]

Without FRAMES_DIR only the synthetic click-mode checks run. With a folder of
PNG frames showing several people (a crowded live-action clip was used to build
this), text mode is checked too.
"""
import glob
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PACK))
frames_dir = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = [sys.argv[0]]
sys.path.insert(0, COMFY)
os.chdir(COMFY)

import numpy as np
import torch
from PIL import Image

FAILURES = []


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}" + ("" if condition or not detail else f"\n         {detail}"))
    if not condition:
        FAILURES.append(label)


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else 1.0


# Load the pack's modules without its __init__ (which needs a running server).
pkg = types.ModuleType("batpack")
pkg.__path__ = [PACK]
sys.modules["batpack"] = pkg
from batpack import bat_sam31_runtime as rt          # noqa: E402
from batpack import bat_sam31_segmenter as seg       # noqa: E402

if not rt.list_models():
    sys.exit("SKIP: no SAM 3 checkpoint in models/checkpoints")
free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
if free < 4:
    sys.exit(f"SKIP: only {free:.1f} GB VRAM free")

print("parse_state")
st = seg.parse_state("not json")
check("junk -> empty", st == {"points": [], "box": None, "instances": []})
st = seg.parse_state(json.dumps({"points": [{"x": 2, "y": -1, "label": 0}, {"x": "a"}], "box": [0.1, 0.1, 0.1, 0.5]}))
check("points clamped, bad point dropped", st["points"] == [{"x": 1.0, "y": 0.0, "label": 0}])
check("degenerate box dropped", st["box"] is None)

print("match_instances")
found = [{"box": [0, 0, .2, .2]}, {"box": [.5, .5, .7, .7]}, {"box": [.8, .8, .9, .9]}]
saved = [{"box": [0.01, 0, .2, .21], "keep": False}, {"box": [.5, .5, .7, .7], "keep": True}]
check("by IoU, unmatched kept", rt.match_instances(found, saved) == [False, True, True])

# ── synthetic clip: a disc moving left -> right over noise ──────────────────
N, H, W = 24, 360, 640
rng = np.random.default_rng(0)
clip = rng.uniform(0.2, 0.4, (N, H, W, 3)).astype(np.float32)
truth = np.zeros((N, H, W), bool)
yy, xx = np.mgrid[0:H, 0:W]
for i in range(N):
    cx, cy = 120 + i * 16, 180
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < 55 ** 2
    clip[i][disc] = [0.9, 0.2, 0.15]
    truth[i] = disc
frames = torch.from_numpy(clip)
node = seg.BatSam31Segmenter()
k = 12
pt = {"x": (120 + k * 16) / W, "y": 180 / H, "label": 1}

print("click mode, bidirectional from frame 12")
t = time.time()
out = node.segment(frames, "click", "", k, False, prompt_state=json.dumps({"points": [pt]}))
masks, preview = out["result"]
print(f"     ({time.time() - t:.1f}s incl. load)")
ious = [iou(masks[i].numpy(), truth[i]) for i in range(N)]
check("every frame tracked (IoU > 0.85)", min(ious) > 0.85, f"min IoU {min(ious):.3f} at {int(np.argmin(ious))}")
check("preview is an IMAGE batch", tuple(preview.shape) == (N, H, W, 3))
token = out["ui"]["token"][0]

print("direction semantics")
for direction, empty in (("forward", range(0, k)), ("backward", range(k + 1, N))):
    adv = dict(seg.DEFAULTS, tracking_direction=direction)
    m = node.segment(frames, "click", "", k, False, prompt_state=json.dumps({"points": [pt]}),
                     advanced=adv)["result"][0]
    check(f"{direction}: other side empty", all(m[i].sum() == 0 for i in empty))
    check(f"{direction}: annotation frame filled", iou(m[k].numpy(), truth[k]) > 0.85)
adv = dict(seg.DEFAULTS, max_frames_to_track=3)
m = node.segment(frames, "click", "", k, False, prompt_state=json.dumps({"points": [pt]}), advanced=adv)["result"][0]
check("max_frames_to_track=3 -> frames 9..15 only",
      [i for i in range(N) if m[i].sum() > 0] == list(range(k - 3, k + 4)))

print("input_mask seed")
m = node.segment(frames, "click", "", k, False, prompt_state="",
                 input_mask=torch.from_numpy(truth.astype(np.float32)))["result"][0]
check("seeded from input_mask", iou(m[N - 1].numpy(), truth[N - 1]) > 0.85)
# No prompt is not an error: the run loads the clip (that is how the editor
# gets its first picture) and returns an empty mask with a note.
out0 = node.segment(frames, "click", "", k, False, prompt_state="")
check("no prompt -> empty mask, clip registered, note",
      out0["result"][0].sum() == 0 and out0["ui"].get("note") and out0["ui"]["token"][0])
check("...and its frames are served for the plate",
      seg.run_frame(out0["ui"]["token"][0], 0, "plate", 128)[1] == "image/jpeg")
out0 = node.segment(frames, "text", "  ", k, False, prompt_state="")
check("text mode with no text -> same", out0["result"][0].sum() == 0 and "text_prompt" in out0["ui"]["note"][0])
try:
    node.segment(frames, "click", "", N, False, prompt_state=json.dumps({"points": [pt]}))
    check("out-of-range frame -> error", False)
except ValueError as e:
    check("out-of-range frame -> error", "out of range" in str(e))

print("live preview")
body = {"source": {"token": token, "frame": k}, "points": [pt], "box": None}
payload, status = seg.live_mask(body)
check("live mask from run token", status == 200 and payload.get("mask"), str(payload)[:200])
payload, status = seg.live_mask(body)
check(f"cached-encode click fast ({payload.get('ms')} ms decode, {payload.get('encode_ms')} ms encode)",
      payload.get("ms", 1e9) < 40 and payload.get("encode_ms", 1e9) < 5)
live = np.asarray(Image.open(__import__("io").BytesIO(__import__("base64").b64decode(payload["mask"])))) > 127
check("live preview == tracked mask on the annotation frame", iou(live, masks[k].numpy()) > 0.95,
      f"IoU {iou(live, masks[k].numpy()):.3f}")
import base64, io  # noqa: E401,E402
buf = io.BytesIO()
Image.fromarray((clip[k] * 255).astype(np.uint8)).save(buf, format="PNG")
payload, status = seg.live_mask({"source": {"image": base64.b64encode(buf.getvalue()).decode()},
                                 "points": [pt]})
check("live mask from an uploaded plate", status == 200 and payload.get("mask"))
payload, status = seg.live_mask({"source": {"token": "nope"}, "points": [pt]})
check("unknown source -> 409", status == 409)

print("run frames route")
data, ctype = seg.run_frame(token, 3, "plate", 256)
check("plate JPEG", ctype == "image/jpeg" and max(Image.open(io.BytesIO(data)).size) == 256)
data, ctype = seg.run_frame(token, 3, "mask", 4096)
check("mask PNG matches run", ctype == "image/png"
      and iou(np.asarray(Image.open(io.BytesIO(data))) > 127, masks[3].numpy()) > 0.99)

# ── text mode on real footage ───────────────────────────────────────────────
if frames_dir:
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))[:40]
    real = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])).float() / 255
    RN, RH, RW = real.shape[:3]
    print(f"text mode on {RN} real frames {RW}x{RH}")
    buf = io.BytesIO()
    Image.fromarray((real[0].numpy() * 255).astype(np.uint8)).save(buf, format="JPEG", quality=90)
    t = time.time()
    payload, _ = seg.live_detect({"source": {"image": base64.b64encode(buf.getvalue()).decode()},
                                  "text": "person"})
    inst = payload.get("instances", [])
    print(f"     {len(inst)} people in {time.time() - t:.2f}s")
    check("several people found", len(inst) >= 3)
    keep_one = [{"box": inst[0]["box"], "keep": True}] + [{"box": i["box"], "keep": False} for i in inst[1:]]
    st = json.dumps({"instances": keep_one})
    one = node.segment(real, "text", "person", 0, False, prompt_state=st)["result"][0]
    allp = node.segment(real, "text", "person", 0, False, prompt_state="")["result"][0]
    check("keeping 1 of N -> smaller mask than all", one[0].sum() < 0.7 * allp[0].sum(),
          f"{one[0].sum():.0f} vs {allp[0].sum():.0f}")
    redet = node.segment(real, "text", "person", 0, True, prompt_state=st)["result"][0]
    dropped = np.asarray(Image.open(io.BytesIO(base64.b64decode(inst[1]["mask"]))).resize((RW, RH))) > 127
    check("re-detect keeps dropped people out (frame 0)",
          (redet[0].numpy() > 0.5)[dropped].mean() < 0.2,
          f"{(redet[0].numpy() > 0.5)[dropped].mean():.2f} of a dropped person covered")
    check("re-detect keeps dropped people out (frame 10)",
          (redet[10].numpy() > 0.5)[dropped].mean() < 0.3)

print()
print("All checks passed." if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
