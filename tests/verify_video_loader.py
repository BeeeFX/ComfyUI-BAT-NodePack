#!/usr/bin/env python
"""
Checks for Bat_VideoLoader.

Four claims here are the kind that fail silently — nothing throws, the node
just quietly serves the wrong thing — so they get a test rather than a comment:

1. **A stale load is a wrong load.** `IS_CHANGED` has to strip the path the
   same way `load()` does (a dragged-in path arrives quoted) and has to notice
   a rewrite that lands on the same whole second. Get either wrong and the node
   returns pre-rewrite frames forever, with no error anywhere.

2. **The uint8 -> float32 scale is exact.** The conversion writes into a
   preallocated buffer to keep the peak down; that optimisation is only worth
   anything if it is bit-identical to the obvious `astype(float32) / 255`.

3. **The preview routes only serve video.** `/bat/video-stream` hands its path
   to FileResponse, so the extension allowlist is the whole defence against it
   reading arbitrary files. This pack ships publicly, so it has to hold on a
   default install, with no env var set.

4. **The trim maths.** end_frame = -1 means "to the end", and
   select_every_nth counts from start_frame.

    python tests/verify_video_loader.py
"""

import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Harness: the module registers aiohttp routes on import, so stub `server`
# before loading it. The stubs also capture the handlers so the route-level
# gating can be driven directly below.
# ---------------------------------------------------------------------------

ROUTES = {}


class _Resp:
    def __init__(self, status=200, text=None, body=None, content_type=None):
        self.status = status
        self.text = text
        self.body = body
        self.content_type = content_type


class _JsonResp:
    def __init__(self, payload):
        self.payload = payload
        self.status = 200


class _FileResp:
    def __init__(self, path):
        self.path = path
        self.status = 200


def _install_server_stub():
    web = types.SimpleNamespace(
        Response=_Resp, json_response=_JsonResp, FileResponse=_FileResp,
    )

    class _Routes:
        def get(self, path):
            def deco(fn):
                ROUTES[path] = fn
                return fn
            return deco

    mod = types.ModuleType("server")
    mod.web = web
    mod.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_Routes())
    )
    sys.modules["server"] = mod


_install_server_stub()

spec = importlib.util.spec_from_file_location(
    "bat_video_loader", os.path.join(PACK, "bat_video_loader.py"))
vl = importlib.util.module_from_spec(spec)
sys.modules["bat_video_loader"] = vl
spec.loader.exec_module(vl)


def call_route(route, **query):
    req = types.SimpleNamespace(rel_url=types.SimpleNamespace(query=query))
    return asyncio.run(ROUTES[route](req))


# ---------------------------------------------------------------------------
# A test clip. Lossless FFV1 so nothing here depends on codec rounding.
# ---------------------------------------------------------------------------

FRAMES, W, H = 12, 64, 48


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def make_clip(directory, name="clip.mkv", frames=FRAMES, shade=0,
              extra=None, alpha=False):
    ffmpeg = _ffmpeg()
    path = os.path.join(directory, name)
    # Each frame a flat, distinct grey so a decoded frame identifies itself,
    # with an off-grid value in the low bits so a 10-bit source has something
    # an 8-bit decode could not represent.
    channels = 4 if alpha else 3
    raw = np.zeros((frames, H, W, channels), dtype=np.uint16)
    for i in range(frames):
        raw[i, :, :, :3] = (2600 + shade * 257 + i * 5100) % 65536
    if alpha:
        # Transparent down the left half, opaque down the right, so a mask can
        # be told apart from its own inverse.
        raw[:, :, :W // 2, 3] = 0
        raw[:, :, W // 2:, 3] = 65535
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo",
         "-pix_fmt", "rgba64le" if alpha else "rgb48le",
         "-s", f"{W}x{H}", "-r", "24", "-i", "-"]
        + (extra or ["-c:v", "ffv1", "-pix_fmt", "bgr0"]) + [path],
        input=raw.tobytes(), capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(path):
        return None, proc.stderr.decode("utf-8", "replace")
    return path, None


def make_clip_with_audio(directory, name="withaudio.mov"):
    """A clip with a real stereo 48k audio track under it."""
    ffmpeg = _ffmpeg()
    path = os.path.join(directory, name)
    seconds = FRAMES / 24.0
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:r=24:d={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-ac", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", path],
        capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(path):
        return None, proc.stderr.decode("utf-8", "replace")
    return path, None


# ---------------------------------------------------------------------------
# 1. IS_CHANGED
# ---------------------------------------------------------------------------

def test_is_changed(clip):
    IC = vl.VideoLoader.IS_CHANGED

    plain = IC(clip, 0, -1, 1)
    quoted = IC(f'"{clip}"', 0, -1, 1)
    check("IS_CHANGED strips a quoted path", plain == quoted,
          f"\n      {plain}\n      {quoted}")
    check("IS_CHANGED is not the constant 'missing' for a real file",
          "missing" not in plain, plain)

    # A rewrite that keeps the same second but changes the size. A float
    # getmtime() cannot see this; mtime_ns + size can.
    st = os.stat(clip)
    with open(clip, "ab") as fh:
        fh.write(b"\0" * 1024)
    os.utime(clip, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = IC(clip, 0, -1, 1)
    check("IS_CHANGED sees a same-mtime rewrite (size moved)", after != plain,
          f"\n      before {plain}\n      after  {after}")

    # Same size, mtime bumped by less than a second.
    os.utime(clip, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    bumped = IC(clip, 0, -1, 1)
    check("IS_CHANGED sees a sub-second mtime change", bumped != after)

    # Missing paths vary per path rather than collapsing to one constant.
    m1 = IC("/nope/one.mp4", 0, -1, 1)
    m2 = IC("/nope/two.mp4", 0, -1, 1)
    check("IS_CHANGED varies for different missing paths", m1 != m2)

    # The trim inputs are part of the fingerprint.
    check("IS_CHANGED varies with the trim inputs",
          len({IC(clip, 0, -1, 1), IC(clip, 2, -1, 1),
               IC(clip, 0, 5, 1), IC(clip, 0, -1, 2)}) == 4)


# ---------------------------------------------------------------------------
# 2. uint8 -> float32 conversion is exact
# ---------------------------------------------------------------------------

def test_scale_exact(clip):
    raw = vl.load_frames(clip, 0, FRAMES - 1, 1)
    vl._FRAME_CACHE.clear()
    out = vl.VideoLoader().load(clip, 0, -1, 1)[0]
    ref = raw.astype(np.float32) / 255.0
    check("load() dtype is float32", out.dtype == torch.float32, str(out.dtype))
    check("load() is bit-identical to astype(float32)/255",
          out.shape == ref.shape and np.array_equal(out.numpy(), ref),
          f"shapes {tuple(out.shape)} vs {ref.shape}")
    check("load() output is in [0,1]",
          float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

    # A non-contiguous decoder result (the torchvision backend slices) must
    # survive the conversion too.
    strided = np.ascontiguousarray(raw)[::3]
    src = torch.from_numpy(np.ascontiguousarray(strided))
    buf = torch.empty(src.shape, dtype=torch.float32)
    torch.div(src, 255.0, out=buf)
    check("conversion handles a strided decoder result",
          np.array_equal(buf.numpy(), strided.astype(np.float32) / 255.0))


# ---------------------------------------------------------------------------
# 3. The preview routes only serve video files
# ---------------------------------------------------------------------------

def test_route_gating(clip, tmp):
    secret = os.path.join(tmp, "secret.txt")
    with open(secret, "w") as fh:
        fh.write("not for the browser")

    r = call_route("/bat/video-stream", path=secret)
    check("video-stream refuses a non-video file",
          isinstance(r, _Resp) and r.status == 404, f"got {type(r).__name__}")
    r = call_route("/bat/video-stream", path=clip)
    check("video-stream serves a video file",
          isinstance(r, _FileResp) and r.path == clip)
    r = call_route("/bat/video-stream", path=os.path.join(tmp, "nope.mp4"))
    check("video-stream refuses a missing file",
          isinstance(r, _Resp) and r.status == 404)

    r = call_route("/bat/video-frame", path=secret, frame="0")
    check("video-frame refuses a non-video file",
          isinstance(r, _Resp) and r.status == 404)
    r = call_route("/bat/video-frame", path=clip, frame="3")
    check("video-frame returns a JPEG",
          isinstance(r, _Resp) and r.content_type == "image/jpeg"
          and r.body and r.body[:2] == b"\xff\xd8",
          getattr(r, "text", "") or "")
    r = call_route("/bat/video-frame", path=clip, frame="x", max_w="junk")
    check("video-frame survives junk frame/max_w",
          isinstance(r, _Resp) and r.content_type == "image/jpeg")

    r = call_route("/bat/video-info", path=secret)
    check("video-info refuses a non-video file",
          isinstance(r, _JsonResp) and r.payload.get("ok") is False)
    r = call_route("/bat/video-info", path=f'"{clip}"')
    check("video-info accepts a quoted path and reports the frame count",
          isinstance(r, _JsonResp) and r.payload.get("ok") is True
          and r.payload.get("frame_count") == FRAMES,
          str(getattr(r, "payload", "")))

    # The route reports what the load will actually do, so the node face can
    # say so before the artist runs it.
    r = call_route("/bat/video-info", path=clip)
    p = r.payload if isinstance(r, _JsonResp) else {}
    check("video-info reports bit depth, alpha and audio",
          p.get("bit_depth") in (8, 16) and "has_alpha" in p and "has_audio" in p,
          str({k: p.get(k) for k in ("bit_depth", "has_alpha", "has_audio", "pix_fmt")}))

    # Opt-in confinement.
    os.environ["BAT_STRICT_PATHS"] = "1"
    try:
        inside = vl._is_safe_path(os.path.join(os.path.abspath("."), "x.mp4"))
        outside = vl._is_safe_path("/etc/passwd")
        r = call_route("/bat/video-stream", path=clip)
    finally:
        del os.environ["BAT_STRICT_PATHS"]
    check("BAT_STRICT_PATHS confines to the ComfyUI tree",
          inside and not outside)
    check("BAT_STRICT_PATHS blocks an out-of-tree video",
          isinstance(r, _Resp) and r.status == 404)
    check("without BAT_STRICT_PATHS any tree is allowed",
          vl._is_safe_path("/etc/passwd"))

    # getpath lists the clip's directory but not the .txt beside it.
    r = call_route("/bat/getpath", path=tmp + os.sep,
                   extensions=",".join(vl.VIDEO_EXTENSIONS))
    names = r.payload if isinstance(r, _JsonResp) else []
    check("getpath lists videos and hides other files",
          os.path.basename(clip) in names and "secret.txt" not in names,
          str(names))


# ---------------------------------------------------------------------------
# 4. Trim maths
# ---------------------------------------------------------------------------

def test_trim(clip):
    node = vl.VideoLoader()

    def n(start, end, stride):
        vl._FRAME_CACHE.clear()
        return node.load(clip, start, end, stride)[0].shape[0]

    def rate(start, end, stride):
        vl._FRAME_CACHE.clear()
        return node.load(clip, start, end, stride)[3]

    check("end_frame=-1 loads to the end", n(0, -1, 1) == FRAMES)
    check("end_frame past the end clamps", n(0, 999, 1) == FRAMES)
    check("inclusive range", n(2, 5, 1) == 4)
    check("stride counts from start_frame", n(0, -1, 3) == 4)
    check("stride over a sub-range", n(1, 8, 2) == 4)
    check("end < start yields one frame", n(6, 2, 1) == 1)
    check("start past the end clamps to the last frame", n(999, -1, 1) == 1)
    check("stride 0 is treated as 1", n(0, -1, 0) == FRAMES)

    # frame_rate describes the LOADED batch, so a stride divides it — encode
    # the result at that rate and it keeps the source's timing.
    check("frame_rate is the source rate at stride 1",
          abs(rate(0, -1, 1) - 24.0) < 1e-6, str(rate(0, -1, 1)))
    check("frame_rate divides by the stride",
          abs(rate(0, -1, 3) - 8.0) < 1e-6, str(rate(0, -1, 3)))


# ---------------------------------------------------------------------------
# 5. The in-process cache is keyed on the file's content, not a coarse mtime
# ---------------------------------------------------------------------------

def test_cache(clip, tmp):
    node = vl.VideoLoader()
    vl._FRAME_CACHE.clear()
    first = node.load(clip, 0, -1, 1)[0]
    again = node.load(clip, 0, -1, 1)[0]
    check("a repeat load hits the cache", first is again)

    # Re-render the clip in place — different length and content, but pinned
    # to the same mtime, as a farm rewrite on a coarse-granularity mount looks.
    # A cache keyed on a float mtime serves `first` again here.
    st = os.stat(clip)
    replacement, err = make_clip(tmp, "other.mkv", frames=FRAMES + 5, shade=7)
    if replacement is None:
        check("cache invalidates on a same-second rewrite", False, err or "")
        return
    with open(replacement, "rb") as fh:
        data = fh.read()
    with open(clip, "wb") as fh:
        fh.write(data)
    os.utime(clip, ns=(st.st_atime_ns, st.st_mtime_ns))
    third = node.load(clip, 0, -1, 1)[0]
    check("cache invalidates on a same-second rewrite",
          third is not first and third.shape[0] == FRAMES + 5,
          f"{third.shape[0]} frames")

    # Audio is counted against the budget too, so an entry holding a
    # waveform has to weigh more than the same entry without one.
    vl._FRAME_CACHE.clear()
    node.load(clip, 0, -1, 1)
    entry = next(iter(vl._FRAME_CACHE.values()))
    bare = vl._entry_nbytes(entry)
    entry["audio"] = {"waveform": torch.zeros((1, 2, 48000)),
                      "sample_rate": 48000}
    check("the cache accounts for a cached waveform",
          vl._entry_nbytes(entry) == bare + 2 * 48000 * 4,
          f"{bare} -> {vl._entry_nbytes(entry)}")

    # The byte budget actually evicts.
    vl._FRAME_CACHE.clear()
    saved = vl._FRAME_CACHE_MAX_BYTES
    try:
        vl._FRAME_CACHE_MAX_BYTES = 1
        node.load(clip, 0, -1, 1)
        check("the byte budget evicts", len(vl._FRAME_CACHE) == 0,
              f"{len(vl._FRAME_CACHE)} entries left")
    finally:
        vl._FRAME_CACHE_MAX_BYTES = saved
        vl._FRAME_CACHE.clear()


# ---------------------------------------------------------------------------
# 6. The output signature
# ---------------------------------------------------------------------------

def test_outputs(clip):
    check("four outputs, in the documented order",
          vl.VideoLoader.RETURN_TYPES == ("IMAGE", "MASK", "AUDIO", "FLOAT")
          and vl.VideoLoader.RETURN_NAMES
              == ("images", "mask", "audio", "frame_rate"),
          str(vl.VideoLoader.RETURN_TYPES))
    check("every output is documented",
          len(vl.VideoLoader.OUTPUT_TOOLTIPS) == len(vl.VideoLoader.RETURN_TYPES))
    check("the AUDIO slot constant matches the signature",
          vl.VideoLoader.RETURN_TYPES[vl._AUDIO_SLOT] == "AUDIO")

    vl._FRAME_CACHE.clear()
    images, mask, audio, rate = vl.VideoLoader().load(clip, 0, -1, 1)
    check("mask is (N,H,W) alongside a (N,H,W,3) image batch",
          mask.shape == images.shape[:3], f"{tuple(mask.shape)}")
    check("a clip without alpha gives an empty mask",
          float(mask.abs().max()) == 0.0)
    check("a clip without audio gives None", audio is None)
    check("frame_rate is a plain float", isinstance(rate, float))


# ---------------------------------------------------------------------------
# 7. Audio is only extracted when something asks for it
# ---------------------------------------------------------------------------

def test_audio_gating(tmp):
    clip, err = make_clip_with_audio(tmp)
    if clip is None:
        print(f"SKIP  audio — could not build a clip with an audio track: {err}")
        return

    probe = vl.probe_ffmpeg(clip)
    check("the probe sees the audio stream",
          bool(probe and probe["has_audio"]), str(probe))

    # Nothing wired to slot 2: no extraction, and the entry stays undecided so
    # connecting the output later still works.
    prompt = {"7": {"inputs": {"images": ["7", 0]}}}
    vl._FRAME_CACHE.clear()
    node = vl.VideoLoader()
    _i, _m, audio, _r = node.load(clip, 0, -1, 1, unique_id="7", prompt=prompt)
    entry = next(iter(vl._FRAME_CACHE.values()))
    check("an unconnected audio output is not extracted",
          audio is None and entry["audio"] is vl._UNSET)

    # Wired to slot 2.
    prompt = {"7": {"inputs": {}},
              "9": {"inputs": {"audio": ["7", vl._AUDIO_SLOT]}}}
    _i, _m, audio, _r = node.load(clip, 0, -1, 1, unique_id="7", prompt=prompt)
    ok = isinstance(audio, dict) and "waveform" in audio
    check("a connected audio output is extracted", ok, str(type(audio)))
    if not ok:
        return
    check("the waveform is (1, channels, samples)",
          audio["waveform"].dim() == 3 and audio["waveform"].shape[0] == 1,
          str(tuple(audio["waveform"].shape)))
    check("the sample rate is read from the source",
          audio["sample_rate"] == 48000, str(audio["sample_rate"]))
    check("stereo is detected", audio["waveform"].shape[1] == 2,
          str(audio["waveform"].shape[1]))
    # Half a second of the clip, taken from the middle.
    half = vl.VideoLoader().load(clip, FRAMES // 2, FRAMES - 1, 1,
                                 unique_id="7", prompt=prompt)[2]
    check("the audio is trimmed to the frame range",
          half["waveform"].shape[2] < audio["waveform"].shape[2],
          f"{half['waveform'].shape[2]} vs {audio['waveform'].shape[2]}")
    check("no prompt at all still extracts (uncertainty costs, not breaks)",
          isinstance(vl.VideoLoader().load(clip, 0, -1, 1)[2], dict))


# ---------------------------------------------------------------------------
# 8. High bit depth and alpha go through ffmpeg
# ---------------------------------------------------------------------------

def test_bit_depth_and_alpha(tmp):
    for name, args, expect in (
        ("prores10.mov",
         ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
         {"bit_depth": 16, "has_alpha": False}),
        ("prores4444.mov",
         ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
          "-alpha_bits", "16"],
         {"bit_depth": 16, "has_alpha": True}),
        ("alpha8.mov",
         ["-c:v", "qtrle", "-pix_fmt", "rgba"],
         {"bit_depth": 8, "has_alpha": True}),
    ):
        path, err = make_clip(tmp, name, extra=args, alpha=expect["has_alpha"])
        if path is None:
            print(f"SKIP  {name} — ffmpeg could not encode it")
            continue
        probe = vl.probe_ffmpeg(path)
        if probe is None:
            check(f"{name}: probed", False, "no probe")
            continue
        check(f"{name}: bit depth read as {expect['bit_depth']}",
              probe["bit_depth"] == expect["bit_depth"],
              f"pix_fmt={probe['pix_fmt']!r} depth={probe['bit_depth']}")
        check(f"{name}: alpha {'seen' if expect['has_alpha'] else 'not claimed'}",
              probe["has_alpha"] == expect["has_alpha"],
              f"pix_fmt={probe['pix_fmt']!r}")

        vl._FRAME_CACHE.clear()
        images, mask, meta = vl.load_batch(path, 0, FRAMES - 1, 1,
                                           want_progress=False)
        check(f"{name}: decoded by ffmpeg", meta["backend"] == "ffmpeg",
              meta["backend"])
        check(f"{name}: full frame count", images.shape[0] == FRAMES,
              str(images.shape[0]))
        check(f"{name}: values stay in [0,1]",
              float(images.min()) >= 0.0 and float(images.max()) <= 1.0,
              f"{float(images.min()):.4f}..{float(images.max()):.4f}")
        if expect["has_alpha"]:
            # The generated clip is transparent down its left half, so the
            # mask (1-alpha) must be 1 there and 0 on the right.
            w = mask.shape[2]
            left = float(mask[:, :, :w // 4].mean())
            right = float(mask[:, :, -w // 4:].mean())
            check(f"{name}: mask is 1-alpha, not alpha",
                  left > 0.9 and right < 0.1, f"left {left:.3f} right {right:.3f}")
        else:
            check(f"{name}: opaque source gives an empty mask",
                  float(mask.abs().max()) == 0.0)

        # A 10-bit source must carry values a 8-bit decode cannot represent,
        # otherwise the "keeps its precision" claim is empty.
        if expect["bit_depth"] == 16:
            quantised = (images * 255.0).round() / 255.0
            check(f"{name}: values are finer than 8-bit steps",
                  float((images - quantised).abs().max()) > 1e-4,
                  f"max deviation {float((images - quantised).abs().max()):.6f}")


# ---------------------------------------------------------------------------
# 9. Progress and cancellation
# ---------------------------------------------------------------------------

def test_progress_and_cancel(clip):
    seen = []
    real = vl._progress_bar

    class Bar:
        def update(self, k): seen.append(k)

    vl._progress_bar = lambda total: Bar()
    try:
        vl._FRAME_CACHE.clear()
        vl.load_batch(clip, 0, FRAMES - 1, 1)
    finally:
        vl._progress_bar = real
    check("progress is reported as frames arrive",
          len(seen) > 1 and sum(seen) == FRAMES, f"{seen}")

    # Cancellation: the poll raises, and it has to get out of the backend
    # rather than being swallowed by its except-and-try-the-next-one loop.
    class Cancelled(BaseException):
        pass

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] >= 3:
            raise Cancelled()

    real_poll = vl._raise_if_interrupted
    vl._raise_if_interrupted = boom
    try:
        vl._FRAME_CACHE.clear()
        try:
            vl.load_batch(clip, 0, FRAMES - 1, 1)
            outcome = "returned"
        except Cancelled:
            outcome = "raised"
        except Exception as e:
            outcome = f"wrong error: {type(e).__name__}: {e}"
    finally:
        vl._raise_if_interrupted = real_poll
    check("a cancel mid-decode propagates out of the backend",
          outcome == "raised", outcome)
    check("a cancelled load leaves nothing in the cache",
          len(vl._FRAME_CACHE) == 0, f"{len(vl._FRAME_CACHE)} entries")


# ---------------------------------------------------------------------------
# 10. The banner parser, on the shapes ffmpeg actually prints
# ---------------------------------------------------------------------------

def test_banner_parser():
    cases = [
        # (banner line, pix_fmt, alpha, depth)
        ("    Stream #0:0(eng): Video: prores (apch / 0x68637061), yuv422p10le(tv, bt709), "
         "1920x1080, 117440 kb/s, 24 fps, 24 tbr, 24k tbn", "yuv422p10le", False, 16),
        ("    Stream #0:0: Video: prores (ap4h / 0x68347061), yuva444p10le(tv), "
         "2048x1080, 25 fps, 25 tbr", "yuva444p10le", True, 16),
        ("    Stream #0:0: Video: h264 (High), yuv420p(progressive), 1920x1080, "
         "30 fps, 30 tbr, 90k tbn", "yuv420p", False, 8),
        ("    Stream #0:0: Video: qtrle (rle / 0x20656C72), rgba, 640x480, "
         "12 fps, 12 tbr", "rgba", True, 8),
        ("    Stream #0:0: Video: ffv1 (FFV1 / 0x31564646), bgr0, 64x48, "
         "24 fps, 24 tbr", "bgr0", False, 8),
        ("    Stream #0:0: Video: vp9 (Profile 0), yuva420p, 512x512, "
         "30 fps, 30 tbr", "yuva420p", True, 8),
        ("    Stream #0:0: Video: rawvideo (RGB[24] / 0x18424752), rgb24, "
         "320x240, 25 fps", "rgb24", False, 8),
        ("    Stream #0:0: Video: exr, gbrapf32le, 2048x1152, 24 fps",
         "gbrapf32le", True, 16),
    ]
    for line, pix, alpha, depth in cases:
        got = vl._parse_ffmpeg_banner("Duration: 00:00:05.00, start: 0.0\n" + line)
        label = pix
        check(f"banner: {label} pixel format", got["pix_fmt"] == pix,
              f"got {got['pix_fmt']!r}")
        check(f"banner: {label} alpha={alpha}", got["has_alpha"] == alpha,
              f"got {got['has_alpha']} from {got['pix_fmt']!r}")
        check(f"banner: {label} depth={depth}", got["bit_depth"] == depth,
              f"got {got['bit_depth']} from {got['pix_fmt']!r}")

    # rgb0 / bgr0 name a padding byte, not alpha. Reading them as alpha would
    # put a garbage mask on ordinary footage.
    check("banner: a padding byte is not alpha",
          not vl._parse_ffmpeg_banner(
              "Stream #0:0: Video: ffv1, bgr0, 64x48, 24 fps")["has_alpha"])

    full = vl._parse_ffmpeg_banner(
        "  Duration: 00:01:03.50, start: 0.000000, bitrate: 1000 kb/s\n"
        "    Stream #0:0: Video: h264, yuv420p, 1920x1080, 23.976 fps, 24 tbr\n"
        "    Stream #0:1: Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s")
    check("banner: dimensions", (full["width"], full["height"]) == (1920, 1080),
          f"{full['width']}x{full['height']}")
    check("banner: fractional fps", abs(full["fps"] - 23.976) < 1e-6,
          str(full["fps"]))
    check("banner: duration in seconds", abs(full["duration"] - 63.5) < 1e-6,
          str(full["duration"]))
    check("banner: audio stream seen", full["has_audio"] is True)
    check("banner: no audio stream", vl._parse_ffmpeg_banner(
        "Stream #0:0: Video: h264, yuv420p, 8x8, 24 fps")["has_audio"] is False)

    for layout, expect in (("mono", 1), ("stereo", 2), ("5.1(side)", 6),
                           ("7.1", 8), ("16 channels", 16), ("weird", 2)):
        rate, ch = vl._parse_audio_layout(
            f"    Stream #0:1: Audio: pcm_f32le, 48000 Hz, {layout}, flt")
        check(f"audio layout: {layout} -> {expect}ch", ch == expect and rate == 48000,
              f"got {ch}ch {rate}Hz")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="bat_video_loader_") as tmp:
        clip, err = make_clip(tmp)
        if clip is None:
            print("SKIP  could not build a test clip with ffmpeg:")
            print(err)
            sys.exit(1)
        if vl.probe_video(clip) is None:
            print("SKIP  no video decoder available "
                  "(install decord, opencv-python, or torchvision)")
            sys.exit(1)
        test_banner_parser()
        test_trim(clip)
        test_outputs(clip)
        test_scale_exact(clip)
        test_route_gating(clip, tmp)
        test_progress_and_cancel(clip)
        test_bit_depth_and_alpha(tmp)
        test_audio_gating(tmp)
        test_cache(clip, tmp)
        # Last: it mutates the clip's mtime/size.
        test_is_changed(clip)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
