#!/usr/bin/env python
"""
Prove 🦇 Video Combine's preview routes work on ffmpeg 9 as well as 7.

    $ ../../../env/bin/python tests/verify_video_combine_trc.py

imageio-ffmpeg 0.8 bundles ffmpeg 9, which dropped two things these routes
leaned on:

* `-vsync` (deprecated since 5.1). /bat/video/frame and /bat/video/save_frame
  passed `-vsync 0`, so every frame grab and hover-scrub thumbnail failed with
  "Unrecognized option 'vsync'". They now use `-fps_mode passthrough`.
* The OpenEXR decoder's `-apply_trc`. The sRGB / Rec.709 views of an EXR
  sequence passed it, so those previews failed to open at all. The curve is
  now a `lutrgb` filter (_trc_filter), which works on 7 and 9 alike.

Everything runs the real bundled ffmpeg against material built in a temp dir:

* The curve itself, read back as 16-bit and compared with the sRGB / BT.709
  formulas (and the no-TRC default with identity), including overbright and
  negative pixels, which clamp the way `-apply_trc` did.
* Alpha survives the filter.
* The three routes end to end — frame, save_frame and the preview transcode —
  on an EXR sequence with each view, and frame / save_frame on a movie, where
  `-fps_mode` is what selects the frame.

Needs OpenImageIO to write the EXRs; without it, or without ffmpeg, it skips.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PACK))
sys.path.insert(0, COMFY)


# ── ComfyUI's `server` module, enough for the route decorators ───────────────
class _Routes:
    def _deco(self, *a, **k):
        def wrap(fn):
            return fn
        return wrap
    get = post = put = delete = patch = _deco


_server = types.ModuleType("server")
_server.PromptServer = types.SimpleNamespace(
    instance=types.SimpleNamespace(routes=_Routes(),
                                   send_sync=lambda *a, **k: None))
_server.web = types.SimpleNamespace(Response=object, json_response=object,
                                    FileResponse=object)
sys.modules.setdefault("server", _server)

try:
    import OpenImageIO as oiio
except ImportError:
    print("SKIP verify_video_combine_trc: OpenImageIO is not installed")
    sys.exit(0)

import importlib.util                                 # noqa: E402

import folder_paths                                   # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "batpack", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
batpack = importlib.util.module_from_spec(_spec)
sys.modules["batpack"] = batpack
_spec.loader.exec_module(batpack)
vc = sys.modules["batpack.bat_video_combine"]

FFMPEG = vc._ffmpeg_bin()
if not FFMPEG or not (shutil.which(FFMPEG) or os.path.isfile(FFMPEG)):
    print("SKIP verify_video_combine_trc: no ffmpeg")
    sys.exit(0)

FAILURES = []
W, H = 256, 4


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: expected {want!r}, got {got!r}")
        print(f"  FAIL {name}: expected {want!r}, got {got!r}")
    else:
        print(f"  ok   {name}")


def near(name, err, tol):
    ok = err <= tol
    if not ok:
        FAILURES.append(f"{name}: max error {err:.6f} > {tol}")
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max error {err:.6f} (tolerance {tol})")


# ── material ────────────────────────────────────────────────────────────────

def ramp(alpha=False):
    """Rows: 0..1 grey, 0..2 overbright, a coloured ramp, and one dipping
    below zero. Alpha, when asked for, is its own ramp running the other way."""
    r = np.linspace(0.0, 1.0, W, dtype=np.float32)
    img = np.stack([r, r, r], -1)[None].repeat(H, 0)
    img[1] *= 2.0
    img[2, :, 1] *= 0.5
    img[2, :, 2] *= 0.25
    img[3] -= 0.1
    if alpha:
        img = np.concatenate([img, (1.0 - r)[None, :, None].repeat(H, 0)], -1)
    return np.ascontiguousarray(img)


def write_exr(path, img):
    spec = oiio.ImageSpec(img.shape[1], img.shape[0], img.shape[2], oiio.FLOAT)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(img)
    out.close()


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def bt709(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.018053968510807, 4.5 * x,
                    1.099296826809442 * np.power(x, 0.45) - 0.099296826809442)


def decode16(path, vf, channels=3):
    """Run `path` through the `-vf` chain and read it back as 0..1 floats."""
    pix = "rgba64le" if channels == 4 else "rgb48le"
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path]
    if vf:
        args += ["-vf", vf]
    raw = subprocess.run(args + ["-f", "rawvideo", "-pix_fmt", pix, "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(H, W, channels).astype(np.float64) / 65535


def run(coro):
    return asyncio.run(coro)


# ── tests ───────────────────────────────────────────────────────────────────

def test_no_removed_options():
    print("\nno option ffmpeg 9 removed is passed any more")
    with open(vc.__file__, encoding="utf-8") as fh:
        src = fh.read()
    check('"-vsync" is gone', '"-vsync"' in src, False)
    check('"-apply_trc" is gone', '"-apply_trc"' in src, False)


def test_filter_shape():
    print("\n_trc_filter only touches EXR, and only for a known view")
    check("no view -> nothing", vc._trc_filter("", "/a/b.exr"), "")
    check("unknown view -> nothing", vc._trc_filter("acescg", "/a/b.exr"), "")
    check("PNG -> nothing", vc._trc_filter("srgb", "/a/b.%05d.png"), "")
    check("upper-case .EXR still counts",
          vc._trc_filter("srgb", "/a/B.%05d.EXR").startswith("format="), True)
    check("view name is case-insensitive",
          vc._trc_filter("Rec709", "/a/b.exr") == vc._trc_filter("rec709", "/a/b.exr"), True)
    check("_with_trc leaves a chain alone without a view",
          vc._with_trc("scale=2:2", "", "/a/b.exr"), "scale=2:2")
    check("...and prepends the curve with one",
          vc._with_trc("scale=2:2", "srgb", "/a/b.exr").endswith(",scale=2:2"), True)


def test_curve(root):
    print("\nthe curve, on this ffmpeg build")
    path = f"{root}/ramp.exr"
    img = ramp()
    write_exr(path, img)
    near("no view is identity", np.abs(decode16(path, "") - np.clip(img, 0, 1)).max(), 0.0005)
    near("sRGB view matches IEC 61966-2-1",
         np.abs(decode16(path, vc._trc_filter("srgb", path)) - srgb(img)).max(), 0.0005)
    near("Rec.709 view matches the BT.709 OETF",
         np.abs(decode16(path, vc._trc_filter("rec709", path)) - bt709(img)).max(), 0.0005)

    apath = f"{root}/ramp_alpha.exr"
    aimg = ramp(alpha=True)
    write_exr(apath, aimg)
    got = decode16(apath, vc._trc_filter("srgb", apath), channels=4)
    near("alpha passes through the curve untouched",
         np.abs(got[..., 3] - aimg[..., 3]).max(), 0.0005)
    near("...while the colour still gets it",
         np.abs(got[..., :3] - srgb(aimg[..., :3])).max(), 0.0005)


def test_routes(root):
    print("\nthe routes, end to end")
    out_dir, tmp_dir = f"{root}/output", f"{root}/temp"
    os.makedirs(f"{out_dir}/seq")
    os.makedirs(tmp_dir)
    folder_paths.set_output_directory(out_dir)
    folder_paths.set_temp_directory(tmp_dir)

    for i in range(1, 4):
        write_exr(f"{out_dir}/seq/shot.{i:05d}.exr", ramp() * (i / 3.0))
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=64x48:rate=24:duration=1",
                    "-pix_fmt", "yuv420p", f"{out_dir}/clip.mp4"], check=True)

    def frame(query):
        return run(vc.bat_video_frame(vc._FakeQueryRequest(query)))

    seq = {"filename": "shot.%05d.exr", "subfolder": "seq", "type": "output"}
    for view in ("", "srgb", "rec709"):
        resp = frame({**seq, "frame": "1", "trc": view})
        check(f"frame: EXR sequence, view {view or 'as-is'} -> PNG",
              (resp.status, bytes(resp.body[:4])), (200, b"\x89PNG"))
    resp = frame({"filename": "clip.mp4", "type": "output", "frame": "5"})
    check("frame: movie frame 5 -> PNG", (resp.status, bytes(resp.body[:4])), (200, b"\x89PNG"))

    class _Body:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    for body in ({"filename": "clip.mp4", "type": "output", "frame": 5},
                 {**seq, "frame": 1, "trc": "srgb"}):
        resp = run(vc.bat_video_save_frame(_Body({**body, "dest_prefix": "grab"})))
        check(f"save_frame: {body['filename']} -> 200", resp.status, 200)
    grabs = [f for f in os.listdir(out_dir) if f.startswith("grab")]
    check("...and both PNGs landed under output/", len(grabs), 2)

    target = vc._resolve_request_target(vc._FakeQueryRequest(seq))
    for view in ("", "srgb", "rec709"):
        cached = f"{tmp_dir}/preview_{view or 'asis'}.mp4"
        err = vc._transcode_preview(target, 24.0, view, cached, tmp_dir)
        check(f"preview transcode, view {view or 'as-is'}",
              (err, os.path.isfile(cached)), (None, True))


def main():
    version = subprocess.run([FFMPEG, "-version"], capture_output=True,
                             text=True).stdout.split("\n")[0]
    print(f"Video Combine preview routes vs {version}")
    root = tempfile.mkdtemp(prefix="bat_vc_trc_")
    try:
        test_no_removed_options()
        test_filter_shape()
        test_curve(root)
        test_routes(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
