#!/usr/bin/env python
"""
Prove 🦇 Loader reads what is on disk, for every kind of thing it accepts.

    $ ../../../env/bin/python tests/verify_loader.py

Everything here is built from scratch into a temp directory and compared against
an independent read of the same bytes — OpenImageIO for the EXRs, Pillow for the
stills — rather than against a stored expectation. A loader that quietly returns
the wrong pixels is the worst failure this pack can have, because every node
downstream inherits it and nothing looks broken.

What is covered
---------------
* Each branch end to end: a still, a ``####`` sequence, a folder, a movie, a
  single EXR and an EXR sequence.
* **Pixel equality**, not "roughly right": RGB and the 1-alpha mask are compared
  exactly against Pillow / OIIO for the paths that involve no resampling.
* **Frame ORDER**, which is the bug a batcher invites: EXR frames are decoded on
  a thread pool and land out of order, so the test writes a different value into
  each frame and checks the batch carries them ascending.
* **Layer aliasing** — the `cryptomatte` output must SHARE its tensors with
  `layers` rather than being a second copy of them. That is the difference
  between ~390 MB and ~780 MB per 2K frame on a comp EXR, and it is invisible
  unless something asserts it.
* **Overscan**: a sequence whose data window moves frame to frame, which is the
  case `auto` exists for. Verified in all three modes, including that `keep`
  still refuses it with a message naming the frame.
* **The skip / every-nth / cap window**, against the frame numbers in the
  filenames, so an off-by-one shows up as the wrong frame rather than the wrong
  count.
* **IS_CHANGED**: that it moves when a frame is rewritten on an identical mtime
  (the NFS case), when a frame is added, and not otherwise.

Needs OpenImageIO for the EXR half; without it those tests skip rather than fail.
"""

import json
import os
import shutil
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

import torch                                          # noqa: E402
from PIL import Image                                 # noqa: E402

import importlib.util                                 # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "batpack", os.path.join(PACK, "__init__.py"),
    submodule_search_locations=[PACK])
batpack = importlib.util.module_from_spec(_spec)
sys.modules["batpack"] = batpack
_spec.loader.exec_module(batpack)

BatLoader = batpack.NODE_CLASS_MAPPINGS["Bat_Loader"]
BatExrLayer = batpack.NODE_CLASS_MAPPINGS["Bat_ExrLayer"]
bat_loader = sys.modules["batpack.bat_loader"]

try:
    import OpenImageIO as oiio
    HAVE_OIIO = True
except ImportError:
    HAVE_OIIO = False

FAILURES = []
W, H = 40, 24


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: expected {want!r}, got {got!r}")
        print(f"  FAIL {name}: expected {want!r}, got {got!r}")
    else:
        print(f"  ok   {name}")


def note(name, value):
    print(f"  ..   {name}: {value}")


# ── material ────────────────────────────────────────────────────────────────

def make_png(path, seed):
    """A frame whose every channel is a known function of `seed`."""
    # int32 throughout, cast once at the end: numpy 2 refuses to fold a
    # Python int into a uint8 expression, and these deliberately wrap.
    xs = np.arange(W, dtype=np.int32)[None, :]
    ys = np.arange(H, dtype=np.int32)[:, None]
    rgba = np.zeros((H, W, 4), dtype=np.int32)
    rgba[:, :, 0] = (xs * 3 + seed) % 256          # R carries the frame number
    rgba[:, :, 1] = (ys * 5) % 256
    rgba[:, :, 2] = 128
    rgba[:, :, 3] = (xs * 6) % 256                 # a real alpha ramp
    rgba = rgba.astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)


def make_exr(path, seed, pad=0):
    """RGBA beauty plus named layers; `pad` grows the DATA window past the format."""
    names = ["R", "G", "B", "A",
             "diffuse.R", "diffuse.G", "diffuse.B",
             "N.X", "N.Y", "N.Z", "depth.Z"]
    spec = oiio.ImageSpec(W + 2 * pad, H + 2 * pad, len(names), "float")
    spec.channelnames = names
    if pad:
        spec.x, spec.y = -pad, -pad
        spec.full_x, spec.full_y = 0, 0
        spec.full_width, spec.full_height = W, H
    data = np.zeros((H + 2 * pad, W + 2 * pad, len(names)), dtype=np.float32)
    data[:, :, 0] = seed / 100.0                   # R identifies the frame
    data[:, :, 1] = np.linspace(0, 1, W + 2 * pad)[None, :]
    data[:, :, 2] = 0.25
    data[:, :, 3] = 0.5
    data[:, :, 4:7] = 0.5
    data[:, :, 7] = 0.1
    data[:, :, 8] = 0.2
    data[:, :, 9] = 1.0
    data[:, :, 10] = 10.0 + seed
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(data)
    out.close()


def build(root):
    """Everything the tests read, and the frame numbers each thing carries."""
    os.makedirs(f"{root}/seq")
    os.makedirs(f"{root}/folder")
    for i, f in enumerate(range(1001, 1009)):
        make_png(f"{root}/seq/plate_{f}.png", i)
        make_png(f"{root}/folder/img_{f}.png", i)
    if HAVE_OIIO:
        os.makedirs(f"{root}/exr")
        os.makedirs(f"{root}/over")
        for i, f in enumerate(range(1001, 1007)):
            make_exr(f"{root}/exr/render_{f}.exr", i)
        # A bounding box that MOVES: 4, 6, 8, 10 px of overscan.
        for i, f in enumerate(range(1001, 1005)):
            make_exr(f"{root}/over/over_{f}.exr", i, pad=4 + i * 2)


def load(node, path, **kw):
    return node.load(path, **kw)["result"]


# ── tests ───────────────────────────────────────────────────────────────────

def test_still(root, node):
    print("\na single still")
    p = f"{root}/seq/plate_1001.png"
    images, mask, audio, layers, lnames, crypto, meta, rp, fps = load(node, p)
    ref = np.asarray(Image.open(p).convert("RGBA"), dtype=np.uint8)
    check("one frame", tuple(images.shape), (1, H, W, 3))
    check("RGB matches Pillow exactly", bool(torch.equal(
        images[0], torch.from_numpy(ref[:, :, :3].astype(np.float32) / 255.0))), True)
    check("mask is 1-alpha exactly", bool(torch.equal(
        mask[0], torch.from_numpy(1.0 - ref[:, :, 3].astype(np.float32) / 255.0))), True)
    check("no audio for a still", audio, None)
    check("no layers for a PNG", (len(layers), len(crypto)), (0, 0))
    check("resolved_path is the file", rp, p)
    check("frame rate is the sequence default", fps, bat_loader.SEQUENCE_FPS)


def test_png_sequence(root, node):
    print("\na #### sequence of stills")
    images, mask, *_ = load(node, f"{root}/seq/plate_####.png")
    check("all eight frames", tuple(images.shape), (8, H, W, 3))
    # R was written as (x*3 + frame_index), so frame N's first pixel is N/255.
    firsts = [round(float(images[i, 0, 0, 0]) * 255) for i in range(8)]
    check("frames are in ascending order", firsts, list(range(8)))

    print("\n  ...with skip / every-nth / cap")
    images, *_ = load(node, f"{root}/seq/plate_####.png", skip_first_frames=2)
    check("skip 2 drops the first two", [round(float(images[i, 0, 0, 0]) * 255)
                                         for i in range(images.shape[0])], [2, 3, 4, 5, 6, 7])
    images, *_ = load(node, f"{root}/seq/plate_####.png", select_every_nth=3)
    check("every 3rd", [round(float(images[i, 0, 0, 0]) * 255)
                        for i in range(images.shape[0])], [0, 3, 6])
    images, *_ = load(node, f"{root}/seq/plate_####.png", frame_load_cap=3)
    check("cap 3", [round(float(images[i, 0, 0, 0]) * 255)
                    for i in range(images.shape[0])], [0, 1, 2])
    images, *_ = load(node, f"{root}/seq/plate_####.png", skip_first_frames=1,
                      select_every_nth=2, frame_load_cap=3)
    check("skip 1, every 2nd, cap 3", [round(float(images[i, 0, 0, 0]) * 255)
                                       for i in range(images.shape[0])], [1, 3, 5])

    print("\n  ...and a pattern that matches nothing")
    try:
        load(node, f"{root}/seq/plate_###.png")     # three hashes, four digits
        check("wrong padding raises", "no error", "FileNotFoundError")
    except FileNotFoundError as exc:
        check("wrong padding raises FileNotFoundError", True, True)
        note("message", str(exc)[:70])


def test_folder(root, node):
    print("\na folder of stills")
    images, *_ = load(node, f"{root}/folder")
    check("every image in the folder", tuple(images.shape), (8, H, W, 3))
    check("sorted by name", [round(float(images[i, 0, 0, 0]) * 255)
                             for i in range(8)], list(range(8)))


def test_exr(root, node):
    print("\na single multi-layer EXR")
    if not HAVE_OIIO:
        print("  .... skipped (no OpenImageIO)")
        return
    p = f"{root}/exr/render_1001.exr"
    images, mask, audio, layers, lnames, crypto, meta, rp, fps = load(node, p)
    check("one frame", tuple(images.shape), (1, H, W, 3))
    check("layers found", sorted(layers.keys()), ["N", "depth", "diffuse"])
    # Declared STRING, so it must BE a string — a list here renders as
    # "['N', 'depth']" in every consumer.
    check("layer names is a newline-separated string", isinstance(lnames, str), True)
    # No `crypto:` entries here: this file has no layer bat_exr recognises as a
    # cryptomatte, and that list is built from the crypto-only dict.
    check("...listing the layers", sorted(lnames.split("\n")),
          ["N", "depth", "diffuse"])
    buf = oiio.ImageBuf(p)
    ref = buf.get_pixels(oiio.FLOAT)
    check("R channel matches OIIO exactly",
          bool(torch.equal(images[0, :, :, 0], torch.from_numpy(ref[:, :, 0]))), True)
    check("metadata parses as JSON", isinstance(json.loads(meta), dict), True)

    layer_node = BatExrLayer()
    img, msk = layer_node.get_layer(layers, "diffuse")
    check("🦇 EXR Layer pulls diffuse", (tuple(img.shape), round(float(img[0, 0, 0, 0]), 3)),
          ((1, H, W, 3), 0.5))
    img, msk = layer_node.get_layer(layers, "depth")
    check("a single-channel layer comes back as image AND mask",
          (tuple(img.shape), tuple(msk.shape), round(float(msk[0, 0, 0]), 1)),
          ((1, H, W, 3), (1, H, W), 10.0))
    try:
        layer_node.get_layer(layers, "nope")
        check("unknown layer raises", "no error", "KeyError")
    except KeyError:
        check("unknown layer raises KeyError", True, True)


def test_exr_sequence(root, node):
    print("\nan EXR sequence")
    if not HAVE_OIIO:
        print("  .... skipped (no OpenImageIO)")
        return
    images, mask, audio, layers, lnames, crypto, meta, rp, fps = load(
        node, f"{root}/exr/render_####.exr")
    check("all six frames", tuple(images.shape), (6, H, W, 3))
    # R was written as seed/100, and the frames decode on a thread pool — this
    # is the assertion that catches a batcher that fills slots out of order.
    check("frames are in ascending order",
          [round(float(images[i, 0, 0, 0]) * 100) for i in range(6)], list(range(6)))
    check("layers are batched too", tuple(layers["diffuse"].shape), (6, H, W, 3))
    check("depth carries its per-frame value",
          [round(float(layers["depth"][i, 0, 0])) for i in range(6)],
          [10, 11, 12, 13, 14, 15])
    check("metadata is one entry per frame", len(json.loads(meta)), 6)

    print("\n  ...and the cryptomatte output aliases the layers")
    shared = sum(1 for k in layers if k in crypto and layers[k] is crypto[k])
    check("every layer tensor is SHARED, not copied", (shared, len(layers)),
          (len(layers), len(layers)))


def test_overscan(root, node):
    print("\noverscan — a bounding box that moves frame to frame")
    if not HAVE_OIIO:
        print("  .... skipped (no OpenImageIO)")
        return
    pattern = f"{root}/over/over_####.exr"
    images, *_ = load(node, pattern, overscan="auto")
    check("auto conforms to the format", tuple(images.shape), (4, H, W, 3))
    images, *_ = load(node, pattern, overscan="crop")
    check("crop conforms to the format", tuple(images.shape), (4, H, W, 3))
    try:
        load(node, pattern, overscan="keep")
        check("keep refuses a moving bbox", "no error", "Exception")
    except Exception as exc:
        check("keep refuses a moving bbox", True, True)
        # Which frame it names is NOT fixed — the decode runs on a thread pool
        # and the batch is sized by whichever frame lands first. What has to
        # hold is that the message names a real frame and the reference it was
        # measured against, so the artist can see which two disagree.
        msg = str(exc)
        named = [n for n in ("1001", "1002", "1003", "1004") if n in msg]
        check("...names the offending frame and its reference", len(named) >= 2, True)
        check("...and says the frames differ in size",
              "aren't all the same size" in msg, True)

    print("\n  ...a constant bounding box is left alone by auto")
    steady = f"{root}/steady"
    os.makedirs(steady, exist_ok=True)
    for f in range(1001, 1004):
        make_exr(f"{steady}/s_{f}.exr", f - 1001, pad=5)
    images, *_ = load(node, f"{steady}/s_####.exr", overscan="auto")
    check("auto keeps the overscan when every frame agrees",
          tuple(images.shape), (3, H + 10, W + 10, 3))
    images, *_ = load(node, f"{steady}/s_####.exr", overscan="crop")
    check("crop strips it on request", tuple(images.shape), (3, H, W, 3))


def test_is_changed(root, node):
    print("\nIS_CHANGED")
    pattern = f"{root}/seq/plate_####.png"
    before = BatLoader.IS_CHANGED(pattern)
    check("stable across two calls", BatLoader.IS_CHANGED(pattern), before)

    # A rewrite pinned to the SAME mtime — the NFS case mtime alone misses.
    victim = f"{root}/seq/plate_1001.png"
    st = os.stat(victim)
    make_png(victim, 99)
    shutil.copystat(f"{root}/seq/plate_1002.png", victim)
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = BatLoader.IS_CHANGED(pattern)
    check("sees a same-mtime rewrite (size moved)", after != before, True)

    make_png(f"{root}/seq/plate_1009.png", 8)
    check("sees a frame appear", BatLoader.IS_CHANGED(pattern) != after, True)

    check("a missing path is path-specific",
          BatLoader.IS_CHANGED("/nope/a_####.exr") != BatLoader.IS_CHANGED("/nope/b_####.exr"),
          True)
    check("an empty path is 'none'", BatLoader.IS_CHANGED(""), "none")
    check("a quoted path reads the same as a bare one",
          BatLoader.IS_CHANGED(f'"{root}/folder"'), BatLoader.IS_CHANGED(f"{root}/folder"))

    print("\n  ...a folder tracks its CONTENTS, not its own mtime")
    folder = f"{root}/folder"
    before = BatLoader.IS_CHANGED(folder)
    make_png(f"{folder}/img_1001.png", 77)
    check("a rewritten file inside changes the fingerprint",
          BatLoader.IS_CHANGED(folder) != before, True)


def test_edges(root, node):
    print("\nedge cases")

    def expect(label, fn, exc):
        try:
            fn()
            FAILURES.append(f"{label}: expected {exc}, got no error")
            print(f"  FAIL {label}: expected {exc}, got no error")
        except Exception as e:
            ok = type(e).__name__ == exc
            print(f"  {'ok  ' if ok else 'FAIL'} {label} -> {type(e).__name__}")
            if not ok:
                FAILURES.append(f"{label}: expected {exc}, got {type(e).__name__}")

    expect("empty path", lambda: load(node, ""), "ValueError")
    expect("missing file", lambda: load(node, "/nope/none.png"), "FileNotFoundError")
    expect("skip past the end",
           lambda: load(node, f"{root}/seq/plate_####.png", skip_first_frames=999),
           "ValueError")

    images, *_ = load(node, f'  "{root}/seq/plate_1001.png"  ')
    check("a quoted, padded path still loads", tuple(images.shape), (1, H, W, 3))
    # INPUT_TYPES says min=1, but an API submit or a converted input can deliver 0.
    images, *_ = load(node, f"{root}/seq/plate_####.png", select_every_nth=0)
    check("select_every_nth=0 is held at 1", tuple(images.shape)[0], 8)
    images, *_ = load(node, f"{root}/seq/plate_####.png", overscan="nonsense")
    check("an unknown overscan mode falls back to auto", tuple(images.shape)[0], 8)


def test_scan(root, node):
    print("\nthe node-face scan")
    scan = bat_loader._scan_path
    i = scan(f"{root}/seq/plate_####.png")
    check("sequence frame count", (i["ok"], i["frames"], i["pattern"]), (True, 8, True))
    i = scan(f"{root}/seq/plate_###.png")
    check("wrong padding reports zero frames (the whole point)",
          (i["ok"], i["frames"], i["pattern"]), (True, 0, True))
    i = scan(f"{root}/folder")
    check("folder", (i["kind"], i["frames"]), ("folder", 8))
    i = scan("/nope/nothing.png")
    check("missing path", i["ok"], False)
    if HAVE_OIIO:
        i = scan(f"{root}/exr/render_1001.exr")
        check("EXR layers are counted from channel prefixes",
              (i["kind"], i["layers"], i["width"]), ("EXR", 3, W))


def main():
    print("BAT Loader verification" + ("" if HAVE_OIIO else "  (no OpenImageIO — EXR tests skip)"))
    root = tempfile.mkdtemp(prefix="bat_loader_")
    try:
        build(root)
        node = BatLoader()
        test_still(root, node)
        test_png_sequence(root, node)
        test_folder(root, node)
        test_exr(root, node)
        test_exr_sequence(root, node)
        test_overscan(root, node)
        test_edges(root, node)
        test_scan(root, node)
        test_is_changed(root, node)     # last: it mutates the material
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
