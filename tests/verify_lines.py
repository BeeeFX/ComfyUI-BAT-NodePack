#!/usr/bin/env python
"""
Checks for 🦇 Lines (bat_lines.py, bat_lines_models.py). CPU only, no weights.

1. **Same geometry as controlnet_aux.** Old workflows migrate onto this node, so
   the processing size, the pad and the crop must be what
   `resize_image_with_pad` did, over awkward sizes, both upscale and downscale.
2. **Same bytes for the no-model filters.** Canny, Scribble XDoG and Standard
   Lineart against a transcription of controlnet_aux 1.1.5 (Apache-2.0, credited
   below) — and against the real pack too when it is importable here.
3. **The preview is the render.** The preview endpoint's function returns what
   the node returned for the cached frame, for every output size.
4. **The downloader only installs verified bytes, only inside models/lines.**

    python tests/verify_lines.py
"""

import importlib.util
import io
import os
import sys
import tempfile
import types

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pkg = types.ModuleType("batpack")
pkg.__path__ = [PACK]
sys.modules["batpack"] = pkg
ui_ref = _load("batpack.bat_ui_ref", os.path.join(PACK, "bat_ui_ref.py"))
models = _load("batpack.bat_lines_models", os.path.join(PACK, "bat_lines_models.py"))
lines = _load("batpack.bat_lines", os.path.join(PACK, "bat_lines.py"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Reference: controlnet_aux 1.1.5, transcribed (Apache-2.0, Fannovel16 et al.)
# custom_controlnet_aux/util.py, canny/, scribble/, lineart_standard/
# ---------------------------------------------------------------------------

def ref_pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def ref_resize_image_with_pad(img, resolution, upscale_method=cv2.INTER_CUBIC):
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=upscale_method if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = ref_pad64(H_target), ref_pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return np.ascontiguousarray(x[:H_target, :W_target, ...].copy()).copy()

    return np.ascontiguousarray(img_padded.copy()).copy(), remove_pad


def ref_canny(u8, lo, hi, res):
    m, rp = ref_resize_image_with_pad(u8, res)
    return rp(cv2.Canny(m, lo, hi))


def ref_xdog(u8, thr, res):
    x, rp = ref_resize_image_with_pad(u8, res)
    g1 = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(x.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(x, dtype=np.uint8)
    result[2 * (255 - dog) > thr] = 255
    return rp(result)[:, :, 0]


def ref_lineart_standard(u8, sigma, thr, res):
    x, rp = ref_resize_image_with_pad(u8, res)
    x = x.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), sigma)
    intensity = np.min(g - x, axis=2).clip(0, 255)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity /= max(16, np.median(intensity[intensity > thr]))
    intensity *= 127
    return rp(intensity.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Synthetic plates (no client material — the repo is public)
# ---------------------------------------------------------------------------

def plate(h, w, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / max(w - 1, 1), y / max(h - 1, 1),
                    0.5 + 0.5 * np.sin((x * x + y * y) / (4.0 * max(h, w)))], axis=2)
    img = (img * 255).astype(np.uint8)
    cv2.circle(img, (w // 3, h // 2), max(2, min(h, w) // 5), (250, 30, 40), -1)
    cv2.rectangle(img, (w // 2, h // 5), (w - 3, h // 2), (10, 10, 10), max(1, min(h, w) // 40))
    cv2.putText(img, "BAT", (3, h - 4), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, min(h, w) / 200), (255, 255, 255), 1)
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)
    return torch.from_numpy(img.astype(np.float32) / 255.0)


SIZES = [(97, 131), (300, 200), (64, 64), (37, 53), (512, 768), (721, 1279)]
RESOLUTIONS = [512, 64, 320, 1024]


def p_for(**kw):
    p = lines._defaults()
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
# 1 + 2. Geometry and the no-model filters
# ---------------------------------------------------------------------------

def test_geometry_and_filters():
    worst = {}
    for (h, w) in SIZES:
        f = plate(h, w, seed=h * w)
        u8 = lines.to_u8(f)
        ref_u8 = np.asarray(f.numpy() * 255.0, dtype=np.uint8)
        check(f"to_u8 truncates like controlnet_aux {h}x{w}", np.array_equal(u8, ref_u8))
        for res in RESOLUTIONS:
            got, (ht, wt) = lines.prepare(u8, res)
            want, rp = ref_resize_image_with_pad(u8, res)
            same = np.array_equal(got, want) and rp(want).shape[:2] == (ht, wt)
            worst.setdefault("prepare", []).append(same)

            for name, ref, p in (
                ("canny", lambda: ref_canny(u8, 100, 200, res), p_for(filter="canny", resolution=res, output_size="processing")),
                ("canny 30/90", lambda: ref_canny(u8, 30, 90, res), p_for(filter="canny", canny_low=30, canny_high=90, resolution=res, output_size="processing")),
                ("xdog scribble", lambda: ref_xdog(u8, 32, res), p_for(filter="xdog", xdog_style="scribble", resolution=res, output_size="processing")),
                ("xdog scribble 5", lambda: ref_xdog(u8, 5, res), p_for(filter="xdog", xdog_style="scribble", xdog_scribble_threshold=5, resolution=res, output_size="processing")),
                ("lineart standard", lambda: ref_lineart_standard(u8, 6.0, 8, res), p_for(filter="lineart standard", resolution=res, output_size="processing")),
                ("lineart standard 2/3", lambda: ref_lineart_standard(u8, 2.0, 3, res), p_for(filter="lineart standard", standard_sigma=2.0, standard_threshold=3, resolution=res, output_size="processing")),
            ):
                out = lines.run_filter(f, p)
                worst.setdefault(name, []).append(np.array_equal(out, ref()))
    for name, oks in worst.items():
        check(f"{name}: byte-equal to controlnet_aux over {len(oks)} size/resolution cases", all(oks),
              f"{oks.count(False)} differ" if not all(oks) else "")


def test_against_real_controlnet_aux():
    """The transcription above is only as good as the transcribing; where the
    real pack is installed, hold the node to it directly."""
    try:
        from custom_controlnet_aux.canny import CannyDetector
        from custom_controlnet_aux.scribble import ScribbleXDog_Detector
        from custom_controlnet_aux.lineart_standard import LineartStandardDetector
    except Exception as exc:
        print(f"SKIP  real controlnet_aux not importable here ({type(exc).__name__})")
        return
    ok = True
    for (h, w) in SIZES[:4]:
        f = plate(h, w, seed=7)
        u8 = np.asarray(f.numpy() * 255.0, dtype=np.uint8)
        for res in (512, 320):
            pairs = (
                (CannyDetector()(u8, 100, 200, detect_resolution=res, output_type="np")[:, :, 0],
                 lines.run_filter(f, p_for(filter="canny", resolution=res, output_size="processing"))),
                (ScribbleXDog_Detector()(u8, detect_resolution=res, thr_a=32, output_type="np")[:, :, 0],
                 lines.run_filter(f, p_for(filter="xdog", xdog_style="scribble", resolution=res, output_size="processing"))),
                (LineartStandardDetector()(u8, guassian_sigma=6.0, intensity_threshold=8, detect_resolution=res, output_type="np")[:, :, 0],
                 lines.run_filter(f, p_for(filter="lineart standard", resolution=res, output_size="processing"))),
            )
            ok &= all(np.array_equal(a, b) for a, b in pairs)
    check("byte-equal to the installed controlnet_aux (canny, xdog scribble, lineart standard)", ok)


def test_output_sizes():
    f = plate(301, 517, seed=1)
    for name in ("canny", "xdog", "lineart standard"):
        for res in (0, 512, 1024):
            a = lines.run_filter(f, p_for(filter=name, resolution=res), out_hw=(301, 517))
            check(f"{name} res={res} output_size=input is the image size", a.shape == (301, 517), str(a.shape))
            b = lines.run_filter(f, p_for(filter=name, resolution=res, output_size="processing"))
            want = lines.processing_size(301, 517, res)
            check(f"{name} res={res} output_size=processing is {want}", b.shape == want, str(b.shape))
    a = lines.run_filter(f, p_for(filter="canny", invert=True))
    b = lines.run_filter(f, p_for(filter="canny"))
    check("invert is 255 - lines", np.array_equal(a, 255 - b))
    soft = lines.run_filter(f, p_for(filter="xdog", xdog_style="soft"))
    check("soft xdog is graded, not binary", len(np.unique(soft)) > 8, f"{len(np.unique(soft))} levels")
    check("soft xdog finds lines", 0.005 < (soft > 128).mean() < 0.3, f"{(soft > 128).mean():.3f} of pixels")
    flat = torch.full((64, 96, 3), 0.1)
    flat[:, 48:] = 0.9
    fl = lines.run_filter(flat, p_for(filter="xdog", xdog_style="soft", resolution=0))
    check("soft xdog leaves flat areas black, dark or bright (no ink fills)",
          fl[:, :40].max() == 0 and fl[:, 56:].max() == 0 and fl[:, 44:52].max() > 0)
    check("manga rounds the resolution up to x256",
          lines.effective_resolution("lineart manga", 300) == 512 and lines.effective_resolution("lineart manga", 512) == 512)


# ---------------------------------------------------------------------------
# 3. The node and its preview
# ---------------------------------------------------------------------------

def test_node_and_preview():
    batch = torch.stack([plate(120, 160, seed=s) for s in range(4)])
    node = lines.BatLines()
    for output_size in ("input", "processing"):
        params = dict(filter="xdog", xdog_style="soft", resolution=256, output_size=output_size, preview_frame=2)
        res = node.run(batch, unique_id="42", **params)
        img, mask = res["result"]
        want_hw = (120, 160) if output_size == "input" else lines.processing_size(120, 160, 256)
        check(f"node outputs IMAGE (N,H,W,3) + MASK (N,H,W), {output_size}",
              tuple(img.shape) == (4, *want_hw, 3) and tuple(mask.shape) == (4, *want_hw),
              f"{tuple(img.shape)} {tuple(mask.shape)}")
        check("IMAGE channels and MASK are the same lines", torch.equal(img[..., 0], mask) and torch.equal(img[..., 2], mask))
        ui = ui_ref.load_ui(res["ui"]) or {}
        token = (ui.get("token") or [None])[0]
        entry = lines._cache_get(token)
        check("execute caches the preview frame under a token", entry is not None)
        if entry is None:
            continue
        p = lines.params_from_request({**params, "token": token})
        for fr in (2, 0):
            u8, info = lines.render_preview(entry, p, fr)
            want = (mask[fr].numpy() * 255.0 + 0.5).astype(np.uint8)
            check(f"preview frame {fr} == node output ({output_size})",
                  np.array_equal(u8, want) and info["exact"], str(info))
    p = lines.params_from_request({"filter": "nonsense", "canny_low": 9999, "resolution": "abc", "xdog_sigma": float("nan")})
    check("request params are coerced and clamped",
          p["filter"] == "canny" and p["canny_low"] == 255 and p["resolution"] == 512 and p["xdog_sigma"] == 0.8, str(p))
    del batch
    import gc
    gc.collect()
    u8, info = lines.render_preview(entry, p, 3)
    check("a freed batch falls back to the anchor frame and says so", info["exact"] is False)


# ---------------------------------------------------------------------------
# 4. Downloader
# ---------------------------------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_downloader():
    payload = b"not really a checkpoint" * 1000
    import hashlib
    good = hashlib.sha256(payload).hexdigest()
    saved = dict(models.MODELS)
    try:
        with tempfile.TemporaryDirectory() as d:
            models.MODELS["fake"] = {"path": "fake/w.pth", "url": "https://example.invalid/w.pth",
                                     "sha256": "0" * 64, "size": len(payload), "licence": "MIT", "project": "x"}
            try:
                models.ensure("fake", dirs=[d], _urlopen=lambda req, timeout=None: _Resp(payload))
                check("a checksum mismatch is refused", False)
            except RuntimeError as e:
                check("a checksum mismatch is refused", "wrong checksum" in str(e))
            left = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
            check("nothing is left behind after a refused download", not left, str(left))

            models.MODELS["fake"]["sha256"] = good
            path = models.ensure("fake", dirs=[d], _urlopen=lambda req, timeout=None: _Resp(payload))
            check("a verified download lands at <lines>/<subfolder>/<file>",
                  path == os.path.realpath(os.path.join(d, "fake/w.pth")) and open(path, "rb").read() == payload)
            again = models.ensure("fake", dirs=[d], _urlopen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("refetched")))
            check("a present file is found, not fetched again", again == os.path.join(d, "fake/w.pth"))

            models.MODELS["evil"] = dict(models.MODELS["fake"], path="../escape.pth")
            try:
                models.ensure("evil", dirs=[d], _urlopen=lambda req, timeout=None: _Resp(payload))
                check("a path outside models/lines is refused", False)
            except RuntimeError as e:
                check("a path outside models/lines is refused", "outside" in str(e))
            check("…and nothing was written outside", not os.path.exists(os.path.join(os.path.dirname(d), "escape.pth")))

            try:
                models.ensure("teed", auto_download=False, dirs=[d])
                check("auto_download off gives manual instructions", False)
            except RuntimeError as e:
                check("auto_download off gives manual instructions",
                      models.MODELS["teed"]["url"] in str(e) and "teed/7_model.pth" in str(e))
    finally:
        models.MODELS.clear()
        models.MODELS.update(saved)

    pinned = all(("/resolve/" in m["url"] and "/main/" not in m["url"]) or "raw.githubusercontent.com" in m["url"]
                 for m in models.MODELS.values())
    check("every weight URL is pinned to a revision or commit", pinned)
    check("every weight has a sha256", all(len(m["sha256"]) == 64 for m in models.MODELS.values()))
    check("TEED never comes from the CC-BY-NC-SA bdsqlsz copy", "bdsqlsz" not in models.MODELS["teed"]["url"])


def test_filters_listed():
    spec = lines.BatLines.INPUT_TYPES()["required"]
    check("the filter dropdown lists the nine filters in order",
          spec["filter"][0] == ["canny", "xdog", "lineart standard", "teed", "hed", "lineart realistic",
                                "lineart anime", "lineart manga", "mlsd"])
    # widgets_values are positional: this order is what saved workflows and the
    # controlnet_aux migrations rely on. Append new widgets; never reorder.
    order = ["image", "filter", "resolution", "output_size", "invert", "canny_low", "canny_high",
             "xdog_style", "xdog_sigma", "xdog_k", "xdog_sharpen", "xdog_threshold", "xdog_softness",
             "xdog_scribble_threshold", "standard_sigma", "standard_threshold", "teed_safe_steps",
             "hed_safe", "lineart_coarse", "mlsd_score", "mlsd_distance", "mlsd_line_width",
             "auto_download", "preview_frame"]
    names = list(spec)
    check("widget order is the saved-workflow contract", names[:len(order)] == order, str(names))


if __name__ == "__main__":
    test_filters_listed()
    test_geometry_and_filters()
    test_against_real_controlnet_aux()
    test_output_sizes()
    test_node_and_preview()
    test_downloader()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all passed")
