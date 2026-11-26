#!/usr/bin/env python
"""
🦇 Lines model filters against comfyui_controlnet_aux, on the same weight files.

Needs the weights (a ``models/lines``-shaped folder) and, for the comparison,
controlnet_aux importable. CI has neither, so it skips there:

    BAT_LINES_MODELS=/path/to/lines PYTHONPATH=<ComfyUI>:<controlnet_aux>/src \
        python tests/verify_lines_models.py

Every filter except TEED must match controlnet_aux (same weights, same pre- and
post-processing). TEED is fed the way its author feeds it (BGR minus the BIPED
mean) rather than controlnet_aux's raw RGB, so for TEED the difference is only
reported, together with how the upstream weights compare with the
CC-BY-NC-SA copy controlnet_aux downloads (when that copy is on disk).
"""

import functools
import glob
import os
import sys

import numpy as np
import torch

WEIGHTS = os.environ.get("BAT_LINES_MODELS", "")
if not WEIGHTS or not os.path.isdir(WEIGHTS):
    print("SKIP  set BAT_LINES_MODELS to a folder holding the 🦇 Lines weights")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)


# verify_lines loads the pack modules under the same names; take them from it
# (importing it after loading our own would silently replace them — and with
# them the search_dirs patch below).
import verify_lines  # noqa: E402
from verify_lines import plate  # noqa: E402  (same synthetic plates)

models = sys.modules["batpack.bat_lines_models"]
lines = sys.modules["batpack.bat_lines"]
# Only ever read the weights from BAT_LINES_MODELS; never search or download
# into the real models folders (auto_download is also off in p_for).
models.search_dirs = lambda: [WEIGHTS]

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def p_for(**kw):
    p = lines._defaults()
    p.update(auto_download=False)
    p.update(kw)
    return p


def w(key):
    return os.path.join(WEIGHTS, models.MODELS[key]["path"])


def sd(key):
    s = torch.load(w(key), map_location="cpu", weights_only=True)
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in s.items()}


def diff(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}", False
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return f"max {d.max()} mean {d.mean():.4f} differing {(d > 0).mean():.4%}", d.max() <= 1 and d.mean() <= 1 / 255


def ca_detectors():
    """controlnet_aux's own detector classes, holding the same weight files."""
    from torch import nn
    from custom_controlnet_aux.lineart import Generator, LineartDetector
    from custom_controlnet_aux.lineart_anime import UnetGenerator, LineartAnimeDetector
    from custom_controlnet_aux.manga_line import LineartMangaDetector
    from custom_controlnet_aux.manga_line.model_torch import res_skip
    from custom_controlnet_aux.mlsd import MLSDdetector
    from custom_controlnet_aux.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large
    from custom_controlnet_aux.hed import HEDdetector, ControlNetHED_Apache2
    from custom_controlnet_aux.teed import TEDDetector
    from custom_controlnet_aux.teed.ted import TED

    def ld(net, key):
        net.load_state_dict(sd(key), strict=True)
        return net.float().eval()

    norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return {
        "lineart": LineartDetector(ld(Generator(3, 1, 3), "lineart"), ld(Generator(3, 1, 3), "lineart_coarse")),
        "anime": LineartAnimeDetector(ld(UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False), "anime")),
        "manga": LineartMangaDetector(ld(res_skip(), "manga")),
        "mlsd": MLSDdetector(ld(MobileV2_MLSD_Large(), "mlsd")),
        "hed": HEDdetector(ld(ControlNetHED_Apache2(), "hed")),
        "teed": TEDDetector(ld(TED(), "teed")),
    }


def main():
    missing = [k for k in models.MODELS if not os.path.isfile(w(k))]
    if missing:
        print(f"SKIP  weights missing in {WEIGHTS}: {missing}")
        sys.exit(0)
    nets_weights = __import__("batpack.bat_lines_nets", fromlist=["weights"]).weights
    check("the networks read weights only from BAT_LINES_MODELS",
          nets_weights is models and all(
              os.path.realpath(models.find(k)).startswith(os.path.realpath(WEIGHTS) + os.sep)
              for k in models.MODELS))
    try:
        ca = ca_detectors()
    except Exception as exc:
        print(f"note  controlnet_aux not importable ({type(exc).__name__}: {exc}); sanity checks only")
        ca = None

    f = plate(300, 420, seed=11)
    u8 = np.asarray(f.numpy() * 255.0, dtype=np.uint8)
    cases = [
        ("lineart realistic", {}, lambda d, r: d["lineart"](u8, coarse=False, detect_resolution=r, output_type="np")),
        ("lineart realistic", {"lineart_coarse": True}, lambda d, r: d["lineart"](u8, coarse=True, detect_resolution=r, output_type="np")),
        ("lineart anime", {}, lambda d, r: d["anime"](u8, detect_resolution=r, output_type="np")),
        ("lineart manga", {}, lambda d, r: d["manga"](u8, detect_resolution=r, output_type="np")),
        ("mlsd", {}, lambda d, r: d["mlsd"](u8, thr_v=0.1, thr_d=0.1, detect_resolution=r, output_type="np")),
        ("hed", {"hed_safe": True}, lambda d, r: d["hed"](u8, detect_resolution=r, safe=True, output_type="np")),
        ("hed", {"hed_safe": False}, lambda d, r: d["hed"](u8, detect_resolution=r, safe=False, output_type="np")),
    ]
    for name, extra, ref in cases:
        for res in (512, 320):
            ours = lines.run_filter(f, p_for(filter=name, resolution=res, output_size="processing", **extra))
            label = f"{name} {extra or ''} res={res}"
            check(f"{label}: runs, uint8 lines", ours.dtype == np.uint8 and ours.max() > ours.min())
            if ca is not None:
                theirs = ref(ca, res)[:, :, 0]
                detail, ok = diff(ours, theirs)
                check(f"{label}: matches controlnet_aux", ok, detail)

    # TEED: upstream preprocessing on purpose — report, don't require.
    for res in (512, 320):
        ours = lines.run_filter(f, p_for(filter="teed", resolution=res, output_size="processing"))
        check(f"teed res={res}: runs, uint8 lines", ours.dtype == np.uint8 and ours.max() > ours.min(),
              f"{(ours > 127).mean():.3%} edge pixels")
        if ca is not None:
            theirs = ca["teed"](u8, detect_resolution=res, safe_steps=2, output_type="np")[:, :, 0]
            detail, _ = diff(ours, theirs)
            print(f"info  teed res={res}: upstream (BGR - mean) vs controlnet_aux (raw RGB) feeding, same weights: {detail}")

    nc = glob.glob(os.path.expanduser(
        "~/packages/comfy_external_nodes/*/platform-linux/arch-x86_64/comfyui_controlnet_aux/ckpts/**/7_model.pth"),
        recursive=True)
    if nc:
        a, b = sd("teed"), torch.load(nc[0], map_location="cpu", weights_only=True)
        same_keys = set(a) == set(b)
        same = same_keys and all(torch.equal(a[k], b[k]) for k in a)
        print(f"info  TEED upstream (MIT) vs controlnet_aux's bdsqlsz copy: "
              f"{'IDENTICAL tensors' if same else ('same keys, different values' if same_keys else 'different keys')}")

    # Output at the plate size; M-LSD redrawn, not resampled.
    for name in ("lineart realistic", "mlsd", "teed"):
        out = lines.run_filter(f, p_for(filter=name, resolution=512), out_hw=(300, 420))
        check(f"{name}: output_size=input is the plate size", out.shape == (300, 420), str(out.shape))
    thick = lines.run_filter(f, p_for(filter="mlsd", resolution=512, mlsd_line_width=4), out_hw=(300, 420))
    thin = lines.run_filter(f, p_for(filter="mlsd", resolution=512, mlsd_line_width=1), out_hw=(300, 420))
    check("mlsd lines are redrawn crisp (binary) at the output size", set(np.unique(thin)) <= {0, 255})
    check("mlsd_line_width thickens the redraw", (thick > 0).sum() > 2 * (thin > 0).sum(),
          f"{(thin > 0).sum()} -> {(thick > 0).sum()} px")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
