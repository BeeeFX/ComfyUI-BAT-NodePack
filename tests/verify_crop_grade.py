#!/usr/bin/env python
"""
Regression checks for the crop / uncrop / grade fixes of 2026-09-23.

Each block pins one bug that shipped, measured the way it was found:

1. **Rotated crops sat half a pixel off.** The sampling grid centred the rect
   at x + w/2 (edge space) while align_corners=True puts pixel INDEX k at k,
   so a rotated Crop→Uncrop round trip landed 0.5–0.7px away from where it
   started. Measured on a ramp, where a shift reads straight off the values.
2. **Animated Crop rounded the rotated per-frame x/y**, but Uncrop pasted at
   the float — the paste jittered ±0.5px frame to frame on a slow move.
3. **Animated Crop resampled a constant rect on every frame** when the first
   key wasn't a multiple of snap_to (the editor's own 960×540 seed on 1080p).
4. **Uncrop clamped the whole composite to [0,1]** — an EXR lost every
   highlight, even outside the rect.
5. **Uncrop feathered edges lying on the frame border**, fading the
   processed crop back to the plate at the image edge.
6. **clamp_black was decorative**: negatives were zeroed before the gamma.
7. **Grade maths**: the fused in-place form must match the old step-by-step
   one, and a mask batch that is neither 1 nor N must not crash.
8. **Preview stride** capped nothing under 480 frames (n // 240).

Plus source-level pins for the JS halves that can't run headless here.

    python tests/verify_crop_grade.py
"""

import importlib
import json
import os
import re
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load_pack_modules():
    """Import the nodes without the pack __init__ (which needs a live server)."""
    pkg = types.ModuleType("batpkg")
    pkg.__path__ = [PACK]
    sys.modules["batpkg"] = pkg
    return tuple(importlib.import_module(f"batpkg.{m}") for m in (
        "bat_crop", "bat_uncrop", "bat_animated_crop",
        "bat_grade", "bat_animated_grade"))


def ramp(H, W, n=1):
    """R = x/W, G = y/H: any sub-pixel shift reads straight off the values."""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    img = torch.stack([xs / W, ys / H, torch.zeros_like(xs)], -1).unsqueeze(0)
    return img.repeat(n, 1, 1, 1)


def mean_shift(back, plate, mask, W, H):
    inside = mask > 0.999
    dx = ((back[..., 0] - plate[..., 0])[inside] * W).mean().item()
    dy = ((back[..., 1] - plate[..., 1])[inside] * H).mean().item()
    return dx, dy


# ─── 1. rotated round trip ───────────────────────────────────────────────────

def test_rotated_round_trip(bc, bu):
    print("\nrotated Crop→Uncrop lands where it started")
    H, W = 128, 160
    img = ramp(H, W)
    crop, unc = bc.BatCrop(), bu.BatUncrop()
    for ang in (5.0, 30.0, 90.0):
        o, info, _, _ = crop.crop(img, 40, 30, 64, 48, ang, 1, False, "free",
                                  False, "black")["result"]
        back, m = unc.uncrop(o, info, "stretch", "bilinear", 0)
        dx, dy = mean_shift(back[0], img[0], m[0], W, H)
        check(f"{ang:g}° round-trip shift < 0.01px", abs(dx) < 0.01 and abs(dy) < 0.01,
              f"({dx:+.3f}, {dy:+.3f})")
    # Just past the 1e-3 rotation threshold the rotated path must agree with
    # the exact axis-aligned slice (it used to be a +0.5px bilinear blend).
    rot = bc._rotated_crop(img, 10, 8, 32, 24, 0.0011)
    exact = img[:, 8:32, 10:42, :]
    check("near-0° rotated crop == exact slice",
          (rot - exact).abs().max().item() < 1e-3,
          f"max|d|={(rot - exact).abs().max().item():.4f}")
    rm = bc._rotated_rect_mask(H, W, 10 + 31 / 2, 8 + 23 / 2, 32, 24, 0.0, 1,
                               "cpu", torch.float32)
    check("axis rect_mask covers exactly w*h pixels", int(rm.sum()) == 32 * 24,
          f"{int(rm.sum())}")


# ─── 2 + 3. animated crop ────────────────────────────────────────────────────

def test_animated_crop(bu, bac):
    print("\nanimated crop")
    H, W = 96, 128
    img = ramp(H, W, 3)
    state = {"keyframes": {
        "0": {"x": 30, "y": 20, "w": 48, "h": 32, "angle": 10.0},
        "2": {"x": 31, "y": 20, "w": 48, "h": 32, "angle": 10.0}}}
    o, info, _, _ = bac.BatAnimatedCrop().crop(
        img, json.dumps(state), 8, False, "free", False, "black")["result"]
    back, m = bu.BatUncrop().uncrop(o, info, "stretch", "bilinear", 0)
    worst = max(max(map(abs, mean_shift(back[f], img[0], m[f], W, H)))
                for f in range(3))
    check("rotated, fractional x: per-frame round-trip shift < 0.01px",
          worst < 0.01, f"worst {worst:.3f}px")

    # A constant rect whose size isn't a multiple of snap_to: every frame must
    # be an exact slice (no resample), at the unchanged snapped output size.
    torch.manual_seed(0)
    noise = torch.rand(4, 60, 80, 3)
    st = {"keyframes": {"0": {"x": 10, "y": 5, "w": 40, "h": 30, "angle": 0.0}}}
    o, info, _, _ = bac.BatAnimatedCrop().crop(
        noise, json.dumps(st), 8, False, "free", True, "black")["result"]
    check("output size is still snap(first key) = 40x24", tuple(o.shape[1:3]) == (24, 40),
          str(tuple(o.shape)))
    f0 = info["frames"][0]
    ix, iy = int(round(f0["x"])), int(round(f0["y"]))
    exact = noise[:, iy:iy + 24, ix:ix + 40, :]
    check("constant rect is sliced, not resampled", torch.equal(o, exact),
          f"max|d|={(o - exact).abs().max().item():.4f}")
    check("snapped rect keeps the drawn centre",
          abs((f0["y"] + f0["h"] / 2) - (5 + 30 / 2)) < 1e-6, str(f0))

    # Preview stride: ceil, so nothing over 240 thumbnails goes out.
    tiny = torch.rand(479, 4, 4, 3)
    ui = bac.BatAnimatedCrop().crop(tiny, "{}", 1, False, "free", True,
                                    "black")["ui"]
    check("479 frames → ≤240 preview thumbnails", len(ui["frames"]) <= 240,
          f"{len(ui['frames'])} (stride {ui['stride'][0]})")


# ─── 4 + 5. uncrop HDR and feather ───────────────────────────────────────────

def test_uncrop(bc, bu):
    print("\nuncrop")
    plate = torch.rand(1, 64, 96, 3) * 0.5
    plate[0, 0, 0] = 4.0                    # a highlight OUTSIDE the rect
    crop, unc = bc.BatCrop(), bu.BatUncrop()
    for ang in (0.0, 10.0):
        o, info, _, _ = crop.crop(plate, 40, 30, 16, 16, ang, 8, False, "free",
                                  True, "black")["result"]
        back, _ = unc.uncrop(o * 3.0, info, "stretch", "bilinear", 0)
        check(f"{ang:g}°: HDR plate pixel outside rect survives",
              abs(back[0, 0, 0, 0].item() - 4.0) < 1e-6, str(back[0, 0, 0].tolist()))
        check(f"{ang:g}°: HDR paste (>1) survives inside rect",
              back.max().item() > 1.0 + 1e-3, f"max {back.max().item():.3f}")
    hot = torch.full((1, 16, 16, 3), 5.0)
    rot = bc._rotated_crop(hot, 0, 0, 8, 8, 20.0, "edge")
    check("rotated crop keeps values > 1", rot.max().item() > 4.99)

    zero = torch.zeros(1, 64, 96, 3)
    o, info, _, _ = crop.crop(zero, 0, 0, 48, 64, 0.0, 8, False, "free",
                              True, "black")["result"]
    back, m = unc.uncrop(torch.ones_like(o), info, "stretch", "bilinear", 16)
    check("no feather on an edge lying on the frame border",
          back[0, 32, 0, 0].item() == 1.0 and back[0, 0, 20, 0].item() == 1.0,
          f"col0={back[0, 32, 0, 0].item():.3f} row0={back[0, 0, 20, 0].item():.3f}")
    check("interior edge is still feathered", 0.0 < m[0, 32, 47].item() < 1.0,
          f"{m[0, 32, 47].item():.3f}")


# ─── 6 + 7. grade ────────────────────────────────────────────────────────────

def _old_grade(img, bp, wp, lift, gain, mult, off, gamma, cw, cb, mask=None):
    """The pre-fix step-by-step maths, for the non-negative regime where the
    fix must not change a thing."""
    base = (img - bp) / max(wp - bp, 1e-6)
    out = (base * (gain - lift) + lift) * mult + off
    out = out.clamp(min=0).pow(1.0 / max(gamma, 1e-6))
    if cw:
        out = out.clamp(max=1)
    if cb:
        out = out.clamp(min=0)
    if mask is not None:
        m = mask.unsqueeze(-1)
        out = out * m + img * (1 - m)
    return out


def test_grade(bg, bag):
    print("\ngrade")
    torch.manual_seed(1)
    img = torch.rand(3, 16, 16, 3)
    mask = torch.rand(3, 16, 16)
    # Every pre-gamma value is >= 0 for these, the regime where the fused
    # form must match the old maths exactly (negatives are block 6).
    for params in ((-0.05, 0.9, 0.02, 1.1, 1.2, 0.03, 1.4, False, False),
                   (0.05, 0.9, 0.02, 1.1, 1.2, 0.03, 1.4, False, True),
                   (0.0, 1.0, 0.0, 1.5, 1.0, 0.0, 0.7, True, True),
                   (-0.1, 1.3, 0.1, 0.8, 0.9, 0.0, 2.2, True, False)):
        new = bg._apply_grade(img, mask, *params)
        old = _old_grade(img, *params, mask=mask)
        d = (new - old).abs().max().item()
        check(f"fused grade == step-by-step {params[:7]}", d < 1e-6, f"max|d|={d:.2e}")

    same = bg._apply_grade(img, None, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, False, False)
    check("identity params return the input unchanged", torch.equal(same, img))

    neg = torch.zeros(1, 1, 1, 3)
    off = bg._apply_grade(neg, None, 0, 1, 0, 1, 1, -0.2, 2.2, False, False)
    on = bg._apply_grade(neg, None, 0, 1, 0, 1, 1, -0.2, 2.2, False, True)
    check("clamp_black OFF keeps negatives through gamma",
          abs(off.flatten()[0].item() + 0.2) < 1e-6, str(off.flatten().tolist()))
    check("clamp_black ON zeroes them", on.min().item() == 0.0, str(on.flatten().tolist()))

    try:
        out = bg.BatGrade().grade(torch.rand(10, 8, 8, 3), 0, 1, 0, 1.2, 1, 0, 1,
                                  False, False, 0, mask=torch.rand(5, 8, 8))
        check("mask batch 5 vs image batch 10 doesn't crash",
              tuple(out["result"][0].shape) == (10, 8, 8, 3))
    except Exception as e:  # noqa: BLE001 — the crash IS the regression
        check("mask batch 5 vs image batch 10 doesn't crash", False, repr(e))

    ui = bag.BatAnimatedGrade().grade(torch.rand(300, 4, 4, 3), "{}")["ui"]
    check("300 frames → ≤240 preview thumbnails", len(ui["frames"]) <= 240,
          f"{len(ui['frames'])}")


# ─── JS source pins ──────────────────────────────────────────────────────────

def test_js_sources():
    print("\njs sources")
    def src(name):
        return open(os.path.join(PACK, "web", name), encoding="utf-8").read()
    for name in ("bat_animated_crop.js", "bat_animated_grade.js"):
        s = src(name)
        m = re.search(r'root\.addEventListener\("keydown".*?\n    \}\);', s, re.S)
        tail = m.group(0) if m else ""
        check(f"{name}: handled keys stopPropagation (Delete used to delete the node)",
              "e.stopPropagation()" in tail)
    for name in ("bat_grade.js", "bat_animated_grade.js"):
        check(f"{name}: preview no longer zeroes negatives before gamma",
              "< 0 ? 0 : Math.pow" not in src(name))
    for name in ("bat_crop.js", "bat_animated_crop.js"):
        s = src(name)
        check(f"{name}: info bar floors like _snap",
              "Math.round(cw / snap) * snap" not in s
              and "Math.round(r.w / snap) * snap" not in s
              and "Math.floor(" in s)


def main():
    bc, bu, bac, bg, bag = load_pack_modules()
    test_rotated_round_trip(bc, bu)
    test_animated_crop(bu, bac)
    test_uncrop(bc, bu)
    test_grade(bg, bag)
    test_js_sources()
    if _failures:
        print(f"\n{len(_failures)} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
