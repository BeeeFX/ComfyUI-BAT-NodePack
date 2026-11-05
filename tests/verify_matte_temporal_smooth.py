"""Checks for bat_matte_temporal_smooth.smooth_matte.

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_matte_temporal_smooth.py [PLATE_DIR ALPHA_NPZ]

CPU only for the synthetic checks. With a plate folder (JPEG/PNG frames) and
an npz holding a uint8 alpha batch under "a" (the CorridorKey2 sweep's
format), it also re-measures the real-shot flicker reduction.
"""
import glob
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
pkg = types.ModuleType("batpack")
pkg.__path__ = [PACK]
sys.modules["batpack"] = pkg
from batpack.bat_matte_temporal_smooth import smooth_matte, GATE_BLUR, MOTION_SPREAD  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok or not detail else f"\n         {detail}"))
    if not ok:
        FAILURES.append(label)


def reference(plate, alpha, radius, tol, falloff=0.6):
    """The formula written out plainly in numpy (replicate-padded box blur)."""
    T, H, W = alpha.shape
    p = GATE_BLUR // 2
    padded = np.pad(plate, ((0, 0), (p, p), (p, p), (0, 0)), mode="edge")
    G = np.zeros_like(plate)
    for dy in range(GATE_BLUR):
        for dx in range(GATE_BLUR):
            G += padded[:, dy:dy + H, dx:dx + W]
    G /= GATE_BLUR ** 2
    out = np.empty_like(alpha)
    for t in range(T):
        acc, ws = alpha[t].copy(), np.ones((H, W))
        for s in range(max(0, t - radius), min(T, t + radius + 1)):
            if s == t:
                continue
            d = np.abs(G[s] - G[t]).max(-1)
            q = MOTION_SPREAD // 2       # max-pool; torch pads with -inf, so the border never wins
            dp = np.pad(d, q, mode="constant", constant_values=-np.inf)
            d = np.max([dp[dy:dy + H, dx:dx + W] for dy in range(MOTION_SPREAD) for dx in range(MOTION_SPREAD)], 0)
            w = np.exp(-d / tol) * falloff ** (abs(s - t) - 1)
            acc += w * alpha[s]; ws += w
        out[t] = acc / ws
    return out


rng = np.random.default_rng(3)

print("matches the plain-numpy formula")
plate = rng.uniform(0, 1, (9, 24, 32, 3)).astype(np.float64)
plate[4:] += 0.2                                   # a jump half way: gated
alpha = rng.uniform(0, 1, (9, 24, 32))
got, _ = smooth_matte(torch.from_numpy(plate).float(), torch.from_numpy(alpha).float(), radius=2,
                      motion_tolerance=0.05)
want = reference(plate, alpha, 2, 0.05)
check("max |torch - numpy| < 1e-5", np.abs(got.numpy() - want).max() < 1e-5,
      f"{np.abs(got.numpy() - want).max():.2e}")

print("static plate: flicker goes, the average stays")
T, H, W = 40, 32, 48
still = np.repeat(rng.uniform(.2, .8, (1, H, W, 3)), T, 0)
base = np.zeros((H, W)); base[:, W // 2:] = 1
noisy = np.clip(base + rng.normal(0, .15, (T, H, W)) * (np.abs(np.arange(W) - W / 2) < 4), 0, 1)
out, _ = smooth_matte(torch.from_numpy(still).float(), torch.from_numpy(noisy).float(), radius=2,
                      motion_tolerance=0.04)
flick = lambda x: np.abs(np.diff(x, axis=0)).mean()
check(f"frame-to-frame change {flick(noisy):.4f} -> {flick(out.numpy()):.4f} (< half)",
      flick(out.numpy()) < 0.5 * flick(noisy))
check("mean alpha preserved", abs(out.numpy().mean() - noisy.mean()) < 0.01)

print("moving object: no trail")
T, H, W = 20, 64, 96
plate = np.zeros((T, H, W, 3)) + [0.8, 0.45, 0.3]      # orange wall
alpha = np.zeros((T, H, W))
yy, xx = np.mgrid[0:H, 0:W]
for t in range(T):
    disc = (xx - (20 + 3 * t)) ** 2 + (yy - 32) ** 2 < 12 ** 2
    plate[t][disc] = [0.75, 0.55, 0.45]                 # skin: same-ish luma as the wall
    alpha[t][disc] = 1
out, _ = smooth_matte(torch.from_numpy(plate).float(), torch.from_numpy(alpha).float(), radius=2,
                      motion_tolerance=0.04)
err = np.abs(out.numpy() - alpha)[2:-2]
check(f"moving edge kept (max error {err.max():.3f} < 0.25, mean {err.mean():.4f})",
      err.max() < 0.25 and err.mean() < 0.01)

print("foreground: same weights, range untouched")
hdr = np.repeat(rng.uniform(0, 4, (1, 16, 16, 3)), 6, 0)
fg_out = smooth_matte(torch.from_numpy(np.repeat(rng.uniform(0, 1, (1, 16, 16, 3)), 6, 0)).float(),
                      torch.ones(6, 16, 16), foreground=torch.from_numpy(hdr).float())[1]
check("HDR foreground above 1.0 survives", abs(fg_out.max().item() - hdr.max()) < 1e-4)

print("errors")
for label, args in (("frame-count mismatch", (torch.zeros(5, 8, 8, 3), torch.zeros(4, 8, 8))),
                    ("foreground size mismatch", (torch.zeros(4, 8, 8, 3), torch.zeros(4, 8, 8),
                                                  torch.zeros(4, 9, 8, 3)))):
    try:
        smooth_matte(*args)
        check(label, False)
    except ValueError:
        check(label, True)

if len(sys.argv) == 3:
    from PIL import Image
    plate_dir, alpha_npz = sys.argv[1:3]
    files = sorted(glob.glob(os.path.join(plate_dir, "*.jpg")) + glob.glob(os.path.join(plate_dir, "*.png")))
    P = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255 for f in files]))
    A = torch.from_numpy(np.load(alpha_npz)["a"].astype(np.float32) / 255)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out, _ = smooth_matte(P, A, radius=2, motion_tolerance=0.04, device=dev)
    blink = lambda a: (((a[1:-1] > .5) != (a[:-2] > .5)) & ((a[1:-1] > .5) != (a[2:] > .5))).reshape(len(a) - 2, -1).sum(1).float().mean().item()
    b0, b1 = blink(A), blink(out)
    print(f"real shot ({len(A)} frames on {dev})")
    check(f"blinking pixels/frame {b0:.0f} -> {b1:.0f} (at least 5x fewer)", b1 * 5 < b0)

print()
print("All checks passed." if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
