"""Checks for 🦇 Merge Mattes (bat_merge_mattes.py). CPU only.

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_merge_mattes.py [PRIMARY_NPZ SECONDARY_NPZ]

Synthetic cases pin each promise the node makes. With two uint8 alpha batches
saved under "a" (the CorridorKey 2 + MEMatte pair it was built on), it also
re-checks the bounds and that the merge flickers no more than the primary.
"""
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
from batpack.bat_merge_mattes import merge_mattes, trust_zone  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok or not detail else f"\n         {detail}"))
    if not ok:
        FAILS.append(label)


H, W = 200, 300
yy, xx = np.mgrid[0:H, 0:W]
disc = lambda cx, cy, r: ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r).astype(np.float32)
T = lambda *a: torch.from_numpy(np.stack(a))

print("bounds")
rng = np.random.default_rng(0)
p, s = rng.uniform(0, 1, (3, H, W)).astype(np.float32), rng.uniform(0, 1, (3, H, W)).astype(np.float32)
m, added, removed, zone = merge_mattes(torch.from_numpy(p), torch.from_numpy(s))
check("never below the primary", bool((m.numpy() >= p - 1e-7).all()))
check("never above max(primary, secondary) — no add/screen brightening", bool((m.numpy() <= np.maximum(p, s) + 1e-7).all()))
check("removed is empty with cleanup off", float(removed.max()) == 0.0)

print("holes")
subject = disc(150, 100, 70)
hole = disc(150, 100, 8)                                  # small hole in the primary
prim = subject * (1 - hole)
m, *_ = merge_mattes(T(prim), T(subject), fill_holes=16)
check("a small hole is filled where the secondary sees subject", m[0][100, 150].item() > 0.99)
m, *_ = merge_mattes(T(prim), T(prim), fill_holes=16)
check("...but never invented when the secondary sees background there too", m[0][100, 150].item() == 0.0)
m, *_ = merge_mattes(T(prim), T(subject), fill_holes=0, edge_reach=0)
check("fill_holes 0 and edge_reach 0: the hole stays", m[0][100, 150].item() == 0.0)
m, *_ = merge_mattes(T(prim), T(subject), fill_holes=0, edge_reach=12)
check("edge_reach alone partly fills a hole smaller than it (a hole is an edge seen from inside)",
      0.0 < m[0][100, 150].item() < 1.0, m[0][100, 150].item())

print("edge reach")
prim = disc(150, 100, 50)
sec = disc(150, 100, 60)                                   # 10 px fuller edge
blob = disc(20, 20, 6)                                     # stray blob far away
m, added, _, zone = merge_mattes(T(prim), T(np.maximum(sec, blob)), edge_reach=12, fill_holes=0)
check("detail 5 px outside the edge comes through (partly)", 0.3 < m[0][100, 150 + 55].item() < 1.0, m[0][100, 205].item())
check("nothing beyond edge_reach", m[0][100, 150 + 63].item() == 0.0)
check("a stray blob far away is ignored", m[0][20, 20].item() == 0.0)
check("zone is 1 inside the primary's subject", zone[0][100, 150].item() == 1.0)
z = trust_zone(prim, 0.5, 0, 12)
check("zone falls off linearly to 0 at edge_reach", abs(z[100, 150 + 50 + 6] - 0.5) < 0.1 and z[100, 150 + 50 + 13] == 0.0,
      (z[100, 206], z[100, 213]))

print("primary-only content")
extra = disc(260, 40, 15)                                  # something only the primary found
m, *_ = merge_mattes(T(np.maximum(prim, extra)), T(sec))
check("kept", m[0][40, 260].item() == 1.0)

print("cleanup")
ghost = (disc(40, 150, 25) * 0.4).astype(np.float32)       # semi-transparent haze, far from the subject
wisp = np.zeros((H, W), np.float32); wisp[100, 211:214] = 0.4   # soft strand just outside the secondary's edge
solid_extra = disc(260, 40, 15)
prim = np.maximum.reduce([disc(150, 100, 50), ghost, wisp, solid_extra])
m0, *_ = merge_mattes(T(prim), T(sec))
m1, _, removed, _ = merge_mattes(T(prim), T(sec), cleanup=1.0, cleanup_distance=12)
check("off: the ghost stays", m0[0][150, 40].item() > 0.35)
check("on: the ghost is removed", m1[0][150, 40].item() < 0.01)
check("on: solid primary-only content is never touched", m1[0][40, 260].item() == 1.0)
check("on: a wisp near the secondary's edge survives", m1[0][100, 212].item() > 0.35)
check("removed output shows what went", removed[0][150, 40].item() > 0.35)

print("batches")
m, *_ = merge_mattes(torch.from_numpy(np.stack([prim] * 4)), T(sec))
check("one secondary mask applies to every frame", m.shape[0] == 4)
try:
    merge_mattes(torch.zeros(3, 10, 10), torch.zeros(2, 10, 10))
    check("mismatched frame counts refused", False)
except ValueError:
    check("mismatched frame counts refused", True)

if len(sys.argv) == 3:
    print("real pair")
    P = torch.from_numpy(np.load(sys.argv[1])["a"].astype(np.float32) / 255)
    S = torch.from_numpy(np.load(sys.argv[2])["a"].astype(np.float32) / 255)
    m, *_ = merge_mattes(P, S)
    blink = lambda a: (((a[1:-1] > .5) != (a[:-2] > .5)) & ((a[1:-1] > .5) != (a[2:] > .5))).reshape(len(a) - 2, -1).sum(1).float().mean().item()
    check("bounds hold on real mattes", bool((m >= P - 1e-6).all() and (m <= torch.maximum(P, S) + 1e-6).all()))
    b0, b1 = blink(P), blink(m)
    check(f"merge flickers no more than the primary ({b0:.0f} -> {b1:.0f} blinks/frame)", b1 <= b0)

print()
print("All checks passed." if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
