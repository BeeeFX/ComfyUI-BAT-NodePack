"""
Checks for 🦇 Grow Mask / 🦇 Erode Mask (bat_mask_morph.py).

The large-radius disc paths must reproduce cv2's MORPH_ELLIPSE output bit for
bit — they are speedups, not a new look — so every case below compares against
a plain `cv2.dilate` / `cv2.erode` with the same kernel and border, which is
what the node always did. The torch fallback is compared against its old
unfold formulation the same way.

Run from the pack root:  python tests/verify_mask_morph.py
"""

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load():
    spec = importlib.util.spec_from_file_location(
        "bat_mask_morph", os.path.join(PACK, "bat_mask_morph.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reference(frame, radius, shape, grow):
    """The node's original per-frame call."""
    import cv2
    k = (np.ones((2 * radius + 1,) * 2, np.uint8) if shape == "square"
         else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    op = cv2.dilate if grow else cv2.erode
    return op(frame, k, iterations=1, borderType=cv2.BORDER_CONSTANT,
              borderValue=0.0 if grow else 1.0)


def old_torch_disc(batch, radius, grow):
    """The fallback's former unfold implementation, kept here as the oracle."""
    import torch
    import torch.nn.functional as F
    x = batch.unsqueeze(1)
    size = 2 * radius + 1
    yy, xx = torch.meshgrid(torch.arange(size) - radius, torch.arange(size) - radius,
                            indexing="ij")
    elem = (yy * yy + xx * xx) <= (radius * radius + 0.5)
    xp = F.pad(x, (radius,) * 4, value=0.0 if grow else 1.0)
    patches = F.unfold(xp, kernel_size=size)
    flat = elem.reshape(-1)
    if grow:
        red = patches.masked_fill(~flat[None, :, None], float("-inf")).max(dim=1).values
    else:
        red = patches.masked_fill(~flat[None, :, None], float("inf")).min(dim=1).values
    return red.view(batch.shape)


def test_exact(mm):
    print("\nlarge-radius disc is bit-exact")
    rng = np.random.default_rng(7)
    bad_bin = bad_soft = n_bin = n_soft = 0
    for trial in range(24):
        h, w = (int(v) for v in rng.integers(6, 80, 2))
        density = [0.002, 0.03, 0.3, 0.9, 0.995][trial % 5]
        binary = (rng.random((1, h, w)) < density).astype(np.float32)
        if trial == 5:
            binary[:] = 0.0          # nothing to grow
        if trial == 6:
            binary[:] = 1.0          # nothing to erode
        soft = rng.random((1, h, w)).astype(np.float32)
        for r in (24, 25, 31, 40, 57, 64, 90):
            for grow in (True, False):
                n_bin += 1
                got = mm._morph_cv2(binary, r, "disc", grow)[0]
                bad_bin += not np.array_equal(got, reference(binary[0], r, "disc", grow))
        for r in (48, 49, 64, 77):
            for grow in (True, False):
                n_soft += 1
                got = mm._morph_cv2(soft, r, "disc", grow)[0]
                bad_soft += not np.array_equal(got, reference(soft[0], r, "disc", grow))
    check(f"binary masks, distance-transform path ({n_bin} cases)", bad_bin == 0,
          f"{bad_bin} differ")
    check(f"soft masks, row-decomposed path ({n_soft} cases)", bad_soft == 0,
          f"{bad_soft} differ")

    # Near-binary (a mask that went through a resize): snapped to exact 0/1,
    # so it may differ from the old output by at most the tolerance.
    nb = (rng.random((1, 50, 70)) < 0.2).astype(np.float32)
    nb = np.where(nb > 0.5, 1.0 - 4e-4, 3e-4).astype(np.float32)
    got = mm._morph_cv2(nb, 30, "disc", True)[0]
    diff = float(np.abs(got - reference(nb[0], 30, "disc", True)).max())
    check("near-binary mask stays within the 1e-3 tolerance", diff <= 1e-3, f"{diff}")

    # Small radii and squares never leave the original code path.
    for r, shape in ((4, "disc"), (23, "disc"), (100, "square")):
        m = (rng.random((1, 40, 40)) < 0.3).astype(np.float32)
        check(f"r={r} {shape} unchanged",
              np.array_equal(mm._morph_cv2(m, r, shape, True)[0],
                             reference(m[0], r, shape, True)))


def test_torch_fallback(mm):
    print("\ntorch fallback")
    import torch
    g = torch.Generator().manual_seed(3)
    bad = n = 0
    for r in (1, 2, 3, 5, 8, 11):
        for grow in (True, False):
            b = torch.rand((2, 23, 31), generator=g)
            n += 1
            bad += not torch.equal(mm._morph_torch(b, r, "disc", grow),
                                   old_torch_disc(b, r, grow))
    check(f"row-wise disc matches the old unfold version ({n} cases)", bad == 0,
          f"{bad} differ")
    # The old version needed (2r+1)^2 copies of the batch; this one a few.
    b = torch.zeros((1, 270, 480))
    b[0, 100:150, 200:260] = 1.0
    t = time.perf_counter()
    out = mm._morph_torch(b, 40, "disc", True)
    check("r=40 on a 480x270 frame runs without the unfold blow-up",
          out.shape == b.shape and float(out[0, 125, 170]) == 1.0 and float(out[0, 125, 150]) == 0.0,
          f"{time.perf_counter() - t:.2f}s")


def test_timing(mm):
    print("\ntiming (one 1080p frame, informational)")
    import cv2
    m = np.zeros((1080, 1920), np.float32)
    cv2.circle(m, (960, 540), 300, 1.0, -1)
    soft = cv2.GaussianBlur(m, (0, 0), 4)
    for label, frame in (("binary", m), ("soft", soft)):
        for r in (64, 128):
            t = time.perf_counter()
            ref = reference(frame, r, "disc", True)
            t_ref = time.perf_counter() - t
            t = time.perf_counter()
            got = mm._morph_cv2(frame[None], r, "disc", True)[0]
            t_new = time.perf_counter() - t
            print(f"  {label:6s} r={r:3d}  kernel {t_ref * 1000:6.0f} ms   "
                  f"now {t_new * 1000:5.0f} ms   same={np.array_equal(ref, got)}")


def main():
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"skip: {e}")
        return 0
    mm = load()
    test_exact(mm)
    test_torch_fallback(mm)
    if "--timing" in sys.argv:
        test_timing(mm)
    print()
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
