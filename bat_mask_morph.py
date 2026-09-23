"""
Bat_GrowMask / Bat_ErodeMask — fast morphological grow (dilate) and
erode of a MASK, with an optional soft edge.

MASK convention (matches the rest of the pack, e.g. bat_roto): a
``(N, H, W)`` float32 tensor in [0, 1]. A single un-batched ``(H, W)``
mask is also accepted and normalised to ``(1, H, W)``.

Why cv2: ``cv2.dilate`` / ``cv2.erode`` are SIMD/threaded C, orders of
magnitude faster than a torch max-pool or a Python loop, and operate
directly on float32 so we skip any 0-255 round-trip. We build a
structuring element once and reuse it across the whole batch, then
morph each frame with its own cv2 call.

(An earlier version of this docstring described stacking the batch
into a single tall ``(N*H, W)`` image to morph it in one call. The
code does NOT do that, deliberately: the kernel would leak across the
frame seams and contaminate the top/bottom rows of adjacent frames.
See the comment in ``_morph_cv2`` — N C-level calls are negligible
next to any Python-level per-pixel work.)

Falls back to a torch max/min-pool implementation if cv2 is missing —
still fully vectorised (no Python per-pixel loop), just a touch slower.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("[Bat_MaskMorph]")

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("cv2 not installed — Bat_GrowMask/Bat_ErodeMask will "
                   "fall back to a torch pooling implementation (slower).")


# ─── helpers ─────────────────────────────────────────────────────────────────


def _as_batch(mask: torch.Tensor) -> torch.Tensor:
    """Normalise a MASK to (N, H, W) float32 on the CPU as a contiguous
    tensor. Accepts (H, W) or (N, H, W)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        # Some upstreams hand a (N, H, W, 1) or (N, 1, H, W) mask — squeeze
        # the trailing/leading singleton channel.
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
    return mask.to(dtype=torch.float32).cpu().contiguous()


def _kernel(radius: int, shape: str) -> np.ndarray:
    """Build a structuring element of the given pixel radius. cv2 kernels
    are (2r+1) square; 'disc' uses an ellipse so growth is rounded rather
    than boxy (what you almost always want for soft mattes)."""
    size = 2 * radius + 1
    if shape == "square":
        return np.ones((size, size), dtype=np.uint8)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _torch_disc(radius: int, shape: str, device) -> torch.Tensor:
    """(1,1,k,k) float kernel used to gate the pooling fallback so 'disc'
    stays round. Values are 1 inside the element, 0 outside."""
    size = 2 * radius + 1
    if shape == "square":
        return torch.ones((size, size), dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device) - radius,
        torch.arange(size, device=device) - radius,
        indexing="ij",
    )
    return ((yy * yy + xx * xx) <= (radius * radius + 0.5)).to(torch.float32)


# ─── large-radius disc ───────────────────────────────────────────────────────
#
# cv2 morphs a non-rectangular kernel by visiting every tap, so the disc costs
# O(r²) per pixel: 1080p, one frame, r=64 ≈ 0.4–1.1 s, r=128 ≈ 1.4–4 s,
# r=256 ≈ 6–20 s (the square stays ~15–50 ms at any radius — it is separable).
# Two EXACT replacements, both reproducing cv2's MORPH_ELLIPSE output bit for
# bit (tests/verify_mask_morph.py checks thousands of random cases):
#
# * binary masks — a distance transform, cost independent of the radius;
# * soft masks — the disc split into its rows: dilating by row half-width w
#   is built up incrementally from w-1 (two maxima), then each kernel row
#   contributes one shifted maximum. O(r) per pixel instead of O(r²).
#
# The obvious shortcut — iterating a small disc — is NOT used: its polygonal
# shape differs from the true disc by up to 0.07 (r=16) … 0.65 (r=256) on a
# feathered edge, which is a visibly different matte.
#
# Below these radii the plain kernel is still the fastest, so small grows —
# the default is 4 — go through exactly the code they always did.
_EDT_MIN_RADIUS = 24
_ROWS_MIN_RADIUS = 48
# "Binary" allows this much slop, so a mask that went through a resize or a
# float round-trip still takes the fast path (its output is then exactly 0/1).
_BINARY_TOL = 1e-3


def _is_near_binary(arr: np.ndarray) -> bool:
    return bool(np.all((arr <= _BINARY_TOL) | (arr >= 1.0 - _BINARY_TOL)))


def _ellipse_half_widths(radius: int) -> list:
    """Half-width of each row of cv2's MORPH_ELLIPSE (2r+1) kernel, top to
    bottom. Read off the real kernel so the fast paths can't drift from it."""
    k = _kernel(radius, "disc")
    return [(int(row.sum()) - 1) // 2 for row in k]


def _dilate_binary_edt(fg: np.ndarray, radius: int) -> np.ndarray:
    """Exact binary dilation of `fg` (H, W bool) by the MORPH_ELLIPSE kernel.

    That kernel is not a Euclidean disc: row dy spans |dx| <= round(√(r²-dy²)),
    i.e. (|dx|-½)² + dy² <= r². So distances are taken on a grid doubled in
    both axes, with each foreground pixel widened to a 3-sample horizontal
    run: from a query sample, the nearest sample of a pixel dx columns away is
    2|dx|-1 half-steps off (0 in the same column), which makes the threshold
    at 2r exactly the kernel's membership test. DIST_MASK_PRECISE is exact.
    Outside the frame counts as background, as BORDER_CONSTANT 0 did.
    """
    h, w = fg.shape
    if not fg.any():
        return np.zeros((h, w), dtype=bool)
    src = np.ones((2 * h - 1, 2 * w + 1), dtype=np.uint8)
    even = src[0::2]
    even[:, 0:-1:2][fg] = 0
    even[:, 1::2][fg] = 0
    even[:, 2::2][fg] = 0
    dist = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist[0::2, 1::2] <= 2 * radius


def _morph_disc_rows(frame: np.ndarray, radius: int, grow: bool) -> np.ndarray:
    """Exact MORPH_ELLIPSE dilate/erode of one float32 frame, O(r) per pixel."""
    half = _ellipse_half_widths(radius)
    bv = 0.0 if grow else 1.0
    red = np.maximum if grow else np.minimum
    h, w = frame.shape
    r = radius
    p = cv2.copyMakeBorder(frame, r, r, r, r, cv2.BORDER_CONSTANT, value=bv)
    row = p[:, r:r + w].copy()   # running row dilation, half-width `cur`
    out = np.full((h, w), bv, dtype=np.float32)
    cur = 0
    for hw in sorted(set(half)):
        while cur < hw:
            cur += 1
            red(row, p[:, r - cur:r - cur + w], out=row)
            red(row, p[:, r + cur:r + cur + w], out=row)
        for i, k_hw in enumerate(half):
            if k_hw == hw:
                red(out, row[i:i + h], out=out)
    return out


def _morph_cv2(batch: np.ndarray, radius: int, shape: str, grow: bool) -> np.ndarray:
    """Dilate (grow) or erode every frame in a (N, H, W) float32 array."""
    n, h, w = batch.shape
    if shape == "disc" and radius >= _EDT_MIN_RADIUS and _is_near_binary(batch):
        out = np.empty_like(batch)
        for i in range(n):
            fg = batch[i] > 0.5
            # Erosion is the complement of dilating the background; outside
            # the frame then counts as foreground, as borderValue 1.0 did.
            hit = _dilate_binary_edt(fg, radius) if grow else ~_dilate_binary_edt(~fg, radius)
            out[i] = hit
        return out
    if shape == "disc" and radius >= _ROWS_MIN_RADIUS:
        out = np.empty_like(batch)
        for i in range(n):
            out[i] = _morph_disc_rows(batch[i], radius, grow)
        return out

    kernel = _kernel(radius, shape)
    op = cv2.dilate if grow else cv2.erode
    # Process each frame independently. Stacking into one tall image would
    # let the kernel leak across frame seams, so we keep it per-frame — the
    # cost is N C-level calls, which is still negligible next to any Python
    # per-pixel work. iterations=1 with a full-radius kernel gives an exact
    # radius (vs. iterating a 3x3, which yields a diamond).
    out = np.empty_like(batch)
    for i in range(n):
        out[i] = op(batch[i], kernel, iterations=1,
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0.0 if grow else 1.0)
    return out


def _morph_torch(batch: torch.Tensor, radius: int, shape: str, grow: bool) -> torch.Tensor:
    """Vectorised pooling fallback. Grow = max-pool, erode = min-pool
    (min-pool = -maxpool(-x)). A disc kernel masks the neighbourhood so
    only in-element pixels contend, matching the cv2 ellipse."""
    x = batch.unsqueeze(1)  # (N,1,H,W)
    size = 2 * radius + 1
    pad = radius
    elem = _torch_disc(radius, shape, x.device)  # (k,k)

    if shape == "square":
        if grow:
            return F.max_pool2d(x, size, stride=1, padding=pad).squeeze(1)
        return (-F.max_pool2d(-x, size, stride=1, padding=pad)).squeeze(1)

    # Disc: reduce row by row. This used to unfold into (N, k*k, H*W) patches,
    # which is k² copies of the batch — ~9 GB for one 1080p frame at r=16, so
    # any real radius ran out of memory. Instead build the horizontal max of
    # half-width w incrementally from w-1 and take one shifted max per kernel
    # row: the same taps (the element above), a few frames of memory.
    # Erode is the same thing on the negated mask (min = -max(-x)).
    xp = F.pad(x, (pad, pad, pad, pad), value=0.0 if grow else 1.0)
    if not grow:
        xp = -xp
    h, w = batch.shape[-2], batch.shape[-1]
    half = [(int(v) - 1) // 2 for v in elem.sum(dim=1).tolist()]
    row = xp[..., :, pad:pad + w].clone()
    out = None
    cur = 0
    for hw in sorted(set(half)):
        while cur < hw:
            cur += 1
            row = torch.maximum(row, xp[..., :, pad - cur:pad - cur + w])
            row = torch.maximum(row, xp[..., :, pad + cur:pad + cur + w])
        for i, k_hw in enumerate(half):
            if k_hw == hw:
                part = row[..., i:i + h, :]
                out = part.clone() if out is None else torch.maximum(out, part)
    red = out if grow else -out
    return red.reshape(batch.shape)


def _feather(batch: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-soften the mask edge. Applied after the morph so it
    softens the grown/eroded silhouette."""
    if radius <= 0:
        return batch
    if _HAS_CV2:
        # Kernel must be wide enough for the sigma or the blur does nothing:
        # int(round(radius))*2+1 gave k=1 for any radius < 0.5, so a sub-pixel
        # feather was silently a no-op.
        k = max(3, int(2 * math.ceil(3.0 * float(radius)) + 1))
        out = np.empty_like(batch)
        for i in range(batch.shape[0]):
            out[i] = cv2.GaussianBlur(batch[i], (k, k), float(radius))
        return out
    # torch gaussian-ish fallback: box blur a few times ≈ gaussian.
    t = torch.from_numpy(batch).unsqueeze(1)
    # Kernel must be wide enough for the sigma or the blur does nothing:
    # int(round(radius))*2+1 gave k=1 for any radius < 0.5.
    k = max(3, int(2 * math.ceil(3.0 * float(radius)) + 1))
    pad = k // 2
    w = torch.ones((1, 1, k, k)) / (k * k)
    for _ in range(3):
        t = F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="replicate"), w)
    return t.squeeze(1).numpy()


def _run(mask: torch.Tensor, amount: int, shape: str, feather: float, grow: bool):
    batch = _as_batch(mask)
    if amount > 0:
        if _HAS_CV2:
            arr = _morph_cv2(batch.numpy(), amount, shape, grow)
        else:
            arr = _morph_torch(batch, amount, shape, grow).numpy()
    else:
        arr = batch.numpy()
    arr = _feather(arr, feather)
    np.clip(arr, 0.0, 1.0, out=arr)
    return (torch.from_numpy(np.ascontiguousarray(arr)),)


# ─── nodes ───────────────────────────────────────────────────────────────────


class BatGrowMask:
    """Grow (dilate) a mask outward by N pixels, optional soft edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 4, "min": 0, "max": 512, "step": 1}),
                "shape": (["disc", "square"], {"default": "disc"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "BAT/Mask"
    DESCRIPTION = (
        "Grow (dilate) a mask outward by `amount` pixels. `disc` gives a "
        "rounded edge, `square` a boxy one. `feather` softens the result. "
        "Batched and cv2-accelerated."
    )

    def run(self, mask, amount, shape, feather):
        return _run(mask, int(amount), shape, float(feather), grow=True)


class BatErodeMask:
    """Erode (shrink) a mask inward by N pixels, optional soft edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 4, "min": 0, "max": 512, "step": 1}),
                "shape": (["disc", "square"], {"default": "disc"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "BAT/Mask"
    DESCRIPTION = (
        "Erode (shrink) a mask inward by `amount` pixels. `disc` gives a "
        "rounded edge, `square` a boxy one. `feather` softens the result. "
        "Batched and cv2-accelerated."
    )

    def run(self, mask, amount, shape, feather):
        return _run(mask, int(amount), shape, float(feather), grow=False)
