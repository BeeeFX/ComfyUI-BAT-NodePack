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


def _morph_cv2(batch: np.ndarray, radius: int, shape: str, grow: bool) -> np.ndarray:
    """Dilate (grow) or erode every frame in a (N, H, W) float32 array."""
    n, h, w = batch.shape
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

    # Disc: unfold into (N, k*k, H*W) patches, mask out-of-element taps to
    # -inf (grow) / +inf (erode), then reduce. One big op, no Python loop.
    if grow:
        xp = F.pad(x, (pad, pad, pad, pad), value=0.0)
    else:
        xp = F.pad(x, (pad, pad, pad, pad), value=1.0)
    patches = F.unfold(xp, kernel_size=size)  # (N, k*k, H*W)
    n, _, l = patches.shape
    patches = patches.view(n, size * size, l)
    flat = elem.reshape(-1).to(torch.bool)  # (k*k,)
    if grow:
        patches = patches.masked_fill(~flat[None, :, None], float("-inf"))
        red = patches.max(dim=1).values
    else:
        patches = patches.masked_fill(~flat[None, :, None], float("inf"))
        red = patches.min(dim=1).values
    return red.view(batch.shape)


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
    CATEGORY = "BAT/mask"
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
    CATEGORY = "BAT/mask"
    DESCRIPTION = (
        "Erode (shrink) a mask inward by `amount` pixels. `disc` gives a "
        "rounded edge, `square` a boxy one. `feather` softens the result. "
        "Batched and cv2-accelerated."
    )

    def run(self, mask, amount, shape, feather):
        return _run(mask, int(amount), shape, float(feather), grow=False)
