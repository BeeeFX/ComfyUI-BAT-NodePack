"""
Bat_Grade — Nuke-style colour grading with a live canvas preview.

Mirrors Nuke's Grade node math:
    base    = (in - blackpoint) / max(whitepoint - blackpoint, 1e-6)
    leveled = base * (gain - lift) + lift
    out     = (leveled * multiply + offset) ** (1 / max(gamma, 1e-6))
              (negative values skip the gamma and pass through linear)
    out     = clamp(out, max=1) if clamp_white; clamp(out, min=0) if clamp_black

An optional MASK input gates the grade: where mask == 1 the grade
applies fully, where mask == 0 the pixels pass through unchanged
(linear mix between them).

The live preview on the node body is driven by the sibling
``web/bat_grade.js`` — Python pushes a downscaled thumbnail of the
input image (and, if connected, the mask) back through the
``{"ui": ...}`` channel on every run, so the JS canvas can apply
the grade formula in real time as the artist drags the sliders.
The actual graded output is computed here on the next run.
"""

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .bat_crop import _broadcast_mask_to_n
from .bat_hdr_preview import hdr_tile


def _first(t: torch.Tensor) -> torch.Tensor:
    return t[0:1] if t is not None and t.shape[0] > 1 else t


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = 384, quality: int = 80) -> str:
    """Downscaled base64 JPEG of an (H,W,3) uint8 numpy array."""
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_png_l(arr_hw: np.ndarray, max_dim: int = 384) -> str:
    """Downscaled base64 PNG of an (H,W) uint8 mask array."""
    im = Image.fromarray(arr_hw, "L")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _apply_grade(
    image: torch.Tensor,      # (N, H, W, 3) float in [0,1]
    mask: torch.Tensor,       # (N, H, W)    float in [0,1], or None
    blackpoint: float,
    whitepoint: float,
    lift: float,
    gain: float,
    multiply: float,
    offset: float,
    gamma: float,
    clamp_white: bool,
    clamp_black: bool,
) -> torch.Tensor:
    """Apply Nuke's Grade formula. See module docstring for the math."""
    img = image.to(torch.float32)
    wp_minus_bp = max(whitepoint - blackpoint, 1e-6)
    g = max(gamma, 1e-6)

    # Pass-through params (the node defaults) are an exact identity; skip the
    # full-batch arithmetic entirely.
    if (blackpoint == 0.0 and whitepoint == 1.0 and lift == 0.0 and gain == 1.0
            and multiply == 1.0 and offset == 0.0 and g == 1.0
            and not clamp_white and not clamp_black):
        return img

    # The levels stage is affine in `in`, so fold it into ONE multiply-add:
    #   ((in - bp)/(wp - bp) * (gain - lift) + lift) * multiply + offset
    #   = in * A + B
    # and do everything after it in place. The step-by-step version kept
    # base / leveled / out alive together: ~6x the batch at peak versus ~2x.
    a = (gain - lift) / wp_minus_bp * multiply
    b = (lift - blackpoint * (gain - lift) / wp_minus_bp) * multiply + offset
    out = img * a
    out.add_(b)

    # pow on negatives is undefined. Negatives pass through LINEAR (Nuke's
    # behaviour) rather than being zeroed, so clamp_black is a real toggle:
    # it used to be a no-op because this step already clamped them to 0.
    if g != 1.0:
        if clamp_black:
            out.clamp_(min=0.0).pow_(1.0 / g)
        else:
            neg = out < 0
            neg_vals = out[neg]
            out.clamp_(min=0.0).pow_(1.0 / g)
            out[neg] = neg_vals

    if clamp_white:
        out.clamp_(max=1.0)
    if clamp_black:
        out.clamp_(min=0.0)

    if mask is not None:
        # A mask batch that is neither 1 nor N used to crash the mix below
        # ("size of tensor a (10) must match … (5)"); hold its last frame /
        # truncate, the same way Bat_Crop brings a mask to the image batch.
        n = img.shape[0]
        if mask.shape[0] not in (1, n):
            mask = _broadcast_mask_to_n(mask, n)
        # mask: (N, H, W) → (N, H, W, 1) for broadcasting against (N,H,W,3)
        m = mask.to(torch.float32).unsqueeze(-1).clamp(0, 1)
        # Resize mask to image spatial size if they differ (e.g. mask is
        # 1/2 res after a downscale).
        if m.shape[1:3] != img.shape[1:3]:
            # NCHW for interpolate
            m_nchw = m.permute(0, 3, 1, 2)
            m_nchw = torch.nn.functional.interpolate(
                m_nchw, size=img.shape[1:3], mode="bilinear", align_corners=False,
            )
            m = m_nchw.permute(0, 2, 3, 1)
        # out*m + img*(1-m), in place.
        out.sub_(img).mul_(m).add_(img)
    return out


class BatGrade:
    """Nuke-style grade with a live canvas preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":       ("IMAGE",),
                "blackpoint":  ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "whitepoint":  ("FLOAT", {"default": 1.0, "min": 0.001, "max": 4.0, "step": 0.001}),
                "lift":        ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "gain":        ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.001}),
                "multiply":    ("FLOAT", {"default": 1.0, "min": 0.0,  "max": 4.0, "step": 0.001}),
                "offset":      ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "gamma":       ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.001}),
                "clamp_white": ("BOOLEAN", {"default": False}),
                "clamp_black": ("BOOLEAN", {"default": False}),
                # Which frame of the input batch the live preview uses.
                # Doesn't affect the actual output (the grade still
                # applies to every frame). Lets the artist pick a more
                # representative frame than the default frame 0 — e.g.
                # the middle of a shot where the action is.
                "preview_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grade"
    CATEGORY = "BAT/Colour"
    DESCRIPTION = (
        "Nuke-style grade with blackpoint/whitepoint, lift/gain, "
        "multiply, offset, and gamma. Optional mask gates the grade. "
        "Live canvas preview on the node body."
    )

    def grade(self, image, blackpoint, whitepoint, lift, gain, multiply,
              offset, gamma, clamp_white, clamp_black, preview_frame=0, mask=None):
        out = _apply_grade(
            image=image,
            mask=mask,
            blackpoint=float(blackpoint),
            whitepoint=float(whitepoint),
            lift=float(lift),
            gain=float(gain),
            multiply=float(multiply),
            offset=float(offset),
            gamma=float(gamma),
            clamp_white=bool(clamp_white),
            clamp_black=bool(clamp_black),
        )

        # Push a preview thumbnail of the SELECTED frame to the JS
        # canvas so it can apply the grade live as sliders move. The
        # frame index is clamped to the batch length so an aspirational
        # widget value (say "frame 200" on a 24-frame clip) falls back
        # to the last frame instead of erroring.
        idx = max(0, min(int(preview_frame), image.shape[0] - 1))
        preview_img = image[idx:idx + 1]
        img_u8 = (preview_img[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        ui = {
            "input_image": [_b64_jpeg(img_u8)],
            "w": [int(image.shape[2])],
            "h": [int(image.shape[1])],
            "preview_frame": [int(idx)],
        }

        # Second, high-precision copy of the same frame for the viewer's
        # Inspect mode. The JPEG above is clamped, 8-bit and lossy, so a
        # 16-bit and an 8-bit source are indistinguishable through it; this
        # tile keeps the source's real quantisation steps so a view exposure
        # can reveal banding. See bat_hdr_preview.py. None on failure, and
        # the JS falls back to the JPEG.
        tile = hdr_tile(preview_img[0])
        if tile is not None:
            ui["hdr_tile"] = [tile]
        if mask is not None:
            # Same-frame index for the mask, again clamped to its own
            # batch length (mask may be a single-frame mask used across
            # the whole batch).
            midx = max(0, min(int(preview_frame), mask.shape[0] - 1))
            preview_mask = mask[midx:midx + 1]
            mask_u8 = (preview_mask[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            ui["input_mask"] = [_b64_png_l(mask_u8)]

        # No .cpu(): forcing the result to host memory cost a full device->host
        # copy (~750MB for a 300-frame 1080p batch) that the next node has to
        # copy straight back. ComfyUI handles device placement.
        return {"ui": ui, "result": (out,)}
