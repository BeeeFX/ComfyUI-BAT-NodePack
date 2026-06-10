"""
Bat_Crop — interactive axis-aligned crop, paired with Bat_Uncrop.

Independent implementation built on the BAT pack's own canvas-widget and
base64-preview-return patterns. The companion Bat_Uncrop reads the
BAT_CROP_INFO socket this node emits to place a processed crop back onto
its original plate at the original position and size.
"""

import base64
import math
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _first(img):
    return img[0:1] if img is not None and img.shape[0] > 1 else img


def _make_crop_grid(x, y, w, h, H, W, angle_deg, n, device, dtype):
    """Build a (n, h, w, 2) sampling grid that maps the output pixels of a
    rotated rect (centre (x+w/2, y+h/2), rotated by `angle_deg`) back into
    normalised source coords for grid_sample (align_corners=True)."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    a = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(a), math.sin(a)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    u = xs - (w - 1) / 2.0
    v = ys - (h - 1) / 2.0
    sx = cx + cos_t * u - sin_t * v
    sy = cy + sin_t * u + cos_t * v
    gx = sx / max(1, W - 1) * 2.0 - 1.0
    gy = sy / max(1, H - 1) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(n, h, w, 2)


def _rotated_crop(image, x, y, w, h, angle_deg):
    """Sample a rotated rect from `image` into an axis-aligned (h, w) output."""
    n, H, W, _ = image.shape
    grid = _make_crop_grid(x, y, w, h, H, W, angle_deg, n,
                           image.device, torch.float32)
    nchw = image.permute(0, 3, 1, 2).to(torch.float32)
    out = F.grid_sample(nchw, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.permute(0, 2, 3, 1).clamp(0, 1).contiguous()


def _rotated_crop_mask(mask, x, y, w, h, angle_deg):
    """Sample a rotated rect from a (n,H,W) mask into (n,h,w)."""
    n, H, W = mask.shape
    grid = _make_crop_grid(x, y, w, h, H, W, angle_deg, n,
                           mask.device, torch.float32)
    nchw = mask.unsqueeze(1).to(torch.float32)
    out = F.grid_sample(nchw, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.squeeze(1).clamp(0, 1).contiguous()


def _broadcast_mask_to_n(mask, n):
    """Bring a (M,H,W) mask up/down to (n,H,W) by repeat-or-truncate."""
    if mask.shape[0] == n:
        return mask
    if mask.shape[0] == 1:
        return mask.expand(n, -1, -1)
    if mask.shape[0] > n:
        return mask[:n]
    pad_n = n - mask.shape[0]
    return torch.cat([mask, mask[-1:].expand(pad_n, -1, -1)], dim=0)


def _rotated_rect_mask(H, W, cx, cy, rw, rh, angle_deg, n, device, dtype):
    """Per-pixel mask on (n, H, W) for a rotated rect of size (rw,rh) at (cx,cy)."""
    a = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(a), math.sin(a)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xs - cx
    dy = ys - cy
    # Inverse rotation: cos(-θ)*dx - sin(-θ)*dy, etc.
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy
    inside = ((u.abs() <= rw / 2.0) & (v.abs() <= rh / 2.0)).to(dtype)
    return inside.unsqueeze(0).expand(n, H, W).contiguous()


def _b64_jpeg(img_1hwc, max_dim=1024, quality=80):
    arr = (img_1hwc[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _snap(v, step):
    step = max(1, int(step))
    return max(step, int(round(v / step)) * step)


class BatCrop:
    """Axis-aligned drag crop with a paired-output channel for Bat_Uncrop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "crop_x":       ("INT", {"default": 0,   "min": 0, "max": 16384}),
                "crop_y":       ("INT", {"default": 0,   "min": 0, "max": 16384}),
                "crop_w":       ("INT", {"default": 512, "min": 1, "max": 16384}),
                "crop_h":       ("INT", {"default": 512, "min": 1, "max": 16384}),
                "crop_angle":   ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "snap_to":      ("INT", {"default": 8,   "min": 1, "max": 256}),
                "aspect_lock":  ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING",  {"default": "free"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    # Outputs: `mask` is the cropped *input* alpha/mask (full-white when no
    # mask is connected); `rect_mask` is the rect's coverage on the original
    # canvas (the previous behaviour of the single `mask` output).
    RETURN_TYPES = ("IMAGE", "BAT_CROP_INFO", "MASK", "MASK")
    RETURN_NAMES = ("image", "crop_info", "mask", "rect_mask")
    CATEGORY = "BAT/image"
    FUNCTION = "crop"

    @classmethod
    def IS_CHANGED(cls, image, crop_x, crop_y, crop_w, crop_h, crop_angle,
                   snap_to, aspect_lock, aspect_ratio, mask=None):
        # The image / mask tensors themselves are hashed by ComfyUI; only the
        # widget values need to be in the cache key.
        return f"{crop_x}|{crop_y}|{crop_w}|{crop_h}|{crop_angle}|{snap_to}|{aspect_lock}|{aspect_ratio}"

    def crop(self, image, crop_x, crop_y, crop_w, crop_h, crop_angle,
             snap_to, aspect_lock, aspect_ratio, mask=None):
        n, H, W, c = image.shape
        # x/y can sit anywhere when angle != 0 (rect rotates about its centre,
        # which may swing the corners outside the image — that's fine, those
        # samples just read as zero from grid_sample's zero-padding). Clamp w/h
        # to be sensible. When angle == 0 we keep the strict axis-aligned
        # clamp for an exact slice (no resampling).
        angle = float(crop_angle)
        device = image.device
        dtype = image.dtype

        # Bring an input mask up to the image's batch + spatial dims if needed.
        in_mask = None
        if mask is not None:
            in_mask = mask.to(torch.float32)
            in_mask = _broadcast_mask_to_n(in_mask, n)
            if in_mask.shape[-2:] != (H, W):
                in_mask = F.interpolate(in_mask.unsqueeze(1), size=(H, W),
                                        mode="bilinear", align_corners=False).squeeze(1).clamp(0, 1)

        if abs(angle) < 1e-3:
            x = max(0, min(int(crop_x), W - 1))
            y = max(0, min(int(crop_y), H - 1))
            w = max(1, min(int(crop_w), W - x))
            h = max(1, min(int(crop_h), H - y))
            cropped = image[:, y:y + h, x:x + w, :]
            # Rect's coverage on the original-size canvas.
            rect_mask = torch.zeros((n, H, W), dtype=dtype, device=device)
            rect_mask[:, y:y + h, x:x + w] = 1.0
            # Cropped input alpha (white if no mask input).
            if in_mask is not None:
                cropped_mask = in_mask[:, y:y + h, x:x + w].contiguous()
            else:
                cropped_mask = torch.ones((n, h, w), dtype=dtype, device=device)
        else:
            x = int(crop_x); y = int(crop_y)
            w = max(1, int(crop_w))
            h = max(1, int(crop_h))
            cropped = _rotated_crop(image, x, y, w, h, angle)
            cx = x + w / 2.0
            cy = y + h / 2.0
            rect_mask = _rotated_rect_mask(H, W, cx, cy, w, h, angle, n,
                                           device, dtype)
            if in_mask is not None:
                cropped_mask = _rotated_crop_mask(in_mask, x, y, w, h, angle)
            else:
                cropped_mask = torch.ones((n, h, w), dtype=dtype, device=device)

        out_w = _snap(w, snap_to)
        out_h = _snap(h, snap_to)
        if out_w != w or out_h != h:
            nchw = cropped.permute(0, 3, 1, 2)
            nchw = F.interpolate(nchw, size=(out_h, out_w), mode="bicubic", align_corners=False)
            out_image = nchw.permute(0, 2, 3, 1).clamp(0, 1).contiguous()
            # Mask follows the same snap-resize as the image.
            m_nchw = cropped_mask.unsqueeze(1).to(torch.float32)
            m_nchw = F.interpolate(m_nchw, size=(out_h, out_w), mode="bilinear", align_corners=False)
            cropped_mask = m_nchw.squeeze(1).clamp(0, 1).contiguous()
        else:
            out_image = cropped.contiguous()

        crop_info = {
            # The Uncrop node uses this batch as the canvas to paste back onto.
            "original_image": image,
            "x": x, "y": y, "w": w, "h": h,
            "angle": angle,
            "original_w": int(W), "original_h": int(H),
            "out_w": int(out_w), "out_h": int(out_h),
            "snap_to": int(snap_to),
        }

        return {
            "ui": {
                "preview": [_b64_jpeg(_first(image))],
                "w": [int(W)], "h": [int(H)],
            },
            "result": (out_image, crop_info, cropped_mask, rect_mask),
        }
