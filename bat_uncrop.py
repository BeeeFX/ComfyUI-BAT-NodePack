"""
Bat_Uncrop — paste a processed crop back onto its original plate.

Takes a BAT_CROP_INFO struct from Bat_Crop (the original image batch +
the crop rectangle that was applied), resizes the processed image batch
back to the rect's pixel size, and composites it into a canvas the size
of the original at the original position. A `feather_px` widget softens
the seam at the rect's edges.

Pair-aware: works regardless of whether intermediate nodes changed the
processed batch's resolution or frame count (lengths are broadcast).
"""

import math

import torch
import torch.nn.functional as F

# Lanczos uses ComfyUI core's resampler when available; otherwise we fall back
# to bicubic transparently.
try:
    from comfy.utils import common_upscale as _common_upscale
    _HAS_COMMON_UPSCALE = True
except Exception:
    _common_upscale = None
    _HAS_COMMON_UPSCALE = False

FILTER_CHOICES = ["bicubic", "bilinear", "lanczos", "area", "nearest"]
FIT_CHOICES = ["fit", "cover", "stretch"]


def _resize_nhwc(t, h, w, mode):
    """Resize a (N,H,W,3) float tensor to (N,h,w,3) using the given filter."""
    if t.shape[1] == h and t.shape[2] == w:
        return t
    nchw = t.permute(0, 3, 1, 2)
    if mode == "lanczos" and _HAS_COMMON_UPSCALE:
        out = _common_upscale(nchw, w, h, "lanczos", "disabled")
    else:
        fmode = mode
        if fmode == "lanczos":
            fmode = "bicubic"
        kwargs = {"size": (h, w), "mode": fmode}
        if fmode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        out = F.interpolate(nchw, **kwargs)
    return out.permute(0, 2, 3, 1).clamp(0, 1).contiguous()


def _broadcast_to(batch, target_n):
    """Repeat / truncate a (N,...) tensor along dim 0 to length target_n."""
    n = batch.shape[0]
    if n == target_n:
        return batch
    if n == 1:
        return batch.repeat((target_n,) + (1,) * (batch.ndim - 1))
    if n > target_n:
        return batch[:target_n]
    # n in (1, target_n): repeat last frame to pad up.
    reps = (target_n + n - 1) // n
    return batch.repeat((reps,) + (1,) * (batch.ndim - 1))[:target_n]


def _feather_local(h, w, feather, device, dtype):
    """A (h, w) tensor of 1s with a `feather` px linear ramp falling off on
    each of the four interior edges. feather=0 → all-ones rect."""
    if w <= 0 or h <= 0:
        return torch.zeros((max(1, h), max(1, w)), device=device, dtype=dtype)
    feather = int(max(0, min(feather, w // 2, h // 2)))
    rx = torch.ones(w, device=device, dtype=dtype)
    ry = torch.ones(h, device=device, dtype=dtype)
    if feather > 0:
        ramp = torch.linspace(0.0, 1.0, feather + 2, device=device, dtype=dtype)[1:-1]
        rx[:feather] = ramp
        rx[w - feather:] = ramp.flip(0)
        ry[:feather] = ramp
        ry[h - feather:] = ramp.flip(0)
    return ry.unsqueeze(1) * rx.unsqueeze(0)


def _feather_rect_mask(H, W, x, y, w, h, feather, device, dtype):
    """A (1,H,W) mask = feathered rect placed at (x,y,w,h) on a HxW canvas."""
    mask = torch.zeros((1, H, W), device=device, dtype=dtype)
    if w <= 0 or h <= 0:
        return mask
    mask[0, y:y + h, x:x + w] = _feather_local(h, w, feather, device, dtype)
    return mask


class BatUncrop:
    """Paste a processed Bat_Crop output back onto its original plate."""

    @classmethod
    def INPUT_TYPES(cls):
        # processed first so a bypassed node forwards the processed image
        # (the desired main output).
        return {
            "required": {
                "processed":  ("IMAGE",),
                "crop_info":  ("BAT_CROP_INFO",),
                "fit_mode":   (FIT_CHOICES,),
                "filter":     (FILTER_CHOICES,),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "BAT/image"
    FUNCTION = "uncrop"

    def uncrop(self, processed, crop_info, fit_mode, filter, feather_px):
        if not isinstance(crop_info, dict) or "original_image" not in crop_info:
            raise ValueError("Bat_Uncrop: missing crop_info from a Bat_Crop node")
        original = crop_info["original_image"]
        x = int(crop_info["x"])
        y = int(crop_info["y"])
        w = int(crop_info["w"])
        h = int(crop_info["h"])
        H = int(crop_info["original_h"])
        W = int(crop_info["original_w"])
        angle = float(crop_info.get("angle", 0.0))  # 0 for back-compat with older crop_info dicts

        device = original.device
        dtype = torch.float32
        proc = processed.to(dtype)
        ph, pw = int(proc.shape[1]), int(proc.shape[2])

        # 1. Reformat the processed batch to land back on the original crop
        #    rect. fit_mode controls aspect-ratio handling (the processed
        #    image may have drifted slightly through intermediate nodes):
        #      • stretch — force-resize to (h, w) ignoring aspect
        #      • fit     — preserve aspect, letterbox inside the rect; the
        #                  un-covered portion of the rect shows the plate
        #      • cover   — preserve aspect, fill the rect, centre-crop
        #    paste_(x,y,w,h) is the *actual* pixel area on the canvas the
        #    processed image ends up occupying.
        if fit_mode == "stretch" or (pw == w and ph == h):
            paste = _resize_nhwc(proc, h, w, filter)
            paste_x, paste_y, paste_w, paste_h = x, y, w, h
        elif fit_mode == "cover":
            scale = max(w / pw, h / ph)
            nw = max(1, int(round(pw * scale)))
            nh = max(1, int(round(ph * scale)))
            big = _resize_nhwc(proc, nh, nw, filter)
            cx = max(0, (nw - w) // 2)
            cy = max(0, (nh - h) // 2)
            paste = big[:, cy:cy + h, cx:cx + w, :].contiguous()
            paste_x, paste_y, paste_w, paste_h = x, y, w, h
        else:  # "fit"
            scale = min(w / pw, h / ph)
            nw = max(1, int(round(pw * scale)))
            nh = max(1, int(round(ph * scale)))
            paste = _resize_nhwc(proc, nh, nw, filter)
            paste_x = x + (w - nw) // 2
            paste_y = y + (h - nh) // 2
            paste_w, paste_h = nw, nh

        # 2. Batch length alignment (broadcast the shorter side).
        target_n = max(proc.shape[0], original.shape[0])
        base = _broadcast_to(original.to(dtype), target_n).clone()
        paste = _broadcast_to(paste, target_n)

        if abs(angle) < 1e-3:
            # Axis-aligned fast path.
            # Feather spans the ACTUAL paste rect (so 'fit' letterbox bars
            # aren't part of the blended region).
            m = _feather_rect_mask(H, W, paste_x, paste_y, paste_w, paste_h,
                                   feather_px, device, dtype)
            m_batch = m.expand(target_n, H, W)
            m4 = m_batch.unsqueeze(-1)
            paste_canvas = torch.zeros_like(base)
            paste_canvas[:, paste_y:paste_y + paste_h, paste_x:paste_x + paste_w, :] = paste
            out = (base * (1.0 - m4) + paste_canvas * m4).clamp(0, 1)
            return (out.contiguous(), m_batch.contiguous())

        # Rotated path: paste the (paste_w × paste_h) image rotated about the
        # crop rect's centre. The paste is always centred on the rect centre
        # regardless of fit_mode (fit letterboxes inside the rect, cover/stretch
        # fill it), so we don't need paste_x/y here.
        rect_cx = x + w / 2.0
        rect_cy = y + h / 2.0
        a = math.radians(angle)
        cos_t, sin_t = math.cos(a), math.sin(a)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        dx = xs - rect_cx
        dy = ys - rect_cy
        # Inverse rotation maps canvas displacement → paste-local axes.
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        pu = u + paste_w / 2.0
        pv = v + paste_h / 2.0
        gx = pu / max(1, paste_w - 1) * 2.0 - 1.0
        gy = pv / max(1, paste_h - 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(target_n, H, W, 2)

        # Sample paste through the inverse-rotation grid (zeros outside paste).
        paste_nchw = paste.permute(0, 3, 1, 2)
        sampled = F.grid_sample(paste_nchw, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        paste_canvas = sampled.permute(0, 2, 3, 1).clamp(0, 1)

        # Sample a paste-local feather mask through the same grid so the
        # feather rotates with the paste.
        local = _feather_local(paste_h, paste_w, feather_px, device, dtype)
        local_n = local.unsqueeze(0).unsqueeze(0).expand(target_n, 1, paste_h, paste_w)
        m_sampled = F.grid_sample(local_n, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=True)
        m_batch = m_sampled[:, 0].clamp(0, 1)
        m4 = m_batch.unsqueeze(-1)

        out = (base * (1.0 - m4) + paste_canvas * m4).clamp(0, 1)
        return (out.contiguous(), m_batch.contiguous())
