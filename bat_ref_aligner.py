"""
Bat_RefAligner — align a reference image onto a plate-sized canvas.

Built for Wan 2.2 Animate: the reference animates better when its framing and
scale roughly match the subject in the driving plate. This node gives a
Photoshop-style free-transform (move / uniform scale / rotate) of the reference
over a canvas the size of the plate, bakes the transform into widgets, and
outputs the repositioned reference plus a coverage mask.

The transform is authored on the node's interactive canvas (see
web/bat_ref_aligner.js); this module just applies the baked widget values via
a single affine grid_sample.
"""

import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _first(img):
    """First frame of an IMAGE batch as (1,H,W,C)."""
    return img[0:1] if img is not None and img.shape[0] > 1 else img


def _parse_rgb(s, default=(0.5, 0.5, 0.5)):
    try:
        parts = [float(x) for x in str(s).split(",")]
        if len(parts) >= 3:
            return tuple(min(1.0, max(0.0, p / 255.0)) for p in parts[:3])
    except Exception:
        pass
    return default


def _b64_jpeg(img_1hwc, max_dim=1024, quality=80):
    """Downscaled base64 JPEG of a (1,H,W,3) float tensor for the editor."""
    arr = (img_1hwc[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_png_rgba(img_1hwc, alpha_1chw, max_dim=1024):
    """Downscaled base64 PNG of a (1,H,W,3) RGB tensor + (1,1,H,W) alpha tensor.

    Used for the reference preview so the editor canvas shows the ref cut by
    its mask (or by its own embedded alpha) — `<img>` + `ctx.drawImage` honour
    PNG alpha natively, so no frontend math is needed. The alpha is sourced
    from the same `alpha_src` used downstream for compositing, so what the
    artist sees in the preview matches the output.
    """
    arr = (img_1hwc[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)  # (H,W,3)
    a_t = alpha_1chw[0, 0].clamp(0, 1).cpu().numpy()                        # (H,W)
    # alpha may already match (the common case), but stay defensive against
    # callers passing a mask of a different shape.
    if a_t.shape != arr.shape[:2]:
        a_im = Image.fromarray((a_t * 255.0).astype(np.uint8), "L").resize(
            (arr.shape[1], arr.shape[0]), Image.BILINEAR,
        )
        a_arr = np.array(a_im, dtype=np.uint8)
    else:
        a_arr = (a_t * 255.0).astype(np.uint8)
    rgba = np.dstack((arr, a_arr))                                          # (H,W,4)
    im = Image.fromarray(rgba, "RGBA")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class RefAligner:
    """Transform a reference image onto a plate-sized canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # NOTE: `reference` is declared before `plate` on purpose. ComfyUI's
            # bypass rewire connects each output to the first input of the
            # matching type by declaration order, so listing reference first
            # makes a bypassed node forward the reference (which IS this node's
            # output) instead of the plate.
            "required": {
                "reference": ("IMAGE",),
                "plate": ("IMAGE",),
                "translate_x": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "translate_y": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "edge_mode": (["edge_pixel", "cut"],),
                "bg_color": ("STRING", {"default": "128,128,128"}),
                "premultiply": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "BAT/wan"
    FUNCTION = "align"

    def align(self, plate, reference, translate_x, translate_y, scale, rotation,
              edge_mode, bg_color, premultiply=True, mask=None):
        plate = _first(plate)
        reference = _first(reference)
        _, H, W, _ = plate.shape
        rh, rw = reference.shape[1], reference.shape[2]
        device = reference.device
        dtype = torch.float32

        # Reference alpha: explicit MASK input wins, else the reference's own
        # 4th channel if present, else fully opaque.
        if mask is not None:
            m = _first(mask).to(dtype)            # (1,mh,mw)
            alpha_src = m.unsqueeze(1)            # (1,1,mh,mw)
            if alpha_src.shape[-2:] != (rh, rw):
                alpha_src = F.interpolate(alpha_src, size=(rh, rw),
                                          mode="bilinear", align_corners=False)
        elif reference.shape[-1] == 4:
            alpha_src = reference[..., 3:4].permute(0, 3, 1, 2).to(dtype)  # (1,1,rh,rw)
        else:
            alpha_src = torch.ones((1, 1, rh, rw), device=device, dtype=dtype)
        alpha_src = alpha_src.to(device)
        reference = reference[..., :3]

        # Build the inverse map: for each output canvas pixel, find the ref
        # pixel it samples. Forward transform places the ref scaled+rotated
        # about its centre at canvas_centre + (tx, ty); we invert it.
        cx = W / 2.0 + float(translate_x)
        cy = H / 2.0 + float(translate_y)
        s = max(1e-6, float(scale))
        theta = np.deg2rad(float(rotation))
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)  # rotate by -rotation

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        dx = xs - cx
        dy = ys - cy
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        u = rx / s + rw / 2.0           # ref-space x
        v = ry / s + rh / 2.0           # ref-space y
        # Normalise to [-1, 1] (align_corners=True convention).
        gx = u / max(1, rw - 1) * 2.0 - 1.0
        gy = v / max(1, rh - 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

        ref_nchw = reference.permute(0, 3, 1, 2).to(dtype)
        pad = "border" if edge_mode == "edge_pixel" else "zeros"
        sampled = F.grid_sample(ref_nchw, grid, mode="bilinear",
                                padding_mode=pad, align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1)  # (1,H,W,3)

        # Warp the reference alpha → on-canvas subject coverage (zeros padding,
        # so anything outside the ref rectangle is transparent).
        cov = F.grid_sample(alpha_src, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        alpha = cov[:, 0].clamp(0, 1)            # (1,H,W) — the output MASK

        # Solid background fill.
        r, g, b = _parse_rgb(bg_color)
        bg = torch.zeros((1, H, W, 3), device=device, dtype=dtype)
        bg[..., 0], bg[..., 1], bg[..., 2] = r, g, b

        # Fill behaviour: composite the subject over bg whenever premultiply is
        # on or edge_mode is "cut"; only "edge_pixel" *without* premultiply
        # keeps the border-extended pixels filling the canvas.
        if premultiply or edge_mode == "cut":
            a = alpha.unsqueeze(-1)
            image = (sampled * a + bg * (1.0 - a)).clamp(0, 1)
        else:
            image = sampled.clamp(0, 1)

        mask = alpha
        return {
            "ui": {
                "plate": [_b64_jpeg(plate)],
                # Reference goes back as RGBA PNG carrying `alpha_src` so
                # the editor's preview shows the ref cut by the mask (or
                # by its own embedded alpha for a 4-channel input). Falls
                # back to a fully-opaque alpha when neither was provided
                # — see `alpha_src` assignment above — so workflows
                # without a mask still see the full RGB rectangle, no
                # visual change vs. the previous JPEG behaviour.
                "reference": [_b64_png_rgba(reference, alpha_src)],
                "plate_w": [int(W)], "plate_h": [int(H)],
                "ref_w": [int(rw)], "ref_h": [int(rh)],
            },
            "result": (image.cpu(), mask.cpu()),
        }
