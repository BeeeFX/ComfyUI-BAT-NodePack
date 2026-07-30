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


# Outside-area fill modes. `black` / `gray` are constant fills; `edge`
# replicates the nearest border pixel; `reflect` mirrors the image at the
# edge. grid_sample has no constant-value-other-than-0 mode, so for the
# rotated path `gray` is realised by pre-shifting: we sample with zeros then
# add the grey level back where the sample missed (via a coverage mask).
_GRAY_VALUE = 0.5

# grid_sample padding_mode per fill (used by the rotated path; the axis-
# aligned path realises the same modes with F.pad, which is exact).
_GRID_PAD = {
    "black":   "zeros",
    "gray":    "zeros",      # gray handled specially (see _rotated_crop)
    "edge":    "border",
    "reflect": "reflection",
}
# F.pad mode per fill (axis-aligned path). None → constant fill.
_PAD_MODE = {
    "black":   ("constant", 0.0),
    "gray":    ("constant", _GRAY_VALUE),
    "edge":    ("replicate", None),
    "reflect": ("reflect", None),
}


def _rotated_crop(image, x, y, w, h, angle_deg, fill="black"):
    """Sample a rotated rect from `image` into an axis-aligned (h, w) output.

    Out-of-canvas samples are filled per `fill` (black / gray / edge /
    reflect)."""
    n, H, W, _ = image.shape
    grid = _make_crop_grid(x, y, w, h, H, W, angle_deg, n,
                           image.device, torch.float32)
    nchw = image.permute(0, 3, 1, 2).to(torch.float32)
    pad_mode = _GRID_PAD.get(fill, "zeros")
    out = F.grid_sample(nchw, grid, mode="bilinear",
                        padding_mode=pad_mode, align_corners=True)
    if fill == "gray":
        # Where the sample fell outside (coverage 0), paint grey.
        ones = torch.ones((n, 1, H, W), device=image.device, dtype=torch.float32)
        cov = F.grid_sample(ones, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        out = out + (1.0 - cov) * _GRAY_VALUE
    return out.permute(0, 2, 3, 1).clamp(0, 1).contiguous()


def _rotated_crop_mask(mask, x, y, w, h, angle_deg, fill="black"):
    """Sample a rotated rect from a (n,H,W) mask into (n,h,w).

    The mask's outside fill mirrors the image edge behaviour for `edge` /
    `reflect`; `black` / `gray` both leave the outside mask at 0 (there's no
    alpha out there)."""
    n, H, W = mask.shape
    grid = _make_crop_grid(x, y, w, h, H, W, angle_deg, n,
                           mask.device, torch.float32)
    nchw = mask.unsqueeze(1).to(torch.float32)
    pad_mode = _GRID_PAD.get(fill, "zeros")
    if fill == "gray":
        pad_mode = "zeros"
    out = F.grid_sample(nchw, grid, mode="bilinear",
                        padding_mode=pad_mode, align_corners=True)
    return out.squeeze(1).clamp(0, 1).contiguous()


def _extract_axis_aligned(image_1hwc_or_nhwc, x, y, w, h, fill):
    """Exact axis-aligned crop of a (…,H,W,C) tensor at integer (x,y,w,h),
    realising the outside-area `fill` with F.pad (no resample). Returns a
    (…,h,w,C) tensor. Also returns the source-overlap rect so callers can
    build the rect_mask / cropped alpha consistently."""
    *lead, H, W, C = image_1hwc_or_nhwc.shape
    x = int(round(x)); y = int(round(y)); w = int(w); h = int(h)
    # Overhang on each side (how far the crop pokes past the image edge).
    pl = max(0, -x)
    pt = max(0, -y)
    pr = max(0, (x + w) - W)
    pb = max(0, (y + h) - H)
    src = image_1hwc_or_nhwc
    if pl or pt or pr or pb:
        mode, val = _PAD_MODE.get(fill, ("constant", 0.0))
        # F.pad expects (N,C,H,W); our tensor is (…,H,W,C).
        nchw = src.permute(0, 3, 1, 2) if src.dim() == 4 else src.unsqueeze(0).permute(0, 3, 1, 2)
        # reflect/replicate need the pad < dim; clamp so huge overhangs don't
        # error — fall back to constant(0) for the portion beyond a full mirror.
        if mode in ("reflect", "replicate"):
            max_pad_w = W - 1
            max_pad_h = H - 1
            if pl > max_pad_w or pr > max_pad_w or pt > max_pad_h or pb > max_pad_h:
                mode, val = "constant", 0.0
        if mode == "constant":
            nchw = F.pad(nchw, (pl, pr, pt, pb), mode="constant", value=float(val))
        else:
            nchw = F.pad(nchw, (pl, pr, pt, pb), mode=mode)
        src = nchw.permute(0, 2, 3, 1)
        if image_1hwc_or_nhwc.dim() == 3:
            src = src.squeeze(0)
    # After padding, the crop origin shifts by (pl, pt).
    ox, oy = x + pl, y + pt
    cropped = src[..., oy:oy + h, ox:ox + w, :]
    return cropped.contiguous()


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


def rotated_bbox_size(w, h, angle_deg):
    """Axis-aligned bounding-box size of a w×h rect rotated by angle_deg.

    bw = |w·cos| + |h·sin| ,  bh = |w·sin| + |h·cos|
    """
    a = math.radians(float(angle_deg))
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return (abs(w) * c + abs(h) * s, abs(w) * s + abs(h) * c)


def constrain_rotated_rect(x, y, w, h, angle_deg, W, H):
    """Slide a ROTATED rect fully inside a W×H canvas without changing its size
    or angle. Returns the corrected top-left (x, y).

    `constrain_to_canvas` used to be gated behind `abs(angle) < 1e-3`, so the
    moment an artist rotated the crop the toggle silently became a no-op and the
    rect could swing off the plate (filled/black corners). Constraining a
    rotated rect means keeping its rotated BOUNDING BOX on the canvas, which is
    a clamp on the centre rather than on the origin.

    When the rotated bbox is larger than the canvas on an axis, the rect is
    centred on that axis — we never shrink or unrotate it (that would silently
    change the artist's framing), so some edge fill is unavoidable in that case.
    """
    bw, bh = rotated_bbox_size(w, h, angle_deg)
    # Centre implied by the requested top-left.
    cx = float(x) + float(w) / 2.0
    cy = float(y) + float(h) / 2.0
    # Valid centre range so the bbox stays inside [0, W] x [0, H].
    if bw <= W:
        cx = min(max(cx, bw / 2.0), W - bw / 2.0)
    else:
        cx = W / 2.0          # too wide to fit — centre it
    if bh <= H:
        cy = min(max(cy, bh / 2.0), H - bh / 2.0)
    else:
        cy = H / 2.0
    return int(round(cx - float(w) / 2.0)), int(round(cy - float(h) / 2.0))


def _b64_jpeg(img_1hwc, max_dim=1024, quality=80):
    arr = (img_1hwc[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _snap(v, step, mode="floor"):
    """Snap `v` to a multiple of `step`.

    mode="floor"   → round DOWN (default). Keeps the crop inside what the artist
                     drew, and — combined with snapping the rect BEFORE slicing
                     — means no resampling ever happens.
    mode="nearest" → round to nearest (the old behaviour).

    Previously this was nearest-only AND floored at `step`, so it could only
    ever grow: a 4-px crop with snap_to=8 became 8, and any non-multiple drag
    forced a bicubic resample. `step<=1` is now a true no-op.
    """
    step = max(1, int(step))
    v = int(v)
    if step == 1:
        return max(1, v)
    if mode == "nearest":
        # floor(v/step + 0.5) rather than round(), which is banker's rounding in
        # Python 3 and would snap an exact .5 case DOWN (516/8 = 64.5 -> 64).
        snapped = int(v // step + (1 if (v % step) * 2 >= step else 0)) * step
    else:
        snapped = (v // step) * step
    # Never return 0 — a zero-size crop is meaningless. Fall back to one step.
    return max(step, snapped)


class BatCrop:
    """Axis-aligned drag crop with a paired-output channel for Bat_Uncrop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                # crop_x/y allow negatives so the rect can be positioned
                # partly off the top/left edge when constrain_to_canvas is off.
                "crop_x":       ("INT", {"default": 0,   "min": -16384, "max": 16384}),
                "crop_y":       ("INT", {"default": 0,   "min": -16384, "max": 16384}),
                "crop_w":       ("INT", {"default": 512, "min": 1, "max": 16384}),
                "crop_h":       ("INT", {"default": 512, "min": 1, "max": 16384}),
                "crop_angle":   ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "snap_to":      ("INT", {"default": 8,   "min": 1, "max": 256}),
                "aspect_lock":  ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING",  {"default": "free"}),
                # When on (default) an axis-aligned crop is clamped to stay
                # fully inside the image. When off, the rect may extend past
                # the canvas edge and the out-of-frame area is zero-padded —
                # useful for framing a subject that runs off the edge (e.g. a
                # face leaving frame). Matches Bat_AnimatedCrop's behaviour.
                "constrain_to_canvas": ("BOOLEAN", {"default": True}),
                # How to fill the area OUTSIDE the source image when the crop
                # extends past the canvas edge (constrain off) or is rotated.
                #   black   — solid black (0)
                #   gray    — solid mid-grey (0.5)
                #   edge    — replicate the nearest border pixel (clamp)
                #   reflect — mirror the image across the edge
                "outside_fill": (["black", "gray", "edge", "reflect"],
                                 {"default": "black"}),
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
    DESCRIPTION = (
        "Interactive crop with an on-node canvas: drag the rect, 8 sizing "
        "handles, rotation, aspect lock and snap. Outputs the cropped image "
        "plus a BAT_CROP_INFO socket that Bat_Uncrop uses to paste the "
        "processed result back onto the plate."
    )
    FUNCTION = "crop"

    @classmethod
    def IS_CHANGED(cls, image, crop_x, crop_y, crop_w, crop_h, crop_angle,
                   snap_to, aspect_lock, aspect_ratio,
                   constrain_to_canvas=True, outside_fill="black", mask=None):
        # The image / mask tensors themselves are hashed by ComfyUI; only the
        # widget values need to be in the cache key.
        return (f"{crop_x}|{crop_y}|{crop_w}|{crop_h}|{crop_angle}|{snap_to}|"
                f"{aspect_lock}|{aspect_ratio}|{constrain_to_canvas}|{outside_fill}")

    def crop(self, image, crop_x, crop_y, crop_w, crop_h, crop_angle,
             snap_to, aspect_lock, aspect_ratio,
             constrain_to_canvas=True, outside_fill="black", mask=None):
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

        # Snap the REQUESTED size up-front so the slice itself lands on a
        # multiple of `snap_to`. Previously the crop was sliced at the drawn
        # size and then bicubic-resampled to the snapped size, AND crop_info
        # recorded the *pre-snap* w/h — which Uncrop reads — so any
        # non-multiple-of-8 drag (the default snap) round-tripped through a
        # size that never existed on the plate and shifted the seam.
        # Snapping first means the common path does zero resampling.
        crop_w = _snap(crop_w, snap_to)
        crop_h = _snap(crop_h, snap_to)

        if abs(angle) < 1e-3 and constrain_to_canvas:
            # Slide the rect in rather than shrinking it: clamping x to W-1 and
            # then w to W-x could yield a 1-px-wide crop when the rect was
            # dragged off the right edge, which Uncrop then faithfully pasted.
            x = max(0, min(int(crop_x), max(0, W - int(crop_w))))
            y = max(0, min(int(crop_y), max(0, H - int(crop_h))))
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
        elif abs(angle) < 1e-3:
            # Unconstrained axis-aligned: keep the exact drawn w/h and fill the
            # region outside the source image per `outside_fill`. The image
            # uses the chosen fill; the rect_mask / cropped alpha reflect only
            # the REAL image overlap (the filled area isn't real coverage).
            x = int(round(crop_x))
            y = int(round(crop_y))
            w = max(1, int(crop_w))
            h = max(1, int(crop_h))
            cropped = _extract_axis_aligned(image, x, y, w, h, outside_fill)
            src_x0 = max(0, x)
            src_y0 = max(0, y)
            src_x1 = min(W, x + w)
            src_y1 = min(H, y + h)
            dst_x0 = src_x0 - x
            dst_y0 = src_y0 - y
            has_overlap = src_x1 > src_x0 and src_y1 > src_y0
            rect_mask = torch.zeros((n, H, W), dtype=dtype, device=device)
            if has_overlap:
                rect_mask[:, src_y0:src_y1, src_x0:src_x1] = 1.0
            if in_mask is not None:
                cropped_mask = torch.zeros((n, h, w), device=device, dtype=dtype)
                if has_overlap:
                    cropped_mask[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                                    dst_x0:dst_x0 + (src_x1 - src_x0)] = \
                        in_mask[:, src_y0:src_y1, src_x0:src_x1]
            else:
                cropped_mask = torch.ones((n, h, w), dtype=dtype, device=device)
        else:
            x = int(crop_x); y = int(crop_y)
            w = max(1, int(crop_w))
            h = max(1, int(crop_h))
            # constrain_to_canvas now applies to ROTATED crops too. Previously
            # this branch ignored it entirely (the toggle was gated behind
            # angle≈0), so rotating a crop silently disabled the constraint.
            # Slide the rect in by clamping its centre so the rotated bounding
            # box stays on the plate; size and angle are never modified.
            if constrain_to_canvas:
                x, y = constrain_rotated_rect(x, y, w, h, angle, W, H)
            cropped = _rotated_crop(image, x, y, w, h, angle, outside_fill)
            cx = x + w / 2.0
            cy = y + h / 2.0
            rect_mask = _rotated_rect_mask(H, W, cx, cy, w, h, angle, n,
                                           device, dtype)
            if in_mask is not None:
                cropped_mask = _rotated_crop_mask(in_mask, x, y, w, h, angle,
                                                  outside_fill)
            else:
                cropped_mask = torch.ones((n, h, w), dtype=dtype, device=device)

        # `crop_w/crop_h` were already snapped above, so w/h are normally
        # already multiples of snap_to and this is a no-op (no resampling).
        # It can still fire in the constrained branch when the rect had to be
        # clipped to the canvas edge.
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
            # Provenance: how the outside area was filled. Uncrop only pastes
            # the real-image overlap back (the filled border lands outside the
            # plate and is clipped), so this is informational — but carrying it
            # keeps the crop→uncrop round-trip self-describing.
            "outside_fill": outside_fill,
        }

        return {
            "ui": {
                "preview": [_b64_jpeg(_first(image))],
                "w": [int(W)], "h": [int(H)],
            },
            "result": (out_image, crop_info, cropped_mask, rect_mask),
        }
