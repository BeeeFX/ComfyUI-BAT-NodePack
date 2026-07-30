"""
Bat_AnimatedCrop — same crop primitive as Bat_Crop but the rect is
keyframed across the input batch. Pair with Bat_Uncrop: when this node
is upstream Bat_Uncrop will honour the per-frame rect list and paste
each processed frame back at the right place.

State model (matches what the JS editor serialises into the hidden
`state` widget):

    {
        "keyframes": {
            "0":  {"x": 0, "y": 0, "w": 512, "h": 512, "angle": 0.0},
            "24": {"x": 48, "y": 12, "w": 512, "h": 512, "angle": 0.0},
            ...
        }
    }

Between keyframes every field lerps independently. Before the first
keyframe / after the last we hold (no extrapolation). With zero
keyframes we fall back to a 512×512 rect at the origin so the node
doesn't explode on a freshly placed graph.

Output resolution: snapped from the FIRST keyframe's `w` and `h`
using `snap_to`. Every frame of the output is resized to that uniform
size so the result stays a regular `(N, out_h, out_w, 3)` tensor.

`crop_info` always carries the scalar fields Bat_Crop wrote (for
backward-compat with code that doesn't iterate per-frame), with the
first frame's rect as the scalar value. The new `frames` list is the
authoritative per-frame source-rect; Bat_Uncrop reads it when present.
"""

import base64
import hashlib
import json
import logging
import math
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the crop primitives from Bat_Crop so the math is identical
# between the static and animated nodes.
from .bat_crop import (
    _broadcast_mask_to_n,
    _extract_axis_aligned,
    _rotated_crop,
    _rotated_crop_mask,
    _rotated_rect_mask,
    _snap,
    constrain_rotated_rect,
    rotated_bbox_size,
)

logger = logging.getLogger("[Bat_AnimatedCrop]")


def _first(img):
    return img[0:1] if img is not None and img.shape[0] > 1 else img


def _resolve_rect_at_frame(kfs: dict, frame: int) -> dict:
    """Interpolate the rect (x, y, w, h, angle) at `frame` from a
    `{"<int frame>": rect}` keyframe dict. Holds at the ends; lerps
    between adjacent keyframes."""
    if not kfs:
        return {"x": 0, "y": 0, "w": 512, "h": 512, "angle": 0.0}
    keys = sorted(int(k) for k in kfs.keys())
    if frame <= keys[0]:
        return dict(kfs[str(keys[0])])
    if frame >= keys[-1]:
        return dict(kfs[str(keys[-1])])
    prev = keys[0]
    nxt = keys[-1]
    for k in keys:
        if k <= frame:
            prev = k
        if k >= frame and k != prev:
            nxt = k
            break
    if prev == nxt:
        return dict(kfs[str(prev)])
    a = kfs[str(prev)]
    b = kfs[str(nxt)]
    t = (frame - prev) / (nxt - prev)
    return {
        "x": a["x"] + (b["x"] - a["x"]) * t,
        "y": a["y"] + (b["y"] - a["y"]) * t,
        "w": a["w"] + (b["w"] - a["w"]) * t,
        "h": a["h"] + (b["h"] - a["h"]) * t,
        "angle": a["angle"] + (b["angle"] - a["angle"]) * t,
    }


def _resize_nhwc(t, h, w, mode="bicubic"):
    """Resize a (N,H,W,C) image tensor to (N,h,w,C)."""
    if t.shape[1] == h and t.shape[2] == w:
        return t
    nchw = t.permute(0, 3, 1, 2)
    kw = {"size": (h, w), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kw["align_corners"] = False
    out = F.interpolate(nchw, **kw)
    return out.permute(0, 2, 3, 1).clamp(0, 1).contiguous()


def _resize_nhw(t, h, w):
    """Resize a (N,H,W) mask tensor to (N,h,w)."""
    if t.shape[1] == h and t.shape[2] == w:
        return t
    nchw = t.unsqueeze(1).to(torch.float32)
    out = F.interpolate(nchw, size=(h, w), mode="bilinear", align_corners=False)
    return out.squeeze(1).clamp(0, 1).contiguous()


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = 720, quality: int = 78) -> str:
    """Downscaled base64 JPEG of an (H,W,3) uint8 numpy array."""
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize(
            (max(1, int(im.width * r)), max(1, int(im.height * r))),
            Image.BILINEAR,
        )
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class BatAnimatedCrop:
    """Keyframed crop rect across a frame batch. Compatible with Bat_Uncrop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "state": ("STRING", {
                    "default": '{"keyframes":{}}',
                    "multiline": False,
                }),
                "snap_to":      ("INT", {"default": 8, "min": 1, "max": 256}),
                "aspect_lock":  ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING",  {"default": "free"}),
                # When on (default) each frame's rect is clamped to stay
                # inside the canvas. When off, the rect may run off the edge
                # and the out-of-frame area is zero-padded — matches Bat_Crop.
                "constrain_to_canvas": ("BOOLEAN", {"default": True}),
                # Fill for the area outside the source image (crop off-canvas
                # or rotated): black / gray / edge (replicate) / reflect.
                "outside_fill": (["black", "gray", "edge", "reflect"],
                                 {"default": "black"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    # Same return shape as Bat_Crop so this is a drop-in replacement when
    # the downstream code already knows what to do with the outputs.
    RETURN_TYPES = ("IMAGE", "BAT_CROP_INFO", "MASK", "MASK")
    RETURN_NAMES = ("image", "crop_info", "mask", "rect_mask")
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Interactive crop whose rectangle is KEYFRAMED across the input batch "
        "— scrub the timeline, move/resize/rotate the rect, and it records a "
        "keyframe. Pair with Bat_Uncrop, which honours the per-frame rect "
        "list and pastes each processed frame back at its own source rect."
    )
    FUNCTION = "crop"

    @classmethod
    def IS_CHANGED(cls, image, state, snap_to, aspect_lock, aspect_ratio,
                   constrain_to_canvas=True, outside_fill="black", mask=None):
        # State JSON drives the rect; everything else lives in the cache key.
        # sha1 rather than hash() — see the note in bat_roto.IS_CHANGED: hash()
        # is salted per process, so the key changed on every restart.
        try:
            digest = hashlib.sha1(
                str(state).encode("utf-8", "surrogatepass")).hexdigest()
            return (f"state:{digest}|{snap_to}|{aspect_lock}|"
                    f"{aspect_ratio}|{constrain_to_canvas}|{outside_fill}")
        except Exception:
            return "state:err"

    def crop(self, image, state, snap_to, aspect_lock, aspect_ratio,
             constrain_to_canvas=True, outside_fill="black", mask=None):
        n, H, W, c = image.shape
        device = image.device
        dtype = image.dtype

        try:
            doc = json.loads(state) if state else {}
        except (TypeError, ValueError):
            logger.warning("Bat_AnimatedCrop: state JSON invalid; using empty.")
            doc = {}
        kfs = doc.get("keyframes") or {}

        # Bring any input mask up to (n, H, W).
        in_mask = None
        if mask is not None:
            in_mask = mask.to(torch.float32)
            in_mask = _broadcast_mask_to_n(in_mask, n)
            if in_mask.shape[-2:] != (H, W):
                in_mask = F.interpolate(
                    in_mask.unsqueeze(1), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(1).clamp(0, 1)

        # Uniform output size = snap of the first keyframe's dimensions.
        # Falls back to 512×512 (snapped) when no keyframes exist.
        first_rect = _resolve_rect_at_frame(kfs, 0)
        out_w = _snap(max(1, int(round(first_rect["w"]))), snap_to)
        out_h = _snap(max(1, int(round(first_rect["h"]))), snap_to)

        out_frames = []
        out_masks = []
        rect_masks = []
        per_frame_rects = []

        for f in range(n):
            rect = _resolve_rect_at_frame(kfs, f)
            x = float(rect["x"])
            y = float(rect["y"])
            w = max(1, int(round(rect["w"])))
            h = max(1, int(round(rect["h"])))
            angle = float(rect["angle"])
            # Constrain to canvas — applies to ROTATED rects too.
            #
            # Previously this was gated behind `abs(angle) < 1e-3`, so rotating
            # a keyframe silently disabled the toggle and the rect swung off the
            # plate. For a rotated rect, constraining means keeping its rotated
            # BOUNDING BOX inside the canvas, i.e. clamping the centre.
            #
            # Also: w/h are no longer shrunk to the canvas. `min(w, W)` narrowed
            # the rect on some frames and not others, changing the per-frame
            # aspect and making the output "breathe" across the clip. We only
            # slide the rect now; extraction already clips to the real overlap.
            if constrain_to_canvas:
                if abs(angle) < 1e-3:
                    x = min(max(0.0, x), float(max(0, W - w)))
                    y = min(max(0.0, y), float(max(0, H - h)))
                else:
                    cx_, cy_ = constrain_rotated_rect(int(round(x)), int(round(y)),
                                                      w, h, angle, W, H)
                    x, y = float(cx_), float(cy_)
            per_frame_rects.append({"x": x, "y": y, "w": w, "h": h, "angle": angle})

            frame_img = image[f:f + 1]   # (1, H, W, 3)
            if in_mask is not None:
                frame_mask_in = in_mask[f:f + 1]
            else:
                frame_mask_in = None

            if abs(angle) < 1e-3:
                # Axis-aligned: exact slice, filling the outside area per
                # `outside_fill`. rect_m / cropped_m reflect only the real
                # image overlap (the filled area isn't real coverage).
                ix = int(round(x))
                iy = int(round(y))
                src_x0 = max(0, ix)
                src_y0 = max(0, iy)
                src_x1 = min(W, ix + w)
                src_y1 = min(H, iy + h)
                dst_x0 = src_x0 - ix
                dst_y0 = src_y0 - iy
                cropped = _extract_axis_aligned(frame_img, ix, iy, w, h, outside_fill)
                rect_m = torch.zeros((1, H, W), dtype=dtype, device=device)
                if src_x1 > src_x0 and src_y1 > src_y0:
                    rect_m[:, src_y0:src_y1, src_x0:src_x1] = 1.0
                if frame_mask_in is not None:
                    cropped_m = torch.zeros((1, h, w), device=device, dtype=dtype)
                    if src_x1 > src_x0 and src_y1 > src_y0:
                        cropped_m[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                                     dst_x0:dst_x0 + (src_x1 - src_x0)] = \
                            frame_mask_in[:, src_y0:src_y1, src_x0:src_x1]
                else:
                    cropped_m = torch.ones((1, h, w), device=device, dtype=dtype)
            else:
                cropped = _rotated_crop(frame_img, int(round(x)), int(round(y)),
                                        w, h, angle, outside_fill)
                cx = x + w / 2.0
                cy = y + h / 2.0
                rect_m = _rotated_rect_mask(H, W, cx, cy, w, h, angle, 1,
                                            device, dtype)
                if frame_mask_in is not None:
                    cropped_m = _rotated_crop_mask(
                        frame_mask_in, int(round(x)), int(round(y)), w, h, angle,
                        outside_fill,
                    )
                else:
                    cropped_m = torch.ones((1, h, w), device=device, dtype=dtype)

            # Resize this frame's cropped image / mask to the uniform output.
            cropped = _resize_nhwc(cropped, out_h, out_w)
            cropped_m = _resize_nhw(cropped_m, out_h, out_w)

            out_frames.append(cropped)
            out_masks.append(cropped_m)
            rect_masks.append(rect_m)

        out_image = torch.cat(out_frames, dim=0).contiguous()
        out_mask = torch.cat(out_masks, dim=0).contiguous()
        rect_mask = torch.cat(rect_masks, dim=0).contiguous()

        first = per_frame_rects[0] if per_frame_rects else {
            "x": 0.0, "y": 0.0, "w": out_w, "h": out_h, "angle": 0.0,
        }
        crop_info = {
            "original_image": image,
            # Scalar fields populated from the FIRST frame's rect — keeps
            # any Uncrop-style consumer that doesn't know about `frames`
            # behaving sensibly (it'll paste back as if the entire batch
            # shared frame 0's rect).
            "x": int(round(first["x"])),
            "y": int(round(first["y"])),
            "w": int(first["w"]),
            "h": int(first["h"]),
            "angle": float(first["angle"]),
            "original_w": int(W),
            "original_h": int(H),
            "out_w": int(out_w),
            "out_h": int(out_h),
            "snap_to": int(snap_to),
            # New: per-frame source rect list. Bat_Uncrop iterates this
            # when present; absent for static Bat_Crop (back-compat).
            "frames": per_frame_rects,
            "outside_fill": outside_fill,
        }

        # Push input frames as base64 JPEGs for the canvas editor (same
        # subsampling pattern as Bat_Roto so long sequences don't blow up
        # the ui payload).
        frames_b64 = []
        max_preview_frames = 240
        stride = max(1, n // max_preview_frames) if n > max_preview_frames else 1
        for i in range(0, n, stride):
            arr = (image[i].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            frames_b64.append(_b64_jpeg(arr))

        return {
            "ui": {
                "frames": frames_b64,
                "w": [int(W)],
                "h": [int(H)],
                "stride": [int(stride)],
                "frame_count": [int(n)],
            },
            "result": (out_image, crop_info, out_mask, rect_mask),
        }
