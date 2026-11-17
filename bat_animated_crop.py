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
            "24": {"x": 48, "y": 12, "w": 512, "h": 512, "angle": 0.0,
                   "ease": "ease_in_out"},
            ...
        },
        "seed": "centre"
    }

Between keyframes every field interpolates independently, through the
curve named by the EARLIER key's optional `ease` (bat_easing.py; absent =
linear, so older states interpolate exactly as before). Before the first
keyframe / after the last we hold (no extrapolation).

With zero keyframes a state carrying `"seed": "centre"` (the default for
new nodes) renders the centred half-size rect the editor draws before the
first edit. A state without it renders the old 512×512 rect at the origin,
which is what such a saved workflow always produced on its first run.

The clip's size and frame count are NOT in the state: `state` is a prompt
input, so the editor keeps them in node.properties (older states still
carry `imgW` / `imgH` / `frameCount`; they are ignored here).

Output resolution: snapped from the FIRST keyframe's `w` and `h`
using `snap_to`. Each frame's rect is scaled about its centre by the same
snap ratio, so frames at the first keyframe's size slice at exactly that
size; any other frame is resized to it, so the result stays a regular
`(N, out_h, out_w, 3)` tensor.

`crop_info` always carries the scalar fields Bat_Crop wrote (for
backward-compat with code that doesn't iterate per-frame), with the
first frame's rect as the scalar value. The new `frames` list is the
authoritative per-frame source-rect; Bat_Uncrop reads it when present.
"""

import base64
import functools
import hashlib
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the crop primitives from Bat_Crop so the math is identical
# between the static and animated nodes.
from .bat_crop import (
    _GRAY_VALUE,
    _GRID_PAD,
    _broadcast_mask_to_n,
    _extract_axis_aligned,
    _make_crop_grid,
    _rotated_rect_mask,
    _snap,
    constrain_rotated_rect,
)
from .bat_easing import apply_ease, ease_name
from .bat_ui_ref import stash_ui

logger = logging.getLogger("[Bat_AnimatedCrop]")


def _first(img):
    return img[0:1] if img is not None and img.shape[0] > 1 else img


# Frames per batched slice / grid_sample / resize are capped by this much
# scratch memory (inputs gathered for the batch, grids, pre-resize crops).
_CHUNK_BYTES = 256 << 20


def _seed_rect(doc, W, H) -> dict:
    """The rect a state with no keyframes renders on a W×H clip.

    It used to be 512×512 at the origin whatever the clip, while the editor
    drew (and on its first ingest wrote) a centred half-size rect, so the very
    first run never matched the canvas. New nodes carry `"seed": "centre"` in
    their default state and get the editor's rect; a state saved without the
    marker keeps the origin rect it has always rendered.
    """
    if isinstance(doc, dict) and doc.get("seed") == "centre":
        # floor(v + 0.5) is the editor's Math.round; round() is banker's and
        # would land an odd-sized plate's rect a pixel off the canvas.
        def rnd(v):
            return int(math.floor(v + 0.5))
        return {"x": rnd(W * 0.25), "y": rnd(H * 0.25),
                "w": max(1, rnd(W * 0.5)), "h": max(1, rnd(H * 0.5)),
                "angle": 0.0}
    return {"x": 0, "y": 0, "w": 512, "h": 512, "angle": 0.0}


def _resolve_rect_at_frame(kfs: dict, frame: int, empty: dict = None) -> dict:
    """Interpolate the rect (x, y, w, h, angle) at `frame` from a
    `{"<int frame>": rect}` keyframe dict. Holds at the ends; between
    adjacent keyframes each field follows the earlier key's `ease`
    (linear when it has none). `empty` is the rect for an empty dict
    (see _seed_rect)."""
    if not kfs:
        return dict(empty) if empty else {"x": 0, "y": 0, "w": 512, "h": 512, "angle": 0.0}
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
    t = apply_ease((frame - prev) / (nxt - prev), ease_name(a))
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
    # Clamp bicubic overshoot to the source's OWN range, not [0,1] — the
    # latter clipped HDR plates on every resized frame. Per frame, so a
    # batched resize clamps exactly as resizing each frame alone did.
    lo = t.amin(dim=(1, 2, 3), keepdim=True)
    hi = t.amax(dim=(1, 2, 3), keepdim=True)
    return out.permute(0, 2, 3, 1).clamp(lo, hi).contiguous()


def _resize_nhw(t, h, w):
    """Resize a (N,H,W) mask tensor to (N,h,w)."""
    if t.shape[1] == h and t.shape[2] == w:
        return t
    nchw = t.unsqueeze(1).to(torch.float32)
    out = F.interpolate(nchw, size=(h, w), mode="bilinear", align_corners=False)
    return out.squeeze(1).clamp(0, 1).contiguous()


def _axis_key(r):
    return int(round(r["x"])), int(round(r["y"]))


def _rot_key(r):
    return r["x"], r["y"], r["angle"]


def _runs(rects, idx, key):
    """Split `idx` (ascending frame indices) into (i0, i1) spans of
    consecutive frames whose rects share `key` — one slice / one grid each."""
    spans = []
    i0 = 0
    for i in range(1, len(idx) + 1):
        if (i == len(idx) or idx[i] != idx[i - 1] + 1
                or key(rects[idx[i]]) != key(rects[idx[i0]])):
            spans.append((i0, i))
            i0 = i
    return spans


def _frames(idx, device):
    """Index for a list of frames: a slice (a view) when they are a run."""
    if idx[-1] - idx[0] + 1 == len(idx):
        return slice(idx[0], idx[-1] + 1)
    return torch.tensor(idx, device=device)


def _put(dst, idx, val):
    dst[_frames(idx, dst.device)] = val


def _crop_axis_chunk(image, in_mask, rect_mask, rects, idx, w, h, fill):
    """Axis-aligned frames of one size: an exact slice per run of identical
    integer rects, filling the outside area per `fill`. rect_mask / the
    cropped mask reflect only the real image overlap (the filled area isn't
    real coverage). Without an input mask the cropped mask is one plane of
    ones, broadcast on write."""
    _, H, W, _ = image.shape
    device, dtype = image.device, image.dtype
    crops, masks = [], []
    for i0, i1 in _runs(rects, idx, _axis_key):
        a, b = idx[i0], idx[i1 - 1] + 1
        ix, iy = _axis_key(rects[a])
        src_x0 = max(0, ix)
        src_y0 = max(0, iy)
        src_x1 = min(W, ix + w)
        src_y1 = min(H, iy + h)
        dst_x0 = src_x0 - ix
        dst_y0 = src_y0 - iy
        overlap = src_x1 > src_x0 and src_y1 > src_y0
        crops.append(_extract_axis_aligned(image[a:b], ix, iy, w, h, fill))
        if overlap:
            rect_mask[a:b, src_y0:src_y1, src_x0:src_x1] = 1.0
        if in_mask is not None:
            m = torch.zeros((b - a, h, w), device=device, dtype=dtype)
            if overlap:
                m[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                     dst_x0:dst_x0 + (src_x1 - src_x0)] = \
                    in_mask[a:b, src_y0:src_y1, src_x0:src_x1]
            masks.append(m)
    cropped = crops[0] if len(crops) == 1 else torch.cat(crops)
    if in_mask is None:
        return cropped, torch.ones((1, h, w), device=device, dtype=dtype)
    return cropped, masks[0] if len(masks) == 1 else torch.cat(masks)


def _crop_rotated_chunk(image, in_mask, rect_mask, rects, idx, w, h, fill,
                        memo):
    """Rotated frames of one size in one grid_sample. The (k, h, w, 2) grid
    is built a run of identical rects at a time (a static rotated rect is a
    single expanded grid); the sampling is bat_crop._rotated_crop /
    _rotated_crop_mask, which only take one rect for a whole batch. `memo`
    carries the last rect's grid and full-canvas rect mask into the next
    chunk, so a static rect builds each once for the whole clip.

    x/y stay float: grid_sample is sub-pixel, and Uncrop pastes at the float
    rect it reads from `frames`. Rounding them made the paste land up to
    0.5px off, alternating frame to frame on an interpolated move (shimmer)."""
    _, H, W, _ = image.shape
    device, dtype = image.device, image.dtype
    f32 = torch.float32
    k = len(idx)
    runs = _runs(rects, idx, _rot_key)
    grid = None
    for i0, i1 in runs:
        a, b = idx[i0], idx[i1 - 1] + 1
        r = rects[a]
        key = (w, h) + _rot_key(r)
        if memo.get("key") != key:
            memo["key"] = key
            memo["grid"] = _make_crop_grid(r["x"], r["y"], w, h, H, W,
                                           r["angle"], 1, device, f32)
            memo["rect"] = _rotated_rect_mask(
                H, W, r["x"] + (w - 1) / 2.0, r["y"] + (h - 1) / 2.0,
                w, h, r["angle"], 1, device, dtype)
        g = memo["grid"].expand(b - a, h, w, 2)
        if len(runs) == 1:
            grid = g
        else:
            if grid is None:
                grid = torch.empty((k, h, w, 2), device=device, dtype=f32)
            grid[i0:i1] = g
        rect_mask[a:b] = memo["rect"]
    sel = _frames(idx, device)
    pad_mode = _GRID_PAD.get(fill, "zeros")      # gray samples zeros too
    out = F.grid_sample(image[sel].permute(0, 3, 1, 2).to(f32), grid,
                        mode="bilinear", padding_mode=pad_mode,
                        align_corners=True)
    if fill == "gray":
        # Where the sample fell outside (coverage 0), paint grey.
        ones = torch.ones((1, 1, H, W), device=device, dtype=f32).expand(k, 1, H, W)
        cov = F.grid_sample(ones, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        out = out + (1.0 - cov) * _GRAY_VALUE
    cropped = out.permute(0, 2, 3, 1).contiguous()
    if in_mask is None:
        return cropped, torch.ones((1, h, w), device=device, dtype=dtype)
    cropped_m = F.grid_sample(in_mask[sel].unsqueeze(1).to(f32), grid,
                              mode="bilinear", padding_mode=pad_mode,
                              align_corners=True)
    return cropped, cropped_m.squeeze(1).clamp(0, 1).contiguous()


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


def _preview_strip(image):
    """Input frames as base64 JPEGs for the canvas editor, and their stride
    (same subsampling pattern as Bat_Roto so long sequences don't blow up
    the ui payload)."""
    n = image.shape[0]
    max_preview_frames = 240
    # ceil, not floor: n // 240 is 1 for anything under 480 frames, so the
    # "cap" let up to 479 thumbnails through.
    stride = max(1, math.ceil(n / max_preview_frames))

    def one(i):
        arr = (image[i].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return _b64_jpeg(arr)

    # PIL releases the GIL while it resizes and encodes, so the thumbnails go
    # in parallel (same bytes, same order). Once the crop itself was batched,
    # encoding them one by one was ~90% of this node's run time.
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        frames_b64 = list(pool.map(one, range(0, n, stride)))
    return frames_b64, stride


class BatAnimatedCrop:
    """Keyframed crop rect across a frame batch. Compatible with Bat_Uncrop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # New nodes carry the seed marker (see _seed_rect); a saved
                # workflow keeps whatever state string it was saved with.
                "state": ("STRING", {
                    "default": '{"keyframes":{},"seed":"centre"}',
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
    CATEGORY = "BAT/Transform"
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
        empty = _seed_rect(doc, W, H)

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

        # Uniform output size = snap of the first keyframe's dimensions
        # (of the seed rect when no keyframes exist).
        first_rect = _resolve_rect_at_frame(kfs, 0, empty)
        ref_w = max(1, int(round(first_rect["w"])))
        ref_h = max(1, int(round(first_rect["h"])))
        out_w = _snap(ref_w, snap_to)
        out_h = _snap(ref_h, snap_to)
        # Every frame's rect is scaled about its centre by out/ref BEFORE
        # extraction — the per-frame equivalent of Bat_Crop snapping its rect
        # before slicing. A frame at the reference size then slices at exactly
        # out_w×out_h with no resample; before, a non-multiple size (the
        # editor's default 960×540 seed on 1080p → 536) bicubic-resampled every
        # frame, and Uncrop resampled it back. Scaling rather than snapping
        # each frame keeps an animated zoom smooth instead of stepping in
        # snap_to increments.
        scale_w = out_w / ref_w
        scale_h = out_h / ref_h

        # Every frame's rect is resolved first, then frames sharing an output
        # shape are cropped as batches: one slice per run of identical integer
        # rects, one grid_sample per chunk of rotated frames, one resize per
        # chunk, all written into preallocated outputs. Frame-at-a-time
        # (a full-canvas mask allocated per frame, a resize per frame, three
        # lists torch.cat'd at the end) made this ~25x slower than Bat_Crop
        # on the same static rect. The maths per frame is unchanged, so the
        # result is bit-identical (tests/verify_animated_crop.py).
        per_frame_rects = []
        for f in range(n):
            rect = _resolve_rect_at_frame(kfs, f, empty)
            w = max(1, int(round(float(rect["w"]) * scale_w)))
            h = max(1, int(round(float(rect["h"]) * scale_h)))
            # Keep the drawn centre where the artist put it.
            x = float(rect["x"]) + (float(rect["w"]) - w) / 2.0
            y = float(rect["y"]) + (float(rect["h"]) - h) / 2.0
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

        rotated = [not abs(r["angle"]) < 1e-3 for r in per_frame_rects]
        # Output dtypes exactly as torch.cat promoted the per-frame pieces:
        # rotated crops and resized masks come back float32. (All float32 for
        # a normal IMAGE; this only matters for another input dtype.)
        f32 = torch.float32
        img_dts = {f32 if rot else dtype for rot in rotated}
        mask_dts = {
            f32 if ((r["w"], r["h"]) != (out_w, out_h)
                    or (rot and in_mask is not None)) else dtype
            for r, rot in zip(per_frame_rects, rotated)}
        img_dt = functools.reduce(torch.promote_types, img_dts, next(iter(img_dts), dtype))
        mask_dt = functools.reduce(torch.promote_types, mask_dts, next(iter(mask_dts), dtype))
        out_image = torch.empty((n, out_h, out_w, c), dtype=img_dt, device=device)
        out_mask = torch.empty((n, out_h, out_w), dtype=mask_dt, device=device)
        rect_mask = torch.zeros((n, H, W), dtype=dtype, device=device)

        groups = {}
        for f, r in enumerate(per_frame_rects):
            groups.setdefault((rotated[f], r["w"], r["h"]), []).append(f)
        memo = {}
        for (rot, w, h), frames in groups.items():
            per_frame = 4 * (H * W * (c + 1) + 3 * h * w * (c + 2)
                             + out_h * out_w * (c + 1))
            step = max(1, _CHUNK_BYTES // per_frame)
            for s in range(0, len(frames), step):
                idx = frames[s:s + step]
                if rot:
                    cropped, cropped_m = _crop_rotated_chunk(
                        image, in_mask, rect_mask, per_frame_rects, idx,
                        w, h, outside_fill, memo)
                else:
                    cropped, cropped_m = _crop_axis_chunk(
                        image, in_mask, rect_mask, per_frame_rects, idx,
                        w, h, outside_fill)
                cropped = _resize_nhwc(cropped, out_h, out_w)
                cropped_m = _resize_nhw(cropped_m, out_h, out_w)
                _put(out_image, idx, cropped)
                _put(out_mask, idx, cropped_m)

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

        frames_b64, stride = _preview_strip(image)

        # The strip goes to a sidecar file (bat_ui_ref): inline, every run
        # kept MBs of base64 in ComfyUI's prompt history.
        return {
            "ui": stash_ui({
                "frames": frames_b64,
                "w": [int(W)],
                "h": [int(H)],
                "stride": [int(stride)],
                "frame_count": [int(n)],
            }),
            "result": (out_image, crop_info, out_mask, rect_mask),
        }
