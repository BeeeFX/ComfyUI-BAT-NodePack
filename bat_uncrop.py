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
    # n in (1, target_n): HOLD the last frame to pad up.
    #
    # This used to `batch.repeat(reps)[:target_n]`, which cyclically TILES the
    # batch rather than holding — a 3-frame batch into a 10-frame plate became
    # 0,1,2,0,1,2,0,1,2,0 instead of 0,1,2,2,2,2,2,2,2,2. Since this helper
    # broadcasts both the plate and the paste, an artist who trimmed a batch
    # mid-graph got frames composited from the WRONG source frames with no
    # warning. Matches bat_crop._broadcast_mask_to_n, which always did this
    # correctly.
    pad_n = target_n - n
    return torch.cat(
        [batch, batch[-1:].expand((pad_n,) + (-1,) * (batch.ndim - 1))], dim=0
    )


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
    """A (1,H,W) mask = feathered rect placed at (x,y,w,h) on a HxW canvas.

    Bounds-safe: when the rect extends off the canvas (e.g. an animated
    crop where the source rect swings past the image edge mid-clip), we
    clip both the destination slice on the canvas and the source slice
    on the feathered patch so they always match. The previous bare
    `mask[0, y:y+h, x:x+w] = …` would silently misbehave with negative x
    or y — Python's negative slice semantics meant the feather got
    written to entirely the wrong part of the canvas, producing
    misaligned mattes during animations where the rect went off-screen.
    """
    mask = torch.zeros((1, H, W), device=device, dtype=dtype)
    if w <= 0 or h <= 0:
        return mask
    src_x0 = max(0, x)
    src_y0 = max(0, y)
    src_x1 = min(W, x + w)
    src_y1 = min(H, y + h)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return mask
    feathered = _feather_local(h, w, feather, device, dtype)
    dst_x0 = src_x0 - x
    dst_y0 = src_y0 - y
    mask[0, src_y0:src_y1, src_x0:src_x1] = feathered[
        dst_y0:dst_y0 + (src_y1 - src_y0),
        dst_x0:dst_x0 + (src_x1 - src_x0),
    ]
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
    DESCRIPTION = (
        "Paste a processed crop back onto its original plate using the "
        "BAT_CROP_INFO from Bat_Crop or Bat_AnimatedCrop. Handles rotated "
        "rects, per-frame animated rects, edge feathering and aspect fit "
        "modes."
    )
    FUNCTION = "uncrop"

    def uncrop(self, processed, crop_info, fit_mode, filter, feather_px):
        if not isinstance(crop_info, dict) or "original_image" not in crop_info:
            raise ValueError("Bat_Uncrop: missing crop_info from a Bat_Crop node")
        original = crop_info["original_image"]
        H = int(crop_info["original_h"])
        W = int(crop_info["original_w"])

        # New: per-frame source rect list, written by Bat_AnimatedCrop.
        # When present, dispatch to the animated path; otherwise fall
        # back to the original single-rect implementation untouched.
        per_frame = crop_info.get("frames")
        if isinstance(per_frame, list) and per_frame:
            return self._uncrop_animated(
                processed, original, per_frame, H, W,
                fit_mode, filter, feather_px,
            )

        x = int(crop_info["x"])
        y = int(crop_info["y"])
        w = int(crop_info["w"])
        h = int(crop_info["h"])
        angle = float(crop_info.get("angle", 0.0))  # 0 for back-compat with older crop_info dicts

        # w/h is the destination rect on the plate — correct for pasting.
        # Bat_Crop now snaps the crop rect BEFORE slicing, so the image it
        # emitted is exactly w×h and the `pw == w and ph == h` fast path below
        # hits, i.e. no resampling. (Legacy crop_info dicts written before that
        # fix may carry a pre-snap w/h with the image actually at out_w×out_h;
        # those fall through to the fit_mode branches as they always did.)

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
            # Bounds-safe blit: the paste rect can start off the top/left edge
            # (negative paste_x/y) or run past the right/bottom when the crop
            # was made with constrain_to_canvas off. Clip the destination slice
            # on the canvas and the matching source slice on the paste so they
            # stay the same size — a bare negative-index slice would silently
            # write to the wrong region (same fix as _feather_rect_mask).
            src_x0 = max(0, paste_x)
            src_y0 = max(0, paste_y)
            src_x1 = min(W, paste_x + paste_w)
            src_y1 = min(H, paste_y + paste_h)
            if src_x1 > src_x0 and src_y1 > src_y0:
                dst_x0 = src_x0 - paste_x
                dst_y0 = src_y0 - paste_y
                paste_canvas[:, src_y0:src_y1, src_x0:src_x1, :] = \
                    paste[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                             dst_x0:dst_x0 + (src_x1 - src_x0), :]
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

    def _uncrop_animated(self, processed, original, per_frame, H, W,
                         fit_mode, filter, feather_px):
        """Animated path: paste each processed frame back at its own source
        rect (read from `crop_info["frames"]`). Frame counts on both sides
        are broadcast against the per-frame list — same convention as the
        scalar path. Slower than the scalar path because each frame builds
        its own resample grid, but typical batch sizes (a few hundred
        frames) are well within budget.

        fit_mode is intentionally ignored here. Bat_AnimatedCrop does an
        unconditional non-uniform stretch from each per-frame rect
        (w_i, h_i) to the fixed uniform output (out_w, out_h) — it has to,
        because the per-frame rect aspect varies but the output tensor
        shape must not. The only correct reversal of an unconditional
        stretch is another unconditional stretch. With fit_mode="fit"
        (the default) the reverse would letterbox inside the rect on any
        frame where the aspect drifted from the first keyframe's;
        "cover" would center-crop. Both produce visible misalignment as
        the rect is scaled non-uniformly across the clip.
        """
        # Forced for the animated path — see docstring.
        fit_mode = "stretch"
        device = original.device
        dtype = torch.float32

        target_n = max(processed.shape[0], original.shape[0], len(per_frame))
        base = _broadcast_to(original.to(dtype), target_n).clone()
        proc = _broadcast_to(processed.to(dtype), target_n)

        ph, pw = int(proc.shape[1]), int(proc.shape[2])

        out_frames = []
        out_masks = []

        # Pixel grid for the rotated path, built ONCE. It depends only on
        # (H, W, device, dtype) — the per-frame rect centre and angle are applied
        # to it as cheap arithmetic below. It used to be rebuilt inside the loop,
        # so a 300-frame 1080p batch constructed 300 redundant 2M-element grids.
        # Built lazily: the axis-aligned branch never needs it.
        _grid_cache = {}

        def _pixel_grid():
            g = _grid_cache.get("yx")
            if g is None:
                g = torch.meshgrid(
                    torch.arange(H, device=device, dtype=dtype),
                    torch.arange(W, device=device, dtype=dtype),
                    indexing="ij",
                )
                _grid_cache["yx"] = g
            return g

        for i in range(target_n):
            # Per-frame rect; clamp index in case `per_frame` is shorter
            # than the batch (caller may have padded the image batch
            # after the crop node — we hold the last rect, like the
            # interpolator does on the encode side).
            rect = per_frame[min(i, len(per_frame) - 1)]
            x = float(rect["x"])
            y = float(rect["y"])
            w = max(1, int(rect["w"]))
            h = max(1, int(rect["h"]))
            angle = float(rect.get("angle", 0.0))

            # Match the scalar path's fit_mode arithmetic but on one
            # frame at a time.
            if fit_mode == "stretch" or (pw == w and ph == h):
                paste = _resize_nhwc(proc[i:i + 1], h, w, filter)
                paste_x, paste_y, paste_w, paste_h = int(round(x)), int(round(y)), w, h
            elif fit_mode == "cover":
                scale = max(w / pw, h / ph)
                nw = max(1, int(round(pw * scale)))
                nh = max(1, int(round(ph * scale)))
                big = _resize_nhwc(proc[i:i + 1], nh, nw, filter)
                cx = max(0, (nw - w) // 2)
                cy = max(0, (nh - h) // 2)
                paste = big[:, cy:cy + h, cx:cx + w, :].contiguous()
                paste_x, paste_y, paste_w, paste_h = int(round(x)), int(round(y)), w, h
            else:  # "fit"
                scale = min(w / pw, h / ph)
                nw = max(1, int(round(pw * scale)))
                nh = max(1, int(round(ph * scale)))
                paste = _resize_nhwc(proc[i:i + 1], nh, nw, filter)
                paste_x = int(round(x)) + (w - nw) // 2
                paste_y = int(round(y)) + (h - nh) // 2
                paste_w, paste_h = nw, nh

            base_i = base[i:i + 1]

            if abs(angle) < 1e-3:
                # Axis-aligned fast path for this frame.
                m = _feather_rect_mask(H, W, paste_x, paste_y, paste_w, paste_h,
                                       feather_px, device, dtype)
                m4 = m.unsqueeze(-1)
                paste_canvas = torch.zeros_like(base_i)
                # Clip the paste region to the canvas so out-of-bounds rects
                # don't blow up the slice.
                src_x0 = max(0, paste_x)
                src_y0 = max(0, paste_y)
                src_x1 = min(W, paste_x + paste_w)
                src_y1 = min(H, paste_y + paste_h)
                dst_x0 = src_x0 - paste_x
                dst_y0 = src_y0 - paste_y
                if src_x1 > src_x0 and src_y1 > src_y0:
                    paste_canvas[:, src_y0:src_y1, src_x0:src_x1, :] = \
                        paste[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                                 dst_x0:dst_x0 + (src_x1 - src_x0), :]
                frame_out = (base_i * (1.0 - m4) + paste_canvas * m4).clamp(0, 1)
                out_frames.append(frame_out)
                out_masks.append(m)
                continue

            # Rotated path: same math as the scalar branch but for one frame.
            rect_cx = x + w / 2.0
            rect_cy = y + h / 2.0
            a = math.radians(angle)
            cos_t, sin_t = math.cos(a), math.sin(a)
            ys, xs = _pixel_grid()   # built once, reused every frame
            dx = xs - rect_cx
            dy = ys - rect_cy
            u = cos_t * dx + sin_t * dy
            v = -sin_t * dx + cos_t * dy
            pu = u + paste_w / 2.0
            pv = v + paste_h / 2.0
            gx = pu / max(1, paste_w - 1) * 2.0 - 1.0
            gy = pv / max(1, paste_h - 1) * 2.0 - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

            paste_nchw = paste.permute(0, 3, 1, 2)
            sampled = F.grid_sample(paste_nchw, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
            paste_canvas = sampled.permute(0, 2, 3, 1).clamp(0, 1)

            local = _feather_local(paste_h, paste_w, feather_px, device, dtype)
            local_n = local.unsqueeze(0).unsqueeze(0)
            m_sampled = F.grid_sample(local_n, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=True)
            m_batch = m_sampled[:, 0].clamp(0, 1)
            m4 = m_batch.unsqueeze(-1)

            frame_out = (base_i * (1.0 - m4) + paste_canvas * m4).clamp(0, 1)
            out_frames.append(frame_out)
            out_masks.append(m_batch)

        out = torch.cat(out_frames, dim=0).contiguous()
        m_out = torch.cat(out_masks, dim=0).contiguous()
        return (out, m_out)
