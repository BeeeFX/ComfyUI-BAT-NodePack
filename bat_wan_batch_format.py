import torch

VAE_TEMPORAL_STRIDE = 4


def _nearest_wan_compatible_num_frames(current_frames, context_frames, context_overlap,
                                       with_ref_or_end_frame, round_up):
    cf_lat = (context_frames - 1) // VAE_TEMPORAL_STRIDE + 1
    co_lat = context_overlap // VAE_TEMPORAL_STRIDE
    delta_lat = cf_lat - co_lat
    if delta_lat <= 0:
        return current_frames

    target_lvl = (current_frames - 1) // VAE_TEMPORAL_STRIDE + 1 + (1 if with_ref_or_end_frame else 0)
    base_lvl = target_lvl - (1 if with_ref_or_end_frame else 0)

    candidates = []
    n_windows = 1
    while True:
        candidate = cf_lat + (n_windows - 1) * delta_lat
        candidate_base = candidate - (1 if with_ref_or_end_frame else 0)
        if candidate_base > base_lvl + 200 and candidates:
            break
        if candidate_base >= 1:
            candidates.append(candidate_base)
        n_windows += 1
        if n_windows > 1000:
            break

    pixel_candidates = [(b - 1) * VAE_TEMPORAL_STRIDE + 1 for b in candidates]
    if not pixel_candidates:
        return current_frames

    if round_up:
        above_or_eq = [p for p in pixel_candidates if p >= current_frames]
        if above_or_eq:
            return min(above_or_eq)

    return min(pixel_candidates, key=lambda p: abs(p - current_frames))


def _normalize_mask(t):
    if t is None:
        return None
    if t.ndim == 2:
        return t.unsqueeze(0)
    return t


def _pad_image_like(t, pad_count, pad_method, pad_position, grey_value):
    """Pad an [N, H, W, C] image tensor."""
    if pad_method == "wan_inpaint_grey":
        H, W, C = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
        return torch.full((pad_count, H, W, C), grey_value, dtype=t.dtype, device=t.device)
    edge = t[-1:] if pad_position == "end" else t[0:1]
    return edge.repeat(pad_count, 1, 1, 1)


def _pad_mask_like(t, pad_count, pad_method, pad_position):
    """Pad an [N, H, W] mask tensor."""
    if pad_method == "wan_inpaint_grey":
        H, W = int(t.shape[-2]), int(t.shape[-1])
        return torch.ones((pad_count, H, W), dtype=t.dtype, device=t.device)
    edge = t[-1:] if pad_position == "end" else t[0:1]
    return edge.repeat(pad_count, 1, 1)


# Distinct, eye-friendly palette. Cycles for window counts > len.
_WINDOW_PALETTE = [
    (0.95, 0.30, 0.30),   # red
    (0.30, 0.85, 0.40),   # green
    (0.30, 0.55, 0.95),   # blue
    (0.95, 0.85, 0.25),   # yellow
    (0.95, 0.55, 0.20),   # orange
    (0.65, 0.40, 0.95),   # purple
    (0.25, 0.85, 0.85),   # cyan
    (0.95, 0.50, 0.75),   # pink
]


def _compute_static_windows(lvl, cf_lat, co_lat):
    """Mirror static_standard from WanVideoWrapper's context_windows/context.py."""
    if cf_lat <= 0 or co_lat >= cf_lat:
        return [list(range(lvl))] if lvl > 0 else []
    if lvl <= cf_lat:
        return [list(range(lvl))]
    delta_lat = cf_lat - co_lat
    windows = []
    start_idx = 0
    while True:
        ending = start_idx + cf_lat
        if ending >= lvl:
            final_delta = ending - lvl
            final_start_idx = start_idx - final_delta
            windows.append(list(range(final_start_idx, final_start_idx + cf_lat)))
            break
        windows.append(list(range(start_idx, start_idx + cf_lat)))
        start_idx += delta_lat
    return windows


def _build_window_viz(N, H, W, context_frames, context_overlap, use_ref_or_end_frame,
                       frames_added_start, frames_added_end):
    """Return an [N, H, W, 3] image batch where each frame's color reflects its
    sliding-context window membership (split into horizontal bands when in an
    overlap region). Padded frames are dimmed to ~40% intensity."""
    cf_lat = (context_frames - 1) // VAE_TEMPORAL_STRIDE + 1
    co_lat = context_overlap // VAE_TEMPORAL_STRIDE
    lvl = (N - 1) // VAE_TEMPORAL_STRIDE + 1 + (1 if use_ref_or_end_frame else 0)
    windows = _compute_static_windows(lvl, cf_lat, co_lat)

    out = torch.zeros((N, H, W, 3), dtype=torch.float32)

    for p in range(N):
        # Pixel p maps to latent index lat (with ref offset baked in)
        lat = (p + (VAE_TEMPORAL_STRIDE - 1)) // VAE_TEMPORAL_STRIDE + (1 if use_ref_or_end_frame else 0)
        containing = [i for i, w in enumerate(windows) if lat in w]

        is_padded = (p < frames_added_start) or (p >= N - frames_added_end)
        dim_factor = 0.4 if is_padded else 1.0

        if not containing:
            color = (0.08, 0.08, 0.08)
            out[p, :, :, 0] = color[0]
            out[p, :, :, 1] = color[1]
            out[p, :, :, 2] = color[2]
            continue

        # Horizontal bands, one per containing window — easy to spot overlaps visually
        n_bands = len(containing)
        band_h = max(1, H // n_bands)
        for bi, win_idx in enumerate(containing):
            color = _WINDOW_PALETTE[win_idx % len(_WINDOW_PALETTE)]
            y0 = bi * band_h
            y1 = (bi + 1) * band_h if bi < n_bands - 1 else H
            out[p, y0:y1, :, 0] = color[0] * dim_factor
            out[p, y0:y1, :, 1] = color[1] * dim_factor
            out[p, y0:y1, :, 2] = color[2] * dim_factor

    return out


class VoltWanBatchFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["nearest_wan_compatible", "specific_num_frames", "arbitrary_pad"], {
                    "default": "nearest_wan_compatible",
                    "tooltip": "nearest_wan_compatible = pad to closest WAN-friendly length using context params. specific_num_frames = pad to exactly target_num_frames. arbitrary_pad = ignore target, just add pad_frames at the chosen position.",
                }),
                "pad_method": (["wan_inpaint_grey", "repeat_edge"], {
                    "default": "wan_inpaint_grey",
                    "tooltip": "wan_inpaint_grey = grey RGB + white alpha (mask=1) so WAN can fill in. repeat_edge = duplicate the first or last real frame.",
                }),
                "pad_position": (["end", "start", "both"], {"default": "end",
                                                            "tooltip": "Where to add the padded frames. 'both' splits evenly (extra frame goes to end if odd)."}),
                "target_num_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1,
                                              "tooltip": "Used when mode=specific_num_frames. Will not truncate if shorter than input — use the cropper afterwards if you need exact trimming."}),
                "pad_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                       "tooltip": "Used when mode=arbitrary_pad. Number of frames to add at the chosen position (preroll if start, postroll if end)."}),
                "context_frames": ("INT", {"default": 81, "min": 5, "max": 1001, "step": 4,
                                           "tooltip": "Used when mode=nearest_wan_compatible."}),
                "context_overlap": ("INT", {"default": 16, "min": 0, "max": 100, "step": 4,
                                            "tooltip": "Used when mode=nearest_wan_compatible."}),
                "use_ref_or_end_frame": ("BOOLEAN", {"default": True,
                                                     "tooltip": "Used when mode=nearest_wan_compatible. Mirrors the calculator's flag."}),
                "round_up": ("BOOLEAN", {"default": True,
                                         "tooltip": "Used when mode=nearest_wan_compatible. Pick smallest valid length >= input length so no input frames get cropped."}),
                "grey_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Grey level for wan_inpaint_grey padded RGB."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "control_video": ("IMAGE", {"tooltip": "Optional control image batch (same length as image). Will be padded the same way and returned via the control_video output."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "control_video", "debug_window_viz",
                    "frames_added_start", "frames_added_end", "total_frames")
    FUNCTION = "format"
    CATEGORY = "BAT/WAN"
    DESCRIPTION = "Format an input image batch into the WAN sliding-context structure. Pads / trims to target frame count, applies static-standard window timing, and emits a debug viz of window boundaries."

    def format(self, mode, pad_method, pad_position, target_num_frames, pad_frames,
               context_frames, context_overlap, use_ref_or_end_frame, round_up, grey_value,
               image=None, mask=None, control_video=None):
        mask = _normalize_mask(mask)

        # Resolve reference length and HxW from whichever input was provided
        sources = []
        if image is not None:
            sources.append(("image", int(image.shape[0]), int(image.shape[1]), int(image.shape[2])))
        if control_video is not None:
            sources.append(("control_video", int(control_video.shape[0]), int(control_video.shape[1]), int(control_video.shape[2])))
        if mask is not None:
            sources.append(("mask", int(mask.shape[0]), int(mask.shape[-2]), int(mask.shape[-1])))

        if not sources:
            raise ValueError("VoltWanBatchFormat: provide at least one of image, mask, or control_video.")

        # All provided sources must agree on frame count
        ref_count = sources[0][1]
        for name, n, _, _ in sources[1:]:
            if n != ref_count:
                raise ValueError(
                    f"VoltWanBatchFormat: input length mismatch — {sources[0][0]} has {ref_count} frames "
                    f"but {name} has {n} frames. They must match."
                )

        current_frames = ref_count
        H, W = sources[0][2], sources[0][3]

        # Manufacture missing companions so we can always output all three
        if image is None:
            image = torch.full((current_frames, H, W, 3), grey_value, dtype=torch.float32)
        if mask is None:
            mask = torch.zeros((current_frames, H, W), dtype=torch.float32)
        # control_video is left as None if absent — we'll output a sensible default below

        # Resolve pad count
        if mode == "arbitrary_pad":
            pad_count = max(0, int(pad_frames))
        elif mode == "specific_num_frames":
            pad_count = max(0, int(target_num_frames) - current_frames)
        else:  # nearest_wan_compatible
            target = _nearest_wan_compatible_num_frames(
                current_frames, context_frames, context_overlap, use_ref_or_end_frame, round_up
            )
            pad_count = max(0, target - current_frames)

        if pad_count <= 0:
            cv_out = control_video if control_video is not None else torch.full((current_frames, H, W, 3), grey_value, dtype=torch.float32)
            viz = _build_window_viz(current_frames, H, W, context_frames, context_overlap,
                                     use_ref_or_end_frame, 0, 0)
            return (image, mask, cv_out, viz, 0, 0, current_frames)

        # Decide split between start and end
        if pad_position == "end":
            added_start, added_end = 0, pad_count
        elif pad_position == "start":
            added_start, added_end = pad_count, 0
        else:  # both — even split, odd extra goes to the end
            added_start = pad_count // 2
            added_end = pad_count - added_start

        def _build_segment(t, pad_method, count, side, grey_value, is_mask):
            if count <= 0:
                return None
            if is_mask:
                return _pad_mask_like(t, count, pad_method, side)
            return _pad_image_like(t, count, pad_method, side, grey_value)

        # Build per-side pads
        pad_img_start = _build_segment(image, pad_method, added_start, "start", grey_value, False)
        pad_img_end = _build_segment(image, pad_method, added_end, "end", grey_value, False)
        pad_msk_start = _build_segment(mask, pad_method, added_start, "start", grey_value, True)
        pad_msk_end = _build_segment(mask, pad_method, added_end, "end", grey_value, True)
        pad_ctrl_start = _build_segment(control_video, pad_method, added_start, "start", grey_value, False) if control_video is not None else None
        pad_ctrl_end = _build_segment(control_video, pad_method, added_end, "end", grey_value, False) if control_video is not None else None

        def _stitch(parts):
            parts = [p for p in parts if p is not None]
            return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        out_image = _stitch([pad_img_start, image, pad_img_end])
        out_mask = _stitch([pad_msk_start, mask, pad_msk_end])
        if control_video is not None:
            out_control = _stitch([pad_ctrl_start, control_video, pad_ctrl_end])
        else:
            total_now = int(out_image.shape[0])
            out_control = torch.full((total_now, H, W, 3), grey_value, dtype=torch.float32)

        total_now = int(out_image.shape[0])
        viz = _build_window_viz(total_now, H, W, context_frames, context_overlap,
                                 use_ref_or_end_frame, added_start, added_end)
        return (out_image, out_mask, out_control, viz, added_start, added_end, total_now)


class VoltWanBatchCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_added_start": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                               "tooltip": "Pass through the frames_added_start output of VoltWanBatchFormat."}),
                "frames_added_end": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                             "tooltip": "Pass through the frames_added_end output of VoltWanBatchFormat."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "control_video": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "control_video")
    FUNCTION = "crop"
    CATEGORY = "BAT/WAN"
    DESCRIPTION = "Crop a WAN-formatted batch back to its original frame count after generation, removing the start/end padding produced by ETC WAN Batch Format."

    def crop(self, frames_added_start, frames_added_end, image=None, mask=None, control_video=None):
        if image is None and mask is None and control_video is None:
            raise ValueError("VoltWanBatchCrop: provide at least one of image, mask, or control_video.")

        mask = _normalize_mask(mask)
        s = max(0, int(frames_added_start))
        e = max(0, int(frames_added_end))

        def _crop_one(t, label):
            if t is None:
                return None
            total = int(t.shape[0])
            end_idx = total - e if e > 0 else total
            if s >= total or end_idx <= s:
                raise ValueError(
                    f"VoltWanBatchCrop: cropping {s} from start and {e} from end leaves nothing "
                    f"in a {total}-frame {label}."
                )
            return t[s:end_idx]

        return (_crop_one(image, "image"), _crop_one(mask, "mask"), _crop_one(control_video, "control_video"))
