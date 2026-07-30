"""
Bat WAN Batch Frame Format — a slimmed-down sibling of
bat_wan_batch_format.py's VoltWanBatchFormat.

Keeps the parts most people actually reach for — the three padding
modes, pad_method, pad_position, pad_frames — and the WAN 4n+1
frame-length constraint. Drops the sliding-context machinery
(context_frames / context_overlap / window viz / control_video),
so "nearest WAN-compatible" simply rounds to the closest 4n+1.

Takes image + mask, returns image + mask (plus the padding counts
so the result can be cropped back with VoltWanBatchCrop).
"""

import torch

from .bat_wan_batch_format import (
    VAE_TEMPORAL_STRIDE,
    _normalize_mask,
    _pad_image_like,
    _pad_mask_like,
)


def _nearest_4n_plus_1(current_frames, round_up):
    """Closest WAN-compatible length (1, 5, 9, 13, ... = 4k+1) to current_frames.

    round_up=True picks the smallest 4k+1 >= current_frames so no input
    frames would ever need cropping. round_up=False picks the nearest.
    """
    if current_frames <= 1:
        return 1
    k = (current_frames - 1) // VAE_TEMPORAL_STRIDE
    lower = VAE_TEMPORAL_STRIDE * k + 1
    upper = lower if lower >= current_frames else lower + VAE_TEMPORAL_STRIDE

    if round_up:
        # `upper` is already >= current_frames by construction (it's `lower`
        # when lower >= current_frames, else lower + stride), so the old
        # `upper if upper >= current_frames else upper + STRIDE` fallback was
        # dead code that could only ever overshoot by a whole stride.
        return upper

    return lower if abs(current_frames - lower) <= abs(upper - current_frames) else upper


class BatWanBatchFrameFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["nearest_wan_compatible", "specific_num_frames", "arbitrary_pad"], {
                    "default": "nearest_wan_compatible",
                    "tooltip": "nearest_wan_compatible = pad to the closest WAN-friendly 4n+1 length. specific_num_frames = pad to exactly target_num_frames. arbitrary_pad = ignore target, just add pad_frames at the chosen position.",
                }),
                "pad_method": (["wan_inpaint_grey", "repeat_edge"], {
                    "default": "wan_inpaint_grey",
                    "tooltip": "wan_inpaint_grey = grey RGB + white mask (mask=1) so WAN can fill in. repeat_edge = duplicate the first or last real frame.",
                }),
                "pad_position": (["end", "start", "both"], {
                    "default": "end",
                    "tooltip": "Where to add the padded frames. 'both' splits evenly (extra frame goes to end if odd).",
                }),
                "target_num_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1,
                                              "tooltip": "Used when mode=specific_num_frames. Never truncates if shorter than input — crop afterwards if you need exact trimming."}),
                "pad_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                       "tooltip": "Used when mode=arbitrary_pad. Number of frames to add at the chosen position."}),
                "round_up": ("BOOLEAN", {"default": True,
                                         "tooltip": "Used when mode=nearest_wan_compatible. Pick the smallest valid 4n+1 length >= input length so no input frames get cropped."}),
                "grey_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Grey level for wan_inpaint_grey padded RGB."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "control_video": ("IMAGE", {"tooltip": "Optional control image batch (same length as image). Padded the same way and returned via the control_video output."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "control_video", "frames_added_start", "frames_added_end", "total_frames")
    FUNCTION = "format"
    CATEGORY = "BAT/wan"
    DESCRIPTION = "Pad an image+mask (+optional control_video) batch to a WAN-compatible 4n+1 frame length (or a specific/arbitrary count). Returns the padding counts so VoltWanBatchCrop can trim back after generation."

    def format(self, mode, pad_method, pad_position, target_num_frames, pad_frames,
               round_up, grey_value, image=None, mask=None, control_video=None):
        mask = _normalize_mask(mask)

        sources = []
        if image is not None:
            sources.append(("image", int(image.shape[0]), int(image.shape[1]), int(image.shape[2])))
        if control_video is not None:
            sources.append(("control_video", int(control_video.shape[0]), int(control_video.shape[1]), int(control_video.shape[2])))
        if mask is not None:
            sources.append(("mask", int(mask.shape[0]), int(mask.shape[-2]), int(mask.shape[-1])))

        if not sources:
            raise ValueError("BatWanBatchFrameFormat: provide at least one of image, mask, or control_video.")

        ref_count = sources[0][1]
        for name, n, _, _ in sources[1:]:
            if n != ref_count:
                raise ValueError(
                    f"BatWanBatchFrameFormat: input length mismatch — {sources[0][0]} has {ref_count} frames "
                    f"but {name} has {n} frames. They must match."
                )

        current_frames = ref_count
        H, W = sources[0][2], sources[0][3]

        # Manufacture the missing companion so we always output both.
        #
        # Inherit device AND dtype from whichever input we do have: these used to
        # be hardcoded CPU/float32, so a CUDA control_video hit
        # "Expected all tensors to be on the same device" on the torch.cat below.
        _ref = None
        for _cand in (image, mask, control_video):
            if _cand is not None and hasattr(_cand, "device"):
                _ref = _cand
                break
        _dev = _ref.device if _ref is not None else None
        _dt = _ref.dtype if _ref is not None else torch.float32
        if image is None:
            image = torch.full((current_frames, H, W, 3), grey_value,
                               dtype=_dt, device=_dev)
        if mask is None:
            mask = torch.zeros((current_frames, H, W), dtype=_dt, device=_dev)

        # Resolve pad count
        if mode == "arbitrary_pad":
            pad_count = max(0, int(pad_frames))
        elif mode == "specific_num_frames":
            pad_count = max(0, int(target_num_frames) - current_frames)
        else:  # nearest_wan_compatible
            target = _nearest_4n_plus_1(current_frames, round_up)
            pad_count = max(0, target - current_frames)

        if pad_count <= 0:
            # Same device/dtype inheritance as the manufacture block above — this
            # early-return path also hardcoded CPU/float32.
            cv_out = (control_video if control_video is not None
                      else torch.full((current_frames, H, W, 3), grey_value,
                                      dtype=_dt, device=_dev))
            return (image, mask, cv_out, 0, 0, current_frames)

        # Decide split between start and end
        if pad_position == "end":
            added_start, added_end = 0, pad_count
        elif pad_position == "start":
            added_start, added_end = pad_count, 0
        else:  # both — even split, odd extra goes to the end
            added_start = pad_count // 2
            added_end = pad_count - added_start

        def _build_segment(t, count, side, is_mask):
            if count <= 0:
                return None
            if is_mask:
                return _pad_mask_like(t, count, pad_method, side)
            return _pad_image_like(t, count, pad_method, side, grey_value)

        pad_img_start = _build_segment(image, added_start, "start", False)
        pad_img_end = _build_segment(image, added_end, "end", False)
        pad_msk_start = _build_segment(mask, added_start, "start", True)
        pad_msk_end = _build_segment(mask, added_end, "end", True)
        pad_ctrl_start = _build_segment(control_video, added_start, "start", False) if control_video is not None else None
        pad_ctrl_end = _build_segment(control_video, added_end, "end", False) if control_video is not None else None

        def _stitch(parts):
            parts = [p for p in parts if p is not None]
            return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        out_image = _stitch([pad_img_start, image, pad_img_end])
        out_mask = _stitch([pad_msk_start, mask, pad_msk_end])
        total_now = int(out_image.shape[0])

        if control_video is not None:
            out_control = _stitch([pad_ctrl_start, control_video, pad_ctrl_end])
        else:
            out_control = torch.full((total_now, H, W, 3), grey_value, dtype=torch.float32)

        return (out_image, out_mask, out_control, added_start, added_end, total_now)
