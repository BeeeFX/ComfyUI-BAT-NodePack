import torch

from .etc_wan_batch_format import _normalize_mask, _pad_image_like, _pad_mask_like


# Each entry: pixel-frame stride. Valid pixel frame counts are F such that
# (F - 1) % stride == 0, i.e. F ∈ {1, stride+1, 2*stride+1, ...}.
# All three target models use causal video VAEs with these strides.
MODEL_STRIDES = {
    "WAN (stride 4)": 4,
    "Hunyuan (stride 4)": 4,
    "LTX (stride 8)": 8,
    "Custom": None,
}


def _nearest_valid_num_frames(current_frames, stride, round_up):
    if stride <= 0 or current_frames <= 1:
        return max(1, current_frames)
    n_floor = (current_frames - 1) // stride
    f_floor = stride * n_floor + 1
    if f_floor == current_frames:
        return current_frames
    f_ceil = stride * (n_floor + 1) + 1
    if round_up:
        return f_ceil
    return f_floor if (current_frames - f_floor) <= (f_ceil - current_frames) else f_ceil


class VoltVideoBatchFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_STRIDES.keys()), {"default": "WAN (stride 4)",
                                                       "tooltip": "Target video model. Pick the one you'll feed this batch into."}),
                "custom_temporal_stride": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1,
                                                   "tooltip": "Used when model=Custom. Pixel frames per latent frame in the model's VAE."}),
                "mode": (["nearest_compatible", "specific_num_frames", "arbitrary_pad"], {
                    "default": "nearest_compatible",
                    "tooltip": "nearest_compatible = pad to closest VAE-friendly length. specific_num_frames = pad to exactly target_num_frames. arbitrary_pad = just add pad_frames at the chosen position.",
                }),
                "pad_method": (["grey_inpaint", "repeat_edge"], {"default": "grey_inpaint",
                                                                  "tooltip": "grey_inpaint = grey RGB + white alpha (mask=1) so the model can fill in. repeat_edge = duplicate the first or last real frame."}),
                "pad_position": (["end", "start", "both"], {"default": "end",
                                                            "tooltip": "Where to add the padded frames. 'both' splits evenly (extra frame goes to end if odd)."}),
                "target_num_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1,
                                              "tooltip": "Used when mode=specific_num_frames. Will not truncate if shorter than input."}),
                "pad_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                       "tooltip": "Used when mode=arbitrary_pad. Number of frames to add at the chosen position (preroll if start, postroll if end)."}),
                "round_up": ("BOOLEAN", {"default": True,
                                         "tooltip": "Used when mode=nearest_compatible. Pick smallest valid length >= input length so no input frames get cropped."}),
                "grey_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Grey level for grey_inpaint padded RGB."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "control_video": ("IMAGE", {"tooltip": "Optional control image batch (same length as image). Will be padded the same way."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "control_video", "frames_added_start", "frames_added_end", "total_frames")
    FUNCTION = "format"
    CATEGORY = "BAT/Video"
    DESCRIPTION = "Generic video batch formatter — pad / trim / normalise an image batch to a target frame count using the selected model's temporal stride."

    def format(self, model, custom_temporal_stride, mode, pad_method, pad_position,
               target_num_frames, pad_frames, round_up, grey_value,
               image=None, mask=None, control_video=None):
        mask = _normalize_mask(mask)

        # Resolve reference length and HxW
        sources = []
        if image is not None:
            sources.append(("image", int(image.shape[0]), int(image.shape[1]), int(image.shape[2])))
        if control_video is not None:
            sources.append(("control_video", int(control_video.shape[0]), int(control_video.shape[1]), int(control_video.shape[2])))
        if mask is not None:
            sources.append(("mask", int(mask.shape[0]), int(mask.shape[-2]), int(mask.shape[-1])))

        if not sources:
            raise ValueError("VoltVideoBatchFormat: provide at least one of image, mask, or control_video.")

        ref_count = sources[0][1]
        for name, n, _, _ in sources[1:]:
            if n != ref_count:
                raise ValueError(
                    f"VoltVideoBatchFormat: input length mismatch — {sources[0][0]} has {ref_count} frames "
                    f"but {name} has {n} frames. They must match."
                )
        current_frames = ref_count
        H, W = sources[0][2], sources[0][3]

        if image is None:
            image = torch.full((current_frames, H, W, 3), grey_value, dtype=torch.float32)
        if mask is None:
            mask = torch.zeros((current_frames, H, W), dtype=torch.float32)

        # Resolve stride
        stride = MODEL_STRIDES[model]
        if stride is None:
            stride = max(1, int(custom_temporal_stride))

        # Resolve pad count
        if mode == "arbitrary_pad":
            pad_count = max(0, int(pad_frames))
        elif mode == "specific_num_frames":
            pad_count = max(0, int(target_num_frames) - current_frames)
        else:  # nearest_compatible
            target = _nearest_valid_num_frames(current_frames, stride, round_up)
            pad_count = max(0, target - current_frames)

        if pad_count <= 0:
            cv_out = control_video if control_video is not None else torch.full((current_frames, H, W, 3), grey_value, dtype=torch.float32)
            return (image, mask, cv_out, 0, 0, current_frames)

        # Decide split
        if pad_position == "end":
            added_start, added_end = 0, pad_count
        elif pad_position == "start":
            added_start, added_end = pad_count, 0
        else:  # both
            added_start = pad_count // 2
            added_end = pad_count - added_start

        def _build_segment(t, count, side, is_mask):
            if count <= 0 or t is None:
                return None
            if is_mask:
                # The mask helper treats "grey_inpaint" specially via an exact string match.
                # Map our generic name to the WAN-side string so behavior is identical.
                pm = "wan_inpaint_grey" if pad_method == "grey_inpaint" else pad_method
                return _pad_mask_like(t, count, pm, side)
            pm = "wan_inpaint_grey" if pad_method == "grey_inpaint" else pad_method
            return _pad_image_like(t, count, pm, side, grey_value)

        pad_img_s = _build_segment(image, added_start, "start", False)
        pad_img_e = _build_segment(image, added_end, "end", False)
        pad_msk_s = _build_segment(mask, added_start, "start", True)
        pad_msk_e = _build_segment(mask, added_end, "end", True)
        pad_ctrl_s = _build_segment(control_video, added_start, "start", False) if control_video is not None else None
        pad_ctrl_e = _build_segment(control_video, added_end, "end", False) if control_video is not None else None

        def _stitch(parts):
            parts = [p for p in parts if p is not None]
            return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        out_image = _stitch([pad_img_s, image, pad_img_e])
        out_mask = _stitch([pad_msk_s, mask, pad_msk_e])
        if control_video is not None:
            out_control = _stitch([pad_ctrl_s, control_video, pad_ctrl_e])
        else:
            total_now = int(out_image.shape[0])
            out_control = torch.full((total_now, H, W, 3), grey_value, dtype=torch.float32)

        total_now = int(out_image.shape[0])
        return (out_image, out_mask, out_control, added_start, added_end, total_now)
