"""
Bat Batch Format — pad an image / mask / control_video batch to a frame
count the target video model will actually accept.

Supersedes two nodes that had drifted into near-duplicates:

  * ``bat_video_batch_format.py`` (Bat_VideoBatchFormat) — had the model
    dropdown but hardcoded a ``stride*k + 1`` grid and built its padding
    tensors on CPU/float32, which crashed on a CUDA ``control_video``.
  * ``bat_wan_frame_format.py`` (Bat_WanBatchFrameFormat) — had the
    device/dtype fix but a WAN-only 4n+1 grid.

Both are migrated into this node by ``web/bat-migrations.js``.

The generalisation that made one node possible: valid lengths are
``stride*k + offset``, not ``stride*k + 1``. WAN / Hunyuan / LTX / Cosmos
all sit at offset 1, but MiniMax H3's grid is 17k + 5 (5, 22, 39, 56, ...
124) — see ``comfy_extras/nodes_minimax_h3.py::align_frame_count``. A
node that assumes +1 silently produces lengths H3 will reject.

Sliding-context WAN work still belongs in bat_wan_batch_format.py's
VoltWanBatchFormat, whose "nearest compatible" additionally snaps to a
whole number of context windows. This node deliberately knows nothing
about windowing.
"""

import torch

from .bat_wan_batch_format import (
    _normalize_mask,
    _pad_image_like,
    _pad_mask_like,
)


# name -> (temporal stride, offset, minimum valid length)
#
# Adding a model is a single line here — deliberately no "Custom" escape
# hatch, since a hand-typed stride is a silent-wrong-output trap and every
# real model's grid belongs in this table where it can be checked.
#
# Valid frame counts F satisfy (F - offset) % stride == 0 and F >= minimum.
# Strides come from each model's causal video VAE; verified against the
# latent-shape maths in comfy_extras (nodes_lt.py, nodes_hunyuan.py,
# nodes_cosmos.py, nodes_minimax_h3.py).
MODEL_SPECS = {
    "WAN (4k+1)":        (4, 1, 1),
    "Hunyuan (4k+1)":    (4, 1, 1),
    "LTX 2.3 (8k+1)":    (8, 1, 1),
    "Cosmos (8k+1)":     (8, 1, 1),
    "MiniMax H3 (17k+5)": (17, 5, 5),
}

DEFAULT_MODEL = "WAN (4k+1)"


def _nearest_valid_num_frames(current_frames, stride, offset, minimum, round_up):
    """Closest length on the model's stride*k + offset grid.

    round_up=True picks the smallest valid length >= current_frames, so no
    input frame is ever left needing a crop. round_up=False picks whichever
    of the two neighbouring valid lengths is closer, ties going downward.
    """
    stride = max(1, int(stride))
    offset = max(0, int(offset))
    minimum = max(1, int(minimum))

    if current_frames <= minimum:
        return minimum

    # k such that stride*k + offset is the largest valid length <= current.
    k = (current_frames - offset) // stride
    lower = stride * k + offset
    if lower < minimum:
        # current_frames sits above `minimum` but below the first grid point
        # at or above it — e.g. H3 with 3 frames. Only the ceiling is legal.
        lower = minimum
    upper = lower if lower >= current_frames else lower + stride

    if round_up:
        # `upper` is >= current_frames by construction, so there is no case
        # left where we would need to add another whole stride.
        return upper

    if lower < minimum:
        return upper
    return lower if (current_frames - lower) <= (upper - current_frames) else upper


class BatBatchFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_SPECS.keys()), {
                    "default": DEFAULT_MODEL,
                    "tooltip": "Target video model — picks the frame-count grid. Most models want stride*k+1; MiniMax H3 wants 17k+5. Only used when mode=nearest_compatible.",
                }),
                "mode": (["nearest_compatible", "specific_num_frames", "arbitrary_pad"], {
                    "default": "nearest_compatible",
                    "tooltip": "nearest_compatible = pad to the closest length the selected model accepts. specific_num_frames = pad to exactly target_num_frames. arbitrary_pad = ignore the grid, just add pad_frames at the chosen position.",
                }),
                "pad_method": (["repeat_edge", "grey_inpaint"], {
                    "default": "repeat_edge",
                    "tooltip": "repeat_edge = duplicate the first or last real frame. grey_inpaint = grey RGB + white mask (mask=1) so the model can fill in.",
                }),
                "pad_position": (["start", "end", "both"], {
                    "default": "start",
                    "tooltip": "Where to add the padded frames. 'both' splits evenly (extra frame goes to end if odd).",
                }),
                "auto_target_frames": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Used when mode=specific_num_frames. On = read the length from the connected inputs and snap it to the selected model's grid (the target_num_frames box hides). Off = use the target_num_frames you type."}),
                "target_num_frames": ("INT", {"default": 81, "min": 1, "max": 100000, "step": 1,
                                              "tooltip": "Used when mode=specific_num_frames and auto_target_frames is off. Never truncates if shorter than input — crop afterwards if you need exact trimming."}),
                "pad_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                       "tooltip": "Used when mode=arbitrary_pad. Number of frames to add at the chosen position (preroll if start, postroll if end)."}),
                "round_up": ("BOOLEAN", {"default": True,
                                         "tooltip": "Used when mode=nearest_compatible. Pick the smallest valid length >= input length so no input frames get cropped."}),
                "grey_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Grey level for grey_inpaint padded RGB."}),
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
    CATEGORY = "BAT/Video"
    DESCRIPTION = "Pad an image+mask (+optional control_video) batch to a frame count the target video model accepts (WAN/Hunyuan 4k+1, LTX/Cosmos 8k+1, MiniMax H3 17k+5), or to a specific/arbitrary count. Returns the padding counts so BAT WAN Batch Crop can trim back after generation."

    def format(self, model, mode, pad_method, pad_position,
               target_num_frames, pad_frames, round_up, grey_value,
               auto_target_frames=False,
               image=None, mask=None, control_video=None):
        # Retro-compat: workflows migrated from the two predecessor nodes may
        # still send their old enum values if the JS migration didn't run
        # (headless API calls, farm submissions of an un-migrated workflow).
        # IS_CHANGED can't see these, so normalise them here rather than
        # relying on the frontend alone.
        if mode in ("nearest_wan_compatible", "nearest_ltx23_compatible"):
            model = "WAN (4k+1)" if mode == "nearest_wan_compatible" else "LTX 2.3 (8k+1)"
            mode = "nearest_compatible"
        if pad_method == "wan_inpaint_grey":
            pad_method = "grey_inpaint"
        if model not in MODEL_SPECS:
            model = DEFAULT_MODEL

        mask = _normalize_mask(mask)

        sources = []
        if image is not None:
            sources.append(("image", int(image.shape[0]), int(image.shape[1]), int(image.shape[2])))
        if control_video is not None:
            sources.append(("control_video", int(control_video.shape[0]), int(control_video.shape[1]), int(control_video.shape[2])))
        if mask is not None:
            sources.append(("mask", int(mask.shape[0]), int(mask.shape[-2]), int(mask.shape[-1])))

        if not sources:
            raise ValueError("BatBatchFormat: provide at least one of image, mask, or control_video.")

        ref_count = sources[0][1]
        for name, n, _, _ in sources[1:]:
            if n != ref_count:
                raise ValueError(
                    f"BatBatchFormat: input length mismatch — {sources[0][0]} has {ref_count} frames "
                    f"but {name} has {n} frames. They must match."
                )

        current_frames = ref_count
        H, W = sources[0][2], sources[0][3]

        # Manufacture the missing companion so we always output both.
        #
        # Inherit device AND dtype from whichever input we do have. Both
        # predecessor nodes hardcoded CPU/float32 here, so a CUDA
        # control_video hit "Expected all tensors to be on the same device"
        # on the torch.cat below.
        _ref = None
        for _cand in (image, mask, control_video):
            if _cand is not None and hasattr(_cand, "device"):
                _ref = _cand
                break
        _dev = _ref.device if _ref is not None else None
        _dt = _ref.dtype if _ref is not None else torch.float32
        if image is None:
            image = torch.full((current_frames, H, W, 3), grey_value, dtype=_dt, device=_dev)
        if mask is None:
            mask = torch.zeros((current_frames, H, W), dtype=_dt, device=_dev)

        # Resolve the target grid.
        stride, offset, minimum = MODEL_SPECS[model]

        # Resolve pad count.
        if mode == "arbitrary_pad":
            pad_count = max(0, int(pad_frames))
        elif mode == "specific_num_frames":
            if auto_target_frames:
                target = _nearest_valid_num_frames(current_frames, stride, offset, minimum, round_up)
            else:
                target = int(target_num_frames)
            pad_count = max(0, target - current_frames)
        else:  # nearest_compatible
            target = _nearest_valid_num_frames(current_frames, stride, offset, minimum, round_up)
            pad_count = max(0, target - current_frames)

        if pad_count <= 0:
            cv_out = (control_video if control_video is not None
                      else torch.full((current_frames, H, W, 3), grey_value,
                                      dtype=_dt, device=_dev))
            return (image, mask, cv_out, 0, 0, current_frames)

        # Decide split between start and end.
        if pad_position == "end":
            added_start, added_end = 0, pad_count
        elif pad_position == "start":
            added_start, added_end = pad_count, 0
        else:  # both — even split, odd extra goes to the end
            added_start = pad_count // 2
            added_end = pad_count - added_start

        # The shared helpers match "wan_inpaint_grey" by exact string, so
        # translate our generic name back at the call site.
        _pm = "wan_inpaint_grey" if pad_method == "grey_inpaint" else pad_method

        def _build_segment(t, count, side, is_mask):
            if count <= 0 or t is None:
                return None
            if is_mask:
                return _pad_mask_like(t, count, _pm, side)
            return _pad_image_like(t, count, _pm, side, grey_value)

        pad_img_start = _build_segment(image, added_start, "start", False)
        pad_img_end = _build_segment(image, added_end, "end", False)
        pad_msk_start = _build_segment(mask, added_start, "start", True)
        pad_msk_end = _build_segment(mask, added_end, "end", True)
        pad_ctrl_start = _build_segment(control_video, added_start, "start", False)
        pad_ctrl_end = _build_segment(control_video, added_end, "end", False)

        def _stitch(parts):
            parts = [p for p in parts if p is not None]
            return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

        out_image = _stitch([pad_img_start, image, pad_img_end])
        out_mask = _stitch([pad_msk_start, mask, pad_msk_end])
        total_now = int(out_image.shape[0])

        if control_video is not None:
            out_control = _stitch([pad_ctrl_start, control_video, pad_ctrl_end])
        else:
            out_control = torch.full((total_now, H, W, 3), grey_value, dtype=_dt, device=_dev)

        return (out_image, out_mask, out_control, added_start, added_end, total_now)
