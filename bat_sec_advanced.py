"""
Bat SeC Advanced Params — optional settings pipe for Bat_SecSegmenter.
=====================================================================

Everything the segmenter node deliberately hides. Leave it out of the graph
and the segmenter uses these same defaults; add it and wire the one output to
the segmenter's ``advanced`` input to override them.

Kept as a separate node rather than widgets-on-a-toggle so one settings node
can drive several segmenters in a shot — and so the segmenter itself stays a
three-control node for the people it's aimed at.
"""

from . import bat_sec_runtime as rt

# The single source of truth for defaults. bat_sec_segmenter imports this and
# applies it when nothing is wired to `advanced`, so the two nodes cannot drift
# apart. Values match the tested SeC setup these nodes were factored out of —
# note `tracking_direction` and `offload_video_to_cpu` differ from upstream
# SecNodes' own defaults ("forward" / False).
DEFAULTS = {
    "model_file": None,           # None -> highest-priority model on disk
    "device": "auto",
    "use_flash_attn": True,
    "allow_mask_overlap": True,
    "tracking_direction": "bidirectional",
    "object_id": 1,
    "max_frames_to_track": -1,
    "mllm_memory_size": 12,
    "offload_video_to_cpu": True,
    "auto_download": True,
}

TRACKING_DIRECTIONS = ["bidirectional", "forward", "backward"]


def resolve(advanced):
    """Merge a wired ``SEC_ADVANCED`` payload over DEFAULTS."""
    settings = dict(DEFAULTS)
    if advanced:
        settings.update({k: v for k, v in advanced.items() if k in DEFAULTS})
    return settings


class BatSecAdvancedParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (rt.model_choices(), {
                    "tooltip": "SeC checkpoint from ComfyUI/models/sams. Each file's "
                               "precision is taken from its name and used as-is.",
                }),
                "device": (rt.device_choices(), {
                    "default": DEFAULTS["device"],
                    "tooltip": "auto = gpu0 when CUDA is available, else CPU. "
                               "CPU inference is forced to float32 and is very slow.",
                }),
                "tracking_direction": (TRACKING_DIRECTIONS, {
                    "default": DEFAULTS["tracking_direction"],
                    "tooltip": "bidirectional tracks both ways from the annotation frame "
                               "(two passes, ~2x the time). forward/backward cover one side "
                               "only — the other side comes back empty.",
                }),
                "object_id": ("INT", {
                    "default": DEFAULTS["object_id"], "min": 1, "max": 9999,
                    "tooltip": "Label for the tracked object. Only matters when you run "
                               "several segmenters and want their masks distinguishable.",
                }),
                "max_frames_to_track": ("INT", {
                    "default": DEFAULTS["max_frames_to_track"], "min": -1, "max": 999999,
                    "tooltip": "-1 = the whole batch. Otherwise this many frames BEYOND the "
                               "annotation frame, per direction; the rest stay empty.",
                }),
                "mllm_memory_size": ("INT", {
                    "default": DEFAULTS["mllm_memory_size"], "min": 1, "max": 20,
                    "tooltip": "Keyframes the vision-language model keeps as concept memory. "
                               "No VRAM cost. The paper used 7; 12 tracks better through cuts.",
                }),
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": DEFAULTS["offload_video_to_cpu"],
                    "tooltip": "Keep decoded frames in system RAM instead of VRAM. Saves a lot "
                               "of VRAM on long clips for roughly 3% more time.",
                }),
                "use_flash_attn": ("BOOLEAN", {
                    "default": DEFAULTS["use_flash_attn"],
                    "tooltip": "Flash Attention 2. Ignored for float32 (no fp32 kernel exists).",
                }),
                "allow_mask_overlap": ("BOOLEAN", {
                    "default": DEFAULTS["allow_mask_overlap"],
                    "tooltip": "Let tracked objects overlap. Turn off to force strictly "
                               "disjoint masks between objects.",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": DEFAULTS["auto_download"],
                    "tooltip": "Fetch the ~7.35GB SeC-4B-fp16 checkpoint from HuggingFace into "
                               "models/sams when none is installed. Turn off on metered or "
                               "air-gapped machines — the node then errors with manual "
                               "download instructions instead.",
                }),
            }
        }

    RETURN_TYPES = ("SEC_ADVANCED",)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = "BAT/Segmentation"
    DESCRIPTION = (
        "Optional advanced settings for the BAT SeC Segmenter. Wire the output into "
        "the segmenter's 'advanced' input. Without it the segmenter uses these same "
        "defaults, so you only need this node when you want to change something."
    )

    def build(self, model_file, device, tracking_direction, object_id,
              max_frames_to_track, mllm_memory_size, offload_video_to_cpu,
              use_flash_attn, allow_mask_overlap, auto_download):
        return ({
            "model_file": None if model_file == rt.NO_MODEL_SENTINEL else model_file,
            "device": device,
            "tracking_direction": tracking_direction,
            "object_id": object_id,
            "max_frames_to_track": max_frames_to_track,
            "mllm_memory_size": mllm_memory_size,
            "offload_video_to_cpu": offload_video_to_cpu,
            "use_flash_attn": use_flash_attn,
            "allow_mask_overlap": allow_mask_overlap,
            "auto_download": auto_download,
        },)
