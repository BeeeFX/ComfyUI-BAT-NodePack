"""
Bat SeC Segmenter — SeC video object segmentation with the points editor built in.
=================================================================================

One node instead of three. Replaces the SeC Model Loader + SeC Video
Segmentation + BAT Points Editor chain: click the object on the annotation
frame, get a tracked mask for the whole batch.

Exposed controls are deliberately minimal — the points editor canvas,
``frame_index_select``, ``auto_unload_model`` and the display-only
``mask_preview``. Everything else defaults to
the values in ``bat_sec_advanced.DEFAULTS`` and is only reachable by wiring a
``🦇 SeC Advanced Params`` node into the ``advanced`` input.

The editor's positive points, negative points AND bbox all feed SeC as prompts
(shift+click, shift+right-click, ctrl+drag respectively).

Widget layout note: the seven editor plumbing widgets have to keep their exact
upstream names — ``points_store``, ``coordinates``, ``neg_coordinates``,
``bbox_store``, ``bboxes``, ``width``, ``height`` — because
``web/bat_points_editor/editor_base.js`` looks them up by name. The JS hides
all of them behind the canvas. See ``web/bat_sec_segmenter.js``.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from . import bat_sec_advanced as adv
from . import bat_sec_runtime as rt

# Preview strip pushed back to the editor after a run, so scrubbing
# frame_index_select re-previews without re-running a multi-minute
# segmentation. Same shape and budget as Bat_Roto's strip.
PREVIEW_MAX_FRAMES = 240
PREVIEW_MAX_DIM = 720
PREVIEW_JPEG_QUALITY = 78


class BatSecSegmenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "The clip to segment, as a frame batch.",
                }),

                # ── points editor plumbing (hidden by the JS) ──────────────
                "points_store": ("STRING", {"multiline": False, "advanced": True}),
                "coordinates": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "neg_coordinates": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "bbox_store": ("STRING", {"multiline": False, "advanced": True}),
                "bboxes": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1, "advanced": True}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 1, "advanced": True}),

                # ── the two controls that matter ───────────────────────────
                "frame_index_select": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Which frame you place the points on. The editor preview jumps to "
                               "this frame, and tracking spreads out from it through the rest of "
                               "the clip.",
                }),
                "auto_unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Free the 4B model after each run. Leave on unless you're "
                               "iterating — off keeps it resident so the next run skips the "
                               "~30s load, at the cost of the VRAM.",
                }),
                # Appended rather than slotted next to frame_index_select: ComfyUI
                # stores widget values as a positional array, so inserting in the
                # middle would shift every saved workflow's values by one.
                "mask_preview": (rt.MASK_PREVIEW_MODES, {
                    "default": rt.MASK_PREVIEW_DEFAULT,
                    "tooltip": "How the 'preview' output draws the mask over the clip. Solid "
                               "replaces the masked pixels; the 50% variants let you see the "
                               "image through the tint, which is what you want for checking "
                               "edges. Display only — never affects the 'masks' output.",
                }),
            },
            # Socket order follows this dict (required first, then optional), and
            # every entry above except `frames` renders as a widget rather than a
            # socket — so these two land as sockets 2 and 3 under `frames`.
            "optional": {
                "input_mask": ("MASK", {
                    "tooltip": "Optional mask to seed the object instead of clicking it, applied "
                               "on the selected frame. Points or a bbox override it; positive "
                               "points outside it are dropped.",
                }),
                "advanced": ("SEC_ADVANCED", {
                    "tooltip": "Optional 🦇 SeC Advanced Params node. Unconnected = defaults.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "preview")
    FUNCTION = "segment"
    CATEGORY = "BAT/Segmentation"
    TITLE = "🦇 SeC Segmenter"
    DESCRIPTION = """
## SeC video object segmentation, all in one node

Concept-driven tracking — a vision-language model works out *what* the object
is, so it survives occlusion, motion blur and hard cuts far better than pure
feature matching.

**Shift + click** — add a positive point (the object)
**Shift + right click** — add a negative point (not the object)
**Right click a point** — delete it
**Ctrl + drag** — draw a bounding box
**Right click the bbox** — delete it

The `preview` output composites the tracked mask back over the clip so you can
eyeball the result — set the look with `mask_preview`.

Drop or paste an image onto the node to set the reference plate manually;
otherwise the preview follows `frame_index_select` from the connected clip.

Needs a SeC checkpoint in `ComfyUI/models/sams` — see `bat_sec/NOTICE.md`.
Wire a **🦇 SeC Advanced Params** node into `advanced` for model choice,
device, tracking direction and the rest.
"""

    def _preview_payload(self, frames):
        """Strided, downscaled JPEG strip for the editor canvas.

        Coordinates stay in full-resolution frame space — the true w/h ride
        along so the JS scales between display and image space itself.
        """
        count = int(frames.shape[0])
        stride = max(1, -(-count // PREVIEW_MAX_FRAMES))  # ceil, so the strip never exceeds the cap

        strip = []
        for i in range(0, count, stride):
            arr = (frames[i].detach().cpu().float().clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] > 3:
                arr = arr[..., :3]
            img = Image.fromarray(arr, "RGB")
            if max(img.size) > PREVIEW_MAX_DIM:
                r = PREVIEW_MAX_DIM / max(img.size)
                img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.BILINEAR)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=PREVIEW_JPEG_QUALITY)
            strip.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        return {
            "frames": strip,
            "w": [int(frames.shape[2])],
            "h": [int(frames.shape[1])],
            "stride": [int(stride)],
            "frame_count": [count],
        }

    def segment(self, frames, points_store, coordinates, neg_coordinates, bbox_store,
                bboxes, width, height, frame_index_select, auto_unload_model,
                mask_preview=rt.MASK_PREVIEW_DEFAULT, advanced=None, input_mask=None):
        settings = adv.resolve(advanced)

        model_file = settings["model_file"]
        if not model_file:
            # No advanced node wired (or it reported "no model"): take whatever
            # list_sec_models ranks first, which is the fp16 single file when
            # present. Nothing installed -> "" and get_model auto-downloads it.
            available = rt.list_sec_models()
            model_file = available[0]["name"] if available else ""

        model, cache_key = rt.get_model(
            model_file,
            device=settings["device"],
            use_flash_attn=settings["use_flash_attn"],
            allow_mask_overlap=settings["allow_mask_overlap"],
            auto_download=settings["auto_download"],
        )

        try:
            masks = rt.segment(
                model,
                frames,
                positive_points=coordinates,
                negative_points=neg_coordinates,
                bbox=bboxes,
                input_mask=input_mask,
                # The editor's coord space is its width/height widgets, which
                # track its background plate. If someone loaded a proxy as the
                # plate, the clicks are in proxy space — rescale them onto the
                # real frames instead of segmenting the wrong pixels.
                editor_size=(int(width), int(height)),
                tracking_direction=settings["tracking_direction"],
                # The widget reads 'frame_index_select' for artists; the runtime
                # keeps SeC's own term for the same thing.
                annotation_frame_idx=frame_index_select,
                object_id=settings["object_id"],
                max_frames_to_track=settings["max_frames_to_track"],
                mllm_memory_size=settings["mllm_memory_size"],
                offload_video_to_cpu=settings["offload_video_to_cpu"],
            )
        finally:
            if auto_unload_model:
                rt.release(cache_key)

        return {
            "ui": self._preview_payload(frames),
            "result": (masks, rt.draw_mask_overlay(frames, masks, mask_preview)),
        }
