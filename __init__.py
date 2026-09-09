"""
ComfyUI-BAT-NodePack
====================
Personal node pack — batch / video / VACE / WAN helpers. Authored by
BeeeFX. Not part of the ETC pipeline but kept in the ETC Suite
*About* listing so the nodes are still discoverable.

Historical note: this pack was previously named ``ComfyUI-ETC_Tools``
and exposed nodes under ``Volt_*`` class_type keys. Class names were
renamed to ``Bat_*`` and the menu category bumped to ``BAT/...`` so
the pack reads as a personal project. Old workflows that still
reference the legacy ``Volt_*`` class names trigger ETC_Core's
"Deprecated nodes detected" migration popup on load — see
``ComfyUI-ETC_Core/web/js/etc-node-migration.js`` and
``web/bat-migrations.js`` below.

2026-09-02: ``Bat_VideoBatchFormat`` and ``Bat_WanBatchFrameFormat``
were merged into the single ``Bat_BatchFormat`` ("🦇 Batch Format"),
which covers every model's frame-count grid rather than one each.
Both old keys migrate via the same popup.

2026-09-07: ``Bat_BypassSwitch`` ("🦇 Bypass Switch") joins the pack as its
first *graph-management* node — it has no data path at all and never executes.
It carries named toggles that bypass saved selections of nodes, subgraphs and
backdrops, for making a template someone else can drive. Everything happens in
``web/bat_bypass_switch.js``; the Python is a stub whose only job is to declare
the widget the state serialises into.

2026-09-08: ``web/bat_canvas_zoom.js`` unlocks zooming further out on the graph
canvas. No node and no Python at all — it lowers litegraph's
``canvas.ds.min_scale`` floor from 10% to 5% (adjustable, *Settings → 🦇 BAT →
Canvas*), which is the single field every zoom gesture in the frontend clamps
against.
"""

from .bat_video_grid_split import VideoGridSplit
from .bat_vace_batch import VaceBatchTool
from .bat_wan_context_calculator import VoltWanContextCalculator
from .bat_wan_batch_format import VoltWanBatchFormat, VoltWanBatchCrop
from .bat_batch_format import BatBatchFormat
from .bat_video_loader import VideoLoader
from .bat_frame_picker import BatFramePicker
from .bat_points_editor import BatPointsEditor
from .bat_ref_aligner import RefAligner
from .bat_crop import BatCrop
from .bat_uncrop import BatUncrop
from .bat_animated_crop import BatAnimatedCrop
from .bat_video_combine import BatVideoCombine
from .bat_grade import BatGrade
from .bat_animated_grade import BatAnimatedGrade
from .bat_hdr_tonal_composite import BatHDRTonalComposite
from .bat_advanced_blend import BatAdvancedBlend
from .bat_layered_images import BatLayeredImages
from .bat_rescale import BatRescale
from .bat_exposure_bracket import BatExposureBracket, BatExposureMerge
from .bat_roto import BatRoto
from .bat_filename_prefix import BatFilenamePrefix
from .bat_mask_morph import BatGrowMask, BatErodeMask
from .bat_framehold import BatFramehold
from .bat_sec_segmenter import BatSecSegmenter
from .bat_sec_advanced import BatSecAdvancedParams
from .bat_bypass_switch import BatBypassSwitch

# ─── Execution profiler ──────────────────────────────────────────────
# Not a node: a sidebar panel that instruments *every* node in the graph
# (not just BAT's) with per-node time / RAM / VRAM / payload / disk-I/O,
# so you can see which node is eating the box. It wraps two module-level
# functions in ComfyUI's execution.py; see bat_profiler.py for why those
# two and what the measurements do and do not mean. It is on by default
# so an unattended OOM is captured without having to reproduce it, and
# every probe is failure-isolated — a profiler fault disables profiling,
# it never fails a prompt.
from . import bat_profiler

bat_profiler.install()

# class_type keys — bumped from Volt_* to Bat_* with the rename. The
# in-UI migration tool (ETC_Core) detects the old keys on workflow
# load and offers a one-click replacement that preserves position,
# parameters, and links.
NODE_CLASS_MAPPINGS = {
    "Bat_VideoGridSplit":       VideoGridSplit,
    "Bat_VaceBatchTool":        VaceBatchTool,
    "Bat_WanContextCalculator": VoltWanContextCalculator,
    "Bat_WanBatchFormat":       VoltWanBatchFormat,
    "Bat_WanBatchCrop":         VoltWanBatchCrop,
    "Bat_BatchFormat":          BatBatchFormat,
    "Bat_VideoLoader":          VideoLoader,
    "Bat_FramePicker":          BatFramePicker,
    "Bat_PointsEditor":         BatPointsEditor,
    "Bat_RefAligner":           RefAligner,
    "Bat_Crop":                 BatCrop,
    "Bat_AnimatedCrop":         BatAnimatedCrop,
    "Bat_Uncrop":               BatUncrop,
    "Bat_VideoCombine":         BatVideoCombine,
    "Bat_Grade":                BatGrade,
    "Bat_AnimatedGrade":        BatAnimatedGrade,
    "Bat_HDRTonalComposite":    BatHDRTonalComposite,
    "Bat_AdvancedBlend":        BatAdvancedBlend,
    "Bat_LayeredImages":        BatLayeredImages,
    "Bat_Rescale":              BatRescale,
    "Bat_ExposureBracket":      BatExposureBracket,
    "Bat_ExposureMerge":        BatExposureMerge,
    "Bat_Roto":                 BatRoto,
    "Bat_FilenamePrefix":       BatFilenamePrefix,
    "Bat_GrowMask":             BatGrowMask,
    "Bat_ErodeMask":            BatErodeMask,
    "Bat_Framehold":            BatFramehold,
    "Bat_SecSegmenter":         BatSecSegmenter,
    "Bat_SecAdvancedParams":    BatSecAdvancedParams,
    "Bat_BypassSwitch":         BatBypassSwitch,
}

# Display names — bat emoji prefix so the nodes stand out as
# personal-pack helpers in the right-click "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Bat_VideoGridSplit":       "🦇 Video Grid Split",
    "Bat_VaceBatchTool":        "🦇 VACE Batch Tool",
    "Bat_WanContextCalculator": "🦇 WAN Context Calculator",
    "Bat_WanBatchFormat":       "🦇 WAN Batch Format",
    "Bat_WanBatchCrop":         "🦇 Batch Crop",
    "Bat_BatchFormat":          "🦇 Batch Format",
    "Bat_VideoLoader":          "🦇 Video Loader",
    "Bat_FramePicker":          "🦇 BAT Frame Picker",
    "Bat_PointsEditor":         "🦇 Points Editor",
    "Bat_RefAligner":           "🦇 Wan Reference Aligner",
    "Bat_Crop":                 "🦇 Crop",
    "Bat_AnimatedCrop":         "🦇 Animated Crop",
    "Bat_Uncrop":               "🦇 Uncrop",
    "Bat_VideoCombine":         "🦇 Video Combine",
    "Bat_Grade":                "🦇 Grade",
    "Bat_AnimatedGrade":        "🦇 Animated Grade",
    "Bat_HDRTonalComposite":    "🦇 HDR Tonal Composite",
    "Bat_AdvancedBlend":        "🦇 Advanced Blend",
    "Bat_LayeredImages":        "🦇 Layered Images",
    "Bat_Rescale":              "🦇 Rescale",
    "Bat_ExposureBracket":      "🦇 Exposure Bracket",
    "Bat_ExposureMerge":        "🦇 Exposure Merge",
    "Bat_Roto":                 "🦇 Roto",
    "Bat_FilenamePrefix":       "🦇 Filename Prefix",
    "Bat_GrowMask":             "🦇 Grow Mask",
    "Bat_ErodeMask":            "🦇 Erode Mask",
    "Bat_Framehold":            "🦇 Framehold",
    "Bat_SecSegmenter":         "🦇 SeC Segmenter",
    "Bat_SecAdvancedParams":    "🦇 SeC Advanced Params",
    "Bat_BypassSwitch":         "🦇 Bypass Switch",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
