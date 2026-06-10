"""
ComfyUI-BAT-NodePack
====================
Personal node pack — batch / video / VACE / WAN helpers. Authored by
Baptiste. Not part of the ETC pipeline but kept in the ETC Suite
*About* listing so the nodes are still discoverable.

Historical note: this pack was previously named ``ComfyUI-ETC_Tools``
and exposed nodes under ``Volt_*`` class_type keys. Class names were
renamed to ``Bat_*`` and the menu category bumped to ``BAT/...`` so
the pack reads as a personal project. Old workflows that still
reference the legacy ``Volt_*`` class names trigger ETC_Core's
"Deprecated nodes detected" migration popup on load — see
``ComfyUI-ETC_Core/web/js/etc-node-migration.js`` and
``web/bat-migrations.js`` below.
"""

from .bat_video_grid_split import VideoGridSplit
from .bat_vace_batch import VaceBatchTool
from .bat_wan_context_calculator import VoltWanContextCalculator
from .bat_wan_batch_format import VoltWanBatchFormat, VoltWanBatchCrop
from .bat_video_batch_format import VoltVideoBatchFormat
from .bat_video_loader import VideoLoader
from .bat_points_editor import BatPointsEditor
from .bat_wan_frame_format import BatWanBatchFrameFormat
from .bat_vri_picker import VriPicker
from .bat_ref_aligner import RefAligner
from .bat_crop import BatCrop
from .bat_uncrop import BatUncrop
from .bat_video_combine import BatVideoCombine

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
    "Bat_VideoBatchFormat":     VoltVideoBatchFormat,
    "Bat_VideoLoader":          VideoLoader,
    "Bat_PointsEditor":         BatPointsEditor,
    "Bat_WanBatchFrameFormat":  BatWanBatchFrameFormat,
    "Bat_VriPicker":            VriPicker,
    "Bat_RefAligner":           RefAligner,
    "Bat_Crop":                 BatCrop,
    "Bat_Uncrop":               BatUncrop,
    "Bat_VideoCombine":         BatVideoCombine,
}

# Display names — bat emoji prefix so the nodes stand out as
# personal-pack helpers in the right-click "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Bat_VideoGridSplit":       "🦇 Video Grid Split",
    "Bat_VaceBatchTool":        "🦇 VACE Batch Tool",
    "Bat_WanContextCalculator": "🦇 WAN Context Calculator",
    "Bat_WanBatchFormat":       "🦇 WAN Batch Format",
    "Bat_WanBatchCrop":         "🦇 WAN Batch Crop",
    "Bat_VideoBatchFormat":     "🦇 Video Batch Format",
    "Bat_VideoLoader":          "🦇 Video Loader",
    "Bat_PointsEditor":         "🦇 Points Editor",
    "Bat_WanBatchFrameFormat":  "🦇 WAN Batch Frame Format",
    "Bat_VriPicker":            "🦇 VRI Frame Picker",
    "Bat_RefAligner":           "🦇 Wan Reference Aligner",
    "Bat_Crop":                 "🦇 Crop",
    "Bat_Uncrop":               "🦇 Uncrop",
    "Bat_VideoCombine":         "🦇 Video Combine",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
