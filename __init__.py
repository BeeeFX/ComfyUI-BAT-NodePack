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

2026-09-15: ``Bat_Loader`` ("🦇 Loader") joins the pack as a unified media
loader — one ``path`` field that takes a still, a ``####`` frame sequence, a
movie or a folder of images, and returns multi-layer EXR layers, cryptomattes
and header metadata alongside the frames. It sits BESIDE 🦇 Video Loader rather
than replacing it: that node stays the one for interactively trimming a movie
with a scrubber, and no existing workflow changes. Its movie branch imports
``bat_video_loader``'s decode stack rather than copying it. Two companion nodes
(``Bat_ExrLayer``, ``Bat_CryptomatteMatte``) consume the layer outputs.

2026-09-15: ``web/bat_fullscreen.js`` adds a ⛶ maximise button to the
advanced editors (Roto, Animated Crop, Animated Grade, Layered Images, HDR
Tonal Composite, Rescale). It moves the live editor root into a full-window
overlay rather than building a second copy, so the fullscreen view IS the node's
editor. See that file's header for why ``widget.hidden`` is the load-bearing
part.

2026-09-17: ``Bat_VideoCombine`` joins them, driven from the ⛶ already in its
transport bar rather than a second button (``addBatFullscreen(..., {button:
false})``). It previously called ``videoEl.requestFullscreen()``, which
promotes the ``<video>`` alone and discards the transport, scrub bar, loop
pins, hover thumbnails and every keyboard shortcut.

2026-09-17: ten small utility nodes join the pack so two third-party packs can
be retired from the studio installs — ``ComfyUI-Easy-Use`` (runtime ``pip
install`` into the shared venv; an arbitrary-file-write in its ``easy saveText``
before v1.4.1) and ``comfyui-art-venture`` (same runtime-pip behaviour). Only
``easy showAnything`` and ``StringToInt`` were actually used in studio
workflows; the rest of these cover the type/logic gaps that core ComfyUI still
does not fill, so nobody reaches for those packs again. Where core DOES cover
it, the migration popup points at the *core* node rather than a BAT one —
``StringToInt`` goes to ``ComfyNumberConvert``, the primitives go to
``Primitive*`` — because the cheapest node to maintain is the one we do not
own. See ``web/bat-migrations.js``.

2026-09-08: ``web/bat_canvas_zoom.js`` unlocks zooming further out on the graph
canvas. No node and no Python at all — it lowers litegraph's
``canvas.ds.min_scale`` floor from 10% to 5% (adjustable, *Settings → 🦇 BAT →
Canvas*), which is the single field every zoom gesture in the frontend clamps
against.

2026-09-23: node modules are imported one by one through ``_guarded`` — a
module that fails to import now costs only its own nodes, not the whole pack.
``bat_prompt_hooks`` stamps the four nodes that skip unwired outputs (Loader,
Video Loader, Advanced Blend, HDR Tonal Composite) with what is wired, so the
output cache can no longer hand a newly connected output a stub from a run that
skipped it.
"""

import importlib
import logging

_log = logging.getLogger("BAT")


def _guarded(module, *names):
    """Import `names` from one of the pack's modules, or None for each if it fails.

    One module that cannot import — a missing optional dependency, a syntax slip
    in a file being edited on the shared install — used to take the whole pack
    down: ComfyUI marks the package IMPORT FAILED and every node (and the
    profiler hooks) vanish, so every workflow using any BAT node breaks at once.
    Now only that module's nodes are missing, and the log names the reason.
    """
    try:
        mod = importlib.import_module(f".{module}", __name__)
        return tuple(getattr(mod, n) for n in names)
    except Exception:
        _log.exception("[BAT] %s failed to import; its nodes (%s) are unavailable",
                       module, ", ".join(names))
        return (None,) * len(names)


(VideoGridSplit,) = _guarded("bat_video_grid_split", "VideoGridSplit")
(BatGridMerge,) = _guarded("bat_grid_merge", "BatGridMerge")
(VaceBatchTool,) = _guarded("bat_vace_batch", "VaceBatchTool")
(VoltWanContextCalculator,) = _guarded("bat_wan_context_calculator", "VoltWanContextCalculator")
VoltWanBatchFormat, VoltWanBatchCrop = _guarded(
    "bat_wan_batch_format", "VoltWanBatchFormat", "VoltWanBatchCrop")
(BatBatchFormat,) = _guarded("bat_batch_format", "BatBatchFormat")
(VideoLoader,) = _guarded("bat_video_loader", "VideoLoader")
(BatLoader,) = _guarded("bat_loader", "BatLoader")
BatExrLayer, BatCryptomatteMatte = _guarded(
    "bat_exr_layers", "BatExrLayer", "BatCryptomatteMatte")
(BatFramePicker,) = _guarded("bat_frame_picker", "BatFramePicker")
(BatPointsEditor,) = _guarded("bat_points_editor", "BatPointsEditor")
(RefAligner,) = _guarded("bat_ref_aligner", "RefAligner")
(BatCrop,) = _guarded("bat_crop", "BatCrop")
(BatUncrop,) = _guarded("bat_uncrop", "BatUncrop")
(BatAnimatedCrop,) = _guarded("bat_animated_crop", "BatAnimatedCrop")
(BatVideoCombine,) = _guarded("bat_video_combine", "BatVideoCombine")
(BatGrade,) = _guarded("bat_grade", "BatGrade")
(BatAnimatedGrade,) = _guarded("bat_animated_grade", "BatAnimatedGrade")
(BatHDRTonalComposite,) = _guarded("bat_hdr_tonal_composite", "BatHDRTonalComposite")
(BatAdvancedBlend,) = _guarded("bat_advanced_blend", "BatAdvancedBlend")
(BatLayeredImages,) = _guarded("bat_layered_images", "BatLayeredImages")
(BatRescale,) = _guarded("bat_rescale", "BatRescale")
BatExposureBracket, BatExposureMerge = _guarded(
    "bat_exposure_bracket", "BatExposureBracket", "BatExposureMerge")
(BatRoto,) = _guarded("bat_roto", "BatRoto")
(BatFilenamePrefix,) = _guarded("bat_filename_prefix", "BatFilenamePrefix")
BatGrowMask, BatErodeMask = _guarded("bat_mask_morph", "BatGrowMask", "BatErodeMask")
(BatFramehold,) = _guarded("bat_framehold", "BatFramehold")
(BatSecSegmenter,) = _guarded("bat_sec_segmenter", "BatSecSegmenter")
(BatSecAdvancedParams,) = _guarded("bat_sec_advanced", "BatSecAdvancedParams")
(BatBypassSwitch,) = _guarded("bat_bypass_switch", "BatBypassSwitch")
BatShowAny, BatShowTensorShape = _guarded("bat_show", "BatShowAny", "BatShowTensorShape")
BatConvertAny, BatAnyToString, BatNumberToString = _guarded(
    "bat_convert", "BatConvertAny", "BatAnyToString", "BatNumberToString")
BatCompare, BatIndexSwitch, BatListLength, BatListIndex, BatListBatch, BatListLengthList = _guarded(
    "bat_logic", "BatCompare", "BatIndexSwitch", "BatListLength", "BatListIndex",
    "BatListBatch", "BatListLengthList")

# ─── Execution profiler ──────────────────────────────────────────────
# Not a node: a sidebar panel that instruments *every* node in the graph
# (not just BAT's) with per-node time / RAM / VRAM / payload / disk-I/O,
# so you can see which node is eating the box. It wraps two module-level
# functions in ComfyUI's execution.py; see bat_profiler.py for why those
# two and what the measurements do and do not mean. The hooks are always
# installed but idle — profiling is off until a browser arms it from the
# panel, and every probe is failure-isolated: a profiler fault disables
# profiling, it never fails a prompt.
try:
    from . import bat_profiler

    bat_profiler.install()
except Exception:
    _log.exception("[BAT] the execution profiler could not be installed")

# ─── Output pruning vs the cache ─────────────────────────────────────
# Stamps each node that skips unwired outputs with what is wired, so the
# output cache cannot hand a newly wired consumer a stub from a run where
# that output was skipped. See bat_prompt_hooks.py.
try:
    from . import bat_prompt_hooks

    bat_prompt_hooks.install()
except Exception:
    _log.exception("[BAT] the output-pruning prompt hook could not be installed")

# ─── Preview payloads out of the history ─────────────────────────────
# The editors' multi-MB `ui` previews are written to a temp sidecar and sent
# as a token (web/bat_ui_ref.js resolves it), because ComfyUI keeps every
# executed `ui` in its prompt history. Imported here so /bat/ui/<token> is
# registered even before any node module pulls it in. See bat_ui_ref.py.
try:
    from . import bat_ui_ref  # noqa: F401
except Exception:
    _log.exception("[BAT] the preview sidecar route could not be registered")

# class_type keys — bumped from Volt_* to Bat_* with the rename. The
# in-UI migration tool (ETC_Core) detects the old keys on workflow
# load and offers a one-click replacement that preserves position,
# parameters, and links.
NODE_CLASS_MAPPINGS = {
    "Bat_VideoGridSplit":       VideoGridSplit,
    "Bat_GridMerge":            BatGridMerge,
    "Bat_VaceBatchTool":        VaceBatchTool,
    "Bat_WanContextCalculator": VoltWanContextCalculator,
    "Bat_WanBatchFormat":       VoltWanBatchFormat,
    "Bat_WanBatchCrop":         VoltWanBatchCrop,
    "Bat_BatchFormat":          BatBatchFormat,
    "Bat_VideoLoader":          VideoLoader,
    "Bat_Loader":               BatLoader,
    "Bat_ExrLayer":             BatExrLayer,
    "Bat_CryptomatteMatte":     BatCryptomatteMatte,
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

    # ─── Utility nodes (2026-09-17) ──────────────────────────────────
    "Bat_ShowAny":              BatShowAny,
    "Bat_ShowTensorShape":      BatShowTensorShape,
    "Bat_ConvertAny":           BatConvertAny,
    "Bat_AnyToString":          BatAnyToString,
    "Bat_NumberToString":       BatNumberToString,
    "Bat_Compare":              BatCompare,
    "Bat_IndexSwitch":          BatIndexSwitch,
    "Bat_ListLength":           BatListLength,
    "Bat_ListIndex":            BatListIndex,
    "Bat_ListBatch":            BatListBatch,
    "Bat_ListLengthList":       BatListLengthList,
}

# Display names — bat emoji prefix so the nodes stand out as
# personal-pack helpers in the right-click "Add Node" menu.
NODE_DISPLAY_NAME_MAPPINGS = {
    "Bat_VideoGridSplit":       "🦇 Video Grid Split",
    "Bat_GridMerge":            "🦇 Video Grid Merge",
    "Bat_VaceBatchTool":        "🦇 VACE Batch Tool",
    "Bat_WanContextCalculator": "🦇 WAN Context Calculator",
    "Bat_WanBatchFormat":       "🦇 WAN Batch Format",
    "Bat_WanBatchCrop":         "🦇 Batch Crop",
    "Bat_BatchFormat":          "🦇 Batch Format",
    "Bat_VideoLoader":          "🦇 Video Loader",
    "Bat_Loader":               "🦇 Loader",
    "Bat_ExrLayer":             "🦇 EXR Layer",
    "Bat_CryptomatteMatte":     "🦇 Cryptomatte Matte",
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

    "Bat_ShowAny":              "🦇 Show Any",
    "Bat_ShowTensorShape":      "🦇 Show Tensor Shape",
    "Bat_ConvertAny":           "🦇 Convert Any",
    "Bat_AnyToString":          "🦇 Any to String",
    "Bat_NumberToString":       "🦇 Number to String",
    "Bat_Compare":              "🦇 Compare",
    "Bat_IndexSwitch":          "🦇 Index Switch",
    "Bat_ListLength":           "🦇 List Length",
    "Bat_ListIndex":            "🦇 List Index",
    "Bat_ListBatch":            "🦇 List Batch",
    "Bat_ListLengthList":       "🦇 List Length (list)",
}

# Drop whatever failed to import (see _guarded), so ComfyUI never sees a None
# class — and the display names follow, or the menu lists a node it can't place.
NODE_CLASS_MAPPINGS = {k: v for k, v in NODE_CLASS_MAPPINGS.items() if v is not None}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for k, v in NODE_DISPLAY_NAME_MAPPINGS.items()
                              if k in NODE_CLASS_MAPPINGS}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
