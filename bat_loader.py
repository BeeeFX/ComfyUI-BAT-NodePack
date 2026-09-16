"""
Bat_Loader — one node for every kind of plate a comp graph starts from.

Type a path into the single `path` field and the node works out what it is:

    /jobs/show/plate.exr            a single frame
    /jobs/show/plate_####.exr       a frame sequence (the #'s are the padding)
    /jobs/show/render_####.png      ...of anything Pillow reads, not just EXR
    /jobs/show/edit.mov             a movie (frames + audio)
    /jobs/show/frames/              a directory of stills

Why one node rather than four: the thing an artist changes between takes is the
PATH, not the loader. Swapping a plate for a render, or an EXR sequence for the
mov the editor sent, otherwise means deleting a node and rewiring every output.

What it gives you beyond a plain image load
-------------------------------------------
* **Multi-layer EXR.** A render is not one RGBA image — it is a flat list of
  named channels that bat_exr groups back into layers (diffuse, N, depth,
  cryptomatte…). Those come out on `layers` and `cryptomatte`, with `layer
  names` listing what was found and `metadata` carrying the full header.
* **Overscan.** EXRs carry a data window (the bbox pixels were written for) and
  a display window (the format). A Nuke render with overscan has a bigger data
  window, and a bbox that MOVES between frames makes a sequence unbatchable.
  The `overscan` widget decides what to do about that — see _plan_overscan.
* **High bit depth, alpha and audio on movies**, by reusing the decode stack
  🦇 Video Loader already has (10/12-bit ProRes keeps its precision; a real
  alpha channel lands on `mask`).

What it deliberately does NOT do
--------------------------------
Trimming with a scrubber. 🦇 Video Loader stays the node for "give me frames
40-120 of this mov, and show me both ends while I pick them"; this one takes a
frame range the way a sequence does, with skip / every-nth / cap.

Cost control: decoding every layer of a comp EXR is the most expensive thing
here, and on a plain plate load nothing reads them. Both the layer outputs and
the audio are built only when something in the prompt is actually wired to
them — see _consumes_slots.
"""

import concurrent.futures
import json
import logging
import os
import threading

import numpy as np
import torch
from PIL import Image

from .bat_exr import OIIO_AVAILABLE, ExrProcessor
from .bat_preview import generate_preview_for_comfyui
from .bat_sequence import SequenceHandler

import server

# The movie branch is 🦇 Video Loader's decode stack, imported rather than
# copied. That node is unchanged and stays the one for interactive trimming;
# this is only about not keeping two copies of the decord/OpenCV/ffmpeg
# selection, the high-bit-depth and alpha handling, and the frame cache — which
# would drift the first time either was fixed.
from .bat_video_loader import (
    VIDEO_EXTENSIONS,
    _output_is_consumed,
    _strip_path,
    load_audio,
    load_batch,
    probe_ffmpeg,
    probe_video,
)

logger = logging.getLogger(__name__)

BIGMAX = 2 ** 31 - 1

# Output slots, in RETURN_NAMES order:
#   0 images  1 mask  2 audio  3 layers  4 layer names
#   5 cryptomatte  6 metadata  7 resolved_path  8 frame_rate
_AUDIO_SLOT = 2
# Everything built out of the EXR's extra channels. If nothing in the prompt
# reads one of these, the file is read for its beauty channels alone.
_LAYER_OUTPUT_SLOTS = (3, 4, 5, 6)

# Stills Pillow will open for the non-EXR paths. Deliberately not "everything
# Pillow supports": a directory load globs on this list, and sweeping up .psd
# or .pdf files that happen to sit beside the frames is worse than missing them.
IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "tga", "exr")

# Frame rate reported for a still sequence. A sequence of files carries no rate
# at all — the number lives in the edit, not on disk — but `frame_rate` has to
# be something an encoder can accept, and 0.0 is not. 24 matches the same
# fallback in 🦇 Video Loader so the two nodes agree; override it downstream if
# the show is 25 or 30.
SEQUENCE_FPS = 24.0

# Values of the `overscan` widget. See _plan_overscan.
OVERSCAN_MODES = ("auto", "crop", "keep")


def _consumes_slots(prompt, unique_id, slots):
    """Whether anything in this prompt reads one of our output `slots`.

    An input link in a ComfyUI prompt is ``[source_node_id, source_slot]``, so a
    link from our slot 5 is ``[our_id, 5]``.

    Returns True whenever we can't tell — no prompt (an API submit that didn't
    send one, or a test), an unparseable prompt, our own id missing. Building
    the layers needlessly costs RAM; skipping them when something wanted them
    hands that consumer an empty dict, so uncertainty falls the expensive way.
    """
    if not isinstance(prompt, dict) or unique_id is None:
        return True
    me = str(unique_id)
    if me not in {str(k) for k in prompt}:
        return True
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and str(value[0]) == me and value[1] in slots):
                return True
    return False


def _frame_slot(res, path):
    """The tensor at `path` inside one frame's process_exr_data result.

    `res` is that function's 7-tuple; a path is ("rgb",) / ("alpha",) or
    ("layers"|"crypto", key) for a plain layer, plus a third element for one
    part of a `*_layer_group` list. Returns None when the frame doesn't carry
    that slot, which is how a ragged sequence is detected rather than crashed
    on.
    """
    kind = path[0]
    if kind == "rgb":
        return res[0]
    if kind == "alpha":
        return res[1]
    value = (res[2] if kind == "crypto" else res[3]).get(path[1])
    if len(path) == 3:
        if not isinstance(value, (list, tuple)) or path[2] >= len(value):
            return None
        value = value[path[2]]
    return value if isinstance(value, torch.Tensor) else None


class _SequenceBatcher:
    """Collect per-frame EXR results into preallocated [N, ...] batches.

    This replaces a `torch.cat` per output, and answers two separate RAM
    problems:

      * `torch.cat` allocates the whole batch while every per-frame tensor is
        still alive, so the peak is exactly 2x the payload — a 100-frame 2K
        RGBA load needing 3.5 GB peaks at 7.1 GB. Frames are copied into their
        slot and released as they come off the thread pool instead, so the
        overhead is the handful of decodes in flight rather than the whole clip
        a second time.

      * `process_exr_data` returns the SAME tensor object under more than one
        name. Every non-crypto layer appears in both the `layers` dict and the
        combined `cryptomatte` dict, and a `*_layer_group` entry is a list of
        tensors already present under their own keys. Batching each name on its
        own turns every alias into an independent full-length copy — on a
        60-channel cryptomatte EXR (28 such layers, ~390 MB per 2K frame) the
        CRYPTOMATTE output was a complete second copy of LAYERS. One batch is
        allocated per distinct tensor *object* and handed to every name that
        aliased it, so the batched outputs alias exactly the way a single
        frame's do.

    Reading the alias map off one frame is sound because it is a property of
    process_exr_data's code rather than of the pixels — `all_for_crypto[k] is
    layers_dict[k]` holds for the same keys on every frame.
    """

    def __init__(self, count, sample, labels=None, ragged_hint="", reference=None):
        self._count = count
        # Real frame numbers, so "frame 1031" in an error names the file on
        # disk rather than its slot in the batch.
        self._labels = labels
        self._ragged_hint = ragged_hint
        # Which frame's dimensions every other frame is measured against. It is
        # whichever one came off the thread pool FIRST, not frame one — so an
        # error has to name it rather than say "the sequence starts at", which
        # is only true when the pool happens to finish in order.
        self._reference = reference
        self._list_len = {}
        self._passthrough = {}
        paths = [("rgb",), ("alpha",)]
        for kind, source in (("layers", sample[3]), ("crypto", sample[2])):
            for key, value in source.items():
                if isinstance(value, (list, tuple)):
                    self._list_len[(kind, key)] = len(value)
                    paths.extend((kind, key, i) for i in range(len(value)))
                elif isinstance(value, torch.Tensor):
                    paths.append((kind, key))
                else:
                    # Neither a tensor nor a list of them. Nothing to batch, so
                    # carry the first frame's value through rather than drop
                    # the key.
                    self._passthrough[(kind, key)] = value
        self._paths = paths

        # One batch per distinct tensor object; `_reps` is the subset of paths
        # actually copied per frame, one per batch.
        self._batch_of_path = {}
        self._reps = []
        by_object = {}
        for path in paths:
            source = _frame_slot(sample, path)
            if source is None:
                self._batch_of_path[path] = None
                continue
            if source.shape[0] != 1:
                raise Exception(
                    f"Bat_Loader: expected one frame per EXR read, got "
                    f"{tuple(source.shape)} for {'.'.join(map(str, path))}."
                )
            batch = by_object.get(id(source))
            if batch is None:
                batch = torch.empty((count,) + tuple(source.shape[1:]),
                                    dtype=source.dtype)
                by_object[id(source)] = batch
                self._reps.append((path, batch))
            self._batch_of_path[path] = batch

    def _label(self, index):
        """How to name batch slot `index` in an error."""
        if self._labels is not None and index < len(self._labels):
            return self._labels[index]
        return index

    def put(self, index, res):
        """Copy frame `index` into its slot in every batch, in place."""
        for path, batch in self._reps:
            source = _frame_slot(res, path)
            if source is None:
                raise Exception(
                    f"Bat_Loader: frame {self._label(index)} of this sequence "
                    f"is missing {'.'.join(map(str, path))}, which the other "
                    "frames have."
                )
            try:
                batch[index] = source[0]
            except RuntimeError as exc:
                against = (f"frame {self._reference}" if self._reference is not None
                           else "the rest of the sequence")
                raise Exception(
                    f"Bat_Loader: frame {self._label(index)} is "
                    f"{tuple(source.shape[1:])} where {against} is "
                    f"{tuple(batch.shape[1:])} — the frames aren't all the "
                    f"same size.{self._ragged_hint} ({exc})"
                ) from exc

    def finish(self, loaded):
        """Return (rgb, alpha, cryptomatte, layers) for the frames in `loaded`.

        `loaded` is the ascending list of slots that actually got a frame. In
        the normal case that is every slot and the batches are returned as they
        stand; when a frame failed to decode the holes are compacted out, which
        costs a copy but only on that error path.
        """
        keep = None
        if len(loaded) != self._count:
            keep = torch.tensor(loaded, dtype=torch.long)

        compacted = {}

        def resolve(batch):
            if batch is None or keep is None:
                return batch
            out = compacted.get(id(batch))
            if out is None:
                # index_select once per batch, not once per alias, so the
                # outputs keep sharing after compaction.
                out = compacted[id(batch)] = batch.index_select(0, keep)
            return out

        layers, crypto = {}, {}
        for (kind, key), value in self._passthrough.items():
            (layers if kind == "layers" else crypto)[key] = value
        for (kind, key), length in self._list_len.items():
            (layers if kind == "layers" else crypto)[key] = [None] * length
        for path in self._paths:
            if path[0] in ("rgb", "alpha"):
                continue
            target = layers if path[0] == "layers" else crypto
            value = resolve(self._batch_of_path[path])
            if len(path) == 3:
                target[path[1]][path[2]] = value
            else:
                target[path[1]] = value

        return (resolve(self._batch_of_path[("rgb",)]),
                resolve(self._batch_of_path[("alpha",)]),
                crypto, layers)


def _scan_windows(selected_frames, max_workers=8):
    """`(data, display)` windows for each of `selected_frames`, in order.

    Header reads only — no pixels — threaded because on an NFS mount the open()
    dominates. A frame that won't open comes back as None instead of raising:
    it will fail again in the decode pass, which already reports and skips a bad
    frame, and there is no reason for the overscan decision to be the thing that
    kills the load.
    """
    def read(file_info):
        try:
            return ExrProcessor.read_windows(file_info[1])
        except Exception as exc:
            logger.warning("[Bat_Loader] could not read EXR windows from %s: %s",
                           file_info[1], exc)
            return None

    workers = max(1, min(max_workers, len(selected_frames)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(read, selected_frames))


def _plan_overscan(selected_frames, mode):
    """Whether to conform this sequence's frames to their display window.

    An EXR carries a data window (the bbox it wrote pixels for) that need not
    match its display window (the format). Renders with overscan have a bigger
    one, anything that shrank its bbox has a smaller one — and either can move
    and resize frame to frame, which is what makes a sequence unbatchable: the
    frames are genuinely different sizes.

    `mode` is the node's `overscan` widget:

      keep  Never conform; batch the frames exactly as they sit on disk. For a
            graph that wants the overscan pixels — a round trip that has to line
            back up on the original bbox — and has no other way to ask.
      crop  Always conform, so the batch is the format whatever the bbox does.
      auto  Conform only when the frames disagree with each other. The default,
            and the only one of the three that cannot change the result of a
            load that works today: a sequence whose data window is constant is
            batched untouched (overscan and all), and one whose data window
            moves is otherwise a hard error.

    `auto` triggers on *any* difference between the frames' data windows, not
    just a difference in size: a bbox that shifts while keeping its dimensions
    batches happily and silently misaligns the frames, which is the worse
    failure of the two.

    Raises when the frames' display windows differ, since conforming can't
    square up a sequence that changes format mid-shot — better to say so than
    let it fall through to the batcher's "aren't all the same size".
    """
    if mode == "keep" or len(selected_frames) < 1:
        return False
    if mode == "crop":
        return True

    windows = [w for w in _scan_windows(selected_frames) if w is not None]
    if not windows:
        return False

    displays = {w[1] for w in windows}
    datas = {w[0] for w in windows}
    if len(datas) == 1:
        data, display = windows[0]
        if data != display and len(displays) == 1:
            logger.info(
                "[Bat_Loader] every frame is %dx%d at offset %d,%d against a "
                "%dx%d format — set `overscan` to `crop` to load the format "
                "only.", data[2], data[3], data[0], data[1], display[2], display[3])
        return False

    if len(displays) > 1:
        raise Exception(
            "Bat_Loader: this sequence changes format mid-shot — its frames "
            f"carry {len(displays)} different display windows "
            f"({', '.join(f'{d[2]}x{d[3]}' for d in sorted(displays))}). "
            "Cropping the overscan can't make them one batch; load the parts "
            "separately, or reformat them to a single resolution first."
        )

    display = windows[0][1]
    shown = sorted(f"{d[2]}x{d[3]}@{d[0]},{d[1]}" for d in datas)[:4]
    logger.info(
        "[Bat_Loader] the frames' bounding boxes differ (%d distinct data "
        "windows: %s%s) — conforming every frame to the %dx%d format. Set "
        "`overscan` to `keep` to batch them as written instead.",
        len(datas), " / ".join(shown), " / ..." if len(datas) > len(shown) else "",
        display[2], display[3])
    return True


def _raise_if_interrupted():
    """Raise comfy's InterruptProcessingException if the user hit Cancel.

    Imported lazily and failure-tolerantly: comfy.model_management is always
    present in a real ComfyUI process, but this module is also exercised by the
    standalone tests in this package, where it isn't.
    """
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except Exception:
        return
    throw_exception_if_processing_interrupted()


def _progress_bar(total):
    """ComfyUI's per-node progress bar, or None. Skipped for a single frame."""
    if total <= 1:
        return None
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _select(items, skip, nth, cap):
    """The skip / every-nth / cap window over an ordered frame list."""
    out = items[skip::nth]
    if cap > 0:
        out = out[:cap]
    return out


def _is_video(path):
    return path.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS if "." in path else False


class BatLoader:
    """Unified media loader: stills, ``####`` sequences, movies and EXR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "plate.exr, render_####.exr, edit.mov, or a folder",
                    # Read by web/bat_path_widget.js for the autocomplete popup.
                    "bat_path_extensions": ",".join(
                        sorted(set(IMAGE_EXTENSIONS) | set(VIDEO_EXTENSIONS))),
                }),
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": BIGMAX, "step": 1,
                    "tooltip": "Stop after this many frames. 0 loads all of them.",
                }),
                "skip_first_frames": ("INT", {
                    "default": 0, "min": 0, "max": BIGMAX, "step": 1,
                    "tooltip": "Drop this many frames from the start.",
                }),
                "select_every_nth": ("INT", {
                    "default": 1, "min": 1, "max": BIGMAX, "step": 1,
                    "tooltip": "Take every Nth frame. 1 takes them all.",
                }),
            },
            # `optional`, not `required`, so an API prompt written before this
            # widget existed still validates — ComfyUI rejects a prompt missing
            # a required key, whereas a missing optional one just leaves the
            # default here.
            "optional": {
                "overscan": (list(OVERSCAN_MODES), {
                    "default": "auto",
                    "tooltip":
                        "EXRs carry a bounding box (data window) that can be "
                        "bigger than the format (display window) when the "
                        "render has overscan, or smaller when something shrank "
                        "it — and it can change from frame to frame, which "
                        "makes the sequence unbatchable.\n"
                        "auto: load the frames as written, but crop them to "
                        "the format when their bounding boxes differ.\n"
                        "crop: always crop to the format, filling black where "
                        "the bounding box falls short of it.\n"
                        "keep: always load the bounding box as written (a "
                        "sequence whose box moves will error).",
                }),
            },
            "hidden": {
                # Used only to see which outputs are wired — see _consumes_slots.
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "LAYERS", "STRING",
                    "CRYPTOMATTE", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("images", "mask", "audio", "layers", "layer names",
                    "cryptomatte", "metadata", "resolved_path", "frame_rate")
    OUTPUT_TOOLTIPS = (
        "The loaded frames.",
        "The alpha channel as 1-alpha, so a fully opaque source gives an empty "
        "mask. Matches ComfyUI's mask convention.",
        "Audio under the loaded range. Movies only, and only extracted when "
        "this output is connected.",
        "Named layers from a multi-layer EXR — feed 🦇 EXR Layer to pull one "
        "out by name.",
        "The layer names found in the file, one per line.",
        "Cryptomatte rank stacks (plus every other layer, so one picker can "
        "reach them all) — feed 🦇 Cryptomatte Matte.",
        "The EXR header as JSON. A sequence gives a JSON list, one entry per "
        "frame.",
        "The path actually read, after quote-stripping and sequence expansion.",
        "Frames per second of the LOADED batch — a movie's source rate divided "
        f"by select_every_nth. A still sequence has no rate on disk, so it "
        f"reports {SEQUENCE_FPS:g}.",
    )
    CATEGORY = "BAT/Loader"
    DESCRIPTION = (
        "Load a still, a #### frame sequence, a movie or a folder of images "
        "from one path field. Multi-layer EXRs come back with their layers, "
        "cryptomattes and header metadata; movies keep 10/12-bit precision and "
        "alpha and can hand back their audio. The layer outputs and the audio "
        "are only decoded when something is wired to them."
    )
    FUNCTION = "load"

    # ── dispatch ────────────────────────────────────────────────────────────

    def load(self, path, frame_load_cap=0, skip_first_frames=0,
             select_every_nth=1, overscan="auto", unique_id=None, prompt=None):
        # INPUT_TYPES constrains the WIDGET, not an API submit or an input
        # converted from a widget. Hold the floors here, where every route in
        # has to pass: select_every_nth lands in a `frames[skip::nth]` slice, so
        # a 0 gives "slice step cannot be zero" instead of a frame range.
        select_every_nth = max(1, _as_int(select_every_nth, 1))
        skip_first_frames = max(0, _as_int(skip_first_frames, 0))
        frame_load_cap = max(0, _as_int(frame_load_cap, 0))
        if overscan not in OVERSCAN_MODES:
            if overscan is not None:
                logger.warning("[Bat_Loader] unknown overscan mode %r — using "
                               "'auto'", overscan)
            overscan = "auto"

        resolved_path = _strip_path(path or "")
        if not resolved_path:
            raise ValueError("Bat_Loader: the path is empty.")

        is_sequence = SequenceHandler.detect_sequence_pattern(resolved_path)
        if not is_sequence and not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Bat_Loader: path does not exist: {resolved_path!r}")

        is_exr = resolved_path.lower().endswith(".exr")

        if is_exr:
            if not OIIO_AVAILABLE:
                raise RuntimeError(
                    "Bat_Loader: reading EXR needs OpenImageIO, which isn't "
                    "installed in this environment (pip install OpenImageIO).")
            # Only pay for the extra layers if something reads them.
            want_layers = _consumes_slots(prompt, unique_id, _LAYER_OUTPUT_SLOTS)
            if not want_layers:
                logger.info("[Bat_Loader] no layer outputs connected — loading "
                            "the beauty channels only")
            out = self._load_exr(resolved_path, frame_load_cap, skip_first_frames,
                                 select_every_nth, is_sequence, want_layers,
                                 overscan)
        elif not is_sequence and _is_video(resolved_path) and os.path.isfile(resolved_path):
            out = self._load_video(resolved_path, frame_load_cap,
                                   skip_first_frames, select_every_nth,
                                   unique_id, prompt)
        else:
            out = self._load_standard(resolved_path, frame_load_cap,
                                      skip_first_frames, select_every_nth,
                                      is_sequence)

        images, mask, audio, layers, layer_names, crypto, metadata, fps = out
        result = (images, mask, audio, layers, layer_names, crypto, metadata,
                  resolved_path, fps)

        # Node-face thumbnail. Sampled from the middle of the batch so a
        # sequence shows something representative rather than always frame one.
        preview = None
        try:
            multi = images is not None and images.dim() == 4 and images.shape[0] > 1
            preview = generate_preview_for_comfyui(
                images, resolved_path, is_sequence=multi,
                frame_index=(images.shape[0] // 2 if multi else 0))
        except Exception as exc:
            logger.warning("[Bat_Loader] preview generation failed for %s: %s",
                           resolved_path, exc)

        return {"ui": {"images": preview or []}, "result": result}

    # ── EXR ─────────────────────────────────────────────────────────────────

    def _load_exr(self, path, frame_load_cap, skip_first_frames,
                  select_every_nth, is_sequence, want_layers, overscan):
        if not is_sequence:
            # One frame can't be ragged, so `auto` has nothing to compare
            # against and loads the file as written, same as `keep`; only an
            # explicit `crop` strips the overscan here.
            res = ExrProcessor.process_exr_data(
                path, False, want_layers=want_layers,
                conform=(overscan == "crop"))
            if isinstance(res, dict):
                res = res["result"]
            return (res[0], res[1], None, res[3], _names_text(res[4]), res[2],
                    res[6], SEQUENCE_FPS)

        files = SequenceHandler.find_sequence_files(path)
        if not files:
            raise FileNotFoundError(f"Bat_Loader: no files match {path!r}")

        frame_info = SequenceHandler.extract_frame_numbers(files)
        selected_frames = _select(frame_info, skip_first_frames,
                                  select_every_nth, frame_load_cap)
        if not selected_frames:
            raise ValueError(
                f"Bat_Loader: no frames left after skip={skip_first_frames} / "
                f"every-{select_every_nth} over {len(frame_info)} frames.")

        # Decided before any pixels are read, because it has to hold for every
        # frame in the batch — a per-frame decision would be exactly the ragged
        # batch this is avoiding.
        conform = _plan_overscan(selected_frames, overscan)
        ragged_hint = (
            " Their display windows (formats) must differ too, so cropping "
            "the overscan can't square them up."
            if conform else
            " If these are renders whose bounding box moves, set `overscan` "
            "to `crop` (or `auto`) to load them at the format."
        )

        # Frames are drained into preallocated batches as they arrive (see
        # _SequenceBatcher) rather than collected and cat'ed at the end.
        #
        # Decoded frames are handed over through `pending` and the worker
        # returns only the index: a Future KEEPS its result until the Future
        # itself is dropped, and every Future is held for the duration (they are
        # needed to cancel on interrupt) — so returning the frame from the
        # worker would pin the entire clip no matter what the consumer loop did
        # with it, costing a second full copy of the batch.
        pending = {}
        pending_lock = threading.Lock()

        def load_single_frame(idx, file_info):
            fn, fpath = file_info
            try:
                # want_preview=False: process_exr_data's own preview writes a
                # full PNG per frame and this path throws every one away —
                # load() makes the node-face thumbnail from the finished batch.
                res = ExrProcessor.process_exr_data(
                    fpath, False, want_preview=False, want_layers=want_layers,
                    conform=conform)
                if isinstance(res, dict):
                    res = res["result"]
            except Exception as exc:
                logger.error("[Bat_Loader] error loading frame %s: %s", fn, exc)
                return idx
            with pending_lock:
                pending[idx] = res
            return idx

        # EXR decode is CPU-bound and only partly releases the GIL, so past ~8
        # threads throughput flattens while peak RAM keeps climbing: every
        # in-flight frame holds its full layer dict until the batcher copies it
        # out. Override with BAT_EXR_WORKERS on a machine that wants otherwise.
        try:
            env_workers = int(os.environ.get("BAT_EXR_WORKERS", "") or 0)
        except ValueError:
            env_workers = 0
        max_workers = env_workers if env_workers > 0 else min(8, (os.cpu_count() or 1))
        max_workers = max(1, min(max_workers, len(selected_frames)))

        pbar = _progress_bar(len(selected_frames))

        batcher = None
        loaded = []
        meta_json = {}
        names_by_index = {}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [executor.submit(load_single_frame, i, f)
                       for i, f in enumerate(selected_frames)]
            try:
                for future in concurrent.futures.as_completed(futures):
                    idx = future.result()
                    with pending_lock:
                        res = pending.pop(idx, None)
                    if res is not None:
                        if batcher is None:
                            # The first frame to land sizes every batch and
                            # fixes the alias map.
                            batcher = _SequenceBatcher(
                                len(selected_frames), res,
                                labels=[f[0] for f in selected_frames],
                                ragged_hint=ragged_hint,
                                reference=selected_frames[idx][0])
                        batcher.put(idx, res)
                        loaded.append(idx)
                        names_by_index[idx] = res[4]
                        meta_json[idx] = res[6]
                    # Drop this frame's tensors now they're copied. Without
                    # this the whole clip stays live until the end, which is
                    # the 2x peak the batcher exists to avoid.
                    del res
                    if pbar is not None:
                        pbar.update(1)
                    # "Cancel current run" only sets a flag and ProgressBar
                    # does not check it. Poll once per completed frame.
                    _raise_if_interrupted()
            except BaseException:
                # Covers InterruptProcessingException (a BaseException, so a
                # bare `except Exception` would miss it) and any load error.
                # Drop queued work so shutdown doesn't wait on frames that
                # haven't started; in-flight decodes can't be preempted.
                for f in futures:
                    f.cancel()
                raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if batcher is None:
            raise RuntimeError(
                f"Bat_Loader: none of the {len(selected_frames)} frames of "
                f"{path!r} could be decoded.")

        # `loaded` came off as_completed, so sort it back into frame order for
        # the compaction step — and so layer names come from the lowest-numbered
        # frame that loaded.
        loaded.sort()
        layer_names = names_by_index[loaded[0]]
        metadata_list = [json.loads(meta_json[i]) for i in loaded]

        rgb, alpha, crypto, layers = batcher.finish(loaded)
        return (rgb, alpha, None, layers, _names_text(layer_names), crypto,
                json.dumps(metadata_list), SEQUENCE_FPS)

    # ── movies ──────────────────────────────────────────────────────────────

    def _load_video(self, path, frame_load_cap, skip_first_frames,
                    select_every_nth, unique_id, prompt):
        """Frames, alpha and audio from a movie, via 🦇 Video Loader's stack."""
        info = probe_video(path)
        if info is None:
            raise RuntimeError(f"Bat_Loader: no decoder could open {path!r}")
        n = int(info.get("frame_count") or 0)
        if n <= 0:
            probe = probe_ffmpeg(path)
            if probe and probe["duration"] > 0 and probe["fps"] > 0:
                n = int(round(probe["duration"] * probe["fps"]))
            if n <= 0:
                raise RuntimeError(
                    f"Bat_Loader: could not determine the frame count of "
                    f"{path!r}. The container may be streamed or damaged.")

        # Raise rather than clamp, to match what the sequence paths do with the
        # same input: clamping would quietly hand back the last frame alone,
        # which looks like a working load of a one-frame clip.
        if skip_first_frames >= n:
            raise ValueError(
                f"Bat_Loader: skip_first_frames={skip_first_frames} is past the "
                f"end of {os.path.basename(path)}, which has {n} frames.")
        start = skip_first_frames
        end = n - 1
        if frame_load_cap > 0:
            # `cap` counts KEPT frames, so the window it spans depends on the
            # stride: 10 frames every 3rd reaches 27 frames past the start.
            end = min(end, start + (frame_load_cap - 1) * select_every_nth)

        images, mask, meta = load_batch(path, start, end, select_every_nth)

        source_fps = meta.get("fps") or info.get("fps") or 0.0
        if source_fps <= 0:
            logger.warning("[Bat_Loader] no frame rate reported for %s — the "
                           "frame_rate output falls back to %g",
                           os.path.basename(path), SEQUENCE_FPS)
            source_fps = SEQUENCE_FPS
        # The batch is every Nth frame of the source, so it plays at the source
        # rate divided by the stride.
        frame_rate = source_fps / select_every_nth

        audio = None
        if _output_is_consumed(prompt, unique_id, _AUDIO_SLOT):
            # The time span of the trim, which the stride does not change:
            # taking every 3rd frame samples the same seconds more coarsely.
            audio = load_audio(path, start / source_fps,
                               (end - start + 1) / source_fps)

        return (images, mask, audio, {}, "", {}, "", frame_rate)

    # ── stills, directories and non-EXR sequences ───────────────────────────

    def _load_standard(self, path, frame_load_cap, skip_first_frames,
                       select_every_nth, is_sequence):
        if is_sequence:
            all_files = SequenceHandler.find_sequence_files(path)
            if not all_files:
                raise FileNotFoundError(f"Bat_Loader: no files match {path!r}")
            frame_info = SequenceHandler.extract_frame_numbers(all_files)
            selected = _select(frame_info, skip_first_frames,
                               select_every_nth, frame_load_cap)
            if not selected:
                raise ValueError(
                    f"Bat_Loader: no frames left after skip={skip_first_frames} "
                    f"/ every-{select_every_nth} over {len(frame_info)} frames.")
            files = [f[1] for f in selected]
        elif os.path.isdir(path):
            files = sorted(
                os.path.join(path, f) for f in os.listdir(path)
                if "." in f and f.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise FileNotFoundError(
                    f"Bat_Loader: no images in {path!r} (looked for "
                    f"{', '.join(IMAGE_EXTENSIONS)})")
            files = _select(files, skip_first_frames, select_every_nth,
                            frame_load_cap)
            if not files:
                raise ValueError(
                    f"Bat_Loader: no frames left after skip={skip_first_frames} "
                    f"/ every-{select_every_nth}.")
        else:
            files = [path]

        pbar = _progress_bar(len(files))

        # Preallocated from the first frame, for the same two reasons as the
        # EXR path: `images.append(t[..., :3])` and `masks.append(t[..., 3])`
        # are both strided VIEWS of the frame's RGBA tensor, so each frame keeps
        # its whole 4-channel buffer alive until the cat — and torch.cat then
        # allocates the finished batch on top of all of it.
        final_images = None
        final_masks = None
        for i, f in enumerate(files):
            _raise_if_interrupted()
            with Image.open(f) as opened:
                img_np = np.asarray(opened.convert("RGBA"), dtype=np.uint8)
            if final_images is None:
                h, w = img_np.shape[0], img_np.shape[1]
                final_images = torch.empty((len(files), h, w, 3), dtype=torch.float32)
                final_masks = torch.empty((len(files), h, w), dtype=torch.float32)
            elif (img_np.shape[0] != final_images.shape[1]
                    or img_np.shape[1] != final_images.shape[2]):
                raise Exception(
                    f"Bat_Loader: {os.path.basename(f)} is "
                    f"{img_np.shape[1]}x{img_np.shape[0]} where the sequence "
                    f"starts at {final_images.shape[2]}x{final_images.shape[1]} "
                    f"— the frames aren't all the same size.")
            # Scaled straight into the preallocated slot through numpy, using
            # the tensor's own buffer as the output. Going via
            # torch.from_numpy(img_np) instead would warn on every load —
            # Pillow hands back a READ-ONLY array and PyTorch refuses to treat
            # one as a tensor quietly — and the usual fix for that (copying the
            # array to make it writable) allocates a second full frame to
            # produce bytes we only ever read. `.numpy()` on a contiguous CPU
            # slice is a zero-copy view, so this writes once, in place.
            img_out = final_images[i].numpy()
            mask_out = final_masks[i].numpy()
            np.divide(img_np[:, :, :3], 255.0, out=img_out)
            # 1-alpha, to match ComfyUI's mask convention and the movie path:
            # an opaque still gives an empty mask.
            np.divide(img_np[:, :, 3], 255.0, out=mask_out)
            np.subtract(1.0, mask_out, out=mask_out)
            del img_np, img_out, mask_out
            if pbar is not None:
                pbar.update(1)

        if final_images is None:
            raise FileNotFoundError(f"Bat_Loader: nothing loadable at {path!r}")

        return (final_images, final_masks, None, {}, "", {}, "", SEQUENCE_FPS)

    # ── caching ─────────────────────────────────────────────────────────────

    @classmethod
    def IS_CHANGED(cls, path, frame_load_cap=0, skip_first_frames=0,
                   select_every_nth=1, overscan="auto", **kwargs):
        """A fingerprint that moves only when the referenced media does.

        **kwargs swallows the hidden inputs. PROMPT in particular must stay out
        of it: it changes on every queue, so folding it in would defeat caching
        entirely.

        The trim inputs are NOT in the fingerprint either — ComfyUI already
        hashes every raw widget value into the cache signature, so repeating
        them here buys nothing.
        """
        resolved_path = _strip_path(path or "")
        if not resolved_path:
            return "none"

        if SequenceHandler.detect_sequence_pattern(resolved_path):
            # Fast path: one directory scan fingerprints every frame. Falls
            # through to the glob below when it can't apply (a '#' in the
            # directory component, an unreadable dir, a frame vanishing
            # mid-scan).
            stats = SequenceHandler.scan_sequence_stats(resolved_path)
            if stats is not None:
                count, max_mtime_ns, total_size = stats
                if not count:
                    return _empty(resolved_path)
                return f"seq:{count}:{max_mtime_ns}:{total_size}:{resolved_path}"

            files = SequenceHandler.find_sequence_files(resolved_path)
            if not files:
                return _empty(resolved_path)
            # st_mtime_ns + total size rather than a float mtime: NFS timestamp
            # granularity plus clock skew between a render node and a
            # workstation can leave a rewritten frame on an identical
            # second-resolution mtime. Size moves whenever content does, so the
            # pair catches a same-timestamp rewrite that mtime alone misses.
            max_mtime_ns = 0
            total_size = 0
            try:
                for f in files:
                    st = os.stat(f)
                    if st.st_mtime_ns > max_mtime_ns:
                        max_mtime_ns = st.st_mtime_ns
                    total_size += st.st_size
            except OSError:
                # A frame vanished mid-scan. Fingerprint the failure WITH the
                # path so it still varies per sequence rather than collapsing
                # to one constant shared by every broken sequence.
                return f"seq-error:{len(files)}:{resolved_path}"
            return f"seq:{len(files)}:{max_mtime_ns}:{total_size}:{resolved_path}"

        if os.path.isdir(resolved_path):
            # A directory load depends on its listing, not on the directory's
            # own mtime — which does not move when a file inside is rewritten.
            count = 0
            max_mtime_ns = 0
            total_size = 0
            try:
                with os.scandir(resolved_path) as entries:
                    for entry in entries:
                        ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
                        if ext not in IMAGE_EXTENSIONS:
                            continue
                        st = entry.stat()
                        count += 1
                        max_mtime_ns = max(max_mtime_ns, st.st_mtime_ns)
                        total_size += st.st_size
            except OSError:
                return f"dir-error:{resolved_path}"
            return f"dir:{count}:{max_mtime_ns}:{total_size}:{resolved_path}"

        try:
            st = os.stat(resolved_path)
        except OSError:
            # Carries the path so a missing file still varies per input rather
            # than collapsing to one constant shared by every broken path.
            return f"missing:{resolved_path}"
        return f"file:{st.st_mtime_ns}:{st.st_size}:{resolved_path}"


def _names_text(names):
    """The layer-name list as the STRING the output is declared to be.

    bat_exr hands back a Python list, and returning that on a slot typed STRING
    is a lie every consumer pays for: a text display renders "['diffuse', 'N']"
    and anything doing string work on it breaks in a way that points at the
    wrong node. One name per line is what a list of names wants to be anyway.
    """
    if isinstance(names, str):
        return names
    if not names:
        return ""
    return "\n".join(str(n) for n in names)


def _empty(resolved_path):
    """Fingerprint for "this pattern matches no files *yet*".

    Carries the path rather than being a bare constant, for the same reason
    bat_video_loader's _fingerprint does: a constant shared by every
    unresolvable path is a value that cannot distinguish two of them, and a
    cache key that can't tell two inputs apart is one rewrite away from serving
    one node's frames to another. The transition to real frames still
    invalidates either way — this only closes the collision.
    """
    return f"seq-empty:{resolved_path}"


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ─── node-face scan ─────────────────────────────────────────────────────────
#
# What the path resolves to, cheaply enough to answer on every keystroke: a
# directory listing, plus an EXR *header* read or a (cached) movie probe. No
# frame is ever decoded here.
#
# It exists for one failure in particular. A `####` pattern that matches nothing
# — three hashes where the files have four digits, a typo in the directory, a
# sequence that hasn't rendered yet — looks exactly like one that matches 300
# frames until you press Queue. Answering "0 frames" while the artist is still
# typing turns that into a glance.

@server.PromptServer.instance.routes.get("/bat/loader-scan")
async def bat_loader_scan(request):
    raw = request.rel_url.query.get("path", "")
    path = _strip_path(raw or "")
    if not path:
        return server.web.json_response({"ok": False, "error": "no path"})

    try:
        info = _scan_path(path)
    except Exception as exc:                      # never 500 the node face
        logger.debug("[Bat_Loader] scan failed for %s: %s", path, exc)
        return server.web.json_response({"ok": False, "error": "unreadable"})
    return server.web.json_response(info)


def _scan_path(path):
    is_pattern = SequenceHandler.detect_sequence_pattern(path)
    out = {"ok": True, "pattern": is_pattern, "frames": 0, "width": 0,
           "height": 0, "layers": 0, "has_audio": False, "kind": ""}

    if is_pattern:
        stats = SequenceHandler.scan_sequence_stats(path)
        count = stats[0] if stats is not None else len(
            SequenceHandler.find_sequence_files(path))
        out["frames"] = count
        is_exr = path.lower().endswith(".exr")
        out["kind"] = "EXR sequence" if is_exr else "sequence"
        if count:
            first = SequenceHandler.find_sequence_files(path)[:1]
            if first:
                _describe_frame(first[0], out)
        return out

    if not os.path.exists(path):
        return {"ok": False, "error": "not found"}

    if os.path.isdir(path):
        try:
            names = [n for n in os.listdir(path)
                     if "." in n and n.rsplit(".", 1)[-1].lower() in IMAGE_EXTENSIONS]
        except OSError:
            return {"ok": False, "error": "unreadable folder"}
        out["kind"] = "folder"
        out["frames"] = len(names)
        if names:
            _describe_frame(os.path.join(path, sorted(names)[0]), out)
        return out

    if _is_video(path):
        out["kind"] = "movie"
        probe = probe_video(path)
        if probe:
            out["frames"] = int(probe.get("frame_count") or 0)
            out["width"] = int(probe.get("width") or 0)
            out["height"] = int(probe.get("height") or 0)
        ff = probe_ffmpeg(path)
        if ff:
            out["has_audio"] = bool(ff["has_audio"])
            out["width"] = out["width"] or ff["width"]
            out["height"] = out["height"] or ff["height"]
            if ff["bit_depth"] > 8:
                out["kind"] = "movie · 16-bit"
        return out

    out["kind"] = "EXR" if path.lower().endswith(".exr") else "still"
    out["frames"] = 1
    _describe_frame(path, out)
    return out


def _describe_frame(path, out):
    """Fill width/height/layers from ONE frame's header. Never decodes pixels."""
    if path.lower().endswith(".exr"):
        if not OIIO_AVAILABLE:
            return
        try:
            meta = ExrProcessor.scan_exr_metadata(path)
        except Exception:
            return
        subs = meta.get("subimages") or []
        if subs:
            out["width"] = int(subs[0].get("width") or 0)
            out["height"] = int(subs[0].get("height") or 0)
        # Named layers, not raw channels: "18 layers" is what the artist will
        # see on the `layers` output, whereas "60 channels" is bookkeeping.
        #
        # Both of the ways an EXR names a layer count, because renderers use
        # both and a file often mixes them: a multi-part EXR puts each layer in
        # its own named SUBIMAGE, while a single-part one prefixes the channel
        # ("diffuse.R", "N.X"). Counting only subimage names reported 0 layers
        # for a perfectly ordinary single-part render.
        names = {s.get("name") for s in subs if s.get("name")}
        names.discard("default")
        for sub in subs:
            for channel in sub.get("channel_names") or []:
                if "." in channel:
                    names.add(channel.rsplit(".", 1)[0])
        out["layers"] = len(names)
        return
    try:
        with Image.open(path) as img:              # header only, no decode
            out["width"], out["height"] = img.width, img.height
    except Exception:
        pass
