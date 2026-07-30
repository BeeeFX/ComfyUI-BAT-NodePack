"""
Bat_FramePicker — browse any video or frame sequence as a contact-sheet grid
and output the selected frame.

A sibling of the ETC VRI Frame Picker, but path-driven instead of VRI-driven:
the user types a path to a video file *or* a frame-sequence pattern (e.g.
/shots/sh010/render.####.exr) — with the same filesystem autocomplete the
Volt/BAT loaders use — and gets a scrollable grid of every frame. Clicking a
cell sets `frame_index`; only that one full-res frame is decoded on execute.

Self-contained: image/sequence helpers live here; video decoding reuses this
pack's bat_video_loader.
"""

import io
import os
import re
import glob
import logging
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

import server

from .bat_video_loader import probe_video, load_frames

logger = logging.getLogger(__name__)

IMG_EXTS = (".exr", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

_FRAME_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_FRAME_CACHE_MAX = 8


# ─── Path / sequence helpers ──────────────────────────────────────────────────


def _strip_path(p):
    return (p or "").strip().strip('"').strip("'")


def _sequence_patterns(names):
    """Group numbered file names into name.####.ext sequence patterns.

    Two guards keep phantom entries off the autocomplete list:
      1. A pattern is only emitted when ≥2 distinct files share the
         same (prefix, digit-count, ext) — a lone `clip_001.mp4` should
         NOT show up as `clip_###.mp4` because there's no sequence to
         load. Previously every numbered file produced a pattern.
      2. Only image extensions count toward sequences. A pile of
         independent `clip_001.mp4 / clip_002.mp4` videos isn't a frame
         sequence; treating it as one is misleading and pulls those
         files into a Frame Picker pretending it's an image sequence.
    """
    buckets: dict[tuple, int] = {}
    for f in names:
        m = re.search(r"^(.*?)(\d+)(\.[^.]+)$", f)
        if not m:
            continue
        prefix, digits, ext = m.groups()
        if len(digits) < 2:
            continue
        if ext.lower() not in IMG_EXTS:
            continue
        buckets[(prefix, len(digits), ext)] = buckets.get((prefix, len(digits), ext), 0) + 1
    return sorted(
        f"{prefix}{'#' * d}{ext}"
        for (prefix, d, ext), n in buckets.items()
        if n >= 2
    )


def find_sequence_files(pattern_path):
    """All files on disk matching a name.####.ext pattern, frame-sorted."""
    pattern_path = pattern_path.replace("\\", "/")
    m = re.search(r"#+", pattern_path)
    if not m:
        return []
    pad = m.group(0)
    glob_pat = pattern_path.replace(pad, "*")
    found = [f.replace("\\", "/") for f in glob.glob(glob_pat)]
    escaped = re.escape(pattern_path)
    rx = re.compile("^" + escaped.replace(re.escape(pad), rf"\d{{{len(pad)}}}") + "$",
                    re.IGNORECASE)
    return sorted(f for f in found if rx.match(f))


def _source_type(path):
    if "#" in path:
        return "sequence"
    if path.lower().endswith(VIDEO_EXTS):
        return "video"
    return "image"


def count_frames(path):
    t = _source_type(path)
    if t == "sequence":
        return len(find_sequence_files(path))
    if t == "video":
        info = probe_video(path)
        return int(info["frame_count"]) if info else 0
    return 1 if os.path.isfile(path) else 0


# ─── Frame decoding ────────────────────────────────────────────────────────────


def _load_image_file(path):
    """Load a still image (incl. EXR) → float32 RGB ndarray (H,W,3)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        try:
            import cv2
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if arr is None:
                raise RuntimeError("cv2 returned None")
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr[:, :, :3][:, :, ::-1]  # BGR→RGB
            return np.ascontiguousarray(arr).astype(np.float32)
        except Exception as e:
            try:
                import imageio.v3 as iio
                arr = iio.imread(path)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                return np.ascontiguousarray(arr[:, :, :3]).astype(np.float32)
            except Exception:
                raise RuntimeError(f"Could not read EXR {path}: {e}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def resolve_frame_file(path, frame_index):
    """The actual file a (path, frame_index) pair resolves to, or None for video.

    Used so callers can test the REAL file's extension rather than the sequence
    pattern: `render.####.EXR` (uppercase) or a float `.tif` used to miss the
    `path.lower().endswith(".exr")` check done against the pattern string.
    """
    if _source_type(path) == "sequence":
        files = find_sequence_files(path)
        if not files:
            return None
        idx = max(0, min(int(frame_index), len(files) - 1))
        return files[idx]
    if _source_type(path) == "video":
        return None
    return path


def is_float_source(path, frame_index=0):
    """True when the resolved frame is high-dynamic-range float data whose values
    may legitimately exceed 1.0 (EXR / float TIFF / HDR)."""
    f = resolve_frame_file(path, frame_index)
    if not f:
        return False
    return os.path.splitext(f)[1].lower() in (".exr", ".hdr", ".tif", ".tiff")


def load_one_frame(path, frame_index):
    """Return a single frame as (H,W,3) float32 from a video / sequence / image."""
    t = _source_type(path)
    if t == "sequence":
        files = find_sequence_files(path)
        if not files:
            raise RuntimeError(f"No frames for sequence {path!r}")
        idx = max(0, min(int(frame_index), len(files) - 1))
        return _load_image_file(files[idx])
    if t == "video":
        arr = load_frames(path, int(frame_index), int(frame_index))  # (1,H,W,3) uint8
        return arr[0].astype(np.float32) / 255.0
    return _load_image_file(path)


def _linear_to_srgb(arr):
    a = np.clip(arr, 0.0, None)
    low = a <= 0.0031308
    out = np.where(low, a * 12.92, 1.055 * np.power(np.clip(a, 0.0031308, None), 1 / 2.4) - 0.055)
    return out


# ─── ComfyUI node ──────────────────────────────────────────────────────────────


class BatFramePicker:
    """Browse a video / sequence on the node and output the selected frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Autocomplete via /bat/frame-picker/getpath (front-end widget).
                "path": ("STRING", {
                    "default": "",
                    "bat_path_extensions": ",".join(
                        e.lstrip(".") for e in (IMG_EXTS + VIDEO_EXTS)
                    ),
                }),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "linear_to_srgb": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "BAT/io"
    DESCRIPTION = (
        "Browse any video or frame sequence as a contact-sheet grid on the "
        "node and output the selected frame. Path-driven: type a path to a "
        "video file or a sequence pattern, click a thumbnail to pick the "
        "frame."
    )
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, path, frame_index, linear_to_srgb):
        p = _strip_path(path)
        mt = 0.0
        try:
            # For a sequence pattern, stat the first matching file.
            stat_target = p
            if _source_type(p) == "sequence":
                files = find_sequence_files(p)
                stat_target = files[0] if files else p
            mt = os.path.getmtime(stat_target)
        except OSError:
            pass
        return f"{p}|{frame_index}|{linear_to_srgb}|{mt}"

    def load(self, path, frame_index, linear_to_srgb=False):
        path = _strip_path(path)
        if not path:
            raise ValueError("Bat_FramePicker: no path provided")
        if _source_type(path) != "sequence" and not os.path.isfile(path):
            raise FileNotFoundError(f"Bat_FramePicker: not a file: {path!r}")

        key = (path, int(frame_index), bool(linear_to_srgb))
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(key)
            return (cached,)

        arr = load_one_frame(path, frame_index)  # (H,W,3) float32

        # Gate on the RESOLVED file's extension, not the pattern string: a
        # sequence declared as `render.####.EXR` (uppercase) or a float .tif
        # silently skipped the conversion when this tested `path`.
        is_float = is_float_source(path, frame_index)
        if linear_to_srgb and is_float:
            arr = _linear_to_srgb(arr)

        # Preserve HDR range for float sources.
        #
        # This used to `np.clip(arr, 0, 1)` unconditionally, which permanently
        # destroyed EXR superwhites (a specular at 6.0 arrived as flat 1.0,
        # indistinguishable from diffuse white) BEFORE any downstream grade could
        # recover them. 8-bit sources (video / PIL) are already normalised to
        # 0-1 by construction, so clipping them was a no-op anyway; we only clamp
        # the lower bound to keep negatives out of the pipeline.
        if is_float:
            arr = np.clip(arr, 0.0, None)
        else:
            arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1,H,W,3)
        _FRAME_CACHE[key] = tensor
        while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
            _FRAME_CACHE.popitem(last=False)
        return (tensor,)


# ─── API routes ──────────────────────────────────────────────────────────────


@server.PromptServer.instance.routes.get("/bat/frame-picker/getpath")
async def bat_frame_picker_getpath(request):
    """Directory autocomplete — mirrors Volt Loader's behaviour and adds
    detected frame-sequence patterns (so a whole sequence is one entry)."""
    query = request.rel_url.query
    raw = query.get("path", "")
    if not raw:
        return server.web.Response(status=204)

    path = os.path.abspath(_strip_path(raw))
    if not os.path.isdir(path):
        return server.web.json_response([])

    valid_extensions = query.get("extensions")
    exts = (
        set(e.strip().lower() for e in valid_extensions.split(",") if e.strip())
        if valid_extensions else None
    )

    items = []
    file_names = []
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir():
                    items.append(entry.name + "/")
                    continue
                ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
                if exts is None or ext in exts:
                    items.append(entry.name)
                    file_names.append(entry.name)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[Bat FramePicker] getpath error: {e}")
        return server.web.json_response([])

    # Collapse numbered stills into sequence patterns as selectable entries.
    items.extend(_sequence_patterns(file_names))

    # Sort newest-first by mtime, stat'ing each entry AT MOST ONCE.
    #
    # This used to call os.stat() inside the sort key, so on a network mount the
    # directory got re-stat'ed throughout the comparison sort — on every
    # autocomplete keystroke. Worse, a single OSError on any one entry threw the
    # whole sort away and silently fell back to plain alphabetical.
    mtimes = {}
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    mtimes[entry.name] = entry.stat().st_mtime
                except OSError:
                    mtimes[entry.name] = 0.0
    except OSError:
        mtimes = {}
    # Sequence patterns ("name.####.ext") aren't real files — keep them at 0 so
    # they group together, as before.
    items.sort(key=lambda f: mtimes.get(f, 0.0) if "#" not in f else 0.0)
    return server.web.json_response(items)


@server.PromptServer.instance.routes.get("/bat/frame-picker/info")
async def bat_frame_picker_info(request):
    path = _strip_path(request.rel_url.query.get("path", ""))
    if not path:
        return server.web.json_response({"ok": False, "error": "no path"})
    t = _source_type(path)
    if t != "sequence" and not os.path.isfile(path):
        return server.web.json_response({"ok": False, "error": "not a file"})
    frames = count_frames(path)
    if not frames:
        return server.web.json_response({"ok": False, "error": "no frames found", "type": t})
    return server.web.json_response({"ok": True, "type": t, "frames": frames})


@server.PromptServer.instance.routes.get("/bat/frame-picker/frame")
async def bat_frame_picker_frame(request):
    q = request.rel_url.query
    path = _strip_path(q.get("path", ""))
    try:
        frame = int(q.get("frame", "0"))
    except ValueError:
        frame = 0
    try:
        max_w = int(q.get("max_w", "256"))
    except ValueError:
        max_w = 256

    if not path or (_source_type(path) != "sequence" and not os.path.isfile(path)):
        return server.web.Response(status=404, text="bad path")
    try:
        arr = load_one_frame(path, frame)
    except Exception as e:
        return server.web.Response(status=500, text=f"decode error: {e}")

    if path.lower().endswith(".exr"):
        arr = _linear_to_srgb(arr)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    if img.width > max_w:
        h = max(1, int(img.height * max_w / img.width))
        img = img.resize((max_w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return server.web.Response(body=buf.getvalue(), content_type="image/jpeg")
