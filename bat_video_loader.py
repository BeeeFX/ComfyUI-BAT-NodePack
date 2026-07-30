"""
Bat_VideoLoader — load a trimmed range of frames from a video file.

The user types a path (with directory autocomplete provided by the
``/bat/getpath`` route, mirroring Volt Loader's behaviour) and uses a
dual-handle range slider on the node to pick a start/end frame. The
front-end queries ``/bat/video-info`` for duration/fps and
``/bat/video-frame`` for thumbnail previews of the trim endpoints.

Decoder selection order (lazy import, first that works wins):
  1. decord  — fastest seek for arbitrary frames
  2. torchvision.io.read_video  — full read, then slice
  3. opencv (cv2)  — broad codec compatibility, frame-by-frame seek
"""

import io
import logging
import os
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

import server

# Bounded LRU of decoded frame batches, keyed by (abs_path, mtime, start, end).
#
# Bounded by BYTES, not entry count. The old `_FRAME_CACHE_MAX = 4` counted
# entries, which says nothing about memory: four cached 500-frame 1080p float32
# batches is 4 x 500 x 1920 x 1080 x 3 x 4 B ≈ 50 GB. A byte budget keeps the
# cache useful for the common "re-run with the same trim window" case while
# making the worst case bounded and predictable.
_FRAME_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
# Entry-count cap is kept as a secondary guard against pathological tiny entries.
_FRAME_CACHE_MAX = 8
_FRAME_CACHE_MAX_BYTES = 4 * 1024 ** 3   # 4 GiB


def _tensor_nbytes(t) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


def _trim_frame_cache():
    """Evict oldest entries until both the byte budget and the entry cap hold."""
    total = sum(_tensor_nbytes(v) for v in _FRAME_CACHE.values())
    while _FRAME_CACHE and (total > _FRAME_CACHE_MAX_BYTES
                            or len(_FRAME_CACHE) > _FRAME_CACHE_MAX):
        _key, victim = _FRAME_CACHE.popitem(last=False)
        total -= _tensor_nbytes(victim)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg")


# ─── Decoder backends ───────────────────────────────────────────────────────


def _probe_decord(path: str):
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        return {
            "backend": "decord",
            "frame_count": len(vr),
            "fps": float(vr.get_avg_fps() or 0.0),
            "width": int(vr[0].shape[1]),
            "height": int(vr[0].shape[0]),
            "_reader": vr,
        }
    except Exception:
        return None


def _probe_cv2(path: str):
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return None
        info = {
            "backend": "cv2",
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        return info
    except Exception:
        return None


def _probe_torchvision(path: str):
    try:
        from torchvision.io import read_video, read_video_timestamps  # type: ignore
        pts, fps = read_video_timestamps(path, pts_unit="sec")
        if not pts:
            return None
        return {
            "backend": "torchvision",
            "frame_count": len(pts),
            "fps": float(fps or 0.0),
            "width": 0,
            "height": 0,
        }
    except Exception:
        return None


def probe_video(path: str) -> Optional[dict]:
    """Return basic info about a video, or None if no backend can read it."""
    for fn in (_probe_decord, _probe_cv2, _probe_torchvision):
        info = fn(path)
        if info is not None:
            info.pop("_reader", None)
            return info
    return None


# ─── Frame extraction ───────────────────────────────────────────────────────


def _frames_decord(path: str, start: int, end: int, stride: int = 1) -> Optional[np.ndarray]:
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        n = len(vr)
        s = max(0, min(start, n - 1))
        e = max(s, min(end, n - 1))
        # get_batch accepts an arbitrary index list, so a stride costs nothing:
        # we decode only the frames we keep instead of the whole span.
        indices = list(range(s, e + 1, max(1, int(stride))))
        batch = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
        return batch
    except Exception as e:
        logger.warning(f"[Bat_VideoLoader] decord failed: {e}")
        return None


def _frames_cv2(path: str, start: int, end: int, stride: int = 1) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        s = max(0, min(start, total - 1))
        e = max(s, min(end, total - 1))
        st = max(1, int(stride))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        out = []
        # grab() advances without decoding/converting; retrieve() only decodes
        # the frames we actually keep. Skipping the colour conversion for
        # discarded frames is where the stride saving comes from.
        for i in range(s, e + 1):
            if (i - s) % st == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if not cap.grab():
                    break
        cap.release()
        return np.stack(out, axis=0) if out else None
    except Exception as e:
        logger.warning(f"[Bat_VideoLoader] cv2 failed: {e}")
        return None


def _frames_torchvision(path: str, start: int, end: int, stride: int = 1) -> Optional[np.ndarray]:
    try:
        from torchvision.io import read_video  # type: ignore
        # read_video returns (T, H, W, C) uint8.
        video, _, _ = read_video(path, pts_unit="sec")
        if video.numel() == 0:
            return None
        n = video.shape[0]
        s = max(0, min(start, n - 1))
        e = max(s, min(end, n - 1))
        # read_video has already decoded everything, so the stride is just a
        # slice here — no decode saving available on this backend.
        return video[s:e + 1:max(1, int(stride))].numpy()
    except Exception as e:
        logger.warning(f"[Bat_VideoLoader] torchvision failed: {e}")
        return None


def load_frames(path: str, start: int, end: int, stride: int = 1) -> np.ndarray:
    """Return frames [start..end] (inclusive), every `stride`-th, as uint8 (N,H,W,3).

    The stride is pushed DOWN into the decoder so we never decode frames that
    are about to be discarded (see _frames_decord / _frames_cv2).
    """
    for fn in (_frames_decord, _frames_cv2, _frames_torchvision):
        out = fn(path, start, end, stride)
        if out is not None and len(out) > 0:
            return out
    raise RuntimeError(
        f"No working video decoder produced frames for {path!r}. "
        f"Install one of: decord, opencv-python, or torchvision with video support."
    )


def _single_frame(path: str, index: int) -> Optional[np.ndarray]:
    arr = load_frames(path, index, index)
    return arr[0] if arr is not None and len(arr) else None


# ─── Path safety ────────────────────────────────────────────────────────────


def _strip_path(p: str) -> str:
    return p.strip().strip('"').strip("'")


def _is_video(name: str) -> bool:
    return name.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS


# ─── ComfyUI node ───────────────────────────────────────────────────────────


class VideoLoader:
    """Load a trimmed image batch from a video file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "bat_path_extensions": ",".join(VIDEO_EXTENSIONS),
                }),
                "start_frame":       ("INT", {"default": 0,  "min": 0,  "max": 999999}),
                "end_frame":         ("INT", {"default": -1, "min": -1, "max": 999999}),
                "select_every_nth":  ("INT", {"default": 1,  "min": 1,  "max": 999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "BAT/video"
    DESCRIPTION = (
        "Load a trimmed range of frames from a video file. Type a path (with "
        "directory autocomplete) and set the range with a dual-handle trim "
        "slider showing thumbnails of the in/out frames."
    )
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, path, start_frame, end_frame, select_every_nth):
        try:
            return f"{path}|{os.path.getmtime(path)}|{start_frame}|{end_frame}|{select_every_nth}"
        except OSError:
            return f"{path}|missing|{start_frame}|{end_frame}|{select_every_nth}"

    def load(self, path: str, start_frame: int, end_frame: int, select_every_nth: int = 1):
        path = _strip_path(path)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Bat_VideoLoader: not a file: {path!r}")
        abs_path = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(abs_path)
        except OSError:
            mtime = 0.0

        info = probe_video(path)
        if info is None:
            raise RuntimeError(f"Bat_VideoLoader: no decoder could open {path!r}")
        n = info["frame_count"]
        start = max(0, min(int(start_frame), n - 1))
        end = int(end_frame)
        if end < 0 or end >= n:
            end = n - 1
        if end < start:
            end = start

        stride = max(1, int(select_every_nth))
        cache_key = (abs_path, mtime, start, end, stride)
        cached = _FRAME_CACHE.get(cache_key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(cache_key)
            return (cached,)

        # Decode only the frames we're going to KEEP. This used to decode the
        # whole [start, end] range and then throw most of it away with
        # `frames[::stride]` — with select_every_nth=10 over a 3000-frame range
        # that decoded 3000 frames to return 300. Volt_Loader already applies its
        # stride during frame selection; this brings the two in line.
        frames = load_frames(path, start, end, stride=stride)
        frames = frames.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames)
        _FRAME_CACHE[cache_key] = tensor
        _trim_frame_cache()
        return (tensor,)


# ─── API routes ─────────────────────────────────────────────────────────────


@server.PromptServer.instance.routes.get("/bat/getpath")
async def bat_getpath(request):
    """Directory autocomplete — mirrors Volt Loader's /volt-loader/getpath."""
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
        if valid_extensions
        else None
    )

    # Collect names AND mtimes in the single scandir pass — the DirEntry already
    # carries the stat, so sorting is free. This used to re-stat every entry
    # inside the sort key (`os.stat(...)` per comparison) on every autocomplete
    # keystroke, which is slow on network mounts; and one OSError anywhere threw
    # the whole mtime ordering away and fell back to alphabetical.
    items = []
    mtimes = {}
    try:
        for entry in os.scandir(path):
            try:
                try:
                    mtimes[entry.name] = entry.stat().st_mtime
                except OSError:
                    mtimes[entry.name] = 0.0
                if entry.is_dir():
                    items.append(entry.name + "/")
                    mtimes[entry.name + "/"] = mtimes.get(entry.name, 0.0)
                    continue
                ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
                if exts is None or ext in exts:
                    items.append(entry.name)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"[Bat] getpath error: {e}")
        return server.web.json_response([])

    items.sort(key=lambda f: mtimes.get(f, 0.0))
    return server.web.json_response(items)


@server.PromptServer.instance.routes.get("/bat/video-info")
async def bat_video_info(request):
    raw = request.rel_url.query.get("path", "")
    path = _strip_path(raw)
    if not path or not os.path.isfile(path):
        return server.web.json_response({"ok": False, "error": "not a file"})
    info = probe_video(path)
    if info is None:
        return server.web.json_response({"ok": False, "error": "no decoder could open file"})
    return server.web.json_response({"ok": True, **info})


@server.PromptServer.instance.routes.get("/bat/video-stream")
async def bat_video_stream(request):
    """Serve the raw video file with Range support so an HTML5 <video>
    element can seek smoothly. aiohttp.web.FileResponse handles Range
    headers automatically."""
    raw = request.rel_url.query.get("path", "")
    path = _strip_path(raw)
    if not path or not os.path.isfile(path):
        return server.web.Response(status=404, text="not a file")
    return server.web.FileResponse(path)


@server.PromptServer.instance.routes.get("/bat/video-frame")
async def bat_video_frame(request):
    q = request.rel_url.query
    path = _strip_path(q.get("path", ""))
    try:
        frame_index = int(q.get("frame", "0"))
    except ValueError:
        frame_index = 0
    max_w = int(q.get("max_w", "320"))
    if not path or not os.path.isfile(path):
        return server.web.Response(status=404, text="not a file")
    try:
        arr = _single_frame(path, frame_index)
    except Exception as e:
        return server.web.Response(status=500, text=f"decode error: {e}")
    if arr is None:
        return server.web.Response(status=500, text="no frame")

    img = Image.fromarray(arr)
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return server.web.Response(body=buf.getvalue(), content_type="image/jpeg")
