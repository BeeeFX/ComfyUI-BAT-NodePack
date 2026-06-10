"""
Bat_VriPicker — browse a VRI's frames as a contact-sheet grid and output one.

Workflow: paste a VRI, the node resolves it to a directory and enumerates its
"components" (image sequences, single images, videos). A lightweight *preview*
component (default jpg) drives an on-node scrollable grid of every frame; the
user clicks one, and the node decodes only that single frame from the heavier
*output* component (default exr) — so scrubbing stays cheap and only the chosen
full-res frame is ever loaded.

Self-contained: the small VRI path/sequence helpers are ported from Volt
Loader rather than imported, so this node has no cross-pack dependency. Video
decoding reuses this pack's own bat_video_loader helpers.
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

logger = logging.getLogger(__name__)

IMG_EXTS = (".exr", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

# Small LRU for decoded output frames, keyed by (dir, component, frame, mtime).
_FRAME_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_FRAME_CACHE_MAX = 8


# ─── VRI resolution + component detection (ported from Volt Loader) ──────────


def vri_to_path(vri):
    """provider:project:path…:name-type@version →
    /jobs/{provider}/{project}/library/{type}/{path}/{name}/{name}_v{NNN}"""
    try:
        vri = (vri or "").strip()
        if "@" not in vri:
            return None
        vri_base, version = vri.split("@")
        parts = vri_base.split(":")
        if len(parts) < 3:
            return None
        provider, project = parts[0], parts[1]
        path_parts, name_type = parts[2:-1], parts[-1]
        if "-" not in name_type:
            return None
        name, asset_type = name_type.split("-")
        path_str = "/".join(path_parts)
        version_str = f"v{int(version):03d}"
        resolved = os.path.join(
            "/jobs", provider, project, "library", asset_type,
            path_str, name, f"{name}_{version_str}",
        )
        return resolved.replace("\\", "/")
    except Exception as e:
        logger.error(f"[Bat] VRI parse error for {vri!r}: {e}")
        return None


def _sequence_patterns(files):
    """Group numbered files into name.####.ext patterns."""
    patterns = set()
    for f in files:
        m = re.search(r"^(.*?)(\d+)(\.[^.]+)$", f)
        if m:
            prefix, digits, ext = m.groups()
            if len(digits) >= 2:
                patterns.add(f"{prefix}{'#' * len(digits)}{ext}")
    return sorted(patterns)


def detect_components(directory):
    """Return [{name, type, frames}] for a resolved VRI directory."""
    if not directory or not os.path.isdir(directory):
        return []
    try:
        files = os.listdir(directory)
    except OSError:
        return []

    out = []
    img_files = [f for f in files if f.lower().endswith(IMG_EXTS)]
    patterns = _sequence_patterns(img_files)
    # A pattern like "shot.####.exr" → regex "^shot\.\d\d\d\d\.exr$".
    pattern_regexes = [
        re.compile("^" + p.replace(".", r"\.").replace("#", r"\d") + "$")
        for p in patterns
    ]
    for p in patterns:
        out.append({"name": p, "type": "sequence",
                    "frames": len(find_sequence_files(os.path.join(directory, p)))})

    # Single images (not part of any detected sequence).
    for f in img_files:
        if not any(rx.match(f) for rx in pattern_regexes):
            out.append({"name": f, "type": "image", "frames": 1})

    # Videos.
    for f in files:
        if f.lower().endswith(VIDEO_EXTS):
            out.append({"name": f, "type": "video",
                        "frames": _video_frame_count(os.path.join(directory, f))})

    out.sort(key=lambda c: c["name"])
    return out


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


def _video_frame_count(path):
    try:
        from .bat_video_loader import probe_video
        info = probe_video(path)
        return int(info["frame_count"]) if info else 0
    except Exception:
        return 0


# ─── Component model ─────────────────────────────────────────────────────────


def _component_type(name):
    low = name.lower()
    if "#" in name:
        return "sequence"
    if low.endswith(VIDEO_EXTS):
        return "video"
    return "image"


def _component_frame_count(directory, name):
    t = _component_type(name)
    if t == "sequence":
        return len(find_sequence_files(os.path.join(directory, name)))
    if t == "video":
        return _video_frame_count(os.path.join(directory, name))
    return 1


# ─── Frame decoding ──────────────────────────────────────────────────────────


def _load_image_file(path):
    """Load a still image (incl. EXR) → float32 RGB ndarray (H,W,3), 0..1
    for LDR; EXR returns linear values (may exceed 1)."""
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
    # LDR via PIL.
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_video_frame(path, frame_index):
    from .bat_video_loader import load_frames
    arr = load_frames(path, frame_index, frame_index)  # uint8 (1,H,W,3)
    return arr[0].astype(np.float32) / 255.0


def load_one_frame(directory, component, frame_index):
    """Return a single frame as (H,W,3) float32 from any component type."""
    t = _component_type(component)
    if t == "sequence":
        files = find_sequence_files(os.path.join(directory, component))
        if not files:
            raise RuntimeError(f"No frames for sequence {component!r}")
        idx = max(0, min(int(frame_index), len(files) - 1))
        return _load_image_file(files[idx])
    if t == "video":
        return _load_video_frame(os.path.join(directory, component), int(frame_index))
    # single image
    return _load_image_file(os.path.join(directory, component))


def _linear_to_srgb(arr):
    a = np.clip(arr, 0.0, None)
    low = a <= 0.0031308
    out = np.where(low, a * 12.92, 1.055 * np.power(np.clip(a, 0.0031308, None), 1 / 2.4) - 0.055)
    return out


# ─── ComfyUI node ────────────────────────────────────────────────────────────


class VriPicker:
    """Browse a VRI's frames on the node and output the selected one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vri": ("STRING", {"default": "", "bat_vri": True}),
                # Combo options are filled client-side from /bat/vri-info.
                "preview_component": ("STRING", {"default": "", "bat_component": "preview"}),
                "output_component": ("STRING", {"default": "", "bat_component": "output"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "linear_to_srgb": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "BAT/io"
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, vri, preview_component, output_component, frame_index, linear_to_srgb):
        directory = vri_to_path(vri) or ""
        mt = 0.0
        try:
            mt = os.path.getmtime(directory)
        except OSError:
            pass
        return f"{vri}|{output_component}|{frame_index}|{linear_to_srgb}|{mt}"

    def load(self, vri, preview_component, output_component, frame_index, linear_to_srgb=False):
        directory = vri_to_path(vri)
        if not directory or not os.path.isdir(directory):
            raise FileNotFoundError(f"Bat_VriPicker: VRI did not resolve to a directory: {vri!r}")
        component = output_component or preview_component
        if not component:
            raise ValueError("Bat_VriPicker: no output component selected")

        try:
            mt = os.path.getmtime(directory)
        except OSError:
            mt = 0.0
        key = (directory, component, int(frame_index), mt, bool(linear_to_srgb))
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(key)
            return (cached,)

        arr = load_one_frame(directory, component, frame_index)  # (H,W,3) float32
        if linear_to_srgb and component.lower().endswith(".exr"):
            arr = _linear_to_srgb(arr)
        arr = np.clip(arr, 0.0, 1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1,H,W,3)
        _FRAME_CACHE[key] = tensor
        while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
            _FRAME_CACHE.popitem(last=False)
        return (tensor,)


# ─── API routes ──────────────────────────────────────────────────────────────


@server.PromptServer.instance.routes.get("/bat/vri-info")
async def bat_vri_info(request):
    vri = request.rel_url.query.get("vri", "")
    directory = vri_to_path(vri)
    if not directory or not os.path.isdir(directory):
        return server.web.json_response({"ok": False, "error": "VRI did not resolve to a directory", "dir": directory or ""})
    return server.web.json_response({"ok": True, "dir": directory, "components": detect_components(directory)})


@server.PromptServer.instance.routes.get("/bat/vri-frame")
async def bat_vri_frame(request):
    q = request.rel_url.query
    vri = q.get("vri", "")
    component = q.get("component", "")
    try:
        frame = int(q.get("frame", "0"))
    except ValueError:
        frame = 0
    try:
        max_w = int(q.get("max_w", "256"))
    except ValueError:
        max_w = 256

    directory = vri_to_path(vri)
    if not directory or not os.path.isdir(directory) or not component:
        return server.web.Response(status=404, text="bad vri/component")
    try:
        arr = load_one_frame(directory, component, frame)
    except Exception as e:
        return server.web.Response(status=500, text=f"decode error: {e}")

    # EXR preview cells are unlikely, but tone-map just in case so they're visible.
    if component.lower().endswith(".exr"):
        arr = _linear_to_srgb(arr)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    if img.width > max_w:
        h = max(1, int(img.height * max_w / img.width))
        img = img.resize((max_w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return server.web.Response(body=buf.getvalue(), content_type="image/jpeg")
