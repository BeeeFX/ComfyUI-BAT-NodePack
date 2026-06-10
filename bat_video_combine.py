"""
Bat_VideoCombine — encode an IMAGE batch to a video file with an
interactive on-canvas player that supports ProRes preview, frame-accurate
scrubbing, frame-by-frame stepping, and a save-current-frame button.

The encoder side is ffmpeg-driven (subprocess.Popen, stdin-piped raw RGB
frames) and reads format definitions from `bat_video_formats/*.json`,
so adding a new codec is one JSON file. Each format declares whether
browsers can decode it natively; for those that can't (ProRes, FFV1,
h265, raw image sequences) the preview path automatically transcodes
through a lightweight H.264/MP4 endpoint so the on-canvas `<video>`
element always has something to play.

Sibling JS file: ``web/bat_video_combine.js`` owns the player UI.
"""

import io
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from PIL import Image

import folder_paths
import server
from aiohttp import web

logger = logging.getLogger("[Bat_VideoCombine]")

# ─── Format catalogue ───────────────────────────────────────────────────────

_FORMATS_DIR = os.path.join(os.path.dirname(__file__), "bat_video_formats")
_FORMATS: dict = {}


def _load_formats() -> dict:
    """Scan bat_video_formats/*.json once on first call, cache the result."""
    if _FORMATS:
        return _FORMATS
    if not os.path.isdir(_FORMATS_DIR):
        logger.warning(f"Format dir missing: {_FORMATS_DIR}")
        return _FORMATS
    for fn in sorted(os.listdir(_FORMATS_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(_FORMATS_DIR, fn), "r", encoding="utf-8") as f:
                data = json.load(f)
            label = data.get("label") or fn[:-5]
            _FORMATS[label] = data
        except Exception as exc:
            logger.warning(f"Could not load format {fn}: {exc}")
    return _FORMATS


def _format_choices() -> list:
    """Return the list of format labels for the COMBO widget."""
    return list(_load_formats().keys()) or ["video/h264-mp4"]


def _expand_widget_args(args: list, widget_values: dict) -> list:
    """Replace `{name}` placeholders in args with widget values.

    Two cases handled:
      - Whole-arg placeholder (`"{pix_fmt}"`) → substituted to the value.
        Skipped entirely when the value is None (lets a format declare an
        optional widget without leaving an empty arg behind).
      - Embedded placeholder (`"paletteuse=dither={dither}"`) → resolved
        via str.format_map so the surrounding string is preserved. The
        gif filter graph needs this.
    """
    str_vals = {k: ("" if v is None else str(v)) for k, v in widget_values.items()}
    out = []
    for a in args:
        if not isinstance(a, str):
            out.append(str(a))
            continue
        if a.startswith("{") and a.endswith("}") and a.count("{") == 1:
            key = a[1:-1]
            if widget_values.get(key) is None:
                continue
            out.append(str_vals[key])
        elif "{" in a and "}" in a:
            try:
                out.append(a.format_map(str_vals))
            except (KeyError, ValueError):
                out.append(a)
        else:
            out.append(a)
    return out


# ─── Helpers ────────────────────────────────────────────────────────────────


def _ffmpeg_bin() -> str:
    """Find ffmpeg. Prefer imageio_ffmpeg's bundled binary (already a hard
    dep of several ComfyUI custom packs); fall back to PATH."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _tensor_to_bytes(images: torch.Tensor) -> bytes:
    """Flatten an (N, H, W, 3 or 4) float tensor to packed uint8 RGB(A) bytes."""
    arr = images.detach().cpu().numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr.tobytes()


def _audio_to_pcm_path(audio: dict, temp_dir: str) -> Optional[str]:
    """Write ComfyUI's AUDIO dict (`waveform` tensor + `sample_rate`) to a
    16-bit PCM .wav so ffmpeg can mux it as the audio track. Returns the
    path, or None if `audio` is empty/missing."""
    if not audio or "waveform" not in audio:
        return None
    wave = audio["waveform"]
    if hasattr(wave, "detach"):
        wave = wave.detach().cpu().numpy()
    wave = np.asarray(wave)
    while wave.ndim > 2:
        wave = wave[0]
    if wave.ndim == 1:
        wave = wave[None, :]
    sr = int(audio.get("sample_rate", 44100))
    pcm = np.clip(wave * 32767.0, -32768, 32767).astype(np.int16).T  # (samples, channels)

    path = os.path.join(temp_dir, f"bat_audio_{int(time.time() * 1000)}.wav")
    ch = pcm.shape[1]
    n = pcm.shape[0]
    data_bytes = pcm.tobytes()
    byte_rate = sr * ch * 2
    block_align = ch * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data_bytes)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block_align, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)
    return path


# ─── Encoder ────────────────────────────────────────────────────────────────


def _encode(
    images: torch.Tensor,
    fmt: dict,
    widget_values: dict,
    frame_rate: float,
    output_path: str,
    audio_path: Optional[str],
    loop_count: int,
    pingpong: bool,
) -> None:
    """Pipe raw frames through ffmpeg to produce a video at output_path."""
    n, h, w, c = images.shape
    if c not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA tensor, got {c} channels.")

    if pingpong and n > 2:
        # Append reversed middle frames so the sequence palindromes.
        rev = torch.flip(images[1:-1], dims=[0])
        images = torch.cat([images, rev], dim=0)
        n = images.shape[0]

    input_pix = "rgba" if c == 4 else "rgb24"

    ffmpeg = _ffmpeg_bin()
    args = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", input_pix,
        "-s", f"{w}x{h}",
        "-r", f"{frame_rate}",
        "-i", "-",                                # raw frames on stdin
    ]
    if audio_path:
        args += ["-i", audio_path]

    args += _expand_widget_args(fmt.get("video_args", []), widget_values)
    audio_args = fmt.get("audio_args")
    if audio_path and audio_args:
        args += _expand_widget_args(audio_args, widget_values)
        args += ["-shortest"]
    elif not audio_path:
        args += ["-an"]

    # GIFs and WebP use the format's own loop arg; for normal videos we
    # respect loop_count by post-processing only the saved file's metadata
    # (browser <video loop> is the real loop; loop_count > 0 here is a
    # no-op pending demand — VHS has the same caveat).
    if loop_count and fmt.get("extension") in ("gif", "webp"):
        # Last "-loop" in args wins; append override if user wants finite loops.
        args += ["-loop", str(int(loop_count))]

    args.append(output_path)

    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        proc.stdin.write(_tensor_to_bytes(images))
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (rc={rc}):\n{err}")
        if err.strip():
            logger.debug(f"ffmpeg stderr: {err.strip()[:500]}")
    except BrokenPipeError:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg pipe broke:\n{err}")


# ─── The node ───────────────────────────────────────────────────────────────


class BatVideoCombine:
    """Encode an IMAGE batch to a video file and preview it on the node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "frame_rate":      ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "loop_count":      ("INT",   {"default": 0, "min": 0, "max": 100}),
                "filename_prefix": ("STRING", {"default": "BatVideo"}),
                # H264 is the studio default — broad player support, sane
                # quality/size trade-off, hardware-decoded on every artist
                # box. The legacy implicit-first-alphabetical default was
                # `av1-webm`, which is now gone from the formats dir.
                "format":          (_format_choices(), {"default": "video/h264-mp4"}),
                "pingpong":        ("BOOLEAN", {"default": False}),
                "save_output":     ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # Plain STRING return — the absolute path of the saved file. Chains
    # into anything that accepts a path (re-loaders, uploaders, prompt
    # builders) without locking us to any particular sibling pack.
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = "BAT/video"
    DESCRIPTION = (
        "Encode an IMAGE batch to a video file (h264, ProRes, webm, gif, "
        "EXR sequence, …) with a frame-accurate canvas preview. ProRes "
        "and other non-web-playable formats are transcoded for the preview "
        "automatically — no setting to flip."
    )

    def combine(self, images, frame_rate, loop_count, filename_prefix, format,
                pingpong, save_output, audio=None, prompt=None, extra_pnginfo=None,
                **format_widget_kwargs):
        fmt = _load_formats().get(format)
        if fmt is None:
            raise RuntimeError(f"Unknown format {format!r}")

        # Resolve format widget values from their JSON defaults. We don't
        # expose every codec knob on the node UI itself (would mean a
        # dozen widgets that nobody touches); instead defaults from the
        # JSON win unless the caller passes overrides explicitly. The JS
        # widget on the node body lets the user pick a small, curated
        # subset (currently: just the format COMBO).
        widget_values = {}
        for name, spec in (fmt.get("widgets") or {}).items():
            widget_values[name] = format_widget_kwargs.get(name, spec.get("default"))

        # Output path under output/ or temp/ depending on save_output.
        if save_output:
            output_dir = folder_paths.get_output_directory()
            preview_type = "output"
        else:
            output_dir = folder_paths.get_temp_directory()
            preview_type = "temp"
        full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images.shape[2], images.shape[1],
        )
        os.makedirs(full_dir, exist_ok=True)

        ext = fmt["extension"]
        # Image-sequence formats use printf-style extensions (`%05d.png`);
        # land them in a subfolder so a hundred PNGs don't clutter output/.
        if "%" in ext:
            seq_dir = os.path.join(full_dir, f"{base}_{counter:05d}")
            os.makedirs(seq_dir, exist_ok=True)
            output_path = os.path.join(seq_dir, ext)
            preview_filename = f"{base}_{counter:05d}/{ext.replace('%', '%%')}"
            preview_subfolder = os.path.relpath(seq_dir, output_dir)
        else:
            output_path = os.path.join(full_dir, f"{base}_{counter:05d}.{ext}")
            preview_filename = os.path.basename(output_path)
            preview_subfolder = subfolder

        # Audio side car (optional).
        audio_path = None
        if audio is not None and fmt.get("audio_args"):
            audio_path = _audio_to_pcm_path(audio, folder_paths.get_temp_directory())

        try:
            _encode(
                images=images,
                fmt=fmt,
                widget_values=widget_values,
                frame_rate=float(frame_rate),
                output_path=output_path,
                audio_path=audio_path,
                loop_count=int(loop_count),
                pingpong=bool(pingpong),
            )
        finally:
            if audio_path and os.path.isfile(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        n_frames = int(images.shape[0]) * (2 if pingpong and images.shape[0] > 2 else 1)
        if pingpong and images.shape[0] > 2:
            n_frames = images.shape[0] * 2 - 2

        preview = {
            "filename": preview_filename,
            "subfolder": preview_subfolder,
            "type": preview_type,
            "format": fmt.get("label", format),
            "frame_rate": float(frame_rate),
            "frame_count": n_frames,
            "browser_playable": bool(fmt.get("browser_playable", True)),
            "fullpath": output_path,
        }

        # `gifs` is the conventional ComfyUI `ui` key for video previews —
        # the canvas player picks it up from `onExecuted(message)`.
        return {
            "ui": {"gifs": [preview]},
            "result": (output_path,),
        }


# ─── API routes ─────────────────────────────────────────────────────────────
#
# All under /bat/video/* so they're easy to grep and don't collide with
# the existing /bat/* endpoints from bat_video_loader.py.

_PREVIEW_CACHE_DIR_NAME = "bat_video_preview_cache"
_PREVIEW_CACHE_MAX = 32


def _safe_under(root: str, *parts: str) -> Optional[str]:
    """Join under root, refuse if the join escapes root via traversal."""
    if not root:
        return None
    candidate = os.path.realpath(os.path.join(root, *(p or "" for p in parts)))
    root_real = os.path.realpath(root)
    try:
        common = os.path.commonpath([root_real, candidate])
    except ValueError:
        return None
    if common != root_real:
        return None
    return candidate


def _resolve_request_path(request) -> Optional[str]:
    """Resolve query (filename, type, subfolder) into an absolute path under
    output/ or temp/. Refuses anything else. Used by every API route below."""
    q = request.rel_url.query
    filename = q.get("filename") or q.get("file") or ""
    typ = q.get("type", "output")
    subfolder = q.get("subfolder", "") or ""
    if not filename:
        return None
    if typ == "output":
        root = folder_paths.get_output_directory()
    elif typ == "temp":
        root = folder_paths.get_temp_directory()
    else:
        return None
    if subfolder:
        path = _safe_under(root, subfolder, filename)
    else:
        path = _safe_under(root, filename)
    if path is None or not os.path.isfile(path):
        return None
    return path


@server.PromptServer.instance.routes.get("/bat/video/meta")
async def bat_video_meta(request):
    """Probe duration / fps / frame count / dimensions / audio flag with
    ffprobe. Used by the canvas player to size the scrubber and to label
    frames. Falls back gracefully if ffprobe is unavailable."""
    path = _resolve_request_path(request)
    if path is None:
        return web.json_response({"error": "Not found"}, status=404)
    try:
        out = subprocess.run(
            [_ffprobe_bin(), "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", path],
            capture_output=True, timeout=10,
        )
        info = json.loads(out.stdout.decode("utf-8", errors="replace") or "{}")
    except Exception as exc:
        return web.json_response({"error": f"ffprobe failed: {exc}"}, status=500)

    streams = info.get("streams") or []
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt_info = info.get("format") or {}

    duration = float(fmt_info.get("duration") or 0)
    fps = 0.0
    frame_count = 0
    width = height = 0
    if vstream:
        width = int(vstream.get("width") or 0)
        height = int(vstream.get("height") or 0)
        rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/1"
        try:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            fps = 0.0
        frame_count = int(vstream.get("nb_frames") or 0) or int(round(duration * fps))

    return web.json_response({
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "has_audio": astream is not None,
    })


@server.PromptServer.instance.routes.get("/bat/video/preview")
async def bat_video_preview(request):
    """On-demand transcode to H.264 MP4 for browser playback.

    Used for ProRes, FFV1, h265, and anything else flagged
    `browser_playable=false` in its format JSON. The transcoded file is
    cached under temp/bat_video_preview_cache so a node redraw doesn't
    re-encode every time. Cache key: source mtime + size, so editing the
    source invalidates the cache automatically.
    """
    path = _resolve_request_path(request)
    if path is None:
        return web.Response(status=404)

    cache_dir = os.path.join(folder_paths.get_temp_directory(), _PREVIEW_CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        st = os.stat(path)
        key = f"{abs(hash(path))}_{int(st.st_mtime)}_{st.st_size}.mp4"
    except OSError:
        return web.Response(status=404)
    cached = os.path.join(cache_dir, key)

    if not os.path.isfile(cached):
        try:
            subprocess.run([
                _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "warning",
                "-i", path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                cached,
            ], check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            return web.json_response({"error": "transcode failed", "stderr": stderr[-2000:]}, status=500)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

        # Trim the cache loosely. Drops the oldest files by mtime once we
        # exceed _PREVIEW_CACHE_MAX. Cheap to do on every miss.
        try:
            entries = sorted(
                (os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".mp4")),
                key=lambda p: os.path.getmtime(p),
            )
            for old in entries[:-_PREVIEW_CACHE_MAX]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except Exception:
            pass

    return web.FileResponse(cached, headers={"Content-Type": "video/mp4"})


@server.PromptServer.instance.routes.get("/bat/video/frame")
async def bat_video_frame(request):
    """Return a single frame as PNG. Used both for the save-current-frame
    button (frontend re-POSTs to /save_frame to land it under output/) and
    for the hover-scrub thumbnail (frontend asks for many small frames)."""
    path = _resolve_request_path(request)
    if path is None:
        return web.Response(status=404)
    try:
        frame = int(request.rel_url.query.get("frame", "0"))
    except ValueError:
        return web.Response(status=400)
    frame = max(0, frame)

    # `select=eq(n,N)` jumps to the Nth decoded frame. Add `-vsync 0` so
    # ffmpeg doesn't pad or drop around the target frame.
    try:
        out = subprocess.run([
            _ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-vf", f"select=eq(n\\,{frame}),scale=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vsync", "0", "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ], capture_output=True, check=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        return web.json_response({"error": "frame extract failed", "stderr": stderr[-500:]}, status=500)
    return web.Response(body=out.stdout, headers={"Content-Type": "image/png"})


@server.PromptServer.instance.routes.post("/bat/video/save_frame")
async def bat_video_save_frame(request):
    """Save a single frame from a saved video as a PNG under output/.

    Body: { filename, type, subfolder, frame, dest_prefix }.
    Server-authoritative — no base64 round-trip through the browser, and
    file lands under output/ where every artist can see it without a
    download dialog.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    # Reuse _resolve_request_path's logic by smuggling values via a fake
    # request-like with rel_url.query. Simpler: replicate the small bit.
    typ = (body.get("type") or "output").strip()
    subfolder = (body.get("subfolder") or "").strip()
    filename = (body.get("filename") or "").strip()
    frame = int(body.get("frame") or 0)
    dest_prefix = _safe_name((body.get("dest_prefix") or "framegrab").strip()) or "framegrab"
    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    src_root = (folder_paths.get_output_directory() if typ == "output"
                else folder_paths.get_temp_directory())
    src_path = _safe_under(src_root, subfolder, filename) if subfolder else _safe_under(src_root, filename)
    if src_path is None or not os.path.isfile(src_path):
        return web.json_response({"error": "source not found"}, status=404)

    out_dir = folder_paths.get_output_directory()
    full_dir, base, counter, sub, _ = folder_paths.get_save_image_path(
        dest_prefix, out_dir, 1, 1,
    )
    os.makedirs(full_dir, exist_ok=True)
    out_name = f"{base}_{counter:05d}_f{frame:06d}.png"
    out_path = os.path.join(full_dir, out_name)

    try:
        subprocess.run([
            _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", src_path,
            "-vf", f"select=eq(n\\,{frame})",
            "-vsync", "0", "-frames:v", "1",
            out_path,
        ], capture_output=True, check=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        return web.json_response({"error": "frame extract failed", "stderr": stderr[-500:]}, status=500)

    return web.json_response({
        "filename": out_name,
        "subfolder": sub,
        "type": "output",
    })


# ─── Registration ───────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {"Bat_VideoCombine": BatVideoCombine}
NODE_DISPLAY_NAME_MAPPINGS = {"Bat_VideoCombine": "🦇 Video Combine"}
