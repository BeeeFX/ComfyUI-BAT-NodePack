"""
Bat_VideoLoader — load a trimmed range of frames from a video file.

The user types a path (with directory autocomplete provided by the
``/bat/getpath`` route) and uses a dual-handle range slider on the node
to pick a start/end frame. The
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
import re
import shutil
import subprocess
from collections import OrderedDict
from typing import Optional, Tuple

# Sentinel for "not worked out yet", where None is a real answer (no ffmpeg on
# this machine; a clip with no audio track).
_UNSET = object()

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
_FRAME_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
# Entry-count cap is kept as a secondary guard against pathological tiny entries.
_FRAME_CACHE_MAX = 8
_FRAME_CACHE_MAX_BYTES = 4 * 1024 ** 3   # 4 GiB


def _tensor_nbytes(t) -> int:
    if t is None:
        return 0
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


def _entry_nbytes(entry) -> int:
    """Bytes held by one cache entry — images, mask and any audio waveform."""
    if not isinstance(entry, dict):
        return _tensor_nbytes(entry)
    total = _tensor_nbytes(entry.get("images")) + _tensor_nbytes(entry.get("mask"))
    audio = entry.get("audio")
    if isinstance(audio, dict):
        total += _tensor_nbytes(audio.get("waveform"))
    return total


def _trim_frame_cache():
    """Evict oldest entries until both the byte budget and the entry cap hold."""
    total = sum(_entry_nbytes(v) for v in _FRAME_CACHE.values())
    while _FRAME_CACHE and (total > _FRAME_CACHE_MAX_BYTES
                            or len(_FRAME_CACHE) > _FRAME_CACHE_MAX):
        _key, victim = _FRAME_CACHE.popitem(last=False)
        total -= _entry_nbytes(victim)

logger = logging.getLogger(__name__)

# Backends already reported as unavailable. decord and torchvision are both
# optional, so on a plain install one of them raises ImportError on *every*
# frame fetch — and the thumbnail routes fire on every keystroke in the path
# field. Warn once per backend, at debug level thereafter.
_MISSING_BACKENDS = set()


def _backend_failed(backend: str, exc: Exception):
    if isinstance(exc, ImportError):
        if backend not in _MISSING_BACKENDS:
            _MISSING_BACKENDS.add(backend)
            logger.info(f"[Bat_VideoLoader] {backend} not installed — "
                        f"skipping that backend")
        else:
            logger.debug(f"[Bat_VideoLoader] {backend} not installed")
        return
    logger.warning(f"[Bat_VideoLoader] {backend} failed: {exc}")

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



# ─── ffmpeg backend: high bit depth, alpha, audio ───────────────────────────
#
# decord / OpenCV / torchvision all hand back 8-bit RGB, which is fine for an
# ordinary mp4 and wrong for the two things a compositor actually brings to a
# graph: a 10- or 12-bit ProRes, and a clip with an alpha channel. Both are
# decoded here instead, straight from ffmpeg's raw output — 16-bit RGBA for a
# high-depth source, 8-bit RGBA for an 8-bit one with alpha.


def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary, or None.

    imageio_ffmpeg ships one and ComfyUI already depends on it, so this
    normally resolves without anything being installed system-wide.
    """
    global _FFMPEG_EXE
    if _FFMPEG_EXE is _UNSET:
        exe = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = shutil.which("ffmpeg")
        _FFMPEG_EXE = exe
        if not exe:
            logger.info("[Bat_VideoLoader] no ffmpeg binary found — "
                        "high-bit-depth and alpha sources will load as 8-bit "
                        "RGB, and audio is unavailable")
    return _FFMPEG_EXE


# Pixel-format name fragments that mean "carries a real alpha channel". Note
# what is NOT here: rgb0 / bgr0 / 0rgb name a padding byte, not alpha, and
# treating them as alpha would produce a garbage mask on ordinary footage.
_ALPHA_TOKENS = ("yuva", "rgba", "bgra", "argb", "abgr", "gbrap", "ya8",
                 "ya16", "ayuv", "yuvap")

# ...and fragments that mean "more than 8 bits per channel". A token list
# rather than parsing the name: the cost of missing one is an 8-bit decode of
# something that could have been 16 (what every other backend does anyway),
# whereas a mis-parse would reshape the raw pipe wrongly and produce garbage.
_HIGH_DEPTH_TOKENS = ("p10", "p12", "p14", "p16", "rgb48", "bgr48", "rgba64",
                      "bgra64", "gbrp10", "gbrp12", "gbrp14", "gbrp16",
                      "gray10", "gray12", "gray14", "gray16", "f32", "f16",
                      "p010", "p210", "p410", "xyz12")

_FFMPEG_EXE = _UNSET
_PROBE_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PROBE_CACHE_MAX = 64


def _parse_ffmpeg_banner(info: str) -> dict:
    """Pull what we need out of `ffmpeg -i` stderr.

    ffprobe is not part of the imageio_ffmpeg download, so it is often absent
    on a bare venv; the banner ffmpeg prints for its own input is always
    available and carries everything here.
    """
    out = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0,
           "pix_fmt": "", "has_alpha": False, "bit_depth": 8,
           "has_audio": False, "codec": ""}

    dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if dur:
        out["duration"] = (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60
                           + float(dur.group(3)))

    for line in info.split("\n"):
        if "Audio:" in line and "Stream #" in line:
            out["has_audio"] = True
            continue
        if "Video:" not in line or "Stream #" not in line or out["width"]:
            continue
        codec = re.search(r"Video:\s*([\w.-]+)", line)
        if codec:
            out["codec"] = codec.group(1)
        # The pixel format is the token after the codec (and any profile in
        # parentheses), and it is the only comma-separated field that names a
        # known pixel layout — so match it by shape rather than by position,
        # which moves about between ffmpeg versions.
        for field in line.split(",")[1:]:
            token = field.strip().split(" ")[0].split("(")[0].lower()
            if re.fullmatch(r"[a-z0-9]*(yuv|rgb|bgr|gbr|gray|ya|nv|pal|xyz|p0|p2|p4)[a-z0-9]*",
                            token) and token not in ("progressive",):
                out["pix_fmt"] = token
                break
        dim = re.search(r"\b(\d{2,5})x(\d{2,5})\b", line)
        if dim:
            out["width"], out["height"] = int(dim.group(1)), int(dim.group(2))
        fps = re.search(r"([\d.]+)\s*fps", line)
        if fps:
            try:
                out["fps"] = float(fps.group(1))
            except ValueError:
                pass

    pf = out["pix_fmt"]
    out["has_alpha"] = any(t in pf for t in _ALPHA_TOKENS)
    out["bit_depth"] = 16 if any(t in pf for t in _HIGH_DEPTH_TOKENS) else 8
    return out


def probe_ffmpeg(path: str) -> Optional[dict]:
    """Container/stream facts from ffmpeg, or None when ffmpeg can't read it.

    Cached per (path, mtime_ns:size): the preview routes call this on every
    keystroke in the path field, and it spawns a process.
    """
    exe = ffmpeg_exe()
    if not exe:
        return None
    key = (os.path.abspath(path), _fingerprint(path))
    hit = _PROBE_CACHE.get(key)
    if hit is not None:
        _PROBE_CACHE.move_to_end(key)
        return hit
    try:
        res = subprocess.run([exe, "-hide_banner", "-i", path],
                             capture_output=True, timeout=30)
        banner = res.stderr.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"[Bat_VideoLoader] ffmpeg probe failed for {path!r}: {e}")
        return None
    info = _parse_ffmpeg_banner(banner)
    if not info["width"] or not info["height"]:
        return None
    _PROBE_CACHE[key] = info
    while len(_PROBE_CACHE) > _PROBE_CACHE_MAX:
        _PROBE_CACHE.popitem(last=False)
    return info


def _select_expr(start: int, end: int, stride: int) -> str:
    """An ffmpeg `select` expression for frames [start..end] every `stride`-th.

    Applied as a filter so the frames we discard are never pushed through the
    pipe, and (for the leading skip) never fully decoded.
    """
    expr = f"gte(n\\,{start})*lte(n\\,{end})"
    if stride > 1:
        expr += f"*eq(mod(n-{start}\\,{stride})\\,0)"
    return f"select='{expr}'"


def _iter_ffmpeg_rgba(path: str, start: int, end: int, stride: int,
                      probe: dict):
    """Yield RGBA frames from ffmpeg as (H,W,4) uint8 or uint16 arrays.

    Raises on a decoder failure rather than returning None: by the time we get
    here the probe has already said this source needs ffmpeg, so falling back
    silently to an 8-bit RGB backend would quietly drop the precision or the
    alpha the caller asked for.
    """
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg is required for this source but wasn't found")

    deep = probe["bit_depth"] > 8
    pix_out = "rgba64le" if deep else "rgba"
    dtype = np.dtype("<u2") if deep else np.dtype(np.uint8)
    w, h = probe["width"], probe["height"]
    stride_bytes = w * h * 4 * dtype.itemsize
    wanted = len(range(start, end + 1, stride))

    args = [exe, "-hide_banner", "-v", "error", "-an", "-i", path,
            "-vf", _select_expr(start, end, stride),
            "-pix_fmt", pix_out, "-frames:v", str(wanted),
            "-f", "rawvideo", "-fps_mode", "passthrough", "-"]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        while True:
            # readinto a fresh bytearray rather than read(): `bytes` is
            # immutable, so a tensor built on it triggers PyTorch's
            # non-writable-array warning, and the usual fix (wrapping in
            # bytearray) copies the whole frame a second time. A fresh buffer
            # per frame is the same single allocation read() would have made,
            # and the frames stay independent of each other.
            raw = bytearray(stride_bytes)
            view = memoryview(raw)
            got = 0
            while got < stride_bytes:
                # A pipe read can come back short of what was asked for.
                chunk = proc.stdout.readinto(view[got:])
                if not chunk:
                    break
                got += chunk
            if got < stride_bytes:
                break        # clean end of stream, or a truncated tail frame
            yield np.frombuffer(raw, dtype=dtype).reshape(h, w, 4)
    finally:
        # Covers the normal end, an interrupt, and the consumer stopping early
        # (a cache-key mismatch, a size error). Without the kill, ffmpeg sits
        # blocked on a pipe nobody is reading for the life of the process.
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        err = b""
        try:
            err = proc.stderr.read() or b""
            proc.stderr.close()
        except Exception:
            pass
        proc.wait()
        if err.strip():
            logger.debug("[Bat_VideoLoader] ffmpeg: %s",
                         err.decode("utf-8", "replace").strip())


def load_audio(path: str, start_time: float = 0.0,
               duration: float = 0.0) -> Optional[dict]:
    """The AUDIO dict ComfyUI expects, or None when there's no audio to take.

    `-ss` goes before `-i` so ffmpeg seeks instead of decoding-and-discarding
    the head of a long clip.
    """
    exe = ffmpeg_exe()
    if not exe:
        return None
    probe = probe_ffmpeg(path)
    if probe is not None and not probe["has_audio"]:
        return None

    args = [exe, "-hide_banner", "-v", "info"]
    if start_time > 0:
        args += ["-ss", f"{start_time:.6f}"]
    args += ["-i", path]
    if duration > 0:
        args += ["-t", f"{duration:.6f}"]
    args += ["-vn", "-f", "f32le", "-"]
    try:
        res = subprocess.run(args, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"[Bat_VideoLoader] audio extraction failed: {e}")
        return None
    if not res.stdout:
        return None

    banner = res.stderr.decode("utf-8", "replace")
    rate, channels = _parse_audio_layout(banner)
    samples = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
    if channels > 1 and samples.numel() % channels:
        # A layout we misread would silently interleave the channels wrongly;
        # mono is the one reading that cannot scramble the samples.
        logger.warning("[Bat_VideoLoader] audio sample count %d doesn't divide "
                       "by %d channels — loading as mono",
                       samples.numel(), channels)
        channels = 1
    waveform = samples.reshape(-1, channels).transpose(0, 1).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": rate}


# "48000 Hz, stereo, fltp" — the layout word, or "N channels" for anything
# without a name.
_AUDIO_LAYOUTS = {"mono": 1, "stereo": 2, "2.1": 3, "3.0": 3, "4.0": 4,
                  "quad": 4, "5.0": 5, "5.1": 6, "6.1": 7, "7.1": 8,
                  "downmix": 2}


def _parse_audio_layout(banner: str) -> Tuple[int, int]:
    """(sample_rate, channels) from an ffmpeg banner, with safe defaults."""
    rate, channels = 44100, 2
    m = re.search(r"Audio:.*?(\d+)\s*Hz,\s*([^,]+),", banner)
    if not m:
        return rate, channels
    rate = int(m.group(1))
    layout = m.group(2).strip().lower()
    n = re.match(r"(\d+)\s*channels?", layout)
    if n:
        return rate, max(1, int(n.group(1)))
    # "5.1(side)" and friends.
    return rate, _AUDIO_LAYOUTS.get(layout.split("(")[0].strip(), 2)


# ─── Progress / cancellation ────────────────────────────────────────────────
#
# Both are imported lazily and failure-tolerantly: comfy is always there in a
# real ComfyUI process, but this module is also driven by the standalone tests
# in tests/, where it isn't.


def _progress_bar(total: int):
    """ComfyUI's per-node progress bar, or None. Skipped for a single frame."""
    if total <= 1:
        return None
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _raise_if_interrupted():
    """Raise comfy's InterruptProcessingException if the user hit Cancel.

    "Cancel current run" only sets a flag, and nothing checks it unless we do —
    ProgressBar.update does not. Without this poll a 3000-frame load runs to
    completion after the artist has already given up on it.
    """
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except Exception:
        return
    throw_exception_if_processing_interrupted()



# ─── Frame extraction ───────────────────────────────────────────────────────


# How many frames to ask decord for at a time. One get_batch for the whole
# clip is marginally faster but allocates the entire batch inside decord before
# we see a single frame, so neither the progress bar nor Cancel can do anything
# until it finishes.
_DECORD_CHUNK = 32


def _frames_decord(path: str, start: int, end: int, stride: int = 1,
                   progress=None) -> Optional[np.ndarray]:
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
        out = None
        done = 0
        for i in range(0, len(indices), _DECORD_CHUNK):
            chunk = vr.get_batch(indices[i:i + _DECORD_CHUNK]).asnumpy()
            if out is None:
                out = np.empty((len(indices),) + chunk.shape[1:],
                               dtype=chunk.dtype)
            out[i:i + len(chunk)] = chunk
            done += len(chunk)
            del chunk
            if progress:
                progress(done)
        return out
    except Exception as e:
        _backend_failed("decord", e)
        return None


def _frames_cv2(path: str, start: int, end: int, stride: int = 1,
                progress=None) -> Optional[np.ndarray]:
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
        wanted = len(range(s, e + 1, st))
        # Preallocated and written in place. The old `out.append(...)` plus
        # `np.stack` held every frame AND the finished array at once, so the
        # uint8 peak was twice the clip.
        out = None
        kept = 0
        # grab() advances without decoding/converting; read() only decodes the
        # frames we actually keep. Skipping the colour conversion for discarded
        # frames is where the stride saving comes from.
        for i in range(s, e + 1):
            if (i - s) % st == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if out is None:
                    out = np.empty((wanted,) + rgb.shape, dtype=rgb.dtype)
                elif rgb.shape != out.shape[1:]:
                    # A mid-clip resolution change (concatenated sources).
                    # Stop at the last frame that fits rather than throwing
                    # away the ones already decoded.
                    logger.warning("[Bat_VideoLoader] frame %d is %dx%d where "
                                   "the clip started at %dx%d — stopping there",
                                   i, rgb.shape[1], rgb.shape[0],
                                   out.shape[2], out.shape[1])
                    break
                out[kept] = rgb
                kept += 1
                del frame, rgb
                if progress:
                    progress(kept)
            else:
                if not cap.grab():
                    break
        cap.release()
        if out is None or kept == 0:
            return None
        return out[:kept]
    except Exception as e:
        _backend_failed("cv2", e)
        return None


def _frames_torchvision(path: str, start: int, end: int, stride: int = 1,
                        progress=None) -> Optional[np.ndarray]:
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
        # slice here — no decode saving available on this backend, and no
        # partial progress to report either.
        out = video[s:e + 1:max(1, int(stride))].numpy()
        if progress:
            progress(len(out))
        return out
    except Exception as e:
        _backend_failed("torchvision", e)
        return None


def load_frames(path: str, start: int, end: int, stride: int = 1,
                progress=None) -> np.ndarray:
    """Return frames [start..end] (inclusive), every `stride`-th, as uint8 (N,H,W,3).

    The stride is pushed DOWN into the decoder so we never decode frames that
    are about to be discarded (see _frames_decord / _frames_cv2).

    `progress` is called with the running frame count as they are decoded. It
    may raise to abort the load — that is how cancellation gets out of a
    backend mid-clip.

    8-bit RGB, whatever the source. Use load_batch() for a load that keeps a
    high-bit-depth source's precision and a clip's alpha channel.
    """
    for fn in (_frames_decord, _frames_cv2, _frames_torchvision):
        out = fn(path, start, end, stride, progress)
        if out is not None and len(out) > 0:
            return out
    raise RuntimeError(
        f"No working video decoder produced frames for {path!r}. "
        f"Install one of: decord, opencv-python, or torchvision with video support."
    )


def load_batch(path: str, start: int, end: int, stride: int = 1,
               want_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Decode frames [start..end] every `stride`-th into ComfyUI tensors.

    Returns (images, mask, meta):

      images  float32 (N,H,W,3), 0..1
      mask    float32 (N,H,W) — ComfyUI's convention is 1-alpha, so a fully
              opaque source (which is most of them) gives zeros
      meta    {"backend", "bit_depth", "has_alpha", "fps"}

    A source that ffmpeg reports as >8-bit or as carrying alpha is decoded by
    ffmpeg, because every other backend here returns 8-bit RGB and would throw
    away exactly what makes those sources worth loading. Everything else takes
    the faster 8-bit path.

    Frames are written into one preallocated batch as they arrive, so the peak
    is the batch plus a frame or two rather than the batch twice.
    """
    stride = max(1, int(stride))
    indices = range(start, end + 1, stride)
    wanted = len(indices)
    if wanted <= 0:
        raise ValueError(f"Bat_VideoLoader: empty frame range {start}..{end}")

    probe = probe_ffmpeg(path)
    deep_or_alpha = probe is not None and (probe["bit_depth"] > 8
                                           or probe["has_alpha"])
    meta = {"backend": "", "bit_depth": 8, "has_alpha": False,
            "fps": (probe or {}).get("fps", 0.0)}

    pbar = _progress_bar(wanted) if want_progress else None
    seen = 0

    def report(done):
        # ProgressBar.update takes an increment, and it does NOT check the
        # interrupt flag, so the poll has to be here too.
        nonlocal seen
        if pbar is not None and done > seen:
            pbar.update(done - seen)
        seen = done
        _raise_if_interrupted()

    if not deep_or_alpha:
        frames = load_frames(path, start, end, stride, progress=report)
        src = torch.from_numpy(np.ascontiguousarray(frames))
        del frames
        images = torch.empty(src.shape, dtype=torch.float32)
        torch.div(src, 255.0, out=images)
        del src
        mask = torch.zeros(images.shape[:3], dtype=torch.float32)
        meta["backend"] = "8-bit"
        return images, mask, meta

    meta.update(backend="ffmpeg", bit_depth=probe["bit_depth"],
                has_alpha=probe["has_alpha"])
    scale = 65535.0 if probe["bit_depth"] > 8 else 255.0
    images = mask = None
    kept = 0
    for frame in _iter_ffmpeg_rgba(path, start, end, stride, probe):
        if kept >= wanted:
            break
        if images is None:
            h, w = frame.shape[0], frame.shape[1]
            images = torch.empty((wanted, h, w, 3), dtype=torch.float32)
            mask = torch.empty((wanted, h, w), dtype=torch.float32)
        src = torch.from_numpy(np.ascontiguousarray(frame))
        torch.div(src[:, :, :3], scale, out=images[kept])
        if probe["has_alpha"]:
            torch.div(src[:, :, 3], scale, out=mask[kept])
            # 1-alpha, in place: opaque reads as 0.
            mask[kept].neg_().add_(1.0)
        else:
            # rgba64le pads alpha to opaque on a source that has none; writing
            # 0 is the same answer without pretending the alpha was real.
            mask[kept].zero_()
        del src
        kept += 1
        report(kept)

    if images is None or kept == 0:
        raise RuntimeError(f"Bat_VideoLoader: ffmpeg decoded no frames from "
                           f"{path!r} (pixel format {probe['pix_fmt']!r})")
    if kept < wanted:
        # Short clip, or a frame count the container lied about. Keep what
        # arrived rather than handing back a batch padded with empty frames.
        logger.info("[Bat_VideoLoader] %d of %d frames decoded from %s",
                    kept, wanted, os.path.basename(path))
        images, mask = images[:kept], mask[:kept]
    return images, mask, meta


def _single_frame(path: str, index: int) -> Optional[np.ndarray]:
    arr = load_frames(path, index, index)
    return arr[0] if arr is not None and len(arr) else None


# ─── Path safety ────────────────────────────────────────────────────────────


def _strip_path(p: str) -> str:
    return p.strip().strip('"').strip("'")


def _is_video(name: str) -> bool:
    return name.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS


def _is_safe_path(path: str) -> bool:
    """Opt-in confinement of the browse/preview routes to the ComfyUI tree.

    Off by default, because an artist browses wherever the footage lives
    (a job mount is nowhere near the install), so confining it out of the box
    would break the node for its actual users. Set BAT_STRICT_PATHS=1 on a
    ComfyUI that is reachable by anyone you don't trust.
    """
    if not os.environ.get("BAT_STRICT_PATHS"):
        return True
    base = os.path.abspath(".")
    try:
        return os.path.commonpath([base, os.path.abspath(path)]) == base
    except ValueError:      # different drive on Windows
        return False


def _resolve_media(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Clean and validate a `path` query param for the preview routes.

    Returns (path, None) or (None, reason).

    The extension allowlist is the load-bearing part: /bat/video-stream hands
    the file straight to FileResponse, so without it any ComfyUI reachable
    over a network served *any* file on disk to anyone who could name it. This
    pack ships publicly, so the routes have to be safe on a default install —
    the strict-path check above is opt-in, this one is not.
    """
    path = _strip_path(raw or "")
    if not path:
        return None, "no path"
    if not _is_video(path):
        return None, "not a video file"
    if not os.path.isfile(path):
        return None, "not a file"
    if not _is_safe_path(path):
        return None, "path not permitted"
    return path, None


def _fingerprint(path: str) -> str:
    """(mtime_ns, size) for `path`, or "missing:<path>" when it can't be stat'd.

    st_mtime_ns + st_size rather than a float getmtime(): on a network mount
    the second-granularity timestamp, plus clock skew between a render node
    and a workstation, can leave a rewritten file on an identical mtime. Size
    moves whenever content does, so the pair catches the same-timestamp
    rewrite that mtime alone misses.

    The failure string carries the path so a missing file still varies per
    input rather than collapsing to one constant shared by every broken path.
    """
    try:
        st = os.stat(path)
    except OSError:
        return f"missing:{path}"
    return f"{st.st_mtime_ns}:{st.st_size}"


# ─── ComfyUI node ───────────────────────────────────────────────────────────

# Output slot of the AUDIO return, for the connection check below.
_AUDIO_SLOT = 2


def _output_is_consumed(prompt, unique_id, slot) -> bool:
    """Whether anything in this prompt reads output `slot` of this node.

    Pulling the audio out of a clip is a second full pass over the file with
    ffmpeg, and on an ordinary image load nothing wants it — the graph takes
    `images` and nothing else. So look at what the prompt actually wires up: an
    input link in a ComfyUI prompt is `[source_node_id, source_slot]`, so a
    link from our slot 2 is `[our_id, 2]`.

    Returns True whenever we can't tell — no prompt (an API submit that didn't
    send one, or a test), an unparseable prompt, our own id missing. Extracting
    audio nobody reads costs a second; handing None to something that wanted it
    breaks the graph. So uncertainty falls the expensive way.
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
                    and str(value[0]) == me and value[1] == slot):
                return True
    return False


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
            },
            "hidden": {
                # Only used to see whether the `audio` output is connected —
                # see _output_is_consumed().
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "mask", "audio", "frame_rate")
    OUTPUT_TOOLTIPS = (
        "The trimmed frames. 10- and 12-bit sources (ProRes, FFV1) keep their "
        "precision; everything else loads as 8-bit.",
        "1 - alpha, so a clip without an alpha channel gives an empty mask.",
        "The audio under the trimmed range. Only extracted when this output "
        "is connected.",
        "Frames per second of the LOADED batch — the source rate divided by "
        "select_every_nth. Feed it to 🦇 Video Combine and the output keeps "
        "the source's timing.",
    )
    CATEGORY = "BAT/Video"
    DESCRIPTION = (
        "Load a trimmed range of frames from a video file. Type a path (with "
        "directory autocomplete) and set the range with a dual-handle trim "
        "slider showing thumbnails of the in/out frames. Outputs the frames, "
        "an alpha mask, the audio under the range, and the frame rate."
    )
    FUNCTION = "load"

    @classmethod
    def IS_CHANGED(cls, path, start_frame=0, end_frame=-1,
                   select_every_nth=1, **kwargs):
        # **kwargs swallows the hidden inputs. The PROMPT in particular must
        # stay out of the fingerprint: it changes on every queue, so folding it
        # in would defeat caching entirely.
        # _strip_path first, exactly as load() does. A dragged-in path arrives
        # quoted; load() accepts it, but stat-ing the raw string here threw
        # OSError and returned the *constant* "missing" fingerprint, which
        # never changes. That pinned the node in cache for the life of the
        # process: the file could be re-rendered underneath it and the stale
        # frames were served forever.
        clean = _strip_path(path)
        return (f"{clean}|{_fingerprint(clean)}"
                f"|{start_frame}|{end_frame}|{select_every_nth}")

    def load(self, path: str, start_frame: int, end_frame: int,
             select_every_nth: int = 1, unique_id=None, prompt=None):
        path = _strip_path(path)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Bat_VideoLoader: not a file: {path!r}")
        abs_path = os.path.abspath(path)
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
        # Same fingerprint as IS_CHANGED, for the same reason: keyed on a
        # float getmtime() this cache re-served frames from before a
        # same-second rewrite.
        cache_key = (abs_path, _fingerprint(abs_path), start, end, stride)
        entry = _FRAME_CACHE.get(cache_key)
        if entry is not None:
            _FRAME_CACHE.move_to_end(cache_key)
        else:
            # Decode only the frames we're going to KEEP. This used to decode
            # the whole [start, end] range and then throw most of it away with
            # `frames[::stride]` — with select_every_nth=10 over a 3000-frame
            # range that decoded 3000 frames to return 300.
            images, mask, meta = load_batch(path, start, end, stride)
            source_fps = meta.get("fps") or info.get("fps") or 0.0
            entry = {"images": images, "mask": mask,
                     "fps": float(source_fps), "audio": _UNSET}
            _FRAME_CACHE[cache_key] = entry
            _trim_frame_cache()

        source_fps = entry["fps"]
        if source_fps <= 0:
            # Nothing readable said what rate this is. 0.0 would go straight
            # into an encoder as an invalid -r, so pick the common default and
            # say so rather than handing on a number that can't work.
            logger.warning("[Bat_VideoLoader] no frame rate reported for %s — "
                           "the frame_rate output falls back to 24",
                           os.path.basename(path))
            source_fps = 24.0
        # The batch is every `stride`-th frame of the source, so it plays at
        # the source rate divided by the stride. Encoding it at the SOURCE rate
        # would run it `stride` times too fast.
        frame_rate = source_fps / stride

        audio = entry["audio"]
        if audio is _UNSET:
            if _output_is_consumed(prompt, unique_id, _AUDIO_SLOT):
                # The time span of the trim, which the stride does not change:
                # taking every 3rd frame samples the same seconds more coarsely.
                start_time = start / source_fps
                duration = (end - start + 1) / source_fps
                audio = load_audio(path, start_time, duration)
                entry["audio"] = audio
                _trim_frame_cache()
            else:
                # Left as _UNSET so connecting the output later still extracts
                # it, instead of caching "no audio" for a clip that has some.
                audio = None

        return (entry["images"], entry["mask"], audio, frame_rate)


# ─── API routes ─────────────────────────────────────────────────────────────


@server.PromptServer.instance.routes.get("/bat/getpath")
async def bat_getpath(request):
    """Directory autocomplete for the node's path widget."""
    query = request.rel_url.query
    raw = query.get("path", "")
    if not raw:
        return server.web.Response(status=204)

    path = os.path.abspath(_strip_path(raw))
    if not os.path.isdir(path) or not _is_safe_path(path):
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
    path, err = _resolve_media(request.rel_url.query.get("path", ""))
    if err:
        return server.web.json_response({"ok": False, "error": err})
    info = probe_video(path)
    if info is None:
        return server.web.json_response({"ok": False, "error": "no decoder could open file"})
    # What load() will actually do with this source, so the node face can say
    # "10-bit · alpha · audio" instead of the artist finding out on execute.
    probe = probe_ffmpeg(path)
    if probe is not None:
        info = {**info,
                "bit_depth": probe["bit_depth"],
                "has_alpha": probe["has_alpha"],
                "has_audio": probe["has_audio"],
                "pix_fmt": probe["pix_fmt"],
                "codec": probe["codec"],
                # ffmpeg's banner is the better source for these two; the
                # frame count stays with whichever backend counted it.
                "fps": info.get("fps") or probe["fps"],
                "duration": probe["duration"]}
    return server.web.json_response({"ok": True, **info})


@server.PromptServer.instance.routes.get("/bat/video-stream")
async def bat_video_stream(request):
    """Serve the raw video file with Range support so an HTML5 <video>
    element can seek smoothly. aiohttp.web.FileResponse handles Range
    headers automatically.

    _resolve_media gates this: FileResponse will happily serve anything, so
    the extension allowlist is what stops this route being an arbitrary-file
    read. Every rejection answers 404 rather than 403 so the route can't be
    used to probe which paths exist."""
    path, err = _resolve_media(request.rel_url.query.get("path", ""))
    if err:
        return server.web.Response(status=404, text="not a file")
    return server.web.FileResponse(path)


@server.PromptServer.instance.routes.get("/bat/video-frame")
async def bat_video_frame(request):
    q = request.rel_url.query
    path, err = _resolve_media(q.get("path", ""))
    if err:
        return server.web.Response(status=404, text="not a file")
    try:
        frame_index = int(q.get("frame", "0"))
    except ValueError:
        frame_index = 0
    try:
        max_w = max(1, int(q.get("max_w", "320")))
    except ValueError:
        max_w = 320
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
