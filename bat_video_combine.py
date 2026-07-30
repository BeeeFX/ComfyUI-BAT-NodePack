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

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Optional: drives the node's progress bar during a long encode. Imported
# defensively so the module still loads outside a running ComfyUI (unit tests,
# tooling) where `comfy` isn't importable.
try:
    from comfy.utils import ProgressBar as _ProgressBar
except Exception:  # pragma: no cover - depends on host app
    _ProgressBar = None

import folder_paths
import server
from aiohttp import web

logger = logging.getLogger("[Bat_VideoCombine]")

# Alpha-capable pix_fmt → the equivalent WITHOUT alpha, same chroma layout and
# bit depth. Used when a format's `derived` rule picks an alpha pix_fmt (e.g.
# ProRes profile 4444 → yuva444p10le) but the incoming IMAGE has only 3
# channels, which is the norm in ComfyUI. Without this, ffmpeg was asked to
# write an alpha plane it had no data for.
_ALPHA_PIX_FALLBACK = {
    "yuva444p10le": "yuv444p10le",
    "yuva444p12le": "yuv444p12le",
    "yuva444p16le": "yuv444p16le",
    "yuva444p":     "yuv444p",
    "yuva422p10le": "yuv422p10le",
    "yuva422p":     "yuv422p",
    "yuva420p":     "yuv420p",
    "rgba64le":     "rgb48le",
    "rgba64be":     "rgb48be",
    "rgba":         "rgb24",
    "bgra":         "bgr24",
}

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


def _resolve_widget_values(fmt: dict, overrides: dict) -> dict:
    """Resolve every codec-widget value for one format.

    For each declared widget: take the caller's override if present, else the
    JSON default. Then apply `derived` mappings, which compute one widget's
    value from another's (e.g. ProRes pix_fmt from the chosen profile, so the
    UI only needs to expose `profile`). Derived widgets are resolved *after*
    the base pass and always win over any incoming value — they exist so an
    invalid hand-passed combination (4444 profile + 4:2:2 pix_fmt) can't slip
    through, whether it comes from the UI's hidden widget or a raw API call."""
    widgets = fmt.get("widgets") or {}
    values = {}
    for name, spec in widgets.items():
        if name in overrides and overrides[name] is not None:
            values[name] = overrides[name]
        else:
            values[name] = spec.get("default")

    for target, rule in (fmt.get("derived") or {}).items():
        source = rule.get("from")
        src_val = values.get(source)
        mapping = rule.get("map") or {}
        # str() so int/enum source values still key the (JSON string) map.
        values[target] = mapping.get(str(src_val), rule.get("default", values.get(target)))

    return values


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


def _ffprobe_bin() -> Optional[str]:
    """ffprobe path if it's on PATH, else None. It is NOT bundled with
    imageio_ffmpeg (which ships ffmpeg only), so on a bare venv it's usually
    absent — callers must fall back to _ffmpeg_probe."""
    return shutil.which("ffprobe")


def _ffmpeg_probe(path: str) -> dict:
    """Probe a media file using `ffmpeg -i` (always available via
    imageio_ffmpeg) when ffprobe isn't installed. ffmpeg prints stream/format
    info to stderr; we parse the bits the canvas player needs. Returns the
    same shape as the route's response."""
    try:
        out = subprocess.run(
            [_ffmpeg_bin(), "-hide_banner", "-i", path],
            capture_output=True, timeout=10,
        )
        err = out.stderr.decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"ffmpeg probe failed: {exc}")

    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    width = height = 0
    fps = 0.0
    vmatch = re.search(r"Stream #\d+:\d+.*?:\s*Video:.*", err)
    if vmatch:
        vline = vmatch.group(0)
        dim = re.search(r"(\d{2,5})x(\d{2,5})", vline)
        if dim:
            width, height = int(dim.group(1)), int(dim.group(2))
        fm = re.search(r"([\d.]+)\s*fps", vline)
        if fm:
            try:
                fps = float(fm.group(1))
            except ValueError:
                fps = 0.0

    has_audio = re.search(r"Stream #\d+:\d+.*?:\s*Audio:", err) is not None
    frame_count = int(round(duration * fps)) if (duration and fps) else 0

    return {
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "has_audio": has_audio,
    }


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _tensor_to_bytes(images: torch.Tensor, bit_depth: int = 8) -> bytes:
    """Flatten an (N, H, W, 3 or 4) float tensor to packed RGB(A) bytes.

    bit_depth=8 packs uint8 (rgb24/rgba). bit_depth=16 packs little-endian
    uint16 (rgb48le/rgba64le) so codecs that carry more than 8 bits per
    channel — ProRes (10-bit), FFV1, 16-bit PNG — receive the full precision
    of the incoming float tensor instead of a value pre-truncated to 256
    levels. Feeding 8-bit input into a 10-bit codec (the previous behaviour)
    produced a file *labelled* 10-bit but with only 8 bits of real data;
    banding in smooth gradients survived the round-trip. ComfyUI IMAGE
    tensors are float in [0,1], so the extra precision is genuinely present
    whenever upstream nodes work in float (grades, blurs, VAE decode)."""
    arr = images.detach().cpu().numpy()
    if bit_depth == 16:
        arr = np.clip(arr * 65535.0 + 0.5, 0, 65535).astype("<u2")
        return arr.tobytes()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr.tobytes()


def _frame_to_bytes(frame: torch.Tensor, bit_depth: int = 8) -> bytes:
    """Pack a single (H, W, 3 or 4) float frame to raw RGB(A) bytes.

    Same packing as _tensor_to_bytes but for one frame at a time, so the
    streaming encoder can convert-and-pipe frame N while ffmpeg encodes frame
    N-1 — no full-clip intermediate array. Kept separate (rather than looping
    _tensor_to_bytes over a length-1 slice) to avoid the per-call unsqueeze /
    reshape churn on the hot path."""
    arr = frame.detach().cpu().numpy()
    if bit_depth == 16:
        return np.clip(arr * 65535.0 + 0.5, 0, 65535).astype("<u2").tobytes()
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()


def _iter_encode_frames(images: torch.Tensor, pingpong: bool):
    """Yield frames in encode order without materialising a second copy.

    pingpong appends the reversed middle frames (1..n-2) after the forward
    pass so the clip palindromes — but as a lazy index walk, not a
    torch.cat of the whole reversed tensor. Each yielded item is a single
    (H, W, C) view into the original tensor; padding/packing happen
    downstream per frame."""
    n = images.shape[0]
    for i in range(n):
        yield images[i]
    if pingpong and n > 2:
        for i in range(n - 2, 0, -1):
            yield images[i]


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


# ─── Project metadata (workflow / prompt embedding) ──────────────────────────
#
# Mirrors what core's Save Image does: stash the API `prompt` and the editor
# `workflow` (carried in `extra_pnginfo`) inside the saved file so dragging the
# output back onto the ComfyUI canvas reloads the project. Each container reads
# these back differently — the ComfyUI frontend metadata parsers
# (src/scripts/metadata/{isobmff,ebml}.ts and pnginfo.ts) look for keys named
# WORKFLOW / PROMPT — so the *how* is keyed off the format JSON's `metadata`
# field rather than sniffing the extension here:
#   "ffmpeg_mov" → MP4/MOV: ffmpeg ffmetadata input + -movflags
#                  use_metadata_tags (writes the udta/meta/keys/ilst atoms the
#                  isobmff parser reads)
#   "ffmpeg_mkv" → WebM/MKV: ffmpeg ffmetadata input (native Matroska SimpleTags)
#   "webp_exif"  → animated WebP: lossless EXIF RIFF chunk spliced into the file
#   "png_text"   → PNG sequence: lossless iTXt chunk spliced into each frame
# The mov/mkv paths feed metadata through a `-f ffmetadata -i <file>` input,
# not inline `-metadata` args, so large workflows don't blow the argv limit
# (see _write_ffmetadata_file). The webp/png post-encode paths splice a chunk
# straight into the encoded bytes rather than re-saving through PIL — re-saving
# would re-compress (generation loss), drop the user's quality/lossless choice,
# downconvert 16-bit PNGs to 8-bit, and mangle non-ASCII via EXIF-ASCII. Raw
# chunk injection avoids all four. Formats without a `metadata` key (gif, exr,
# ffv1-mkv) get nothing — the frontend has no loader that would read them
# back, matching core/VHS.


def _build_metadata(prompt, extra_pnginfo) -> "OrderedDict[str, str]":
    """Return the {key: json-string} pairs to embed. Keys are upper-cased
    (WORKFLOW / PROMPT / …) so they match the ComfyUI frontend parsers, which
    compare case-insensitively but key the recovered dict off the stored name.

    `extra_pnginfo` is the hidden EXTRA_PNGINFO input — a dict that normally
    holds {"workflow": <editor graph>}. `prompt` is the hidden PROMPT input
    (the API-format graph). Both can legitimately be None (metadata disabled,
    or an API call without extra_pnginfo); `extra_pnginfo` has historically
    also arrived as a non-dict, so we guard the type. Never raises — a
    metadata problem must not be able to abort the caller's render."""
    meta: "OrderedDict[str, str]" = OrderedDict()

    def _add(key: str, value) -> None:
        # Serialise each entry independently so one unserialisable value
        # (e.g. a stray object in the API prompt) can't drop the others —
        # the editor `workflow` and the API `prompt` must not share fate.
        try:
            meta[key] = json.dumps(value)
        except (TypeError, ValueError) as exc:
            logger.warning("Bat_VideoCombine: skipping unserialisable metadata %r: %s", key, exc)

    if prompt is not None:
        _add("PROMPT", prompt)
    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            _add(str(key).upper(), value)
    elif extra_pnginfo is not None:
        logger.warning(
            "Bat_VideoCombine: extra_pnginfo is %s, not a dict; "
            "skipping workflow metadata.", type(extra_pnginfo).__name__,
        )
    return meta


def _escape_ffmetadata(value: str) -> str:
    """Escape a value for an ffmetadata file. Per the ffmetadata spec, the
    special characters `=`, `;`, `#`, `\\` and a literal newline must be
    backslash-escaped."""
    out = []
    for ch in value:
        if ch in ("=", ";", "#", "\\"):
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\\n")
        else:
            out.append(ch)
    return "".join(out)


def _write_ffmetadata_file(metadata: dict, temp_dir: str) -> str:
    """Serialise `metadata` to an ffmetadata file and return its path.

    Why a file and not `-metadata key=value` on the command line: real
    ComfyUI workflows routinely exceed 128 KB of JSON, and a single argv
    string is capped at MAX_ARG_STRLEN (128 KiB on Linux) regardless of the
    larger total ARG_MAX — so inline metadata raises "Argument list too long"
    for non-trivial graphs. ffmpeg reads this file via `-f ffmetadata -i`.

    The name carries a uuid (not a millisecond stamp) so two concurrent
    encodes — ComfyUI runs nodes in parallel, and farm workers share an
    output dir — can't land on the same path and read each other's JSON."""
    path = os.path.join(temp_dir, f"bat_meta_{uuid.uuid4().hex}.ffmeta")
    with open(path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for key, value in metadata.items():
            f.write(f"{key}={_escape_ffmetadata(value)}\n")
    return path


# EXIF ASCII tags PIL/exiftool conventionally map to plain strings; the
# frontend's parseExifData reads any type-2 entry and decodes it as UTF-8, so
# raw UTF-8 bytes survive here (unlike strict ASCII). One tag per metadata key.
_WEBP_EXIF_TAGS = [0x010F, 0x010E, 0x0131, 0x013B, 0x8298]  # Make, ImageDescription, Software, Artist, Copyright


def _build_exif_block(items: list) -> bytes:
    """Hand-build a little-endian TIFF/EXIF block: one IFD of ASCII (type-2)
    entries, each value the UTF-8 bytes of "key:value". This is what the
    ComfyUI frontend's getWebpMetadata → parseExifData reads back."""
    base = 8 + 2 + len(items) * 12 + 4  # header + entry count + entries + next-IFD ptr
    entries = b""
    blobs = b""
    offset = base
    for tag, (key, value) in zip(_WEBP_EXIF_TAGS, items):
        s = (f"{key.lower()}:{value}").encode("utf-8") + b"\x00"
        entries += struct.pack("<HHII", tag, 2, len(s), offset)
        blobs += s
        offset += len(s)
    header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    ifd = struct.pack("<H", len(items)) + entries + struct.pack("<I", 0)
    return header + ifd + blobs


def _iter_riff_chunks(body: bytes):
    """Yield (fourcc, start_offset, advance) for each RIFF chunk in `body`
    (the bytes after the leading 'WEBP' tag). `advance` includes the RIFF
    odd-size pad byte."""
    off = 0
    while off + 8 <= len(body):
        cid = body[off:off + 4]
        size = struct.unpack("<I", body[off + 4:off + 8])[0]
        yield cid, off, 8 + size + (size & 1)
        off += 8 + size + (size & 1)


def _webp_dimensions(cid: bytes, payload: bytes) -> Optional[tuple]:
    """Recover canvas (width, height) from a VP8 or VP8L bitstream chunk —
    needed to synthesise a VP8X header for a simple (non-extended) WebP."""
    if cid == b"VP8 ":
        i = payload.find(b"\x9d\x01\x2a")  # VP8 keyframe start code
        if i < 0 or i + 7 > len(payload):
            return None
        w = struct.unpack("<H", payload[i + 3:i + 5])[0] & 0x3FFF
        h = struct.unpack("<H", payload[i + 5:i + 7])[0] & 0x3FFF
        return w, h
    if cid == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
        bits = int.from_bytes(payload[1:5], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def _embed_webp_metadata(path: str, metadata: dict) -> None:
    """Splice an EXIF chunk carrying `metadata` into a WebP, losslessly.

    ffmpeg's libwebp muxer drops `-metadata`, and decoding+re-encoding through
    PIL would re-compress the (already lossy) frames, silently drop the user's
    quality/lossless choice, and force ASCII. Instead we inject the EXIF RIFF
    chunk straight into the encoded bytes: animated output is already a VP8X
    (extended) container, so we just set its EXIF flag and append the chunk; a
    single-frame VP8/VP8L file is upgraded to VP8X first. Pixels are untouched."""
    if not metadata:
        return
    items = list(metadata.items())
    if len(items) > len(_WEBP_EXIF_TAGS):
        logger.warning(
            "Bat_VideoCombine: %d metadata keys exceed WebP EXIF tag pool (%d); "
            "embedding first %d.", len(items), len(_WEBP_EXIF_TAGS), len(_WEBP_EXIF_TAGS),
        )
        items = items[: len(_WEBP_EXIF_TAGS)]

    with open(path, "rb") as f:
        src = f.read()
    if src[:4] != b"RIFF" or src[8:12] != b"WEBP":
        logger.warning("Bat_VideoCombine: %s is not a RIFF/WEBP file; skipping metadata.", path)
        return

    exif_payload = _build_exif_block(items)
    exif_chunk = b"EXIF" + struct.pack("<I", len(exif_payload)) + exif_payload
    if len(exif_payload) & 1:
        exif_chunk += b"\x00"

    body = src[12:]
    chunks = list(_iter_riff_chunks(body))
    if chunks and chunks[0][0] == b"VP8X":
        # Extended container already: flip the EXIF-present flag (bit 3 of the
        # first flags byte) and append the chunk after the existing payload.
        _, vp8x_off, vp8x_adv = chunks[0]
        flags = bytearray(body[vp8x_off + 8:vp8x_off + vp8x_adv])
        if flags:
            flags[0] |= (1 << 3)
        new_vp8x = b"VP8X" + struct.pack("<I", len(flags)) + bytes(flags)
        new_body = b"WEBP" + new_vp8x + body[vp8x_off + vp8x_adv:] + exif_chunk
    else:
        # Simple container: build a VP8X header (flags + 3-byte (w-1),(h-1)).
        dims = None
        for cid, off, adv in chunks:
            if cid in (b"VP8 ", b"VP8L"):
                dims = _webp_dimensions(cid, body[off + 8:off + adv])
                break
        if dims is None:
            logger.warning("Bat_VideoCombine: could not size %s for VP8X upgrade; skipping metadata.", path)
            return
        w, h = dims
        vp8x = bytearray(10)
        vp8x[0] = (1 << 3)  # EXIF metadata present
        wv, hv = w - 1, h - 1
        vp8x[4], vp8x[5], vp8x[6] = wv & 0xFF, (wv >> 8) & 0xFF, (wv >> 16) & 0xFF
        vp8x[7], vp8x[8], vp8x[9] = hv & 0xFF, (hv >> 8) & 0xFF, (hv >> 16) & 0xFF
        new_vp8x = b"VP8X" + struct.pack("<I", 10) + bytes(vp8x)
        new_body = b"WEBP" + new_vp8x + body + exif_chunk

    out = b"RIFF" + struct.pack("<I", len(new_body)) + new_body
    with open(path, "wb") as f:
        f.write(out)


def _png_itxt_chunk(key: str, value: str) -> bytes:
    """Build a PNG iTXt chunk (UTF-8, uncompressed). iTXt — not tEXt — because
    tEXt is Latin-1 only; iTXt carries the workflow JSON's non-ASCII intact,
    and getPngMetadata reads both."""
    # keyword(latin-1) \0 compression_flag(0) compression_method(0)
    # language_tag \0 translated_keyword \0 text(utf-8)
    data = (
        key.encode("latin-1", "replace") + b"\x00"
        + b"\x00" + b"\x00"
        + b"\x00" + b"\x00"
        + value.encode("utf-8")
    )
    return struct.pack(">I", len(data)) + b"iTXt" + data + struct.pack(
        ">I", zlib.crc32(b"iTXt" + data) & 0xFFFFFFFF
    )


def _embed_png_sequence_metadata(seq_dir: str, metadata: dict) -> None:
    """Splice an iTXt chunk into every PNG under `seq_dir`, losslessly.

    A printf-pattern sequence isn't a single droppable file, but an individual
    frame dragged onto the canvas reloads the project. We insert the chunk
    before each file's IEND rather than re-saving through PIL — re-saving would
    downconvert the 16-bit pix_fmts (rgb48be/rgba64be) the format offers to
    8-bit. Keys are lower-cased ("workflow"/"prompt") to match getPngMetadata."""
    if not metadata:
        return
    blocks = b"".join(_png_itxt_chunk(key.lower(), value) for key, value in metadata.items())
    if not blocks:
        return

    for fn in sorted(os.listdir(seq_dir)):
        if not fn.lower().endswith(".png"):
            continue
        fpath = os.path.join(seq_dir, fn)
        try:
            with open(fpath, "rb") as f:
                raw = f.read()
            if raw[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            # IEND is the final chunk; its 4-byte length precedes the type, so
            # the chunk (incl. its length field) starts 8 bytes before "IEND".
            iend = raw.rfind(b"IEND")
            if iend < 4:
                continue
            insert_at = iend - 4
            with open(fpath, "wb") as f:
                f.write(raw[:insert_at] + blocks + raw[insert_at:])
        except Exception as exc:
            logger.warning("Bat_VideoCombine: could not embed PNG metadata in %s: %s", fn, exc)


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
    metadata: Optional[dict] = None,
) -> None:
    """Pipe raw frames through ffmpeg to produce a video at output_path."""
    n, h, w, c = images.shape
    if c not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA tensor, got {c} channels.")

    # pingpong is now applied lazily by _iter_encode_frames (no torch.cat of
    # the reversed tensor); compute the resulting frame count for logging only.
    if pingpong and n > 2:
        n = 2 * n - 2

    # Even-dimensions guard: codecs paired with YUV 4:2:0/4:2:2 chroma
    # subsampling (h264, h265, vp9, av1, prores, animated webp) refuse
    # odd width/height — ffmpeg errors out with "width not divisible by
    # 2". Pad up by a single black row/column on the right/bottom when
    # the format JSON declares `requires_even_dims: true`. Single-pixel
    # padding is imperceptible and preserves the original pixel data
    # bit-for-bit, vs. a scale filter which would resample everything.
    # The pad is applied per-frame in the streaming loop below (pad_h/pad_w)
    # so we never allocate a second full-batch tensor; here we only work out
    # the padded output dimensions ffmpeg needs for its -s arg.
    pad_h = pad_w = 0
    if fmt.get("requires_even_dims") and (h % 2 != 0 or w % 2 != 0):
        pad_h = h % 2
        pad_w = w % 2
        new_h, new_w = h + pad_h, w + pad_w
        logger.info(
            "Bat_VideoCombine: %s requires even dimensions; padding "
            "input from %dx%d to %dx%d (black at right/bottom).",
            fmt.get("label", "format"), w, h, new_w, new_h,
        )
        h, w = new_h, new_w

    # Input bit depth is a per-format property: codecs that store >8 bits
    # (ProRes, FFV1, 16-bit PNG) declare `input_color_depth: "16bit"` in their
    # JSON so we pipe rgb48le/rgba64le and preserve the float tensor's
    # precision. Everything else stays 8-bit rgb24/rgba. Mirrors VHS's
    # input_color_depth handling (videohelpersuite/nodes.py).
    bit_depth = 16 if fmt.get("input_color_depth") == "16bit" else 8
    if bit_depth == 16:
        input_pix = "rgba64le" if c == 4 else "rgb48le"
    else:
        input_pix = "rgba" if c == 4 else "rgb24"

    # Reconcile the OUTPUT pix_fmt with the channel count we actually have.
    #
    # `derived` rules map e.g. ProRes profile 4444 -> yuva444p10le without
    # seeing the tensor, but ComfyUI IMAGE is nearly always 3-channel. Asking
    # ffmpeg for an alpha pix_fmt with no alpha in the input made it invent an
    # alpha plane; the mirror case (4-channel input, non-alpha profile) silently
    # dropped the alpha with nothing in the log.
    out_pix = widget_values.get("pix_fmt") if isinstance(widget_values, dict) else None
    if out_pix:
        has_alpha_pix = ("a" in out_pix.split("p")[0]) or out_pix.startswith("yuva") \
                        or out_pix.startswith("rgba") or out_pix.startswith("bgra")
        if has_alpha_pix and c != 4:
            fallback = _ALPHA_PIX_FALLBACK.get(out_pix)
            if fallback:
                logger.warning(
                    "Bat_VideoCombine: %s selects pix_fmt %s (with alpha) but the "
                    "input image has %d channels — encoding as %s instead.",
                    fmt.get("label", "format"), out_pix, c, fallback,
                )
                widget_values["pix_fmt"] = fallback
        elif (not has_alpha_pix) and c == 4:
            logger.warning(
                "Bat_VideoCombine: input has an alpha channel but pix_fmt %s "
                "cannot store it — the alpha will be discarded. Pick an alpha-"
                "capable profile (e.g. ProRes 4444) to keep it.",
                out_pix,
            )

    # Project metadata for containers ffmpeg can tag directly (mov/mkv
    # family). webp/png-sequence are handled post-encode by the caller. We
    # feed it as a separate `-f ffmetadata -i <file>` input rather than inline
    # `-metadata` args because real workflows exceed the argv length cap (see
    # _write_ffmetadata_file). Input indices: stdin is 0; audio (if present)
    # is 1; the ffmetadata input is therefore the next index.
    meta_mode = fmt.get("metadata")
    meta_file = None
    meta_input_index = None
    if metadata and meta_mode in ("ffmpeg_mov", "ffmpeg_mkv"):
        meta_file = _write_ffmetadata_file(metadata, os.path.dirname(output_path) or ".")
        meta_input_index = 1 + (1 if audio_path else 0)

    ffmpeg = _ffmpeg_bin()
    try:
        args = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", input_pix,
            "-s", f"{w}x{h}",
            "-r", f"{frame_rate}",
            "-i", "-",                                # raw frames on stdin (input 0)
        ]
        if audio_path:
            args += ["-i", audio_path]                # audio (input 1)
        if meta_file:
            args += ["-f", "ffmetadata", "-i", meta_file]  # metadata (next input)

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

        if meta_file:
            # Pull global metadata from the ffmetadata input. For the MP4/MOV
            # (isobmff) family the keys/ilst atoms only appear with
            # `use_metadata_tags`. ffmpeg's -movflags is a single last-wins
            # field — the '+' combines with the option default, NOT with an
            # earlier -movflags token — so a second standalone token would
            # silently drop the format JSON's "+faststart". We therefore emit
            # the full combined set here. Matroska/WebM stores tags natively.
            args += ["-map_metadata", str(meta_input_index)]
            if meta_mode == "ffmpeg_mov":
                args += ["-movflags", "+faststart+use_metadata_tags"]

        args.append(output_path)

        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Drain stderr on a background thread for the whole encode. Without
        # this, streaming frames deadlocks: ffmpeg blocks writing to a full
        # stderr pipe (~64 KB) → stops draining stdin → our stdin.write()
        # blocks → we never get to read stderr. The thread keeps stderr
        # flowing so both pipes make progress. We join it after closing stdin.
        stderr_chunks: list = []

        def _drain_stderr(pipe):
            try:
                for chunk in iter(lambda: pipe.read(65536), b""):
                    stderr_chunks.append(chunk)
            except (OSError, ValueError):
                pass

        stderr_thread = threading.Thread(
            target=_drain_stderr, args=(proc.stderr,), daemon=True,
        )
        stderr_thread.start()

        def _stderr_text() -> str:
            stderr_thread.join(timeout=5)
            return b"".join(stderr_chunks).decode("utf-8", errors="replace")

        try:
            # Convert-and-pipe one frame at a time. pingpong ordering and the
            # even-dims pad are applied here, per frame, so peak memory stays
            # at ~one frame of overhead instead of a full converted copy of
            # the clip — and ffmpeg encodes frame N-1 while we pack frame N.
            # Pad buffer is allocated ONCE and reused per frame (it used to be a
            # fresh torch.zeros per frame — an allocation per frame for the whole
            # clip).
            pad_buf = None
            # Progress feedback: a long ProRes 4K encode otherwise shows a
            # completely frozen node with no indication it's working.
            pbar = None
            try:
                if _ProgressBar is not None:
                    total = images.shape[0]
                    if pingpong and total > 2:
                        total += total - 2
                    pbar = _ProgressBar(total)
            except Exception:
                pbar = None

            for frame in _iter_encode_frames(images, pingpong):
                if pad_h or pad_w:
                    fh, fw, fc = frame.shape
                    if pad_buf is None:
                        pad_buf = torch.empty(
                            (fh + pad_h, fw + pad_w, fc),
                            dtype=frame.dtype, device=frame.device,
                        )
                    pad_buf[:fh, :fw, :] = frame
                    # REPLICATE the edge rather than leaving black. With 4:2:0
                    # chroma subsampling a black pad row/column gets averaged
                    # into the last real row of pixels, visibly darkening the
                    # bottom/right edge. Replicating is free and keeps the edge
                    # colour intact.
                    if pad_h:
                        pad_buf[fh:, :fw, :] = frame[fh - 1:fh, :, :]
                    if pad_w:
                        pad_buf[:fh, fw:, :] = frame[:, fw - 1:fw, :]
                    if pad_h and pad_w:
                        pad_buf[fh:, fw:, :] = frame[fh - 1:fh, fw - 1:fw, :]
                    frame = pad_buf
                proc.stdin.write(_frame_to_bytes(frame, bit_depth))
                if pbar is not None:
                    try: pbar.update(1)
                    except Exception: pbar = None
            proc.stdin.close()
            rc = proc.wait()
            err = _stderr_text()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed (rc={rc}):\n{err}")
            if err.strip():
                logger.debug(f"ffmpeg stderr: {err.strip()[:500]}")
        except BrokenPipeError:
            # ffmpeg died mid-stream (e.g. bad args). Reap it, then surface the
            # real complaint from the drained stderr.
            proc.wait()
            raise RuntimeError(f"ffmpeg pipe broke:\n{_stderr_text()}")
    finally:
        if meta_file and os.path.isfile(meta_file):
            try:
                os.remove(meta_file)
            except OSError:
                pass


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
        # JSON default wins unless the caller passes an override. The JS adds
        # one widget per entry in the format's `widgets` map when that format
        # is selected (see web/bat_video_combine.js), so the artist can tune
        # crf / preset / profile / etc.; callers that don't (e.g. the API)
        # just get the defaults. `derived` widgets are then computed from
        # another widget's value — see _resolve_widget_values.
        widget_values = _resolve_widget_values(fmt, format_widget_kwargs)

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
        seq_dir = None
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

        # Build the project metadata to embed (workflow + prompt), so the saved
        # file reopens its graph when dragged onto the ComfyUI canvas.
        # _build_metadata never raises — a metadata problem must never abort an
        # artist's (potentially long) render; worst case the file just saves
        # without an embedded workflow.
        metadata = _build_metadata(prompt, extra_pnginfo)

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
                metadata=metadata,
            )
        finally:
            if audio_path and os.path.isfile(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        # Post-encode metadata for containers ffmpeg can't tag itself. Best
        # effort — a failure here must not lose the already-encoded output.
        meta_mode = fmt.get("metadata")
        if metadata:
            try:
                if meta_mode == "webp_exif":
                    _embed_webp_metadata(output_path, metadata)
                elif meta_mode == "png_text" and seq_dir is not None:
                    _embed_png_sequence_metadata(seq_dir, metadata)
            except Exception as exc:
                logger.warning(
                    "Bat_VideoCombine: failed to embed %s metadata: %s",
                    meta_mode, exc,
                )

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


@server.PromptServer.instance.routes.get("/bat/video/formats")
async def bat_video_formats(request):
    """Serve the per-format codec-widget specs to the node's JS.

    The JS can't read bat_video_formats/*.json directly, so it asks here.
    Returns, per format label, the ordered widget definitions (name, type,
    label, options/min/max, default, hidden) plus the `derived` rules. The
    front-end renders one widget per non-hidden entry when that format is
    selected and mirrors `derived` so a profile change updates a hidden
    pix_fmt live; the backend re-applies `derived` authoritatively at encode
    time (see _resolve_widget_values), so the JS copy is only for UX."""
    out = {}
    for label, fmt in _load_formats().items():
        widgets = []
        for name, spec in (fmt.get("widgets") or {}).items():
            w = {
                "name": name,
                "type": spec.get("type", "STRING"),
                "label": spec.get("label", name),
                "default": spec.get("default"),
                "hidden": bool(spec.get("hidden", False)),
            }
            if "options" in spec:
                w["options"] = spec["options"]
            if "min" in spec:
                w["min"] = spec["min"]
            if "max" in spec:
                w["max"] = spec["max"]
            widgets.append(w)
        out[label] = {"widgets": widgets, "derived": fmt.get("derived") or {}}
    return web.json_response(out)


@server.PromptServer.instance.routes.get("/bat/video/meta")
async def bat_video_meta(request):
    """Probe duration / fps / frame count / dimensions / audio flag for the
    canvas player. Prefers ffprobe (precise nb_frames), but ffprobe is not
    bundled with imageio_ffmpeg, so on a bare venv we fall back to parsing
    `ffmpeg -i` (which is always available). Either way returns 200 with the
    best info we have rather than 500-ing the player."""
    path = _resolve_request_path(request)
    if path is None:
        return web.json_response({"error": "Not found"}, status=404)

    ffprobe = _ffprobe_bin()
    if ffprobe:
        try:
            out = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", path],
                capture_output=True, timeout=10,
            )
            info = json.loads(out.stdout.decode("utf-8", errors="replace") or "{}")
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
        except Exception as exc:
            logger.warning("Bat_VideoCombine: ffprobe failed (%s); falling back to ffmpeg.", exc)

    # Fallback: parse `ffmpeg -i` stderr (no ffprobe needed).
    try:
        return web.json_response(_ffmpeg_probe(path))
    except Exception as exc:
        return web.json_response({"error": f"probe failed: {exc}"}, status=500)


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
        # Stable cache key — Python's built-in hash() is salted per process
        # (PYTHONHASHSEED), so it never re-hit this cache across restarts and
        # re-transcoded every launch. A hashlib digest is deterministic; the
        # mtime+size suffix still busts the entry when the source file changes.
        digest = hashlib.md5(path.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        key = f"{digest}_{int(st.st_mtime)}_{st.st_size}.mp4"
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
    # Guard the cast and clamp negatives: `int("abc")` raised straight out of the
    # handler as a 500 + traceback, and a negative value was interpolated into
    # the ffmpeg filter string as `select=eq(n\,-1)`.
    try:
        frame = max(0, int(body.get("frame") or 0))
    except (TypeError, ValueError):
        return web.json_response({"error": "frame must be an integer"}, status=400)
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
