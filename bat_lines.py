"""
🦇 Lines — one node for line and edge extraction (ControlNet lines, comp mattes).

It replaces the line extractors of comfyui_controlnet_aux (Canny, XDoG scribble,
standard lineart, TEED, HED, lineart realistic / anime / manga, M-LSD) with one
node whose `filter` dropdown picks the method, so that pack — 4 GB of weights
and several non-commercial ones — can leave the studio installs.

Two things are kept identical to controlnet_aux on purpose, because old
workflows migrate onto this node and must keep producing the same lines:

* **The geometry.** A frame is quantised to uint8 (truncating, as there), its
  short side scaled to `resolution` (INTER_AREA when shrinking, the filter's own
  upscale method otherwise, with `round()`ed sizes), padded right and bottom to
  a multiple of 64 by edge replication, filtered, and the pad cropped off.
* **The no-model filters.** Canny, XDoG scribble and standard lineart are the
  same OpenCV calls in the same order, uint8 wrap-around included (XDoG's
  ``2 * (255 - dog)`` overflows in uint8 there too). `tests/verify_lines.py`
  holds this to byte equality against a transcription of the original code.

What is new: `output_size` brings the lines back to the plate's size (the
default — controlnet_aux left them at processing size, which is what migrated
nodes keep), a MASK output, `invert`, a graded (not thresholded) edge-only XDoG, and
a live preview (see web/bat_lines.js) that re-runs the filter on the cached
frame as widgets change, through the same `run_filter()` the node uses.

The networks are in ``bat_lines_nets/`` and their weights in
``bat_lines_models.py``.
"""

import asyncio
import io
import json
import logging
import threading
import uuid
import weakref
from collections import OrderedDict

import cv2
import numpy as np
import torch
from PIL import Image

from .bat_ui_ref import stash_ui

logger = logging.getLogger("[Bat_Lines]")

# Widget value -> description. Order is the dropdown's order.
FILTERS = OrderedDict([
    ("canny",             "Canny edges (OpenCV). Hard 1-px edges between the two thresholds."),
    ("xdog",              "Difference of Gaussians. 'soft' = graded ink lines, 'scribble' = controlnet_aux's Scribble XDoG."),
    ("lineart standard",  "controlnet_aux's Standard Lineart: dark detail against a blur, no model."),
    ("teed",              "TEED soft edges (model, 0.25 MB, MIT)."),
    ("hed",               "HED soft edges (model, 29 MB, Apache-2.0)."),
    ("lineart realistic", "informative-drawings line art for photos (model, 17 MB, MIT)."),
    ("lineart anime",     "Anime2Sketch line art (model, 218 MB, MIT)."),
    ("lineart manga",     "MangaLineExtraction line art (model, 173 MB, MIT)."),
    ("mlsd",              "M-LSD straight line segments (model, 6 MB, Apache-2.0). Lines are redrawn crisp at the output size."),
])

# Filters that need weights (bat_lines_nets), and the model keys they load.
MODEL_FILTERS = {"teed", "hed", "lineart realistic", "lineart anime",
                 "lineart manga", "mlsd"}

# controlnet_aux's upscale method per filter (shrinking is always INTER_AREA).
_UPSCALE = {"mlsd": cv2.INTER_AREA}

OUTPUT_SIZES = ["input", "processing"]
XDOG_STYLES = ["soft", "scribble"]

# One cached preview frame is the source at full size, float32. Same budget
# reasoning as Bat_Rescale.
CACHE_MAX_BYTES = 400 << 20
MAX_RESOLUTION = 16384


# ---------------------------------------------------------------------------
# Geometry (controlnet_aux's resize_image_with_pad, kept exact)
# ---------------------------------------------------------------------------

def to_u8(frame):
    """(H,W,C) float tensor -> (H,W,3) uint8, as controlnet_aux quantises.

    `np.asarray(x * 255, dtype=uint8)` truncates. Values are clamped first: the
    cast of an out-of-range float is undefined in numpy, so an HDR plate would
    otherwise give platform-dependent garbage rather than clipped highlights.
    """
    a = np.asarray(frame.detach().to("cpu", torch.float32).clamp(0, 1).numpy() * 255.0,
                   dtype=np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    c = a.shape[2]
    if c == 1:
        return np.concatenate([a, a, a], axis=2)
    if c == 4:     # controlnet_aux's HWC3: composite over white
        color = a[:, :, :3].astype(np.float32)
        alpha = a[:, :, 3:4].astype(np.float32) / 255.0
        return (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(a[:, :, :3])


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def processing_size(h, w, resolution):
    """(H_target, W_target) for a frame of (h, w): the short side at
    `resolution`, or the frame's own size when it is 0."""
    if resolution <= 0:
        return h, w
    k = float(resolution) / float(min(h, w))
    return int(np.round(float(h) * k)), int(np.round(float(w) * k))


def prepare(u8, resolution, upscale=cv2.INTER_CUBIC):
    """Scale + pad one uint8 frame. Returns (padded, (H_target, W_target))."""
    h, w = u8.shape[:2]
    ht, wt = processing_size(h, w, resolution)
    if (ht, wt) != (h, w):
        k = float(resolution) / float(min(h, w))
        u8 = cv2.resize(u8, (wt, ht), interpolation=upscale if k > 1 else cv2.INTER_AREA)
    padded = np.pad(u8, [[0, pad64(ht)], [0, pad64(wt)], [0, 0]], mode="edge")
    return np.ascontiguousarray(padded), (ht, wt)


def effective_resolution(filter_name, resolution):
    """What the frame is really processed at: controlnet_aux's manga detector
    rounds the resolution up to a multiple of 256."""
    if filter_name == "lineart manga" and resolution > 0:
        return 256 * int(np.ceil(float(resolution) / 256.0))
    return resolution


# ---------------------------------------------------------------------------
# No-model filters. Each takes the padded uint8 RGB frame and returns a uint8
# (H, W) map of white lines on black at the padded size.
# ---------------------------------------------------------------------------

def f_canny(img, p):
    return cv2.Canny(img, int(p["canny_low"]), int(p["canny_high"]))


def f_xdog_scribble(img, p):
    g1 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.5)
    g2 = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 5.0)
    dog = (255 - np.min(g2 - g1, axis=2)).clip(0, 255).astype(np.uint8)
    out = np.zeros(img.shape[:2], dtype=np.uint8)
    # uint8 arithmetic on purpose — it wraps, exactly as in controlnet_aux.
    out[2 * (255 - dog) > int(p["xdog_scribble_threshold"])] = 255
    return out


def f_xdog_soft(img, p):
    """Graded difference-of-Gaussians lines on luma, white on black.

    The edge-only form of XDoG: D = sharpen * (G_ks - G_s) is the response on
    the dark side of an edge, and lines = tanh(softness * (D - threshold)) where
    D passes the threshold. Winnemöller's full form, (1 + p) G_s - p G_ks, also
    carries the luma itself, so every area darker than the threshold fills with
    ink — a stylised look, not lines. Flat areas here stay black at any
    brightness.
    """
    luma = (img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
    s = float(p["xdog_sigma"])
    g1 = cv2.GaussianBlur(luma, (0, 0), s)
    g2 = cv2.GaussianBlur(luma, (0, 0), s * float(p["xdog_k"]))
    d = float(p["xdog_sharpen"]) * (g2 - g1)
    eps = float(p["xdog_threshold"])
    lines = np.where(d > eps, np.tanh(float(p["xdog_softness"]) * (d - eps)), 0.0)
    return (lines.clip(0, 1) * 255.0 + 0.5).astype(np.uint8)


def f_lineart_standard(img, p):
    x = img.astype(np.float32)
    g = cv2.GaussianBlur(x, (0, 0), float(p["standard_sigma"]))
    intensity = np.min(g - x, axis=2).clip(0, 255)
    above = intensity[intensity > float(p["standard_threshold"])]
    # An empty selection is NaN in controlnet_aux too, and max(16, nan) is 16.
    med = np.median(above) if above.size else float("nan")
    intensity /= max(16, med)
    intensity *= 127
    return intensity.clip(0, 255).astype(np.uint8)


def _no_model(filter_name, img, p):
    if filter_name == "canny":
        return f_canny(img, p)
    if filter_name == "xdog":
        return f_xdog_scribble(img, p) if p["xdog_style"] == "scribble" else f_xdog_soft(img, p)
    if filter_name == "lineart standard":
        return f_lineart_standard(img, p)
    raise ValueError(f"unknown filter {filter_name!r}")


# ---------------------------------------------------------------------------
# One frame, end to end
# ---------------------------------------------------------------------------

def run_filter(frame, p, out_hw=None):
    """Lines for one (H,W,C) float frame, as a (H', W') uint8 array.

    `out_hw` is the size to return at — the frame's own size for
    output_size=input — or None to stay at processing size. Both the node and
    the preview endpoint come through here, so the preview IS the render.
    """
    name = p["filter"]
    u8 = to_u8(frame)
    res = effective_resolution(name, int(p["resolution"]))
    img, (ht, wt) = prepare(u8, res, _UPSCALE.get(name, cv2.INTER_CUBIC))

    segments = None
    if name in MODEL_FILTERS:
        from . import bat_lines_nets
        result = bat_lines_nets.run(name, img, p)
        if isinstance(result, dict):           # mlsd: segments + their raster
            segments, lines = result["segments"], result["raster"]
        else:
            lines = result
    else:
        lines = _no_model(name, img, p)
    lines = np.ascontiguousarray(lines[:ht, :wt])

    if out_hw is not None and tuple(out_hw) != (ht, wt):
        oh, ow = int(out_hw[0]), int(out_hw[1])
        if segments is not None:
            # Straight segments are geometry, not pixels: redraw them at the
            # output size instead of resampling a 1-px raster into mush.
            lines = draw_segments(segments, oh, ow, ow / wt, oh / ht,
                                  int(p["mlsd_line_width"]))
        else:
            shrink = oh <= ht and ow <= wt
            lines = cv2.resize(lines, (ow, oh),
                               interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    elif segments is not None and int(p["mlsd_line_width"]) != 1:
        lines = draw_segments(segments, ht, wt, 1.0, 1.0, int(p["mlsd_line_width"]))

    if p["invert"]:
        lines = 255 - lines
    return lines


def draw_segments(segments, h, w, sx, sy, width):
    out = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in segments:
        cv2.line(out, (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy)),
                 255, max(1, int(width)))
    return out


# ---------------------------------------------------------------------------
# Live preview cache (Bat_Rescale's pattern: per-run token, CPU frame, weak
# reference to the batch so other frames can be previewed without pinning it)
# ---------------------------------------------------------------------------

_cache = OrderedDict()          # token -> entry
_cache_lock = threading.Lock()
# The node and the preview endpoint share models and the GPU; one at a time.
_run_lock = threading.Lock()


def _entry_bytes(entry):
    t = entry.get("frame")
    return 0 if t is None else t.numel() * t.element_size()


def _cache_put(token, node_id, frame, batch, meta):
    entry = {
        "node_id": str(node_id),
        "frame": frame.detach().to("cpu", torch.float32).contiguous(),
        "batch": _weak(batch),
        "meta": meta,
    }
    with _cache_lock:
        for old_token, old in list(_cache.items()):
            if old["node_id"] == entry["node_id"]:
                _cache.pop(old_token, None)
        _cache[token] = entry
        total = sum(_entry_bytes(e) for e in _cache.values())
        while len(_cache) > 1 and total > CACHE_MAX_BYTES:
            _, dropped = _cache.popitem(last=False)
            total -= _entry_bytes(dropped)


def _weak(t):
    try:
        return weakref.ref(t)
    except TypeError:
        return None


def _cache_get(token):
    if not token:
        return None
    with _cache_lock:
        entry = _cache.get(token)
        if entry is not None:
            _cache.move_to_end(token)
        return entry


def _frame_for(entry, index):
    """(frame (H,W,C) CPU float, exact). `exact` is False when the batch has
    been freed and the cached anchor frame stands in for the one asked for."""
    anchor = int(entry["meta"].get("frame", 0))
    index = int(index)
    if index == anchor:
        return entry["frame"][0], True
    ref = entry.get("batch")
    batch = ref() if ref is not None else None
    if batch is None:
        return entry["frame"][0], False
    i = max(0, min(index, int(batch.shape[0]) - 1))
    return batch[i].detach().to("cpu", torch.float32), i == index


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _defaults():
    """Every parameter's default, from INPUT_TYPES — the preview endpoint fills
    gaps in a request from here, so the two can't disagree."""
    out = {}
    for name, spec in BatLines.INPUT_TYPES()["required"].items():
        if name == "image":
            continue
        kind, opts = spec[0], (spec[1] if len(spec) > 1 else {})
        out[name] = opts.get("default", kind[0] if isinstance(kind, list) else None)
    return out


class BatLines:
    """Line / edge extraction with a live preview."""

    @classmethod
    def INPUT_TYPES(cls):
        # Every filter's widgets are declared here, always, in a fixed order;
        # web/bat_lines.js only hides the ones the current filter ignores.
        # Hidden widgets still serialise, so widgets_values never shift. New
        # widgets go at the END only.
        return {
            "required": {
                "image": ("IMAGE",),
                "filter": (list(FILTERS), {"default": "canny", "tooltip":
                    "The extraction method. " + " ".join(
                        f"{k}: {v}" for k, v in FILTERS.items())}),
                "resolution": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64,
                    "tooltip": "Short side the filter works at (the detail level). "
                               "0 = the image's own size. Lines from a model filter "
                               "look best near 512-1024."}),
                "output_size": (OUTPUT_SIZES, {"default": "input", "tooltip":
                    "input = lines at the image's size (fits a ControlNet or a comp "
                    "directly). processing = at the working size, as "
                    "comfyui_controlnet_aux returned them."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip":
                    "Black lines on white instead of white on black."}),
                "canny_low": ("INT", {"default": 100, "min": 0, "max": 255, "tooltip":
                    "Edges weaker than this are dropped (0-255 gradient)."}),
                "canny_high": ("INT", {"default": 200, "min": 0, "max": 255, "tooltip":
                    "Edges stronger than this always stay; weaker ones only where "
                    "they connect to one. Core ComfyUI's Canny node uses 0-1 "
                    "thresholds at the input size; this one matches controlnet_aux."}),
                "xdog_style": (XDOG_STYLES, {"default": "soft", "tooltip":
                    "soft = graded lines from a difference of Gaussians. scribble = "
                    "controlnet_aux's Scribble XDoG, binary."}),
                "xdog_sigma": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 10.0, "step": 0.05,
                    "tooltip": "Line scale in pixels at the processing size."}),
                "xdog_k": ("FLOAT", {"default": 1.6, "min": 1.1, "max": 5.0, "step": 0.05,
                    "tooltip": "Ratio of the two blurs. Higher = thicker, bolder lines."}),
                "xdog_sharpen": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Gain on the edge response. Higher = fainter edges become lines."}),
                "xdog_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Edges weaker than this draw nothing. Raise it to drop grain and texture."}),
                "xdog_softness": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "How quickly a line reaches full white. Low = soft graded strokes, high = hard lines."}),
                "xdog_scribble_threshold": ("INT", {"default": 32, "min": 1, "max": 64, "tooltip":
                    "Scribble only: lower = more lines (controlnet_aux's 'threshold')."}),
                "standard_sigma": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Blur the detail is measured against. Bigger = thicker lines."}),
                "standard_threshold": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 16.0, "step": 0.5,
                    "tooltip": "Detail weaker than this is ignored when levelling the lines."}),
                "teed_safe_steps": ("INT", {"default": 2, "min": 0, "max": 10, "tooltip":
                    "Quantise the soft edges into this many levels (0 = off). Two "
                    "keeps ControlNet from copying faint texture."}),
                "hed_safe": ("BOOLEAN", {"default": True, "tooltip":
                    "Quantise HED's soft edges (controlnet_aux 'safe')."}),
                "lineart_coarse": ("BOOLEAN", {"default": False, "tooltip":
                    "Coarse line art: fewer, bolder lines (the second informative-"
                    "drawings model)."}),
                "mlsd_score": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "Minimum confidence for a line. Higher = fewer lines."}),
                "mlsd_distance": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 20.0, "step": 0.01,
                    "tooltip": "Minimum segment length. Higher = drops short segments."}),
                "mlsd_line_width": ("INT", {"default": 1, "min": 1, "max": 16, "tooltip":
                    "Drawn line thickness in output pixels."}),
                "auto_download": ("BOOLEAN", {"default": True, "tooltip":
                    "Fetch a model filter's weights on first use (pinned + "
                    "checksummed) into models/lines. Off = fail with instructions."}),
                "preview_frame": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip":
                    "Which frame of the batch the live preview shows."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("lines", "mask")
    OUTPUT_TOOLTIPS = (
        "The lines as an RGB image (white on black unless inverted).",
        "The same lines as a single-channel mask.",
    )
    FUNCTION = "run"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Line and edge extraction for ControlNet and comp: Canny, XDoG, standard "
        "lineart, TEED, HED, lineart realistic / anime / manga and M-LSD in one "
        "node, with a live preview. Every method is commercially usable; model "
        "weights are pinned and checksummed (models/lines)."
    )

    def run(self, image, unique_id=None, **params):
        if image.ndim != 4:
            raise ValueError(f"expected an (N,H,W,C) batch, got {tuple(image.shape)}")
        p = _defaults()
        p.update({k: v for k, v in params.items() if k in p})
        if p["filter"] not in FILTERS:
            raise ValueError(f"unknown filter {p['filter']!r}")

        n, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        out_hw = (h, w) if p["output_size"] == "input" else None

        if p["filter"] in MODEL_FILTERS:
            # Fail before the loop, with the download instructions, rather than
            # after the first frame.
            from . import bat_lines_nets
            bat_lines_nets.ensure_weights(p["filter"], p)

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(n)
        except Exception:
            pbar = None

        out = None
        with _run_lock:
            for i in range(n):
                lines = run_filter(image[i], p, out_hw)
                if out is None:
                    out = torch.empty((n, lines.shape[0], lines.shape[1]), dtype=torch.float32)
                out[i] = torch.from_numpy(lines.astype(np.float32) / 255.0)
                if pbar is not None:
                    pbar.update(1)

        lines_img = out.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        ui = {"frames": [n], "src_w": [w], "src_h": [h],
              "out_w": [int(out.shape[2])], "out_h": [int(out.shape[1])]}
        if unique_id is not None:
            try:
                ui.update(self._cache_preview(unique_id, image, out,
                                              int(p["preview_frame"]), n, w, h))
            except Exception as exc:
                logger.warning("could not cache the preview frame (%s)", exc)
        return {"ui": stash_ui(ui), "result": (lines_img, out)}

    def _cache_preview(self, unique_id, image, out, preview_frame, n, w, h):
        idx = max(0, min(preview_frame, n - 1))
        frame = image[idx:idx + 1].detach()
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        frame = frame.clone()      # a view would keep the whole batch alive
        token = uuid.uuid4().hex[:16]
        _cache_put(token, unique_id, frame, image,
                   {"frames": n, "frame": idx, "w": w, "h": h})
        return {"token": [token], "preview_frame": [idx],
                "thumb": [_b64_jpeg_u8((out[idx].numpy() * 255.0 + 0.5).astype(np.uint8))],
                "source": [_b64_jpeg_u8(to_u8(frame[0]))]}


def _b64_jpeg_u8(arr, max_dim=768, quality=88):
    import base64
    im = Image.fromarray(arr)
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Preview endpoints
# ---------------------------------------------------------------------------

def _coerce(name, value, default):
    """A request value, as the type its widget has, clamped to its range."""
    spec = BatLines.INPUT_TYPES()["required"][name]
    kind, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    if isinstance(kind, list):
        return value if value in kind else default
    if kind == "BOOLEAN":
        return bool(value)
    try:
        v = (int if kind == "INT" else float)(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return max(opts.get("min", v), min(opts.get("max", v), v))


def params_from_request(body):
    p = _defaults()
    for k in list(p):
        if k in body:
            p[k] = _coerce(k, body[k], p[k])
    return p


def render_preview(entry, p, frame_index):
    """The preview's lines for one cached frame, at the size the node outputs."""
    frame, exact = _frame_for(entry, frame_index)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    out_hw = (h, w) if p["output_size"] == "input" else None
    with _run_lock:
        lines = run_filter(frame, p, out_hw)
    return lines, {"exact": exact, "w": int(lines.shape[1]), "h": int(lines.shape[0])}


def source_preview(entry, frame_index, max_dim=1600):
    """The cached source frame as 8-bit RGB (long edge capped), for the
    viewer's wipe. Returns (u8, exact) like _frame_for."""
    frame, exact = _frame_for(entry, frame_index)
    u8 = to_u8(frame)
    h, w = u8.shape[:2]
    if max(h, w) > max_dim:
        r = max_dim / max(h, w)
        u8 = cv2.resize(u8, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
    return u8, exact


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.get("/bat/lines/info")
    async def _bat_lines_info(request):
        entry = _cache_get(request.rel_url.query.get("token", ""))
        if entry is None:
            return web.json_response({"ok": False})
        ref = entry.get("batch")
        return web.json_response({
            "ok": True,
            "frames": int(entry["meta"].get("frames", 1)),
            "frame": int(entry["meta"].get("frame", 0)),
            "live_batch": bool(ref is not None and ref() is not None),
        })

    @server.PromptServer.instance.routes.post("/bat/lines/render")
    async def _bat_lines_render(request):
        """Re-run the filter on the cached frame. POST {token, frame, ...params}
        -> PNG of the lines, or 409 when nothing is cached for the token.
        With {"want": "source"} it returns that frame of the source as a JPEG
        instead, for the viewer's wipe."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        entry = _cache_get(str(body.get("token", "")))
        if entry is None:
            return web.json_response(
                {"error": "no cached frame for this token; run the node once"}, status=409)
        p = params_from_request(body)
        try:
            frame_index = int(body.get("frame", entry["meta"].get("frame", 0)))
        except (TypeError, ValueError):
            frame_index = int(entry["meta"].get("frame", 0))
        loop = asyncio.get_running_loop()
        if body.get("want") == "source":
            u8, exact = await loop.run_in_executor(None, source_preview, entry, frame_index)
            buf = io.BytesIO()
            Image.fromarray(u8).save(buf, format="JPEG", quality=90)
            return web.Response(body=buf.getvalue(), content_type="image/jpeg",
                                headers={"Cache-Control": "no-store",
                                         "X-Bat-Lines": json.dumps({"exact": exact})})
        try:
            lines, info = await loop.run_in_executor(None, render_preview, entry, p, frame_index)
        except Exception as exc:
            logger.warning("preview render failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
        buf = io.BytesIO()
        Image.fromarray(lines).save(buf, format="PNG", compress_level=1)
        return web.Response(body=buf.getvalue(), content_type="image/png",
                            headers={"Cache-Control": "no-store",
                                     "X-Bat-Lines": json.dumps(info)})

except ImportError as _exc:      # pragma: no cover - no ComfyUI server (tests)
    logger.debug("preview endpoints not registered (%s)", _exc)
except Exception as _exc:        # pragma: no cover
    logger.warning("could not register the preview endpoints: %s", _exc)
