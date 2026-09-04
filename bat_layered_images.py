"""
Bat_LayeredImages — stack several images with per-layer blend modes and masks.

What it is for
--------------
``Bat_AdvancedBlend`` does two plates very well and cannot do three. This does N,
which turns out to be what a real comp needs: a plate, a regraded version of it
through a matte, a glow pass on screen, a dirt layer on multiply. The frequency
separation is deliberately absent — that is the other node's job, and wiring one
into a layer of this one is the intended way to combine them.

Layer order
-----------
``image_1`` is the BOTTOM layer and higher numbers stack on top, which is the
order the words "layer on top of" imply. The on-node panel lists them the other
way up, top layer first, because that is how every compositing application shows
a stack and reading it the other way round is a constant small tax.

Compositing model
-----------------
Full W3C compositing, not a lerp chain. Each layer carries an alpha — its mask
times its opacity, or just its opacity where no mask is wired — and the
accumulator tracks alpha as well as colour::

    Cb = premultiplied backdrop / ab          (the backdrop as a colour)
    co = as*(1-ab)*Cs + as*ab*B(Cb,Cs) + (1-as)*co_prev
    ao = as + ab*(1-as)

Doing it properly rather than lerping matters at the bottom of the stack: layer 1
composites over nothing, so with a mask it is genuinely semi-transparent, and the
node's second output is that accumulated alpha. Send it and ``image`` to an
Uncrop or a downstream Merge and the result composites correctly over whatever is
under it. A lerp chain would have silently treated the bottom layer as opaque and
quietly baked black into the transparent parts.

The blend modes live in ``bat_blend_modes.py`` — the full W3C/Photoshop set
including the four non-separable colour modes — because the live preview needs
the identical formulas and a table per node is how those drift.

Per-layer settings
------------------
Mode, opacity and visibility are NOT widgets. They live in one hidden STRING
widget as JSON, the way ``Bat_Roto`` stores its shapes, and the artist edits them
in a layer panel on the node body.

That is a considered choice, not a shortcut. Per-layer widgets would mean
``mode_1``, ``opacity_1``, ``mode_2`` … growing and shrinking with the input
count — and ComfyUI restores ``widgets_values`` positionally, with at least one
code path (copy/paste) that ignores ``serialize = false``. A node whose widget
list changes shape is a node whose saved values shift, which is exactly the
corruption that hit ``Bat_AdvancedBlend``'s advanced toggle. One STRING widget in
a fixed position cannot do that, and a panel is a better UI for a stack anyway.

Range
-----
Range-agnostic like the rest of the pack: ``clamp_output`` is off by default and
nothing truncates on the way through. The caveat from the blend-mode module
applies — ``screen``, ``overlay`` and friends are defined against white = 1.0, so
on scene-linear input above white they compute but stop meaning what they say.
``normal``, ``plus``, ``minus``, ``darken`` and ``lighten`` are the safe ones
there.
"""

import asyncio
import base64
import json
import logging
import threading
from collections import OrderedDict
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .bat_blend_modes import MODES, NON_SEPARABLE, SEPARABLE, blend
from .bat_hdr_preview import hdr_tile

logger = logging.getLogger("[Bat_LayeredImages]")

EPS = 1e-6

# Hard ceiling on layers. Eight is already a busy comp, and every one costs a
# full-resolution buffer in the preview cache.
MAX_LAYERS = 8

# Long edge of the draft preview tiles. Smaller than Bat_AdvancedBlend's 768
# because a single payload carries one per connected layer — up to eight lossless
# 16-bit buffers — and unlike that node there is no high-frequency judgement here
# that demands the extra pixels. The full-resolution server layer covers detail.
PREVIEW_TILE_DIM = 512

# Frames per chunk. A 4K layer is ~100MB of float32 and the accumulator needs two
# more, so a deep stack on a long clip has to be metered.
CHUNK_FRAMES = 4

RESIZE_MODES = ["match_first", "match_largest", "match_smallest"]
RESIZE_FILTERS = ["lanczos", "bicubic", "bilinear", "area", "nearest"]

DEFAULT_LAYER = {"mode": "normal", "opacity": 1.0, "enabled": True}


# ---------------------------------------------------------------------------
# Layer state
# ---------------------------------------------------------------------------

def parse_layers(text: str, count: int):
    """Decode the layer panel's JSON into exactly `count` sane entries.

    Forgiving by design: this string is written by the frontend and stored in a
    workflow, so it can be absent, truncated, from an older version, or hand
    edited. Anything unusable falls back to the default layer rather than
    failing the run — a missing opacity should not cost someone a render.
    """
    doc = None
    if text:
        try:
            doc = json.loads(text)
        except (TypeError, ValueError):
            logger.warning("layer state is not valid JSON; using defaults")
    raw = (doc or {}).get("layers")
    if not isinstance(raw, list):
        raw = []

    out = []
    for i in range(count):
        entry = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        mode = entry.get("mode", DEFAULT_LAYER["mode"])
        if mode not in MODES:
            if mode is not None and i < len(raw):
                logger.warning("layer %d: unknown blend mode %r; using normal",
                               i + 1, mode)
            mode = "normal"
        try:
            opacity = float(entry.get("opacity", 1.0))
        except (TypeError, ValueError):
            opacity = 1.0
        if not np.isfinite(opacity):
            opacity = 1.0
        out.append({
            "mode": mode,
            "opacity": max(0.0, min(1.0, opacity)),
            "enabled": bool(entry.get("enabled", True)),
        })
    return out


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite(layers, settings, clamp_output=False):
    """Composite a bottom-to-top list of (image, alpha) onto transparency.

    `layers` is [(rgb (N,H,W,3), alpha (N,H,W,1) or None), ...] bottom first.
    `settings` is the matching list of {mode, opacity, enabled}.

    Returns (rgb, alpha). See the module docstring for the algebra; the shape of
    the loop is the W3C one, kept literally so it can be checked against the
    spec rather than reverse-engineered.
    """
    ref = layers[0][0]
    acc = torch.zeros_like(ref)                       # premultiplied colour
    ab = torch.zeros_like(ref[..., :1])               # accumulated alpha

    for (cs, mask), cfg in zip(layers, settings):
        if not cfg["enabled"] or cfg["opacity"] <= 0.0:
            continue
        a_s = mask if mask is not None else torch.ones_like(ab)
        if cfg["opacity"] != 1.0:
            a_s = a_s * cfg["opacity"]

        # The backdrop as a colour rather than premultiplied, which is what the
        # blend functions expect. Where nothing has been laid down yet this is
        # 0 and the blend term is gated to nothing by `ab` anyway.
        cb = acc / ab.clamp(min=EPS)
        blended = blend(cb, cs, cfg["mode"])

        acc = a_s * (1.0 - ab) * cs + a_s * ab * blended + (1.0 - a_s) * acc
        ab = a_s + ab * (1.0 - a_s)

    out = acc / ab.clamp(min=EPS)
    # Un-premultiply leaves the fully transparent pixels undefined; they carry
    # whatever the division produced. Zero them so the RGB output is clean where
    # the alpha says there is nothing.
    out = torch.where(ab > 0.0, out, torch.zeros_like(out))
    if clamp_output:
        out = out.clamp(0.0, 1.0)
    return out, ab


# ---------------------------------------------------------------------------
# Resampling / helpers (shared with Bat_AdvancedBlend)
# ---------------------------------------------------------------------------

from .bat_advanced_blend import _num, _resize          # noqa: E402


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = PREVIEW_TILE_DIM,
              quality: int = 82) -> str:
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prep_mask(mask, n, device, dtype):
    """MASK -> (N,H,W) at its own resolution, frame count aligned to `n`."""
    if mask is None:
        return None
    m = mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and n > 1:
        m = m.expand(n, -1, -1)
    elif m.shape[0] < n:
        m = torch.cat([m, m[-1:].expand(n - m.shape[0], -1, -1)], dim=0)
    elif m.shape[0] > n:
        m = m[:n]
    return m


class BatLayeredImages:
    """Stack N images with per-layer blend modes, masks and opacity."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(MAX_LAYERS):
            opt[f"image_{i + 1}"] = ("IMAGE", {
                "tooltip": f"Layer {i + 1}. image_1 is the BOTTOM of the stack; "
                           f"higher numbers sit on top."})
            opt[f"mask_{i + 1}"] = ("MASK", {
                "tooltip": f"Optional alpha for layer {i + 1}. White = this layer "
                           f"shows, black = what is underneath shows. Multiplied "
                           f"with the layer's opacity."})
        opt["resize_mode"] = (RESIZE_MODES, {
            "default": "match_first",
            "tooltip": "Layers are rarely all the same size. This picks the "
                       "working resolution: match_first uses image_1's, which is "
                       "usually the plate everything else is being laid onto."})
        opt["resize_filter"] = (RESIZE_FILTERS, {
            "default": "lanczos",
            "tooltip": "Kernel used to conform the other layers. Runs in float32 "
                       "and preserves values above white."})
        opt["clamp_output"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Clamp the result to [0,1]. OFF by default so scene-linear "
                       "and HDR layers pass through with their range intact."})
        # The layer panel's state. A STRING in a FIXED position, deliberately —
        # see the module docstring on why this is not a set of per-layer widgets.
        opt["layers"] = ("STRING", {
            "default": "{\"layers\":[]}",
            "tooltip": "Per-layer mode / opacity / visibility, as JSON. Edited "
                       "through the panel on the node body; there is no reason to "
                       "type in here, but it is readable and diffable if you want "
                       "to."})
        opt["preview_frame"] = ("INT", {
            "default": 0, "min": 0, "max": 9999,
            "tooltip": "Which frame of the batch the on-node preview shows. "
                       "Display only, and it takes effect on the next Run."})
        return {"optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    OUTPUT_TOOLTIPS = (
        "The composited stack.",
        "Accumulated alpha. Only below 1 where the bottom layer is masked or "
        "partly transparent — send it downstream with the image to composite the "
        "result correctly over something else.",
    )
    FUNCTION = "run"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Stack up to 8 images with per-layer blend mode, opacity and optional "
        "mask. image_1 is the bottom. Full W3C/Photoshop mode set including the "
        "non-separable colour modes. Layers are edited in a panel on the node "
        "body, with a live preview of the composite and per-layer solo."
    )

    # ------------------------------------------------------------------
    @staticmethod
    def _collect(kwargs, n_declared=MAX_LAYERS):
        """Gather the connected layers, bottom first, closing any gaps.

        A gap is normal — someone unwires layer 2 of four — and the stack should
        simply become three layers rather than growing a black hole in the
        middle. The panel is told the original slot numbers so its rows stay
        recognisable.
        """
        found = []
        for i in range(n_declared):
            img = kwargs.get(f"image_{i + 1}")
            if img is None:
                continue
            found.append((i, img, kwargs.get(f"mask_{i + 1}")))
        return found

    @staticmethod
    def _target_size(images, mode):
        sizes = [(int(t.shape[1]), int(t.shape[2])) for t in images]
        if mode == "match_largest":
            return max(sizes, key=lambda s: s[0] * s[1])
        if mode == "match_smallest":
            return min(sizes, key=lambda s: s[0] * s[1])
        return sizes[0]

    # ------------------------------------------------------------------
    def run(self, resize_mode="match_first", resize_filter="lanczos",
            clamp_output=False, layers="", preview_frame=0, unique_id=None,
            **kwargs):
        found = self._collect(kwargs)
        if not found:
            raise ValueError(
                "Bat_LayeredImages: no image inputs connected. Wire at least "
                "image_1 — it is the bottom of the stack.")

        if resize_mode not in RESIZE_MODES:
            resize_mode = "match_first"
        if resize_filter not in RESIZE_FILTERS:
            resize_filter = "lanczos"

        imgs = []
        masks = []
        for _, img, mask in found:
            t = img.to(torch.float32)
            if t.shape[-1] > 3:
                # Drop any alpha the upstream node carried: it would otherwise
                # be blended as a colour channel. A real alpha belongs on the
                # mask input, where it can act as one.
                t = t[..., :3]
            imgs.append(t)
            masks.append(mask)

        device = imgs[0].device
        imgs = [t.to(device) for t in imgs]

        # Frame counts: a single frame broadcasts (a still over a clip), and
        # otherwise the stack is truncated to the shortest with a warning rather
        # than silently holding or looping.
        counts = [int(t.shape[0]) for t in imgs]
        n = max(counts)
        if len(set(counts)) > 1:
            multi = [c for c in counts if c > 1]
            if multi and min(multi) != max(multi):
                n = min(multi)
                logger.warning("layers disagree on frame count (%s); using the "
                               "first %d", counts, n)
            else:
                n = max(counts)
        imgs = [t.expand(n, -1, -1, -1) if t.shape[0] == 1 else t[:n] for t in imgs]

        out_h, out_w = self._target_size(imgs, resize_mode)
        settings = parse_layers(layers, len(imgs))
        prepared_masks = [_prep_mask(m, n, device, torch.float32) for m in masks]

        idx = max(0, min(int(preview_frame), n - 1))
        pv = None

        outs, alphas = [], []
        for s in range(0, n, CHUNK_FRAMES):
            e = min(s + CHUNK_FRAMES, n)
            chunk = []
            for t, m in zip(imgs, prepared_masks):
                c = _resize(t[s:e], out_h, out_w, resize_filter)
                a = None
                if m is not None:
                    a = m[s:e].unsqueeze(-1)
                    if a.shape[1:3] != (out_h, out_w):
                        a = torch.nn.functional.interpolate(
                            a.permute(0, 3, 1, 2), size=(out_h, out_w),
                            mode="bilinear", align_corners=False,
                        ).permute(0, 2, 3, 1)
                chunk.append((c, a))
            o, al = composite(chunk, settings, clamp_output)
            outs.append(o)
            alphas.append(al)
            if pv is None and s <= idx < e:
                k = idx - s
                # .clone(), not a view: a view pins the whole chunk, which for a
                # deep stack at 4K is most of a gigabyte.
                pv = [(c[k:k + 1].clone(),
                       None if a is None else a[k:k + 1].clone())
                      for (c, a) in chunk]

        out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        alpha = torch.cat(alphas, dim=0) if len(alphas) > 1 else alphas[0]

        ui = self._preview_payload(pv, found, settings, idx, n, out_w, out_h,
                                   unique_id)
        return {"ui": ui, "result": (out, alpha[..., 0])}

    # ------------------------------------------------------------------
    def _preview_payload(self, pv, found, settings, idx, frames, out_w, out_h,
                         unique_id):
        """Tiles and metadata for the two preview layers.

        One tile per layer, all conformed and all off the same area-sampler, so
        the JS can composite them with a single counter. Masks ride along as
        8-bit PNGs rather than 16-bit tiles: a matte's precision is not what this
        node is judged on, and eight more lossless buffers would double an
        already-large payload.
        """
        if not pv:
            return {}
        ui = {
            "slots": [[int(i) + 1 for (i, _, _) in found]],
            "settings": [settings],
            "frames": [int(frames)],
            "preview_frame": [int(idx)],
            "w": [int(out_w)],
            "h": [int(out_h)],
            "modes": [list(MODES)],
        }

        tiles, mask_pngs = [], []
        for (c, a) in pv:
            t = hdr_tile(c[0], PREVIEW_TILE_DIM, sample="area")
            if t is None:
                logger.warning("could not build a preview tile; the live canvas "
                               "will stay on its 8-bit fallback this run")
                tiles = []
                break
            tiles.append(t)
            if a is None:
                mask_pngs.append(None)
            else:
                u8 = (a[0, ..., 0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
                im = Image.fromarray(u8, "L")
                if max(im.size) > PREVIEW_TILE_DIM:
                    r = PREVIEW_TILE_DIM / max(im.size)
                    im = im.resize((max(1, int(im.width * r)),
                                    max(1, int(im.height * r))), Image.BILINEAR)
                buf = BytesIO()
                im.save(buf, format="PNG", compress_level=3)
                mask_pngs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        if tiles:
            ui["tiles"] = [tiles]
            ui["masks"] = [mask_pngs]

        # 8-bit fallback of the bottom layer, so a reopened workflow has
        # something to show before the first run.
        u8 = (pv[0][0][0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        ui["jpeg_base"] = [_b64_jpeg(u8)]

        if unique_id is not None:
            try:
                _cache_put(str(unique_id), pv, settings,
                           {"w": int(out_w), "h": int(out_h),
                            "frames": int(frames), "frame": int(idx)})
            except Exception as exc:
                logger.warning("could not cache the frame for full-resolution "
                               "preview: %s", exc)
        return ui


# ---------------------------------------------------------------------------
# Full-resolution preview service
# ---------------------------------------------------------------------------
#
# Same two-layer arrangement as Bat_AdvancedBlend, and the same reasoning: the
# browser composites downscaled tiles for instant feedback, and this renders the
# real thing at real resolution over the region on screen a moment later. It is
# not a mirror of the render, it IS the render.

CACHE_MAX_BYTES = 900 << 20

_cache = OrderedDict()
_cache_lock = threading.Lock()


def _entry_bytes(entry):
    n = 0
    for (c, a) in entry["layers"]:
        n += c.numel() * c.element_size()
        if a is not None:
            n += a.numel() * a.element_size()
    return n


def _cache_put(node_id, pv, settings, meta):
    """Cache the conformed preview frame, CPU-side.

    A deep stack is the expensive case here — eight 4K layers is ~800MB — so the
    LRU matters more than it does for the two-plate node. Holding it in VRAM
    would be taking that away from whatever the artist is actually rendering.
    """
    entry = {
        "layers": [(c.detach().to("cpu", torch.float32).contiguous(),
                    None if a is None else a.detach().to("cpu", torch.float32).contiguous())
                   for (c, a) in pv],
        "settings": [dict(s) for s in settings],
        "meta": meta,
    }
    with _cache_lock:
        _cache.pop(node_id, None)
        _cache[node_id] = entry
        total = sum(_entry_bytes(e) for e in _cache.values())
        while len(_cache) > 1 and total > CACHE_MAX_BYTES:
            _, dropped = _cache.popitem(last=False)
            total -= _entry_bytes(dropped)
            logger.debug("preview cache: evicted an entry to stay under %d MB",
                         CACHE_MAX_BYTES >> 20)


def _cache_get(node_id):
    with _cache_lock:
        entry = _cache.get(node_id)
        if entry is not None:
            _cache.move_to_end(node_id)
        return entry


MAX_OUT_DIM = 4096


def render_region(entry, settings, roi, out_w, out_h, view="result",
                  clamp_output=False):
    """Composite one region of the cached frame at full resolution.

    `view` is "result" or "layer:<i>" to solo one layer. Unlike
    Bat_AdvancedBlend there is no blur here, so no context margin is needed —
    every operation is per-pixel, and a region is exactly independent of its
    surroundings.
    """
    layers = entry["layers"]
    fh, fw = layers[0][0].shape[1], layers[0][0].shape[2]

    x, y, w, h = (int(v) for v in roi)
    x = max(0, min(x, max(fw - 1, 0)))
    y = max(0, min(y, max(fh - 1, 0)))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))

    device = torch.device("cuda") if torch.cuda.is_available() else layers[0][0].device

    def slab(t):
        return None if t is None else t[:, y:y + h, x:x + w].to(device, non_blocking=True)

    try:
        chunk = [(slab(c), slab(a)) for (c, a) in layers]
        if view.startswith("layer:"):
            i = int(view.split(":", 1)[1])
            if not (0 <= i < len(chunk)):
                raise ValueError(f"no layer {i}")
            # Solo: that layer alone over transparency, at full opacity and
            # `normal`, so what you see is the layer itself rather than the layer
            # as its mode happens to render it against nothing.
            img, _ = composite([chunk[i]], [dict(DEFAULT_LAYER)], clamp_output)
        else:
            img, _ = composite(chunk, settings, clamp_output)
    except torch.cuda.OutOfMemoryError:
        logger.warning("full-resolution preview did not fit in VRAM; falling "
                       "back to CPU for this request")
        torch.cuda.empty_cache()
        chunk = [(None if c is None else c[:, y:y + h, x:x + w],
                  None if a is None else a[:, y:y + h, x:x + w])
                 for (c, a) in layers]
        if view.startswith("layer:"):
            i = int(view.split(":", 1)[1])
            img, _ = composite([chunk[i]], [dict(DEFAULT_LAYER)], clamp_output)
        else:
            img, _ = composite(chunk, settings, clamp_output)

    if (out_h, out_w) != (img.shape[1], img.shape[2]):
        filt = "area" if (out_w < img.shape[2] or out_h < img.shape[1]) else "lanczos"
        img = _resize(img, max(1, int(out_h)), max(1, int(out_w)), filt)

    return (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _settings_from_request(body, count):
    """Layer settings from the client's JSON, coerced rather than trusted."""
    raw = body.get("settings")
    if not isinstance(raw, list):
        raw = []
    out = []
    for i in range(count):
        entry = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        mode = str(entry.get("mode", "normal"))
        if mode not in MODES:
            mode = "normal"
        out.append({
            "mode": mode,
            "opacity": _num(entry.get("opacity", 1.0), 1.0, 0.0, 1.0),
            "enabled": bool(entry.get("enabled", True)),
        })
    return out


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/bat/layered_images/render")
    async def _bat_layered_render(request):
        """Composite one region of a node's cached stack at full resolution."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        entry = _cache_get(str(body.get("node_id", "")))
        if entry is None:
            return web.json_response(
                {"error": "no cached frame for this node; run it once"}, status=409)

        try:
            layers = entry["layers"]
            settings = _settings_from_request(body, len(layers))
            roi = body.get("roi") or [0, 0, layers[0][0].shape[2],
                                      layers[0][0].shape[1]]
            out_w = max(1, min(int(body.get("out_w") or 512), MAX_OUT_DIM))
            out_h = max(1, min(int(body.get("out_h") or 512), MAX_OUT_DIM))
            view = str(body.get("view", "result"))
            clamp = bool(body.get("clamp_output", False))
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": f"bad request: {exc}"}, status=400)

        loop = asyncio.get_running_loop()
        try:
            # Off the event loop: a full-resolution composite of a deep stack is
            # a good fraction of a second, and blocking here would stall every
            # other websocket message including execution progress.
            u8 = await loop.run_in_executor(
                None, render_region, entry, settings, roi, out_w, out_h, view, clamp)
        except Exception as exc:
            logger.warning("full-resolution preview failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

        buf = BytesIO()
        Image.fromarray(u8, "RGB").save(buf, format="PNG", compress_level=1)
        return web.Response(body=buf.getvalue(), content_type="image/png",
                            headers={"Cache-Control": "no-store"})

except ImportError as _exc:    # pragma: no cover - import-time only
    logger.debug("full-resolution preview endpoint not registered (%s)", _exc)
except Exception as _exc:      # pragma: no cover - import-time only
    logger.warning("could not register the full-resolution preview endpoint; "
                   "the on-node canvas will stay on its draft layer: %s", _exc)
