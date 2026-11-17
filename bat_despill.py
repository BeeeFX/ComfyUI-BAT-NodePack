"""
Bat_Despill — despill a keyed foreground and review the key, with an on-node viewer.
===================================================================================

Takes a STRAIGHT (unpremultiplied) foreground and its alpha from any keyer or
matting model — CorridorKey 2, MEMatte, SAM + a matting pass — and returns:

    foreground     straight, despilled
    premultiplied  foreground x alpha
    comp           foreground over checker / black / grey / white, in linear light

Built for CorridorKey 2 (2026-11-05): despill is the last thing its key node
does to each frame, but as a widget there, changing it re-keys the whole clip.
Set the key node's despill_strength to 0 and wire it in here, and only this
cheap node re-runs. ComfyUI-BAT-CorridorKey2's tests/test_bat_despill_parity.py
checks that this node reproduces the key node's despill and checker preview.

The despill is the classic "average" despill — spill = G - (R + B) / 2,
clamped at 0, taken out of G and given half each to R and B — mirrored for
blue, blended with the original by strength. The comp is done in linear light
(IEC 61966-2-1 sRGB, sign-mirrored for negative values) over a 64 px
0.75 / 0.5 checker, matching CorridorKey 2's preview. Everything here is BAT's
own code (MIT); nothing is taken from CorridorKey.

Why the straight foreground looks soft on its own: it is meant to be multiplied
by alpha. Where alpha is near zero the colour has no effect in a comp, so keyers
carry the subject's colour on past the matte edge (on the shot this was built
for, 10-20 px of hair colour where alpha < 0.05), and nothing pins it from frame
to frame. Judge and render from the premultiplied or comp outputs.

The viewer is SERVER-rendered: every view is drawn by ``render_view``, the code
that makes the outputs, over the visible region only — zoomed in you see real
pixels, and the viewer cannot drift from the render. Frames are reached through
a per-run token holding WEAK references to the input batches, so the viewer
costs no memory beyond ComfyUI's own cache and says so once ComfyUI drops them.
"""

import asyncio
import hashlib
import io
import logging
import threading
import time
import weakref
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

COLORSPACES = ["sRGB", "linear"]
SCREENS = ["green", "blue"]
BACKGROUNDS = ["checker", "black", "grey", "white"]
VIEWS = ["comp", "alpha", "foreground", "premultiplied", "spill", "plate"]
_BG_SRGB = {"black": 0.0, "grey": 0.5, "white": 1.0}
CHECKER_SIZE, CHECKER_LIGHT, CHECKER_DARK = 64, 0.75, 0.5
SPILL_GAIN = 4.0          # the spill view is |raw - despilled| x this, so small removals show


# ── colour ──────────────────────────────────────────────────────────────────

def srgb_to_linear(x):
    """IEC 61966-2-1 decode, mirrored for negative values (scene-linear can be < 0)."""
    a = x.abs()
    lin = torch.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055).pow(2.4))
    return torch.where(x >= 0, lin, -lin)


def linear_to_srgb(x):
    """IEC 61966-2-1 encode, mirrored for negative values."""
    a = x.abs()
    enc = torch.where(a <= 0.0031308, a * 12.92, 1.055 * a.clamp(min=0.0031308).pow(1.0 / 2.4) - 0.055)
    return torch.where(x >= 0, enc, -enc)


def despill(fg, strength, screen="green"):
    """Average despill on [..., 3] channels-last RGB. strength 0 = untouched, 1 = full."""
    strength = float(strength)
    if strength <= 0.0:
        return fg
    r, g, b = fg[..., 0:1], fg[..., 1:2], fg[..., 2:3]
    if screen == "blue":
        spill = (b - (r + g) * 0.5).clamp(min=0.0)
        full = torch.cat([r + spill * 0.5, g + spill * 0.5, b - spill], dim=-1)
    else:
        spill = (g - (r + b) * 0.5).clamp(min=0.0)
        full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=-1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def checker_srgb(h, w, y0=0, x0=0, device=None):
    """The checker for the frame region starting at (y0, x0), so a crop matches the full frame."""
    y = torch.arange(y0, y0 + h, device=device).view(-1, 1)
    x = torch.arange(x0, x0 + w, device=device).view(1, -1)
    light = (((y // CHECKER_SIZE) + (x // CHECKER_SIZE)) % 2 == 0).float()
    return (light * CHECKER_LIGHT + (1.0 - light) * CHECKER_DARK).expand(h, w)


def background(name, h, w, y0=0, x0=0, device=None):
    """The comp background in linear light, [H,W,1]."""
    if name == "checker":
        bg = checker_srgb(h, w, y0, x0, device)
    else:
        bg = torch.full((h, w), _BG_SRGB.get(name, 0.5), device=device)
    return srgb_to_linear(bg)[..., None]


def composite(fg, alpha, bg_name, linear, y0=0, x0=0):
    """Foreground over the background in LINEAR light. sRGB in -> clipped sRGB out
    (like CorridorKey 2's preview); linear in -> unclamped linear out."""
    a = alpha.clamp(0.0, 1.0)[..., None]
    bg = background(bg_name, fg.shape[-3], fg.shape[-2], y0, x0, fg.device)
    fg_lin = fg if linear else srgb_to_linear(fg)
    comp = fg_lin * a + bg * (1.0 - a)
    return comp if linear else linear_to_srgb(comp.clamp(min=0.0)).clamp(0.0, 1.0)


def to_display(x, linear):
    """Anything in the foreground's encoding -> sRGB 0-1 for the viewer."""
    if linear:
        x = linear_to_srgb(x.clamp(min=0.0))
    return x.clamp(0.0, 1.0)


def render_view(fg, alpha, plate, view, strength, screen, bg_name, linear, y0=0, x0=0):
    """One view of one frame region, display sRGB [H,W,3]. Same maths as the outputs."""
    clean = despill(fg, strength, screen)
    a = alpha.clamp(0.0, 1.0)
    if view == "alpha":
        return a[..., None].expand(*a.shape, 3)
    if view == "foreground":
        return to_display(clean, linear)
    if view == "premultiplied":
        return to_display(clean * a[..., None], linear)
    if view == "spill":
        removed = (fg - clean).abs().amax(-1, keepdim=True) * a[..., None]
        return (removed * SPILL_GAIN).clamp(0.0, 1.0).expand(*removed.shape[:-1], 3)
    if view == "plate" and plate is not None:
        return to_display(plate[..., :3], linear)
    out = composite(clean, alpha, bg_name, linear, y0, x0)
    return to_display(out, linear) if linear else out


# ── run tokens: the viewer's way back to the frames ─────────────────────────

class Runs:
    MAX = 8

    def __init__(self):
        self._lock = threading.Lock()
        self._runs = OrderedDict()

    @staticmethod
    def _ref(t):
        if t is None:
            return lambda: None
        try:
            return weakref.ref(t)
        except TypeError:
            return lambda: t

    def register(self, fg, alpha, plate, linear):
        token = hashlib.sha1(f"{id(fg)}:{time.time()}".encode()).hexdigest()[:16]
        with self._lock:
            self._runs[token] = {"fg": self._ref(fg), "alpha": self._ref(alpha),
                                 "plate": self._ref(plate), "linear": linear,
                                 "count": int(fg.shape[0]), "h": int(fg.shape[1]),
                                 "w": int(fg.shape[2]), "has_plate": plate is not None}
            while len(self._runs) > self.MAX:
                self._runs.popitem(last=False)
        return token

    def get(self, token):
        with self._lock:
            e = self._runs.get(token)
            if e is not None:
                self._runs.move_to_end(token)
            return e


RUNS = Runs()


def _render_device():
    """GPU for the viewer while ComfyUI is idle, CPU while a job runs.

    A full-frame comp is ~6 ms on an A5000 vs ~100 ms on CPU (same pixels to
    2e-7), but a running job — a CorridorKey 2 key sizes its window from free
    VRAM — must never be pushed out of memory by a viewer render.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        import server
        if server.PromptServer.instance.prompt_queue.currently_running:
            return torch.device("cpu")
    except Exception:
        pass
    return torch.device("cuda", torch.cuda.current_device())


def view_rgb(body):
    """Render a viewer request -> ([H,W,3] uint8, None) or (None, error)."""
    e = RUNS.get(str(body.get("token", "")))
    if e is None:
        return None, "unknown run — run the node again"
    fg, alpha, plate = e["fg"](), e["alpha"](), e["plate"]()
    if fg is None or alpha is None:
        return None, "ComfyUI has dropped this run's frames — run the node again"
    i = max(0, min(int(body.get("frame", 0)), e["count"] - 1))
    H, W = e["h"], e["w"]
    x, y, w, h = (int(round(float(v))) for v in (body.get("roi") or [0, 0, W, H]))
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
    out_w = max(1, min(int(body.get("out_w", w)), 4096))
    out_h = max(1, min(int(body.get("out_h", h)), 4096))
    view = body.get("view") if body.get("view") in VIEWS else "comp"
    strength = 0.0 if body.get("raw") else float(body.get("strength", 0.5))

    pick = lambda b: None if b is None else b[min(i, b.shape[0] - 1)]
    a_frame = pick(alpha)
    if a_frame.dim() == 3:
        a_frame = a_frame[..., 0]
    dev = _render_device()
    crop = lambda t: None if t is None else t[y:y + h, x:x + w].to(dev, non_blocking=True).float()
    img = render_view(crop(pick(fg)[..., :3]), crop(a_frame), crop(pick(plate)), view, strength,
                      body.get("screen", "green"), body.get("background", "checker"), e["linear"],
                      y0=y, x0=x)
    chw = img.permute(2, 0, 1).unsqueeze(0)
    # Full resolution first, THEN resize: reducing first would despill averaged
    # pixels, and despill is non-linear (measured up to 88/255 off on the spill view).
    if out_w < w or out_h < h:
        chw = F.interpolate(chw, size=(out_h, out_w), mode="area")
    elif out_w != w or out_h != h:
        chw = F.interpolate(chw, size=(out_h, out_w), mode="nearest")   # zoomed in: real pixels
    return (chw[0].permute(1, 2, 0).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy(), None


def view_png(body):
    """view_rgb as PNG bytes (tests, and any client that wants an image file)."""
    u8, err = view_rgb(body)
    if u8 is None:
        return None, err
    buf = io.BytesIO()
    Image.fromarray(u8, "RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), None


class BatDespill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE", {
                    "tooltip": "Straight (unpremultiplied) foreground, e.g. CorridorKey 2's with its "
                               "despill_strength at 0."}),
                "alpha": ("MASK", {"tooltip": "The matte for that foreground."}),
                "colorspace": (COLORSPACES, {
                    "default": "sRGB",
                    "tooltip": "The foreground's encoding. linear keeps foreground and comp unclamped "
                               "scene-linear. For CorridorKey 2, match its input_colorspace.",
                }),
                "screen": (SCREENS, {"default": "green",
                                     "tooltip": "Which colour to pull out of the foreground."}),
                "despill_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 = untouched, 1 = full average despill. Only this node re-runs "
                               "when it changes.",
                }),
                "background": (BACKGROUNDS, {
                    "default": "checker",
                    "tooltip": "What the 'comp' output is laid over. checker = CorridorKey 2's preview.",
                }),
            },
            "optional": {
                "plate": ("IMAGE", {"tooltip": "Optional original clip, for the viewer's Plate view."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "premultiplied", "comp")
    OUTPUT_TOOLTIPS = (
        "Straight (unpremultiplied) foreground, despilled. Multiply by alpha before laying it over "
        "anything — on its own its edges look soft: the colour carries on past the matte.",
        "Foreground x alpha, in the foreground's encoding. Lay over a background with 'over'.",
        "Foreground over the chosen background, composited in linear light.",
    )
    FUNCTION = "run"
    CATEGORY = "BAT/mask"
    DESCRIPTION = (
        "Despill a keyed foreground and review the key. Outputs the despilled straight "
        "foreground, the premultiplied foreground and a comp. The viewer shows comp / alpha / "
        "foreground / premultiplied / spill removed / plate, with zoom to real pixels and hold-B "
        "to compare with no despill.")

    def run(self, foreground, alpha, colorspace, screen, despill_strength, background, plate=None):
        if alpha.dim() == 4:
            alpha = alpha[..., 0]
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(0)
        if foreground.shape[0] != alpha.shape[0] or foreground.shape[1:3] != alpha.shape[1:3]:
            raise ValueError(f"foreground {tuple(foreground.shape[:3])} and alpha {tuple(alpha.shape)} "
                             f"must be the same frames at the same size.")
        linear = colorspace == "linear"
        fg = foreground[..., :3].float()
        clean, premult, comp = [], [], []
        for i in range(fg.shape[0]):          # per frame: a 4K batch x 3 temporaries adds up
            c = despill(fg[i], despill_strength, screen)
            a = alpha[i].float()
            clean.append(c)
            premult.append(c * a.clamp(0.0, 1.0)[..., None])
            comp.append(composite(c, a, background, linear))
        token = RUNS.register(foreground, alpha, plate, linear)
        ui = {"despill_token": [token], "w": [int(fg.shape[2])], "h": [int(fg.shape[1])],
              "frame_count": [int(fg.shape[0])], "has_plate": [plate is not None]}
        return {"ui": ui, "result": (torch.stack(clean), torch.stack(premult), torch.stack(comp))}


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/bat/despill/view")
    async def _bat_despill_view(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        try:
            t0 = time.perf_counter()
            u8, err = await asyncio.to_thread(view_rgb, body)
        except Exception as exc:
            logger.warning("Bat_Despill: viewer render failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
        if u8 is None:
            return web.json_response({"error": err}, status=410)
        # Raw RGB, not PNG: PNG encoding was 84 of ~100 ms per render (the despill and
        # comp are ~7 ms on the GPU). Lossless either way; ~2 MB is nothing locally.
        return web.Response(body=u8.tobytes(), content_type="application/octet-stream", headers={
            "Cache-Control": "no-store", "X-Width": str(u8.shape[1]), "X-Height": str(u8.shape[0]),
            "X-Render-Ms": f"{(time.perf_counter() - t0) * 1000:.0f}"})

    @server.PromptServer.instance.routes.get("/bat/despill/info")
    async def _bat_despill_info(request):
        e = RUNS.get(request.rel_url.query.get("token", ""))
        if e is None or e["fg"]() is None or e["alpha"]() is None:
            return web.json_response({"ok": False})
        return web.json_response({"ok": True, "count": e["count"], "w": e["w"], "h": e["h"],
                                  "has_plate": e["has_plate"] and e["plate"]() is not None})

except ImportError:          # pragma: no cover — tests / standalone import
    pass
except Exception as _exc:    # pragma: no cover — server module without a running instance
    logger.warning("Bat_Despill: viewer routes not registered: %s", _exc)
