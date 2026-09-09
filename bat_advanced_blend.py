"""
Bat_AdvancedBlend — two-plate blend with optional frequency separation and a
live on-node preview.

What it is for
--------------
On the face of it this is an ordinary compositing blend: two IMAGE inputs, a
mode dropdown, an opacity, an optional mask. That is the whole node if you
leave ``frequency_separation`` off, and it is a perfectly good reason to use
it.

The reason it exists, though, is the case a plain blend cannot solve: dialling
back an upscale. SEEDVR2 (and every refiner like it) hands back a picture that
is genuinely more detailed than the original *and* too sharp — over-crisp
edges, invented micro-texture, a plasticky grain floor. Cross-dissolving it
with the original at 60% doesn't fix that, because it takes 60% of the
upscale's *structure* along with 60% of its sharpness. The two things you want
to control separately are welded together.

Frequency separation unwelds them. Split each plate into a low band (the blur
— tone, colour, large-scale structure) and a high band (everything the blur
threw away — edges, texture, grain), then mix the bands independently:

* ``low_mix 1.0`` / ``high_mix 0.4`` — keep the upscale's structure and
  resolution, take only 40% of its sharpening. The usual answer to "SEEDVR2
  made it too sharp".
* ``low_mix 0.0`` / ``high_mix 1.0`` — the original's tone and colour
  untouched, the upscale's detail laid on top. "Add only the detail of plate A
  onto plate B."
* ``soften_a`` / ``soften_b`` — Gaussian-blur either plate *before* any of
  this. A blur is a lowpass, so this is exactly "take the high frequency out
  of plate A (or B) before the blend", with a radius rather than a switch.
  Useful for killing an upscaler's grain floor at source instead of
  attenuating it afterwards.

The three domains
-----------------
Resolutions almost never match here — that is the point of an upscale — so
the node conforms them first (``resize_mode`` / ``resize_filter``), and every
radius the artist types is in pixels **of the conformed working resolution**,
not of either input.

The resamplers are implemented here rather than borrowed from
``comfy.utils.common_upscale`` for one specific reason: that module's
``lanczos`` path round-trips through PIL, which clamps to [0,1] and quantises
to 8-bit. Passing the original plate through that before a frequency-
separation subtract injects quantisation steps straight into the low band —
which is the one artefact this node exists to avoid manufacturing. The
separable float resampler below stays in float32 and preserves values above
white, so a scene-linear plate survives the conform.

Two detail models
-----------------
``detail_mode`` picks how a band is extracted and put back:

* ``subtract`` — ``high = x - blur(x)``, recombined by addition. The classic
  retouching split. Signed, symmetric, and the right default.
* ``divide`` — ``high = x / blur(x)``, recombined by multiplication. Scale
  invariant: the same texture over a bright region and a dark one produces the
  same ratio, so transferring detail into shadows doesn't crush them and into
  highlights doesn't blow them. Worth reaching for on scene-linear plates, and
  on shots with a very wide tonal range. ``detail_limit`` bounds it, because a
  ratio against a near-black lowpass is where fireflies come from.

Blend modes and where they apply
--------------------------------
With frequency separation OFF, ``blend_mode`` combines the two plates whole.
With it ON, ``blend_mode`` combines the two **low** bands only, and the high
bands are always a straight lerp. That is not an omission: a subtract-mode
high band is signed around zero, and ``multiply`` / ``screen`` / ``overlay``
are all defined on [0,1] values with black and white as meaningful anchors.
Applied to signed detail they produce garbage that looks like a bug. Tone and
structure are what blend modes are for; detail is what a mix is for.

``mix`` is the global wet/dry over the whole operation and behaves identically
in both cases: 0 returns plate B, 1 returns the full result, and an optional
``mask`` gates it per pixel. "Plate B" there means B *after* ``soften_b``,
because that blur is part of preparing the plate rather than part of the blend
— at the default of 0 the two are the same thing.

Range: nothing here truncates
-----------------------------
The node is range-agnostic end to end, and ``clamp_output`` is OFF by default
so it stays that way. A scene-linear or HDR plate comes out the other side with
its values above white intact:

* the resamplers run in float32 and are unclamped (which is the second reason
  they aren't ``comfy.utils.common_upscale``'s — that path clips to [0,1] on
  top of quantising);
* the Gaussian, the band split, the band mixes and the ``mix`` lerp are all
  linear operations with no anchor at white;
* the ``detail`` and ``difference`` outputs are never clamped at all, in either
  direction, so a Grade downstream can expose into them.

The one place white still matters is ``blend_mode``. ``over``, ``add``, ``min``
and ``max`` are range-agnostic. ``multiply``, ``screen``, ``overlay``,
``soft_light`` and ``difference`` are defined against white = 1.0 by
construction — they will compute happily on a value of 12.0, but what they
compute stops corresponding to what the mode means. On HDR input, reach for the
first group, or for the frequency controls, which have no such anchor.

``divide`` detail mode is worth a mention here too: being a ratio rather than a
difference, it is scale-invariant, which makes it the better-behaved of the two
splits on a plate with a very wide range. That is the same property that makes
it prone to fireflies against a near-black lowpass, hence ``detail_limit``.

The preview is the exception, and deliberately so. Its tiles are
range-normalised rather than clamped (see ``bat_hdr_preview.py``), so above-
white data does survive the trip to the browser and the blend still runs on it
unclamped — but the canvas has no viewer exposure to go and look at it with, so
what you see is the [0,1] slice. That was a deliberate trade: the judgement
this preview exists for is per-pixel texture, so the budget went on resolution
instead. See the "Resolution" note in ``web/bat_advanced_blend.js``.

The live preview
----------------
Same transport as ``Bat_Grade`` and ``Bat_HDRTonalComposite``: on every run
Python pushes downscaled tiles of the two conformed plates back through the
``{"ui": ...}`` channel, and ``web/bat_advanced_blend.js`` re-runs this exact
algorithm on them as the artist drags sliders. The preview is a fast
approximation at tile resolution — the real result is computed here on the
next run. Radii are scaled by the tile ratio on the JS side so a blur reads
the same in the preview as it will at full res.

Two notes on the node's face. The advanced controls — everything below ``mix``
— are folded into a collapsible section that starts closed, so the node reads
as a plain blend until you open it; that is done with the frontend's own
advanced-widget mechanism, and the details (including why it has to be set two
different ways) are in ``web/bat_advanced_blend.js``. And the preview is
zoomable: at 1:1 and above it draws the tile's real pixels with no smoothing,
which is the only honest way to judge sharpening.
"""

import base64
import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from . import bat_interrupt as _interrupt
from .bat_hdr_preview import hdr_tile

logger = logging.getLogger("[Bat_AdvancedBlend]")

EPS = 1e-6


def _num(value, default: float, lo: float, hi: float) -> float:
    """Coerce a widget/request value to a finite float in range.

    Used by both the node and the preview endpoint, for the same reason from two
    directions: neither can assume its input is sane. The endpoint is reachable
    over HTTP, and a saved workflow can carry impossible values — an early build
    of this node shifted them on copy/paste and wrote NaN into a radius, which
    reaches `_blur` as `int(round(nan))` and raises mid-render. A silently
    defaulted radius is a far better outcome than a traceback halfway through a
    queue, and the frontend repairs the node on load anyway.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return max(lo, min(hi, v))

# Long edge of the DRAFT preview tiles — the ones the browser blends itself, for
# instant feedback while a slider is moving.
#
# 768. This has been up and down: 512 originally, 768, back to 512 when the
# full-resolution server layer arrived (on the reasoning that the draft only has
# to track a drag), and now 768 again — because in practice the draft is what you
# are looking at during the moments you are actually deciding something, and 512
# was too coarse to see a sharpening change in.
#
# The draft's interactive resolution is no longer tied to this number, which is
# what makes a bigger tile affordable: the JS keeps a mip ladder and picks the
# finest level it can repaint inside its frame budget, measured live. So a
# machine that cannot hold 768 during a drag steps down on its own rather than
# everyone paying for the slowest case.
#
# What it does still cost is payload, on every execution. Measured, two tiles of
# a 4K plate with full-frame grain, as base64 through the UI channel:
#
#     512 -> 2.2MB    768 -> 5.0MB    896 -> 6.8MB    1024 -> 8.9MB
#
# and the ceiling is information-theoretic rather than a matter of tuning: the
# tiles are LOSSLESS 16-bit, and fine grain — exactly the signal this node
# manipulates — is incompressible. Lossy would break the node rather than merely
# cost less: JPEG at q92-q98 was measured destroying 75-105% of the high band's
# RMS on grainy content, which would show a detail transfer that does not exist.
# That is the whole reason the truth is a server render rather than a bigger tile.
PREVIEW_TILE_DIM = 768

# Frames per chunk through the blur/blend pipeline. A 4K frame carries ~100MB
# of float32 intermediates across the two bands of both plates, so a 300-frame
# batch processed whole would need tens of GB. Chunking bounds it without
# changing the result — every operation here is per-frame.
CHUNK_FRAMES = 8

BLEND_MODES = [
    "over", "add", "multiply", "screen", "overlay",
    "soft_light", "difference", "min", "max",
]

RESIZE_MODES = ["match_a", "match_b", "match_larger", "match_smaller"]
RESIZE_FILTERS = ["lanczos", "bicubic", "bilinear", "area", "nearest"]
DETAIL_MODES = ["subtract", "divide"]


# ---------------------------------------------------------------------------
# Resampling — separable, float-safe. See the module docstring for why this
# isn't comfy.utils.common_upscale.
# ---------------------------------------------------------------------------

def _kernel(name: str):
    """Return (fn, support) for a separable resampling kernel.

    `fn` takes the signed distance in input pixels (already divided by the
    antialiasing widening factor) and returns a weight. Supports are the
    conventional ones: lanczos3 = 3 taps either side, bicubic = 2, bilinear
    = 1, area = a half-pixel box that the widening factor turns into a true
    area average when downscaling.
    """
    if name == "lanczos":
        def f(x):
            # sinc(x) * sinc(x / 3), windowed to +/-3 taps. torch.sinc is
            # sin(pi x) / (pi x) and already evaluates the limit at x == 0.
            return torch.sinc(x) * torch.sinc(x / 3.0) * (x.abs() < 3.0)
        return f, 3.0
    if name == "bicubic":
        # Keys cubic with a = -0.5, which is what PIL and torch's bicubic use.
        a = -0.5

        def f(x):
            ax = x.abs()
            ax2 = ax * ax
            ax3 = ax2 * ax
            inner = (a + 2.0) * ax3 - (a + 3.0) * ax2 + 1.0
            outer = a * ax3 - 5.0 * a * ax2 + 8.0 * a * ax - 4.0 * a
            return torch.where(ax < 1.0, inner,
                               torch.where(ax < 2.0, outer, torch.zeros_like(ax)))
        return f, 2.0
    if name == "area":
        def f(x):
            return (x.abs() < 0.5).to(x.dtype)
        return f, 0.5
    # bilinear / fallback
    def f(x):
        return (1.0 - x.abs()).clamp(min=0.0)
    return f, 1.0


def _resample_matrix(n_in: int, n_out: int, filt: str, device, dtype, _cache={}):
    """Dense (n_out, n_in) weight matrix for a 1-D resample.

    Dense rather than banded because the largest case in practice is a 4K axis
    against a 1080p one — a 3840x1920 float32 matrix is 29MB, and a single
    matmul against it is far faster than the gather-and-pad machinery a banded
    form would need.

    Memoised because the frame loop is chunked: the same two matrices are
    wanted for every chunk of the batch, and rebuilding a 29MB one per chunk
    costs more than the resample itself. The cache is bounded because the key
    space is tiny in practice (two axes, one filter, one device per run).
    """
    key = (n_in, n_out, filt, str(device), str(dtype))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    if len(_cache) > 32:
        _cache.clear()

    scale = n_out / n_in
    # Downscaling widens the kernel so it averages the pixels it is skipping
    # over; upscaling leaves it at unit width. This is the standard
    # antialiasing correction, and skipping it is what makes a naive
    # nearest/bilinear downscale alias.
    widen = 1.0 / scale if scale < 1.0 else 1.0
    fn, support = _kernel(filt)

    # Output pixel centres mapped back into input-pixel coordinates.
    centers = (torch.arange(n_out, device=device, dtype=torch.float32) + 0.5) / scale
    j = torch.arange(n_in, device=device, dtype=torch.float32) + 0.5
    d = (j[None, :] - centers[:, None]) / widen          # (n_out, n_in)

    w = fn(d)
    s = w.sum(dim=1, keepdim=True)
    # A degenerate kernel (every tap zero) would divide by zero; fall back to
    # nearest for those rows rather than emitting NaN.
    dead = s.abs() < 1e-12
    if bool(dead.any()):
        nearest = (centers.clamp(0, n_in - 1)).round().long().clamp(0, n_in - 1)
        w[dead.squeeze(1)] = 0.0
        rows = torch.nonzero(dead.squeeze(1), as_tuple=False).squeeze(1)
        w[rows, nearest[rows]] = 1.0
        s = w.sum(dim=1, keepdim=True)
    out = (w / s).to(dtype)
    _cache[key] = out
    return out


def _resize(img: torch.Tensor, out_h: int, out_w: int, filt: str) -> torch.Tensor:
    """Resize an (N,H,W,C) float tensor. No clamping, no quantisation."""
    n, h, w, c = img.shape
    if (h, w) == (out_h, out_w):
        return img
    if filt == "nearest":
        ys = ((torch.arange(out_h, device=img.device, dtype=torch.float32) + 0.5)
              * (h / out_h)).long().clamp(0, h - 1)
        xs = ((torch.arange(out_w, device=img.device, dtype=torch.float32) + 0.5)
              * (w / out_w)).long().clamp(0, w - 1)
        return img[:, ys][:, :, xs]

    out = img
    if w != out_w:
        mx = _resample_matrix(w, out_w, filt, img.device, img.dtype)
        # (N,H,W,C) -> (N,H,C,W) @ (W,out_w) -> (N,H,C,out_w) -> back
        out = (out.movedim(2, -1) @ mx.transpose(0, 1)).movedim(-1, 2)
    if h != out_h:
        my = _resample_matrix(h, out_h, filt, img.device, img.dtype)
        out = (out.movedim(1, -1) @ my.transpose(0, 1)).movedim(-1, 1)
    return out


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------

def _gauss_kernel(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian of length 2*radius+1.

    Renormalised after truncation — clipping the radius to fit a narrow image
    otherwise throws away part of the kernel's mass and quietly darkens the
    lowpass, which in a subtract split turns into a bright halo in the detail
    band along that edge.
    """
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable Gaussian on an (N,H,W,C) tensor, reflect-padded.

    Reflect rather than zero padding: a zero-padded lowpass darkens toward the
    frame edge, so the high band picks up a bright border that reads as a
    glow once it is transferred.
    """
    if radius <= 0.0:
        return x
    r = int(round(radius))
    if r <= 0:
        return x
    sigma = max(radius / 2.0, 0.5)

    c = x.shape[-1]
    v = x.movedim(-1, 1)                                    # (N,C,H,W)
    for axis in (3, 2):
        rr = min(r, max(v.shape[axis] - 1, 0))
        if rr <= 0:
            continue
        k = _gauss_kernel(rr, sigma, v.device, v.dtype)
        if axis == 3:
            v = torch.nn.functional.pad(v, (rr, rr, 0, 0), mode="reflect")
            v = torch.nn.functional.conv2d(v, k.view(1, 1, 1, -1).expand(c, 1, 1, -1),
                                           groups=c)
        else:
            v = torch.nn.functional.pad(v, (0, 0, rr, rr), mode="reflect")
            v = torch.nn.functional.conv2d(v, k.view(1, 1, -1, 1).expand(c, 1, -1, 1),
                                           groups=c)
    return v.movedim(1, -1)


# ---------------------------------------------------------------------------
# Blend modes. `base` is plate B (the bottom), `top` is plate A.
# Mirrored verbatim in web/bat_advanced_blend.js — change both or neither.
# ---------------------------------------------------------------------------

def _blend(base: torch.Tensor, top: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "over":
        # No alpha channel to composite through, so "over" is a straight
        # replace. `mix` and `mask` are what gate it.
        return top
    if mode == "add":
        return base + top
    if mode == "multiply":
        return base * top
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - top)
    if mode == "overlay":
        return torch.where(base <= 0.5,
                           2.0 * base * top,
                           1.0 - 2.0 * (1.0 - base) * (1.0 - top))
    if mode == "soft_light":
        # W3C / CSS compositing definition, which is also Photoshop's.
        d = torch.where(base <= 0.25,
                        ((16.0 * base - 12.0) * base + 4.0) * base,
                        torch.sqrt(base.clamp(min=0.0)))
        return torch.where(top <= 0.5,
                           base - (1.0 - 2.0 * top) * base * (1.0 - base),
                           base + (2.0 * top - 1.0) * (d - base))
    if mode == "difference":
        return (base - top).abs()
    if mode == "min":
        return torch.minimum(base, top)
    if mode == "max":
        return torch.maximum(base, top)
    return top


# ---------------------------------------------------------------------------
# Preview payload helpers (same shapes bat_grade.py / bat_hdr_tonal_composite
# .py use, so the JS side's decode path is shared)
# ---------------------------------------------------------------------------

def _consumed_slots(prompt, unique_id):
    """Which of this node's output slots anything downstream reads.

    Returns None when it cannot tell, and every caller treats that as "all of
    them" — so a failure costs memory rather than correctness. The API prompt
    is {id: {"inputs": {name: value | [src_id, slot]}}}, so a consumer of our
    slot N shows up as the pair [our_id, N].
    """
    if not prompt or unique_id is None:
        return None
    try:
        me = str(unique_id)
        used = set()
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            for v in (node.get("inputs") or {}).values():
                if (isinstance(v, (list, tuple)) and len(v) == 2
                        and str(v[0]) == me and isinstance(v[1], int)):
                    used.add(v[1])
        return used
    except Exception as exc:
        logger.debug("could not read the prompt for output pruning (%s)", exc)
        return None


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = 384, quality: int = 82) -> str:
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_png_l(arr_hw: np.ndarray, max_dim: int = PREVIEW_TILE_DIM) -> str:
    im = Image.fromarray(arr_hw, "L")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _split(x: torch.Tensor, radius: float, mode: str):
    """Split into (low, high) by `mode`. Inverse is _recombine."""
    low = _blur(x, radius)
    if mode == "divide":
        # Ratio against the lowpass. The floor is what stops a near-black
        # lowpass turning a few counts of noise into a huge multiplier.
        return low, x / low.clamp(min=EPS)
    return low, x - low


def _recombine(low: torch.Tensor, high: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "divide":
        return low * high
    return low + high


def _neutral_high(mode: str) -> float:
    """The high-band value that leaves the low band untouched."""
    return 1.0 if mode == "divide" else 0.0


class BatAdvancedBlend:
    """Two-plate blend with frequency separation and a live canvas preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {
                    "tooltip": "The TOP plate — the one being blended in. For the "
                               "upscale case this is the SEEDVR2 / refiner result.",
                }),
                "image_b": ("IMAGE", {
                    "tooltip": "The BOTTOM plate — the base the result falls back "
                               "to. For the upscale case this is the original "
                               "generation. mix=0 returns this exactly.",
                }),
                "blend_mode": (BLEND_MODES, {
                    "default": "over",
                    "tooltip": "How A combines with B. 'over' is a straight "
                               "replace (there is no alpha channel to composite "
                               "through) — mix and mask are what gate it.\n"
                               "With frequency_separation ON this applies to the "
                               "LOW bands only; the high bands are always a lerp, "
                               "because a signed detail band has no meaningful "
                               "black or white for these formulas to key off.\n"
                               "On HDR / scene-linear input: over, add, min and "
                               "max are range-agnostic and stay correct above "
                               "white. multiply, screen, overlay, soft_light and "
                               "difference are all defined against white = 1.0, "
                               "so they still compute but stop meaning what they "
                               "mean on a display-referred plate.",
                }),
                "mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Global wet/dry over the whole operation. 0 returns "
                               "plate B (after soften_b), 1 is the full result.",
                }),
                "resize_mode": (RESIZE_MODES, {
                    "default": "match_a",
                    "tooltip": "The two plates almost never match resolution — "
                               "that is the point of an upscale. This picks the "
                               "working resolution: match_a (default) resamples B "
                               "up to A's size, so the upscale's resolution is the "
                               "deliverable. Every radius below is in pixels of "
                               "the working resolution.",
                }),
                "resize_filter": (RESIZE_FILTERS, {
                    "default": "lanczos",
                    "tooltip": "Kernel used to conform the other plate. lanczos is "
                               "the sharpest and the right default here; area is "
                               "the honest choice when downscaling a long way. "
                               "All of them run in float32 and preserve values "
                               "above white.",
                }),

                # ── frequency separation ─────────────────────────────────
                "frequency_separation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "OFF: an ordinary blend, and everything below is "
                               "ignored except soften_a/soften_b.\n"
                               "ON: both plates are split into a low band (blur — "
                               "tone, colour, structure) and a high band (edges, "
                               "texture, grain) which are then mixed separately.",
                }),
                "split_radius": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 128.0, "step": 0.1,
                    "tooltip": "Blur radius, in working-resolution pixels, that "
                               "divides low from high. Everything finer than this "
                               "counts as detail. Rule of thumb: just wide enough "
                               "that the blurred plate has no texture left, only "
                               "shapes.",
                }),
                "detail_mode": (DETAIL_MODES, {
                    "default": "subtract",
                    "tooltip": "subtract: high = x - blur(x), put back by adding. "
                               "The classic retouching split.\n"
                               "divide: high = x / blur(x), put back by "
                               "multiplying. Scale-invariant, so transferring "
                               "detail into shadows doesn't crush them and into "
                               "highlights doesn't blow them — worth it on "
                               "scene-linear or very wide-range plates.",
                }),
                "low_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "How far the LOW band moves from B's toward the "
                               "blend of A and B. 0 keeps B's tone, colour and "
                               "large-scale structure exactly.",
                }),
                "high_mix": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "How far the HIGH band moves from B's detail to "
                               "A's. This is the 'SEEDVR2 is too sharp' dial — "
                               "drop it to 0.3-0.5 to keep the upscale's "
                               "resolution but not its crunch.\n"
                               "low_mix 0 + high_mix 1 = only A's detail, on B.",
                }),
                "detail_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "Scales the mixed high band before it goes back on. "
                               "Below 1 softens the result overall, above 1 "
                               "sharpens it. Independent of high_mix: that picks "
                               "WHOSE detail, this picks HOW MUCH.",
                }),
                "detail_limit": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 8.0, "step": 0.01,
                    "tooltip": "Clamp on how far the high band may depart from "
                               "neutral. 0 = off. Mainly for detail_mode=divide, "
                               "where a ratio taken against a near-black lowpass "
                               "is where fireflies come from; a value of 2 there "
                               "means 'no pixel may be pushed more than 2x or "
                               "less than 1/2 its local average'.",
                }),

                # ── per-input pre-blur ───────────────────────────────────
                "soften_a": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1,
                    "tooltip": "Gaussian blur applied to plate A BEFORE anything "
                               "else. A blur is a lowpass, so this is 'take the "
                               "high frequency out of A first', with a radius "
                               "rather than a switch. Kills an upscaler's grain "
                               "floor at source instead of attenuating it "
                               "afterwards. Applies whether or not frequency "
                               "separation is on.",
                }),
                "soften_b": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1,
                    "tooltip": "Same, for plate B.",
                }),

                "clamp_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp the blended image to [0,1]. OFF by "
                               "default: nothing in this node truncates range "
                               "unless you ask it to, so a scene-linear or HDR "
                               "plate passes through with its values above "
                               "white intact. Turn it ON only when you want a "
                               "guaranteed display-referred result. Never "
                               "applies to the detail or difference outputs.",
                }),
                "preview_frame": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which frame of the batch the on-node preview "
                               "shows. Display only — the blend still applies to "
                               "every frame.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Gates the whole operation per pixel: white = full "
                               "result, black = plate B untouched. Multiplied with "
                               "mix.",
                }),
            },
            # Hidden inputs are not widgets, so this cannot disturb any saved
            # workflow's widgets_values. It is how the full-resolution preview
            # endpoint knows which cached frame belongs to which node.
            # PROMPT lets the node see which of its three outputs anything
            # downstream reads, so it can skip allocating the rest. At 100
            # frames of 2K that is 2.8 GB per output not paid for.
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "detail", "difference")
    OUTPUT_TOOLTIPS = (
        "The blended result.",
        "The high-frequency band that was actually applied, offset to 0.5 grey "
        "so it is viewable. Unclamped — push it through a Grade to inspect it.",
        "Signed (A - B) at the working resolution, offset to 0.5 grey. Shows "
        "exactly where and how far the upscale departed from the original.",
    )
    FUNCTION = "blend"
    CATEGORY = "BAT/Colour"
    DESCRIPTION = (
        "Blend two plates, with optional frequency separation so tone and "
        "detail move independently. Built for folding an upscale (SEEDVR2 and "
        "friends) back over the original without taking its over-sharpening "
        "along: low_mix 1 / high_mix 0.4 keeps the resolution and drops the "
        "crunch, low_mix 0 / high_mix 1 puts only A's detail onto B. "
        "soften_a / soften_b lowpass either plate before the blend. "
        "Live canvas preview on the node body."
    )

    # ------------------------------------------------------------------
    @staticmethod
    def _target_size(a: torch.Tensor, b: torch.Tensor, mode: str):
        ah, aw = int(a.shape[1]), int(a.shape[2])
        bh, bw = int(b.shape[1]), int(b.shape[2])
        if mode == "match_b":
            return bh, bw
        if mode == "match_larger":
            return (ah, aw) if ah * aw >= bh * bw else (bh, bw)
        if mode == "match_smaller":
            return (ah, aw) if ah * aw <= bh * bw else (bh, bw)
        return ah, aw

    @staticmethod
    def _conform_batch(a: torch.Tensor, b: torch.Tensor):
        """Reconcile frame counts.

        A single frame on either side broadcasts across the other's batch —
        that is the "blend a still over a clip" case and it is unambiguous.
        Two different multi-frame lengths are not unambiguous, so the result
        is truncated to the shorter and the mismatch is logged rather than
        silently held or looped.
        """
        na, nb = int(a.shape[0]), int(b.shape[0])
        if na == nb:
            return a, b
        if na == 1:
            return a.expand(nb, -1, -1, -1), b
        if nb == 1:
            return a, b.expand(na, -1, -1, -1)
        n = min(na, nb)
        logger.warning(
            "frame count mismatch: image_a has %d, image_b has %d — using the "
            "first %d. Conform them upstream if that is not what you want.",
            na, nb, n)
        return a[:n], b[:n]

    @staticmethod
    def _prep_mask(mask, n: int, device, dtype):
        """Align a MASK's frame count to the batch. Spatial resize happens per
        chunk in `blend`, so a long clip never materialises a full-resolution
        mask for every frame at once."""
        if mask is None:
            return None
        m = mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[0] == 1 and n > 1:
            m = m.expand(n, -1, -1)
        elif m.shape[0] < n:
            # Hold the last frame rather than erroring — a short mask is
            # nearly always a still that was meant to cover the whole clip.
            m = torch.cat([m, m[-1:].expand(n - m.shape[0], -1, -1)], dim=0)
        elif m.shape[0] > n:
            m = m[:n]
        return m

    @staticmethod
    def _mask_chunk(m, h: int, w: int):
        """One chunk of the aligned mask, resized to the working resolution and
        shaped (N,H,W,1) for broadcasting against (N,H,W,3)."""
        if m is None:
            return None
        if m.shape[1:3] != (h, w):
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=(h, w), mode="bilinear",
                align_corners=False).squeeze(1)
        return m.unsqueeze(-1)

    # ------------------------------------------------------------------
    @staticmethod
    def _core(a, b, p):
        """The whole algorithm for one chunk of frames.

        A staticmethod because the full-resolution preview endpoint calls it
        too, on a region rather than a chunk — there is exactly one
        implementation of this algorithm on the Python side and both the render
        and the preview service go through it.

        Returns (out, detail, difference). Every tensor here is (N,H,W,3)
        float32 at the working resolution, already conformed by the caller.
        """
        soft_a = _blur(a, p["soften_a"])
        soft_b = _blur(b, p["soften_b"])

        neutral = _neutral_high(p["detail_mode"])

        if not p["frequency_separation"]:
            result = _blend(soft_b, soft_a, p["blend_mode"])
            # Nothing was separated, so there is no applied detail band; report
            # neutral rather than inventing one, which keeps the `detail`
            # output honest about the fact that the feature is off.
            detail = torch.full_like(result, neutral)
        else:
            r = p["split_radius"]
            dm = p["detail_mode"]
            low_a, high_a = _split(soft_a, r, dm)
            low_b, high_b = _split(soft_b, r, dm)

            # Low band: the blend mode, then dialled back toward B's own low.
            low_blend = _blend(low_b, low_a, p["blend_mode"])
            low = low_b + (low_blend - low_b) * p["low_mix"]

            # High band: always a lerp. See the module docstring.
            high = high_b + (high_a - high_b) * p["high_mix"]

            # detail_gain scales the band's departure from neutral, so it means
            # the same thing in both detail modes (0 -> no detail at all, 1 ->
            # as extracted) instead of multiplying a ratio toward zero.
            g = p["detail_gain"]
            if g != 1.0:
                high = neutral + (high - neutral) * g
            lim = p["detail_limit"]
            if lim > 0.0:
                if dm == "divide":
                    # Symmetric in ratio terms: `lim` times up, `lim` times
                    # down. A limit below 1 would invert the bounds (min > max),
                    # which torch resolves silently to a flat image rather than
                    # raising, so floor it — "no more than 0.5x brighter" is not
                    # a thing anyone means to ask for.
                    hi = max(lim, 1.0 + EPS)
                    high = high.clamp(1.0 / hi, hi)
                else:
                    high = high.clamp(-lim, lim)

            result = _recombine(low, high, dm)
            detail = high

        mix = p["mix"]
        gate = mix if p["mask_chunk"] is None else p["mask_chunk"] * mix
        out = soft_b + (result - soft_b) * gate

        if p["clamp_output"]:
            out = out.clamp(0.0, 1.0)

        # Both inspection outputs are offset to mid grey and deliberately NOT
        # clamped — a Grade downstream can expose them, and clamping here would
        # throw away exactly the excursions you look at them to find.
        detail_view = (detail - neutral) + 0.5
        difference = (a - b) + 0.5
        return out, detail_view, difference

    # ------------------------------------------------------------------
    def blend(self, image_a, image_b, blend_mode, mix, resize_mode, resize_filter,
              frequency_separation, split_radius, detail_mode, low_mix, high_mix,
              detail_gain, detail_limit, soften_a, soften_b, clamp_output,
              preview_frame=0, mask=None, unique_id=None, prompt=None):

        a = image_a.to(torch.float32)
        b = image_b.to(torch.float32)
        if b.device != a.device:
            b = b.to(a.device)

        # Drop any alpha the upstream node carried; every operation here is
        # RGB and a 4-channel tensor would silently blend alpha as colour.
        if a.shape[-1] > 3:
            a = a[..., :3]
        if b.shape[-1] > 3:
            b = b[..., :3]

        a, b = self._conform_batch(a, b)

        # Size is derived from shapes alone, so the conform itself can wait
        # until inside the frame loop. Resampling a 300-frame batch to 4K up
        # front would be 30GB of intermediates for a result we consume 8 frames
        # at a time.
        if resize_mode not in RESIZE_MODES:
            logger.warning("unknown resize_mode %r; using match_a", resize_mode)
            resize_mode = "match_a"
        if resize_filter not in RESIZE_FILTERS:
            logger.warning("unknown resize_filter %r; using lanczos", resize_filter)
            resize_filter = "lanczos"

        out_h, out_w = self._target_size(a, b, resize_mode)

        n = int(a.shape[0])
        m = self._prep_mask(mask, n, a.device, a.dtype)

        # Coerced rather than trusted — see _num(). Ranges match INPUT_TYPES.
        p = {
            "blend_mode": blend_mode if blend_mode in BLEND_MODES else "over",
            "mix": _num(mix, 1.0, 0.0, 1.0),
            "frequency_separation": bool(frequency_separation),
            "split_radius": _num(split_radius, 4.0, 0.5, 128.0),
            "detail_mode": detail_mode if detail_mode in DETAIL_MODES else "subtract",
            "low_mix": _num(low_mix, 1.0, 0.0, 1.0),
            "high_mix": _num(high_mix, 1.0, 0.0, 1.0),
            "detail_gain": _num(detail_gain, 1.0, 0.0, 4.0),
            "detail_limit": _num(detail_limit, 0.0, 0.0, 8.0),
            "soften_a": _num(soften_a, 0.0, 0.0, 64.0),
            "soften_b": _num(soften_b, 0.0, 0.0, 64.0),
            "clamp_output": bool(clamp_output),
            "mask_chunk": None,
        }

        # The preview wants ONE conformed frame, and the loop below is about to
        # conform every frame anyway — so take it from there rather than running
        # the resampler a second time. On a single-frame 4K batch that redundant
        # conform was 1.2s, i.e. most of the node's runtime, spent recomputing
        # something that had just been thrown away.
        idx = max(0, min(int(preview_frame), n - 1))
        pv_a = pv_b = None

        # Which outputs anything downstream actually reads. None = unknown, so
        # compute all three; a failure here costs memory, never correctness.
        used = _consumed_slots(prompt, unique_id)
        want = [used is None or k in used for k in range(3)]

        # Preallocated and written by slice, NOT accumulated into lists and
        # torch.cat'ed. The old version held every chunk AND the concatenated
        # copy at the same time — a full-size tensor of chunks plus a full-size
        # result, for each of three outputs, so the cat alone roughly doubled
        # peak. Measured 6.11 GB on a 25-frame 2K pair.
        shape = (n, out_h, out_w, 3)
        stub = torch.zeros((1, 1, 1, 3), dtype=a.dtype, device=a.device)
        out    = torch.empty(shape, dtype=a.dtype, device=a.device) if want[0] else stub
        detail = torch.empty(shape, dtype=a.dtype, device=a.device) if want[1] else stub
        diff   = torch.empty(shape, dtype=a.dtype, device=a.device) if want[2] else stub

        for i in range(0, n, CHUNK_FRAMES):
            # Once per chunk: a flag read, and the difference between Cancel
            # working and the artist waiting out the whole clip.
            _interrupt.check()
            j = min(i + CHUNK_FRAMES, n)
            ac = _resize(a[i:j], out_h, out_w, resize_filter)
            bc = _resize(b[i:j], out_h, out_w, resize_filter)
            p["mask_chunk"] = self._mask_chunk(
                None if m is None else m[i:j], out_h, out_w)
            o, d, df = self._core(ac, bc, p)
            if want[0]: out[i:j] = o
            if want[1]: detail[i:j] = d
            if want[2]: diff[i:j] = df
            del o, d, df
            if pv_a is None and i <= idx < j:
                # .clone(), not a view: a view keeps the whole chunk alive, and
                # a chunk of 8 conformed 4K frames is 800MB.
                pv_a = ac[idx - i:idx - i + 1].clone()
                pv_b = bc[idx - i:idx - i + 1].clone()

        ui = self._preview_payload(pv_a, pv_b, m, idx, out_w, out_h, n, unique_id)
        return {"ui": ui, "result": (out, detail, diff)}

    # ------------------------------------------------------------------
    def _preview_payload(self, a_conf, b_conf, mask, idx,
                         out_w, out_h, frames, unique_id=None):
        """Everything the two preview layers need.

        `a_conf` / `b_conf` are ONE frame each, already conformed to the working
        resolution by the render loop — the same buffers the render itself
        consumed. That matters beyond saving a resample: the draft tiles below
        are a downscale of exactly what the full-resolution endpoint will render
        from, so the two layers cannot disagree about geometry, only about
        precision.

        Both tiles come off the same area-sampler at the same long-edge cap, so
        they arrive pixel-aligned and the JS loop can index them with a single
        counter. Area-sampled rather than point-sampled (Bat_Grade's default)
        because this tile is composited from and shown across the node body: a
        nearest-neighbour decimate of a 4K frame keeps one pixel in 16, which
        both looks broken and lies about how much detail the high band holds.
        """
        if a_conf is None or b_conf is None:
            return {}
        af, bf = a_conf[0], b_conf[0]

        # Hand the conformed frame to the full-resolution preview service. This
        # copy is what makes that layer possible at all, and it is the feature's
        # real cost — see CACHE_MAX_BYTES.
        if unique_id is not None:
            try:
                mk = None
                if mask is not None:
                    midx = max(0, min(idx, mask.shape[0] - 1))
                    mk = self._mask_chunk(mask[midx:midx + 1], out_h, out_w)
                    mk = None if mk is None else mk[..., 0]
                _cache_put(str(unique_id), a_conf, b_conf, mk,
                           {"w": int(out_w), "h": int(out_h),
                            "frames": int(frames), "frame": int(idx)})
            except Exception as exc:
                # The draft layer does not depend on this, so a failure here
                # costs the full layer and nothing else.
                logger.warning("could not cache the frame for full-resolution "
                               "preview: %s", exc)

        # w/h are the working resolution; the JS divides the tile's own width
        # by `w` to scale every pixel radius, so a blur reads the same in the
        # preview as it will at full res.
        ui = {
            "w": [int(out_w)],
            "h": [int(out_h)],
            "frames": [int(frames)],
            "preview_frame": [int(idx)],
        }

        t_a = hdr_tile(af, PREVIEW_TILE_DIM, sample="area")
        t_b = hdr_tile(bf, PREVIEW_TILE_DIM, sample="area")
        if t_a is not None:
            ui["tile_a"] = [t_a]
        if t_b is not None:
            ui["tile_b"] = [t_b]

        # 8-bit fallback for both plates. Needed in two situations: the tile
        # builder failed, or the browser has no DecompressionStream to inflate
        # it with. Also the only thing small enough to cache in localStorage
        # for the next workflow reopen.
        ui["jpeg_a"] = [_b64_jpeg((af.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8))]
        ui["jpeg_b"] = [_b64_jpeg((bf.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8))]

        if mask is not None:
            midx = max(0, min(idx, mask.shape[0] - 1))
            mk = (mask[midx].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            ui["mask_png"] = [_b64_png_l(mk)]

        if t_a is None or t_b is None:
            logger.debug("preview falling back to the 8-bit JPEGs for at least "
                         "one plate; the live canvas is still usable but its "
                         "blacks are not the ones the node acts on.")
        return ui


# ---------------------------------------------------------------------------
# Full-resolution preview service
# ---------------------------------------------------------------------------
#
# The on-node canvas has two layers, and they exist for different reasons.
#
# The DRAFT layer is `web/bat_advanced_blend.js` re-running this algorithm on a
# small tile in the browser. It is instant, which is what a slider drag needs,
# and it is an approximation: the radii are scaled into tile space, so a blur is
# right in proportion but not in detail.
#
# The FULL layer is this. It runs the real `_core` on the real plates at their
# real resolution and hands back a rendered picture. It is not a mirror of the
# render — it *is* the render — so at fit zoom it is strictly more correct than
# the draft (no radius scaling), and zoomed in it shows native pixels, which is
# the only honest way to judge sharpening. The cost is latency: the client fires
# it once the artist stops moving, a few hundred milliseconds behind the draft.
#
# What makes it possible is caching the conformed preview frame between runs.
# That is real memory — see CACHE_MAX_BYTES — and it is the price of the
# feature. Everything in here degrades to "no full layer, keep the draft"
# rather than failing.

import asyncio
import json
import threading
from collections import OrderedDict

# One conformed 4K frame is 3840*2160*3*4 = 99MB, so a plate pair is ~199MB.
# Two nodes' worth is the working set for a real comp — beyond that the LRU
# evicts. Deliberately a plain number rather than a fraction of system RAM: a
# preview cache that grows with the machine is how you discover it at 3am.
CACHE_MAX_BYTES = 900 << 20

_cache = OrderedDict()        # node_id -> entry
_cache_lock = threading.Lock()


def _entry_bytes(entry):
    n = 0
    for k in ("a", "b", "mask"):
        t = entry.get(k)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def _cache_put(node_id, a, b, mask, meta):
    """Cache one conformed preview frame. CPU-side on purpose.

    Holding a 4K plate pair in VRAM to service a preview would be taking
    hundreds of megabytes away from the thing the artist is actually trying to
    render. The ROI slice is moved to the GPU per request instead, which costs a
    PCIe copy and is invisible next to the round trip.
    """
    if node_id is None:
        return
    entry = {
        "a": a.detach().to("cpu", torch.float32).contiguous(),
        "b": b.detach().to("cpu", torch.float32).contiguous(),
        "mask": (None if mask is None
                 else mask.detach().to("cpu", torch.float32).contiguous()),
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


def _blur_margin(p):
    """Pixels of context an ROI needs so its interior matches a full-frame render.

    `_blur` uses a kernel of half-width `round(radius)`, and `_core` chains at
    most two of them per plate — the pre-blur, then the band split — so the
    influence radius is the sum. Get this wrong and the ROI is correct in the
    middle and wrong in a band around the edge, which is the sort of bug that
    only shows up as a faint seam once the artist pans.
    """
    split = round(p["split_radius"]) if p["frequency_separation"] else 0
    reach = max(round(p["soften_a"]), round(p["soften_b"])) + split
    return int(max(0, reach) + 2)


def render_region(entry, p, roi, out_w, out_h, view="result", amp=1.0):
    """Render one region of the cached frame, at full resolution.

    `roi` is (x, y, w, h) in conformed working-resolution pixels. The region is
    computed at native scale and only then resampled to (out_w, out_h) for
    transport — never the other way round. Rendering at the output size would
    mean scaling the radii, which is exactly the approximation the draft layer
    already makes and this layer exists to remove.

    Returns an (H, W, 3) uint8 array.
    """
    a_full, b_full = entry["a"], entry["b"]
    fh, fw = a_full.shape[1], a_full.shape[2]

    x, y, w, h = (int(v) for v in roi)
    x = max(0, min(x, max(fw - 1, 0)))
    y = max(0, min(y, max(fh - 1, 0)))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))

    m = _blur_margin(p)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(fw, x + w + m), min(fh, y + h + m)

    device = a_full.device
    if torch.cuda.is_available():
        device = torch.device("cuda")

    def slab(t):
        if t is None:
            return None
        return t[:, y0:y1, x0:x1].to(device, non_blocking=True)

    try:
        a = slab(a_full)
        b = slab(b_full)
        mask = slab(entry["mask"])
        pp = dict(p)
        pp["mask_chunk"] = None if mask is None else mask.unsqueeze(-1)
        out, detail, diff = BatAdvancedBlend._core(a, b, pp)
    except torch.cuda.OutOfMemoryError:
        # A preview must never be the reason a render fails. Retry on the host.
        logger.warning("full-resolution preview did not fit in VRAM; "
                       "falling back to CPU for this request")
        torch.cuda.empty_cache()
        a = a_full[:, y0:y1, x0:x1]
        b = b_full[:, y0:y1, x0:x1]
        mask = None if entry["mask"] is None else entry["mask"][:, y0:y1, x0:x1]
        pp = dict(p)
        pp["mask_chunk"] = None if mask is None else mask.unsqueeze(-1)
        out, detail, diff = BatAdvancedBlend._core(a, b, pp)

    # Pick the view. Mirrors the same choice in the JS so the draft and the full
    # layer never disagree about what is being shown, only about how precisely.
    if view == "a":
        img = a
    elif view == "b":
        img = b
    elif view == "detail":
        img = detail
    elif view == "diff":
        img = diff
    else:
        img = out

    if view in ("detail", "diff") and amp != 1.0:
        # Pivoted at mid grey, exactly as the JS does it: these views carry
        # differences of a few code values and are unreadable at unity.
        img = (img - 0.5) * float(amp) + 0.5

    # Drop the context margin now that the blurs have used it.
    img = img[:, (y - y0):(y - y0) + h, (x - x0):(x - x0) + w]

    if (out_h, out_w) != (img.shape[1], img.shape[2]):
        # Area for minification (the fit view), lanczos when the client asked
        # for more pixels than the region has — which only happens past 1:1,
        # where it is honest about being an enlargement.
        filt = "area" if (out_w < img.shape[2] or out_h < img.shape[1]) else "lanczos"
        img = _resize(img, max(1, int(out_h)), max(1, int(out_w)), filt)

    u8 = (img[0].clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return u8


def _params_from_request(body):
    """Build the `_core` parameter dict from the client's JSON.

    Every value is coerced and clamped here rather than trusted: this is an
    HTTP endpoint, and a NaN radius reaching `_blur` is an unbounded allocation.
    """
    def f(key, default, lo, hi):
        return _num(body.get(key, default), default, lo, hi)

    mode = str(body.get("blend_mode", "over"))
    if mode not in BLEND_MODES:
        mode = "over"
    dm = str(body.get("detail_mode", "subtract"))
    if dm not in DETAIL_MODES:
        dm = "subtract"

    return {
        "blend_mode": mode,
        "mix": f("mix", 1.0, 0.0, 1.0),
        "frequency_separation": bool(body.get("frequency_separation", False)),
        "split_radius": f("split_radius", 4.0, 0.5, 128.0),
        "detail_mode": dm,
        "low_mix": f("low_mix", 1.0, 0.0, 1.0),
        "high_mix": f("high_mix", 1.0, 0.0, 1.0),
        "detail_gain": f("detail_gain", 1.0, 0.0, 4.0),
        "detail_limit": f("detail_limit", 0.0, 0.0, 8.0),
        "soften_a": f("soften_a", 0.0, 0.0, 64.0),
        "soften_b": f("soften_b", 0.0, 0.0, 64.0),
        "clamp_output": bool(body.get("clamp_output", False)),
        "mask_chunk": None,
    }


# Largest picture the endpoint will return, per axis. The client asks for its
# own canvas size, so this is a backstop against a malformed request turning
# into a multi-gigapixel allocation, not a limit anyone should ever meet.
MAX_OUT_DIM = 4096


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/bat/advanced_blend/render")
    async def _bat_advanced_blend_render(request):
        """Render one region of a node's cached preview frame at full resolution.

        POST JSON: {node_id, roi: [x, y, w, h], out_w, out_h, view, amp, ...params}
        Returns a PNG, or 409 with a short reason when there is nothing cached
        (the client then simply keeps showing its draft).
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        node_id = str(body.get("node_id", ""))
        entry = _cache_get(node_id)
        if entry is None:
            # Not an error worth shouting about: it just means this node has not
            # run since the server started, or the LRU dropped it.
            return web.json_response(
                {"error": "no cached frame for this node; run it once"},
                status=409)

        try:
            p = _params_from_request(body)
            roi = body.get("roi") or [0, 0, entry["a"].shape[2], entry["a"].shape[1]]
            out_w = max(1, min(int(body.get("out_w") or 512), MAX_OUT_DIM))
            out_h = max(1, min(int(body.get("out_h") or 512), MAX_OUT_DIM))
            view = str(body.get("view", "result"))
            amp = float(body.get("amp", 1.0))
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": f"bad request: {exc}"}, status=400)

        loop = asyncio.get_running_loop()
        try:
            # Off the event loop: this is a full-resolution torch blend and can
            # take a good fraction of a second, which would otherwise stall
            # every other websocket message the frontend is waiting on —
            # including execution progress.
            u8 = await loop.run_in_executor(
                None, render_region, entry, p, roi, out_w, out_h, view, amp)
        except Exception as exc:
            logger.warning("full-resolution preview failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

        buf = BytesIO()
        # PNG, not JPEG. The whole point of this layer is that it shows the real
        # per-pixel result; a DCT codec would put its own texture in the high
        # frequencies, which is precisely what the artist is inspecting.
        Image.fromarray(u8, "RGB").save(buf, format="PNG", compress_level=1)
        return web.Response(
            body=buf.getvalue(), content_type="image/png",
            headers={
                "Cache-Control": "no-store",
                # Lets the client show "1.00:1" honestly rather than guessing.
                "X-Bat-Region": json.dumps({
                    "roi": [int(v) for v in roi],
                    "out": [int(out_w), int(out_h)],
                }),
            })

except ImportError as _exc:    # pragma: no cover - import-time only
    # No ComfyUI server at all: unit tests, or the pack imported standalone. The
    # node still works; the canvas just never gets its full-resolution layer.
    logger.debug("full-resolution preview endpoint not registered (%s)", _exc)
except Exception as _exc:      # pragma: no cover - import-time only
    # The server IS there and registration still failed, which is a real
    # problem rather than an expected environment — say so at a level someone
    # will see, because the symptom otherwise is just a preview that never
    # sharpens and no explanation anywhere.
    logger.warning("could not register the full-resolution preview endpoint; "
                   "the on-node canvas will stay on its draft layer: %s", _exc)
