"""
Bat_HDRTonalComposite — fold an LTX HDR reconstruction into a plate, but only
in the tonal extremes.

The problem this solves
-----------------------
LTX's HDR IC-LoRA does a genuinely good job of inventing detail in blown
highlights and crushed blacks, but ``LTXVHDRDecodePostprocess`` hands back a
whole new picture, not a patch. Everything moves: the midtones drift, the
grade shifts, the plate stops being the plate. If the shot is an AI-edited
SDR plate that has already been approved, that is not a usable trade.

So treat the HDR result as an *enhancement source* rather than a replacement.
Key a weight off the plate's own tone — near zero through the midtones, rising
into the shadows and the highlights — and only let the HDR reconstruction in
where the weight is up. Mids stay the plate. Extremes gain the range LTX
recovered. Because the key is tonal rather than spatial there are no matte
edges to sell.

Wire ``hdr_linear`` (the second output of ``LTXVHDRDecodePostprocess``) into
``hdr_ai``, not ``tonemapped`` — this node wants the raw linear HDR and does
its own display transform at the end.

Where the maths lives
---------------------
Three domains, kept deliberately distinct, because conflating them is what
makes this kind of node feel unpredictable:

* **Key domain** (display-referred luminance, in the plate's own encoding —
  see ``plate_gamma_mode``). Every threshold
  the artist types — ``shadow_start``, ``highlight_start``, ``mid_low`` — is
  read here. This is the one substantive departure from the original design
  brief, which applied those numbers to *linear* luminance. Linear 0.12 is
  about 38% grey on a monitor, so a "shadow" ramp starting there actually
  covered the lower midtones and overlapped the midtone exposure-match band
  (``mid_low`` 0.10) almost exactly. Keying in the display domain makes every
  default mean what it reads as.
* **Blend domain** (scene-linear). All the actual mixing is linear light, as
  it must be for the energy to add up.
* **Output domain** (display-referred again) via the tonemap at the end.

The merge itself
----------------
Target luminance is a straight lerp toward the HDR's luminance, weighted by
the tonal key. Getting there without wrecking the plate's colour is the
interesting part. The obvious move — scale the plate's RGB by
``Y_target / Y_plate`` — preserves hue exactly, and is what the brief
proposed, but it is multiplicative: a pixel crushed to zero stays at zero no
matter how much HDR you point at it. That kills the entire shadow half of the
node, which is precisely the case where the plate has no detail left and the
HDR has some.

This module writes the same operation additively instead::

    dir   = plate_lin / Y_plate         # unit-luminance chroma direction
    out   = plate_lin + (Y_target - Y_plate) * dir

which is *algebraically identical* to the multiply wherever the plate has
signal, but degrades gracefully where it doesn't: as the plate approaches
black there is no plate chroma left to preserve, so ``dir`` crossfades to the
HDR's own chroma direction, and then to neutral if both are black. Note that
``(1,1,1)`` has luminance exactly 1 under Rec.709 weights, so every one of
those directions is unit-luminance and the output lands on ``Y_target``
exactly rather than approximately.

Level is only half the job, though. "Recover the highlight" and "keep the
plate's tone" pull against each other if the only tool is a luminance lerp:
turn it up and the sky gets its detail back but also visibly changes
brightness. So the transfer is split in two, and the split is the point of
the node:

* ``*_tone_transfer`` moves the local *average* toward the HDR.
* ``detail_transfer`` moves the local *contrast* toward the HDR, in the log
  domain, leaving the average where the tone stage put it. This is the
  "more detail, same tone" operation: multiply the plate's neighbourhood
  average by the HDR's neighbourhood-relative structure.

With tone transfer at 0 and ``detail_transfer`` at 1 the output's local
average is the plate's, exactly, and only the texture is LTX's. With both at 1
the weighted regions become the HDR outright.

The tone control is split per tonal end, because the two ends want opposite
things and one slider cannot serve both. Shadows need level: a region crushed
to 0.0 must be lifted before any detail in it is visible, so
``shadow_tone_transfer`` defaults to 1.

Highlights are the harder half, and the reason is the display rather than the
algorithm. A monitor has nothing above white. Lift a clipped sky from 1.0 to
its true 3.5 linear and the shoulder has to put it *somewhere* below 1.0 —
with no headroom reserved, 1.0 and 16.0 land within a few code values of each
other and the whole recovered structure reads as one flat white. The only
escape is to move the plate's own white down and spend the freed range on the
highlights, which is what ``highlight_headroom`` does, in stops.

So the two highlight controls are a pair and neither works alone:
``highlight_headroom`` creates somewhere to go, ``highlight_tone_transfer``
moves the level into it. Raising the second without the first actively makes
things worse — it pushes the sky further up a curve that has already run out
of room. Defaults are 2 stops and 0.5. Set headroom to 0 for a hard clip and
the plate's whites come through exactly, at the cost of seeing nothing above
them.

Finally ``hdr_rgb_mix`` brings in a little of the HDR's own *colour*, for what
a luminance operation structurally cannot reach — a clipped sky is flat *and*
hueless in the plate, so there is no chroma there to scale or modulate, and
only the HDR knows what colour it should be. It is rescaled to the target
luminance before mixing, so it moves hue and saturation without moving
brightness; a plain RGB lerp there would have dragged the level along and
quietly undone the tone stage.

Two outputs, two different worlds
---------------------------------
``linear_out`` is the deliverable: scene-linear, unbounded, whatever range the
HDR carried — 50, 200, 1000 — with no tonemap and no baked exposure. Send it
to EXR and grade it properly downstream. Nothing in this node truncates it by
default; ``hdr_ceiling`` exists to clamp fireflies but is OFF.

``linear_out_primaries`` converts that output's gamut on the way out —
Rec.709 (default, no-op), ACEScg or ACES2065-1. It affects ``linear_out``
only. Everything upstream, including the Rec.709 luma weights the whole tonal
key depends on, works in the plate's primaries; the matrix is the very last
step. If your comp package is colour-managed, the equivalent fix is to leave
this at rec709 and read the EXR as "Linear Rec.709" — do one or the other,
never both.

``image_out`` is a monitoring image: Rec.709/sRGB, [0, 1], because that is
what a display and a ComfyUI preview are. Every argument in this file about
shoulders, knees and ``highlight_headroom`` is about *that* output only —
none of it touches ``linear_out``. If the SDR preview looks like it has lost
the highlights, check ``linear_out`` before believing it.

Live preview
------------
The node body carries a canvas that re-runs this whole algorithm in JS on a
downscaled tile, so thresholds can be dialled in at interactive speed without
re-running the graph. It ships a 16-bit tile of the plate *and* of the linear
HDR (see bat_hdr_preview.py — range-normalised, so values above white survive
the trip), which is what makes a live preview of an HDR composite possible at
all: an 8-bit JPEG of the HDR input would have clipped away the only thing
worth previewing. The canvas has its own viewer exposure, applied to the
scene-linear merge *before* the display transform, so exposing down reveals
what is sitting above white without altering the render.

It also carries its own display transform (auto / OCIO / sRGB / Rec.709 / 2.2 /
2.4 / raw), again display-only — it cannot change either output or the tonal key.
"auto" follows whatever ``output_gamma_mode`` resolves to, which keeps the
Plate view byte-exact against a display-referred plate. The exception is a
scene-linear plate: there is no plate curve to match, ``image_out`` falls back
to sRGB, and the canvas uses Rec.709 instead on the grounds that anyone
feeding scene-linear into a comp node is likelier on a Rec.709 grading monitor
than a generic sRGB desktop. That is the only case where canvas and image_out
disagree, and the control is tinted to say so.

"OCIO" is the one option that is a real view *transform* rather than a curve.
Set ``preview_ocio_view`` to a display transform (out_rec709, say) and the
backend bakes it out of the OCIO config into a 33^3 log-shaped 3D LUT and
ships it with the tiles; the canvas then renders through the same transform
the artist is viewing through. This is the only way to close that gap: an ACES
Output Transform is RRT + ODT, a filmic tone curve with a path-to-white, and
no choice of gamma is its inverse — measured against the shipped aces_1.2
config, un-rendering a plate and re-encoding with BT.709 leaves 18/255 of
drift no gamma setting can remove. The LUT lands within 1.27/255 worst case
and 0.24/255 mean over a real frame. The config is found via
``OCIO_CONFIG_PATH`` / ``$BAT_OCIO_CONFIG`` / ``$OCIO``, falling back to the
one ETC's own colourspace node bundles, which is usually the right answer.

Both tiles are area-sampled at ``preview_resolution`` (512px long edge by
default, ~850 KB for the pair on a real 1080p frame). Bat_Grade point-samples
its tile instead, because its Inspect mode is a bit-depth probe and averaging
would erase the quantisation steps it looks for; this node *displays* its tile
full-width and composites from it, so an aliased nearest-neighbour decimate of
a 1920-wide frame — one pixel in 56 — reads as broken. Worth knowing that
area-sampling costs more on the wire, not less: averaging manufactures
intermediate values where an 8-bit-sourced plate had only 256 distinct codes,
and that extra entropy defeats zlib.

See web/bat_hdr_tonal_composite.js — the pixel loop there mirrors
``_composite_chunk`` below and the two must be changed together.
"""

import base64
import logging
import math
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from . import bat_interrupt as _interrupt
from .bat_hdr_preview import hdr_tile

logger = logging.getLogger("[Bat_HDRTonalComposite]")

# Rec.709 / sRGB luminance weights. Chosen (over Rec.2020 or a naive average)
# because the plate is sRGB and the LTX pipeline's own preview path is sRGB.
_LUMA = (0.2126, 0.7152, 0.0722)

# Numerical floor. Not a widget: it is a guard against divide-by-zero, not an
# artistic control, and exposing it only invites someone to set it to
# something that breaks the maths.
_EPS = 1e-6

# Linear luminance at which the plate is considered to have lost its colour
# identity. Below this, "preserve the plate's chroma" is a meaningless
# instruction — there is no chroma down there, only quantisation noise — so
# the merge crossfades to the HDR's chroma instead. Also the offset for the
# log-domain detail split, which sets how far into the blacks that separation
# stays meaningful. 1e-3 linear is about 3.6% on an sRGB display: solidly
# "black" by eye.
_BLACK_FLOOR = 1e-3

# Feather on the midtone exposure-match band, in the key (display) domain. A
# hard in/out band makes the match ratio jump whenever a threshold crosses a
# histogram peak, which reads as the node being unstable when it isn't.
_MID_FEATHER = 0.05

# ---------------------------------------------------------------------------
# Output primaries
# ---------------------------------------------------------------------------
# Everything upstream of here works in the PLATE's primaries — Rec.709, which
# is also what LTX outputs — and the Rec.709 luma weights above depend on that.
# This is the last step, applied to linear_out only, for pipelines whose
# working space is not Rec.709.
#
# Why bake matrices at all: a colour-managed app should normally do this, and
# reading the EXR as "Linear Rec.709" in Nuke is the equivalent fix with no
# code. But if the house convention is that EXRs are always working-space,
# converting here means `scene_linear` just works.
#
# These do NOT change with the ACES version. AP0/AP1 and Rec.709
# chromaticities are identical from ACES 1.0 through 1.3; what moves between
# versions is the RRT/ODT, which is a viewing transform and no business of
# this node. Both were cross-checked two ways: derived from the published
# primaries with a Bradford D65->D60 adaptation, and probed out of OCIO's
# built-in ACES config. The two agree to 2.7e-08.
_M_REC709_TO_ACESCG = (
    (0.6130974, 0.3395231, 0.0473795),
    (0.0701937, 0.9163539, 0.0134524),
    (0.0206156, 0.1095698, 0.8698146),
)
_M_REC709_TO_ACES2065_1 = (
    (0.4396330, 0.3829887, 0.1773783),
    (0.0897764, 0.8134394, 0.0967841),
    (0.0175412, 0.1115466, 0.8709123),
)
_BAKED_PRIMARIES = {
    "acescg": _M_REC709_TO_ACESCG,
    "aces2065-1": _M_REC709_TO_ACES2065_1,
}

# Point this at an .ocio config to derive the matrices from it instead of
# using the baked ones — set it here, or via $BAT_OCIO_CONFIG, or $OCIO.
# Only needed for a config with non-standard primaries; the baked values are
# correct for every stock ACES config. Falls back to baked (with a warning) if
# the config or PyOpenColorIO is unavailable.
OCIO_CONFIG_PATH = ""

# Configs disagree on naming across ACES versions, so try candidates in order.
# ACES 1.2-era (OCIO v1) names first, then the OCIO v2 built-in style.
_OCIO_NAMES = {
    "rec709": ["Utility - Linear - Rec.709", "Linear Rec.709 (sRGB)",
               "lin_rec709", "Utility - Linear - sRGB"],
    "acescg": ["ACES - ACEScg", "ACEScg", "lin_ap1"],
    "aces2065-1": ["ACES - ACES2065-1", "ACES2065-1", "lin_ap0"],
}

_primaries_cache = {}

# ---------------------------------------------------------------------------
# Viewer LUT — an exact OCIO display transform for the on-node canvas
# ---------------------------------------------------------------------------
# The canvas can encode with a gamma curve, but a gamma curve is not a view
# transform. An ACES Output Transform is RRT + ODT: a filmic tone curve with a
# path-to-white. Un-rendering a plate through it and re-encoding with BT.709 is
# not a round trip — measured on the shipped aces_1.2 config, 18/255 max drift,
# mids ~5% up, deep shadows ~13% down, everything more saturated. No choice of
# gamma fixes that, because the curve was never the difference.
#
# So bake the real transform out of OCIO into a 3D LUT and hand it to the JS.
# Display-only: it never touches image_out, linear_out, or the tonal key.

# Grid size per axis. 33 is the film-industry default for a display LUT and the
# RRT is smooth enough that trilinear between those samples is invisible.
_VIEW_LUT_SIZE = 33

# Shaper range in stops. The LUT axis is log2 of the linear value, because a
# linear axis would spend most of its samples above white where nothing is
# happening and leave the shadows — where the tone curve moves fastest — with
# almost none. +/-12 stops covers 0.00024 to 4096.
_VIEW_LUT_MIN_EV, _VIEW_LUT_MAX_EV = -12.0, 12.0

# Extra hdr_ai inputs the node can carry. One set of settings applied to
# several LTX reconstructions of the same plate — the point being that dialling
# a threshold once retunes every version, instead of copy-pasting the node and
# keeping N copies in sync by hand.
#
# Outputs are INTERLEAVED (image_out, linear_out, image_out_2, linear_out_2, …)
# rather than grouped, because ComfyUI maps outputs to RETURN_TYPES by index
# and the frontend trims unused slots off the END. Grouped, dropping a version
# would shift every linear_out down onto an image_out.
MAX_HDR_VERSIONS = 8

_view_lut_cache = {}


def _ocio_view_choices():
    """Colourspace names for the preview_ocio_view widget, read from the config.

    Enumerated at class-definition time the same way ETC's own node does it, so
    the artist picks from a list rather than typing a name that has to match
    exactly. "(off)" first: the LUT is ~95 KB on every execution and most
    graphs do not need it.
    """
    choices = ["(off)"]
    try:
        import PyOpenColorIO as ocio
        path = _find_ocio_config()
        if path:
            cfg = ocio.Config.CreateFromFile(path)
            names = sorted(cs.getName() for cs in cfg.getColorSpaces())
            outs = [n for n in names if n.lower().startswith("out")]
            choices += outs + [n for n in names if n not in outs]
    except Exception as exc:
        logger.info("no OCIO config for the preview view list (%s)", exc)
    return choices


def _find_ocio_config():
    """Resolve an OCIO config: the module override, the env, or ETC's bundle.

    The last one is the useful bit in this studio — ComfyUI-ETC_Colourspace_Manager
    ships an aces_1.2 config, and it is almost always the one the artist is
    already viewing through, so finding it automatically means the canvas can
    match their viewer with nothing to configure.
    """
    for p in (OCIO_CONFIG_PATH, os.environ.get("BAT_OCIO_CONFIG"),
              os.environ.get("OCIO")):
        if p:
            return p
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, os.pardir, "ComfyUI-ETC_Colourspace_Manager",
                           "aces_1.2", "config.ocio")
    return os.path.normpath(bundled) if os.path.exists(bundled) else None


def _view_lut(view_name: str, working: str = "rec709"):
    """Bake `working`-linear -> `view_name` into a zlib'd 16-bit 3D LUT.

    Returns the JS payload dict, or None (with a warning) if it cannot be
    built — the canvas then falls back to its gamma encodes, which is a worse
    match but never a broken preview.
    """
    if not view_name:
        return None
    path = _find_ocio_config()
    if not path:
        logger.warning("preview_ocio_view is set to %r but no OCIO config was "
                       "found (set OCIO_CONFIG_PATH, $BAT_OCIO_CONFIG or $OCIO)",
                       view_name)
        return None
    key = (path, view_name, working, _VIEW_LUT_SIZE)
    if key in _view_lut_cache:
        return _view_lut_cache[key]
    try:
        import base64
        import zlib

        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromFile(path)
        have = {cs.getName() for cs in cfg.getColorSpaces()}
        src = next((n for n in _OCIO_NAMES[working] if n in have), None)
        if src is None:
            raise KeyError(f"no linear {working} colourspace in {path}")
        if view_name not in have:
            raise KeyError(f"{view_name!r} is not a colourspace in {path}")

        n = _VIEW_LUT_SIZE
        lo, hi = _VIEW_LUT_MIN_EV, _VIEW_LUT_MAX_EV
        axis = 2.0 ** (lo + (np.arange(n, dtype=np.float64) / (n - 1)) * (hi - lo))
        # Index order r-major so the JS can address it as ((r*n + g)*n + b).
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        grid = np.stack([r, g, b], -1).reshape(-1, 3).astype(np.float32)
        cfg.getProcessor(src, view_name).getDefaultCPUProcessor().applyRGB(grid)

        out = np.clip(np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        packed = np.clip(out * 65535.0 + 0.5, 0, 65535).astype("<u2").tobytes()
        payload = {
            "data": base64.b64encode(zlib.compress(packed, 6)).decode("ascii"),
            "size": n, "minEv": lo, "maxEv": hi,
            "name": view_name, "src": src,
            "config": os.path.basename(os.path.dirname(path)) or path,
        }
        _view_lut_cache[key] = payload
        logger.info("baked a %d^3 view LUT for %s -> %s from %s",
                    n, src, view_name, path)
        return payload
    except Exception as exc:
        logger.warning("could not bake the %r view LUT from %s (%s); the canvas "
                       "will fall back to its gamma encodes", view_name, path, exc)
        _view_lut_cache[key] = None
        return None




def _ocio_matrix(target: str):
    """Recover the Rec.709 -> `target` matrix from an OCIO config, or None.

    Probes the processor with the three basis vectors rather than digging
    through the transform graph — exact for a linear-to-linear conversion,
    which is all this can be, and immune to how the config expresses it.
    """
    path = (OCIO_CONFIG_PATH or os.environ.get("BAT_OCIO_CONFIG")
            or os.environ.get("OCIO") or "")
    if not path:
        return None
    try:
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromFile(path)
        have = {cs.getName() for cs in cfg.getColorSpaces()}
        def pick(key):
            for n in _OCIO_NAMES[key]:
                if n in have:
                    return n
            raise KeyError(f"no colorspace for {key!r} in {path}")
        proc = cfg.getProcessor(pick("rec709"), pick(target)).getDefaultCPUProcessor()
        basis = np.eye(3, dtype=np.float32)
        proc.applyRGB(basis)          # mutates ndarrays in place; rows = images
        M = basis.T
        if not np.all(np.isfinite(M)) or abs(float(np.linalg.det(M))) < 1e-6:
            raise ValueError(f"degenerate matrix from config: {M.tolist()}")
        logger.info("primaries matrix for %s taken from OCIO config %s", target, path)
        return M
    except Exception as exc:
        logger.warning("could not derive %s primaries from OCIO config %r (%s); "
                       "using the baked ACES matrix instead", target, path, exc)
        return None


def _primaries_matrix(target: str, device, dtype):
    """Cached Rec.709 -> `target` 3x3. Returns None for 'rec709' (a no-op)."""
    if target == "rec709" or target not in _BAKED_PRIMARIES:
        return None
    if target not in _primaries_cache:
        M = _ocio_matrix(target)
        if M is None:
            M = np.array(_BAKED_PRIMARIES[target], dtype=np.float32)
        _primaries_cache[target] = np.ascontiguousarray(M, dtype=np.float32)
    return torch.from_numpy(_primaries_cache[target]).to(device=device, dtype=dtype)


# Sanity bounds on the auto exposure match. A pathological frame (an almost
# entirely black plate, say) can produce an absurd ratio; better to stop
# matching than to detonate the image.
_K_MIN, _K_MAX = 1.0 / 64.0, 64.0

# Peak bytes to allow per intermediate tensor when chunking the composite.
# The pipeline holds roughly a dozen frame-sized buffers live at once, so a
# 121-frame 1080p batch would want ~35 GB in one shot. Chunking keeps the
# working set flat regardless of clip length — the two full-size output
# tensors are then the only term that still scales with the batch, and those
# are the deliverable. 128 MB works out at ~5 frames per chunk at 1080p.
#
# The result is chunk-size-invariant to float precision but not bit-exact,
# for two reasons that are both torch's rather than ours: conv2d (the two
# detail-stage blurs) picks a different reduction order per batch size, and
# the whole-batch exposure match accumulates its sums per chunk. Both land
# around 1e-7 relative. With detail_radius = 0 and the match off, chunking
# IS bit-exact.
# Measured, not guessed: at 2048x1152 this works out at one frame per chunk,
# which is both LEANER and FASTER than the 128 MB it used to be (4.56 GB / 7.6 s
# vs 5.88 GB / 10.0 s on a 25-frame two-version run). Bigger chunks lose on
# cache locality in the separable blurs, so there is no tradeoff to balance
# here — smaller simply won.
_CHUNK_BYTES = 32 << 20


# ---------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------

def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF. Piecewise, including the linear toe — the pure 2.4 power
    diverges badly enough near black to matter for a node whose whole job is
    the bottom of the curve."""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow(((x.clamp(min=0.04045) + 0.055) / 1.055), 2.4),
    )


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Inverse of the above. Not clamped to 1 on the way out: callers that
    want display range clamp themselves, and the key domain wants the
    unclamped curve so overbright plates still sort above `highlight_full`."""
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x.clamp(min=0.0031308), 1.0 / 2.4) - 0.055,
    )


def _rec709_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Inverse BT.709 OETF — what Nuke's "rec709" colorspace applies.

    Not interchangeable with sRGB despite both being "the Rec.709 primaries":
    the transfer curves differ by up to 65% in the shadows (code 0.05 decodes
    to 0.0039 through sRGB and 0.0111 through BT.709). This node keys its
    shadow ramp off plate luminance, so decoding with the wrong curve moves
    the key, not just the levels.
    """
    return torch.where(x < 0.081,
                       x / 4.5,
                       torch.pow((x.clamp(min=0.081) + 0.099) / 1.099, 1.0 / 0.45))


def _to_linear(img: torch.Tensor, mode: str) -> torch.Tensor:
    """Decode the plate to scene-linear according to how it was encoded."""
    img = img.clamp(min=0.0)
    if mode == "linear":
        return img
    if mode == "rec709":
        return _rec709_to_linear(img)
    if mode == "gamma_2_2":
        return torch.pow(img, 2.2)
    if mode == "gamma_2_4":
        return torch.pow(img, 2.4)
    return _srgb_to_linear(img)


def _luminance(img: torch.Tensor) -> torch.Tensor:
    """(B,H,W,C>=3) -> (B,H,W). Rec.709 weights."""
    return (img[..., 0] * _LUMA[0]
            + img[..., 1] * _LUMA[1]
            + img[..., 2] * _LUMA[2])


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    """Hermite ramp from e0 to e1.

    Reversed edges (e0 > e1) are supported and produce a falling ramp — which
    is how the shadow key is written, and how the upper edge of the midtone
    band is written. The clamp does the work: with e1 < e0 the quotient is
    negative-over-negative below e1, lands above 1, and clamps.
    """
    span = e1 - e0
    if abs(span) < 1e-8:
        span = 1e-8 if span >= 0 else -1e-8
    t = ((x - e0) / span).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Where a `highlight_headroom` of h stops puts the linear value 2**h on the
# display: high enough to read as "near white", low enough that the values
# above it still have somewhere to go.
_SHOULDER_TARGET = 0.95


def _knee_from_headroom(stops: float) -> float:
    """Solve for the shoulder knee that lands 2**`stops` linear on _SHOULDER_TARGET.

    Headroom is the control an artist actually wants — "show me three stops
    above plate white" — while the knee is the parameter the curve needs. With
    the shoulder below, ``f(W) = T`` rearranges to a quadratic in the knee::

        k**2 - 2Tk + T - W(1 - T) = 0   ->   k = T - sqrt(T**2 - T + W(1 - T))

    Zero stops returns 1.0, i.e. a hard clip: no shoulder at all, and the
    plate's whites survive exactly.
    """
    if stops <= 0.0:
        return 1.0
    W = 2.0 ** float(stops)
    T = _SHOULDER_TARGET
    k = T - math.sqrt(max(T * T - T + W * (1.0 - T), 0.0))
    return min(max(k, 0.02), 0.999)


def _soft_rolloff(x: torch.Tensor, knee: float) -> torch.Tensor:
    """Identity below `knee`, hyperbolic shoulder above, asymptotic to 1.

    Slope is exactly 1 on both sides of the knee, so a gradient crossing it
    shows no crease.

    This replaced an exponential shoulder (``1 - exp(-u)``) which had the same
    continuity but a fatally short tail: with the default knee it drove linear
    4.0 and linear 16.0 to the *same* display code, so every recovered
    highlight above two stops collapsed into one flat white. The hyperbola
    never actually reaches 1, which is the point — it keeps the ordering, so
    16x still reads brighter than 4x. Measured plate-white-to-16x separation
    at knee 0.45: 24 display codes on the exponential, 32 on this.

    Note the asymmetry this curve is built around: a display cannot show
    anything brighter than white, so the only way to make a recovered
    highlight read as *bright* is to move the plate's own white down and spend
    that range on the highlight. There is no setting that avoids the trade —
    `highlight_headroom` just makes it explicit.
    """
    if knee >= 1.0:
        return x.clamp(max=1.0)
    knee = max(knee, 0.0)
    s = 1.0 - knee
    u = ((x - knee) / s).clamp(min=0.0)
    return torch.where(x <= knee, x, knee + s * u / (1.0 + u))


def _gauss_kernel(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    """Normalised 1-D Gaussian of length 2*radius+1.

    Built per axis rather than once and sliced, because a truncated kernel has
    to be renormalised — otherwise clipping the radius down to fit a narrow
    image quietly darkens the weight map by whatever fraction of the kernel
    was thrown away.
    """
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur_hw(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable Gaussian on a (B,H,W) map. Reflect-padded so the frame edge
    doesn't darken the weight and pull the composite back toward the plate in
    a border band."""
    if radius <= 0:
        return x
    sigma = max(radius / 2.0, 0.5)
    v = x.unsqueeze(1)  # (B,1,H,W)

    # torch's reflect pad requires pad < dim, so a radius wider than the image
    # is clipped per axis. Nonsense input, but it shouldn't raise.
    for axis in (3, 2):
        r = min(int(radius), max(v.shape[axis] - 1, 0))
        if r <= 0:
            continue
        k = _gauss_kernel(r, sigma, v.device, v.dtype)
        if axis == 3:
            v = torch.nn.functional.pad(v, (r, r, 0, 0), mode="reflect")
            v = torch.nn.functional.conv2d(v, k.view(1, 1, 1, -1))
        else:
            v = torch.nn.functional.pad(v, (0, 0, r, r), mode="reflect")
            v = torch.nn.functional.conv2d(v, k.view(1, 1, -1, 1))
    return v.squeeze(1)


# ---------------------------------------------------------------------------
# Preview payload helpers (same shapes bat_grade.py uses)
# ---------------------------------------------------------------------------

def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = 384, quality: int = 80) -> str:
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _linear_to_rec709(x: torch.Tensor) -> torch.Tensor:
    """Forward BT.709 OETF. Inverse of _rec709_to_linear."""
    return torch.where(x < 0.018,
                       4.5 * x,
                       1.099 * torch.pow(x.clamp(min=0.018), 0.45) - 0.099)


def _encode_from_linear(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Forward of `_to_linear` — scene-linear back to the plate's own encoding.

    `linear` has no display encoding of its own to offer, so it borrows sRGB:
    the key has to live in *some* perceptual domain or the threshold defaults
    stop meaning what they say (linear 0.12 is about 38% grey on screen).
    """
    x = x.clamp(min=0.0)
    if mode == "rec709":
        return _linear_to_rec709(x)
    if mode == "gamma_2_2":
        return torch.pow(x, 1.0 / 2.2)
    if mode == "gamma_2_4":
        return torch.pow(x, 1.0 / 2.4)
    return _linear_to_srgb(x)


def _key_luma(plate_lin: torch.Tensor, mode: str = "srgb") -> torch.Tensor:
    """The signal every threshold widget is read against: display-referred
    luminance, in the plate's OWN encoding.

    Encoding the *luminance* rather than taking the luminance of the encoded
    RGB is the more defensible of the two — this is a tone key, not a colour.

    It re-encodes with `plate_gamma_mode` rather than always with sRGB, which
    is a correctness fix: keying a Rec.709 plate through the sRGB curve put
    every threshold in a domain the artist was not looking at, and the two
    curves are up to 65% apart in the shadows — precisely where the shadow
    ramp lives. Now `shadow_start = 0.12` means "where this plate reads 0.12",
    the same number a pixel probe in Nuke would show. The merge itself was
    never affected; only where the ramps landed.
    """
    return _encode_from_linear(_luminance(plate_lin), mode)


def _mid_mask(y_key: torch.Tensor, mid_low: float, mid_high: float) -> torch.Tensor:
    """Feathered midtone band in the key domain."""
    lo_hi = min(mid_low + _MID_FEATHER, mid_high)
    hi_lo = max(mid_high - _MID_FEATHER, mid_low)
    return (_smoothstep(mid_low, lo_hi, y_key)
            * _smoothstep(mid_high, hi_lo, y_key))


def _tonal_ramps(y_key: torch.Tensor, p: dict) -> tuple:
    """The two tonal keys, kept apart.

    They are returned separately rather than pre-combined because the shadow
    and highlight ends want *opposite* amounts of level transfer, and merging
    them early throws away the only information needed to tell them apart. A
    crushed black has to be lifted off zero before any detail in it can be
    seen; a clipped highlight must NOT be lifted, because the display shoulder
    would then squash the recovered structure back into the last fraction of a
    stop and you would see nothing. Same node, same weight map, opposite
    treatment. See `shadow_tone_transfer` / `highlight_tone_transfer`.
    """
    w_sh = (_smoothstep(p["shadow_full"], p["shadow_start"], y_key)
            .mul(-1.0).add(1.0)
            .mul(p["shadow_strength"]).clamp(0.0, 1.0))
    w_hi = (_smoothstep(p["highlight_start"], p["highlight_full"], y_key)
            .mul(p["highlight_strength"]).clamp(0.0, 1.0))
    if p["blur_radius"] > 0:
        # Blur each ramp, not their union: the union is blurred implicitly and
        # the per-domain tone amounts stay smooth too.
        r = int(p["blur_radius"])
        w_sh = _gaussian_blur_hw(w_sh, r).clamp(0.0, 1.0)
        w_hi = _gaussian_blur_hw(w_hi, r).clamp(0.0, 1.0)
    return w_sh, w_hi


def _log_detail(y: torch.Tensor, radius: int) -> tuple:
    """Split a luminance map into (log local average, log local detail).

    Multiplicative rather than additive separation — ``log(y) - log(blur(y))``
    — because local contrast in light is a ratio, not a difference. Transplant
    an additive detail band from a bright HDR onto a dark plate and it swamps
    it; transplant the ratio and a 10% local ripple stays a 10% local ripple
    wherever it lands.

    ``_BLACK_FLOOR`` is the log offset, not ``_EPS``: the offset sets how far
    into the blacks the separation stays meaningful, and with ``_EPS`` the
    result down there would be dominated by the epsilon rather than by the
    picture.
    """
    base = _gaussian_blur_hw(y, radius).clamp(min=0.0)
    lb = torch.log(base + _BLACK_FLOOR)
    return lb, torch.log(y.clamp(min=0.0) + _BLACK_FLOOR) - lb


def _tonal_ramps_union(y_key: torch.Tensor, p: dict) -> torch.Tensor:
    """max(shadow ramp, highlight ramp) — the weight that drives detail and
    colour. Split out so callers that only want the union (and the tests)
    don't have to reproduce the combine rule."""
    w_sh, w_hi = _tonal_ramps(y_key, p)
    return torch.maximum(w_sh, w_hi)


def _sanitise_hdr(chunk: torch.Tensor) -> torch.Tensor:
    """Scrub NaN/Inf and bound one chunk of hdr_ai.

    Chunk-sized rather than whole-tensor: LTX already clamps its output to
    [0, 1e4], so this is a guard against something else in the chain, and
    paying 2.8 GB of resident copies for a guard is not a trade worth making.

    `nan_to_num` returns a fresh tensor, so the `clamp_` after it is safe in
    place — but note that `chunk.float()` on an already-float32 tensor returns
    a VIEW of the caller's input, which is exactly why the in-place op has to
    come after nan_to_num and never before it.
    """
    return torch.nan_to_num(chunk.float(), nan=0.0, posinf=1e4,
                            neginf=0.0).clamp_(min=0.0, max=1e4)


def _composite_chunk(plate: torch.Tensor, hdr: torch.Tensor, k: float,
                     p: dict) -> torch.Tensor:
    """One chunk of frames -> merged scene-linear RGB.

    plate (B,H,W,3) as supplied, hdr (B,H,W,3) already linear. Mirrored
    pixel-for-pixel by `applyComposite()` in the sibling .js — the two have to
    be changed together or the preview starts lying.
    """
    plate_lin = _to_linear(plate, p["plate_gamma_mode"])
    hdr_m = (hdr * k).clamp(min=0.0)

    y_src = _luminance(plate_lin)
    y_hdr = _luminance(hdr_m)
    y_key = _key_luma(plate_lin, p["plate_gamma_mode"])

    w_sh, w_hi = _tonal_ramps(y_key, p)
    # The union drives detail and colour, which both ends want equally.
    # `max` rather than a sum: where the ramps overlap (only if the artist has
    # crossed the thresholds) the intent is "let the HDR in", not "twice".
    w = torch.maximum(w_sh, w_hi)

    # ── stage 1: level ───────────────────────────────────────────────────
    # Straight linear-light lerp of the target luminance, per tonal end.
    # Linear rather than geometric on purpose: a log-domain lerp here would
    # make deep-shadow behaviour a function of the log offset rather than of
    # the picture, and shadow recovery is exactly what this node must get right.
    a_tone = torch.maximum(w_sh * p["shadow_tone_transfer"],
                           w_hi * p["highlight_tone_transfer"]).clamp(0.0, 1.0)
    # Bounded by the inputs by construction — a lerp cannot leave
    # [min(y_src, y_hdr), max(y_src, y_hdr)] — so nothing here needs a
    # ceiling. Putting one here, as this node originally did (a ratio against
    # the plate), silently truncated the HDR's real scene range: with a plate
    # clipped at white, a max_gain of 8 turned every input above 8.08 linear
    # into 8.08 — precisely the data an SDR->HDR pass exists to recover. The
    # only cap left is on the detail stage below.
    y_level = y_src + (y_hdr - y_src) * a_tone
    y_mix = y_level

    # ── stage 2: local contrast ──────────────────────────────────────────
    # Push the output's neighbourhood-relative structure toward the HDR's
    # without moving the level stage 1 just chose.
    a_det = (w * p["detail_transfer"]).clamp(0.0, 1.0)
    if p["detail_transfer"] > 0.0 and p["detail_radius"] > 0 and float(w.max()) > 0.0:
        _, d_src = _log_detail(y_src, p["detail_radius"])
        _, d_hdr = _log_detail(y_hdr, p["detail_radius"])
        # The level lerp already dragged roughly `a_tone` of the HDR's detail
        # along with it, so only inject the shortfall. This is what keeps
        # tone_transfer = detail_transfer = 1 a clean pass-through to the HDR
        # instead of applying its texture twice.
        extra = (a_det - a_tone).clamp(min=0.0)
        # The one genuinely unbounded term in the node. Frequency separation
        # haloes by nature, and an unclamped exponent turns a single noisy
        # pixel in a crushed black into a firefly.
        lim = math.log(max(p["max_detail_gain"], 1.0 + 1e-6))
        boost = (extra * (d_hdr - d_src)).clamp(-lim, lim)
        y_mix = y_level * torch.exp(boost)

    # Optional absolute ceiling, OFF by default. Faithfulness to the HDR in
    # the extremes is the node's whole job, so it does not get to decide a
    # bright value is "wrong" unless asked — but LTX throws the occasional
    # firefly and clamping it here beats hunting it down in Nuke.
    if p["hdr_ceiling"] > 0.0:
        y_mix = y_mix.clamp(max=p["hdr_ceiling"])
    y_mix = y_mix.clamp(min=0.0)

    # ── stage 3: get there without moving the colour ─────────────────────
    # Unit-luminance chroma directions. (1,1,1) is the neutral fallback and
    # has Rec.709 luminance of exactly 1, so it belongs in this set.
    ones = torch.ones_like(plate_lin)
    dir_plate = torch.where(y_src.unsqueeze(-1) > _EPS,
                            plate_lin / y_src.unsqueeze(-1).clamp(min=_EPS), ones)
    dir_hdr = torch.where(y_hdr.unsqueeze(-1) > _EPS,
                          hdr_m / y_hdr.unsqueeze(-1).clamp(min=_EPS), ones)

    # How much plate chroma there actually is to preserve. Goes to 0 as the
    # plate goes black, which is what stops "preserve_plate_chroma = 1" from
    # meaning "preserve this pixel's meaningless near-zero hue".
    conf = y_src / (y_src + _BLACK_FLOOR)
    c = (conf * p["preserve_plate_chroma"]).unsqueeze(-1)
    direction = dir_plate * c + dir_hdr * (1.0 - c)

    # Additive form of the luminance transplant — algebraically identical to
    # plate * (y_mix / y_src) where the plate has signal, but able to lift a
    # crushed black, which the multiply cannot.
    out = plate_lin + (y_mix - y_src).unsqueeze(-1) * direction
    out = out.clamp(min=0.0)

    # ── stage 4: a little of the HDR's own colour ────────────────────────
    # Chroma ONLY. A plain lerp toward hdr_m here would drag the luminance
    # along with the colour and quietly undo stage 1 — measured on a clipped
    # sky with highlight_tone_transfer at 0, a 0.25 mix pushed the level from
    # 1.0 to 2.0 linear, moving a level the tone stage had deliberately pinned.
    # So rescale the HDR to the luminance we already decided
    # on before mixing. The output's luminance is exactly `y_mix` either way,
    # which is the invariant the whole node rests on.
    if p["hdr_rgb_mix"] > 0.0:
        y_h = y_hdr.unsqueeze(-1)
        hdr_at_target = torch.where(y_h > _EPS,
                                    hdr_m * (y_mix.unsqueeze(-1) / y_h.clamp(min=_EPS)),
                                    out)
        m = (w * p["hdr_rgb_mix"]).clamp(0.0, 1.0).unsqueeze(-1)
        out = out * (1.0 - m) + hdr_at_target * m

    return out


def _display(lin: torch.Tensor, p: dict) -> torch.Tensor:
    """Scene-linear -> display-referred sRGB, via the chosen shoulder.

    `preview_exposure` is baked here; the viewer exposure on the node's canvas
    is a separate, display-only gain that never reaches this function.
    """
    x = lin * (2.0 ** p["preview_exposure"])
    mode = p["preview_tonemap"]
    if mode == "reinhard":
        x = x / (1.0 + x)
    elif mode == "soft_rolloff":
        x = _soft_rolloff(x, _knee_from_headroom(p["highlight_headroom"]))
    # "clip" falls through: the clamp below is the whole transform.
    #
    # Encoded with the OUTPUT curve, not unconditionally sRGB. Getting this
    # wrong meant a Rec.709 plate came back through a preview that had been
    # sRGB-encoded — 0.062 off at zero HDR weight, i.e. the node visibly
    # altered a plate it was supposed to be leaving alone.
    mode = p.get("_out_mode")
    if mode is None:
        om = p.get("output_gamma_mode", "match_plate")
        mode = p.get("plate_gamma_mode", "srgb") if om == "match_plate" else om
    return _encode_from_linear(x.clamp(0.0, 1.0), mode).clamp(0.0, 1.0)


class BatHDRTonalComposite:
    """Fold an LTX HDR reconstruction into a plate, in the tonal extremes only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # PROMPT + UNIQUE_ID let the node see which of its outputs anything
            # downstream actually consumes, so it can skip allocating the rest.
            # On a 100-frame 2K clip each skipped output is 2.8 GB.
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
            "required": {
                "plate_sdr": ("IMAGE", {
                    "tooltip": "The original / AI-edited SDR plate. This is the "
                               "picture the result stays faithful to."}),
                "hdr_ai": ("IMAGE", {
                    "tooltip": "The 'hdr_linear' output of LTXVHDRDecodePostprocess "
                               "— NOT 'tonemapped'. Raw scene-linear HDR."}),
            },
            "optional": {
                "plate_gamma_mode": (["srgb", "rec709", "gamma_2_2", "gamma_2_4", "linear"], {
                    "default": "srgb",
                    "tooltip": "How plate_sdr is encoded, so it can be decoded to linear "
                               "before compositing. Match this to how the plate is read in "
                               "your comp package — sRGB and Rec.709 are NOT the same curve "
                               "and differ by up to 65% in the shadows, which is where this "
                               "node works. gamma_2_4 is BT.1886. Use 'linear' only if the "
                               "plate is already scene-linear."}),

                "auto_match_mids": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Scale the HDR input so its midtones sit on the "
                               "plate's midtones before blending. Without this, a "
                               "global exposure offset in the LTX result leaks into "
                               "the extremes as a brightness step."}),
                "match_scope": (["whole_batch", "per_frame"], {
                    "default": "whole_batch",
                    "tooltip": "whole_batch: one match ratio for the entire clip — "
                               "flicker-free, use this for video. per_frame: match "
                               "each frame independently; tracks a changing scene but "
                               "can pump on cuts or fast lighting changes."}),
                "mid_low": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Lower edge of the midtone band used for "
                                                 "exposure matching (display-referred)."}),
                "mid_high": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Upper edge of the midtone match band "
                                                  "(display-referred)."}),

                "shadow_start": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Below this display-referred luma the HDR "
                                                      "begins to contribute."}),
                "shadow_full": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "At and below this, the HDR contributes "
                                                     "fully. Must be lower than shadow_start."}),
                "shadow_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                              "tooltip": "Scales the whole shadow ramp. 0 disables "
                                                         "shadow recovery."}),

                "highlight_start": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Above this display-referred luma the HDR "
                                                         "begins to contribute."}),
                "highlight_full": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "At and above this, the HDR contributes "
                                                        "fully — i.e. the clipped region."}),
                "highlight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                                 "tooltip": "Scales the whole highlight ramp. 0 "
                                                            "disables highlight recovery."}),

                "shadow_tone_transfer": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "How far the shadow LEVEL moves toward the HDR. "
                                                              "Wants to be high: a region crushed to 0 has to "
                                                              "be lifted off zero before any detail in it can "
                                                              "be seen."}),
                "highlight_tone_transfer": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                                                      "tooltip": "How far the highlight LEVEL moves toward the "
                                                                 "HDR. Wants to be LOW — usually 0. Lifting a "
                                                                 "clipped sky to its true linear value just "
                                                                 "pushes it past the display shoulder, which "
                                                                 "squashes the recovered detail back into "
                                                                 "nothing. Raise it only as far as "
                                                                 "highlight_headroom has given it somewhere to "
                                                                 "go — the two work as a pair, and this one "
                                                                 "alone makes things worse."}),
                "detail_transfer": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "How far the LOCAL CONTRAST moves toward the HDR, "
                                                         "without disturbing the level. This is the "
                                                         "'more detail, same tone' control. Off by default "
                                                         "because it is a frequency separation and costs two "
                                                         "blurs per chunk; turn it up (and pull "
                                                         "tone_transfer down) when you want texture from the "
                                                         "HDR rather than level."}),
                "detail_radius": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1,
                                          "tooltip": "Frequency split, in pixels: structure finer than this "
                                                     "counts as detail, coarser counts as tone. Scales with "
                                                     "resolution — roughly 16 at HD, 32 at UHD. 0 disables "
                                                     "the detail stage."}),

                "preserve_plate_chroma": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                                    "tooltip": "1 = the merge only changes brightness and "
                                                               "keeps the plate's hue/saturation. 0 = adopt "
                                                               "the HDR's colour as brightness moves. Auto-"
                                                               "falls toward the HDR near black, where the "
                                                               "plate has no colour left to keep."}),
                "hdr_rgb_mix": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "How much of the HDR's own COLOUR is mixed in, in the "
                                                     "weighted regions. Brightness is unaffected — the HDR "
                                                     "is rescaled to the target luminance first. Reaches "
                                                     "what a luminance operation cannot: a clipped sky is "
                                                     "flat AND hueless in the plate, so only the HDR knows "
                                                     "what colour it should be."}),

                "max_detail_gain": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 32.0, "step": 0.1,
                                              "tooltip": "How far the local-contrast stage may push one pixel "
                                                         "from its own neighbourhood. Bounds the only term in "
                                                         "the node that can run away. Does NOT limit the HDR's "
                                                         "scene range — that passes through intact."}),
                "hdr_ceiling": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                                          "tooltip": "Absolute clamp on the merged linear luminance. 0 = OFF, "
                                                     "the default: linear_out carries whatever scene values "
                                                     "the HDR had — 50, 200, 1000. Set it only to kill LTX "
                                                     "fireflies, well above your real highlights."}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                        "tooltip": "Softens the tonal weight map. Use if the ramps are tight "
                                                   "enough to crunch on noise."}),

                "output_gamma_mode": (["match_plate", "srgb", "rec709",
                                       "gamma_2_2", "gamma_2_4"], {
                    "default": "match_plate",
                    "tooltip": "Encoding of image_out. 'match_plate' re-encodes with whatever "
                               "plate_gamma_mode decoded, so with no HDR contribution the "
                               "preview returns the plate untouched — which is the property "
                               "that makes it trustworthy. (A 'linear' plate has no display "
                               "curve to match, so it falls back to sRGB and stays viewable.) "
                               "Does not affect linear_out."}),

                "linear_out_primaries": (["rec709", "acescg", "aces2065-1"], {
                    "default": "rec709",
                    "tooltip": "Colour primaries of linear_out. Affects linear_out ONLY — "
                               "image_out stays in the plate's primaries so the ComfyUI "
                               "preview keeps matching the plate. The plate and LTX are both "
                               "Rec.709, so 'rec709' passes through untouched. Pick 'acescg' "
                               "if your convention is that EXRs are working-space and you read "
                               "them as scene_linear. If you would rather let Nuke do it, leave "
                               "this at rec709 and set the Read to 'Utility - Linear - Rec.709' "
                               "(ACES 1.2) or 'Linear Rec.709 (sRGB)' (ACES 1.3) — do ONE or the "
                               "other, never both."}),

                "preview_tonemap": (["soft_rolloff", "reinhard", "clip"], {
                    "default": "soft_rolloff",
                    "tooltip": "soft_rolloff: plate untouched below the knee, hyperbolic shoulder "
                               "above, sized by highlight_headroom — the default, because it leaves "
                               "the mids alone and still shows what is above white. reinhard: matches "
                               "the LTX preview exactly, but darkens the whole plate. clip: hard "
                               "clamp, discards everything the HDR pushed above white."}),
                "highlight_headroom": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.25,
                                                 "tooltip": "How many stops above plate white the preview "
                                                            "makes room for. THIS is what lets a recovered "
                                                            "highlight read as bright rather than just "
                                                            "textured — a display has nothing above white, so "
                                                            "the only way to show one is to move the plate's "
                                                            "white down and spend the range on it. 0 = hard "
                                                            "clip, plate whites exact, nothing above white "
                                                            "visible. 2 = plate white at ~231/255 with four "
                                                            "stops of separation above it. Midtones are "
                                                            "untouched either way."}),
                "preview_exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,
                                               "tooltip": "Exposure in stops applied before the tonemap. "
                                                          "Baked into the output, unlike the viewer exposure "
                                                          "on the preview strip."}),

                "preview_frame": ("INT", {"default": 0, "min": 0, "max": 9999,
                                          "tooltip": "Which frame of the batch feeds the live canvas. Does "
                                                     "not affect the render."}),
                **{f"hdr_ai_{i}": ("IMAGE", {
                    "tooltip": f"Another LTX hdr_linear of the SAME plate. Composited with "
                               f"the identical settings and emitted on image_out_{i} / "
                               f"linear_out_{i}. Added and removed with the +/- HDR version "
                               f"buttons."}) for i in range(2, MAX_HDR_VERSIONS + 1)},

                "preview_ocio_view": (_ocio_view_choices(), {
                    "default": "(off)",
                    "tooltip": "Bake this OCIO view transform into a 3D LUT and hand it to "
                               "the on-node canvas, so the preview matches what you actually "
                               "look at. A gamma setting cannot do this: an ACES Output "
                               "Transform is RRT + ODT, a filmic tone curve, and no choice of "
                               "curve is its inverse. Pick your display transform (e.g. "
                               "out_rec709), then set the canvas's view control to OCIO. "
                               "Display-only — it never touches image_out or linear_out. "
                               "Costs ~95 KB per run, so it is off by default."}),

                "preview_resolution": (["256", "384", "512", "768"], {
                    "default": "512",
                    "tooltip": "Long edge of the live preview, in pixels. The canvas composites from "
                               "this directly, so it is also the sharpness of what you see. Measured on "
                               "a real 1080p frame the two tiles cost about 230 / 480 / 850 / 1790 KB "
                               "per run — drop it if you are iterating hard or driving the UI "
                               "remotely."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE") * MAX_HDR_VERSIONS
    RETURN_NAMES = tuple(
        n for i in range(MAX_HDR_VERSIONS)
        for n in ((f"image_out_{i + 1}", f"linear_out_{i + 1}") if i else
                  ("image_out", "linear_out")))
    OUTPUT_TOOLTIPS = tuple(
        t for i in range(MAX_HDR_VERSIONS) for t in (
            ("Display-referred preview, [0,1]. Safe for PreviewImage / SaveImage."
             if i == 0 else f"Display-referred preview for hdr_ai_{i + 1}."),
            ("The merged result in scene-linear, unclamped and untonemapped — "
             "values above 1.0 intact. This is the EXR/Nuke output; sending it to "
             "a PNG saver will clip it." if i == 0 else
             f"Scene-linear merge for hdr_ai_{i + 1}.")))
    FUNCTION = "composite"
    CATEGORY = "BAT/Colour"
    DESCRIPTION = (
        "Composites an LTX hdr_linear reconstruction into an SDR plate in the "
        "tonal extremes only — shadows and highlights gain the HDR's recovered "
        "range while the midtones and the plate's colour stay put. Tone-keyed, "
        "so there are no matte edges. Two-stage transfer: tone_transfer moves the "
        "level, detail_transfer moves the local contrast without moving the level. "
        "Outputs an sRGB preview and the raw scene-linear merge for EXR. Live canvas "
        "preview on the node body. Use the +/- HDR version buttons to run several "
        "LTX reconstructions of the same plate through one set of settings."
    )

    # ------------------------------------------------------------------
    def composite(self, plate_sdr, hdr_ai,
                  plate_gamma_mode="srgb",
                  auto_match_mids=True, match_scope="whole_batch",
                  mid_low=0.10, mid_high=0.70,
                  shadow_start=0.12, shadow_full=0.03, shadow_strength=1.0,
                  highlight_start=0.75, highlight_full=0.98, highlight_strength=1.0,
                  shadow_tone_transfer=1.00, highlight_tone_transfer=0.50,
                  detail_transfer=0.00, detail_radius=16,
                  preserve_plate_chroma=0.85, hdr_rgb_mix=0.25,
                  max_detail_gain=4.0, hdr_ceiling=0.0, blur_radius=0,
                  output_gamma_mode="match_plate", linear_out_primaries="rec709",
                  preview_tonemap="soft_rolloff", highlight_headroom=2.0,
                  preview_exposure=0.0, preview_frame=0,
                  preview_resolution="512", preview_ocio_view="(off)",
                  prompt=None, unique_id=None, **hdr_versions):

        p = {
            "plate_gamma_mode": str(plate_gamma_mode),
            "mid_low": float(mid_low), "mid_high": float(mid_high),
            "shadow_start": float(shadow_start), "shadow_full": float(shadow_full),
            "shadow_strength": float(shadow_strength),
            "highlight_start": float(highlight_start), "highlight_full": float(highlight_full),
            "highlight_strength": float(highlight_strength),
            "shadow_tone_transfer": float(shadow_tone_transfer),
            "highlight_tone_transfer": float(highlight_tone_transfer),
            "detail_transfer": float(detail_transfer),
            "detail_radius": int(detail_radius),
            "preserve_plate_chroma": float(preserve_plate_chroma),
            "hdr_rgb_mix": float(hdr_rgb_mix),
            "max_detail_gain": float(max_detail_gain),
            "hdr_ceiling": float(hdr_ceiling), "blur_radius": int(blur_radius),
            "preview_tonemap": str(preview_tonemap),
            "output_gamma_mode": str(output_gamma_mode),
            "linear_out_primaries": str(linear_out_primaries),
            # Resolved once here so _display never has to re-derive it.
            "_out_mode": (str(plate_gamma_mode) if output_gamma_mode == "match_plate"
                          else str(output_gamma_mode)),
            "highlight_headroom": float(highlight_headroom),
            "preview_exposure": float(preview_exposure),
        }

        # Version 1 is the required input; the rest arrive as optional kwargs.
        # A gap is tolerated (hdr_ai_2 empty, hdr_ai_3 wired) rather than
        # silently shifting a later version onto an earlier output pair.
        versions = [("hdr_ai", hdr_ai)]
        for i in range(2, MAX_HDR_VERSIONS + 1):
            versions.append((f"hdr_ai_{i}", hdr_versions.get(f"hdr_ai_{i}")))

        used = self._consumed_slots(prompt, unique_id)
        results = []
        first_plate = first_hdr = None
        k_first = 1.0
        for name, img in versions:
            if img is None:
                results.append(None)
                continue
            try:
                pl, hd = self._align(plate_sdr, img)
            except ValueError as exc:
                # Name the offending input: with eight of them, "resolution
                # mismatch" on its own is not enough to find the bad wire.
                raise ValueError(f"{name}: {exc}") from None
            slot = len(results) * 2
            out, lin, k = self._run_one(
                pl, hd, p, bool(auto_match_mids), str(match_scope),
                str(linear_out_primaries),
                want_display=(used is None or slot in used),
                want_linear=(used is None or (slot + 1) in used))
            results.append((out, lin))
            if first_plate is None:
                first_plate, first_hdr, k_first = pl, hd, k

        if first_plate is None:
            raise ValueError(
                "Bat_HDRTonalComposite: hdr_ai is required — wire the hdr_linear "
                "output of LTXVHDRDecodePostprocess to it.")

        ui = self._preview_payload(first_plate, first_hdr, k_first, p, preview_frame,
                                   bool(auto_match_mids), str(match_scope),
                                   int(preview_resolution))
        # The LUT's source is the plate's primaries, which is what the canvas
        # works in — NOT linear_out_primaries, which is applied downstream of
        # everything the canvas ever sees.
        if preview_ocio_view and preview_ocio_view != "(off)":
            lut = _view_lut(str(preview_ocio_view), "rec709")
            if lut is not None:
                ui["view_lut"] = [lut]

        # Every declared slot must be returned even when its input is empty, or
        # the executor's index mapping breaks. Spares repeat version 1 rather
        # than being black, so a connection to a trimmed slot gives a sane
        # picture instead of a mystery.
        spare = results[0]
        flat = []
        for r in results:
            flat.extend(r if r is not None else spare)
        return {"ui": ui, "result": tuple(flat)}

    # ------------------------------------------------------------------
    @staticmethod
    def _consumed_slots(prompt, unique_id):
        """Which of this node's output slots anything downstream reads.

        Returns None when it cannot tell — no prompt, an unexpected shape, our
        own id missing — and every caller treats None as "all of them", so a
        failure here costs memory rather than correctness.

        The API prompt is {id: {"inputs": {name: value | [src_id, slot]}}}, so a
        consumer of our slot N appears as the pair [our_id, N].
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

    def _run_one(self, plate, hdr, p, auto_match_mids, match_scope, primaries,
                 want_display=True, want_linear=True):
        """One plate + one HDR reconstruction -> (image_out, linear_out, k).

        Split out of `composite` so several hdr_ai inputs share one set of
        settings. Each version gets its OWN exposure match: separate LTX runs
        have no reason to land on the same absolute scale, and reusing version
        1's ratio would drag the others off.
        """
        b, h, w, _ = plate.shape
        per_chunk = max(1, int(_CHUNK_BYTES // max(h * w * 3 * 4, 1)))

        # Pass 1 — the exposure match. Accumulated over chunks as two scalars
        # so a 300-frame clip costs no more memory than a 10-frame one.
        k_batch = 1.0
        if auto_match_mids and match_scope == "whole_batch":
            num = den = 0.0
            for s in range(0, b, per_chunk):
                _interrupt.check()
                pl = _to_linear(plate[s:s + per_chunk].float(), p["plate_gamma_mode"])
                hd = _sanitise_hdr(hdr[s:s + per_chunk])
                m = _mid_mask(_key_luma(pl, p["plate_gamma_mode"]),
                              p["mid_low"], p["mid_high"])
                num += float((_luminance(pl) * m).sum())
                den += float((_luminance(hd) * m).sum())
            k_batch = self._ratio(num, den)

        # Not empty_like: `plate` may be an expand()ed view with zero strides
        # after single-frame broadcasting, and preserve_format would inherit them.
        # Only allocate what something downstream will read. A 1x1 stand-in
        # keeps the return arity right for the executor's index mapping without
        # paying for a full-size buffer nobody asked for.
        stub = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=plate.device)
        out = (torch.empty(plate.shape, dtype=torch.float32, device=plate.device)
               if want_display else stub)
        lin = (torch.empty(plate.shape, dtype=torch.float32, device=plate.device)
               if want_linear else stub)
        # Resolved once, outside the chunk loop — an OCIO config lookup per
        # chunk would be absurd, and the cache makes it once per process anyway.
        pm = _primaries_matrix(primaries, plate.device, torch.float32)
        for s in range(0, b, per_chunk):
            # Once per chunk. At one frame per chunk on a 2K clip this is a
            # flag read per frame — free, and it is what makes Cancel work.
            _interrupt.check()
            e = min(s + per_chunk, b)
            pl_raw = plate[s:e].float()
            hd = _sanitise_hdr(hdr[s:e])

            if not auto_match_mids:
                k = 1.0
            elif match_scope == "whole_batch":
                k = k_batch
            else:
                pl_lin = _to_linear(pl_raw, p["plate_gamma_mode"])
                m = _mid_mask(_key_luma(pl_lin, p["plate_gamma_mode"]),
                              p["mid_low"], p["mid_high"])
                k = self._ratio(float((_luminance(pl_lin) * m).sum()),
                                float((_luminance(hd) * m).sum()))

            merged = _composite_chunk(pl_raw, hd, k, p)
            # image_out first: it must stay in the plate's primaries, so it is
            # derived before any gamut conversion.
            if want_display:
                out[s:e] = _display(merged, p)
            # linear_out is the merge itself: no exposure, no tonemap, no clamp
            # at the top. preview_exposure is a look control on the preview and
            # has no business baking itself into a linear deliverable.
            if want_linear:
                if pm is None:
                    lin[s:e] = merged
                else:
                    # (B,H,W,3) @ Mt -> row vectors through the matrix.
                    lin[s:e] = torch.matmul(merged, pm.transpose(0, 1))
        return out, lin, k_batch

    # ------------------------------------------------------------------
    @staticmethod
    def _ratio(num: float, den: float) -> float:
        """Midtone exposure-match ratio, with every degenerate case falling
        back to 'don't touch it'."""
        if not math.isfinite(num) or not math.isfinite(den) or den <= _EPS or num <= 0.0:
            return 1.0
        return float(min(max(num / den, _K_MIN), _K_MAX))

    @staticmethod
    def _align(plate: torch.Tensor, hdr: torch.Tensor):
        """Validate, and broadcast a single-frame side across the other's batch.

        Resolution mismatch is a hard error rather than a silent resample:
        in the LTX HDR graph it almost always means the plate was taken from
        before the ResizeImageMask that feeds the sampler, and quietly
        resampling a plate is not something a comp node should do behind the
        artist's back.
        """
        if plate.ndim != 4 or hdr.ndim != 4:
            raise ValueError(
                f"Bat_HDRTonalComposite expects [B,H,W,C] IMAGE tensors, got "
                f"plate_sdr {tuple(plate.shape)} and hdr_ai {tuple(hdr.shape)}.")
        if plate.shape[1:3] != hdr.shape[1:3]:
            raise ValueError(
                f"Bat_HDRTonalComposite: resolution mismatch — plate_sdr is "
                f"{plate.shape[2]}x{plate.shape[1]} but hdr_ai is "
                f"{hdr.shape[2]}x{hdr.shape[1]}. Feed the plate through the same "
                f"resize the LTX sampler saw, or resize it to match.")
        if plate.shape[3] < 3 or hdr.shape[3] < 3:
            raise ValueError("Bat_HDRTonalComposite: both inputs need at least 3 channels.")
        plate = plate[..., :3]
        hdr = hdr[..., :3]

        bp, bh = plate.shape[0], hdr.shape[0]
        if bp != bh:
            if bp == 1:
                plate = plate.expand(bh, -1, -1, -1)
            elif bh == 1:
                hdr = hdr.expand(bp, -1, -1, -1)
            else:
                raise ValueError(
                    f"Bat_HDRTonalComposite: frame-count mismatch — plate_sdr has "
                    f"{bp} frames, hdr_ai has {bh}. LTX's latent temporal compression "
                    f"often changes the count; trim one side to match (or pass a "
                    f"single frame, which broadcasts).")

        # LTX clamps its HDR to [0, 1e4] but nothing guarantees a clean tensor
        # arrives here if something else is in the chain.

        # LTXVHDRDecodePostprocess has two outputs and they sit next to each
        # other: 'tonemapped' (sRGB, [0,1]) and 'hdr_linear' (scene-linear).
        # Wiring the first one here is silent — it is a valid IMAGE and the
        # node happily treats it as linear — but it throws away the entire
        # point, because there is no above-white data left to recover. A dim
        # linear HDR can also legitimately peak below 1.0, so this warns
        # rather than raising.
        # NOTE: hdr is returned UNSANITISED. nan_to_num + clamp on the whole
        # tensor allocated two full-size copies (2.8 GB each on a 100-frame 2K
        # clip) that stayed resident for the entire run; the chunk loop does it
        # per chunk instead, for a few MB. See _sanitise_hdr.
        if float(hdr.max()) <= 1.0:
            logger.warning(
                "hdr_ai peaks at %.3f — nothing above white. If this came from "
                "LTXVHDRDecodePostprocess, check you wired 'hdr_linear' and not "
                "'tonemapped'; the tonemapped output is display-referred and has "
                "already discarded the HDR range this node exists to composite.",
                float(hdr.max()))
        return plate, hdr

    # ------------------------------------------------------------------
    def _preview_payload(self, plate, hdr, k_batch, p, preview_frame,
                         auto_match, match_scope, tile_dim=512):
        """Everything the JS canvas needs to re-run this algorithm locally.

        Both tiles come off the same point-sampler at the same long-edge cap,
        and the inputs are guaranteed same-resolution by _align, so the two
        arrive pixel-aligned and the JS loop can index them with one counter.
        """
        idx = max(0, min(int(preview_frame), plate.shape[0] - 1))
        pf = plate[idx].float()
        # hdr arrives unsanitised now that the scrub moved into the chunk loop,
        # so a NaN would reach hdr_tile's uint16 cast and pack garbage. One
        # frame, so doing it here costs nothing.
        hf = _sanitise_hdr(hdr[idx])

        u8 = (pf.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        ui = {
            "plate_jpeg": [_b64_jpeg(u8)],
            "w": [int(plate.shape[2])],
            "h": [int(plate.shape[1])],
            "frames": [int(plate.shape[0])],
            "preview_frame": [int(idx)],
            # Shipped so the canvas can use the exact whole-batch ratio rather
            # than re-deriving a one-frame approximation of it — but only while
            # the midtone widgets still hold the values it was computed from.
            "k_batch": [float(k_batch)],
            "k_valid": [bool(auto_match and match_scope == "whole_batch")],
            "k_mid_low": [float(p["mid_low"])],
            "k_mid_high": [float(p["mid_high"])],
            "k_gamma_mode": [p["plate_gamma_mode"]],
        }

        # The plate tile goes down *as received* (still display-referred), so
        # flipping plate_gamma_mode updates the canvas without a re-run.
        #
        # Area-sampled, unlike Bat_Grade's tiles. That node point-samples so its
        # Inspect mode can show real quantisation steps; this one composites
        # from the tile and shows the result full-width on the node body, where
        # a nearest-neighbour decimate of a 1920-wide frame keeps one pixel in
        # 56 and reads as broken. Both tiles use the same filter and long edge,
        # so they stay pixel-aligned for the JS loop.
        t_plate = hdr_tile(pf, tile_dim, sample="area")
        t_hdr = hdr_tile(hf, tile_dim, sample="area")
        if t_plate is not None:
            ui["plate_tile"] = [t_plate]
        if t_hdr is not None:
            ui["hdr_tile"] = [t_hdr]
        elif t_plate is not None:
            # Without the HDR tile there is nothing to composite against, and a
            # canvas showing the bare plate would read as "the node does
            # nothing". Say so instead.
            logger.warning("HDR preview tile could not be built; live canvas "
                           "will show the plate only.")
        return ui
