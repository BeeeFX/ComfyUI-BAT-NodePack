"""
Blend modes, shared by the BAT compositing nodes.

The formulas are the W3C Compositing and Blending spec's, which is also what
Photoshop implements and what Nuke's Merge node calls its "Photoshop" set. They
are here rather than in a node because there are now three implementations of
each one that must agree exactly — this file, ``web/bat_blend_modes.js`` for the
live preview, and the worker that shares that JS — and a table per node is how
they drift.

Separable vs non-separable
--------------------------
Most modes are *separable*: one function applied independently per channel.
Four are not — ``hue``, ``saturation``, ``color`` and ``luminosity`` reason about
the pixel as a colour, so they take and return an RGB triple. Consumers have to
branch on which kind they were handed, which is why ``is_separable()`` exists
rather than a single uniform signature that would have to fake it.

Two things about the non-separable four that look like mistakes and are not
-------------------------------------------------------------------------
* **The luma weights are 0.3 / 0.59 / 0.11, not Rec.709.** Those are the
  coefficients the spec mandates for these four modes, and they are what
  Photoshop uses, so matching them is what makes a Photoshop file and this node
  agree. Elsewhere in the pack (``_luminance`` in the HDR nodes) Rec.709 is
  correct and used; the two are different jobs, not an inconsistency.
* **``clip_color`` can look like it does nothing.** It only engages where a
  luminosity change has pushed a channel outside [0,1], which on ordinary
  footage is a minority of pixels — but without it those pixels shift hue as
  they clip instead of desaturating toward the luminance, which is very visible
  on a saturated highlight.

Range
-----
Nothing here clamps. Every mode is defined against white = 1.0 — that is what
"screen" or "overlay" *mean* — so on scene-linear input above white they still
compute, but what they compute stops corresponding to the mode's name. The
arithmetic modes (``normal``, ``plus``, ``minus``, ``darken``, ``lighten``) are
the range-agnostic ones. Callers wanting a display-referred result clamp
themselves.
"""

import torch

EPS = 1e-6

# W3C's coefficients for the non-separable modes. NOT Rec.709 — see the module
# docstring.
LUM_R, LUM_G, LUM_B = 0.3, 0.59, 0.11


# ---------------------------------------------------------------------------
# Separable modes: f(backdrop, source) applied per channel
# ---------------------------------------------------------------------------

def _normal(cb, cs):
    # No alpha here: the caller composites with the layer's mask/opacity, so
    # "normal" is simply the source and the mix falls out of that.
    return cs


def _multiply(cb, cs):
    return cb * cs


def _screen(cb, cs):
    return cb + cs - cb * cs


def _overlay(cb, cs):
    # Hard-light with the operands swapped, which is the spec's own definition.
    return _hard_light(cs, cb)


def _darken(cb, cs):
    return torch.minimum(cb, cs)


def _lighten(cb, cs):
    return torch.maximum(cb, cs)


def _color_dodge(cb, cs):
    # Spec: 0 where the backdrop is 0 (so a black backdrop stays black rather
    # than being lifted by the division), 1 where the source is 1.
    out = torch.where(
        cb <= 0.0,
        torch.zeros_like(cb),
        torch.where(cs >= 1.0, torch.ones_like(cb),
                    torch.minimum(torch.ones_like(cb), cb / (1.0 - cs).clamp(min=EPS))),
    )
    return out


def _color_burn(cb, cs):
    out = torch.where(
        cb >= 1.0,
        torch.ones_like(cb),
        torch.where(cs <= 0.0, torch.zeros_like(cb),
                    1.0 - torch.minimum(torch.ones_like(cb),
                                        (1.0 - cb) / cs.clamp(min=EPS))),
    )
    return out


def _hard_light(cb, cs):
    return torch.where(cs <= 0.5,
                       _multiply(cb, 2.0 * cs),
                       _screen(cb, 2.0 * cs - 1.0))


def _soft_light(cb, cs):
    d = torch.where(cb <= 0.25,
                    ((16.0 * cb - 12.0) * cb + 4.0) * cb,
                    torch.sqrt(cb.clamp(min=0.0)))
    return torch.where(cs <= 0.5,
                       cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
                       cb + (2.0 * cs - 1.0) * (d - cb))


def _difference(cb, cs):
    return (cb - cs).abs()


def _exclusion(cb, cs):
    return cb + cs - 2.0 * cb * cs


def _plus(cb, cs):
    return cb + cs


def _minus(cb, cs):
    # Backdrop minus source: "take the top layer away from what is under it".
    return cb - cs


def _divide(cb, cs):
    return cb / torch.where(cs.abs() < EPS, torch.full_like(cs, EPS), cs)


SEPARABLE = {
    "normal": _normal,
    "multiply": _multiply,
    "screen": _screen,
    "overlay": _overlay,
    "darken": _darken,
    "lighten": _lighten,
    "color_dodge": _color_dodge,
    "color_burn": _color_burn,
    "hard_light": _hard_light,
    "soft_light": _soft_light,
    "difference": _difference,
    "exclusion": _exclusion,
    "plus": _plus,
    "minus": _minus,
    "divide": _divide,
}


# ---------------------------------------------------------------------------
# Non-separable modes: f(backdrop, source) on whole RGB pixels
# ---------------------------------------------------------------------------

def _lum(c):
    """Luminosity of an (..., 3) tensor, keepdim so it broadcasts back."""
    return (c[..., 0:1] * LUM_R + c[..., 1:2] * LUM_G + c[..., 2:3] * LUM_B)


def _clip_color(c):
    """Pull a colour back inside [0,1] by desaturating toward its luminosity.

    The naive alternative — clamping each channel — shifts hue as it clips,
    which shows up as a saturated highlight turning the wrong colour. This
    scales the whole colour toward its own luminance instead, so the hue
    survives and only the saturation gives way.
    """
    lum = _lum(c)
    n = c.min(dim=-1, keepdim=True).values
    x = c.max(dim=-1, keepdim=True).values

    # Written as `lum + (c - lum) * k`, with k forced to 0 when the colour has
    # no spread left to give. That shape is not cosmetic — it is what makes this
    # numerically stable.
    #
    # The spec divides by (lum - n) and (x - lum), both of which go to zero as
    # the colour approaches neutral. Guarding those denominators with an epsilon
    # (the obvious defence) turns a spread of 1e-8 into a multiplier of ~1e-2,
    # so two implementations that agree to float precision on the INPUT produce
    # visibly different colours on the OUTPUT. The live preview and the render
    # disagreed by 0.16 on exactly such a pixel.
    #
    # Setting k = 0 instead gives the correct limit rather than a guarded
    # approximation of it: a neutral colour cannot be desaturated any further,
    # so ClipColor leaves it at its own luminosity.
    #
    # Applied SEQUENTIALLY, per the spec: L, n and x are computed once, and when
    # both corrections fire the second operates on the output of the first.
    den_low = lum - n
    k_low = torch.where(den_low > EPS, lum / den_low.clamp(min=EPS),
                        torch.zeros_like(lum))
    c = torch.where(n < 0.0, lum + (c - lum) * k_low, c)

    den_high = x - lum
    k_high = torch.where(den_high > EPS, (1.0 - lum) / den_high.clamp(min=EPS),
                         torch.zeros_like(lum))
    c = torch.where(x > 1.0, lum + (c - lum) * k_high, c)
    return c


def _set_lum(c, lum):
    return _clip_color(c + (lum - _lum(c)))


def _sat(c):
    return (c.max(dim=-1, keepdim=True).values
            - c.min(dim=-1, keepdim=True).values)


def _set_sat(c, s):
    """Rescale a colour's saturation, keeping its mid channel proportional.

    Spec form. A fully desaturated input has nothing to rescale — max == min, so
    the span is zero — and the spec's answer is black, which the guard produces.
    """
    mn = c.min(dim=-1, keepdim=True).values
    mx = c.max(dim=-1, keepdim=True).values
    span = (mx - mn)
    # `span > EPS`, not `> 0`: a span of 1e-8 is a neutral colour, and dividing
    # by it amplifies float noise into a wildly saturated result that differs
    # between any two implementations. Same reasoning as _clip_color.
    scaled = (c - mn) * s / span.clamp(min=EPS)
    return torch.where(span > EPS, scaled, torch.zeros_like(c))


def _ns_hue(cb, cs):
    return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))


def _ns_saturation(cb, cs):
    return _set_lum(_set_sat(cb, _sat(cs)), _lum(cb))


def _ns_color(cb, cs):
    return _set_lum(cs, _lum(cb))


def _ns_luminosity(cb, cs):
    return _set_lum(cb, _lum(cs))


NON_SEPARABLE = {
    "hue": _ns_hue,
    "saturation": _ns_saturation,
    "color": _ns_color,
    "luminosity": _ns_luminosity,
}


# Order is the dropdown order, grouped the way a compositor expects rather than
# alphabetically: pass-through, then darkening, then lightening, then contrast,
# then comparative, then arithmetic, then the colour modes.
MODES = [
    "normal",
    "multiply", "darken", "color_burn", "minus",
    "screen", "lighten", "color_dodge", "plus",
    "overlay", "soft_light", "hard_light",
    "difference", "exclusion", "divide",
    "hue", "saturation", "color", "luminosity",
]

assert set(MODES) == set(SEPARABLE) | set(NON_SEPARABLE), "MODES is out of step"


def is_separable(mode: str) -> bool:
    return mode in SEPARABLE


def blend(cb: torch.Tensor, cs: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a blend mode to (..., 3) backdrop and source tensors.

    Unknown modes fall back to ``normal`` rather than raising: this is reachable
    from a saved workflow and from an HTTP request, and a preview is never worth
    failing a run over.
    """
    fn = SEPARABLE.get(mode)
    if fn is not None:
        return fn(cb, cs)
    fn = NON_SEPARABLE.get(mode)
    if fn is not None:
        return fn(cb, cs)
    return cs
