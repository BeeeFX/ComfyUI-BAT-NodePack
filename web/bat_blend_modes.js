/**
 * Blend modes — exact mirror of `bat_blend_modes.py`.
 *
 * The formulas are the W3C Compositing and Blending spec's, which is what
 * Photoshop implements and what Nuke calls its Photoshop set. Read the Python
 * file for the reasoning; the notes worth repeating on this side are:
 *
 * • The non-separable four (hue/saturation/color/luminosity) use luma weights
 *   0.3/0.59/0.11 — the spec's, and Photoshop's. NOT Rec.709, which the pack's
 *   HDR nodes correctly use for a different job.
 * • Nothing clamps. Every mode is defined against white = 1.0, so above white
 *   they still compute but stop meaning what their name says. `normal`, `plus`,
 *   `minus`, `darken` and `lighten` are the range-agnostic ones.
 * • Separable modes take scalars, non-separable ones take and return RGB
 *   triples. Consumers branch on `isSeparable()`; a uniform signature would only
 *   be able to fake it.
 *
 * `tests/verify_layered_images.py` diffs every mode in here against the Python
 * over the full [0,1] grid plus out-of-range values, which is the only reason
 * two hand-written copies of nineteen formulas is a defensible arrangement.
 */

/** Guard for the divisions in colour_dodge / colour_burn / divide / clipColor.
 *  Exported so consumers share one value rather than each picking their own. */
export const EPS = 1e-6;

export const LUM_R = 0.3, LUM_G = 0.59, LUM_B = 0.11;

// ── separable: f(backdrop, source) per channel ───────────────────────────
const multiply = (cb, cs) => cb * cs;
const screen = (cb, cs) => cb + cs - cb * cs;

function hardLight(cb, cs) {
    return cs <= 0.5 ? multiply(cb, 2 * cs) : screen(cb, 2 * cs - 1);
}

export const SEPARABLE = {
    // No alpha here: the caller composites with the layer's mask/opacity, so
    // "normal" is simply the source and the mix falls out of that.
    normal: (cb, cs) => cs,
    multiply,
    screen,
    // Hard-light with the operands swapped, which is the spec's own definition.
    overlay: (cb, cs) => hardLight(cs, cb),
    darken: (cb, cs) => Math.min(cb, cs),
    lighten: (cb, cs) => Math.max(cb, cs),
    color_dodge: (cb, cs) => {
        // 0 where the backdrop is 0, so black stays black instead of being
        // lifted by the division; 1 where the source is 1.
        if (cb <= 0) return 0;
        if (cs >= 1) return 1;
        return Math.min(1, cb / Math.max(1 - cs, EPS));
    },
    color_burn: (cb, cs) => {
        if (cb >= 1) return 1;
        if (cs <= 0) return 0;
        return 1 - Math.min(1, (1 - cb) / Math.max(cs, EPS));
    },
    hard_light: hardLight,
    soft_light: (cb, cs) => {
        const d = cb <= 0.25
            ? ((16 * cb - 12) * cb + 4) * cb
            : Math.sqrt(Math.max(cb, 0));
        return cs <= 0.5
            ? cb - (1 - 2 * cs) * cb * (1 - cb)
            : cb + (2 * cs - 1) * (d - cb);
    },
    difference: (cb, cs) => Math.abs(cb - cs),
    exclusion: (cb, cs) => cb + cs - 2 * cb * cs,
    plus: (cb, cs) => cb + cs,
    // Backdrop minus source: "take the top layer away from what is under it".
    minus: (cb, cs) => cb - cs,
    divide: (cb, cs) => cb / (Math.abs(cs) < EPS ? EPS : cs),
};

// ── non-separable: whole-pixel colour operations ─────────────────────────
// These work on scratch triples the caller owns, so the per-pixel loop can run
// without allocating.

function lum3(r, g, b) { return r * LUM_R + g * LUM_G + b * LUM_B; }

/**
 * Pull a colour back inside [0,1] by desaturating toward its own luminosity.
 *
 * Clamping each channel instead would shift hue as it clips — a saturated
 * highlight turning the wrong colour. This gives up saturation and keeps hue.
 */
function clipColor(out) {
    let [r, g, b] = out;
    const l = lum3(r, g, b);
    const n = Math.min(r, Math.min(g, b));
    const x = Math.max(r, Math.max(g, b));
    // `l + (c - l) * k`, with k forced to 0 when the colour has no spread left
    // to give. Not cosmetic — it is what makes this numerically stable.
    //
    // The spec divides by (l - n) and (x - l), both of which go to zero as the
    // colour approaches neutral. Guarding those denominators with an epsilon —
    // the obvious defence, and what this used to do — turns a spread of 1e-8
    // into a multiplier of ~1e-2, so the preview and the render disagreed by
    // 0.16 on a near-neutral pixel taken above white. k = 0 gives the correct
    // limit instead: a neutral colour cannot desaturate further, so it stays at
    // its own luminosity.
    //
    // Sequential, per the spec: l/n/x computed once, second correction applied
    // to the output of the first.
    if (n < 0) {
        const d = l - n;
        const k = d > EPS ? l / d : 0;
        r = l + (r - l) * k;
        g = l + (g - l) * k;
        b = l + (b - l) * k;
    }
    if (x > 1) {
        const d = x - l;
        const k = d > EPS ? (1 - l) / d : 0;
        r = l + (r - l) * k;
        g = l + (g - l) * k;
        b = l + (b - l) * k;
    }
    out[0] = r; out[1] = g; out[2] = b;
    return out;
}

function setLum(out, r, g, b, l) {
    const d = l - lum3(r, g, b);
    out[0] = r + d; out[1] = g + d; out[2] = b + d;
    return clipColor(out);
}

function satOf(r, g, b) {
    return Math.max(r, Math.max(g, b)) - Math.min(r, Math.min(g, b));
}

/** Rescale saturation, keeping the mid channel proportional. A fully
 *  desaturated input has a zero span and the spec's answer is black. */
function setSat(out, r, g, b, s) {
    const mn = Math.min(r, Math.min(g, b));
    const mx = Math.max(r, Math.max(g, b));
    const span = mx - mn;
    // `> EPS`, not `> 0`: dividing by a 1e-8 span amplifies float noise into a
    // wildly saturated result that no two implementations would agree on. Same
    // reasoning as clipColor.
    if (span > EPS) {
        out[0] = (r - mn) * s / span;
        out[1] = (g - mn) * s / span;
        out[2] = (b - mn) * s / span;
    } else {
        out[0] = 0; out[1] = 0; out[2] = 0;
    }
    return out;
}

export const NON_SEPARABLE = {
    hue: (out, br, bg, bb, sr, sg, sb) => {
        setSat(out, sr, sg, sb, satOf(br, bg, bb));
        return setLum(out, out[0], out[1], out[2], lum3(br, bg, bb));
    },
    saturation: (out, br, bg, bb, sr, sg, sb) => {
        setSat(out, br, bg, bb, satOf(sr, sg, sb));
        return setLum(out, out[0], out[1], out[2], lum3(br, bg, bb));
    },
    color: (out, br, bg, bb, sr, sg, sb) =>
        setLum(out, sr, sg, sb, lum3(br, bg, bb)),
    luminosity: (out, br, bg, bb, sr, sg, sb) =>
        setLum(out, br, bg, bb, lum3(sr, sg, sb)),
};

/** Dropdown order, grouped the way a compositor expects. Mirrors MODES in the .py. */
export const MODES = [
    "normal",
    "multiply", "darken", "color_burn", "minus",
    "screen", "lighten", "color_dodge", "plus",
    "overlay", "soft_light", "hard_light",
    "difference", "exclusion", "divide",
    "hue", "saturation", "color", "luminosity",
];

/** Labels for the UI. Kept apart from the wire names so a rename never
 *  invalidates a saved workflow. */
export const MODE_LABELS = {
    normal: "Normal", multiply: "Multiply", darken: "Darken",
    color_burn: "Colour Burn", minus: "Minus", screen: "Screen",
    lighten: "Lighten", color_dodge: "Colour Dodge", plus: "Plus",
    overlay: "Overlay", soft_light: "Soft Light", hard_light: "Hard Light",
    difference: "Difference", exclusion: "Exclusion", divide: "Divide",
    hue: "Hue", saturation: "Saturation", color: "Colour",
    luminosity: "Luminosity",
};

export function isSeparable(mode) {
    return Object.prototype.hasOwnProperty.call(SEPARABLE, mode);
}
