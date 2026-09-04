/**
 * BAT — display transfer functions, shared by the frontend previews.
 *
 * Exact mirrors of the ones in `bat_hdr_tonal_composite.py`
 * (`_srgb_to_linear` / `_linear_to_srgb` / `_rec709_to_linear` /
 * `_linear_to_rec709` / `_to_linear` / `_encode_from_linear`), which is where
 * the authoritative versions live. Any preview that shows an artist what an
 * exposure or a grade will look like has to run the same curve the render will,
 * or it is lying about the one thing it exists to show.
 *
 * Note on scope: `bat_hdr_tonal_composite.js` carries its own private copies of
 * these, predating this module. They agree today. Folding it onto this module is
 * the obvious cleanup but it is a change to a shipped node with a large preview
 * of its own, so it is deliberately left alone rather than refactored as a side
 * effect of another feature — if you are here to change a curve, change it in
 * three places (the Python, this file, that file) and run
 * `tests/verify_exposure_bracket.py`.
 *
 * Two asymmetries that look like bugs and are not
 * ----------------------------------------------
 * • `toLinear(x, "linear")` is the identity, but `encodeFromLinear(x, "linear")`
 *   applies the sRGB curve. Linear has no display encoding of its own to offer,
 *   so it borrows sRGB — the Python documents this and does the same, and any
 *   consumer that needs a genuine round-trip must not use "linear" for both
 *   directions.
 * • `linearToSrgb` and `linearToRec709` are NOT clamped to 1 on the way out.
 *   Callers wanting display range clamp themselves; the key domains in the
 *   composite want the unclamped curve so overbright values still sort above
 *   their thresholds.
 */

export function srgbToLinear(x) {
    return x <= 0.04045 ? x / 12.92 : Math.pow((Math.max(x, 0.04045) + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(x) {
    return x <= 0.0031308 ? 12.92 * x
                          : 1.055 * Math.pow(Math.max(x, 0.0031308), 1 / 2.4) - 0.055;
}

/**
 * Inverse BT.709 OETF — what Nuke's "rec709" colorspace applies. Not
 * interchangeable with sRGB: the curves differ by up to 65% in the shadows
 * (code 0.05 decodes to 0.0039 through sRGB and 0.0111 through BT.709).
 */
export function rec709ToLinear(x) {
    return x < 0.081 ? x / 4.5 : Math.pow((Math.max(x, 0.081) + 0.099) / 1.099, 1 / 0.45);
}

export function linearToRec709(x) {
    return x < 0.018 ? 4.5 * x : 1.099 * Math.pow(Math.max(x, 0.018), 0.45) - 0.099;
}

/** Decode a display-referred value to scene-linear. */
export function toLinear(x, mode) {
    const v = x < 0 ? 0 : x;
    if (mode === "linear") return v;
    if (mode === "rec709") return rec709ToLinear(v);
    if (mode === "gamma_2_2") return Math.pow(v, 2.2);
    if (mode === "gamma_2_4") return Math.pow(v, 2.4);
    return srgbToLinear(v);
}

/** Scene-linear back to a display encoding. "linear" borrows sRGB — see above. */
export function encodeFromLinear(x, mode) {
    const v = x < 0 ? 0 : x;
    if (mode === "rec709") return linearToRec709(v);
    if (mode === "gamma_2_2") return Math.pow(v, 1 / 2.2);
    if (mode === "gamma_2_4") return Math.pow(v, 1 / 2.4);
    return linearToSrgb(v);
}

/**
 * One exposure of a scene-linear value, display-encoded and clamped — the
 * mirror of `_expose_to_sdr` in bat_exposure_bracket.py.
 *
 * The clamp before the encode is the operation, not housekeeping: it is what
 * turns "expose down" into "bring what was above white into the visible
 * range". `scale` is 2**ev, hoisted by the caller so it isn't recomputed per
 * pixel.
 */
export function exposeToSdr(lin, scale, mode) {
    let v = lin * scale;
    v = v < 0 ? 0 : (v > 1 ? 1 : v);
    const e = encodeFromLinear(v, mode);
    return e < 0 ? 0 : (e > 1 ? 1 : e);
}
