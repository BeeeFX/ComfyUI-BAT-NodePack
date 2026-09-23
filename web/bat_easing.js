/**
 * BAT — keyframe easing shared by the animated editors' live previews.
 *
 * Mirror of bat_easing.py: a keyframe's `ease` names the curve used on the way
 * OUT of it, towards the next key; a key without one is linear, so existing
 * workflows interpolate exactly as before. The two tables must stay identical
 * or the preview and the render disagree mid-segment (tests/verify_easing.py).
 */

export const EASES = ["linear", "ease_in", "ease_out", "ease_in_out", "hold"];
export const DEFAULT_EASE = "linear";

/** Labels for the editors' ease pickers, in EASES order. */
export const EASE_LABELS = {
    linear: "Linear",
    ease_in: "Ease in",
    ease_out: "Ease out",
    ease_in_out: "Ease in-out",
    hold: "Hold",
};

/** The ease of a keyframe object (or the default for anything else). */
export function easeName(key) {
    const name = key && typeof key === "object" ? key.ease : undefined;
    return EASES.includes(name) ? name : DEFAULT_EASE;
}

/** Map a segment fraction t in [0, 1] through the named curve. */
export function applyEase(t, name) {
    t = t <= 0 ? 0 : t >= 1 ? 1 : t;
    switch (name) {
        case "ease_in":     return t * t;
        case "ease_out":    return 1 - (1 - t) * (1 - t);
        case "ease_in_out": return t * t * (3 - 2 * t);
        case "hold":        return 0;
        default:            return t;
    }
}
