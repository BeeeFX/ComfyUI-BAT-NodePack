/**
 * Bat_AdvancedBlend — the blend itself, with no DOM and no ComfyUI.
 *
 * Split out of `bat_advanced_blend.js` so that three consumers can share one
 * implementation:
 *
 *   • the editor, on the main thread, as a fallback;
 *   • `bat_blend_worker.js`, which is where it normally runs — a full-tile blend
 *     at a wide radius is hundreds of milliseconds of tight typed-array loop,
 *     and on the main thread that freezes the node, the graph and the slider
 *     being dragged;
 *   • `tests/verify_advanced_blend.py`, which diffs it against the Python. That
 *     used to work by slicing the top off the editor file, which was fragile.
 *
 * Everything here is a mirror of `bat_advanced_blend.py`. Change one side and
 * the test fails, which is the point.
 */

const EPS = 1e-6;

// ── blend modes — mirrors of _blend() in bat_advanced_blend.py ───────────
// Resolved to a function once per repaint rather than switched on inside the
// pixel loop: a string switch per channel per pixel is the difference between
// a preview that keeps up with a drag and one that doesn't.
export const BLEND_FNS = {
    over:       (base, top) => top,
    add:        (base, top) => base + top,
    multiply:   (base, top) => base * top,
    screen:     (base, top) => 1 - (1 - base) * (1 - top),
    overlay:    (base, top) => (base <= 0.5 ? 2 * base * top
                                            : 1 - 2 * (1 - base) * (1 - top)),
    soft_light: (base, top) => {
        // W3C / CSS compositing definition, which is also Photoshop's.
        const d = base <= 0.25
            ? ((16 * base - 12) * base + 4) * base
            : Math.sqrt(Math.max(base, 0));
        return top <= 0.5
            ? base - (1 - 2 * top) * base * (1 - base)
            : base + (2 * top - 1) * (d - base);
    },
    difference: (base, top) => Math.abs(base - top),
    min:        (base, top) => Math.min(base, top),
    max:        (base, top) => Math.max(base, top),
};

/**
 * Separable Gaussian on an interleaved RGB Float32 buffer (stride 3),
 * reflect-padded. Mirrors _blur() / _gauss_kernel() in the Python, including
 * the renormalisation after truncating the kernel to fit a narrow tile —
 * without it the edge of the lowpass darkens and the detail band picks up a
 * bright border.
 *
 * `radius` is in TILE pixels; the caller has already scaled it down from the
 * widget's working-resolution value.
 */
export function blurRGB(src, w, h, radius) {
    if (!(radius > 0)) return src;
    const r0 = Math.round(radius);
    if (r0 <= 0) return src;
    const sigma = Math.max(radius / 2, 0.5);

    const build = (r) => {
        const k = new Float32Array(2 * r + 1);
        let s = 0;
        for (let i = -r; i <= r; i++) {
            const v = Math.exp(-(i * i) / (2 * sigma * sigma));
            k[i + r] = v; s += v;
        }
        for (let i = 0; i < k.length; i++) k[i] /= s;
        return k;
    };
    const refl = (i, n) => (i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i));

    let cur = src;
    const rx = Math.min(r0, Math.max(w - 1, 0));
    if (rx > 0) {
        const k = build(rx);
        const tmp = new Float32Array(w * h * 3);
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                let ar = 0, ag = 0, ab = 0;
                for (let i = -rx; i <= rx; i++) {
                    const p = (row + refl(x + i, w)) * 3;
                    const kw = k[i + rx];
                    ar += cur[p] * kw; ag += cur[p + 1] * kw; ab += cur[p + 2] * kw;
                }
                const q = (row + x) * 3;
                tmp[q] = ar; tmp[q + 1] = ag; tmp[q + 2] = ab;
            }
        }
        cur = tmp;
    }
    const ry = Math.min(r0, Math.max(h - 1, 0));
    if (ry > 0) {
        const k = build(ry);
        const tmp = new Float32Array(w * h * 3);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let ar = 0, ag = 0, ab = 0;
                for (let i = -ry; i <= ry; i++) {
                    const p = (refl(y + i, h) * w + x) * 3;
                    const kw = k[i + ry];
                    ar += cur[p] * kw; ag += cur[p + 1] * kw; ab += cur[p + 2] * kw;
                }
                const q = (y * w + x) * 3;
                tmp[q] = ar; tmp[q + 1] = ag; tmp[q + 2] = ab;
            }
        }
        cur = tmp;
    }
    return cur;
}

/**
 * Memoised blur. The key includes the identity of the source buffer, so a
 * cached lowpass is dropped the moment the thing it was taken from changes
 * (a new run, or a different soften radius upstream of it).
 */
export function makeBlurCache() {
    const slots = new Map();
    return {
        get(name, src, w, h, radius) {
            const hit = slots.get(name);
            if (hit && hit.src === src && hit.radius === radius) return hit.out;
            const out = blurRGB(src, w, h, radius);
            slots.set(name, { src, radius, out });
            return out;
        },
        clear() { slots.clear(); },
    };
}

/**
 * The algorithm itself — a pure mirror of `BatAdvancedBlend._core` in
 * bat_advanced_blend.py, with no DOM and no widget access.
 *
 * Kept at module scope rather than inside the editor closure so it can be run
 * headlessly against the Python and proved equivalent (see
 * `tests/verify_advanced_blend.py`); a preview that silently drifts from the
 * render is worse than no preview, because the artist trusts it.
 *
 * `a` / `b` are RGB-interleaved Float32 (stride 3) of identical length —
 * Python conforms both plates before it samples either, so that holds by
 * construction. Radii in `p` are already in TILE pixels.
 *
 * @returns {{out: Float32Array, high: Float32Array}} the blended result and
 *   the high band that was applied (neutral everywhere if separation is off).
 */
export function blendTile(a, b, w, h, mask, p, blurs) {
    const n = w * h;
    const blendFn = BLEND_FNS[p.blend_mode] || BLEND_FNS.over;
    const divide = p.detail_mode === "divide";
    const neutral = divide ? 1 : 0;
    const cache = blurs || makeBlurCache();

    // A limit below 1 would invert the ratio bounds (min > max) and flatten the
    // band instead of bounding it, so it is floored exactly as the Python
    // floors it.
    const limitOn = p.detail_limit > 0;
    const limHi = divide ? Math.max(p.detail_limit, 1 + EPS) : p.detail_limit;
    const limLo = divide ? 1 / limHi : -p.detail_limit;

    const softA = cache.get("softA", a, w, h, p.soften_a);
    const softB = cache.get("softB", b, w, h, p.soften_b);
    const lowA = p.frequency_separation ? cache.get("lowA", softA, w, h, p.split_radius) : null;
    const lowB = p.frequency_separation ? cache.get("lowB", softB, w, h, p.split_radius) : null;

    const out = new Float32Array(n * 3);
    const high = new Float32Array(n * 3);

    for (let i = 0, q = 0; i < n; i++, q += 3) {
        const gate = mask ? mask[i] * p.mix : p.mix;
        for (let c = 0; c < 3; c++) {
            const av = softA[q + c];
            const bv = softB[q + c];
            let result, hv;

            if (!p.frequency_separation) {
                result = blendFn(bv, av);
                hv = neutral;
            } else {
                const la = lowA[q + c];
                const lb = lowB[q + c];
                const ha = divide ? av / Math.max(la, EPS) : av - la;
                const hb = divide ? bv / Math.max(lb, EPS) : bv - lb;

                // Low band: the blend mode, dialled back toward B's own low.
                const lowBlend = blendFn(lb, la);
                const low = lb + (lowBlend - lb) * p.low_mix;

                // High band: always a lerp. A signed detail band has no
                // meaningful black or white for a blend mode to key off.
                hv = hb + (ha - hb) * p.high_mix;
                if (p.detail_gain !== 1) hv = neutral + (hv - neutral) * p.detail_gain;
                if (limitOn) hv = hv < limLo ? limLo : (hv > limHi ? limHi : hv);

                result = divide ? low * hv : low + hv;
            }

            let v = bv + (result - bv) * gate;
            if (p.clamp_output) v = v < 0 ? 0 : (v > 1 ? 1 : v);
            out[q + c] = v;
            high[q + c] = hv;
        }
    }
    return { out, high };
}




/**
 * Map a computed blend to 8-bit RGBA for display.
 *
 * Lives here, next to the blend, because the server's `render_region()` makes
 * exactly the same choice on the Python side — if the two drift, the draft and
 * the full-resolution layer show different pictures of the same settings, which
 * is worse than either being wrong on its own.
 *
 * `a` / `b` are the RAW inputs, deliberately: the A and B views answer "what did
 * the node receive", and a soften radius is already visible in Result and Detail.
 */
export function paintView({ view, a, b, out, high, neutral, amp, w, h, rgba }) {
    const n = w * h;
    const dst = rgba || new Uint8ClampedArray(n * 4);
    // Amplification applies only to the two inspection views, which sit on 0.5
    // grey and routinely carry differences of a few code values.
    const pivot = (view === "detail" || view === "diff");
    const g = pivot ? (amp || 1) : 1;
    for (let i = 0, q = 0, r = 0; i < n; i++, q += 3, r += 4) {
        for (let c = 0; c < 3; c++) {
            let show;
            if (view === "a") show = a[q + c];
            else if (view === "b") show = b[q + c];
            else if (view === "detail") show = (high[q + c] - neutral) + 0.5;
            else if (view === "diff") show = (a[q + c] - b[q + c]) + 0.5;
            else show = out[q + c];
            if (g !== 1) show = 0.5 + (show - 0.5) * g;
            dst[r + c] = show > 1 ? 255 : (show < 0 ? 0 : Math.round(show * 255));
        }
        dst[r + 3] = 255;
    }
    return dst;
}

/** The high-band value that leaves the low band untouched. Mirrors _neutral_high. */
export function neutralHigh(detailMode) {
    return detailMode === "divide" ? 1 : 0;
}
