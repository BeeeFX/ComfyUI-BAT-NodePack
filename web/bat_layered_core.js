/**
 * Bat_LayeredImages — the composite, with no DOM and no ComfyUI.
 *
 * A mirror of `composite()` in `bat_layered_images.py`, kept separate so the
 * editor, the worker and `tests/verify_layered_images.py` all use one
 * implementation.
 *
 * The loop is the W3C compositing form written out literally, so it can be
 * checked against the spec rather than reverse-engineered::
 *
 *     Cb = acc / ab                                    the backdrop as a colour
 *     co = as*(1-ab)*Cs + as*ab*B(Cb,Cs) + (1-as)*acc
 *     ao = as + ab*(1-as)
 *
 * Doing it properly rather than as a chain of lerps is what makes the bottom
 * layer's mask mean something: layer 1 composites over nothing, so it can be
 * genuinely semi-transparent, and the accumulated alpha comes out as a second
 * output. A lerp chain would have baked black into those pixels.
 */

import { SEPARABLE, NON_SEPARABLE } from "./bat_blend_modes.js";

/**
 * Composite a bottom-to-top stack.
 *
 * @param {Array<{data: Float32Array, mask: Float32Array|null}>} layers
 *        RGB interleaved (stride 3) and an optional single-channel mask, all at
 *        w x h. Bottom layer first.
 * @param {Array<{mode: string, opacity: number, enabled: boolean}>} settings
 * @param {number} w
 * @param {number} h
 * @param {boolean} clampOutput
 * @returns {{out: Float32Array, alpha: Float32Array}}
 */
export function compositeStack(layers, settings, w, h, clampOutput) {
    const n = w * h;
    const acc = new Float32Array(n * 3);      // premultiplied colour
    const ab = new Float32Array(n);           // accumulated alpha
    // Scratch for the non-separable modes, reused so the pixel loop never
    // allocates — at 512x288 that would otherwise be 147k throwaway arrays per
    // repaint, which costs more than the blend.
    const tmp = [0, 0, 0];

    for (let li = 0; li < layers.length; li++) {
        const cfg = settings[li] || { mode: "normal", opacity: 1, enabled: true };
        if (cfg.enabled === false || !(cfg.opacity > 0)) continue;
        const src = layers[li].data;
        const mask = layers[li].mask || null;
        const op = cfg.opacity;
        const sep = SEPARABLE[cfg.mode];
        const nonSep = sep ? null : (NON_SEPARABLE[cfg.mode] || null);
        // An unknown mode falls back to `normal`, matching the Python. This is
        // reachable from a saved workflow, so it must not throw.
        const fnSep = sep || (nonSep ? null : SEPARABLE.normal);

        for (let i = 0, p = 0; i < n; i++, p += 3) {
            let as = mask ? mask[i] : 1;
            if (op !== 1) as *= op;
            const abi = ab[i];

            if (as <= 0) continue;             // nothing to add for this pixel

            const inv = 1 - abi;
            const invS = 1 - as;
            if (abi <= 0) {
                // Nothing underneath: the blend term is gated out by `ab`, so
                // this reduces to the source over transparency. Skipping the
                // blend here is not just an optimisation — it avoids evaluating
                // a mode against an undefined backdrop.
                acc[p] = as * src[p] + invS * acc[p];
                acc[p + 1] = as * src[p + 1] + invS * acc[p + 1];
                acc[p + 2] = as * src[p + 2] + invS * acc[p + 2];
            } else {
                const k = 1 / abi;
                const cbR = acc[p] * k, cbG = acc[p + 1] * k, cbB = acc[p + 2] * k;
                const sR = src[p], sG = src[p + 1], sB = src[p + 2];
                let bR, bG, bB;
                if (fnSep) {
                    bR = fnSep(cbR, sR); bG = fnSep(cbG, sG); bB = fnSep(cbB, sB);
                } else {
                    nonSep(tmp, cbR, cbG, cbB, sR, sG, sB);
                    bR = tmp[0]; bG = tmp[1]; bB = tmp[2];
                }
                acc[p] = as * inv * sR + as * abi * bR + invS * acc[p];
                acc[p + 1] = as * inv * sG + as * abi * bG + invS * acc[p + 1];
                acc[p + 2] = as * inv * sB + as * abi * bB + invS * acc[p + 2];
            }
            ab[i] = as + abi * invS;
        }
    }

    const out = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 3) {
        const a = ab[i];
        if (a > 0) {
            const k = 1 / a;
            let r = acc[p] * k, g = acc[p + 1] * k, b = acc[p + 2] * k;
            if (clampOutput) {
                r = r < 0 ? 0 : (r > 1 ? 1 : r);
                g = g < 0 ? 0 : (g > 1 ? 1 : g);
                b = b < 0 ? 0 : (b > 1 ? 1 : b);
            }
            out[p] = r; out[p + 1] = g; out[p + 2] = b;
        }
        // else left at 0: un-premultiplying a fully transparent pixel is
        // undefined, so the RGB is zeroed where the alpha says there is nothing.
    }
    return { out, alpha: ab };
}

/**
 * Map a composite to 8-bit RGBA for display.
 *
 * `view` is "result", "layer:<i>" to solo one, or "alpha" to see the
 * accumulated matte. Soloing shows the layer as itself — full opacity, `normal`
 * — rather than as its mode renders it against nothing, which is what someone
 * clicking a layer row is asking to see. Matches `render_region()`'s choice on
 * the Python side.
 */
export function paintLayered({ view, layers, out, alpha, w, h, rgba }) {
    const n = w * h;
    const dst = rgba || new Uint8ClampedArray(n * 4);
    const solo = view.startsWith("layer:") ? parseInt(view.slice(6), 10) : -1;
    const layer = (solo >= 0 && layers[solo]) ? layers[solo] : null;

    for (let i = 0, p = 0, q = 0; i < n; i++, p += 3, q += 4) {
        let r, g, b;
        if (view === "alpha") {
            r = g = b = alpha[i];
        } else if (layer) {
            const a = layer.mask ? layer.mask[i] : 1;
            // Over black, so a masked-out region reads as absent rather than as
            // whatever happened to be in the layer's RGB there.
            r = layer.data[p] * a;
            g = layer.data[p + 1] * a;
            b = layer.data[p + 2] * a;
        } else {
            r = out[p]; g = out[p + 1]; b = out[p + 2];
        }
        dst[q] = r > 1 ? 255 : (r < 0 ? 0 : Math.round(r * 255));
        dst[q + 1] = g > 1 ? 255 : (g < 0 ? 0 : Math.round(g * 255));
        dst[q + 2] = b > 1 ? 255 : (b < 0 ? 0 : Math.round(b * 255));
        dst[q + 3] = 255;
    }
    return dst;
}
