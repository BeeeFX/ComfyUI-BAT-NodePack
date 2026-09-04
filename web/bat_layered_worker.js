/**
 * Bat_LayeredImages — the draft composite, off the main thread.
 *
 * Same arrangement and the same reasoning as `bat_blend_worker.js`: the tile
 * composites at full resolution however long that takes, and the UI stays
 * responsive, because dropping frames is an acceptable degradation and dropping
 * pixels is not.
 *
 * It matters more here than it does for the two-plate node. Cost scales with the
 * depth of the stack, and the four non-separable colour modes are several times
 * dearer per pixel than an arithmetic one — a 512x288 tile of eight `saturation`
 * layers is a lot of min/max/divide to run between two frames of a slider drag.
 *
 * Protocol
 * --------
 *   → {type: "layers", layers: [{data, mask}], w, h}   once per execution
 *   → {type: "render", id, settings, view, clampOutput, wantValues}
 *   ← {type: "rendered", id, rgba, w, h, ms, out?, alpha?}
 *
 * `rgba` is transferred, so the per-frame cost back to the main thread is a
 * pointer rather than the buffer. The float `out`/`alpha` come back only when
 * asked for, which is on a settled frame, because the pixel probe is a hover
 * interaction and shipping them every frame would cost more than the composite.
 */

import { compositeStack, paintLayered } from "./bat_layered_core.js";

let LAYERS = null, W = 0, H = 0;
let rgbaPool = null;

self.onmessage = (e) => {
    const msg = e.data;
    if (!msg) return;

    if (msg.type === "layers") {
        LAYERS = (msg.layers || []).map((l) => ({
            data: new Float32Array(l.data),
            mask: l.mask ? new Float32Array(l.mask) : null,
        }));
        W = msg.w | 0; H = msg.h | 0;
        rgbaPool = null;
        return;
    }

    if (msg.type !== "render") return;
    if (!LAYERS || !LAYERS.length || !W || !H) return;

    const t0 = performance.now();
    let rgba, values;
    try {
        const { out, alpha } = compositeStack(
            LAYERS, msg.settings || [], W, H, !!msg.clampOutput);
        // Reused between frames: allocating this per repaint is pure garbage
        // pressure during a drag. Dropped whenever it has been transferred.
        if (!rgbaPool || rgbaPool.length !== W * H * 4) {
            rgbaPool = new Uint8ClampedArray(W * H * 4);
        }
        rgba = paintLayered({
            view: msg.view || "result", layers: LAYERS,
            out, alpha, w: W, h: H, rgba: rgbaPool,
        });
        if (msg.wantValues) values = { out, alpha };
    } catch (err) {
        self.postMessage({ type: "error", id: msg.id, message: String(err) });
        return;
    }

    const payload = {
        type: "rendered", id: msg.id, rgba, w: W, h: H,
        ms: performance.now() - t0,
    };
    const transfer = [rgba.buffer];
    if (values) {
        payload.out = values.out;
        payload.alpha = values.alpha;
        transfer.push(values.out.buffer, values.alpha.buffer);
    }
    rgbaPool = null;      // transferred away; rebuilt next frame
    self.postMessage(payload, transfer);
};
