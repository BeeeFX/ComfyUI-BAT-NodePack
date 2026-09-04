/**
 * Bat_AdvancedBlend — the draft blend, off the main thread.
 *
 * Why this exists
 * ---------------
 * The preview's whole job is to show what a change does to high-frequency
 * detail, and the earlier answer to "a full-tile blend is too slow to track a
 * slider" was to blend a downscaled copy. That is the one degradation this node
 * cannot afford: downscaling destroys exactly the signal being judged, so a
 * smooth quarter-resolution preview of a sharpening change shows nothing useful.
 *
 * The correct degradation for a detail tool is to drop FRAMES, not PIXELS — a
 * full-resolution picture that arrives a moment late is useful, a fast blurry one
 * is not. Dropping frames on the main thread, though, means freezing the graph
 * and the slider being dragged, because a 768x432 blend at a wide radius is
 * hundreds of milliseconds of uninterruptible typed-array loop.
 *
 * Hence a worker. The tile blends at full resolution, however long it takes, and
 * the UI stays responsive throughout.
 *
 * Protocol
 * --------
 *   → {type: "tiles", a, b, mask, w, h}   once per execution; copies retained here
 *   → {type: "render", id, p, view, amp, wantValues}
 *   ← {type: "rendered", id, rgba, w, h, ms, values?}
 *
 * `rgba` is transferred rather than copied, so the per-frame cost back to the
 * main thread is a pointer rather than 1.3MB. The float `out`/`high` buffers are
 * only returned when `wantValues` is set — the pixel probe needs them, but it is
 * a hover interaction, and shipping 8MB per frame to service it would cost more
 * than the blend.
 *
 * Only the newest render matters: the editor coalesces, and this side answers
 * every request it is given, so a request already superseded is simply ignored
 * when it arrives back.
 */

import { blendTile, makeBlurCache, paintView, neutralHigh } from "./bat_blend_core.js";

/** Persistent across renders, so a drag that does not change a radius reuses the
 *  Gaussians — which is what makes most of the sliders cheap at full resolution. */
const blurs = makeBlurCache();

let A = null, B = null, mask = null, W = 0, H = 0;
let rgbaPool = null;

self.onmessage = (e) => {
    const msg = e.data;
    if (!msg) return;

    if (msg.type === "tiles") {
        A = msg.a ? new Float32Array(msg.a) : null;
        B = msg.b ? new Float32Array(msg.b) : null;
        mask = msg.mask ? new Float32Array(msg.mask) : null;
        W = msg.w | 0; H = msg.h | 0;
        blurs.clear();
        rgbaPool = null;
        return;
    }

    if (msg.type !== "render") return;
    if (!A || !B || !W || !H) return;

    const t0 = performance.now();
    let rgba, values;
    try {
        const { out, high } = blendTile(A, B, W, H, mask, msg.p, blurs);
        // Reused between frames: allocating 1.3MB per repaint is pure garbage
        // pressure during a drag. Dropped whenever it has been transferred away.
        if (!rgbaPool || rgbaPool.length !== W * H * 4) {
            rgbaPool = new Uint8ClampedArray(W * H * 4);
        }
        rgba = paintView({
            view: msg.view, a: A, b: B, out, high,
            neutral: neutralHigh(msg.p.detail_mode),
            amp: msg.amp, w: W, h: H, rgba: rgbaPool,
        });
        if (msg.wantValues) values = { out, high };
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
        payload.high = values.high;
        transfer.push(values.out.buffer, values.high.buffer);
    }
    // Transferred, so this copy is gone from the worker — the pool has to be
    // rebuilt next frame.
    rgbaPool = null;
    self.postMessage(payload, transfer);
};
