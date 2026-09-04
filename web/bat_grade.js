/**
 * Bat_Grade — live canvas preview of the grade settings.
 *
 * Workflow:
 *   1. User runs the node once → backend pushes a small JPEG of the
 *      ungraded input (and optional mask) to the canvas.
 *   2. As the artist drags the blackpoint/whitepoint/lift/gain/multiply/
 *      offset/gamma sliders, the JS applies the same Nuke-style formula
 *      to that thumbnail pixel-by-pixel and redraws.
 *   3. The actual graded full-resolution output is computed by the
 *      Python on the NEXT run — the preview is a fast approximation,
 *      not the bit-exact result.
 *
 * Why pixel-by-pixel JS instead of WebGL? A 384px thumbnail is ~150 k
 * pixels; the formula is ~10 multiply-adds per channel; modern V8
 * crunches that in well under 16 ms which is plenty for slider drags.
 * WebGL would be faster but pulls in a lot of boilerplate for no
 * perceptible win at this size.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import {
    hdrSupported, decodeHdrTile, imageDataToSource, buildInspectBar,
} from "./bat_hdr_preview.js";

const NODE_TYPE = "Bat_Grade";

// localStorage preview cache, keyed by node id. Mirrors the pattern in
// bat_roto.js / bat_animated_crop.js — gives the live canvas something
// to show on workflow reopen without bloating the workflow JSON.
function _previewCacheKey(node) {
    return `bat_grade_preview_${node?.id ?? "_"}`;
}
function _saveCachedPreview(node, data) {
    try { localStorage.setItem(_previewCacheKey(node), JSON.stringify(data)); }
    catch (_) {}
}
function _loadCachedPreview(node) {
    try {
        const raw = localStorage.getItem(_previewCacheKey(node));
        return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
}

function buildPreview(node) {
    const root = document.createElement("div");
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px;
        overflow:hidden;
    `;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:auto; display:block; background:#000;";
    root.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `
        position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;
    `;
    hint.textContent = "Run once to populate the preview.";
    root.appendChild(hint);

    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    // Cached sources so we don't re-decode on every slider drag. Both are the
    // SourceBuffer shape from bat_hdr_preview.js — RGB interleaved Float32,
    // stride 3 — so the pixel loop below never branches on which one is live.
    const state = {
        sourceBitmap: null,    // <img> of the ungraded input (kept for debugging)
        source8:      null,    // SourceBuffer decoded from the 8-bit JPEG
        sourceHdr:    null,    // SourceBuffer decoded from the 16-bit tile, or null
        maskImg:      null,    // <img> of the input mask, or null
        maskCache:    new Map(),   // "WxH" -> Float32Array, rasterised on demand
    };
    node._batGradeState = state;

    // The high-precision tile and the JPEG are different sizes (256px vs 384px
    // long edge), so the mask has to be rasterised per source size rather than
    // once at ingest. Cached because a slider drag repaints every frame.
    function maskFor(w, h) {
        if (!state.maskImg) return null;
        const key = `${w}x${h}`;
        const hit = state.maskCache.get(key);
        if (hit) return hit;
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        const mctx = c.getContext("2d", { willReadFrequently: true });
        // Stretch to match, same as the backend does before applying.
        mctx.drawImage(state.maskImg, 0, 0, w, h);
        const px = mctx.getImageData(0, 0, w, h).data;
        const out = new Float32Array(w * h);
        for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = px[p] / 255;
        state.maskCache.set(key, out);
        return out;
    }

    const inspectBar = buildInspectBar({
        node,
        storageKey: `bat_grade_view_${node?.id ?? "_"}`,
        hasHdr: () => !!state.sourceHdr,
        onChange: () => schedule(),
    });

    function activeSource() {
        return (inspectBar.inspect && state.sourceHdr) ? state.sourceHdr : state.source8;
    }

    // ── widget access ────────────────────────────────────────────────
    const W = (n) => node.widgets.find(w => w.name === n);
    const get = (n) => {
        const w = W(n);
        if (!w) return undefined;
        // BOOLEAN widgets store the value as bool; numeric ones as number.
        return w.value;
    };

    // ── the grade formula — must mirror bat_grade.py's _apply_grade ──
    function applyGrade() {
        const source = activeSource();
        if (!source) return;

        const bp = +get("blackpoint");
        const wp = +get("whitepoint");
        const lift = +get("lift");
        const gain = +get("gain");
        const mult = +get("multiply");
        const off  = +get("offset");
        const gamma = Math.max(+get("gamma"), 0.01);
        const clampW = !!get("clamp_white");
        const clampB = !!get("clamp_black");
        const wpMbp = Math.max(wp - bp, 1e-6);
        const invG  = 1 / gamma;

        // Float32, RGB interleaved (stride 3) whether it came from the JPEG or
        // the 16-bit tile. Grading in float and quantising only at the write
        // below is the whole point: an 8-bit intermediate here would collapse
        // a 16-bit source to the same 256 levels as an 8-bit one, which is
        // exactly what the old Uint8ClampedArray path did.
        const src  = source.data;
        const w    = source.width;
        const h    = source.height;
        const n    = w * h;
        const out  = ctx.createImageData(w, h);
        const dst  = out.data;
        const mask = maskFor(w, h);
        // Viewer exposure — applied after the grade, display only, never part
        // of the rendered result.
        const view = inspectBar.gain;

        for (let i = 0, p = 0, q = 0; i < n; i++, p += 3, q += 4) {
            const r0 = src[p], g0 = src[p+1], b0 = src[p+2];

            // Grade math — one channel at a time, vectorised inline.
            let r = (r0 - bp) / wpMbp;
            let g = (g0 - bp) / wpMbp;
            let b = (b0 - bp) / wpMbp;

            r = r * (gain - lift) + lift;
            g = g * (gain - lift) + lift;
            b = b * (gain - lift) + lift;

            r = r * mult + off;
            g = g * mult + off;
            b = b * mult + off;

            // pow() on negatives is NaN; clamp pre-gamma to >= 0.
            r = r < 0 ? 0 : Math.pow(r, invG);
            g = g < 0 ? 0 : Math.pow(g, invG);
            b = b < 0 ? 0 : Math.pow(b, invG);

            if (clampW) { if (r > 1) r = 1; if (g > 1) g = 1; if (b > 1) b = 1; }
            if (clampB) { if (r < 0) r = 0; if (g < 0) g = 0; if (b < 0) b = 0; }

            // Mask gate (lerp between original and graded by mask value).
            if (mask) {
                const m = mask[i];
                const inv = 1 - m;
                r = r * m + r0 * inv;
                g = g * m + g0 * inv;
                b = b * m + b0 * inv;
            }

            if (view !== 1) { r *= view; g *= view; b *= view; }

            dst[q]   = r > 1 ? 255 : (r < 0 ? 0 : Math.round(r * 255));
            dst[q+1] = g > 1 ? 255 : (g < 0 ? 0 : Math.round(g * 255));
            dst[q+2] = b > 1 ? 255 : (b < 0 ? 0 : Math.round(b * 255));
            dst[q+3] = 255;
        }
        // Resize canvas to match preview size (only if the dimensions changed).
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        ctx.putImageData(out, 0, 0);
        hint.style.display = "none";
    }

    // Schedule a redraw on the next animation frame (coalesces multiple
    // widget callbacks that fire in the same tick — e.g. typing in a
    // number field).
    let raf = 0;
    function schedule() {
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; applyGrade(); });
    }

    // Hook each numeric / boolean widget so any change repaints the
    // preview. Done in onNodeCreated below after the widgets exist; we
    // expose the hook function on the node so the extension can call it.
    node._batGradeWatchWidgets = () => {
        for (const name of ["blackpoint", "whitepoint", "lift", "gain",
                            "multiply", "offset", "gamma",
                            "clamp_white", "clamp_black"]) {
            const w = W(name);
            if (!w) continue;
            const orig = w.callback;
            w.callback = function (v) {
                try { orig?.call(this, v); } catch (_) {}
                schedule();
            };
        }
    };

    // ── ingest helpers (called from onExecuted) ───────────────────────
    node._batGradeIngest = async (b64Image, b64Mask, hdrTile) => {
        const img = new Image();
        await new Promise((res, rej) => {
            img.onload = res; img.onerror = rej;
            img.src = `data:image/jpeg;base64,${b64Image}`;
        });
        state.sourceBitmap = img;
        // Stamp into a backing canvas so we can read pixels.
        const back = document.createElement("canvas");
        back.width = img.naturalWidth;
        back.height = img.naturalHeight;
        const bctx = back.getContext("2d", { willReadFrequently: true });
        bctx.drawImage(img, 0, 0);
        state.source8 = imageDataToSource(
            bctx.getImageData(0, 0, back.width, back.height));

        // Mask is kept as an <img> and rasterised per source size on demand,
        // because the JPEG and the high-precision tile aren't the same size.
        state.maskCache.clear();
        if (b64Mask) {
            const mimg = new Image();
            await new Promise((res, rej) => {
                mimg.onload = res; mimg.onerror = rej;
                mimg.src = `data:image/png;base64,${b64Mask}`;
            });
            state.maskImg = mimg;
        } else {
            state.maskImg = null;
        }

        // High-precision tile. Optional in every direction: the backend may
        // not have produced one, and an old browser may not be able to inflate
        // it — either way the 8-bit path above still works, so a failure here
        // only disables Inspect rather than breaking the preview.
        state.sourceHdr = null;
        if (hdrTile && hdrSupported()) {
            try {
                state.sourceHdr = await decodeHdrTile(hdrTile);
            } catch (e) {
                console.warn("[Bat_Grade] high-precision tile failed to decode:", e);
            }
        }
        inspectBar.refresh();

        // Cache for next reopen — the JPEG only. The 16-bit tile is a few
        // hundred KB and localStorage is a ~5MB origin-wide budget shared with
        // every other BAT node's thumbnail cache, so stashing it there would
        // evict the caches that actually need to survive a reload. Inspect
        // simply waits for the next run.
        _saveCachedPreview(node, { image: b64Image, mask: b64Mask || null });

        schedule();
    };

    // Restore the cached source thumbnail (and mask) on init so the
    // live preview shows something the first time the workflow opens,
    // before any Run.
    node._batGradeRestoreFromCache = () => {
        const cached = _loadCachedPreview(node);
        if (!cached?.image) return;
        node._batGradeIngest(cached.image, cached.mask || null, null);
    };

    root.appendChild(inspectBar.el);
    return root;
}

app.registerExtension({
    name: "Bat_Grade",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = buildPreview(this);
            // Dual-mode sizing: Nodes 2.0 derives node height from
            // computeLayoutSize, so a bare addDOMWidget + this.size left the
            // node and the widget disagreeing (grey band, no resize).
            addBatDOMWidget(this, "bat_grade_preview", "bat_grade_preview", el, {
                minWidth: 360, height: 520, growable: true,
            });
            this._batGradeWatchWidgets?.();
            clampNodeSize(this, 360, 520);
            // Defer the cache restore until after node.id is finalized
            // (LiteGraph sets it as part of construction; using a 0-ms
            // setTimeout puts us at the back of the microtask queue).
            setTimeout(() => this._batGradeRestoreFromCache?.(), 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batGradeIngest) return r;
            const one = (v) => Array.isArray(v) ? v[0] : v;
            const img = one(message.input_image);
            const msk = one(message.input_mask);
            const hdr = one(message.hdr_tile);
            if (img) this._batGradeIngest(img, msk, hdr);
            return r;
        };
    },
});
