/**
 * Bat_LayeredImages — dynamic layer inputs, a layers panel, and a live preview.
 *
 * Three parts, and they are independent of each other:
 *
 * 1. **Dynamic inputs.** `image_N` / `mask_N` pairs appear as they are wired and
 *    disappear when they are not, one spare pair always kept ready. Inputs are
 *    keyed by NAME in the prompt, not by index, so unlike output slots they can
 *    be trimmed from anywhere without repointing a link.
 *
 * 2. **The layers panel.** One row per connected layer: mode, opacity,
 *    visibility, solo. State lives in the node's `layers` STRING widget as JSON,
 *    the way Bat_Roto stores its shapes.
 *
 *    Not per-layer widgets, and that is a scar rather than a preference. A
 *    widget list that changes shape with the input count is a widget list whose
 *    saved values shift, because ComfyUI restores `widgets_values` positionally
 *    and at least one path (copy/paste) ignores `serialize = false`. That
 *    corrupted Bat_AdvancedBlend's nodes when its Advanced toggle was a widget.
 *    One STRING in a fixed position cannot do that.
 *
 * 3. **The preview**, in the same two layers as Bat_AdvancedBlend: a draft
 *    composited in a worker from downscaled tiles for instant feedback, and the
 *    real thing rendered by Python at full resolution a moment later. The
 *    view buttons adapt to the stack — Result, A, B, C… one per connected layer,
 *    plus Alpha.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { hdrSupported, decodeHdrTile, imageDataToSource } from "./bat_hdr_preview.js";
import { batTrack, registerCleanup, batNodeCacheKey, isNodeAlive, batReplayLastExecution } from "./bat_lifecycle.js";
import { attachZoomControl } from "./bat_zoom_control.js";
import { compositeStack, paintLayered } from "./bat_layered_core.js";
import { MODES, MODE_LABELS } from "./bat_blend_modes.js";
import { makeScrubber } from "./bat_scrub.js";

const NODE_TYPE = "Bat_LayeredImages";
const MAX_LAYERS = 8;          // must match MAX_LAYERS in bat_layered_images.py

/** Letters for the view buttons and the panel rows. A is the bottom layer. */
const LETTERS = "ABCDEFGH";

const DEFAULT_LAYER = { mode: "normal", opacity: 1.0, enabled: true };

// ── dynamic inputs ───────────────────────────────────────────────────────

function inputIndex(node, name) {
    return (node.inputs || []).findIndex((s) => s.name === name);
}

/**
 * Grow and shrink the image_N / mask_N pairs to fit what is wired.
 *
 * One spare pair beyond the last connected layer, always — so there is
 * somewhere to drop the next image without hunting for a button, and the node
 * does not carry eight empty slots for the common case of two.
 *
 * Never trims a connected slot. Someone can wire layer 4, unwire layer 2, and
 * expect layer 4 to still be there; cutting their link to tidy up the list
 * would be far worse than briefly showing a gap.
 */
function syncLayerInputs(node) {
    if (!node.inputs) return;

    let lastConnected = 0;
    for (let i = 1; i <= MAX_LAYERS; i++) {
        const img = node.inputs[inputIndex(node, `image_${i}`)];
        const msk = node.inputs[inputIndex(node, `mask_${i}`)];
        if ((img && img.link != null) || (msk && msk.link != null)) lastConnected = i;
    }
    const want = Math.min(MAX_LAYERS, lastConnected + 1);

    let have = 0;
    for (let i = 1; i <= MAX_LAYERS; i++) {
        if (inputIndex(node, `image_${i}`) !== -1) have = i;
    }

    for (let i = have + 1; i <= want; i++) {
        node.addInput(`image_${i}`, "IMAGE");
        node.addInput(`mask_${i}`, "MASK");
    }
    for (let i = have; i > want; i--) {
        for (const nm of [`mask_${i}`, `image_${i}`]) {
            const at = inputIndex(node, nm);
            if (at !== -1 && node.inputs[at].link == null) node.removeInput(at);
        }
    }

    // Label with the letter the preview and the panel use, so a row, a button
    // and a socket all say the same thing about the same layer.
    for (let i = 1; i <= MAX_LAYERS; i++) {
        const at = inputIndex(node, `image_${i}`);
        if (at === -1) continue;
        const letter = LETTERS[i - 1] || String(i);
        setSlotLabel(node, at, `${letter}  image${i === 1 ? " (base)" : ""}`);
        const mat = inputIndex(node, `mask_${i}`);
        if (mat !== -1) setSlotLabel(node, mat, `${letter}  mask`);
    }
    node.setDirtyCanvas?.(true, true);
    node._batLayeredSyncPanel?.();
}

/**
 * Relabel a slot without breaking Nodes 2.0 reactivity.
 *
 * `shallowReactive` tracks array mutation but not a property change on an
 * element already in the array, so assigning a fresh object to the index is
 * what makes Vue notice. Spread keeps `link` and everything litegraph cares
 * about. Same trick as bat_exposure_bracket.js.
 */
function setSlotLabel(node, index, label) {
    const arr = node.inputs;
    const slot = arr?.[index];
    if (!slot || slot.label === label) return;
    try { arr[index] = { ...slot, label }; }
    catch (_) { slot.label = label; }
}

/** Which layer slots are actually connected, bottom first. */
function connectedLayers(node) {
    const out = [];
    for (let i = 1; i <= MAX_LAYERS; i++) {
        const at = inputIndex(node, `image_${i}`);
        if (at !== -1 && node.inputs[at].link != null) out.push(i);
    }
    return out;
}

// ── layer state ──────────────────────────────────────────────────────────

/**
 * Read the layer settings out of the node's STRING widget.
 *
 * Mirrors `parse_layers()` in the Python, including its forgiveness: the string
 * comes from a saved workflow and can be absent, truncated or hand edited, and
 * a missing opacity should not break the node.
 */
export function readLayerState(node, count) {
    const w = node.widgets?.find((x) => x.name === "layers");
    let doc = null;
    try { doc = JSON.parse(w?.value || "{}"); } catch (_) { doc = null; }
    const raw = Array.isArray(doc?.layers) ? doc.layers : [];
    const out = [];
    for (let i = 0; i < count; i++) {
        const e = (raw[i] && typeof raw[i] === "object") ? raw[i] : {};
        const mode = MODES.includes(e.mode) ? e.mode : DEFAULT_LAYER.mode;
        let op = Number(e.opacity);
        if (!Number.isFinite(op)) op = 1;
        out.push({
            mode,
            opacity: Math.max(0, Math.min(1, op)),
            enabled: e.enabled !== false,
        });
    }
    return out;
}

function writeLayerState(node, settings) {
    const w = node.widgets?.find((x) => x.name === "layers");
    if (!w) return;
    w.value = JSON.stringify({ layers: settings });
    // The widget is the serialised truth, so a change to it has to mark the
    // graph dirty or it can be lost on save.
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
}

// ── the editor ───────────────────────────────────────────────────────────

function buildEditor(node) {
    const track = batTrack(node);

    const root = document.createElement("div");
    root.style.cssText = `position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; overflow:hidden;`;

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative; flex:1 1 auto; min-height:0; display:flex; background:#000;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; display:block;";
    stage.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;`;
    hint.textContent = "Wire image_1 and run once.";
    stage.appendChild(hint);

    const badge = document.createElement("div");
    badge.style.cssText = `position:absolute; right:6px; top:4px; font:10px monospace;
        color:#9aa; background:rgba(0,0,0,0.62); padding:2px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; text-align:right; line-height:1.45;
        display:none;`;
    stage.appendChild(badge);

    root.appendChild(stage);

    const viewRow = document.createElement("div");
    viewRow.style.cssText = `display:flex; flex-wrap:wrap; gap:4px; padding:4px 6px;
        background:#111; border-top:1px solid #2a2a2a; flex:0 0 auto; align-items:center;`;
    root.appendChild(viewRow);

    // The layer stack. Scrolls rather than growing the node without limit — at
    // eight layers this would otherwise be taller than the picture.
    const panel = document.createElement("div");
    panel.style.cssText = `display:flex; flex-direction:column; gap:2px; padding:4px 5px;
        background:#0e0e0e; border-top:1px solid #2a2a2a; flex:0 0 auto;
        max-height:44%; overflow-y:auto; font:11px monospace;`;
    root.appendChild(panel);

    const ctx = canvas.getContext("2d");
    const off = document.createElement("canvas");
    const offCtx = off.getContext("2d", { willReadFrequently: true });

    const viewKey = batNodeCacheKey(app, "bat_layered_view", node);
    const saved = (() => {
        try { return JSON.parse(localStorage.getItem(viewKey) || "{}") || {}; }
        catch (_) { return {}; }
    })();

    const state = {
        layers: [],          // [{data, mask, width, height}] bottom first
        slots: [],           // original image_N numbers, for the row labels
        meta: null,
        view: saved.view || "result",
        // Display transform, owned by bat_zoom_control.js. 1 = fit.
        dispZoom: Number.isFinite(saved.dispZoom) ? saved.dispZoom : 1,
        panX: 0, panY: 0, dispScale: 1, offX: 0, offY: 0,
        imgW: 1, imgH: 1,
        tw: 0, th: 0,
        out: null, alpha: null,
        compositeMs: 0,
        needsRun: false,
        runId: 0,
        full: null, fullKey: null, fullRect: null, fullPending: false, fullStale: false,
        serverMs: 0,
    };
    node._batLayeredState = state;

    // ── worker ───────────────────────────────────────────────────────────
    let worker = null, workerReady = false;
    let jobId = 0, jobInFlight = 0, jobPending = null;
    try {
        worker = new Worker(new URL("./bat_layered_worker.js", import.meta.url),
                            { type: "module" });
        worker.onerror = (e) => {
            console.warn("[Bat_LayeredImages] composite worker failed; falling "
                         + "back to the main thread:", e.message || e);
            try { worker.terminate(); } catch (_) {}
            worker = null; workerReady = false;
        };
        worker.onmessage = (e) => onWorkerMessage(e.data);
        track.dispose(() => { try { worker?.terminate(); } catch (_) {} });
    } catch (e) {
        console.warn("[Bat_LayeredImages] no composite worker; compositing on "
                     + "the main thread:", e);
        worker = null;
    }

    function onWorkerMessage(msg) {
        if (!msg || !isNodeAlive(node)) return;
        if (msg.type === "error") {
            console.warn("[Bat_LayeredImages] worker composite failed:", msg.message);
            jobInFlight = 0; flushPending();
            return;
        }
        if (msg.type !== "rendered") return;
        jobInFlight = 0;
        // Anything but the newest job is discarded: during a drag the editor
        // keeps asking, and an answer to a superseded question would flicker
        // the canvas backwards.
        if (msg.id === jobId) {
            state.compositeMs = msg.ms;
            state.tw = msg.w; state.th = msg.h;
            state.imgW = msg.w; state.imgH = msg.h;
            if (off.width !== msg.w || off.height !== msg.h) {
                off.width = msg.w; off.height = msg.h;
            }
            offCtx.putImageData(new ImageData(msg.rgba, msg.w, msg.h), 0, 0);
            if (msg.out) { state.out = msg.out; state.alpha = msg.alpha; }
            hint.style.display = "none";
            present(); paintBadge();
        }
        flushPending();
    }

    function flushPending() {
        if (!jobPending || jobInFlight) return;
        const job = jobPending; jobPending = null; postJob(job);
    }

    function postJob(job) {
        if (!worker || !workerReady) return false;
        if (jobInFlight) { jobPending = job; return true; }
        jobInFlight = ++jobId;
        job.id = jobInFlight;
        worker.postMessage(job);
        return true;
    }

    // ── display ──────────────────────────────────────────────────────────
    /** CSS pixels, which is the contract bat_zoom_control.js reads back. */
    function recomputeDisplay() {
        const cw = canvas.clientWidth || 1, ch = canvas.clientHeight || 1;
        const iw = state.tw || 1, ih = state.th || 1;
        state.imgW = iw; state.imgH = ih;
        const fit = Math.min(cw / iw, ch / ih);
        state.dispScale = fit * (state.dispZoom || 1);
        state.offX = (cw - iw * state.dispScale) / 2 + (state.panX || 0);
        state.offY = (ch - ih * state.dispScale) / 2 + (state.panY || 0);
    }

    function present() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const cw = Math.max(1, stage.clientWidth), ch = Math.max(1, stage.clientHeight);
        const bw = Math.round(cw * dpr), bh = Math.round(ch * dpr);
        if (canvas.width !== bw || canvas.height !== bh) {
            canvas.width = bw; canvas.height = bh;
        }
        recomputeDisplay();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cw, ch);
        if (!state.tw) return;
        ctx.imageSmoothingEnabled = state.dispScale < 1;
        ctx.imageSmoothingQuality = "high";
        // Draft first, so a partially-stale full render shows the approximation
        // at its edges rather than black.
        ctx.drawImage(off, state.offX, state.offY,
                      state.tw * state.dispScale, state.th * state.dispScale);
        if (state.full && state.fullRect) {
            const r = state.fullRect;
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(state.full,
                          state.offX + r.x * state.dispScale,
                          state.offY + r.y * state.dispScale,
                          r.w * state.dispScale, r.h * state.dispScale);
        }
    }

    function paintBadge() {
        const lines = [];
        const m = state.meta;
        if (m?.w) {
            lines.push(`${m.w}x${m.h}   ${state.layers.length} layer`
                + (state.layers.length === 1 ? "" : "s")
                + (m.frames > 1 ? `   f${m.frame}/${m.frames - 1}` : ""));
        }
        const workScale = (m?.w && state.tw)
            ? state.dispScale * (state.tw / m.w) : state.dispScale;
        const cost = state.compositeMs >= 1 ? `  ${Math.round(state.compositeMs)}ms` : "";
        lines.push(`${workScale.toFixed(2)}:1   ` + (state.full
            ? (state.fullStale ? "full res — updating…" : "full res")
            : (state.fullPending ? `draft${cost} — rendering…` : `draft${cost}`)));
        if (state.needsRun) lines.push("⟳ Run to apply preview_frame / resize");
        badge.textContent = lines.join("\n");
        badge.style.display = lines.length ? "block" : "none";
    }

    // ── compositing ──────────────────────────────────────────────────────
    function settings() {
        return readLayerState(node, state.layers.length);
    }

    function clampOutput() {
        const w = node.widgets?.find((x) => x.name === "clamp_output");
        return w ? !!w.value : false;
    }

    /** Main-thread fallback. Correct, and slower than the worker. */
    function computeSync() {
        if (!state.layers.length) return false;
        const w = state.layers[0].width, h = state.layers[0].height;
        const t0 = performance.now();
        const { out, alpha } = compositeStack(state.layers, settings(), w, h,
                                              clampOutput());
        state.compositeMs = performance.now() - t0;
        if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
        const img = offCtx.createImageData(w, h);
        paintLayered({
            view: state.view, layers: state.layers, out, alpha, w, h, rgba: img.data,
        });
        offCtx.putImageData(img, 0, 0);
        state.out = out; state.alpha = alpha;
        state.tw = w; state.th = h;
        hint.style.display = "none";
        return true;
    }

    let raf = 0, settleT = 0;
    const SETTLE_MS = 120;

    function draw(settled) {
        if (!isNodeAlive(node) || !state.layers.length) return;
        if (worker && workerReady) {
            if (postJob({
                type: "render", settings: settings(), view: state.view,
                clampOutput: clampOutput(),
                // 8MB a frame; only ask when the picture has settled and the
                // probe might read them.
                wantValues: !!settled,
            })) { paintBadge(); return; }
        }
        try {
            if (computeSync()) { present(); paintBadge(); }
        } catch (e) {
            console.error("[Bat_LayeredImages] preview failed:", e);
        }
    }

    function schedule(interactive) {
        invalidateFull(interactive, false);
        if (interactive) {
            clearTimeout(settleT);
            settleT = track.timeout(setTimeout(() => draw(true), SETTLE_MS));
        }
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; draw(!interactive); });
    }
    node._batLayeredRepaint = () => schedule(false);

    // ── the full-resolution layer ─────────────────────────────────────────
    const FULL_DELAY = 140;
    let fullTimer = 0, fullAbort = null, lastFullAt = 0;

    function abortFull() {
        state.fullPending = false;
        clearTimeout(fullTimer);
        try { fullAbort?.abort(); } catch (_) {}
        fullAbort = null;
    }
    function dropFull() {
        abortFull();
        state.full = null; state.fullKey = null; state.fullRect = null;
        state.fullStale = false;
    }

    function visibleRegion() {
        const cw = Math.max(1, canvas.clientWidth), ch = Math.max(1, canvas.clientHeight);
        const sc = state.dispScale || 1;
        let x0 = Math.max(0, Math.floor((0 - state.offX) / sc) - 1);
        let y0 = Math.max(0, Math.floor((0 - state.offY) / sc) - 1);
        let x1 = Math.min(state.tw, Math.ceil((cw - state.offX) / sc) + 1);
        let y1 = Math.min(state.th, Math.ceil((ch - state.offY) / sc) + 1);
        if (x1 <= x0 || y1 <= y0) return null;
        const tileRect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
        const k = (state.meta?.w && state.tw) ? state.meta.w / state.tw : 1;
        const world = {
            x: Math.floor(tileRect.x * k), y: Math.floor(tileRect.y * k),
            w: Math.max(1, Math.round(tileRect.w * k)),
            h: Math.max(1, Math.round(tileRect.h * k)),
        };
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const outW = Math.max(1, Math.min(Math.round(tileRect.w * sc * dpr), world.w, 4096));
        const outH = Math.max(1, Math.min(Math.round(tileRect.h * sc * dpr), world.h, 4096));
        return { tileRect, world, outW, outH };
    }

    /** `runId` must stay in this key — see the same note in bat_advanced_blend.js.
     *  A new execution can ship a different frame of the batch, and without the
     *  generation the request would be skipped as redundant and the previous
     *  frame's render would stay on screen. */
    function fullKeyFor(region) {
        return JSON.stringify([state.runId, settings(), state.view,
                               clampOutput(), region.world, region.outW, region.outH]);
    }

    async function requestFull() {
        if (!isNodeAlive(node) || !state.tw || node.id == null) return;
        const region = visibleRegion();
        if (!region) return;
        const key = fullKeyFor(region);
        if (state.fullKey === key && state.full) return;

        const ctl = new AbortController();
        fullAbort = ctl;
        state.fullPending = true;
        paintBadge();
        const started = performance.now();
        try {
            const res = await fetch("/bat/layered_images/render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: ctl.signal,
                body: JSON.stringify({
                    node_id: String(node.id),
                    roi: [region.world.x, region.world.y, region.world.w, region.world.h],
                    out_w: region.outW, out_h: region.outH,
                    view: state.view, settings: settings(),
                    clamp_output: clampOutput(),
                }),
            });
            if (!res.ok) {
                // 409 only means this node has not run since the server came up.
                if (res.status !== 409) {
                    console.warn("[Bat_LayeredImages] full-resolution render failed:",
                                 res.status, await res.text().catch(() => ""));
                }
                if (state.fullStale) dropFull();
                state.fullPending = false;
                present(); paintBadge();
                return;
            }
            const blob = await res.blob();
            const img = (typeof createImageBitmap === "function")
                ? await createImageBitmap(blob)
                : await new Promise((ok, no) => {
                    const i = new Image();
                    i.onload = () => ok(i); i.onerror = no;
                    i.src = URL.createObjectURL(blob);
                });
            if (!isNodeAlive(node) || ctl.signal.aborted) return;
            if (fullKeyFor(region) !== key) { state.fullPending = false; return; }
            state.full = img; state.fullKey = key; state.fullRect = region.tileRect;
            state.fullStale = false; state.fullPending = false;
            state.serverMs = performance.now() - started;
            lastFullAt = performance.now();
            present(); paintBadge();
        } catch (e) {
            if (e?.name !== "AbortError") {
                console.warn("[Bat_LayeredImages] full-resolution render error:", e);
                if (state.fullStale) dropFull();
                present();
            }
            state.fullPending = false;
            paintBadge();
        }
    }

    /**
     * A layer change only alters content, so the sharp-but-stale render is held
     * while its replacement computes rather than dropping back to the draft. A
     * pan or zoom moves the region, so it must go — it would be drawn over the
     * wrong part of the picture. Same rule as Bat_AdvancedBlend's Full setting,
     * and the view-change exception is the fragile half.
     */
    function invalidateFull(interactive, viewChanged) {
        if (!viewChanged && state.full) {
            state.fullStale = true;
            abortFull();
        } else {
            dropFull();
        }
        const rtt = state.serverMs || 0;
        const delay = Math.max(FULL_DELAY, Math.min(rtt, 400));
        const throttle = Math.max(2 * rtt, 180);
        if (interactive && rtt && rtt < 250
            && performance.now() - lastFullAt >= throttle) {
            requestFull();
            return;
        }
        fullTimer = track.timeout(setTimeout(requestFull, delay));
    }

    // ── view buttons, adapting to the stack ──────────────────────────────
    function rebuildViewRow() {
        viewRow.textContent = "";
        const mk = (id, label, title) => {
            const b = document.createElement("button");
            b.textContent = label; b.title = title;
            b.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
                border-radius:3px; padding:2px 7px; font:11px monospace; cursor:pointer;`;
            b.addEventListener("click", () => {
                state.view = id; saveView(); rebuildViewRow(); schedule(false);
            });
            b.addEventListener("pointerdown", (e) => e.stopPropagation());
            const on = state.view === id;
            b.style.background = on ? "#2b4a63" : "#222";
            b.style.color = on ? "#cfe8ff" : "#9aa";
            b.style.borderColor = on ? "#4a7fa8" : "#333";
            viewRow.appendChild(b);
        };
        mk("result", "Result", "The composited stack.");
        state.layers.forEach((_, i) => {
            const letter = LETTERS[i] || String(i + 1);
            mk(`layer:${i}`, letter,
               `Layer ${letter} on its own, through its mask, over black.\n`
               + "Shown at full opacity in Normal — the layer itself, rather than "
               + "how its blend mode happens to render it against nothing.");
        });
        mk("alpha", "Alpha",
           "The accumulated alpha, which is the node's second output.\n"
           + "Below 1 only where the bottom layer is masked or partly "
           + "transparent — everywhere else the stack is opaque.");
    }

    function saveView() {
        try {
            localStorage.setItem(viewKey, JSON.stringify({
                view: state.view, dispZoom: state.dispZoom,
            }));
        } catch (_) {}
    }

    // ── the layers panel ─────────────────────────────────────────────────
    /**
     * Rebuild the stack rows.
     *
     * Displayed TOP layer first, the reverse of the input order, because that is
     * how every compositing application shows a stack. `image_1` is the bottom
     * of the stack and the bottom row.
     */
    function rebuildPanel() {
        panel.textContent = "";
        const cfg = settings();
        const n = state.layers.length;

        if (!n) {
            const empty = document.createElement("div");
            empty.textContent = "No layers connected. Wire image_1 — it is the "
                + "bottom of the stack.";
            empty.style.cssText = "color:#68757f; padding:3px 2px; white-space:normal;";
            panel.appendChild(empty);
            return;
        }

        for (let i = n - 1; i >= 0; i--) {
            const row = document.createElement("div");
            row.style.cssText = `display:flex; align-items:center; gap:5px;
                padding:2px 3px; border:1px solid ${state.view === `layer:${i}` ? "#4a7fa8" : "#232323"};
                border-radius:3px; background:#141414;`;

            const letter = LETTERS[i] || String(i + 1);
            const slot = state.slots[i] ?? (i + 1);

            // Visibility. Not the same thing as opacity 0 — this is the thing you
            // flick on and off to see what a layer is contributing.
            const eye = document.createElement("button");
            eye.textContent = cfg[i].enabled ? "◉" : "○";
            eye.title = "Show or hide this layer";
            eye.style.cssText = `background:none; border:none; cursor:pointer;
                color:${cfg[i].enabled ? "#cde" : "#556"}; font:12px monospace; padding:0 2px;`;
            eye.addEventListener("click", () => {
                const s = settings();
                s[i].enabled = !s[i].enabled;
                writeLayerState(node, s);
                rebuildPanel(); schedule(false);
            });
            eye.addEventListener("pointerdown", (e) => e.stopPropagation());

            const tag = document.createElement("button");
            tag.textContent = letter;
            tag.title = `Solo layer ${letter} in the preview (input image_${slot})`;
            tag.style.cssText = `background:${state.view === `layer:${i}` ? "#2b4a63" : "#1c1c1c"};
                color:#cde; border:1px solid #333; border-radius:3px; width:18px;
                font:11px monospace; cursor:pointer;`;
            tag.addEventListener("click", () => {
                state.view = state.view === `layer:${i}` ? "result" : `layer:${i}`;
                saveView(); rebuildViewRow(); rebuildPanel(); schedule(false);
            });
            tag.addEventListener("pointerdown", (e) => e.stopPropagation());

            const mode = document.createElement("select");
            // Narrower than it was: the opacity bar needs the room more than the
            // mode names do, and they stay readable truncated.
            mode.style.cssText = `background:#1c1c1c; color:#cde; border:1px solid #333;
                border-radius:3px; font:10px monospace; flex:0 1 92px; min-width:62px;
                cursor:pointer;`;
            for (const md of MODES) {
                const o = document.createElement("option");
                o.value = md; o.textContent = MODE_LABELS[md] || md;
                mode.appendChild(o);
            }
            mode.value = cfg[i].mode;
            // The bottom layer has nothing under it, so its mode is inert —
            // say so rather than letting someone wonder why it does nothing.
            if (i === 0) {
                mode.disabled = true;
                mode.title = "The bottom layer has nothing beneath it, so a blend "
                    + "mode has nothing to blend with. Its mask and opacity still "
                    + "apply, and show up in the Alpha output.";
                mode.style.opacity = "0.45";
            }
            mode.addEventListener("change", () => {
                const s = settings();
                s[i].mode = mode.value;
                writeLayerState(node, s);
                schedule(false);
            });
            mode.addEventListener("pointerdown", (e) => e.stopPropagation());

            // A scrubber, not an <input type="range">. In a row this narrow a
            // native range gives a ~12px thumb, drops the drag the moment the
            // pointer leaves, and offers 2.5% per pixel. See bat_scrub.js.
            // The readout lives inside the bar, which is also where the width it
            // used to occupy went.
            const op = makeScrubber({
                value: cfg[i].opacity, min: 0, max: 1, step: 0.01,
                defaultValue: 1,
                format: (v) => `${Math.round(v * 100)}%`,
                title: `Layer ${letter} opacity, multiplied with its mask`,
                onInput: (v) => {
                    const s = settings();
                    s[i].opacity = v;
                    writeLayerState(node, s);
                    schedule(true);
                },
                // The drag itself already repainted; this asks for the settled
                // full-resolution pass once the gesture ends.
                onCommit: () => schedule(false),
            });

            const mk = document.createElement("span");
            mk.textContent = state.layers[i].mask ? "▦" : " ";
            mk.title = state.layers[i].mask
                ? "This layer has a mask wired" : "No mask on this layer";
            mk.style.cssText = "color:#7a8b96; font:10px monospace; width:9px; flex:0 0 auto;";

            row.append(eye, tag, mode, op.el, mk);
            panel.appendChild(row);
        }
    }
    node._batLayeredSyncPanel = () => { rebuildPanel(); rebuildViewRow(); };

    // ── zoom / pan ───────────────────────────────────────────────────────
    try {
        attachZoomControl({
            wrap: stage, canvas, state,
            onChange: () => {
                saveView();
                invalidateFull(false, true);
                present(); paintBadge();
            },
        });
    } catch (e) {
        console.warn("[Bat_LayeredImages] zoom control unavailable:", e);
    }

    // ── ingest ───────────────────────────────────────────────────────────
    node._batLayeredIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        const tiles = one(msg.tiles);
        const masks = one(msg.masks) || [];
        state.meta = {
            w: Number(one(msg.w)) || 0,
            h: Number(one(msg.h)) || 0,
            frames: Number(one(msg.frames)) || 1,
            frame: Number(one(msg.preview_frame)) || 0,
        };
        state.slots = one(msg.slots) || [];

        if (!tiles || !hdrSupported()) {
            hint.textContent = hdrSupported()
                ? "Preview tiles unavailable — see the log."
                : "This browser has no DecompressionStream, so the preview tiles "
                  + "can't be inflated.";
            hint.style.display = "block";
            return;
        }

        const decoded = [];
        try {
            for (let i = 0; i < tiles.length; i++) {
                const t = await decodeHdrTile(tiles[i]);
                let mask = null;
                if (masks[i]) {
                    const im = new Image();
                    await new Promise((ok, no) => {
                        im.onload = ok; im.onerror = no;
                        im.src = `data:image/png;base64,${masks[i]}`;
                    });
                    const c = document.createElement("canvas");
                    c.width = t.width; c.height = t.height;
                    const cx = c.getContext("2d", { willReadFrequently: true });
                    cx.drawImage(im, 0, 0, t.width, t.height);
                    const px = cx.getImageData(0, 0, t.width, t.height).data;
                    mask = new Float32Array(t.width * t.height);
                    for (let k = 0, p = 0; k < mask.length; k++, p += 4) mask[k] = px[p] / 255;
                }
                decoded.push({ data: t.data, mask, width: t.width, height: t.height });
            }
        } catch (e) {
            console.warn("[Bat_LayeredImages] tile decode failed:", e);
            return;
        }
        if (!isNodeAlive(node)) return;

        // Every tile is conformed and area-sampled at the same long edge, so a
        // disagreement means something changed shape mid-payload.
        const w0 = decoded[0]?.width, h0 = decoded[0]?.height;
        if (decoded.some((d) => d.width !== w0 || d.height !== h0)) {
            console.warn("[Bat_LayeredImages] preview tiles disagree on size; skipping.");
            return;
        }

        state.layers = decoded;
        state.tw = w0; state.th = h0;
        state.runId++;
        state.needsRun = false;
        // A new execution can be a different frame entirely, so the render on
        // screen is not stale, it is of something else.
        dropFull();

        if (worker) {
            try {
                worker.postMessage({
                    type: "layers",
                    layers: decoded.map((d) => ({
                        data: d.data.slice(),
                        mask: d.mask ? d.mask.slice() : null,
                    })),
                    w: w0, h: h0,
                });
                workerReady = true;
                jobPending = null; jobInFlight = 0;
            } catch (e) {
                console.warn("[Bat_LayeredImages] could not seed the worker:", e);
                workerReady = false;
            }
        }

        // Settings shipped by Python are the sanitised truth; adopt them so the
        // panel and the render can never disagree about what was just rendered.
        const shipped = one(msg.settings);
        if (Array.isArray(shipped) && shipped.length === decoded.length) {
            writeLayerState(node, shipped);
        }

        if (state.view.startsWith("layer:")
            && parseInt(state.view.slice(6), 10) >= decoded.length) {
            state.view = "result";
        }
        rebuildViewRow(); rebuildPanel();
        schedule(false);
    };

    try {
        const ro = new ResizeObserver(() => { present(); paintBadge(); });
        ro.observe(stage);
        track.observer(ro, stage);
    } catch (_) {}

    track.dispose(() => {
        clearTimeout(settleT);
        dropFull();
        state.layers = []; state.out = null; state.alpha = null;
    });

    rebuildViewRow();
    rebuildPanel();
    return root;
}

app.registerExtension({
    name: "Bat_LayeredImages",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // A graph reload (Ctrl+Z is one) destroys and rebuilds every node, so
        // replay the last run's preview payload into the new instance.
        batReplayLastExecution(nodeType);

        registerCleanup(nodeType);

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const self = this;

            // The layer state widget is the serialised truth, not something to
            // type in. Hidden from the node body; still visible in the
            // parameters panel for anyone who wants to inspect it.
            const lw = this.widgets?.find((w) => w.name === "layers");
            if (lw) {
                lw.hidden = true;
                lw.options = { ...(lw.options || {}), hidden: true };
            }

            for (const name of ["clamp_output"]) {
                const w = this.widgets?.find((x) => x.name === name);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function (v) {
                    try { orig?.call(this, v); } catch (_) {}
                    self._batLayeredRepaint?.();
                };
            }
            // These change what Python SHIPS rather than how the tiles are
            // combined, so nothing here can recompute them — watched only so
            // the node can say a Run is needed instead of appearing to ignore
            // the click.
            for (const name of ["preview_frame", "resize_mode", "resize_filter"]) {
                const w = this.widgets?.find((x) => x.name === name);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function (v) {
                    try { orig?.call(this, v); } catch (_) {}
                    const st = self._batLayeredState;
                    if (st) { st.needsRun = true; self._batLayeredRepaint?.(); }
                };
            }

            const el = buildEditor(this);
            addBatDOMWidget(this, "bat_layered_preview", "bat_layered_preview", el, {
                minWidth: 420, height: 520, growable: true,
            });
            clampNodeSize(this, 420, 520);

            // Deferred: the node is not in the graph yet, and addInput before
            // that point has nowhere to register links.
            setTimeout(() => syncLayerInputs(this), 0);
            return r;
        };

        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConn?.apply(this, arguments);
            // After litegraph has finished updating the link, not during.
            setTimeout(() => syncLayerInputs(this), 0);
            return r;
        };

        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const r = onConfigured?.apply(this, arguments);
            syncLayerInputs(this);
            this._batLayeredSyncPanel?.();
            this._batLayeredRepaint?.();
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            if (message && this._batLayeredIngest) this._batLayeredIngest(message);
            return r;
        };
    },
});
