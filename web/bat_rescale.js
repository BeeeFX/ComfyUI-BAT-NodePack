/**
 * Bat_Rescale — a resolution preview that does not change size on screen.
 *
 * The idea
 * --------
 * Every other preview in ComfyUI scales with the image, so halving the
 * resolution just makes the picture on the node smaller and tells you nothing
 * about the pixels. This viewer inverts that: the picture is held at a
 * CONSTANT size on screen and only its resolution changes underneath, so what
 * you see when you pull `scale` down is detail going away rather than a
 * thumbnail shrinking. That is the whole reason the node exists, and it is why
 * the viewer is region-based (a window onto the source at a chosen zoom)
 * rather than fit-based (the frame squeezed into the node).
 *
 * The consequence worth stating: the screen box is a function of the ZOOM and
 * the node's size, and of nothing else. No widget on this node can change it.
 *
 * Two layers
 * ----------
 * Same arrangement as Bat_AdvancedBlend, for the same reasons:
 *
 *   • DRAFT — this file scaling the tile it already has, via two canvas
 *     `drawImage` calls, on the same frame as your mouse. Approximate: the
 *     browser's resampler is not the node's resampler.
 *   • TRUTH — one POST to `/bat/rescale/render`, which runs the node's own
 *     resampler on the cached frame at full resolution and returns exactly the
 *     region on screen. Region renders are asserted bit-identical to a crop of
 *     a full-frame resize, so this is not a mirror of the render, it IS the
 *     render.
 *
 * The response is a single PNG of twice the requested height: the unscaled
 * region on top, the rescaled-and-magnified region below. One request for both
 * halves is what makes the wipe trustworthy — they are framed from one snapped
 * region on the server, so they cannot disagree by a subpixel and show a step
 * at the divider that reads as detail. It also means dragging the wipe, and
 * holding B to compare, cost nothing at all.
 *
 * Why the viewer controls are not widgets
 * ---------------------------------------
 * Zoom, pan, wipe position, magnify filter and compare mode are all display
 * decisions — they never reach the render — and ComfyUI restores
 * `widgets_values` positionally, so adding serialised widgets for them would
 * shift every saved workflow's values. They live in workflow-scoped
 * localStorage instead, like the other editors' view state. `preview_frame` is
 * the exception: it IS a widget, because which frame Python ships is Python's
 * decision, and the viewer's frame stepper simply drives it.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import {
    batTrack, isNodeAlive, batNodeCacheKey, batReplayLastExecution, batPreviewWillReplay,
} from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_Rescale";

// Zoom presets, in device pixels per source pixel. `null` is fit-to-panel.
const ZOOM_PRESETS = [null, 1, 2, 4];
const MIN_ZOOM = 0.05;
const MAX_ZOOM = 16;

// How close to the panel edge the wipe divider may be parked.
//
// Not cosmetic: dragged flush to an edge the handle is half outside the panel
// and there is nothing left to aim at, and because the position is persisted,
// a workflow reopened in that state came back with a divider that could not be
// grabbed at all. The extremes are not lost — that is what the compare button's
// `scaled` and `original` modes are, and they are reachable in one click
// instead of a pixel-perfect drag.
const WIPE_MARGIN = 0.03;
const clampWipe = (w) => Math.max(WIPE_MARGIN, Math.min(1 - WIPE_MARGIN,
    Number.isFinite(w) ? w : 0.5));

// Debounce before asking the server for the truth. Long enough that a slider
// drag does not queue a request per frame, short enough that letting go feels
// immediate. Only ever one request in flight; a change during it re-fires on
// completion, so a slow render costs latency and never responsiveness.
const TRUTH_DEBOUNCE_MS = 80;


/* ==========================================================================
 * Size planning — the mirror of plan_size() in bat_rescale.py
 * ==========================================================================
 *
 * This is here so the node can show the output resolution BEFORE you run it,
 * which is most of what makes the widgets legible ("0.66 of what, exactly?").
 * Being a second implementation of the render's arithmetic, it is exactly the
 * kind of thing that drifts, so tests/verify_rescale.py drives this function
 * and the Python one over every mode and asserts they agree.
 *
 * `Math.floor(x + 0.5)` rather than `Math.round`: Python's round() is banker's
 * rounding and JS's is not, so a factor landing exactly on .5 (0.5 of a
 * 1921-wide plate) would otherwise disagree between this readout and the file
 * that renders it.
 */

export function roundHalfUp(x) {
    return Math.floor(Number(x) + 0.5);
}

export function snapTo(v, multiple) {
    const m = Math.max(1, Math.floor(multiple || 1));
    if (m <= 1) return Math.max(1, Math.floor(v));
    return Math.max(m, roundHalfUp(v / m) * m);
}

export function planSize(h, w, p) {
    h = Math.max(1, Math.floor(h));
    w = Math.max(1, Math.floor(w));
    const mult = Math.max(1, Math.floor(p.multiple_of || 1));

    if (p.mode === "match_reference") {
        if (p.ref_h && p.ref_w) {
            return { h: snapTo(p.ref_h, mult), w: snapTo(p.ref_w, mult) };
        }
        return { h: snapTo(h, mult), w: snapTo(w, mult) };
    }

    let s;
    if (p.mode === "long_edge")       s = Number(p.target) / Math.max(h, w);
    else if (p.mode === "short_edge") s = Number(p.target) / Math.min(h, w);
    else if (p.mode === "width")      s = Number(p.target) / w;
    else if (p.mode === "height")     s = Number(p.target) / h;
    else if (p.mode === "megapixels") s = Math.sqrt(Math.max(Number(p.megapixels), 1e-6) * 1e6 / (w * h));
    else                              s = Number(p.scale);

    s = Math.max(s, 1e-6);
    return { h: snapTo(roundHalfUp(h * s), mult), w: snapTo(roundHalfUp(w * s), mult) };
}


/* ==========================================================================
 * Pointer geometry
 * ==========================================================================
 *
 * Pulled out of the viewer closure and exported so tests/verify_rescale.py can
 * drive it, because this is the code that has been wrong once already: the
 * divider's grab zone was computed in a different coordinate system from the
 * divider itself, which reads as "the handle is offset from where I can see it,
 * and how far off depends on the graph zoom".
 *
 * There are THREE coordinate systems and the middle one is easy to miss:
 *
 *   displayed px   what a pointer event carries (clientX/clientY)
 *   layout px      what `clientWidth` reports and CSS lays out in
 *   device px      what the canvas backing store and computeView() use
 *
 * Displayed and layout differ by litegraph's graph zoom, which scales the node
 * with a CSS transform that `clientWidth` knows nothing about. Layout and
 * device differ by devicePixelRatio. Applying only the second one — which is
 * what the first version did — is exactly correct at graph zoom 1.0 and
 * progressively wrong either side of it.
 */

/** Grab zone width, in DISPLAYED pixels: the same size under your finger at
 *  any graph zoom, which a fixed device-pixel tolerance is not. */
export const GRAB_DISPLAYED_PX = 14;

/** displayed px -> device px, measured off the element rather than read from
 *  `app.canvas.ds.scale`, so any other transform in the chain (browser page
 *  zoom, a frontend that nests the widget differently) is covered too. */
export function pointerToDevice(rectWidth, layoutWidth, dpr) {
    const graph = rectWidth > 0 && layoutWidth > 0 ? layoutWidth / rectWidth : 1;
    return { graph, toDevice: graph * Math.max(1, dpr || 1) };
}

/** Where the divider and its grip are, in canvas device pixels. */
export function wipeGeom(view, wipe, toDevice) {
    return {
        x: view.drawX + view.outW * wipe,
        tol: GRAB_DISPLAYED_PX * toDevice,
        gripW: 11 * toDevice,
        gripH: 34 * toDevice,
        midY: view.drawY + view.outH / 2,
    };
}

/** Is a canvas-device-pixel point on the divider? Fatter target over the grip. */
export function isOnDivider(geom, view, cx, cy) {
    if (cy < view.drawY - geom.tol || cy > view.drawY + view.outH + geom.tol) {
        return false;
    }
    const nearGrip = Math.abs(cy - geom.midY) <= geom.gripH / 2 + geom.tol;
    return Math.abs(cx - geom.x) <= (nearGrip ? geom.gripW / 2 + geom.tol : geom.tol);
}


/* ==========================================================================
 * View state persistence
 * ========================================================================== */

function viewKey(node) {
    return batNodeCacheKey(app, "bat_rescale_view", node);
}
function cacheKey(node) {
    return batNodeCacheKey(app, "bat_rescale_src", node);
}

function loadJson(key, fallback) {
    try {
        const raw = localStorage.getItem(key);
        return raw ? (JSON.parse(raw) ?? fallback) : fallback;
    } catch (_) { return fallback; }
}
function saveJson(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) {}
}


/* ==========================================================================
 * The viewer
 * ========================================================================== */

function buildViewer(node) {
    const track = batTrack(node);

    const saved = loadJson(viewKey(node), {});
    const state = {
        // Source identity, filled in by a run (or restored from localStorage).
        token: null,
        frames: 1,
        srcW: 0, srcH: 0,
        liveBatch: false,
        thumb: null,              // <img> whole-frame JPEG, the cold-start draft

        // View — display only, none of it reaches the render.
        zoom: saved.zoom === undefined ? null : saved.zoom,   // null = fit
        cx: saved.cx ?? null,     // pan centre in source px (null = frame centre)
        cy: saved.cy ?? null,
        wipe: clampWipe(saved.wipe ?? 0.5),
        compare: saved.compare ?? "wipe",     // "wipe" | "scaled" | "original"
        magnify: saved.magnify ?? "pixels",   // "pixels" | "smooth"

        // Layers.
        pair: null,               // {orig, scaled, roi, outW, outH, info}
        holdOriginal: false,      // B held down
        pending: false,
        inflight: false,
        stale: false,
        error: null,
        lastInfo: null,
    };
    node._batRescale = state;

    /* ---- chrome ---------------------------------------------------------- */

    const root = document.createElement("div");
    root.tabIndex = 0;
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column; outline:none;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px;
        overflow:hidden; font:11px monospace; color:#9aa;
    `;

    const bar = document.createElement("div");
    bar.style.cssText = `
        display:flex; align-items:center; gap:4px; padding:3px 5px;
        background:#141414; border-bottom:1px solid #2a2a2a;
        flex:0 0 auto; user-select:none; flex-wrap:wrap;
    `;

    const btn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.title = title;
        b.style.cssText = `
            background:#222; color:#9aa; border:1px solid #333; border-radius:3px;
            padding:2px 6px; font:11px monospace; cursor:pointer; flex:0 0 auto;
        `;
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };
    const paintBtn = (b, on) => {
        b.style.background = on ? "#2b4a63" : "#222";
        b.style.color = on ? "#cfe8ff" : "#9aa";
        b.style.borderColor = on ? "#4a7fa8" : "#333";
    };
    const sep = () => {
        const s = document.createElement("span");
        s.style.cssText = "width:1px; height:14px; background:#2f2f2f; margin:0 2px;";
        return s;
    };

    const zoomBtns = ZOOM_PRESETS.map((z) => {
        const b = btn(z === null ? "Fit" : `${z}:1`,
                      z === null
                          ? "Fit the whole frame in the panel."
                          : `${z} screen pixel${z > 1 ? "s" : ""} per source pixel. ` +
                            "1:1 is where a resolution judgement is honest.");
        b.addEventListener("click", () => { setZoom(z); });
        return b;
    });

    const magBtn = btn("smooth", "");
    magBtn.addEventListener("click", () => {
        state.magnify = state.magnify === "smooth" ? "pixels" : "smooth";
        persist(); requestTruth(true); paintChrome(); draw();
    });

    const cmpBtn = btn("wipe", "");
    cmpBtn.addEventListener("click", () => {
        state.compare = state.compare === "wipe" ? "scaled"
                      : state.compare === "scaled" ? "original" : "wipe";
        persist(); paintChrome(); draw();
    });

    const resetBtn = btn("reset", "Reset zoom, pan and wipe (or double-click the image).");
    resetBtn.addEventListener("click", () => {
        state.zoom = null; state.cx = null; state.cy = null; state.wipe = 0.5;
        state.hoverWipe = false;
        persist(); paintChrome(); scheduleView();
    });

    const framePrev = btn("◀", "Previous frame  (← key)");
    const frameNext = btn("▶", "Next frame  (→ key)");
    const frameLabel = document.createElement("span");
    frameLabel.style.cssText = "min-width:58px; text-align:center; color:#cde;";
    framePrev.addEventListener("click", () => stepFrame(-1));
    frameNext.addEventListener("click", () => stepFrame(1));

    bar.append(...zoomBtns, sep(), magBtn, cmpBtn, sep(),
               framePrev, frameLabel, frameNext, sep(), resetBtn);

    const wrap = document.createElement("div");
    wrap.style.cssText = `
        position:relative; flex:1 1 auto; min-height:80px; overflow:hidden;
        background:#000; cursor:grab;
    `;
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute; inset:0; width:100%; height:100%; display:block;";
    wrap.appendChild(canvas);
    const ctx = canvas.getContext("2d");

    const badge = document.createElement("div");
    badge.style.cssText = `
        position:absolute; right:5px; top:4px; padding:1px 5px; border-radius:3px;
        background:#000a; pointer-events:none; letter-spacing:0.04em;
    `;

    const hint = document.createElement("div");
    hint.style.cssText = `
        position:absolute; inset:0; display:flex; align-items:center;
        justify-content:center; text-align:center; padding:0 20px; color:#788;
        pointer-events:none;
    `;
    hint.textContent = "Run once to load a frame.";
    wrap.append(badge, hint);

    const status = document.createElement("div");
    status.style.cssText = `
        display:flex; align-items:center; gap:10px; padding:3px 6px;
        background:#141414; border-top:1px solid #2a2a2a; flex:0 0 auto;
        user-select:none; flex-wrap:wrap;
    `;
    const resOut = document.createElement("span");
    const lossOut = document.createElement("span");
    const viewOut = document.createElement("span");
    viewOut.style.cssText = "margin-left:auto; color:#788;";
    status.append(resOut, lossOut, viewOut);

    root.append(bar, wrap, status);

    /* ---- widget access --------------------------------------------------- */

    const W = (n) => node.widgets?.find((w) => w.name === n);
    const val = (n, dflt) => { const w = W(n); return w ? w.value : dflt; };

    function renderParams() {
        return {
            mode: String(val("mode", "factor")),
            scale: Number(val("scale", 1)),
            target: Number(val("target", 1024)),
            megapixels: Number(val("megapixels", 1)),
            filter: String(val("filter", "lanczos")),
            multiple_of: Number(val("multiple_of", 1)),
            ref_h: state.refH || null,
            ref_w: state.refW || null,
        };
    }

    function plannedSize() {
        if (!state.srcW || !state.srcH) return null;
        return planSize(state.srcH, state.srcW, renderParams());
    }

    function currentFrame() {
        return Math.max(0, Math.floor(Number(val("preview_frame", 0)) || 0));
    }

    /* ---- geometry --------------------------------------------------------
     *
     * One function decides everything about what is on screen, and the render
     * parameters are deliberately not among its inputs. That is the node's
     * central promise made structural: `scale` cannot move the picture, because
     * nothing downstream of `scale` is consulted here.
     */

    function computeView() {
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const panelW = Math.max(1, Math.round(wrap.clientWidth * dpr));
        const panelH = Math.max(1, Math.round(wrap.clientHeight * dpr));
        if (!state.srcW || !state.srcH) {
            return { panelW, panelH, outW: panelW, outH: panelH,
                     roi: [0, 0, 1, 1], drawX: 0, drawY: 0, zoom: 1, fit: true };
        }

        const fitZoom = Math.min(panelW / state.srcW, panelH / state.srcH);
        const fit = state.zoom === null;
        const zoom = fit ? fitZoom
                         : Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, state.zoom));

        // Region wanted, in source pixels, clamped to the frame. When the whole
        // frame is smaller than the panel at this zoom the region is the frame
        // and the picture is letterboxed — which is also the fit case.
        let regW = Math.min(state.srcW, panelW / zoom);
        let regH = Math.min(state.srcH, panelH / zoom);

        const cx = state.cx === null ? state.srcW / 2 : state.cx;
        const cy = state.cy === null ? state.srcH / 2 : state.cy;
        let x = Math.round(cx - regW / 2);
        let y = Math.round(cy - regH / 2);
        x = Math.max(0, Math.min(x, state.srcW - Math.ceil(regW)));
        y = Math.max(0, Math.min(y, state.srcH - Math.ceil(regH)));
        regW = Math.max(1, Math.min(Math.ceil(regW), state.srcW - x));
        regH = Math.max(1, Math.min(Math.ceil(regH), state.srcH - y));

        // The screen box the region occupies. In fit mode that is the region at
        // the fit zoom; zoomed in it is the panel (or less, at the frame edge).
        const outW = Math.max(1, Math.min(panelW, Math.round(regW * zoom)));
        const outH = Math.max(1, Math.min(panelH, Math.round(regH * zoom)));

        return {
            panelW, panelH, outW, outH, zoom, fit,
            roi: [x, y, regW, regH],
            drawX: Math.round((panelW - outW) / 2),
            drawY: Math.round((panelH - outH) / 2),
        };
    }

    function setZoom(z, anchor) {
        const before = computeView();
        // Keep the point under the cursor (or the centre) put.
        const ax = anchor ? anchor.x : before.roi[0] + before.roi[2] / 2;
        const ay = anchor ? anchor.y : before.roi[1] + before.roi[3] / 2;
        state.zoom = z;
        if (z !== null && anchor) {
            state.cx = ax; state.cy = ay;
        }
        persist();
        paintChrome();
        scheduleView();
    }

    /** Measure the element and hand back the conversion. See pointerToDevice(). */
    function pointerScale() {
        const rect = wrap.getBoundingClientRect();
        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const { graph, toDevice } = pointerToDevice(rect.width, wrap.clientWidth, dpr);
        return { rect, dpr, graph, toDevice };
    }

    /** Source-pixel coordinate under a pointer event. */
    function eventToSource(ev) {
        const v = computeView();
        const { rect, toDevice } = pointerScale();
        // Canvas device pixels, then the letterbox offset.
        const cx = (ev.clientX - rect.left) * toDevice;
        const cy = (ev.clientY - rect.top) * toDevice;
        const px = cx - v.drawX;
        const py = cy - v.drawY;
        return {
            x: v.roi[0] + (px / Math.max(1, v.outW)) * v.roi[2],
            y: v.roi[1] + (py / Math.max(1, v.outH)) * v.roi[3],
            inside: px >= 0 && py >= 0 && px <= v.outW && py <= v.outH,
            view: v, px, py, cx, cy,
        };
    }

    function wipeGeometry(v) {
        return wipeGeom(v, state.wipe, pointerScale().toDevice);
    }

    function onDivider(hit) {
        if (state.compare !== "wipe" || state.holdOriginal) return false;
        return isOnDivider(wipeGeometry(hit.view), hit.view, hit.cx, hit.cy);
    }

    /* ---- the server layer ------------------------------------------------
     *
     * Every request is stamped with the state it was made for, and a response
     * is only applied if that state is still current. Without it, changing the
     * preview frame while a render was in flight painted the OLD frame when it
     * landed and only then asked for the new one — so stepping through frames
     * showed each previous frame flash up before the one you asked for. A
     * response is evidence about a moment, and the moment can be gone by the
     * time it arrives.
     */

    let truthTimer = 0;
    let reqSeq = 0;

    /** Everything a render depends on, as one comparable string. */
    function requestKey(v) {
        const p = renderParams();
        return JSON.stringify([
            state.token, currentFrame(), v.roi, v.outW, v.outH,
            state.magnify, p.mode, p.scale, p.target, p.megapixels,
            p.filter, p.multiple_of, p.ref_h, p.ref_w,
        ]);
    }

    function scheduleView() {
        // A view change moves the region, so the picture on screen no longer
        // fits where it is drawn — unlike a parameter or frame change, there is
        // nothing honest to hold. Fall back to the whole-frame thumbnail draft
        // until the truth lands.
        dropPair();
        state.stale = false;
        draw();
        requestTruth(true);
    }

    function dropPair() {
        state.pair?.orig?.close?.();
        state.pair?.scaled?.close?.();
        state.pair = null;
    }

    function requestTruth(immediate) {
        if (truthTimer) { clearTimeout(truthTimer); truthTimer = 0; }
        // Only one render in flight at a time, but no `wantAnother` flag any
        // more: the completion handler compares keys and refires if the world
        // moved, which also covers the case where it moved twice.
        if (state.inflight) return;
        // Deliberately not track.timeout(): that keeps every id it is handed,
        // and this fires on every slider tick. One live timer, one disposer.
        truthTimer = setTimeout(fetchTruth, immediate ? 0 : TRUTH_DEBOUNCE_MS);
    }

    async function fetchTruth() {
        truthTimer = 0;
        if (!isNodeAlive(node) || !state.token) return;
        const v = computeView();
        if (!state.srcW) return;

        const key = requestKey(v);
        const seq = ++reqSeq;
        state.inflight = true;
        paintChrome();

        const body = {
            token: state.token,
            frame: currentFrame(),
            roi: v.roi,
            out_w: v.outW,
            out_h: v.outH,
            magnify: state.magnify === "pixels" ? "nearest" : "smooth",
            ...renderParams(),
        };

        /** Has anything this render depended on changed since it was sent? */
        const superseded = () =>
            seq !== reqSeq || requestKey(computeView()) !== key;

        const settle = () => {
            state.inflight = false;
            if (isNodeAlive(node) && superseded()) requestTruth(true);
        };

        let res;
        try {
            res = await fetch("/bat/rescale/render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
        } catch (e) {
            state.inflight = false;
            if (!isNodeAlive(node)) return;
            state.error = "no answer from the server";
            paintChrome();
            return;
        }

        if (!isNodeAlive(node)) { state.inflight = false; return; }
        if (!res.ok) {
            state.inflight = false;
            // 409 is the ordinary "this node has not run since the server
            // started, or the cache dropped it" — not worth shouting about.
            state.error = res.status === 409
                ? "run the node once to load a frame"
                : `preview failed (${res.status})`;
            if (res.status === 409) state.token = null;
            paintChrome();
            draw();
            return;
        }

        let info = null;
        try { info = JSON.parse(res.headers.get("X-Bat-Rescale") || "null"); }
        catch (_) {}

        let bitmap;
        try {
            bitmap = await createImageBitmap(await res.blob());
        } catch (e) {
            state.inflight = false;
            if (!isNodeAlive(node)) return;
            state.error = "could not decode the preview";
            paintChrome();
            return;
        }
        if (!isNodeAlive(node)) { bitmap.close?.(); state.inflight = false; return; }

        // Stale by arrival: throw it away rather than painting a moment that has
        // passed. settle() re-fires for the state that is current now.
        if (superseded()) {
            bitmap.close?.();
            settle();
            draw();
            return;
        }

        // One PNG, two halves: unscaled on top, rescaled below.
        const half = Math.floor(bitmap.height / 2);
        try {
            const orig = await createImageBitmap(bitmap, 0, 0, bitmap.width, half);
            const scaled = await createImageBitmap(bitmap, 0, half, bitmap.width, half);
            dropPair();
            state.pair = {
                orig, scaled,
                roi: (info?.src_rect || body.roi).map(Number),
                outW: bitmap.width, outH: half,
                frame: body.frame,
                info,
            };
        } finally {
            bitmap.close?.();
        }

        state.lastInfo = info;
        state.error = null;
        state.stale = false;
        if (info && info.frame_exact === false) state.liveBatch = false;
        settle();
        paintChrome();
        draw();
    }

    /* ---- drawing --------------------------------------------------------- */

    // Offscreen scratch for the draft's two-step scale. Kept on the closure so
    // a slider drag is not allocating a canvas per frame.
    const scratch = document.createElement("canvas");
    const sctx = scratch.getContext("2d");

    let rafPending = 0;
    function draw() {
        if (rafPending) return;
        rafPending = requestAnimationFrame(() => {
            rafPending = 0;
            if (isNodeAlive(node)) paint();
        });
    }

    function paint() {
        const v = computeView();
        if (canvas.width !== v.panelW || canvas.height !== v.panelH) {
            canvas.width = v.panelW;
            canvas.height = v.panelH;
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, v.panelW, v.panelH);

        const planned = plannedSize();
        // A render that arrived for a different region than the one now on
        // screen (the panel was resized mid-flight, say) would be drawn over the
        // wrong part of the picture. Same rule as a pan: fall back to the draft
        // rather than lie about where you are. Tolerance is 1.5px because the
        // server snaps the region outward to whole source pixels.
        const p = state.pair;
        const fits = p && p.roi && Math.abs(p.roi[0] - v.roi[0]) < 1.5
                                && Math.abs(p.roi[1] - v.roi[1]) < 1.5
                                && Math.abs(p.roi[2] - v.roi[2]) < 1.5
                                && Math.abs(p.roi[3] - v.roi[3]) < 1.5;
        const haveSource = !!(fits || state.thumb);
        hint.style.display = haveSource ? "none" : "flex";
        if (!haveSource) { paintStatus(v, planned); return; }

        const showOriginal = state.holdOriginal || state.compare === "original";
        const wipeX = state.compare === "wipe" && !showOriginal
            ? Math.round(v.drawX + v.outW * state.wipe) : null;

        if (fits) {
            // Truth. Both halves are already at the screen box and framed from
            // one region, so this is a straight blit with a clip.
            ctx.imageSmoothingEnabled = false;
            if (showOriginal) {
                ctx.drawImage(p.orig, v.drawX, v.drawY, v.outW, v.outH);
            } else if (wipeX === null) {
                ctx.drawImage(p.scaled, v.drawX, v.drawY, v.outW, v.outH);
            } else {
                ctx.save();
                ctx.beginPath();
                ctx.rect(v.drawX, v.drawY, wipeX - v.drawX, v.outH);
                ctx.clip();
                ctx.drawImage(p.orig, v.drawX, v.drawY, v.outW, v.outH);
                ctx.restore();
                ctx.save();
                ctx.beginPath();
                ctx.rect(wipeX, v.drawY, v.drawX + v.outW - wipeX, v.outH);
                ctx.clip();
                ctx.drawImage(p.scaled, v.drawX, v.drawY, v.outW, v.outH);
                ctx.restore();
            }
        } else if (state.thumb) {
            // Cold start / view change: the whole-frame thumbnail, cropped to
            // the region. Soft, and honest about being a draft.
            drawDraftFromThumb(v, showOriginal, wipeX, planned);
        }

        if (wipeX !== null) drawDivider(v);
        paintStatus(v, planned);
    }

    /**
     * The draft: crop the region out of the whole-frame thumbnail, then scale it
     * DOWN by the same ratio the render would and back UP to the screen box.
     *
     * The down-then-up round trip is the point. Drawing the thumbnail straight
     * to the box would show the source's detail at the wrong resolution — which
     * is precisely the lie this node exists to correct — so even the draft has
     * to make the round trip, just with the browser's resampler instead of the
     * node's.
     */
    function drawDraftFromThumb(v, showOriginal, wipeX, planned) {
        const img = state.thumb;
        const tScale = img.naturalWidth / Math.max(1, state.srcW);   // thumb px per source px
        const [rx, ry, rw, rh] = v.roi;
        const sx = rx * tScale, sy = ry * tScale;
        const sw = Math.max(1, rw * tScale), sh = Math.max(1, rh * tScale);

        const ratio = planned ? planned.w / Math.max(1, state.srcW) : 1;
        const dw = Math.max(1, Math.round(rw * ratio));
        const dh = Math.max(1, Math.round(rh * ratio));

        const drawOne = (destX, clipFrom, clipTo, scaledLayer) => {
            ctx.save();
            ctx.beginPath();
            ctx.rect(clipFrom, v.drawY, clipTo - clipFrom, v.outH);
            ctx.clip();
            if (!scaledLayer) {
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                ctx.drawImage(img, sx, sy, sw, sh, v.drawX, v.drawY, v.outW, v.outH);
            } else {
                if (scratch.width !== dw || scratch.height !== dh) {
                    scratch.width = dw; scratch.height = dh;
                }
                sctx.imageSmoothingEnabled = true;
                sctx.imageSmoothingQuality = "high";
                sctx.clearRect(0, 0, dw, dh);
                sctx.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh);
                ctx.imageSmoothingEnabled = state.magnify !== "pixels";
                ctx.imageSmoothingQuality = "high";
                ctx.drawImage(scratch, 0, 0, dw, dh, v.drawX, v.drawY, v.outW, v.outH);
            }
            ctx.restore();
        };

        if (showOriginal) {
            drawOne(v.drawX, v.drawX, v.drawX + v.outW, false);
        } else if (wipeX === null) {
            drawOne(v.drawX, v.drawX, v.drawX + v.outW, true);
        } else {
            drawOne(v.drawX, v.drawX, wipeX, false);
            drawOne(v.drawX, wipeX, v.drawX + v.outW, true);
        }
    }

    /**
     * The wipe divider: a hairline bar plus a grip you can actually aim at.
     *
     * The grip is drawn from `wipeGeometry()` — the same function the hit-test
     * uses — so what you can see and what you can grab are one thing by
     * construction rather than two numbers that agreed when they were written.
     */
    function drawDivider(v) {
        const g = wipeGeometry(v);
        const x = Math.round(g.x);
        const hot = state.hoverWipe || drag?.mode === "wipe";

        ctx.save();
        ctx.fillStyle = hot ? "#9cf" : "#6af";
        ctx.fillRect(x - 1, v.drawY, 2, v.outH);

        // Grip.
        const w = Math.max(9, g.gripW);
        const h = Math.max(24, g.gripH);
        const y = g.midY - h / 2;
        ctx.beginPath();
        const r = Math.min(w, h) / 2;
        ctx.moveTo(x - w / 2 + r, y);
        ctx.arcTo(x + w / 2, y, x + w / 2, y + h, r);
        ctx.arcTo(x + w / 2, y + h, x - w / 2, y + h, r);
        ctx.arcTo(x - w / 2, y + h, x - w / 2, y, r);
        ctx.arcTo(x - w / 2, y, x + w / 2, y, r);
        ctx.closePath();
        ctx.fillStyle = hot ? "#9cf" : "#6afd";
        ctx.fill();
        ctx.strokeStyle = "#0009";
        ctx.lineWidth = Math.max(1, w / 9);
        ctx.stroke();

        // Two arrows, so it reads as "drag me sideways" without a tooltip.
        ctx.strokeStyle = "#04203a";
        ctx.lineWidth = Math.max(1, w / 7);
        ctx.lineCap = "round";
        const a = w * 0.22, mid = g.midY;
        for (const dir of [-1, 1]) {
            ctx.beginPath();
            ctx.moveTo(x + dir * a * 0.4, mid - a);
            ctx.lineTo(x + dir * a * 1.3, mid);
            ctx.lineTo(x + dir * a * 0.4, mid + a);
            ctx.stroke();
        }
        ctx.restore();
    }

    function paintStatus(v, planned) {
        const src = state.srcW ? `${state.srcW}×${state.srcH}` : "—";
        if (planned) {
            const pct = (planned.w / state.srcW) * 100;
            const mp = (planned.w * planned.h) / 1e6;
            resOut.textContent =
                `${src} → ${planned.w}×${planned.h}  ${pct.toFixed(1)}%  ${mp.toFixed(2)} MP`;
            resOut.style.color = planned.w > state.srcW ? "#e0b070" : "#cde";
        } else {
            resOut.textContent = src;
            resOut.style.color = "#788";
        }

        const info = state.lastInfo;
        if (state.pair && info && "psnr" in info) {
            const rms = (Number(info.rms) || 0) * 100;
            const psnr = info.psnr === null ? "∞" : `${Number(info.psnr).toFixed(1)} dB`;
            lossOut.textContent = `round trip  Δ ${rms.toFixed(2)}% RMS · ${psnr}`;
            // Rough, and labelled as rough by being a colour rather than a
            // verdict: on grain the eye tends to go before ~48 dB, on flat CG
            // long after it.
            const p = info.psnr === null ? 99 : Number(info.psnr);
            lossOut.style.color = p >= 48 ? "#8c8" : p >= 40 ? "#cc8" : "#d88";
        } else {
            lossOut.textContent = "";
        }

        const zoomPct = v.fit ? `fit ${(v.zoom * 100).toFixed(0)}%`
                              : `${v.zoom >= 1 ? v.zoom.toFixed(v.zoom % 1 ? 2 : 0) : v.zoom.toFixed(2)}:1`;
        // Litegraph scales the whole node with a CSS transform, so at graph zoom
        // 70% a "1:1" view is 0.7 screen pixels per source pixel and a sharpness
        // judgement made on it is optimistic. Worth saying out loud on a node
        // whose entire job is that judgement.
        const graph = pointerScale().graph;
        const offGrid = !v.fit && Math.abs(graph - 1) > 0.02;
        viewOut.textContent = `${zoomPct} · ${state.magnify}`
            + (offGrid ? `  ⚠ graph at ${(100 / graph).toFixed(0)}%` : "");
        viewOut.style.color = offGrid ? "#cc8" : "#788";
        viewOut.title = offGrid
            ? "The graph is zoomed, so the node is being scaled by the canvas "
              + "and this view is not really 1:1 on your monitor. Zoom the graph "
              + "to 100% before judging sharpness."
            : "";
    }

    function paintChrome() {
        const v = computeView();
        zoomBtns.forEach((b, i) => paintBtn(b, state.zoom === ZOOM_PRESETS[i]));
        magBtn.textContent = state.magnify;
        magBtn.title = state.magnify === "smooth"
            ? "Magnify back to screen size with lanczos — the round-trip test: "
            + "what looks missing is what the downscale actually threw away. "
            + "Click for `pixels`."
            : "Magnify back with nearest, so one output pixel is one visible "
            + "block and you are looking at the real pixel grid. Click for "
            + "`smooth` to judge information rather than blockiness.";
        paintBtn(magBtn, state.magnify === "pixels");

        cmpBtn.textContent = state.compare;
        cmpBtn.title = state.compare === "wipe"
            ? "Wipe: drag the divider (or click anywhere on it). Click this for "
              + "`scaled` — the rescale alone, no divider."
            : `Showing the ${state.compare} alone — there is no divider to drag `
              + "in this mode. Click through to `wipe` to compare. Hold B for "
              + "the original at any time.";
        paintBtn(cmpBtn, state.compare !== "scaled");

        const n = Math.max(1, state.frames);
        // The asterisk is load-bearing: when the upstream batch has been freed
        // the server can only serve the frame Python shipped, and "this is
        // frame 40" is exactly the claim the artist is relying on when they
        // pick a frame to judge on.
        const exact = state.lastInfo ? state.lastInfo.frame_exact !== false : true;
        frameLabel.textContent = `${currentFrame()}${exact ? "" : "*"}/${n - 1}`;
        const scrubbable = (state.liveBatch && exact) || n === 1;
        frameLabel.style.color = scrubbable ? "#cde" : "#cc8";
        frameLabel.title = scrubbable
            ? "Frame shown in the preview (display only). ← / → to step."
            : "The upstream batch has been freed, so the only frame available "
            + "at full resolution is the one Python shipped — what you are "
            + "looking at is that frame, not this one. Re-run to move.";

        // A held picture of another frame is the one case where saying
        // "refreshing" is not enough: the number in the stepper has already
        // moved, so the badge has to name what the pixels actually are.
        const shown = state.pair ? state.pair.frame : null;
        const wrongFrame = shown !== null && shown !== currentFrame();

        let text = "", colour = "#8c8";
        if (state.error) { text = state.error; colour = "#d88"; }
        else if (!state.token) { text = "run once"; colour = "#cc8"; }
        else if (wrongFrame) { text = `frame ${shown} → ${currentFrame()}…`; colour = "#cc8"; }
        else if (state.inflight) { text = state.pair ? "refreshing" : "rendering"; colour = "#cc8"; }
        else if (!state.pair) { text = "draft"; colour = "#cc8"; }
        else if (state.stale) { text = "stale"; colour = "#cc8"; }
        else { text = "1:1 truth"; colour = "#8c8"; }
        badge.textContent = text;
        badge.style.color = colour;
        badge.style.display = text ? "block" : "none";

        hint.textContent = state.error && !state.pair && !state.thumb
            ? state.error : "Run once to load a frame.";
        void v;
    }

    function persist() {
        saveJson(viewKey(node), {
            zoom: state.zoom, cx: state.cx, cy: state.cy,
            wipe: state.wipe, compare: state.compare, magnify: state.magnify,
        });
    }

    /* ---- interaction ----------------------------------------------------- */

    let drag = null;

    track.listener(wrap, "pointerdown", (ev) => {
        if (ev.button !== 0 && ev.button !== 1) return;
        ev.stopPropagation();
        root.focus({ preventScroll: true });
        const hit = eventToSource(ev);
        const v = hit.view;
        const grabbing = ev.button === 0 && onDivider(hit);
        drag = {
            mode: grabbing ? "wipe" : "pan",
            x: ev.clientX, y: ev.clientY,
            cx: state.cx === null ? state.srcW / 2 : state.cx,
            cy: state.cy === null ? state.srcH / 2 : state.cy,
            zoom: v.zoom,
        };
        // Take the wipe to the pointer on press rather than only on the
        // subsequent move, so a click anywhere on the bar (or a click that
        // lands a pixel or two off it) does what it looks like it should.
        if (grabbing) {
            state.wipe = clampWipe(hit.px / Math.max(1, v.outW));
            draw();
        }
        wrap.style.cursor = grabbing ? "col-resize" : "grabbing";
        try { wrap.setPointerCapture(ev.pointerId); } catch (_) {}
    });

    track.listener(wrap, "pointermove", (ev) => {
        if (!drag) {
            // Cursor feedback for the divider without committing to a drag.
            // Gated on having a picture rather than on having a full render —
            // the divider is drawn over the draft too, so it has to be grabbable
            // there.
            if (state.compare === "wipe" && (state.pair || state.thumb)) {
                const near = onDivider(eventToSource(ev));
                wrap.style.cursor = near ? "col-resize" : "grab";
                if (near !== state.hoverWipe) { state.hoverWipe = near; draw(); }
            }
            return;
        }
        ev.stopPropagation();
        const { toDevice } = pointerScale();
        if (drag.mode === "wipe") {
            const hit = eventToSource(ev);
            state.wipe = clampWipe(hit.px / Math.max(1, hit.view.outW));
            draw();
            return;
        }
        // Pan. In fit mode there is nothing to pan, so a drag moves nothing
        // rather than jumping the view.
        if (state.zoom === null) return;
        // Same conversion as the hit-test: a drag measured in displayed pixels
        // has to come back to layout device pixels before it can be divided by
        // the view zoom, or the picture lags or outruns the pointer under graph
        // zoom.
        const dx = (ev.clientX - drag.x) * toDevice / drag.zoom;
        const dy = (ev.clientY - drag.y) * toDevice / drag.zoom;
        state.cx = drag.cx - dx;
        state.cy = drag.cy - dy;
        dropPair();             // the region moved; the old render no longer fits
        draw();
    });

    const endDrag = (ev) => {
        if (!drag) return;
        const wasPan = drag.mode === "pan";
        drag = null;
        wrap.style.cursor = "grab";
        try { wrap.releasePointerCapture?.(ev.pointerId); } catch (_) {}
        persist();
        if (wasPan) requestTruth(true); else draw();
    };
    track.listener(wrap, "pointerleave", () => {
        if (state.hoverWipe) { state.hoverWipe = false; draw(); }
    });
    track.listener(wrap, "pointerup", endDrag);
    track.listener(wrap, "pointercancel", endDrag);

    track.listener(wrap, "wheel", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const hit = eventToSource(ev);
        const base = hit.view.zoom;
        const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM,
            base * (ev.deltaY < 0 ? 1.25 : 1 / 1.25)));
        // Snap to fit rather than sitting a hair below it — that band is where
        // the picture is smaller than the panel and the pan does nothing.
        const fitZoom = Math.min(hit.view.panelW / state.srcW,
                                 hit.view.panelH / state.srcH);
        state.zoom = next <= fitZoom * 1.02 ? null : next;
        if (state.zoom !== null && hit.inside) { state.cx = hit.x; state.cy = hit.y; }
        persist(); paintChrome(); scheduleView();
    }, { passive: false });

    track.listener(wrap, "dblclick", (ev) => {
        ev.stopPropagation();
        state.zoom = state.zoom === null ? 1 : null;
        persist(); paintChrome(); scheduleView();
    });

    track.listener(root, "keydown", (ev) => {
        const k = ev.key;
        if (k === "ArrowLeft" || k === "ArrowRight") {
            stepFrame(k === "ArrowLeft" ? -1 : 1);
        } else if (k === "b" || k === "B") {
            if (state.holdOriginal) return;
            state.holdOriginal = true;
            draw();
        } else if (k === "1") { setZoom(1); }
        else if (k === "2") { setZoom(2); }
        else if (k === "f" || k === "F") { setZoom(null); }
        else { return; }
        ev.preventDefault();
        ev.stopPropagation();
    });
    track.listener(root, "keyup", (ev) => {
        if ((ev.key === "b" || ev.key === "B") && state.holdOriginal) {
            state.holdOriginal = false;
            draw();
        }
    });
    // Releasing B outside the node would otherwise leave it stuck on.
    track.listener(window, "blur", () => {
        if (state.holdOriginal) { state.holdOriginal = false; draw(); }
    });

    function stepFrame(delta) {
        const w = W("preview_frame");
        if (!w) return;
        const n = Math.max(1, state.frames);
        const next = Math.max(0, Math.min(n - 1, currentFrame() + delta));
        if (next === currentFrame()) return;
        w.value = next;
        try { w.callback?.(next); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
        // The pair is deliberately NOT dropped. Dropping it fell back to the
        // whole-frame thumbnail, which is the frame PYTHON shipped — so asking
        // for frame 6 flashed the anchor frame, i.e. a different picture
        // entirely, before frame 6 arrived. Holding the last real render
        // instead means the region and the framing stay put and only the
        // content updates, the way a video scrubber behaves. The badge and the
        // frame counter say which frame is actually on screen.
        paintChrome();
        draw();
        requestTruth(true);
    }

    /* ---- widget watching -------------------------------------------------
     *
     * A parameter change keeps the picture that is on screen and marks it
     * stale, rather than dropping back to the draft: the region has not moved,
     * so what is drawn is still the right part of the right frame, just one
     * scale factor out of date. Flicking between two sharpnesses mid-drag is
     * worse than a badge saying "stale" for 80 ms.
     */

    node._batRescaleWatch = () => {
        for (const name of ["mode", "scale", "target", "megapixels", "filter",
                            "multiple_of"]) {
            const w = W(name);
            if (!w || w._batRescaleHooked) continue;
            w._batRescaleHooked = true;
            const orig = w.callback;
            w.callback = function (v) {
                try { orig?.call(this, v); } catch (_) {}
                state.stale = true;
                paintChrome();
                draw();
                requestTruth(false);
            };
        }
        const pf = W("preview_frame");
        if (pf && !pf._batRescaleHooked) {
            pf._batRescaleHooked = true;
            const orig = pf.callback;
            pf.callback = function (v) {
                try { orig?.call(this, v); } catch (_) {}
                // Same reasoning as stepFrame(): hold the picture, don't fall
                // back to the anchor thumbnail. Debounced rather than
                // immediate because dragging a litegraph number widget emits a
                // value per pixel of mouse travel.
                paintChrome();
                draw();
                requestTruth(false);
            };
        }
    };

    /* ---- ingest ---------------------------------------------------------- */

    async function loadThumb(b64) {
        if (!b64) return;
        const img = new Image();
        await new Promise((res, rej) => {
            img.onload = res; img.onerror = rej;
            img.src = `data:image/jpeg;base64,${b64}`;
        });
        state.thumb = img;
    }

    node._batRescaleIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        state.token = one(msg.token) || state.token;
        state.frames = Number(one(msg.frames)) || 1;
        state.srcW = Number(one(msg.src_w)) || state.srcW;
        state.srcH = Number(one(msg.src_h)) || state.srcH;
        state.liveBatch = true;      // fresh run: the batch is in Comfy's cache
        state.error = null;
        const thumb = one(msg.thumb);
        if (thumb) {
            await loadThumb(thumb);
            // Only the JPEG and the token go to localStorage: enough for a
            // reopened workflow to show something and to ask the server whether
            // its frame is still cached, and small enough not to evict the
            // other BAT nodes' caches out of the ~5MB origin budget.
            saveJson(cacheKey(node), {
                thumb, token: state.token, frames: state.frames,
                src_w: state.srcW, src_h: state.srcH,
            });
        }
        dropPair();
        paintChrome();
        draw();
        requestTruth(true);
    };

    node._batRescaleRestore = async () => {
        // A full-res draft replayed from this session's last run beats the
        // cached thumbnail; both this decode and the token revalidation below
        // would otherwise land on top of it.
        if (batPreviewWillReplay(node)) { paintChrome(); return; }
        const cached = loadJson(cacheKey(node), null);
        if (!cached) { paintChrome(); return; }
        state.frames = Number(cached.frames) || 1;
        state.srcW = Number(cached.src_w) || 0;
        state.srcH = Number(cached.src_h) || 0;
        try { await loadThumb(cached.thumb); } catch (_) {}
        paintChrome();
        draw();

        // Is the server still holding the frame this token points at? If so the
        // viewer comes back fully — truth layer included — with no re-run.
        if (!cached.token) return;
        try {
            const r = await fetch(`/bat/rescale/info?token=${encodeURIComponent(cached.token)}`);
            const j = await r.json();
            if (!isNodeAlive(node) || !j?.ok) return;
            state.token = cached.token;
            state.frames = Number(j.frames) || state.frames;
            state.srcW = Number(j.src_w) || state.srcW;
            state.srcH = Number(j.src_h) || state.srcH;
            state.liveBatch = !!j.live_batch;
            paintChrome();
            requestTruth(true);
        } catch (_) { /* server restarted: the draft is what we have */ }
    };

    /* ---- reference input ------------------------------------------------- */

    // match_reference needs the reference plate's size to show the output
    // resolution before a run. Nothing in the frontend knows an image's size,
    // so the readout says so rather than guessing.
    node._batRescaleRefSize = (h, w) => { state.refH = h; state.refW = w; draw(); };

    /* ---- keep up with the panel ------------------------------------------ */

    track.observer(new ResizeObserver(() => {
        if (!isNodeAlive(node)) return;
        draw();
        // The screen box changed, so the region did too.
        requestTruth(false);
    }), wrap);

    track.dispose(() => {
        if (truthTimer) { clearTimeout(truthTimer); truthTimer = 0; }
        dropPair();
        state.thumb = null;
    });

    paintChrome();
    return root;
}


/* ==========================================================================
 * Registration
 * ========================================================================== */

app.registerExtension({
    name: "Bat_Rescale",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        // A graph reload (Ctrl+Z is one) destroys and rebuilds every node, so
        // replay the last run's preview payload into the new instance.
        batReplayLastExecution(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = buildViewer(this);
            addBatDOMWidget(this, "bat_rescale_viewer", "bat_rescale_viewer", el, {
                minWidth: 380, height: 460, growable: true,
            });
            this._batRescaleWatch?.();
            clampNodeSize(this, 380, 520);
            // Deferred: node.id is only final once litegraph has finished
            // constructing, and every localStorage key here is scoped by it.
            setTimeout(() => this._batRescaleRestore?.(), 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && this._batRescaleIngest) this._batRescaleIngest(message);
            return r;
        };
    },
});
