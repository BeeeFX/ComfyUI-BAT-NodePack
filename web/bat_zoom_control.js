/**
 * BAT — shared display-zoom + pan control for the on-node canvas editors
 * (Bat_Crop, Bat_AnimatedCrop, Bat_Roto).
 *
 * This is a DISPLAY-only transform: it changes how much of the image (and
 * the area around it) the preview canvas shows, so the artist can pull
 * back — or zoom in and pan — to work on a crop / roto that extends past
 * the frame edge. It never touches the node's output — the backend only
 * ever sees the widget values / state JSON, not this view transform.
 *
 * Contract with the host editor:
 *   • The editor keeps `state.dispZoom` (1 = fit) and `state.panX/panY`
 *     (display-pixel offset from centred, default 0). The editor's
 *     recomputeDisplay() must:
 *         const fit = Math.min(cw/imgW, ch/imgH);
 *         state.dispScale = fit * (state.dispZoom || 1);
 *         state.offX = (cw - imgW*state.dispScale)/2 + (state.panX || 0);
 *         state.offY = (ch - imgH*state.dispScale)/2 + (state.panY || 0);
 *     Every c2d/d2c-based draw and hit-test then works unchanged.
 *   • attachZoomControl() seeds those fields, overlays a small −/%/+
 *     control on `wrap`, and wires wheel-zoom, middle-mouse pan, and
 *     double-click reset.
 *   • After recomputeDisplay() runs, the editor leaves the resulting
 *     state.dispScale/offX/offY on `state`; this control reads them back
 *     to anchor zoom-toward-cursor and to convert pan deltas. So onChange()
 *     MUST run the editor's render()/recomputeDisplay() before this control
 *     next needs those values — which it does, since render() is what
 *     onChange points at.
 *
 * Zoom range is 0.2×–4×. Below 1× you see letterboxed space around the
 * image (what you want for out-of-frame crop/roto); above 1× you zoom in.
 * Middle-mouse drag pans; the wheel zooms toward the cursor.
 *
 * A middle-click inside an on-node DOM widget would otherwise paste the user's
 * clipboard nodes onto the graph on Linux/X11; this control marks its canvas and
 * arms the shared guard in bat_paste_guard.js, which handles the rest. The
 * returned `destroy()` is a no-op kept for callers' convenience.
 *
 * `state.imgW` / `state.imgH` are REQUIRED, not optional: zoomTo() reads them
 * to anchor zoom-toward-cursor. A host that tracks its image size under other
 * names still has to mirror it onto these two, or the wheel will drift.
 */

import { armBatPasteGuard, markBatWidget } from "./bat_paste_guard.js";

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 4.0;
// Multiplicative step so each click / wheel-notch feels even across the
// range (0.5→0.6 and 2.0→2.4 are the same *ratio*).
const STEP = 1.2;

const clampZoom = (z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));

/* ── Middle-drag pan under Nodes 2.0 ────────────────────────────────────────
 *
 * On a Vue node the canvas's own middle-button listeners never run:
 * TransformPane forwards every middle-button pointerdown/move/up to the graph
 * canvas in the CAPTURE phase and stops it there (GraphCanvas.vue
 * forwardPanEvent → useCanvasInteractions.forwardEventToCanvas), so a
 * middle-drag over the picture panned the whole graph. The only listener that
 * runs before TransformPane's is one on `window`, capture phase. So: one such
 * router per page (not per node — a per-instance window listener is the
 * accumulating leak bat_lifecycle.js exists to prevent), which finds the
 * zoom-controlled canvas under the press through a marker attribute + WeakMap,
 * drives that instance's pan, and stops the pointerdown/move/up sequence
 * while the pan is live. It only claims presses inside a Vue node (`.lg-node`):
 * under Nodes 1.0, and in fullscreen, nothing sits in front and the canvas's
 * own listeners handle it exactly as before.
 */
const ZOOM_CANVAS_ATTR = "data-bat-zoom-canvas";
const PANS = new WeakMap();        // canvas → { begin, move, end }
let routedPan = null;              // { pan, pointerId } while a routed drag is live
let routerInstalled = false;

function installPanRouter() {
    if (routerInstalled) return;
    // No window (a headless harness, a worker): there is no TransformPane to
    // route around either, and the canvas's own listeners still pan.
    if (typeof window === "undefined" || typeof window.addEventListener !== "function") return;
    routerInstalled = true;
    const claim = (e) => { e.preventDefault(); e.stopPropagation(); };
    try {
        window.addEventListener("pointerdown", (e) => {
            if (e.button !== 1 || routedPan) return;
            const c = e.target?.closest?.(`[${ZOOM_CANVAS_ATTR}]`);
            if (!c || !c.closest(".lg-node")) return;
            const pan = PANS.get(c);
            if (!pan) return;
            routedPan = { pan, pointerId: e.pointerId };
            pan.begin(e);
            claim(e);
        }, true);
        window.addEventListener("pointermove", (e) => {
            if (!routedPan || e.pointerId !== routedPan.pointerId) return;
            routedPan.pan.move(e);
            claim(e);
        }, true);
        const up = (e) => {
            if (!routedPan || e.pointerId !== routedPan.pointerId) return;
            const { pan } = routedPan;
            routedPan = null;
            pan.end(e);
            claim(e);
        };
        window.addEventListener("pointerup", up, true);
        window.addEventListener("pointercancel", up, true);
    } catch (e) {
        globalThis.console?.error?.("[BAT.zoom] could not install the pan router:", e);
        routerInstalled = false;
    }
}

/**
 * @param {object}   opts
 * @param {HTMLElement} opts.wrap      canvas wrapper (position:relative)
 * @param {HTMLCanvasElement} opts.canvas  the editor canvas (for wheel/pan/dblclick)
 * @param {object}   opts.state        editor state object (gets .dispZoom/.panX/.panY)
 * @param {Function} opts.onChange     called after any view change (→ render)
 * @param {string}   [opts.corner]     "tl"|"tr"|"bl"|"br" (default "bl")
 * @param {HTMLElement} [opts.scope]   the element that holds keyboard focus
 *                                     while the artist works in the editor
 *                                     (default `wrap`) — see below
 * @returns {{setZoom:Function, getZoom:Function, resetView:Function, refresh:Function}}
 */
export function attachZoomControl({ wrap, canvas, state, onChange, corner = "bl", scope = null }) {
    // Nodes 2.0: TransformPane forwards every wheel event to the graph in the
    // CAPTURE phase (GraphCanvas.vue → useCanvasInteractions.forwardEventToCanvas)
    // and stops it there, so the canvas listener below never ran — the wheel
    // zoomed the graph instead. The one exemption is a target inside a
    // `[data-capture-wheel="true"]` element that contains the focused element.
    // So mark the element the editor focuses: after one click in the editor the
    // wheel is ours, before it the graph still pans over the node, which is the
    // frontend's own convention. Inert under Nodes 1.0 and inert if nothing
    // inside `scope` ever takes focus — only an editor that focuses itself
    // gains wheel zoom here; one that doesn't would have to take focus first,
    // and then stop its keys reaching core's Delete binding (see bat_roto.js).
    try { (scope || wrap)?.setAttribute?.("data-capture-wheel", "true"); } catch (_) {}

    if (typeof state.dispZoom !== "number" || !isFinite(state.dispZoom)) {
        state.dispZoom = 1;
    }
    if (typeof state.panX !== "number" || !isFinite(state.panX)) state.panX = 0;
    if (typeof state.panY !== "number" || !isFinite(state.panY)) state.panY = 0;

    // A middle-drag here is a deliberate pan, so mark the canvas explicitly
    // rather than relying on an ancestor having been marked. Hosts that route
    // through addBatDOMWidget already are, but the control is also usable on an
    // element that isn't. See bat_paste_guard.js.
    markBatWidget(canvas);

    const pos = {
        tl: "left:8px; top:6px;",
        tr: "right:8px; top:6px;",
        bl: "left:8px; bottom:6px;",
        br: "right:8px; bottom:6px;",
    }[corner] || "left:8px; bottom:6px;";

    const box = document.createElement("div");
    box.style.cssText = `
        position:absolute; ${pos} display:flex; align-items:center; gap:2px;
        background:rgba(0,0,0,0.55); border:1px solid #2a2f37; border-radius:4px;
        padding:1px 2px; font:11px monospace; color:#cdd; z-index:6;
        user-select:none;
    `;

    const mkBtn = (txt, title) => {
        const b = document.createElement("button");
        b.textContent = txt; b.title = title;
        b.style.cssText = `
            background:none; border:none; color:#cdd; cursor:pointer;
            font:13px monospace; line-height:1; padding:1px 5px; border-radius:3px;
        `;
        b.onmouseover = () => { b.style.background = "rgba(76,158,255,0.2)"; };
        b.onmouseout  = () => { b.style.background = "none"; };
        // Don't let clicks fall through to the canvas / node drag.
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };

    const outBtn = mkBtn("−", "Zoom out (scroll down)");
    const readout = document.createElement("span");
    readout.style.cssText = "min-width:34px; text-align:center; cursor:pointer;";
    readout.title = "Display zoom — click to reset view (double-click canvas also resets). "
                  + "Middle-mouse drag to pan.";
    const inBtn = mkBtn("+", "Zoom in (scroll up)");
    // Explicit reset. The percentage readout has always been click-to-reset and
    // the canvas has always been double-click-to-reset, but neither is
    // discoverable — nothing about a number suggests it is a button. A visible
    // control costs 18 pixels and saves the guess.
    const resetBtn = mkBtn("⤾", "Reset view — back to fit, centred (100%).\n"
                                + "Also: click the percentage, or double-click the image.");
    box.append(outBtn, readout, inBtn, resetBtn);
    wrap.appendChild(box);

    function refresh() {
        readout.textContent = `${Math.round(state.dispZoom * 100)}%`;
    }

    // Cursor position in CSS-pixel canvas coords (matches the editors'
    // own localMouse() convention: getBoundingClientRect scaled by the
    // client/box ratio so LiteGraph graph-zoom doesn't skew it).
    function cursorCanvas(e) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (canvas.clientWidth  / r.width),
            y: (e.clientY - r.top)  * (canvas.clientHeight / r.height),
        };
    }

    // Zoom to `nz`, keeping the image point currently under (ax, ay) —
    // canvas CSS px — fixed on screen (zoom-toward-cursor).
    //
    // recomputeDisplay() will rebuild offX from an auto-centre term plus
    // panX:   offX = centre + panX,  centre = (cw - imgW*scale)/2.
    // We don't have cw here, but centre = offX_old - panX_old before the
    // zoom, and centre only shifts by the change in imgW*scale/2 (cw is
    // constant). So:
    //   centre_new = centre_old + (oldScale - newScale) * imgW/2
    // and we want the anchor's image point fixed:
    //   offX_new = ax - imgX*newScale
    //   panX_new = offX_new - centre_new
    function zoomTo(nz, ax, ay) {
        nz = clampZoom(nz);
        if (nz === state.dispZoom) { refresh(); return; }

        const oldScale = state.dispScale || 1;
        const oldOffX = state.offX || 0;
        const oldOffY = state.offY || 0;
        const imgW = state.imgW || 0;
        const imgH = state.imgH || 0;

        // Image-space point under the anchor before the zoom.
        const imgX = (ax - oldOffX) / oldScale;
        const imgY = (ay - oldOffY) / oldScale;

        const newScale = oldScale * (nz / state.dispZoom);
        state.dispZoom = nz;

        const centreXold = oldOffX - (state.panX || 0);
        const centreYold = oldOffY - (state.panY || 0);
        const centreXnew = centreXold + (oldScale - newScale) * imgW / 2;
        const centreYnew = centreYold + (oldScale - newScale) * imgH / 2;

        state.panX = (ax - imgX * newScale) - centreXnew;
        state.panY = (ay - imgY * newScale) - centreYnew;

        refresh();
        onChange?.();
    }

    function setZoom(z) { zoomTo(z); }      // centre-anchored (button path)

    function resetView() {
        state.dispZoom = 1;
        state.panX = 0;
        state.panY = 0;
        refresh();
        onChange?.();
    }

    // Buttons zoom toward the canvas centre (no cursor context).
    outBtn.onclick = (e) => {
        e.stopPropagation();
        const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
        zoomTo(state.dispZoom / STEP, cx, cy);
    };
    inBtn.onclick = (e) => {
        e.stopPropagation();
        const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
        zoomTo(state.dispZoom * STEP, cx, cy);
    };
    readout.onclick = (e) => { e.stopPropagation(); resetView(); };
    resetBtn.onclick = (e) => { e.stopPropagation(); resetView(); };

    // Scroll-wheel zooms toward the cursor (and swallows the event so the
    // LiteGraph canvas underneath doesn't also zoom the whole graph).
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const c = cursorCanvas(e);
        zoomTo(e.deltaY < 0 ? state.dispZoom * STEP : state.dispZoom / STEP, c.x, c.y);
    }, { passive: false });


    // ── Middle-mouse pan ─────────────────────────────────────────────
    // Captured in the CAPTURE phase so it runs before the editor's own
    // pointerdown (draw / crop / marquee) and can stop it — the editors
    // don't filter on e.button, so without this a middle-click would also
    // start a draw. Left/right buttons fall straight through untouched.
    let pan = null;
    const beginPan = (e) => {
        const r = canvas.getBoundingClientRect();
        pan = {
            startX: e.clientX, startY: e.clientY,
            baseX: state.panX || 0, baseY: state.panY || 0,
            // display px per client px (LiteGraph graph-zoom compensation)
            sx: canvas.clientWidth  / Math.max(1, r.width),
            sy: canvas.clientHeight / Math.max(1, r.height),
            prevCursor: canvas.style.cursor,
        };
        canvas.style.cursor = "grabbing";
        armBatPasteGuard();
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    };
    const movePan = (e) => {
        if (!pan) return;
        state.panX = pan.baseX + (e.clientX - pan.startX) * pan.sx;
        state.panY = pan.baseY + (e.clientY - pan.startY) * pan.sy;
        onChange?.();
    };
    const finishPan = (e) => {
        if (!pan) return;
        // Re-arm: the paste arrives on release, not on press.
        armBatPasteGuard();
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        canvas.style.cursor = pan.prevCursor || "";
        pan = null;
    };
    // Nodes 2.0 reaches the three above through the window router instead.
    try { canvas.setAttribute(ZOOM_CANVAS_ATTR, ""); } catch (_) {}
    PANS.set(canvas, { begin: beginPan, move: movePan, end: finishPan });
    installPanRouter();

    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;         // middle button only
        e.preventDefault();
        e.stopPropagation();
        beginPan(e);
    }, true);
    canvas.addEventListener("pointermove", (e) => {
        if (!pan) return;
        e.preventDefault();
        e.stopPropagation();
        movePan(e);
    }, true);
    const endPan = (e) => {
        if (!pan) return;
        e.stopPropagation();
        finishPan(e);
    };
    canvas.addEventListener("pointerup", endPan, true);
    canvas.addEventListener("pointercancel", endPan, true);
    // Some browsers emit the middle-button "paste"/autoscroll on auxclick —
    // suppress it over the canvas so a middle-click never triggers it.
    canvas.addEventListener("auxclick", (e) => {
        if (e.button === 1) { e.preventDefault(); e.stopPropagation(); }
    }, true);


    // Double-click empty canvas → reset view (zoom + pan). Resetting the
    // view doesn't disturb the crop/roto data, only the display.
    canvas.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetView();
    });

    refresh();

    /**
     * Nothing to tear down.
     *
     * Every listener this module installs is on `canvas` or on its own buttons,
     * all of which die with the widget's DOM. The X11 paste guard used to live
     * here and did need releasing; it now lives in bat_paste_guard.js as a
     * page-wide singleton, precisely because the editors that predate any
     * teardown (Crop, Animated Crop, Roto) would never have released it.
     *
     * Kept as a no-op so hosts can call it unconditionally, and so this stays
     * the obvious place to hang a real teardown if one is ever needed.
     */
    function destroy() {}

    return { setZoom, getZoom: () => state.dispZoom, resetView, refresh, destroy };
}
