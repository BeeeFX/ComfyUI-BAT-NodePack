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
 */

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 4.0;
// Multiplicative step so each click / wheel-notch feels even across the
// range (0.5→0.6 and 2.0→2.4 are the same *ratio*).
const STEP = 1.2;

const clampZoom = (z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));

/**
 * @param {object}   opts
 * @param {HTMLElement} opts.wrap      canvas wrapper (position:relative)
 * @param {HTMLCanvasElement} opts.canvas  the editor canvas (for wheel/pan/dblclick)
 * @param {object}   opts.state        editor state object (gets .dispZoom/.panX/.panY)
 * @param {Function} opts.onChange     called after any view change (→ render)
 * @param {string}   [opts.corner]     "tl"|"tr"|"bl"|"br" (default "bl")
 * @returns {{setZoom:Function, getZoom:Function, resetView:Function, refresh:Function}}
 */
export function attachZoomControl({ wrap, canvas, state, onChange, corner = "bl" }) {
    if (typeof state.dispZoom !== "number" || !isFinite(state.dispZoom)) {
        state.dispZoom = 1;
    }
    if (typeof state.panX !== "number" || !isFinite(state.panX)) state.panX = 0;
    if (typeof state.panY !== "number" || !isFinite(state.panY)) state.panY = 0;

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
    box.append(outBtn, readout, inBtn);
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
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 1) return;         // middle button only
        e.preventDefault();
        e.stopPropagation();
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
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    }, true);
    canvas.addEventListener("pointermove", (e) => {
        if (!pan) return;
        e.preventDefault();
        e.stopPropagation();
        state.panX = pan.baseX + (e.clientX - pan.startX) * pan.sx;
        state.panY = pan.baseY + (e.clientY - pan.startY) * pan.sy;
        onChange?.();
    }, true);
    const endPan = (e) => {
        if (!pan) return;
        e.stopPropagation();
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        canvas.style.cursor = pan.prevCursor || "";
        pan = null;
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
    return { setZoom, getZoom: () => state.dispZoom, resetView, refresh };
}
