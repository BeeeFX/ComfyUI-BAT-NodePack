/**
 * Bat_Roto — multi-shape bezier roto editor with timeline + keyframes.
 *
 * Persistent state lives in a hidden STRING widget called `state`,
 * shaped as:
 *
 *   { shapes: [{ id, name, color, opacity, invert, feather, keyframes }] }
 *
 * where keyframes is { "<frame>": [point, point, ...] } and each
 * point is [x, y, hxi, hyi, hxo, hyo] in absolute pixel coordinates
 * of the underlying image.
 *
 * Interactions (Nuke-style):
 *   - Click on empty canvas: start a new shape (or extend the active
 *     one if it isn't closed yet).
 *   - Click on the first point of the active open shape: close it.
 *   - Click a point: select it. Drag: move it.
 *   - Drag a tangent handle: shape the curve.
 *   - Alt + click point: delete that point.
 *   - Shift + click on an edge: insert a new point at that edge.
 *   - Esc: deselect / cancel new-shape mode.
 *
 * Keyframes:
 *   - Move any point while Auto-key is ON → keyframe at the current
 *     frame for the active shape.
 *   - Move any point while Auto-key is OFF → only changes the current
 *     keyframe (if there is one at this frame), or creates one.
 *   - "K" key adds a manual keyframe at the current frame.
 *   - Delete a keyframe by right-clicking its marker on the timeline.
 *
 * Playback:
 *   - Space: play/pause through the frame range.
 *   - ← / →: prev / next frame.
 *
 * The Python backend reads the same state JSON, interpolates between
 * keyframes for each frame, and rasterises the bezier shapes into a
 * (N, H, W) MASK tensor.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { batTrack, batNodeCacheKey } from "./bat_lifecycle.js";
import { api } from "../../scripts/api.js";
import { attachZoomControl } from "./bat_zoom_control.js";

const NODE_TYPE = "Bat_Roto";

// ── tiny utilities ─────────────────────────────────────────────────────────
const uid = () => Math.random().toString(36).slice(2, 10);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function defaultState() {
    return { shapes: [] };
}

// ── localStorage preview cache ──────────────────────────────────────────
// Keyed by node id so reopening a workflow restores the bg thumbnail
// (and the original image dimensions) immediately, before the artist
// has to hit Run. Per-machine by design — we don't bloat the workflow
// JSON with base64 thumbnails, and a workflow shared with a colleague
// just shows a black bg until they Run on their side.
function _previewCacheKey(node) {
    // Workflow-scoped: keying on node.id ALONE collided across graphs — opening
    // another workflow whose node 14 is also this node type restored the wrong
    // shot's plate (at the wrong imgW/imgH, so shapes drew against a bogus
    // reference). batNodeCacheKey folds in the workflow identity.
    return batNodeCacheKey(app, "bat_roto_preview", node);
}
function _saveCachedPreview(node, data) {
    try { localStorage.setItem(_previewCacheKey(node), JSON.stringify(data)); }
    catch (_) { /* quota exceeded or storage disabled — fine, just skip */ }
}
function _loadCachedPreview(node) {
    try {
        const raw = localStorage.getItem(_previewCacheKey(node));
        return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
}

const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

/**
 * De Casteljau subdivision of a cubic bezier at parameter t.
 *
 * Returns the new control points needed to insert a new anchor between
 * two existing ones WITHOUT changing the visible curve. Inputs:
 *   p0 = previous anchor xy
 *   p1 = previous anchor's OUT handle xy
 *   p2 = next anchor's IN handle xy
 *   p3 = next anchor xy
 *
 * Returns {
 *   prevHout:  new OUT handle for the previous anchor,
 *   newHin:    IN  handle for the new anchor,
 *   newPoint:  xy of the new anchor (the curve sample at t),
 *   newHout:   OUT handle for the new anchor,
 *   nextHin:   new IN handle for the next anchor,
 * }
 *
 * Because linear interpolation of bezier control points at fixed t
 * commutes with bezier sampling, applying this subdivision to every
 * keyframe at the SAME t guarantees that the click position the user
 * picked on an interpolated frame lands exactly there on screen —
 * the in-between animation is preserved.
 */
function deCasteljauSubdivide(p0, p1, p2, p3, t) {
    const q0 = lerp2(p0, p1, t);
    const q1 = lerp2(p1, p2, t);
    const q2 = lerp2(p2, p3, t);
    const r0 = lerp2(q0, q1, t);
    const r1 = lerp2(q1, q2, t);
    const s  = lerp2(r0, r1, t);
    return {
        prevHout: q0,
        newHin:   r0,
        newPoint: s,
        newHout:  r1,
        nextHin:  q2,
    };
}

/**
 * Find the edge on a shape (at its CURRENT interpolated point list)
 * closest to a query position, and return {edgeIndex, t, distance}.
 *
 * Edge `i` runs from points[i] → points[(i+1) % N]. We sample each edge
 * coarsely then refine around the best sample. Used for the
 * "click anywhere on a closed shape to add a point" interaction.
 */
function closestEdgeAndT(points, qx, qy, coarseN = 16, refineN = 12) {
    if (!points || points.length < 2) return null;
    let bestIdx = 0, bestT = 0, bestDist = Infinity;
    for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        const p0 = [a[0], a[1]];
        const p1 = [a[4], a[5]];
        const p2 = [b[2], b[3]];
        const p3 = [b[0], b[1]];
        // Coarse scan.
        let edgeBestT = 0, edgeBestDist = Infinity;
        for (let s = 0; s <= coarseN; s++) {
            const t = s / coarseN;
            const mt = 1 - t;
            const x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0];
            const y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1];
            const d = (x - qx) * (x - qx) + (y - qy) * (y - qy);
            if (d < edgeBestDist) { edgeBestDist = d; edgeBestT = t; }
        }
        // Refine around the best coarse sample.
        const lo = Math.max(0, edgeBestT - 1 / coarseN);
        const hi = Math.min(1, edgeBestT + 1 / coarseN);
        for (let s = 0; s <= refineN; s++) {
            const t = lo + (hi - lo) * (s / refineN);
            const mt = 1 - t;
            const x = mt*mt*mt*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t*t*t*p3[0];
            const y = mt*mt*mt*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t*t*t*p3[1];
            const d = (x - qx) * (x - qx) + (y - qy) * (y - qy);
            if (d < edgeBestDist) { edgeBestDist = d; edgeBestT = t; }
        }
        if (edgeBestDist < bestDist) {
            bestDist = edgeBestDist; bestIdx = i; bestT = edgeBestT;
        }
    }
    return { edgeIndex: bestIdx, t: bestT, distance: Math.sqrt(bestDist) };
}

/**
 * Subdivide ONE keyframe's point list at (edgeIndex, t) using De
 * Casteljau, returning a NEW point list with the inserted anchor.
 * Existing point objects are not mutated.
 */
function subdivideKeyframeAt(points, edgeIndex, t) {
    const i = edgeIndex;
    const j = (i + 1) % points.length;
    const a = points[i];
    const b = points[j];
    const sub = deCasteljauSubdivide(
        [a[0], a[1]], [a[4], a[5]],
        [b[2], b[3]], [b[0], b[1]],
        t,
    );
    const aNew = [a[0], a[1], a[2], a[3], sub.prevHout[0], sub.prevHout[1]];
    const newPt = [sub.newPoint[0], sub.newPoint[1],
                   sub.newHin[0],   sub.newHin[1],
                   sub.newHout[0],  sub.newHout[1]];
    const bNew = [b[0], b[1], sub.nextHin[0], sub.nextHin[1], b[4], b[5]];
    const out = points.slice();
    out[i] = aNew;
    out[j] = bNew;
    // Insert the new point right after index i (or at the end when j wrapped).
    if (j > i) out.splice(j, 0, newPt);
    else       out.push(newPt);
    return out;
}

function makeShape() {
    return {
        id: uid(),
        name: "shape",
        color: pickShapeColor(),
        // value × opacity = the final grey level this shape paints
        // into the MASK output. They're separate fields so the artist
        // can animate the fade-in (opacity) independently from the
        // target grey level (value).
        value: 1.0,
        opacity: 1.0,
        invert: false,
        feather: 0.0,
        // `closed=false` while the artist is still drawing: clicks
        // append new anchors at the cursor. Once the shape is closed
        // (click on the first point, ≥3 anchors), clicks on the curve
        // subdivide via De Casteljau — preserving animation.
        closed: false,
        // Show / hide on canvas AND skip in the rasteriser. Default
        // visible. Per-shape eye toggle in the sidebar.
        visible: true,
        keyframes: {}, // "<frame>": [point, point, ...]
    };
}

function pickShapeColor() {
    const palette = [
        "#7ab8ff", "#f4b860", "#7ed957", "#ff7eb6",
        "#a78bfa", "#fb923c", "#34d399", "#facc15",
    ];
    return palette[Math.floor(Math.random() * palette.length)];
}

// ── editor build ───────────────────────────────────────────────────────────
function buildEditor(node) {
    const root = document.createElement("div");
    root.tabIndex = 0;
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px;
        outline:none; min-height:480px; font:11px sans-serif; color:#cde;
    `;

    // ── top row: canvas + sidebar ────────────────────────────────────
    const topRow = document.createElement("div");
    topRow.style.cssText = "display:flex; flex:1; min-height:0; gap:0;";
    root.appendChild(topRow);

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative; flex:1; min-width:0; background:#000;";
    topRow.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; display:block; touch-action:none;";
    canvasWrap.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `
        position:absolute; left:8px; top:6px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;
    `;
    hint.textContent = "Run once to load the image. Then click to add roto points.";
    canvasWrap.appendChild(hint);

    // ── floating help overlay (toggled by the ? button) ──────────────
    const helpOverlay = document.createElement("div");
    helpOverlay.style.cssText = `
        position:absolute; right:8px; top:6px; max-width:280px;
        background:rgba(15,18,24,0.96); border:1px solid #2a2f37;
        border-radius:5px; padding:10px 12px; font:11px sans-serif;
        color:#cde; display:none; line-height:1.6;
        box-shadow:0 6px 18px rgba(0,0,0,0.5); z-index:10;
    `;
    helpOverlay.innerHTML = `
        <div style="font-weight:bold; margin-bottom:6px;">Shortcuts</div>
        <div><b>Pen tool</b> (new shape, dashed outline):</div>
        <div>&nbsp;&nbsp;<b>Click</b> empty area — drop an anchor (corner)</div>
        <div>&nbsp;&nbsp;<b>Click + drag</b> empty area — drop an anchor with smooth tangents</div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;<i>(hold Shift while dragging to break tangent symmetry)</i></div>
        <div>&nbsp;&nbsp;<b>Click</b> first anchor (≥3 points) — close shape</div>
        <div>&nbsp;&nbsp;<b>Enter / Esc</b> — also closes the shape</div>
        <div style="margin-top:4px;"><b>Once closed</b>:</div>
        <div>&nbsp;&nbsp;<b>Click</b> a point — select it (reduces multi-selection to that one)</div>
        <div>&nbsp;&nbsp;<b>Drag</b> a point — move it</div>
        <div>&nbsp;&nbsp;<b>Click</b> a tangent handle — drag to shape the curve</div>
        <div>&nbsp;&nbsp;<b>Shift</b>+drag tangent — break handle symmetry</div>
        <div>&nbsp;&nbsp;<b>Alt</b>+click a point — delete it (across every keyframe)</div>
        <div>&nbsp;&nbsp;<b>Right-click</b> on a curve — Add point here</div>
        <div>&nbsp;&nbsp;<b>Right-click</b> on a point — Add / Reset / Break / Link tangents + Delete</div>
        <div>&nbsp;&nbsp;<b>Click</b> inside another shape — switch to it</div>
        <div>&nbsp;&nbsp;<b>Click</b> empty area — deselect everything</div>
        <div style="margin-top:6px; border-top:1px solid #2a2f37; padding-top:6px;">
            <b>Space</b> — play / pause<br>
            <b>← / →</b> — step one frame<br>
            <b>K</b> — keyframe at current frame<br>
            <b>Ctrl+Z / Ctrl+Shift+Z</b> — undo / redo<br>
            <b>Delete</b> — remove selected anchors, or active shape if none selected<br>
            <b>Esc</b> — deselect (and close an open shape if it has ≥3 anchors)
        </div>
        <div style="margin-top:6px; border-top:1px solid #2a2f37; padding-top:6px;">
            Multi-select:<br>
            <b>Drag</b> empty area — marquee (replace selection)<br>
            <b>Shift+drag</b> empty area — additive marquee<br>
            <b>Shift+click</b> an anchor — toggle its selection<br>
            With ≥2 selected, a <b>transform box</b> appears:<br>
            &nbsp;&nbsp;Drag <b>inside the box</b> — translate the group<br>
            &nbsp;&nbsp;Drag a <b>corner</b> — scale (Shift = uniform)<br>
            &nbsp;&nbsp;Drag the <b>orange handle</b> — rotate
        </div>
        <div style="margin-top:6px; border-top:1px solid #2a2f37; padding-top:6px;">
            Timeline (three strips):<br>
            <b>Top strip</b> — click / drag to seek the playhead.<br>
            <b>Middle strip</b> ("keys"):<br>
            &nbsp;&nbsp;<b>Click</b> a marker — seek + select that key<br>
            &nbsp;&nbsp;<b>Shift+click</b> — add / remove a key from the selection<br>
            &nbsp;&nbsp;<b>Drag</b> selected markers — move them all together<br>
            &nbsp;&nbsp;<b>Right-click</b> — delete a key<br>
            &nbsp;&nbsp;<b>Delete</b> with selected keys — delete them all<br>
            &nbsp;&nbsp;<b>Click empty area</b> — clear the key selection<br>
            <b>Bottom strip</b> ("view") — zoom minimap:<br>
            &nbsp;&nbsp;<b>Drag</b> the blue window — pan the viewport<br>
            &nbsp;&nbsp;<b>Drag</b> the window's left/right edge — zoom<br>
            &nbsp;&nbsp;<b>Click</b> outside the window — recenter on click<br>
            &nbsp;&nbsp;<b>Right-click</b> — reset to full range
        </div>
        <div style="margin-top:6px; border-top:1px solid #2a2f37; padding-top:6px;">
            Per-shape: <b>Mask value</b> × <b>Opacity</b> = final intensity.<br>
            <b>Global feather</b> blurs the unioned mask AFTER shapes
            combine — different from per-shape feather, which softens
            each shape before union.
        </div>
    `;
    canvasWrap.appendChild(helpOverlay);

    const ctx = canvas.getContext("2d");

    // Sidebar — shape list + per-shape controls
    const sidebar = document.createElement("div");
    sidebar.style.cssText = `
        width:160px; background:#15181d; border-left:1px solid #222;
        display:flex; flex-direction:column; padding:6px; gap:4px;
        overflow-y:auto;
    `;
    topRow.appendChild(sidebar);

    // ── bottom row: timeline + transport ─────────────────────────────
    // Full-width timeline on top, controls row beneath.
    //
    // The timeline strips used to sit INLINE among the buttons in a single
    // flex row. Roto has the busiest row in the pack (8 buttons + auto-key +
    // the view dropdown + counter + help), so the scrub bar was squeezed into
    // whatever was left — often ~200px for a 240-frame range, i.e. under 1px
    // per frame. Stacking gives the timeline the editor's full width and
    // matches a classic player (timeline above, transport below).
    const transport = document.createElement("div");
    transport.style.cssText = `
        display:flex; flex-direction:column; gap:4px; padding:4px 6px;
        background:#15181d; border-top:1px solid #222;
    `;
    root.appendChild(transport);

    // Row that holds the buttons / auto-key / view mode / counter / help,
    // under the timeline. flex-wrap so the many controls degrade gracefully
    // instead of overflowing when the node is narrow.
    const controlsRow = document.createElement("div");
    controlsRow.style.cssText = "display:flex; align-items:center; gap:6px; flex-wrap:wrap;";

    const btn = (txt, title) => {
        const b = document.createElement("button");
        b.textContent = txt; b.title = title;
        // white-space:nowrap + inline-flex keep multi-glyph labels (|◀, ▶|)
        // on a single line — otherwise narrow widths can split the bar
        // and triangle onto two rows.
        b.style.cssText = `
            background:none; border:1px solid #2a2f37; color:#cdd;
            padding:2px 6px; font-size:11px; border-radius:3px; cursor:pointer;
            min-width:22px;
            white-space:nowrap; line-height:1; display:inline-flex;
            align-items:center; justify-content:center;
        `;
        b.onmouseover = () => { b.style.background = "rgba(76,158,255,0.15)"; };
        b.onmouseout  = () => { b.style.background = "none"; };
        return b;
    };

    const playBtn = btn("▶", "Play / Pause (Space)");
    const prevBtn = btn("|◀", "Previous frame (←)");
    const nextBtn = btn("▶|", "Next frame (→)");
    const addKeyBtn = btn("◆+", "Add keyframe at current frame for active shape (K)");
    const delKeyBtn = btn("◆-", "Delete keyframe at current frame for active shape");
    const undoBtn = btn("↶", "Undo (Ctrl+Z)");
    const redoBtn = btn("↷", "Redo (Ctrl+Shift+Z)");
    const helpBtn = btn("?", "Show shortcuts");
    const autoKeyToggle = document.createElement("label");
    autoKeyToggle.style.cssText = "display:flex; gap:4px; align-items:center; cursor:pointer; padding:0 4px;";
    const autoKeyInput = document.createElement("input");
    autoKeyInput.type = "checkbox"; autoKeyInput.checked = true;
    autoKeyToggle.append(autoKeyInput, document.createTextNode("Auto-key"));

    // Three stacked strips in a vertical flex column, occupying the
    // transport's full width (above the controls row):
    //   1. FRAME strip — playhead + scrub.
    //   2. KEYFRAME strip — kf markers (drag, multi-select).
    //   3. RANGE strip — minimap of the full clip with a draggable
    //      window indicating the current viewport on strips 1 & 2.
    //
    // `overflow:hidden` on each strip is defensive: it stops absolute-
    // positioned children (markers, window, handles) from leaking
    // visually below the strip when an interaction is mid-flight.
    // Full width now that it owns its own row (was flex:1 competing with the
    // buttons). Strips are slightly taller too — easier to grab a keyframe.
    const timelineStack = document.createElement("div");
    timelineStack.style.cssText = "display:flex; flex-direction:column; width:100%; gap:2px; min-width:0;";

    const timelineWrap = document.createElement("div");
    timelineWrap.style.cssText = "position:relative; height:20px; background:#1a1d22; border:1px solid #222; border-radius:3px; cursor:pointer; user-select:none; overflow:hidden;";
    const timelineProgress = document.createElement("div");
    timelineProgress.style.cssText = "position:absolute; left:0; top:0; bottom:0; width:0; background:rgba(76,158,255,0.25);";
    timelineWrap.appendChild(timelineProgress);
    const timelinePlayhead = document.createElement("div");
    timelinePlayhead.style.cssText = "position:absolute; top:-2px; bottom:-2px; width:2px; background:#7ab8ff; pointer-events:none;";
    timelineWrap.appendChild(timelinePlayhead);

    const keyframeStrip = document.createElement("div");
    keyframeStrip.style.cssText = "position:relative; height:16px; background:#11141a; border:1px solid #1c2128; border-radius:3px; user-select:none; overflow:hidden;";

    const rangeStrip = document.createElement("div");
    rangeStrip.style.cssText = "position:relative; height:16px; background:#0d1015; border:1px solid #1c2128; border-radius:3px; cursor:pointer; user-select:none; overflow:hidden;";
    const rangeWindow = document.createElement("div");
    // Seed with full-range bounds so the window is visible the moment
    // the node renders — even before any Run has populated frames.
    rangeWindow.style.cssText = "position:absolute; top:0; bottom:0; left:0%; width:100%; background:rgba(76,158,255,0.25); border:1px solid #7ab8ff; cursor:grab; box-sizing:border-box;";
    rangeStrip.appendChild(rangeWindow);
    const rangeLeftHandle = document.createElement("div");
    rangeLeftHandle.style.cssText = "position:absolute; left:-3px; top:0; bottom:0; width:6px; cursor:ew-resize; background:rgba(122,184,255,0.6);";
    rangeWindow.appendChild(rangeLeftHandle);
    const rangeRightHandle = document.createElement("div");
    rangeRightHandle.style.cssText = "position:absolute; right:-3px; top:0; bottom:0; width:6px; cursor:ew-resize; background:rgba(122,184,255,0.6);";
    rangeWindow.appendChild(rangeRightHandle);

    timelineStack.append(timelineWrap, keyframeStrip, rangeStrip);

    // Reposition the range window from current state.viewStart / End.
    // When the node hasn't yet been run (frameCount <= 1) we keep the
    // window at full width so it's visible and clickable — the artist
    // sees what they'll get once a clip is loaded.
    function syncRangeWindow() {
        if (state.frameCount <= 1) {
            rangeWindow.style.left  = "0%";
            rangeWindow.style.width = "100%";
            return;
        }
        const maxF = state.frameCount - 1;
        const left  = (state.viewStart / maxF) * 100;
        const width = ((state.viewEnd - state.viewStart) / maxF) * 100;
        rangeWindow.style.left  = `${left}%`;
        rangeWindow.style.width = `${Math.max(2, width)}%`;
    }

    // Range strip interactions — pan, resize, recenter, reset.
    {
        let drag = null;   // { mode: "pan"|"left"|"right", startX, startVS, startVE, totalPx }
        const onDown = (ev, mode) => {
            ev.stopPropagation();
            ev.preventDefault();
            if (ev.button !== 0) return;
            const r = rangeStrip.getBoundingClientRect();
            drag = {
                mode,
                startX: ev.clientX,
                startVS: state.viewStart,
                startVE: state.viewEnd,
                totalPx: Math.max(1, r.width),
            };
            try { rangeWindow.setPointerCapture(ev.pointerId); } catch (_) {}
        };
        rangeWindow.addEventListener("pointerdown", (ev) => onDown(ev, "pan"));
        rangeLeftHandle.addEventListener("pointerdown", (ev) => onDown(ev, "left"));
        rangeRightHandle.addEventListener("pointerdown", (ev) => onDown(ev, "right"));

        rangeWindow.addEventListener("pointermove", (ev) => {
            if (!drag) return;
            const maxF = Math.max(0, state.frameCount - 1);
            const dxFrames = Math.round(
                ((ev.clientX - drag.startX) / drag.totalPx) * maxF,
            );
            if (drag.mode === "pan") {
                const span = drag.startVE - drag.startVS;
                let vs = clamp(drag.startVS + dxFrames, 0, maxF - span);
                state.viewStart = vs;
                state.viewEnd   = vs + span;
            } else if (drag.mode === "left") {
                state.viewStart = clamp(drag.startVS + dxFrames, 0, drag.startVE - 1);
            } else if (drag.mode === "right") {
                state.viewEnd = clamp(drag.startVE + dxFrames, drag.startVS + 1, maxF);
            }
            clampViewport();
            syncRangeWindow();
            // The main timeline + keyframe strip rebuild from
            // viewStart/End on the next refresh.
            timelineProgress.style.width = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
            timelinePlayhead.style.left  = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
            renderTimelineKeyframes();
        });
        const onUp = (ev) => {
            if (!drag) return;
            try { rangeWindow.releasePointerCapture(ev.pointerId); } catch (_) {}
            drag = null;
        };
        rangeWindow.addEventListener("pointerup", onUp);
        rangeWindow.addEventListener("pointercancel", onUp);

        // Click empty area of the range strip (outside the window) →
        // recenter the viewport on the click. The strip itself does
        // not capture pointer; the bare click here works as long as
        // the window's listener didn't claim it first.
        rangeStrip.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            // Only fire for clicks on the strip itself — clicks on the
            // window / handles / minidots bubble up here too but they
            // have their own handlers or pointer-events:none.
            if (ev.target !== rangeStrip) return;
            const r = rangeStrip.getBoundingClientRect();
            const maxF = Math.max(0, state.frameCount - 1);
            const target = Math.round(((ev.clientX - r.left) / Math.max(1, r.width)) * maxF);
            const span = state.viewEnd - state.viewStart;
            state.viewStart = clamp(Math.round(target - span / 2), 0, maxF - span);
            state.viewEnd   = state.viewStart + span;
            syncRangeWindow();
            renderTimelineKeyframes();
            timelineProgress.style.width = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
            timelinePlayhead.style.left  = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
        });
        rangeStrip.addEventListener("contextmenu", (ev) => {
            ev.preventDefault();
            const maxF = Math.max(0, state.frameCount - 1);
            state.viewStart = 0;
            state.viewEnd   = maxF;
            syncRangeWindow();
            renderTimelineKeyframes();
            timelineProgress.style.width = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
            timelinePlayhead.style.left  = `${clamp(frameToPct(state.currentFrame), 0, 100)}%`;
        });
    }

    // Click on the empty area of the keyframe strip clears the
    // multi-selection — same convention as Adobe / NLE timelines.
    // Markers stopPropagation on their own pointerdown so we never
    // reach this for an actual marker click.
    keyframeStrip.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        if (state.kfSelection.size > 0) {
            state.kfSelection = new Set();
            refreshSidebar(); render();
        }
    });

    const frameLabel = document.createElement("span");
    // margin-left:auto pushes the counter (and the help button after it) to the
    // right end of the controls row.
    frameLabel.style.cssText = "font-family:monospace; min-width:80px; text-align:right; color:#9ab; margin-left:auto;";
    frameLabel.textContent = "1 / 1";

    // View-mode dropdown — switches the canvas between previewing the
    // input image, the alpha mask, the masked image, and the cutout.
    // Persisted in localStorage so reopen keeps the artist's last view.
    const viewSel = document.createElement("select");
    viewSel.title = "Viewport mode";
    viewSel.style.cssText = "background:#1a1d22; color:#cdd; border:1px solid #2a2f37; border-radius:3px; font-size:11px; padding:1px 2px;";
    for (const [v, label] of [
        ["image",  "Image + shapes"],
        ["alpha",  "Alpha mask"],
        ["premul", "Image × Alpha"],
        ["cutout", "Image × (1-Alpha)"],
    ]) {
        const o = document.createElement("option");
        o.value = v; o.textContent = label;
        viewSel.appendChild(o);
    }
    try { viewSel.value = localStorage.getItem("bat_roto_view") || "image"; } catch (_) {}

    // Timeline first (full width), then the controls beneath it.
    controlsRow.append(prevBtn, playBtn, nextBtn, addKeyBtn, delKeyBtn,
                       undoBtn, redoBtn, autoKeyToggle, viewSel,
                       frameLabel, helpBtn);
    transport.append(timelineStack, controlsRow);

    // ── state ────────────────────────────────────────────────────────
    const W = (n) => node.widgets.find(w => w.name === n);
    const setStateWidget = (s) => {
        const w = W("state");
        if (!w) return;
        w.value = JSON.stringify(s);
        try { w.callback?.call(w, w.value); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };
    const readStateWidget = () => {
        try { return JSON.parse(W("state")?.value || ""); }
        catch { return defaultState(); }
    };

    // Back-fill `closed` / `visible` on shapes coming from older saved
    // states so they keep behaving like they did before — anything
    // already on disk is treated as a finished, visible shape.
    function _hydrateLoadedDoc(d) {
        if (!d || !Array.isArray(d.shapes)) return d;
        for (const sh of d.shapes) {
            if (sh.closed === undefined)  sh.closed  = true;
            if (sh.visible === undefined) sh.visible = true;
        }
        return d;
    }

    // Undo / redo: snapshots of state.doc as JSON strings. Cheap (a few
    // KB per snapshot, capped) and trivial to restore. We snapshot AT
    // ACTION BOUNDARIES (after a click, after a drag releases, etc.) —
    // not on every pointermove tick — so a drag from A→B is one undo
    // step, not a hundred. `pointer` points at the current snapshot;
    // undo decrements, redo increments. The redo branch is truncated
    // whenever a new action is recorded mid-history.
    const history = { snapshots: [], pointer: -1, max: 64 };

    const state = {
        doc: _hydrateLoadedDoc(readStateWidget()) || defaultState(),
        activeId: null,           // selected shape id
        activePoint: -1,          // index of selected point in activeShape
        // image / frames
        bgImage: null,            // current frame's bg
        bgFrames: [],             // [b64Image, ...] one per frame
        imgW: 0, imgH: 0,
        frameCount: 1,
        currentFrame: 0,
        // canvas transform (image-space → element-space)
        dispScale: 1, offX: 0, offY: 0,
        dispZoom: 1,              // display-only zoom (1 = fit-to-canvas)
        // drag mode
        drag: null,               // {mode:'point'|'in'|'out'|'multi'|'marquee', ...}
        // Multi-anchor selection on the active shape (Set of anchor
        // indices). Drag any selected anchor to move them all; press
        // Delete to remove them together. Built via Shift+drag marquee
        // or the "Select all" button in the sidebar.
        selection: new Set(),
        marquee: null,            // {x0,y0,x1,y1} in image space, while dragging
        // Multi-keyframe selection on the active shape (Set of frame
        // indices). Drag any selected marker to move them all by the
        // same delta. Shift+click toggles individual markers in/out.
        kfSelection: new Set(),
        // Visible-range viewport on the main timeline + keyframe strip.
        // The third "range" strip below shows the full clip and a
        // draggable / resizable window that maps onto these values.
        // The frame strip and key strip render only what lies within
        // [viewStart, viewEnd] so the artist can zoom into a dense
        // section of keyframes without losing the rest. Both reset to
        // the full range whenever the input batch length changes.
        viewStart: 0,
        viewEnd: 0,
        // playback
        playing: false,
        playInterval: 0,
        playFps: 24,
        // misc
        autoKey: true,
    };
    if (!Array.isArray(state.doc.shapes)) state.doc.shapes = [];
    node._batRotoState = state;

    // ── helpers ──────────────────────────────────────────────────────
    const persist = () => setStateWidget(state.doc);
    const activeShape = () => state.doc.shapes.find(s => s.id === state.activeId);
    const activeShapeKey = () => String(state.currentFrame);

    // ── undo / redo ──────────────────────────────────────────────────
    // Call recordHistory() AFTER a mutation completes so the snapshot
    // captures the post-action state. The redo branch is truncated on
    // every new mutation — standard NLE / pixel-app behaviour.
    function recordHistory() {
        const snap = JSON.stringify(state.doc);
        if (history.pointer >= 0 && history.snapshots[history.pointer] === snap) {
            return;  // no-op: state didn't actually change
        }
        history.snapshots = history.snapshots.slice(0, history.pointer + 1);
        history.snapshots.push(snap);
        if (history.snapshots.length > history.max) {
            history.snapshots.shift();
        } else {
            history.pointer = history.snapshots.length - 1;
        }
        history.pointer = history.snapshots.length - 1;
    }
    function applySnapshot(idx) {
        const snap = history.snapshots[idx];
        if (!snap) return;
        try {
            state.doc = JSON.parse(snap);
            _hydrateLoadedDoc(state.doc);
        } catch (_) { return; }
        // Drop any in-flight drag / selection that referenced the old
        // point indices — they may not exist any more.
        state.drag = null;
        state.selection = new Set();
        state.kfSelection = new Set();
        state.activePoint = -1;
        // If the active shape no longer exists in this snapshot,
        // clear the selection.
        if (state.activeId && !state.doc.shapes.find(s => s.id === state.activeId)) {
            state.activeId = null;
        }
        persist(); refreshSidebar(); render();
    }
    function undo() {
        if (history.pointer <= 0) return;
        history.pointer -= 1;
        applySnapshot(history.pointer);
    }
    function redo() {
        if (history.pointer >= history.snapshots.length - 1) return;
        history.pointer += 1;
        applySnapshot(history.pointer);
    }

    function shapePointsAtCurrent() {
        const sh = activeShape();
        if (!sh) return null;
        const kfs = sh.keyframes || {};
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        if (!keys.length) return null;
        const f = state.currentFrame;
        if (f <= keys[0]) return kfs[String(keys[0])];
        if (f >= keys[keys.length - 1]) return kfs[String(keys[keys.length - 1])];
        let prev = keys[0], nxt = keys[keys.length - 1];
        for (const k of keys) {
            if (k <= f) prev = k;
            if (k >= f && k !== prev) { nxt = k; break; }
        }
        if (prev === nxt) return kfs[String(prev)];
        const t = (f - prev) / (nxt - prev);
        const a = kfs[String(prev)], b = kfs[String(nxt)];
        if (a.length !== b.length) return a;
        return a.map((pa, i) => {
            const pb = b[i];
            if (pa.length !== pb.length) return pa.slice();
            return pa.map((v, k) => v + (pb[k] - v) * t);
        });
    }

    // Returns the keyframe TO MODIFY when the user drags a point. When
    // auto-key is on we create / replace the keyframe at the current
    // frame. When off, we modify the nearest keyframe ≤ current.
    function getEditableKeyframe(sh) {
        const keyStr = String(state.currentFrame);
        const kfs = sh.keyframes;
        if (state.autoKey || !Object.keys(kfs).length) {
            // Snapshot whatever points the interpolation produces at this
            // frame (so we don't lose the in-between pose), then write.
            if (!kfs[keyStr]) {
                const pts = shapePointsAtCurrent() || [];
                kfs[keyStr] = pts.map(p => p.slice());
            }
            return kfs[keyStr];
        }
        // Modify the keyframe at or before the current frame.
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        let target = keys[0];
        for (const k of keys) if (k <= state.currentFrame) target = k;
        return kfs[String(target)];
    }

    // ── coordinate conversion ────────────────────────────────────────
    function recomputeDisplay() {
        if (!state.imgW || !state.imgH) {
            state.dispScale = 1; state.offX = 0; state.offY = 0; return;
        }
        const cw = canvas.clientWidth || 1;
        const ch = canvas.clientHeight || 1;
        // Fit-to-canvas × display-only zoom + pan. Centred at zoom 1 / pan 0;
        // zooming out reveals the area outside the frame (roto a shape past
        // the canvas edge), pan / zoom-to-cursor shift the centre.
        const fit = Math.min(cw / state.imgW, ch / state.imgH);
        state.dispScale = fit * (state.dispZoom || 1);
        state.offX = (cw - state.imgW * state.dispScale) / 2 + (state.panX || 0);
        state.offY = (ch - state.imgH * state.dispScale) / 2 + (state.panY || 0);
    }
    const i2d = (x, y) => ({ x: state.offX + x * state.dispScale, y: state.offY + y * state.dispScale });
    const d2i = (x, y) => ({ x: (x - state.offX) / state.dispScale, y: (y - state.offY) / state.dispScale });
    function localMouse(e) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (canvas.clientWidth / r.width),
            y: (e.clientY - r.top)  * (canvas.clientHeight / r.height),
        };
    }

    // ── render ───────────────────────────────────────────────────────
    function render() {
        const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        const cssW = Math.max(1, canvas.clientWidth || canvasWrap.offsetWidth || 480);
        const cssH = Math.max(1, canvas.clientHeight || canvasWrap.offsetHeight || 360);
        if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
            canvas.width = Math.floor(cssW * dpr);
            canvas.height = Math.floor(cssH * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        recomputeDisplay();

        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cssW, cssH);

        const mode = viewSel.value || "image";

        // Compute the bg image's screen rect once (used by every mode
        // that wants to show the image somewhere).
        let bgRect = null;
        if (state.bgImage && state.imgW) {
            const o = i2d(0, 0);
            bgRect = { x: o.x, y: o.y,
                       w: state.imgW * state.dispScale,
                       h: state.imgH * state.dispScale };
        }

        // For non-default modes we need a rasterised alpha — fill each
        // visible shape grayscale and union-by-max via the "lighten"
        // composite op. Drawn onto an offscreen canvas so we can blend
        // it with the bg via the right Canvas2D operation per mode.
        let alphaCanvas = null;
        if (mode !== "image") {
            alphaCanvas = document.createElement("canvas");
            alphaCanvas.width  = canvas.width;
            alphaCanvas.height = canvas.height;
            const actx = alphaCanvas.getContext("2d");
            actx.setTransform(dpr, 0, 0, dpr, 0, 0);
            actx.fillStyle = "#000";
            actx.fillRect(0, 0, cssW, cssH);
            // src-over composite, NOT lighten/max: shapes higher in the
            // list paint over the ones below using their `opacity` as
            // the alpha channel. A value=0.5 opacity=1.0 shape forces
            // its area to mid-grey, regardless of what was under it.
            // Mirrors the backend's per-shape `coverage * opacity`
            // alpha-composite (see bat_roto.py `_rasterise_shape`).
            actx.globalCompositeOperation = "source-over";
            for (const sh of state.doc.shapes) {
                if (sh.visible === false) continue;
                if (sh.closed === false) continue;   // open shapes don't fill yet
                const pts = (sh.id === state.activeId)
                    ? shapePointsAtCurrent()
                    : resolveShapeAt(sh, state.currentFrame);
                if (!pts || pts.length < 2) continue;
                fillShapeMask(actx, sh, pts, cssW, cssH);
            }
            // Global feather (matches the backend's post-union blur).
            // Canvas2D's filter property gives us a cheap Gaussian.
            const gf = state.doc.global_feather ?? 0;
            if (gf > 0) {
                const blurCanvas = document.createElement("canvas");
                blurCanvas.width = alphaCanvas.width;
                blurCanvas.height = alphaCanvas.height;
                const bctx = blurCanvas.getContext("2d");
                bctx.filter = `blur(${gf}px)`;
                bctx.drawImage(alphaCanvas, 0, 0);
                alphaCanvas = blurCanvas;
            }
        }

        if (mode === "image" || mode === "premul" || mode === "cutout") {
            // Image (and image-based modes) paint the bg first.
            if (bgRect) {
                ctx.drawImage(state.bgImage, bgRect.x, bgRect.y, bgRect.w, bgRect.h);
            }
        }

        if (mode === "alpha") {
            // Just stamp the alpha canvas onto the main canvas.
            ctx.drawImage(alphaCanvas, 0, 0, cssW, cssH);
        } else if (mode === "premul") {
            // Multiply blend: pixel = image_pixel × alpha_pixel / 255.
            // Outside the alpha (black) → black. Inside (white) → image.
            ctx.globalCompositeOperation = "multiply";
            ctx.drawImage(alphaCanvas, 0, 0, cssW, cssH);
            ctx.globalCompositeOperation = "source-over";
        } else if (mode === "cutout") {
            // Inverted alpha multiply: paint a white-where-NOT-shape
            // mask first, then multiply.
            const invCanvas = document.createElement("canvas");
            invCanvas.width = alphaCanvas.width;
            invCanvas.height = alphaCanvas.height;
            const ictx = invCanvas.getContext("2d");
            ictx.fillStyle = "#fff";
            ictx.fillRect(0, 0, invCanvas.width, invCanvas.height);
            ictx.globalCompositeOperation = "difference";
            ictx.drawImage(alphaCanvas, 0, 0);
            ctx.globalCompositeOperation = "multiply";
            ctx.drawImage(invCanvas, 0, 0, cssW, cssH);
            ctx.globalCompositeOperation = "source-over";
        }

        if (mode === "image") {
            // Outlines + shape adornments live on top of the bg only in
            // the default "image + shapes" mode. The alpha-only and
            // composited modes are meant to show what the OUTPUT looks
            // like; cluttering them with editing chrome defeats the
            // purpose.
            for (const sh of state.doc.shapes) {
                if (sh.visible === false && sh.id !== state.activeId) continue;
                const pts = (sh.id === state.activeId)
                    ? shapePointsAtCurrent()
                    : resolveShapeAt(sh, state.currentFrame);
                if (!pts || pts.length < 2) continue;
                drawShape(sh, pts, sh.id === state.activeId);
            }
            const sh = activeShape();
            if (sh) {
                const pts = shapePointsAtCurrent();
                if (pts) drawHandles(sh, pts);
            }
            // Transform box (+ scale + rotate handles) when ≥2 anchors
            // are multi-selected. Lets the artist scale / rotate the
            // selected anchors as a group — drag a corner to scale
            // (Shift = uniform), drag the rotate handle for rotation.
            drawTransformBox();
        } else {
            // In preview modes, still draw the active shape's outline
            // + handles (faintly) so the artist knows what they're
            // editing — otherwise the canvas becomes uneditable.
            const sh = activeShape();
            if (sh) {
                const pts = shapePointsAtCurrent();
                if (pts) {
                    ctx.globalAlpha = 0.6;
                    drawShape(sh, pts, true);
                    ctx.globalAlpha = 1;
                    drawHandles(sh, pts);
                }
            }
        }

        // Marquee rect — drawn last so it sits on top of everything.
        if (state.marquee) {
            const m = state.marquee;
            const a = i2d(Math.min(m.x0, m.x1), Math.min(m.y0, m.y1));
            const b = i2d(Math.max(m.x0, m.x1), Math.max(m.y0, m.y1));
            ctx.fillStyle = "rgba(76,158,255,0.12)";
            ctx.strokeStyle = "#7ab8ff";
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.rect(a.x, a.y, b.x - a.x, b.y - a.y);
            ctx.fill();
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // Fill a single closed shape as a flat grayscale onto a context.
    // Used to build the alpha mask for non-default view modes. Mirrors
    // the backend rasteriser: shape's "Mask value" (stored under the
    // legacy `opacity` field for back-compat) × invert.
    function fillShapeMask(actx, sh, pts, cssW, cssH) {
        // Mirror the backend rasteriser. Each shape paints with:
        //   rgb   = `value`   (the grey level the artist set)
        //   alpha = coverage × `opacity`
        // where coverage is the post-feather, optionally inverted
        // polygon (so the edge softens BEFORE the value × opacity
        // gate). Composite via src-over so a high-opacity shape
        // occludes everything below it — different from the previous
        // `lighten` (max) which couldn't ever block underlying shapes.
        const v = Math.max(0, Math.min(1, sh.value ?? 1));
        const o = Math.max(0, Math.min(1, sh.opacity ?? 1));
        const ft = Math.max(0, Number(sh.feather) || 0);

        // Build the shape path in screen pixels.
        const path = new Path2D();
        for (let i = 0; i < pts.length; i++) {
            const a = pts[i];
            const b = pts[(i + 1) % pts.length];
            const p0 = i2d(a[0], a[1]);
            const p1 = i2d(a[4], a[5]);
            const p2 = i2d(b[2], b[3]);
            const p3 = i2d(b[0], b[1]);
            if (i === 0) path.moveTo(p0.x, p0.y);
            path.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
        }
        path.closePath();

        // Render the shape's coverage onto a tmp canvas: white where
        // covered, transparent elsewhere. After Gaussian blur the
        // edge alpha ramps softly — that's the per-shape feather.
        const tmp = document.createElement("canvas");
        tmp.width = actx.canvas.width;
        tmp.height = actx.canvas.height;
        const tctx = tmp.getContext("2d");
        const tx = actx.getTransform();
        tctx.setTransform(tx.a, tx.b, tx.c, tx.d, tx.e, tx.f);
        if (ft > 0) tctx.filter = `blur(${ft}px)`;
        tctx.fillStyle = "#fff";
        if (sh.invert) {
            // Even-odd inverse: the shape becomes a hole in a full-
            // canvas rect, so the area OUTSIDE the curve is filled.
            const combo = new Path2D();
            combo.rect(0, 0, cssW, cssH);
            combo.addPath(path);
            tctx.fill(combo, "evenodd");
        } else {
            tctx.fill(path);
        }
        tctx.filter = "none";

        // Recolor the tmp pixels: RGB → value, alpha → existing × opacity.
        // Porter-Duff `source-in` keeps the destination alpha shape but
        // replaces RGB with the source colour. globalAlpha scales the
        // source alpha to apply opacity.
        tctx.globalCompositeOperation = "source-in";
        tctx.globalAlpha = o;
        const vi = Math.round(v * 255);
        tctx.fillStyle = `rgb(${vi}, ${vi}, ${vi})`;
        tctx.fillRect(0, 0, cssW, cssH);
        tctx.globalAlpha = 1;
        tctx.globalCompositeOperation = "source-over";

        // Composite tmp onto the accumulator with default src-over.
        actx.drawImage(tmp, 0, 0, cssW, cssH);
    }

    function drawShape(sh, pts, isActive) {
        // Open shapes (still being drawn) draw the actual bezier curve
        // through every adjacent pair of anchors — using each anchor's
        // tangent handles — but WITHOUT a closing edge between the
        // last and first anchor, and no fill. That way the artist
        // sees the curvature of any handles they drag out while the
        // shape is still being authored.
        const open = sh.closed === false;
        ctx.beginPath();
        if (open) {
            if (pts.length === 1) {
                const o = i2d(pts[0][0], pts[0][1]);
                ctx.moveTo(o.x, o.y);
            } else {
                for (let i = 0; i < pts.length - 1; i++) {
                    const a = pts[i];
                    const b = pts[i + 1];
                    const p0 = i2d(a[0], a[1]);
                    const p1 = i2d(a[4], a[5]);    // a.outHandle
                    const p2 = i2d(b[2], b[3]);    // b.inHandle
                    const p3 = i2d(b[0], b[1]);
                    if (i === 0) ctx.moveTo(p0.x, p0.y);
                    ctx.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                }
            }
            ctx.strokeStyle = sh.color;
            ctx.lineWidth = isActive ? 1.8 : 1.2;
            ctx.setLineDash([4, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
            // Highlight the FIRST point so the artist knows clicking
            // back on it closes the shape.
            if (pts.length >= 3 && isActive) {
                const f = i2d(pts[0][0], pts[0][1]);
                ctx.strokeStyle = "#7ec0ee";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(f.x, f.y, 8, 0, Math.PI * 2);
                ctx.stroke();
            }
            return;
        }
        // Closed shape — proper cubic bezier render.
        for (let i = 0; i < pts.length; i++) {
            const a = pts[i];
            const b = pts[(i + 1) % pts.length];
            const p0 = i2d(a[0], a[1]);
            const p1 = i2d(a[4], a[5]);
            const p2 = i2d(b[2], b[3]);
            const p3 = i2d(b[0], b[1]);
            if (i === 0) ctx.moveTo(p0.x, p0.y);
            ctx.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
        }
        ctx.closePath();
        ctx.fillStyle = isActive
            ? hexWithAlpha(sh.color, 0.18)
            : hexWithAlpha(sh.color, 0.08);
        ctx.fill();
        ctx.strokeStyle = sh.color;
        ctx.lineWidth = isActive ? 1.8 : 1.2;
        ctx.stroke();
    }

    // Draw the multi-selection transform box: axis-aligned bbox of the
    // selected anchors, with 4 corner scale handles and one rotate
    // handle 30 display-px above the top-mid edge.
    function drawTransformBox() {
        const bb = selectionBBox();
        if (!bb) return;
        const tl = i2d(bb.minX, bb.minY);
        const tr = i2d(bb.maxX, bb.minY);
        const br = i2d(bb.maxX, bb.maxY);
        const bl = i2d(bb.minX, bb.maxY);

        // Box outline.
        ctx.save();
        ctx.strokeStyle = "rgba(122,184,255,0.85)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(tl.x, tl.y);
        ctx.lineTo(tr.x, tr.y);
        ctx.lineTo(br.x, br.y);
        ctx.lineTo(bl.x, bl.y);
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);

        // Corner scale handles.
        const HR = 5;
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        for (const c of [tl, tr, br, bl]) {
            ctx.beginPath();
            ctx.rect(c.x - HR, c.y - HR, HR * 2, HR * 2);
            ctx.fill(); ctx.stroke();
        }

        // Rotate handle: connector line + circle 30 display-px above
        // the top-mid edge.
        const topMidX = (tl.x + tr.x) / 2;
        const topMidY = (tl.y + tr.y) / 2;
        const rotY = topMidY - 30;
        ctx.strokeStyle = "#f4b860";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(topMidX, topMidY);
        ctx.lineTo(topMidX, rotY);
        ctx.stroke();
        ctx.fillStyle = "#f4b860";
        ctx.strokeStyle = "#000";
        ctx.beginPath();
        ctx.arc(topMidX, rotY, 6, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.restore();
    }

    function drawHandles(sh, pts) {
        const HANDLE_EPS = 1.0;   // image-space px below which we treat
                                  // a handle as coincident with the anchor.
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            const o = i2d(p[0], p[1]);
            // Skip drawing tangent handles that are essentially at the
            // anchor — they're visual noise on linear segments and
            // they overlap the anchor's click target. Hit-testing
            // (pointerdown handler) uses the same threshold so the
            // user doesn't accidentally drag invisible handles.
            const dxi = p[2] - p[0], dyi = p[3] - p[1];
            const dxo = p[4] - p[0], dyo = p[5] - p[1];
            const showIn  = (dxi * dxi + dyi * dyi) > HANDLE_EPS * HANDLE_EPS;
            const showOut = (dxo * dxo + dyo * dyo) > HANDLE_EPS * HANDLE_EPS;
            if (showIn || showOut) {
                const hi = i2d(p[2], p[3]);
                const ho = i2d(p[4], p[5]);
                ctx.strokeStyle = "rgba(180,200,220,0.5)";
                ctx.lineWidth = 1;
                if (showIn && showOut) {
                    ctx.beginPath(); ctx.moveTo(hi.x, hi.y); ctx.lineTo(ho.x, ho.y); ctx.stroke();
                } else if (showIn) {
                    ctx.beginPath(); ctx.moveTo(hi.x, hi.y); ctx.lineTo(o.x,  o.y);  ctx.stroke();
                } else {
                    ctx.beginPath(); ctx.moveTo(o.x,  o.y);  ctx.lineTo(ho.x, ho.y); ctx.stroke();
                }
                ctx.fillStyle = "#c8d4e0";
                if (showIn)  { ctx.beginPath(); ctx.arc(hi.x, hi.y, 3, 0, Math.PI * 2); ctx.fill(); }
                if (showOut) { ctx.beginPath(); ctx.arc(ho.x, ho.y, 3, 0, Math.PI * 2); ctx.fill(); }
            }
            // Main anchor — multi-selected anchors get a blue halo so
            // it's obvious which ones a marquee picked up.
            const inSelection = state.selection?.has(i);
            ctx.fillStyle = inSelection
                ? "#7ab8ff"
                : (i === state.activePoint ? "#ffd966" : "#ffffff");
            ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
        }
    }

    function hexWithAlpha(hex, a) {
        const m = /^#?([0-9a-f]{6})$/i.exec(hex);
        if (!m) return `rgba(122,184,255,${a})`;
        const r = parseInt(m[1].slice(0, 2), 16);
        const g = parseInt(m[1].slice(2, 4), 16);
        const b = parseInt(m[1].slice(4, 6), 16);
        return `rgba(${r},${g},${b},${a})`;
    }

    function resolveShapeAt(sh, frame) {
        const kfs = sh.keyframes || {};
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        if (!keys.length) return null;
        if (frame <= keys[0]) return kfs[String(keys[0])];
        if (frame >= keys[keys.length - 1]) return kfs[String(keys[keys.length - 1])];
        let prev = keys[0], nxt = keys[keys.length - 1];
        for (const k of keys) {
            if (k <= frame) prev = k;
            if (k >= frame && k !== prev) { nxt = k; break; }
        }
        if (prev === nxt) return kfs[String(prev)];
        const t = (frame - prev) / (nxt - prev);
        const a = kfs[String(prev)], b = kfs[String(nxt)];
        if (a.length !== b.length) return a;
        return a.map((pa, i) => {
            const pb = b[i];
            return pa.map((v, k) => v + (pb[k] - v) * t);
        });
    }

    // ── handle helpers ──────────────────────────────────────────────
    // Treat handles as "mirrored" when the OUT handle is the exact
    // negation of the IN handle around the anchor (within half a pixel
    // to account for float drift). Used at drag start to decide whether
    // a plain in/out drag should auto-mirror the other handle.
    function areHandlesMirrored(pp) {
        if (!pp) return false;
        const inDx  = pp[2] - pp[0], inDy  = pp[3] - pp[1];
        const outDx = pp[4] - pp[0], outDy = pp[5] - pp[1];
        return Math.abs(outDx + inDx) < 0.5 && Math.abs(outDy + inDy) < 0.5;
    }

    // Pull tangent handles out from an anchor that currently has none.
    // We aim the handles along the bisector of the two adjacent edges
    // (so the new tangent is a sensible starting point for smoothing
    // through this point) at 1/3 of the average edge length. If the
    // shape only has one or two anchors we fall back to a horizontal
    // 30-pixel handle pair.
    function addHandlesToAnchor(kf, i) {
        if (!kf[i]) return;
        const p = kf[i];
        const n = kf.length;
        if (n < 2) {
            const fallback = 30;
            p[2] = p[0] - fallback; p[3] = p[1];
            p[4] = p[0] + fallback; p[5] = p[1];
            return;
        }
        const prev = kf[(i - 1 + n) % n];
        const next = kf[(i + 1) % n];
        const vIn  = { x: prev[0] - p[0], y: prev[1] - p[1] };
        const vOut = { x: next[0] - p[0], y: next[1] - p[1] };
        // Tangent direction = bisector of the two edge vectors,
        // perpendicular-flipped so it points along the curve rather
        // than into the shape.
        const lenIn  = Math.hypot(vIn.x,  vIn.y)  || 1;
        const lenOut = Math.hypot(vOut.x, vOut.y) || 1;
        const ux = vIn.x / lenIn  - vOut.x / lenOut;
        const uy = vIn.y / lenIn  - vOut.y / lenOut;
        const uLen = Math.hypot(ux, uy) || 1;
        const dirX = ux / uLen, dirY = uy / uLen;
        const handleLen = Math.min(lenIn, lenOut) / 3;
        p[2] = p[0] - dirX * handleLen; p[3] = p[1] - dirY * handleLen;
        p[4] = p[0] + dirX * handleLen; p[5] = p[1] + dirY * handleLen;
    }

    // Apply a mutator function to EVERY keyframe of the shape, then
    // persist + record one undo step. Topology-affecting handle
    // changes go through this so the animation stays consistent
    // across all keyframes.
    function applyToAllKeyframes(sh, mutateKf) {
        for (const k of Object.keys(sh.keyframes)) {
            mutateKf(sh.keyframes[k]);
        }
        persist(); recordHistory(); render();
    }

    // ── viewport math ───────────────────────────────────────────────
    // The viewport is the visible-range window on the main timeline +
    // keyframe strip. Everything that converts between FRAME and PIXEL
    // on those strips goes through these three helpers so a single
    // source of truth handles "we're zoomed in on frames [120..180] of
    // a 600-frame clip" without dozens of call sites needing to know.
    function viewSpan() { return Math.max(1, state.viewEnd - state.viewStart); }
    // Frame → percentage along the visible strip (0..100). Outside the
    // viewport returns a value < 0 or > 100 — callers use that to
    // hide off-screen markers.
    function frameToPct(f) { return ((f - state.viewStart) / viewSpan()) * 100; }
    // Pixel offset from the left edge of the strip → frame index in
    // the current viewport. Clamped to the full clip range, not just
    // the viewport, so a drag past the strip edge can still target a
    // frame at the very start or end if the maths drift slightly.
    function pxToFrame(dxPx, totalPx) {
        return clamp(
            state.viewStart + (dxPx / Math.max(1, totalPx)) * viewSpan(),
            0, Math.max(0, state.frameCount - 1),
        );
    }
    function clampViewport() {
        const maxF = Math.max(0, state.frameCount - 1);
        if (state.viewEnd > maxF) state.viewEnd = maxF;
        if (state.viewStart < 0) state.viewStart = 0;
        if (state.viewStart >= state.viewEnd) {
            state.viewStart = 0;
            state.viewEnd = maxF;
        }
    }

    // Bounding box of multi-selected anchors (in image space). Used to
    // draw + hit-test the transform handles. Returns null if the
    // selection size is < 2 OR if the box is degenerate (zero area).
    function selectionBBox() {
        if (!state.selection || state.selection.size < 2) return null;
        const sh = activeShape();
        if (!sh) return null;
        const pts = shapePointsAtCurrent();
        if (!pts) return null;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        let n = 0;
        for (const i of state.selection) {
            const p = pts[i];
            if (!p) continue;
            if (p[0] < minX) minX = p[0];
            if (p[1] < minY) minY = p[1];
            if (p[0] > maxX) maxX = p[0];
            if (p[1] > maxY) maxY = p[1];
            n += 1;
        }
        if (n < 2 || maxX - minX < 1e-3 || maxY - minY < 1e-3) return null;
        return { minX, minY, maxX, maxY,
                 cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
    }

    // Transform-handle hit test (returns "tl"|"tr"|"br"|"bl"|"rot"|null).
    function transformHandleAt(ic) {
        const bb = selectionBBox();
        if (!bb) return null;
        const r = 8 / state.dispScale;
        const corners = [
            ["tl", bb.minX, bb.minY],
            ["tr", bb.maxX, bb.minY],
            ["br", bb.maxX, bb.maxY],
            ["bl", bb.minX, bb.maxY],
        ];
        for (const [name, x, y] of corners) {
            if (Math.abs(x - ic.x) <= r && Math.abs(y - ic.y) <= r) return name;
        }
        // Rotate handle: 30 display-pixels above the top-mid edge.
        const offset = 30 / state.dispScale;
        const rx = bb.cx;
        const ry = bb.minY - offset;
        if (Math.hypot(rx - ic.x, ry - ic.y) <= r) return "rot";
        return null;
    }

    // ── right-click context menu ─────────────────────────────────────
    // Replaces the previous implicit "click-on-a-curve-to-add-a-point"
    // behaviour, which was confusing on a closed shape (clicking
    // anywhere on the curve created a new anchor instead of just
    // deselecting). Now adding a point is an explicit Right-click →
    // "Add point here", which uses the same De Casteljau subdivision
    // and so still preserves the animation across every keyframe.
    let contextMenuEl = null;
    function hideContextMenu() {
        if (contextMenuEl) {
            contextMenuEl.remove();
            contextMenuEl = null;
        }
    }
    function showContextMenu(items, clientX, clientY) {
        hideContextMenu();
        if (!items.length) return;
        const m = document.createElement("div");
        m.style.cssText = `
            position:fixed; left:${clientX}px; top:${clientY}px;
            background:#1a1d22; border:1px solid #3a4a66;
            border-radius:4px; padding:3px 0; z-index:9999;
            box-shadow:0 6px 16px rgba(0,0,0,0.5);
            font:11px sans-serif; color:#cde; min-width:140px;
        `;
        for (const item of items) {
            const row = document.createElement("div");
            row.textContent = item.label;
            row.style.cssText = "padding:5px 12px; cursor:pointer;";
            row.onmouseover = () => { row.style.background = "rgba(76,158,255,0.18)"; };
            row.onmouseout  = () => { row.style.background = "none"; };
            row.onclick = () => { hideContextMenu(); item.action(); };
            m.appendChild(row);
        }
        document.body.appendChild(m);
        contextMenuEl = m;
        // Click anywhere else to dismiss.
        setTimeout(() => {
            const dismiss = (ev) => {
                if (contextMenuEl && !contextMenuEl.contains(ev.target)) {
                    hideContextMenu();
                    document.removeEventListener("pointerdown", dismiss, true);
                }
            };
            document.addEventListener("pointerdown", dismiss, true);
        }, 0);
    }

    canvas.addEventListener("contextmenu", (e) => {
        if (!state.imgW) return;
        e.preventDefault();
        const p = localMouse(e);
        const ic = d2i(p.x, p.y);
        const sh = activeShape();
        if (!sh) return;
        const pts = shapePointsAtCurrent();
        const hitRadius = 7 / state.dispScale;
        const items = [];

        // Right-click on an anchor → handle-related actions in addition
        // to "Delete point". The handle actions all mutate EVERY
        // keyframe of the shape so the topology / smoothness stays
        // consistent across the animation.
        if (pts) {
            for (let i = 0; i < pts.length; i++) {
                const pp = pts[i];
                if (Math.hypot(pp[0] - ic.x, pp[1] - ic.y) <= hitRadius) {
                    const hasIn  = Math.hypot(pp[2] - pp[0], pp[3] - pp[1]) > 0.5;
                    const hasOut = Math.hypot(pp[4] - pp[0], pp[5] - pp[1]) > 0.5;
                    const hasAny = hasIn || hasOut;
                    const mirrored = areHandlesMirrored(pp);

                    if (!hasAny) {
                        items.push({
                            label: "Add tangent handles",
                            action: () => applyToAllKeyframes(sh, kf => {
                                addHandlesToAnchor(kf, i);
                            }),
                        });
                    } else {
                        items.push({
                            label: "Reset to sharp corner",
                            action: () => applyToAllKeyframes(sh, kf => {
                                if (!kf[i]) return;
                                kf[i][2] = kf[i][0]; kf[i][3] = kf[i][1];
                                kf[i][4] = kf[i][0]; kf[i][5] = kf[i][1];
                            }),
                        });
                        if (mirrored) {
                            items.push({
                                label: "Break tangents (independent)",
                                action: () => applyToAllKeyframes(sh, kf => {
                                    if (!kf[i]) return;
                                    // Tiny offset to the OUT handle so
                                    // it's no longer perfectly mirrored.
                                    // The drag logic uses areHandlesMirrored
                                    // to decide whether to keep mirroring,
                                    // so this one-pixel nudge is enough
                                    // to flip the behaviour permanently.
                                    kf[i][4] += 1;
                                }),
                            });
                        } else {
                            items.push({
                                label: "Link tangents (mirror)",
                                action: () => applyToAllKeyframes(sh, kf => {
                                    if (!kf[i]) return;
                                    // Mirror OUT across the anchor using
                                    // the current IN direction. If IN is
                                    // zero-length, mirror IN from OUT
                                    // instead. The previously-out length
                                    // is preserved on whichever handle we
                                    // didn't replace.
                                    const inLen  = Math.hypot(kf[i][2] - kf[i][0], kf[i][3] - kf[i][1]);
                                    if (inLen > 0.5) {
                                        kf[i][4] = 2 * kf[i][0] - kf[i][2];
                                        kf[i][5] = 2 * kf[i][1] - kf[i][3];
                                    } else {
                                        kf[i][2] = 2 * kf[i][0] - kf[i][4];
                                        kf[i][3] = 2 * kf[i][1] - kf[i][5];
                                    }
                                }),
                            });
                        }
                    }

                    items.push({
                        label: "Delete point",
                        action: () => {
                            const kf = getEditableKeyframe(sh);
                            kf.splice(i, 1);
                            for (const k of Object.keys(sh.keyframes)) {
                                if (k !== String(state.currentFrame)
                                        && sh.keyframes[k].length > i) {
                                    sh.keyframes[k].splice(i, 1);
                                }
                            }
                            state.activePoint = -1;
                            state.selection.delete(i);
                            persist(); recordHistory(); render();
                        },
                    });
                    showContextMenu(items, e.clientX, e.clientY);
                    return;
                }
            }
        }

        // Right-click on or near a curve segment of a closed shape →
        // offer "Add point here". The cursor doesn't have to be ON the
        // curve exactly; we snap to the closest edge within a generous
        // radius so the gesture isn't fiddly.
        if (sh.closed && pts && pts.length >= 2) {
            const hit = closestEdgeAndT(pts, ic.x, ic.y);
            const edgeRadius = 24 / state.dispScale;
            if (hit && hit.distance <= edgeRadius) {
                items.push({
                    label: "Add point here",
                    action: () => {
                        const kfs = sh.keyframes;
                        const allKeys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
                        if (allKeys.length === 0) {
                            kfs[String(state.currentFrame)] = pts.map(p => p.slice());
                            allKeys.push(state.currentFrame);
                        }
                        for (const k of allKeys) {
                            kfs[String(k)] = subdivideKeyframeAt(
                                kfs[String(k)], hit.edgeIndex, hit.t,
                            );
                        }
                        state.activePoint = (hit.edgeIndex + 1) % (pts.length + 1);
                        persist(); recordHistory(); render();
                    },
                });
            }
        }

        if (items.length) showContextMenu(items, e.clientX, e.clientY);
    });

    // ── pointer interaction ──────────────────────────────────────────
    canvas.addEventListener("pointerdown", (e) => {
        if (!state.imgW) return;
        root.focus();
        const p = localMouse(e);
        const ic = d2i(p.x, p.y);

        const sh = activeShape();
        const pts = sh ? shapePointsAtCurrent() : null;
        const hitRadius = 7 / state.dispScale;

        // Shift+click on an anchor toggles its membership in the
        // multi-selection (Illustrator / Nuke convention). Handled
        // here so it short-circuits BEFORE any drag-related branch.
        if (e.shiftKey && sh && pts && pts.length > 0) {
            for (let i = 0; i < pts.length; i++) {
                const pp = pts[i];
                if (Math.hypot(pp[0] - ic.x, pp[1] - ic.y) <= hitRadius) {
                    if (state.selection.has(i)) state.selection.delete(i);
                    else                        state.selection.add(i);
                    state.activePoint = i;
                    render();
                    return;
                }
            }
            // Fall through — Shift+drag in empty space is handled by
            // the same marquee branch at the end (with additive mode
            // when shiftKey is set on pointerdown).
        }

        // Transform handle (corner / rotate) — only present when ≥2
        // anchors are multi-selected. Tested BEFORE individual anchor
        // hit-tests so a corner handle that happens to land near an
        // anchor doesn't get stolen.
        if (state.selection && state.selection.size >= 2) {
            const handle = transformHandleAt(ic);
            if (handle) {
                const bb = selectionBBox();
                const kf = getEditableKeyframe(sh);
                // Snapshot every selected point's CURRENT position +
                // handles so the transform math is relative to the
                // start, not cumulative across pointermove events.
                const snapshot = {};
                for (const i of state.selection) {
                    if (kf[i]) snapshot[i] = kf[i].slice();
                }
                if (handle === "rot") {
                    state.drag = {
                        mode: "rotate",
                        bb,
                        snapshot,
                        startAngle: Math.atan2(ic.y - bb.cy, ic.x - bb.cx),
                    };
                } else {
                    // Anchor of the scale = opposite corner.
                    const opp = {
                        tl: { x: bb.maxX, y: bb.maxY },
                        tr: { x: bb.minX, y: bb.maxY },
                        br: { x: bb.minX, y: bb.minY },
                        bl: { x: bb.maxX, y: bb.minY },
                    }[handle];
                    const cur = {
                        tl: { x: bb.minX, y: bb.minY },
                        tr: { x: bb.maxX, y: bb.minY },
                        br: { x: bb.maxX, y: bb.maxY },
                        bl: { x: bb.minX, y: bb.maxY },
                    }[handle];
                    state.drag = {
                        mode: "scale",
                        which: handle,
                        bb,
                        snapshot,
                        anchorX: opp.x, anchorY: opp.y,
                        startCornerX: cur.x, startCornerY: cur.y,
                    };
                }
                try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
                e.preventDefault();
                return;
            }
        }

        // Click inside the transform bounding box (and NOT on a corner
        // / rotate handle, which were tested just above) → drag every
        // selected anchor as a group. This is the Nuke / AE
        // convention: the box itself is the move handle.
        if (sh && state.selection && state.selection.size >= 2) {
            const bb = selectionBBox();
            if (bb && ic.x >= bb.minX && ic.x <= bb.maxX
                  && ic.y >= bb.minY && ic.y <= bb.maxY) {
                state.drag = {
                    mode: "multi",
                    startC: { x: ic.x, y: ic.y },
                    // Snapshot the current keyframe points so the delta
                    // is relative to the start, not cumulative.
                    startPoints: getEditableKeyframe(sh).map(p => p.slice()),
                };
                try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
                e.preventDefault();
                return;
            }
        }

        if (pts) {
            // Hit-test order: ANCHOR first (so a zero-length handle
            // doesn't steal the click), then tangent handles, then
            // edge insertion. Handles that are coincident with their
            // anchor are skipped so users can't grab invisible
            // tangents — matches the visual threshold in drawHandles.
            const HANDLE_EPS = 1.0;
            // 1) Anchor points.
            for (let i = 0; i < pts.length; i++) {
                const pp = pts[i];
                if (Math.hypot(pp[0] - ic.x, pp[1] - ic.y) <= hitRadius) {
                    // Pen-tool close: while the shape is still open,
                    // clicking the FIRST anchor (with ≥3 anchors) closes
                    // it instead of starting a drag. Mirrors Photoshop /
                    // Illustrator pen tool behaviour.
                    if (sh.closed === false && i === 0 && pts.length >= 3) {
                        sh.closed = true;
                        state.activePoint = -1;
                        persist(); recordHistory(); render();
                        return;
                    }
                    if (e.altKey) {
                        // Delete point.
                        const kf = getEditableKeyframe(sh);
                        kf.splice(i, 1);
                        // Mirror the deletion across every keyframe of this
                        // shape so the topology stays consistent.
                        for (const k of Object.keys(sh.keyframes)) {
                            if (k !== String(state.currentFrame) && sh.keyframes[k].length > i) {
                                sh.keyframes[k].splice(i, 1);
                            }
                        }
                        state.activePoint = -1;
                        persist(); recordHistory(); render();
                        return;
                    }
                    // Plain (no Shift) click on ANY anchor reduces the
                    // multi-selection to just that one anchor. Matches
                    // Nuke's curve editor — clicking a point makes it
                    // THE selection; Shift+click is the only way to
                    // keep / extend a multi-selection.
                    state.selection = new Set([i]);
                    state.activePoint = i;
                    state.drag = { mode: "point", pi: i };
                    canvas.setPointerCapture(e.pointerId);
                    render();
                    return;
                }
            }
            // 2) Tangent handles — only ones that are visually rendered
            //    (i.e. non-zero-length). Coincident handles are invisible
            //    in drawHandles and we keep them un-clickable to avoid the
            //    "I clicked the point but dragged a handle" trap.
            for (let i = 0; i < pts.length; i++) {
                const pp = pts[i];
                const dxi = pp[2] - pp[0], dyi = pp[3] - pp[1];
                const dxo = pp[4] - pp[0], dyo = pp[5] - pp[1];
                const showIn  = (dxi * dxi + dyi * dyi) > HANDLE_EPS * HANDLE_EPS;
                const showOut = (dxo * dxo + dyo * dyo) > HANDLE_EPS * HANDLE_EPS;
                if (showIn && Math.hypot(pp[2] - ic.x, pp[3] - ic.y) <= hitRadius) {
                    state.activePoint = i;
                    // Capture whether the handles are currently mirrored
                    // so the drag's auto-mirror behaviour mirrors a
                    // previously-mirrored point but DOESN'T re-link a
                    // point the artist has explicitly broken.
                    state.drag = { mode: "in", pi: i, startMirrored: areHandlesMirrored(pp) };
                    canvas.setPointerCapture(e.pointerId);
                    render();
                    return;
                }
                if (showOut && Math.hypot(pp[4] - ic.x, pp[5] - ic.y) <= hitRadius) {
                    state.activePoint = i;
                    state.drag = { mode: "out", pi: i, startMirrored: areHandlesMirrored(pp) };
                    canvas.setPointerCapture(e.pointerId);
                    render();
                    return;
                }
            }
        }

        // Hit-test other shapes (clicking inside a non-active shape selects it).
        // BUT NOT while the active shape is still being drawn (open): a click
        // over another shape's fill must drop the next anchor of the shape in
        // progress, not abandon it to select the other shape. The pen-tool
        // branch below handles the append. (Shift+click still falls through to
        // the marquee/anchor logic, matching the pen branch's guard.)
        const drawingOpen = !!(sh && sh.closed === false);
        if (!(drawingOpen && !e.shiftKey)) {
        for (const cand of state.doc.shapes) {
            if (cand.id === state.activeId) continue;
            const cpts = resolveShapeAt(cand, state.currentFrame);
            if (cpts && pointInPolygon(ic, cpts)) {
                state.activeId = cand.id;
                state.activePoint = -1;
                // Multi-selection indices belong to the previous active
                // shape — clear so they don't highlight random anchors
                // / keyframes on the new one.
                state.selection = new Set();
                state.kfSelection = new Set();
                refreshSidebar();
                render();
                return;
            }
        }
        }

        // ── OPEN shape: pen-tool mode (append at cursor) ─────────────
        // Only takes effect if there's an active OPEN shape — never
        // creates a shape from a stray click. If you want a new shape,
        // use the "+ New shape" button in the sidebar; the first click
        // on the canvas after that drops its first anchor.
        if (sh && sh.closed === false && !e.shiftKey) {
            const target = sh;
            const existing = shapePointsAtCurrent();
            // Click near first point with ≥3 anchors → close the shape.
            if (existing && existing.length >= 3) {
                const p0 = existing[0];
                if (Math.hypot(p0[0] - ic.x, p0[1] - ic.y) <= hitRadius) {
                    target.closed = true;
                    state.activePoint = -1;
                    persist(); recordHistory(); render();
                    return;
                }
            }
            const kf = getEditableKeyframe(target);
            kf.push([ic.x, ic.y, ic.x, ic.y, ic.x, ic.y]);
            state.activePoint = kf.length - 1;
            // Pen-tool "drag-out handles" — same as Illustrator's pen
            // tool: clicking drops an anchor with zero-length handles
            // (sharp corner), and click-AND-drag pulls the handles out
            // symmetrically while the cursor's down (smooth curve).
            // We start a special drag mode; if the user releases without
            // moving, the handles stay at the anchor; if they drag, the
            // out-handle follows the cursor and the in-handle mirrors.
            // History is snapshotted on pointerup (one undo step covers
            // "add point + drag handles"), so we DON'T recordHistory()
            // here — endDrag treats this mode as mutating.
            persist();
            state.drag = { mode: "createHandle", pi: kf.length - 1 };
            try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
            render();
            return;
        }

        // ── CLOSED shape (or no shape): marquee-or-deselect ──────────
        // Drag = marquee selection over the active shape's anchors.
        // Click without movement = clear all selection (anchors and
        // multi-selection). Shift = additive marquee (don't clear the
        // existing selection on commit). Replaces the old implicit
        // click-to-subdivide — adding points is now an explicit
        // right-click → "Add point here" action (see contextmenu
        // handler below).
        state.marquee = { x0: ic.x, y0: ic.y, x1: ic.x, y1: ic.y };
        state.drag = {
            mode: "marquee",
            shift: !!e.shiftKey,
            couldBeClick: true,
            // Snapshot existing selection so a click without movement
            // can still distinguish "nothing changed" from "deselect".
            startSelection: new Set(state.selection),
        };
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        e.preventDefault();
        render();
    });

    canvas.addEventListener("pointermove", (e) => {
        const p = localMouse(e);
        const ic = d2i(p.x, p.y);
        if (!state.drag) return;

        // Marquee: update the rect; commit on pointerup. Doesn't touch
        // the document until release. Once we've actually moved, the
        // gesture is no longer "a click" — flip couldBeClick so the
        // release doesn't get interpreted as a deselect.
        if (state.drag.mode === "marquee") {
            if (state.marquee) {
                state.marquee.x1 = ic.x;
                state.marquee.y1 = ic.y;
                const minSpan = 2 / state.dispScale;
                if (Math.abs(state.marquee.x1 - state.marquee.x0) > minSpan
                        || Math.abs(state.marquee.y1 - state.marquee.y0) > minSpan) {
                    state.drag.couldBeClick = false;
                }
            }
            render();
            return;
        }

        const sh = activeShape();
        if (!sh) return;

        // Scale around the opposite corner: every selected anchor +
        // its tangent handles scale by (sx, sy) relative to the anchor
        // corner. Shift = uniform scale (take the larger of the two
        // factors so the user can't accidentally flatten the shape).
        if (state.drag.mode === "scale") {
            const d = state.drag;
            const kf = getEditableKeyframe(sh);
            let sx = (ic.x - d.anchorX) / (d.startCornerX - d.anchorX);
            let sy = (ic.y - d.anchorY) / (d.startCornerY - d.anchorY);
            if (!Number.isFinite(sx)) sx = 1;
            if (!Number.isFinite(sy)) sy = 1;
            if (e.shiftKey) {
                // Uniform: take the factor whose absolute value is
                // larger so the cursor still feels like it's "leading"
                // the scale.
                const m = Math.abs(sx) >= Math.abs(sy) ? sx : sy;
                sx = sy = m;
            }
            for (const i of state.selection) {
                const start = d.snapshot[i];
                if (!start || !kf[i]) continue;
                // Apply to all three positions stored per point:
                // anchor, in-handle, out-handle.
                kf[i][0] = d.anchorX + (start[0] - d.anchorX) * sx;
                kf[i][1] = d.anchorY + (start[1] - d.anchorY) * sy;
                kf[i][2] = d.anchorX + (start[2] - d.anchorX) * sx;
                kf[i][3] = d.anchorY + (start[3] - d.anchorY) * sy;
                kf[i][4] = d.anchorX + (start[4] - d.anchorX) * sx;
                kf[i][5] = d.anchorY + (start[5] - d.anchorY) * sy;
            }
            persist(); render();
            return;
        }

        // Rotation around the bounding-box centre: same per-anchor
        // application as scale, but with a rotation matrix.
        if (state.drag.mode === "rotate") {
            const d = state.drag;
            const kf = getEditableKeyframe(sh);
            const ang = Math.atan2(ic.y - d.bb.cy, ic.x - d.bb.cx) - d.startAngle;
            const cos = Math.cos(ang), sin = Math.sin(ang);
            const rotate = (x, y) => {
                const dx = x - d.bb.cx, dy = y - d.bb.cy;
                return [
                    d.bb.cx + dx * cos - dy * sin,
                    d.bb.cy + dx * sin + dy * cos,
                ];
            };
            for (const i of state.selection) {
                const start = d.snapshot[i];
                if (!start || !kf[i]) continue;
                const [ax, ay] = rotate(start[0], start[1]);
                const [ix, iy] = rotate(start[2], start[3]);
                const [ox, oy] = rotate(start[4], start[5]);
                kf[i][0] = ax; kf[i][1] = ay;
                kf[i][2] = ix; kf[i][3] = iy;
                kf[i][4] = ox; kf[i][5] = oy;
            }
            persist(); render();
            return;
        }

        // Multi-anchor drag: every selected anchor (and its tangent
        // handles) moves by the same delta from where the drag started.
        // Using startPoints avoids cumulative-drift if the user wiggles
        // back and forth.
        if (state.drag.mode === "multi") {
            const kf = getEditableKeyframe(sh);
            const dx = ic.x - state.drag.startC.x;
            const dy = ic.y - state.drag.startC.y;
            for (const i of state.selection) {
                const start = state.drag.startPoints[i];
                if (!start || !kf[i]) continue;
                kf[i][0] = start[0] + dx;
                kf[i][1] = start[1] + dy;
                kf[i][2] = start[2] + dx;
                kf[i][3] = start[3] + dy;
                kf[i][4] = start[4] + dx;
                kf[i][5] = start[5] + dy;
            }
            persist(); render();
            return;
        }

        const kf = getEditableKeyframe(sh);
        const pp = kf[state.drag.pi];
        if (!pp) return;
        // Pen tool: dragging immediately after a new-anchor click pulls
        // the out-handle to the cursor and mirrors the in-handle across
        // the anchor (symmetric tangents — the standard pen-tool feel).
        // Hold Shift to break the symmetry: only the out-handle follows
        // the cursor, the in-handle stays put.
        if (state.drag.mode === "createHandle") {
            pp[4] = ic.x; pp[5] = ic.y;
            if (!e.shiftKey) {
                pp[2] = 2 * pp[0] - ic.x;
                pp[3] = 2 * pp[1] - ic.y;
            }
            persist(); render();
            return;
        }
        if (state.drag.mode === "point") {
            // Move point AND drag tangent handles by the same delta so
            // their relative positions are preserved.
            const dx = ic.x - pp[0];
            const dy = ic.y - pp[1];
            pp[0] = ic.x; pp[1] = ic.y;
            pp[2] += dx; pp[3] += dy;
            pp[4] += dx; pp[5] += dy;
        } else if (state.drag.mode === "in") {
            pp[2] = ic.x; pp[3] = ic.y;
            // Auto-mirror only when the handles WERE mirrored at drag
            // start. If the artist has explicitly broken them (via
            // Shift+drag last time, or the right-click "Break tangents"
            // action), each handle moves independently from now on
            // until they re-link via right-click. Holding Shift always
            // suppresses mirroring as a one-shot override.
            if (!e.shiftKey && state.drag.startMirrored) {
                pp[4] = 2 * pp[0] - ic.x;
                pp[5] = 2 * pp[1] - ic.y;
            }
        } else if (state.drag.mode === "out") {
            pp[4] = ic.x; pp[5] = ic.y;
            if (!e.shiftKey && state.drag.startMirrored) {
                pp[2] = 2 * pp[0] - ic.x;
                pp[3] = 2 * pp[1] - ic.y;
            }
        }
        persist(); render();
    });

    const endDrag = (e) => {
        if (!state.drag) return;
        const wasMarquee = state.drag.mode === "marquee";
        const wasMutating = !wasMarquee && state.drag.mode !== undefined;
        // Marquee resolves to one of three outcomes:
        //   • Click without movement (couldBeClick) → DESELECT (unless
        //     Shift, which is a no-op on a non-drag).
        //   • Drag with Shift → ADDITIVE to the prior selection.
        //   • Drag without Shift → REPLACE the selection.
        if (wasMarquee && state.marquee) {
            const sh = activeShape();
            const pts = sh ? shapePointsAtCurrent() : null;
            const m = state.marquee;
            const minSpan = 2 / state.dispScale;
            const movedEnough = Math.abs(m.x1 - m.x0) > minSpan
                              || Math.abs(m.y1 - m.y0) > minSpan;
            if (!movedEnough && state.drag.couldBeClick) {
                if (!state.drag.shift) {
                    // Empty-area click → deselect everything (anchors).
                    state.selection = new Set();
                }
            } else if (sh && pts) {
                const x0 = Math.min(m.x0, m.x1);
                const y0 = Math.min(m.y0, m.y1);
                const x1 = Math.max(m.x0, m.x1);
                const y1 = Math.max(m.y0, m.y1);
                const next = state.drag.shift
                    ? new Set(state.drag.startSelection)   // additive
                    : new Set();                           // replace
                for (let i = 0; i < pts.length; i++) {
                    const pp = pts[i];
                    if (pp[0] >= x0 && pp[0] <= x1
                            && pp[1] >= y0 && pp[1] <= y1) {
                        next.add(i);
                    }
                }
                state.selection = next;
            }
        }
        state.drag = null;
        state.marquee = null;
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        // Snapshot the post-drag state as a single undo step. Many
        // pointermove events fire during the drag but we don't want
        // each to be its own undo entry — one drag = one history step
        // matches every comp / NLE. Marquees don't touch the document
        // (selection is runtime-only), so we skip the snapshot.
        if (wasMutating) recordHistory();
        render();
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // Helper for "click inside shape selects it" — uses the bezier polyline.
    function pointInPolygon(p, pts) {
        // Approximate with the anchor polyline (good enough for selection).
        let inside = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const xi = pts[i][0], yi = pts[i][1];
            const xj = pts[j][0], yj = pts[j][1];
            const intersect = ((yi > p.y) !== (yj > p.y))
                && (p.x < (xj - xi) * (p.y - yi) / (yj - yi + 1e-9) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // ── keyboard ─────────────────────────────────────────────────────
    root.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        // Ctrl/Cmd+Z = undo, Ctrl/Cmd+Shift+Z (or Ctrl+Y) = redo.
        if ((e.ctrlKey || e.metaKey) && !e.altKey) {
            if (e.key === "z" || e.key === "Z") {
                if (e.shiftKey) redo(); else undo();
                e.preventDefault();
                return;
            }
            if (e.key === "y" || e.key === "Y") {
                redo();
                e.preventDefault();
                return;
            }
        }
        let handled = true;
        switch (e.key) {
            case " ": togglePlay(); break;
            case "ArrowLeft":  setFrame(state.currentFrame - 1); break;
            case "ArrowRight": setFrame(state.currentFrame + 1); break;
            case "k": case "K": addKeyframeHere(); break;
            case "Enter": {
                // Close the active shape if it's still open and has
                // enough anchors to form a polygon.
                const sh = activeShape();
                if (sh && !sh.closed) {
                    const pts = shapePointsAtCurrent();
                    if (pts && pts.length >= 3) {
                        sh.closed = true;
                        persist(); recordHistory(); render();
                    }
                } else { handled = false; }
                break;
            }
            case "Escape": {
                // Same shortcut as before (deselect). If a shape is
                // currently being drawn AND has enough anchors, also
                // close it on the way out — common pen-tool muscle
                // memory.
                const sh = activeShape();
                let mutated = false;
                if (sh && !sh.closed) {
                    const pts = shapePointsAtCurrent();
                    if (pts && pts.length >= 3) { sh.closed = true; mutated = true; }
                }
                state.activeId = null; state.activePoint = -1;
                state.selection = new Set();
                state.kfSelection = new Set();
                persist();
                if (mutated) recordHistory();
                refreshSidebar(); render();
                break;
            }
            case "Delete": case "Backspace":
                // Priority order:
                //   1. Multi-keyframe selection → delete those kfs
                //   2. Multi-anchor selection   → delete those anchors
                //                                  from every keyframe
                //   3. Otherwise                → delete the active shape
                if (state.kfSelection.size > 0) {
                    const shKF = activeShape();
                    if (shKF) {
                        for (const f of state.kfSelection) {
                            delete shKF.keyframes[String(f)];
                        }
                        state.kfSelection = new Set();
                        persist(); recordHistory(); refreshSidebar(); render();
                    }
                    break;
                }
                if (state.selection.size > 0) {
                    const shA = activeShape();
                    if (shA) {
                        const sorted = [...state.selection].sort((a, b) => b - a);
                        for (const k of Object.keys(shA.keyframes)) {
                            const arr = shA.keyframes[k];
                            for (const idx of sorted) {
                                if (arr.length > idx) arr.splice(idx, 1);
                            }
                        }
                        state.selection = new Set();
                        state.activePoint = -1;
                        persist(); recordHistory(); render();
                    }
                } else {
                    deleteActiveShape();
                }
                break;
            default: handled = false;
        }
        if (handled) e.preventDefault();
    });

    // ── playback ─────────────────────────────────────────────────────
    // Playback wraps inside the range strip's viewport — dragging the
    // viewport handles narrower lets the artist preview a short
    // sub-range without affecting the output (the output still spans
    // every frame, this is purely a preview convenience).
    function loopRange() {
        const start = Math.max(0, Math.min(state.viewStart, state.frameCount - 1));
        const end   = Math.max(start, Math.min(state.viewEnd, state.frameCount - 1));
        return [start, end];
    }
    function nextPlaybackFrame(f) {
        const [start, end] = loopRange();
        if (end <= start) return f;
        const nxt = f + 1;
        return nxt > end ? start : nxt;
    }
    function togglePlay() {
        if (state.playing) {
            clearInterval(state.playInterval);
            state.playInterval = 0;
            state.playing = false;
            playBtn.textContent = "▶";
        } else {
            state.playing = true;
            playBtn.textContent = "⏸";
            // Snap into the loop range if we're sitting outside it,
            // otherwise the first tick would jump us into the range.
            const [start, end] = loopRange();
            if (state.currentFrame < start || state.currentFrame > end) {
                setFrame(start);
            }
            state.playInterval = setInterval(() => {
                setFrame(nextPlaybackFrame(state.currentFrame));
            }, 1000 / state.playFps);
        }
    }
    playBtn.onclick = togglePlay;
    prevBtn.onclick = () => setFrame(state.currentFrame - 1);
    nextBtn.onclick = () => setFrame(state.currentFrame + 1);
    addKeyBtn.onclick = addKeyframeHere;
    delKeyBtn.onclick = () => {
        const sh = activeShape();
        if (!sh) return;
        delete sh.keyframes[String(state.currentFrame)];
        persist(); recordHistory(); refreshSidebar(); render();
    };
    autoKeyInput.onchange = () => { state.autoKey = autoKeyInput.checked; };
    undoBtn.onclick = undo;
    redoBtn.onclick = redo;
    helpBtn.onclick = () => { helpOverlay.style.display = helpOverlay.style.display === "block" ? "none" : "block"; };
    viewSel.onchange = () => {
        try { localStorage.setItem("bat_roto_view", viewSel.value); } catch (_) {}
        render();
    };

    function addKeyframeHere() {
        const sh = activeShape();
        if (!sh) return;
        const keyStr = String(state.currentFrame);
        const pts = shapePointsAtCurrent();
        sh.keyframes[keyStr] = (pts || []).map(p => p.slice());
        persist(); recordHistory(); refreshSidebar(); render();
    }

    function setFrame(f) {
        if (state.frameCount <= 0) return;
        state.currentFrame = clamp(Math.round(f), 0, state.frameCount - 1);
        // Map to nearest cached preview frame (we may have subsampled).
        const previews = state.previewFrames || [];
        if (previews.length) {
            const stride = state.bgStride || 1;
            const idx = Math.min(previews.length - 1,
                Math.round(state.currentFrame / stride));
            const img = previews[idx];
            if (img instanceof Image) state.bgImage = img;
        }
        frameLabel.textContent = `${state.currentFrame + 1} / ${state.frameCount}`;
        // Viewport-aware playhead: clamps to the strip edges if the
        // current frame is outside the visible range. The underlying
        // currentFrame is unchanged — the artist can still scrub the
        // playhead off-screen, just won't see it until the viewport
        // catches up.
        const pct = clamp(frameToPct(state.currentFrame), 0, 100);
        timelineProgress.style.width = `${pct}%`;
        timelinePlayhead.style.left  = `${pct}%`;
        renderTimelineKeyframes();
        render();
    }

    // Timeline interaction — seek using viewport-aware px↔frame mapping
    // so scrubbing in a zoomed-in view targets the right frame.
    timelineWrap.addEventListener("pointerdown", (e) => {
        timelineWrap.setPointerCapture(e.pointerId);
        seekFromMouse(e);
    });
    timelineWrap.addEventListener("pointermove", (e) => {
        if (e.buttons & 1) seekFromMouse(e);
    });
    function seekFromMouse(e) {
        const r = timelineWrap.getBoundingClientRect();
        setFrame(Math.round(pxToFrame(e.clientX - r.left, r.width)));
    }

    // Keyframe markers live on the SEPARATE `keyframeStrip` below the
    // frame timeline. Interactions:
    //   • click            — seek to that frame AND select only this kf
    //   • shift+click      — toggle membership in the multi-selection
    //   • drag             — move every selected kf by the same delta
    //   • right-click      — delete this kf (and drop it from selection)
    //
    // Group-drag preserves the relative spacing between selected
    // keyframes. Collisions with non-selected kfs at destination
    // positions abort the move (cheaper than a confirm dialog and
    // matches NLE behaviour).
    function renderTimelineKeyframes() {
        keyframeStrip.querySelectorAll(".bat-kf").forEach(el => el.remove());
        // Also wipe any minimap dots from the range strip — we rebuild
        // them in lockstep so the artist sees keyframe positions at
        // both zoom levels.
        rangeStrip.querySelectorAll(".bat-kf-mini").forEach(el => el.remove());
        // Keep the range window visually in sync with state.viewStart/End.
        syncRangeWindow();
        const sh = activeShape();
        if (!sh || state.frameCount <= 1) return;
        // Denom is now the viewport span — that's what the keyframe
        // strip width represents. The minimap (range strip) uses the
        // FULL clip length as its denom.
        const denom = viewSpan();
        const fullDenom = Math.max(1, state.frameCount - 1);

        // Drop any stale kfSelection entries that no longer exist
        // (e.g. after an external delete via the inspector).
        for (const f of [...state.kfSelection]) {
            if (sh.keyframes[String(f)] === undefined) state.kfSelection.delete(f);
        }

        // Drag controller scoped at the strip level so any selected
        // marker dragged moves the rest in lockstep. We track which
        // marker the gesture started on, the per-frame snapshot of all
        // selected frames at drag start, and a `moved` flag so a click
        // without drag still acts as "seek".
        let drag = null;   // { startX, startFrame, basis: Map<frame, frame>, moved }

        for (const k of Object.keys(sh.keyframes)) {
            const f = Number(k);
            const inSelection = state.kfSelection.has(f);

            // Always draw a tiny dot on the minimap so the artist
            // sees where keyframes sit at full-clip scale, even when
            // they're zoomed out of the main strip.
            const miniDot = document.createElement("div");
            miniDot.className = "bat-kf-mini";
            miniDot.style.cssText = `
                position:absolute; top:1px; bottom:1px; width:2px;
                background:${inSelection ? "#7ab8ff" : "#f4b860"};
                left:${(f / fullDenom) * 100}%;
                transform:translateX(-50%); pointer-events:none;
            `;
            rangeStrip.appendChild(miniDot);

            // Skip rendering markers outside the viewport — the
            // minimap above already shows them at full scale.
            if (f < state.viewStart || f > state.viewEnd) continue;

            const dot = document.createElement("div");
            dot.className = "bat-kf";
            dot.title = `Keyframe @ frame ${f + 1} — drag to move, Shift+click to add to selection, right-click to delete`;
            dot.style.cssText = `
                position:absolute; top:0; bottom:0; width:8px;
                background:${inSelection ? "#7ab8ff" : "#f4b860"};
                border:1px solid #000;
                left:${frameToPct(f)}%;
                transform:translateX(-50%); cursor:ew-resize; z-index:5;
            `;
            dot.addEventListener("contextmenu", (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                delete sh.keyframes[k];
                state.kfSelection.delete(f);
                persist(); recordHistory(); refreshSidebar(); render();
            });

            dot.addEventListener("pointerdown", (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                if (ev.button !== 0) return;

                // Shift+click toggles selection membership without
                // starting a drag. The visual updates via re-render.
                if (ev.shiftKey) {
                    if (state.kfSelection.has(f)) state.kfSelection.delete(f);
                    else                          state.kfSelection.add(f);
                    // Safe to refresh here — shift+click isn't a drag,
                    // so tearing down the marker doesn't strand a
                    // captured pointer.
                    renderTimelineKeyframes();
                    return;
                }

                // Plain click: if this marker isn't already in the
                // multi-selection, replace the selection with just this
                // one. (Anything already in the selection stays so a
                // group drag works without holding Shift.)
                if (!state.kfSelection.has(f)) {
                    state.kfSelection = new Set([f]);
                }
                // Snapshot every selected frame's CURRENT position so
                // group drag computes new positions relative to the
                // start. Map(originalFrame → liveFrame).
                const basis = new Map();
                for (const sf of state.kfSelection) basis.set(sf, sf);
                drag = {
                    startX: ev.clientX,
                    startFrame: f,
                    basis,
                    moved: false,
                };
                dot.setPointerCapture(ev.pointerId);
                // NB: don't refreshSidebar()/render() here — those
                // tear down and rebuild the keyframe strip (markers
                // re-created from scratch), which detaches the very
                // dot we just set pointer capture on. The drag would
                // die before the first pointermove. The selection's
                // visual state catches up on pointerup.
            });

            dot.addEventListener("pointermove", (ev) => {
                if (!drag) return;
                if (Math.abs(ev.clientX - drag.startX) > 2) drag.moved = true;
                const r = keyframeStrip.getBoundingClientRect();
                // Viewport-aware: dxFrame is the delta within the
                // CURRENT visible span, not the full clip. Otherwise
                // moving the cursor 1 strip-width in a zoomed-in view
                // would still cover the whole clip.
                const dxFrac = (ev.clientX - drag.startX) / Math.max(1, r.width);
                const dxFrame = Math.round(dxFrac * viewSpan());
                keyframeStrip.querySelectorAll(".bat-kf").forEach((el) => {
                    const fromFrame = Number(el.dataset.frame ?? "-1");
                    if (drag.basis.has(fromFrame)) {
                        const nf = clamp(fromFrame + dxFrame, 0, state.frameCount - 1);
                        el.style.left = `${frameToPct(nf)}%`;
                        el.title = `Keyframe → frame ${nf + 1} (release to drop)`;
                    }
                });
            });

            dot.dataset.frame = String(f);

            dot.addEventListener("pointerup", (ev) => {
                if (!drag) return;
                try { dot.releasePointerCapture(ev.pointerId); } catch (_) {}
                if (!drag.moved) {
                    // Click without drag — seek the playhead AND make
                    // this the sole selected keyframe.
                    state.kfSelection = new Set([drag.startFrame]);
                    setFrame(drag.startFrame);
                    drag = null;
                    return;
                }
                // Compute the new frame for every selected kf. Same
                // viewport-aware delta as the live preview above.
                const r = keyframeStrip.getBoundingClientRect();
                const dxFrac = (ev.clientX - drag.startX) / Math.max(1, r.width);
                const dxFrame = Math.round(dxFrac * viewSpan());
                if (dxFrame === 0) { drag = null; refreshSidebar(); render(); return; }
                // Build new-from-old map; abort if any new slot collides
                // with a non-selected existing kf.
                const moves = [];
                const newSet = new Set();
                let collision = false;
                for (const sf of state.kfSelection) {
                    const nf = clamp(sf + dxFrame, 0, state.frameCount - 1);
                    if (newSet.has(nf)) { collision = true; break; }
                    newSet.add(nf);
                    moves.push([sf, nf]);
                }
                // Check against existing un-selected keyframes.
                if (!collision) {
                    for (const [, nf] of moves) {
                        if (state.kfSelection.has(nf)) continue;
                        if (sh.keyframes[String(nf)] !== undefined
                                && !state.kfSelection.has(nf)) {
                            collision = true; break;
                        }
                    }
                }
                if (collision) {
                    // Snap markers back via re-render; no state change.
                    drag = null;
                    refreshSidebar(); render();
                    return;
                }
                // Apply: pull selected payloads aside, delete originals,
                // write to new positions. Two-phase so we don't trample
                // ourselves when keyframes shift into each other's slots.
                const stash = new Map();
                for (const [sf,] of moves) {
                    stash.set(sf, sh.keyframes[String(sf)]);
                    delete sh.keyframes[String(sf)];
                }
                state.kfSelection = new Set();
                for (const [sf, nf] of moves) {
                    sh.keyframes[String(nf)] = stash.get(sf);
                    state.kfSelection.add(nf);
                }
                drag = null;
                persist(); recordHistory(); refreshSidebar(); render();
            });

            dot.addEventListener("pointercancel", (ev) => {
                if (drag) { drag = null; refreshSidebar(); render(); }
            });

            keyframeStrip.appendChild(dot);
        }
    }

    function deleteActiveShape() {
        const sh = activeShape();
        if (!sh) return;
        state.doc.shapes = state.doc.shapes.filter(s => s.id !== sh.id);
        state.activeId = null;
        persist(); recordHistory(); refreshSidebar(); render();
    }

    // ── sidebar (shape list + per-shape controls) ────────────────────
    function refreshSidebar() {
        sidebar.innerHTML = "";

        // Node-scope: a global feather applied to the unioned mask
        // AFTER every shape is composited. Distinct from per-shape
        // feather (which softens each shape's edge before union) —
        // useful for blurring the whole roto silhouette with one knob.
        const globalRow = document.createElement("div");
        globalRow.style.cssText = `
            display:flex; flex-direction:column; gap:1px; padding:5px;
            background:#15181d; border:1px solid #2a2f37; border-radius:4px;
            margin-bottom:6px;
        `;
        const globalHeader = document.createElement("div");
        globalHeader.style.cssText = "display:flex; justify-content:space-between; color:#9ab; font-size:10px;";
        const globalLabel = document.createElement("span");
        globalLabel.textContent = "Global feather (px)";
        globalLabel.title = "Gaussian blur applied to the FINAL unioned mask. Combine with per-shape feather for finer control.";
        const globalValue = document.createElement("span");
        globalValue.style.cssText = "font-family:monospace; color:#cde;";
        const gf = state.doc.global_feather ?? 0;
        globalValue.textContent = `${gf.toFixed(1)}`;
        globalHeader.append(globalLabel, globalValue);
        const globalSlider = document.createElement("input");
        globalSlider.type = "range";
        globalSlider.min = "0"; globalSlider.max = "100"; globalSlider.step = "0.5";
        globalSlider.value = String(gf);
        globalSlider.style.width = "100%";
        globalSlider.oninput = () => {
            const v = parseFloat(globalSlider.value) || 0;
            state.doc.global_feather = v;
            globalValue.textContent = `${v.toFixed(1)}`;
            persist();
            render();
        };
        globalSlider.onchange = () => recordHistory();
        globalSlider.onclick = (ev) => ev.stopPropagation();
        globalRow.append(globalHeader, globalSlider);
        sidebar.appendChild(globalRow);

        const newBtn = document.createElement("button");
        newBtn.textContent = "+ New shape";
        newBtn.style.cssText = `
            padding:5px; background:#1e2530; border:1px solid #2a3a55;
            color:#cde; border-radius:3px; cursor:pointer; font-size:11px;
        `;
        newBtn.onclick = () => {
            const sh = makeShape();
            sh.name = `shape ${state.doc.shapes.length + 1}`;
            state.doc.shapes.push(sh);
            state.activeId = sh.id;
            state.activePoint = -1;
            state.selection = new Set();
            state.kfSelection = new Set();
            persist(); recordHistory(); refreshSidebar(); render();
        };
        sidebar.appendChild(newBtn);

        const list = document.createElement("div");
        list.style.cssText = "display:flex; flex-direction:column; gap:3px; margin-top:6px;";
        sidebar.appendChild(list);

        for (const sh of state.doc.shapes) {
            const isActive = sh.id === state.activeId;
            const row = document.createElement("div");
            row.style.cssText = `
                padding:5px 5px 6px 5px;
                border:1px solid ${isActive ? "#7ab8ff" : "#2a2f37"};
                border-radius:4px; cursor:pointer;
                background:${isActive ? "rgba(76,158,255,0.10)" : "#1a1d22"};
            `;

            // ── header: two-row layout ──
            // Row 1: swatch + name input (full width — the sidebar is
            // narrow and shape names get squished otherwise).
            // Row 2: order / visibility / delete buttons.
            const head = document.createElement("div");
            head.style.cssText = "display:flex; flex-direction:column; gap:4px;";
            const headTop = document.createElement("div");
            headTop.style.cssText = "display:flex; align-items:center; gap:5px;";
            const headBtns = document.createElement("div");
            headBtns.style.cssText = "display:flex; align-items:center; justify-content:flex-end; gap:2px;";
            const swatch = document.createElement("span");
            swatch.style.cssText = `
                width:12px; height:12px; background:${sh.color};
                border:1px solid #000; border-radius:2px; flex-shrink:0;
                cursor:pointer;
            `;
            swatch.title = "Click to change the on-canvas display colour (mask intensity is controlled by Opacity below)";
            // Click the swatch → pop a hidden colour input.
            const colorInput = document.createElement("input");
            colorInput.type = "color"; colorInput.value = sh.color;
            colorInput.style.cssText = "position:absolute; opacity:0; width:0; height:0;";
            swatch.appendChild(colorInput);
            swatch.onclick = (ev) => { ev.stopPropagation(); colorInput.click(); };
            colorInput.onchange = () => {
                sh.color = colorInput.value;
                persist(); recordHistory(); refreshSidebar(); render();
            };

            const name = document.createElement("input");
            name.type = "text";
            name.value = sh.name;
            name.title = "Shape name — Enter to confirm";
            name.style.cssText = `
                flex:1; min-width:0; background:#0f1217;
                border:1px solid ${isActive ? "#3a4a66" : "#2a2f37"};
                color:#cde; font-size:12px; padding:3px 5px; border-radius:3px;
            `;
            // Don't let typing in the name input bubble up to the
            // canvas keyboard handler (otherwise Space pauses playback
            // while you're typing a name with spaces).
            name.onkeydown = (ev) => ev.stopPropagation();
            name.onclick = (ev) => ev.stopPropagation();
            const commit = () => {
                const v = name.value.trim();
                if (v && v !== sh.name) { sh.name = v; persist(); recordHistory(); }
            };
            name.onblur = commit;
            name.onkeyup = (ev) => { if (ev.key === "Enter") { name.blur(); } };

            // Eye / show-hide toggle. Hidden shapes don't render on the
            // canvas and don't contribute to the output mask.
            // Inline SVGs (Lucide-style open eye vs slashed eye) — more
            // evocative than the previous ●/○ dots and they pick up
            // the button's current color via currentColor.
            const SVG_EYE_OPEN = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
                     stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/>
                    <circle cx="12" cy="12" r="3"/>
                </svg>`;
            const SVG_EYE_CLOSED = `
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none"
                     stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                    <path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
                    <line x1="1" y1="1" x2="23" y2="23"/>
                </svg>`;
            const eye = document.createElement("button");
            const setEyeIcon = () => {
                const hidden = sh.visible === false;
                eye.innerHTML = hidden ? SVG_EYE_CLOSED : SVG_EYE_OPEN;
                eye.title = hidden ? "Show shape" : "Hide shape";
                eye.style.color = hidden ? "#666" : "#cde";
            };
            eye.style.cssText = "background:none; border:none; cursor:pointer; padding:1px 4px; display:inline-flex; align-items:center;";
            setEyeIcon();
            eye.onclick = (ev) => {
                ev.stopPropagation();
                sh.visible = sh.visible === false ? true : false;
                setEyeIcon();
                persist(); recordHistory(); render();
            };

            // Raise / lower — move this shape later / earlier in the
            // shapes array. Shapes are composited in array order, so the
            // LAST shape paints on top; "raise" moves toward index N-1.
            // Buttons gray out at the bounds.
            const idx = state.doc.shapes.indexOf(sh);
            const mkOrderBtn = (label, title, delta, disabled) => {
                const b = document.createElement("button");
                b.textContent = label; b.title = title;
                b.style.cssText = `
                    background:none; border:none; color:${disabled ? "#444" : "#888"};
                    cursor:${disabled ? "default" : "pointer"};
                    padding:0 3px; font-size:11px; line-height:1;
                `;
                if (!disabled) {
                    b.onmouseover = () => { b.style.color = "#cde"; };
                    b.onmouseout  = () => { b.style.color = "#888"; };
                    b.onclick = (ev) => {
                        ev.stopPropagation();
                        const i = state.doc.shapes.indexOf(sh);
                        const j = i + delta;
                        if (j < 0 || j >= state.doc.shapes.length) return;
                        const arr = state.doc.shapes;
                        [arr[i], arr[j]] = [arr[j], arr[i]];
                        persist(); recordHistory(); refreshSidebar(); render();
                    };
                }
                return b;
            };
            const upBtn = mkOrderBtn("▲", "Raise (paint on top of shapes below)", +1,
                                     idx >= state.doc.shapes.length - 1);
            const downBtn = mkOrderBtn("▼", "Lower (paint below shapes above)", -1,
                                       idx <= 0);

            const del = document.createElement("button");
            del.textContent = "✕"; del.title = "Delete shape";
            del.style.cssText = "background:none; border:none; color:#888; cursor:pointer; padding:0 4px; font-size:14px;";
            del.onmouseover = () => { del.style.color = "#ff8c8c"; };
            del.onmouseout  = () => { del.style.color = "#888"; };
            del.onclick = (ev) => {
                ev.stopPropagation();
                state.doc.shapes = state.doc.shapes.filter(s => s.id !== sh.id);
                if (state.activeId === sh.id) state.activeId = null;
                persist(); recordHistory(); refreshSidebar(); render();
            };
            headTop.append(swatch, name);
            headBtns.append(downBtn, upBtn, eye, del);
            head.append(headTop, headBtns);
            row.appendChild(head);

            // Clicking the row body (outside the name input) selects it.
            // Suppress the switch while the artist is mid-draw on the
            // active shape — otherwise clicking anywhere in the sidebar
            // (incl. accidental clicks while reaching for the canvas)
            // ends the draw, leaving an open shape behind.
            row.onclick = () => {
                if (state.activeId !== sh.id) {
                    const active = activeShape();
                    if (active && active.closed === false) return;
                    state.selection = new Set();
                    state.kfSelection = new Set();
                }
                state.activeId = sh.id;
                state.activePoint = -1;
                refreshSidebar(); render();
            };

            if (isActive) {
                // ── stacked per-shape controls ──
                const ctrls = document.createElement("div");
                ctrls.style.cssText = "display:flex; flex-direction:column; gap:4px; margin-top:6px; font-size:10px;";

                const labeledSlider = (labelText, current, min, max, step, onChange, fmt = v => v) => {
                    const wrap = document.createElement("div");
                    wrap.style.cssText = "display:flex; flex-direction:column; gap:1px;";
                    const head = document.createElement("div");
                    head.style.cssText = "display:flex; justify-content:space-between; color:#9ab;";
                    const lab = document.createElement("span");
                    lab.textContent = labelText;
                    const val = document.createElement("span");
                    val.style.cssText = "font-family:monospace; color:#cde;";
                    val.textContent = fmt(current);
                    head.append(lab, val);
                    const slider = document.createElement("input");
                    slider.type = "range";
                    slider.min = String(min); slider.max = String(max); slider.step = String(step);
                    slider.value = String(current);
                    slider.style.width = "100%";
                    slider.oninput = () => {
                        const v = parseFloat(slider.value);
                        val.textContent = fmt(v);
                        onChange(v);
                    };
                    // onchange fires once when the artist releases the
                    // slider — that's the right granularity for the
                    // undo history (one entry per drag, not per
                    // pixel).
                    slider.onchange = () => recordHistory();
                    slider.onclick = (ev) => ev.stopPropagation();
                    wrap.append(head, slider);
                    return wrap;
                };

                // "Mask value" — the grey level this shape paints
                // (0=black, 1=white). "Opacity" — how strongly that
                // value contributes vs the rest of the scene. Final
                // per-pixel contribution = value × opacity, but you can
                // animate them independently. `value` defaults to 1.0
                // so legacy state JSON (which only had `opacity`) still
                // produces the same final intensity.
                const valCtl = labeledSlider("Mask value", (sh.value ?? 1),
                    0, 1, 0.01,
                    v => { sh.value = v; persist(); render(); },
                    v => Number(v).toFixed(2));
                const opCtl = labeledSlider("Opacity", (sh.opacity ?? 1),
                    0, 1, 0.01,
                    v => { sh.opacity = v; persist(); render(); },
                    v => Number(v).toFixed(2));
                const ftCtl = labeledSlider("Feather (px)", sh.feather ?? 0, 0, 100, 0.5,
                    v => { sh.feather = v; persist(); render(); },
                    v => `${v.toFixed(1)}`);

                const invRow = document.createElement("label");
                invRow.style.cssText = "display:flex; gap:5px; align-items:center; color:#cde; cursor:pointer;";
                const invCb = document.createElement("input");
                invCb.type = "checkbox"; invCb.checked = !!sh.invert;
                invCb.onclick = (ev) => ev.stopPropagation();
                invCb.onchange = () => {
                    sh.invert = invCb.checked;
                    persist(); recordHistory(); render();
                };
                invRow.append(invCb, document.createTextNode("Invert"));

                // Select-all-anchors button — populates the
                // multi-selection with every anchor of this shape so
                // the artist can drag-to-move the whole curve or
                // press Delete to clear every anchor at once.
                const selRow = document.createElement("div");
                selRow.style.cssText = "display:flex; gap:4px; margin-top:2px;";
                const selAll = document.createElement("button");
                selAll.textContent = "Select all anchors";
                selAll.title = "Add every anchor of this shape to the multi-selection (drag any to move all, Delete to remove all)";
                selAll.style.cssText = `
                    flex:1; padding:3px 5px; background:#1e2530;
                    border:1px solid #2a3a55; color:#cde; border-radius:3px;
                    cursor:pointer; font-size:10px;
                `;
                selAll.onclick = (ev) => {
                    ev.stopPropagation();
                    const pts = shapePointsAtCurrent();
                    if (!pts) return;
                    state.selection = new Set(pts.map((_, i) => i));
                    render();
                };
                const selClear = document.createElement("button");
                selClear.textContent = "Clear";
                selClear.title = "Clear the multi-selection";
                selClear.style.cssText = `
                    padding:3px 6px; background:#1a1d22;
                    border:1px solid #2a2f37; color:#9ab; border-radius:3px;
                    cursor:pointer; font-size:10px;
                `;
                selClear.onclick = (ev) => {
                    ev.stopPropagation();
                    state.selection = new Set();
                    render();
                };
                selRow.append(selAll, selClear);

                ctrls.append(valCtl, opCtl, ftCtl, invRow, selRow);
                row.appendChild(ctrls);
            }
            list.appendChild(row);
        }
        renderTimelineKeyframes();
    }

    // ── State JSON inspector ─────────────────────────────────────────
    // A collapsible panel under the canvas that exposes the entire
    // node state as JSON. Useful for: (1) confirming persistence is
    // alive (you can see the data that ships with the workflow JSON),
    // (2) copy/pasting a roto setup between nodes. Editing the JSON
    // and pressing Apply replaces the in-memory state and re-renders.
    const inspector = document.createElement("div");
    inspector.style.cssText = `
        border-top:1px solid #222; background:#0e1116;
        font:11px sans-serif;
    `;
    const inspectorHeader = document.createElement("div");
    inspectorHeader.style.cssText = `
        display:flex; align-items:center; gap:6px; padding:4px 8px;
        cursor:pointer; user-select:none; color:#9ab;
    `;
    const inspectorChevron = document.createElement("span");
    inspectorChevron.textContent = "▸";
    inspectorChevron.style.cssText = "font-size:9px; width:10px; display:inline-block;";
    const inspectorLabel = document.createElement("span");
    inspectorLabel.textContent = "State JSON (shapes · keyframes · global feather)";
    inspectorLabel.style.flex = "1";
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.title = "Copy the full state JSON to the clipboard";
    copyBtn.style.cssText = "background:none; border:1px solid #2a2f37; color:#cdd; padding:2px 6px; border-radius:3px; cursor:pointer; font-size:10px;";
    const pasteBtn = document.createElement("button");
    pasteBtn.textContent = "Paste";
    pasteBtn.title = "Replace state with the JSON from the clipboard";
    pasteBtn.style.cssText = copyBtn.style.cssText;
    inspectorHeader.append(inspectorChevron, inspectorLabel, copyBtn, pasteBtn);

    const inspectorBody = document.createElement("div");
    inspectorBody.style.cssText = "display:none; padding:6px 8px; gap:6px; flex-direction:column;";
    const inspectorArea = document.createElement("textarea");
    inspectorArea.spellcheck = false;
    inspectorArea.style.cssText = `
        width:100%; min-height:120px; max-height:280px;
        background:#0a0d12; color:#cde; border:1px solid #2a2f37;
        border-radius:3px; font:11px monospace; padding:4px 6px;
        resize:vertical; box-sizing:border-box; white-space:pre;
    `;
    const inspectorMsg = document.createElement("div");
    inspectorMsg.style.cssText = "color:#9ab; font-size:10px;";
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply edits";
    applyBtn.title = "Parse the JSON and replace the state. Validates before applying.";
    applyBtn.style.cssText = "background:#1e2530; border:1px solid #2a3a55; color:#cde; padding:3px 8px; border-radius:3px; cursor:pointer; font-size:11px; align-self:flex-start;";
    inspectorBody.append(inspectorArea, inspectorMsg, applyBtn);
    inspector.append(inspectorHeader, inspectorBody);
    root.appendChild(inspector);

    function inspectorRefresh() {
        try { inspectorArea.value = JSON.stringify(state.doc, null, 2); }
        catch (_) { inspectorArea.value = ""; }
    }
    function inspectorToggle(open) {
        const willOpen = open ?? inspectorBody.style.display === "none";
        inspectorBody.style.display = willOpen ? "flex" : "none";
        inspectorChevron.textContent = willOpen ? "▾" : "▸";
        if (willOpen) inspectorRefresh();
    }
    inspectorHeader.addEventListener("click", (ev) => {
        if (ev.target === copyBtn || ev.target === pasteBtn) return;
        inspectorToggle();
    });
    copyBtn.onclick = async (ev) => {
        ev.stopPropagation();
        const text = JSON.stringify(state.doc, null, 2);
        try {
            await navigator.clipboard.writeText(text);
            copyBtn.textContent = "✓"; setTimeout(() => copyBtn.textContent = "Copy", 900);
        } catch (_) {
            // Fallback: open the panel + select-all so the artist can
            // Ctrl+C manually.
            inspectorToggle(true);
            inspectorArea.value = text;
            inspectorArea.focus(); inspectorArea.select();
        }
    };
    pasteBtn.onclick = async (ev) => {
        ev.stopPropagation();
        try {
            const text = await navigator.clipboard.readText();
            applyJsonText(text);
        } catch (_) {
            inspectorToggle(true);
            inspectorMsg.textContent = "Clipboard read blocked — paste into the textarea and click Apply.";
            inspectorMsg.style.color = "#f4b860";
        }
    };
    applyBtn.onclick = () => applyJsonText(inspectorArea.value);

    function applyJsonText(text) {
        let parsed;
        try {
            parsed = JSON.parse(text);
        } catch (e) {
            inspectorMsg.textContent = `Not valid JSON: ${e.message}`;
            inspectorMsg.style.color = "#ff8c8c";
            return;
        }
        if (typeof parsed !== "object" || parsed === null || !Array.isArray(parsed.shapes)) {
            inspectorMsg.textContent = "Expected an object with a `shapes` array.";
            inspectorMsg.style.color = "#ff8c8c";
            return;
        }
        state.doc = _hydrateLoadedDoc(parsed);
        state.activeId = null;
        state.activePoint = -1;
        state.selection = new Set();
        persist(); recordHistory(); refreshSidebar(); render();
        inspectorRefresh();
        inspectorMsg.textContent = `Applied ${state.doc.shapes.length} shape(s).`;
        inspectorMsg.style.color = "#7ed957";
    }

    // Refresh the inspector textarea on every action boundary. We
    // hook into recordHistory rather than persist so a slider drag
    // (which calls persist hundreds of times per second) doesn't
    // thrash the textarea — recordHistory only fires when the user
    // commits.
    const _origRecordHistory = recordHistory;
    recordHistory = function () {
        _origRecordHistory();
        if (inspectorBody.style.display !== "none") inspectorRefresh();
    };

    // ── ingest from backend (input image batch) ──────────────────────
    // `frames` is a possibly-subsampled list of JPEG thumbnails (the
    // backend caps the payload to ~240 entries; `stride` and the real
    // `frameCount` are reported separately so we still let the artist
    // scrub through every actual frame of the source — we just snap
    // the bg image to the nearest cached preview).
    node._batRotoIngest = async (frames, w, h, frameCount, stride) => {
        state.imgW = w; state.imgH = h;
        state.bgStride = Math.max(1, stride || 1);
        state.previewFrames = await Promise.all(frames.map(b64 => new Promise((res, rej) => {
            const im = new Image();
            im.onload = () => res(im);
            im.onerror = rej;
            im.src = `data:image/jpeg;base64,${b64}`;
        })));
        state.frameCount = Math.max(1, frameCount || state.previewFrames.length || 1);
        // Reset the viewport on a new ingest — the artist almost
        // certainly wants to see the whole new clip first; they can
        // re-zoom from there.
        state.viewStart = 0;
        state.viewEnd = Math.max(0, state.frameCount - 1);
        hint.style.display = state.previewFrames.length ? "none" : "block";
        // Persist image dimensions inside the state JSON so shape
        // positions render at correct relative scale on workflow reload
        // even on a different machine where the localStorage cache is
        // absent. Cheap (two integers) and ships with the workflow.
        state.doc.imgW = w; state.doc.imgH = h;
        state.doc.frameCount = state.frameCount;
        persist();
        // Cache the first frame's thumbnail locally so the next time
        // this workflow opens on THIS machine the canvas already shows
        // something instead of black-and-waiting-for-Run.
        if (frames.length > 0) {
            _saveCachedPreview(node, {
                firstFrame: frames[0],
                imgW: w, imgH: h, frameCount: state.frameCount,
            });
        }
        setFrame(Math.min(state.currentFrame, state.frameCount - 1));
        refreshSidebar();
    };

    // ── restore from cache on init ──────────────────────────────────
    // Two layers of restoration when no Run has happened yet:
    //   1. state.doc.imgW/imgH (shipped with the workflow JSON) —
    //      ensures shape positions are at the correct relative scale.
    //   2. localStorage thumb (per-machine) — gives a visible bg too.
    function _restoreCachedPreview() {
        // 1) dimensions from the workflow state.
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW;
            state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
            // Default the viewport to the full range when restoring
            // from a workflow load; we don't persist the viewport
            // because it's a UI preference, not a creative decision.
            state.viewStart = 0;
            state.viewEnd = Math.max(0, state.frameCount - 1);
        }
        // 2) thumbnail from localStorage.
        const cached = _loadCachedPreview(node);
        if (cached?.firstFrame) {
            const im = new Image();
            im.onload = () => {
                state.previewFrames = [im];
                state.bgImage = im;
                state.bgStride = Math.max(1, state.frameCount); // 1 cached frame for the whole range
                if (cached.imgW) state.imgW = cached.imgW;
                if (cached.imgH) state.imgH = cached.imgH;
                hint.style.display = "none";
                setFrame(state.currentFrame);
                render();
            };
            im.src = `data:image/jpeg;base64,${cached.firstFrame}`;
        }
        // Re-render once even if no cache so the persisted shape
        // dimensions take effect immediately.
        render();
    }

    // Initial paint (state restored from workflow). The restore call
    // below will re-paint with the cached bg + dimensions if we have
    // them; otherwise render() runs against the empty canvas.
    refreshSidebar();
    recordHistory();   // seed entry so the first user action has a target to undo to.
    // Tracked so the observer, the playback interval and the retained preview
    // frames are all released when the node is deleted (see bat_lifecycle).
    const track = batTrack(node);
    track.observer(new ResizeObserver(render), canvasWrap);
    track.dispose(() => {
        // Playback kept firing setFrame() -> render() against a detached canvas.
        if (state.playInterval) {
            clearInterval(state.playInterval);
            state.playInterval = null;
        }
        // Up to a few hundred decoded Image objects per node otherwise stay
        // reachable from the closure for the rest of the session.
        state.previewFrames = null;
        state.img = null;
    });

    // Display-only zoom control (bottom-left of the canvas). Lets the artist
    // pull back to see and roto shapes that extend past the frame edge.
    attachZoomControl({ wrap: canvasWrap, canvas, state, onChange: render, corner: "bl" });

    _restoreCachedPreview();

    // Expose a "reload from widget" hook so the registerExtension
    // wrapper can re-sync after ComfyUI restores widgets_values from
    // a workflow JSON. The state widget value is set by ComfyUI AFTER
    // onNodeCreated runs, so the canvas would otherwise stay frozen
    // at the default empty state even though the persisted data is
    // sitting in the widget. Calling this in onConfigure brings the
    // canvas back in sync without losing in-progress edits made
    // BEFORE the workflow load (rare but possible) — we always trust
    // the widget value as the source of truth at load time.
    node._batRotoReloadFromWidget = () => {
        try {
            const raw = W("state")?.value || "";
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (!parsed || !Array.isArray(parsed.shapes)) return;
            state.doc = _hydrateLoadedDoc(parsed);
        } catch (_) { return; }
        // Reset runtime-only state that may reference indices /
        // shape ids that no longer match the loaded document.
        state.activeId = null;
        state.activePoint = -1;
        state.selection = new Set();
        state.kfSelection = new Set();
        state.drag = null;
        state.marquee = null;
        // Reset undo history to the loaded state — restore-then-undo
        // back to "blank" would surprise the artist on reload.
        history.snapshots = [];
        history.pointer = -1;
        recordHistory();
        // If the persisted state carries image dimensions, lift them
        // over to runtime so positions render at correct scale even
        // before Run. (Cached-thumb restore is fired separately below.)
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW;
            state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
        }
        // Reset the viewport on a workflow reload — viewport state
        // is a UI preference, not persisted, so we always come back
        // to the full clip view.
        state.viewStart = 0;
        state.viewEnd = Math.max(0, state.frameCount - 1);
        setFrame(0);
        refreshSidebar();
        render();
        // Also retry the cached-thumb restore — useful when the node
        // is loaded fresh and we hadn't yet attempted it with the
        // restored dimensions.
        _restoreCachedPreview();
    };

    return root;
}

app.registerExtension({
    name: "Bat_Roto",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Hide the raw state widget from the node body — it stores
            // the JSON blob but we never want the artist editing it
            // directly. Setting type:"hidden" alone leaks the value as
            // raw text on some ComfyUI front-end builds, so we also
            // collapse its layout slot AND no-op its draw.
            const stateW = this.widgets?.find(w => w.name === "state");
            if (stateW) {
                stateW.type = "hidden";
                stateW.computeSize = () => [0, -4];
                stateW.draw = () => {};
                stateW.hidden = true;
            }

            const el = buildEditor(this);
            // Dual-mode sizing: Nodes 2.0 derives node height from
            // computeLayoutSize, so a bare addDOMWidget + this.size left the
            // node and the widget disagreeing (grey band, no resize).
            addBatDOMWidget(this, "bat_roto_editor", "bat_roto_editor", el, {
                minWidth: 640, height: 540, growable: true,
            });
            clampNodeSize(this, 640, 540);
            return r;
        };

        // onConfigure fires after ComfyUI has restored widgets_values
        // from the workflow JSON — i.e. after the `state` widget now
        // actually holds the artist's saved roto. We re-read it into
        // the in-memory state.doc so the canvas reflects what was
        // saved. Without this hook the canvas stays empty on every
        // workflow reload / F5, even though the data is intact on
        // disk, which is exactly the "did I lose my work?" trap.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // Defer one tick so any other extension's onConfigure also
            // has a chance to settle. setTimeout(0) is sufficient —
            // we just need to escape the current call stack.
            const node = this;
            setTimeout(() => { node._batRotoReloadFromWidget?.(); }, 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batRotoIngest) return r;
            const frames = message.frames || [];
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const w = one(message.w) || 0;
            const h = one(message.h) || 0;
            const fc = one(message.frame_count) || frames.length || 1;
            const stride = one(message.stride) || 1;
            if (frames.length && w && h) this._batRotoIngest(frames, w, h, fc, stride);
            return r;
        };
    },
});
