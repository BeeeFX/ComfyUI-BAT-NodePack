/**
 * Bat_AnimatedCrop — same rect editor as Bat_Crop but the rect lives
 * in a per-frame keyframe map; a timeline at the bottom drives scrub +
 * playback + keyframe management exactly like Bat_Roto.
 *
 * Persistent state (hidden STRING widget `state`):
 *   {
 *     "keyframes": {
 *       "<frame>": { x, y, w, h, angle },
 *       ...
 *     }
 *   }
 *
 * Auto-key ON  → every rect edit creates / replaces the keyframe at the
 *                current frame.
 * Auto-key OFF → edits modify the nearest keyframe ≤ current frame
 *                (or the only existing keyframe).
 *
 * Rect interaction matches Bat_Crop's: 8 sizing handles, rotation
 * handle, rotation-aware resize math, aspect-ratio lock. The drag math
 * is lifted from bat_crop.js — only the "where do we write the
 * resulting numbers" differs (here we mutate a JSON state instead of
 * separate INT widgets).
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { batTrack, batNodeCacheKey } from "./bat_lifecycle.js";
import { attachZoomControl } from "./bat_zoom_control.js";

const NODE_TYPE = "Bat_AnimatedCrop";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// localStorage preview cache, keyed by node id. Same idea as in
// bat_roto.js — gives us a bg thumbnail on reopen without bloating the
// workflow JSON.
function _previewCacheKey(node) {
    // Workflow-scoped: keying on node.id ALONE collided across graphs — opening
    // another workflow whose node 14 is also this node type restored the wrong
    // shot's plate (at the wrong imgW/imgH, so shapes drew against a bogus
    // reference). batNodeCacheKey folds in the workflow identity.
    return batNodeCacheKey(app, "bat_animcrop_preview", node);
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

function parseRatio(s) {
    if (!s || typeof s !== "string") return null;
    s = s.trim().toLowerCase();
    if (s === "" || s === "free") return null;
    if (s.includes(":")) {
        const [a, b] = s.split(":").map(Number);
        if (a > 0 && b > 0) return a / b;
        return null;
    }
    const n = Number(s);
    return n > 0 ? n : null;
}

function buildEditor(node) {
    const root = document.createElement("div");
    root.tabIndex = 0;
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px;
        outline:none; min-height:480px; font:11px sans-serif; color:#cde;
    `;

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative; flex:1; min-height:0; background:#000;";
    root.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; display:block; touch-action:none;";
    canvasWrap.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `
        position:absolute; left:8px; top:6px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;
    `;
    hint.textContent = "Run once to load the frames. Then drag a rect to crop.";
    canvasWrap.appendChild(hint);

    const info = document.createElement("div");
    info.style.cssText = `
        position:absolute; right:8px; top:6px; font:11px monospace;
        color:#cde; background:rgba(0,0,0,0.55); padding:2px 6px;
        border-radius:3px; pointer-events:none;
    `;
    canvasWrap.appendChild(info);

    const ctx = canvas.getContext("2d");

    // ── transport: full-width timeline on top, controls row beneath ───
    // The timeline strips used to sit INLINE among the buttons in a single
    // flex row, so the scrub bar got whatever width was left over — often
    // ~200px for a 240-frame range (under 1px per frame, nearly impossible to
    // scrub or hit a keyframe). Stacking it gives the timeline the editor's
    // full width and matches a classic player (timeline above, transport
    // below).
    const transport = document.createElement("div");
    transport.style.cssText = `
        display:flex; flex-direction:column; gap:4px; padding:4px 6px;
        background:#15181d; border-top:1px solid #222;
    `;
    root.appendChild(transport);

    // Row that holds the buttons / auto-key / frame counter, under the timeline.
    const controlsRow = document.createElement("div");
    controlsRow.style.cssText = "display:flex; align-items:center; gap:6px;";

    const btn = (txt, title) => {
        const b = document.createElement("button");
        b.textContent = txt; b.title = title;
        // white-space:nowrap + inline-flex keep multi-glyph icons (|◀, ▶|)
        // on a single line at any narrow width.
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

    const playBtn   = btn("▶", "Play / Pause (Space)");
    const prevBtn   = btn("|◀", "Previous frame (←)");
    const nextBtn   = btn("▶|", "Next frame (→)");
    const addKeyBtn = btn("◆+", "Add keyframe at current frame (K)");
    const delKeyBtn = btn("◆-", "Delete keyframe at current frame");

    const autoKeyToggle = document.createElement("label");
    autoKeyToggle.style.cssText = "display:flex; gap:4px; align-items:center; cursor:pointer; padding:0 4px;";
    const autoKeyInput = document.createElement("input");
    autoKeyInput.type = "checkbox"; autoKeyInput.checked = true;
    autoKeyToggle.append(autoKeyInput, document.createTextNode("Auto-key"));

    // Two stacked strips: FRAME (playhead, seek) on top, KEYFRAMES on
    // bottom. Mirrors Bat_Roto — sharing one strip the previous way
    // meant clicking a keyframe marker also snapped the playhead,
    // which made keyframe dragging unusable.
    // Full width now that it owns its own row (was flex:1 competing with the
    // buttons). Strips are slightly taller too — easier to grab a keyframe.
    const timelineStack = document.createElement("div");
    timelineStack.style.cssText = "display:flex; flex-direction:column; width:100%; gap:2px; min-width:0;";

    const timelineWrap = document.createElement("div");
    timelineWrap.style.cssText = "position:relative; height:20px; background:#1a1d22; border:1px solid #222; border-radius:3px; cursor:pointer; user-select:none;";
    const timelineProgress = document.createElement("div");
    timelineProgress.style.cssText = "position:absolute; left:0; top:0; bottom:0; width:0; background:rgba(76,158,255,0.25);";
    timelineWrap.appendChild(timelineProgress);
    const timelinePlayhead = document.createElement("div");
    timelinePlayhead.style.cssText = "position:absolute; top:-2px; bottom:-2px; width:2px; background:#7ab8ff; pointer-events:none;";
    timelineWrap.appendChild(timelinePlayhead);

    const keyframeStrip = document.createElement("div");
    keyframeStrip.style.cssText = "position:relative; height:16px; background:#11141a; border:1px solid #1c2128; border-radius:3px; user-select:none; overflow:hidden;";

    // Range strip — viewport minimap. Drag the window to pan, drag a
    // handle to zoom. Playback wraps inside this range so the artist
    // can preview a sub-clip without affecting the actual output.
    const rangeStrip = document.createElement("div");
    rangeStrip.style.cssText = "position:relative; height:16px; background:#0d1015; border:1px solid #1c2128; border-radius:3px; cursor:pointer; user-select:none; overflow:hidden;";
    const rangeWindow = document.createElement("div");
    rangeWindow.style.cssText = "position:absolute; top:0; bottom:0; left:0%; width:100%; background:rgba(76,158,255,0.25); border:1px solid #7ab8ff; cursor:grab; box-sizing:border-box;";
    rangeStrip.appendChild(rangeWindow);
    const rangeLeftHandle = document.createElement("div");
    rangeLeftHandle.style.cssText = "position:absolute; left:-3px; top:0; bottom:0; width:6px; cursor:ew-resize; background:rgba(122,184,255,0.6);";
    rangeWindow.appendChild(rangeLeftHandle);
    const rangeRightHandle = document.createElement("div");
    rangeRightHandle.style.cssText = "position:absolute; right:-3px; top:0; bottom:0; width:6px; cursor:ew-resize; background:rgba(122,184,255,0.6);";
    rangeWindow.appendChild(rangeRightHandle);

    timelineStack.append(timelineWrap, keyframeStrip, rangeStrip);

    // Click empty area of the keyframe strip → clear multi-selection.
    keyframeStrip.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        if (state.kfSelection.size > 0) {
            state.kfSelection = new Set();
            render();
        }
    });

    const frameLabel = document.createElement("span");
    // margin-left:auto pins the counter to the right end of the controls row.
    frameLabel.style.cssText = "font-family:monospace; min-width:80px; text-align:right; color:#9ab; margin-left:auto;";
    frameLabel.textContent = "1 / 1";

    // Timeline first (full width), then the controls beneath it.
    controlsRow.append(prevBtn, playBtn, nextBtn, addKeyBtn, delKeyBtn,
                       autoKeyToggle, frameLabel);
    transport.append(timelineStack, controlsRow);

    // ── state ────────────────────────────────────────────────────────
    const W = (name) => node.widgets.find(w => w.name === name);
    const get = (name) => W(name)?.value;
    // Whether each frame's rect must stay inside the canvas. Defaults to
    // true when the widget is missing (older graphs) so behaviour matches.
    const constrained = () => {
        const w = W("constrain_to_canvas");
        return w ? !!w.value : true;
    };
    const setStateWidget = (s) => {
        const w = W("state");
        if (!w) return;
        w.value = JSON.stringify(s);
        try { w.callback?.call(w, w.value); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };
    const readStateWidget = () => {
        try { return JSON.parse(W("state")?.value || ""); }
        catch { return { keyframes: {} }; }
    };

    const state = {
        doc: readStateWidget() || { keyframes: {} },
        // Multi-keyframe selection (Set of frame indices). Drag any
        // selected marker to move them all by the same delta;
        // Shift+click to toggle individual markers; Delete removes
        // them as a group.
        kfSelection: new Set(),
        // image / frames
        imgW: 0, imgH: 0,
        bgImage: null,
        previewFrames: [],
        bgStride: 1,
        frameCount: 1,
        currentFrame: 0,
        // Viewport window into the timeline (zoom + loop range). The
        // FRAME strip + keyframe markers map [viewStart..viewEnd] across
        // the full strip width; the RANGE strip stays at full extent
        // and shows a draggable window matching this viewport.
        viewStart: 0, viewEnd: 0,
        // canvas transform (image-space → element-space)
        dispScale: 1, offX: 0, offY: 0,
        dispZoom: 1,               // display-only zoom (1 = fit-to-canvas)
        // drag mode (mirrors bat_crop.js)
        drag: null,
        handles: [],
        // playback
        playing: false,
        playInterval: 0,
        playFps: 24,
        autoKey: true,
    };
    if (!state.doc.keyframes) state.doc.keyframes = {};
    node._batAnimCropState = state;

    // ── helpers ──────────────────────────────────────────────────────
    const persist = () => setStateWidget(state.doc);

    // Rect at current frame (interpolated). Returns the default rect when
    // no keyframes exist — same defaults the backend falls back to.
    function rectAtCurrent() {
        return rectAtFrame(state.currentFrame);
    }
    function rectAtFrame(f) {
        const kfs = state.doc.keyframes;
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        if (!keys.length) {
            return { x: 0, y: 0, w: 512, h: 512, angle: 0 };
        }
        if (f <= keys[0]) return { ...kfs[String(keys[0])] };
        if (f >= keys[keys.length - 1]) return { ...kfs[String(keys[keys.length - 1])] };
        let prev = keys[0], nxt = keys[keys.length - 1];
        for (const k of keys) {
            if (k <= f) prev = k;
            if (k >= f && k !== prev) { nxt = k; break; }
        }
        if (prev === nxt) return { ...kfs[String(prev)] };
        const t = (f - prev) / (nxt - prev);
        const a = kfs[String(prev)], b = kfs[String(nxt)];
        return {
            x: a.x + (b.x - a.x) * t,
            y: a.y + (b.y - a.y) * t,
            w: a.w + (b.w - a.w) * t,
            h: a.h + (b.h - a.h) * t,
            angle: a.angle + (b.angle - a.angle) * t,
        };
    }

    // Get the keyframe object to WRITE rect edits into. With auto-key,
    // we snapshot the interpolated rect into a new keyframe at the
    // current frame if there isn't one yet; otherwise edit the nearest
    // existing keyframe ≤ current.
    function editableKeyframe() {
        const kfs = state.doc.keyframes;
        const keyStr = String(state.currentFrame);
        if (state.autoKey || !Object.keys(kfs).length) {
            if (!kfs[keyStr]) {
                kfs[keyStr] = rectAtCurrent();
            }
            return kfs[keyStr];
        }
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        let target = keys[0];
        for (const k of keys) if (k <= state.currentFrame) target = k;
        return kfs[String(target)];
    }

    function writeRect(r) {
        const kf = editableKeyframe();
        kf.x = r.x; kf.y = r.y; kf.w = r.w; kf.h = r.h; kf.angle = r.angle;
        persist();
        renderTimelineKeyframes();
        render();
    }

    // ── coordinate conversion ────────────────────────────────────────
    function recomputeDisplay() {
        if (!state.imgW || !state.imgH) {
            state.dispScale = 1; state.offX = 0; state.offY = 0; return;
        }
        const cw = canvas.clientWidth || 1;
        const ch = canvas.clientHeight || 1;
        // Fit-to-canvas × display-only zoom + pan. Centred at zoom 1 / pan 0;
        // zooming out reveals the out-of-frame area a crop can extend into,
        // pan / zoom-to-cursor shift the centre.
        const fit = Math.min(cw / state.imgW, ch / state.imgH);
        state.dispScale = fit * (state.dispZoom || 1);
        state.offX = (cw - state.imgW * state.dispScale) / 2 + (state.panX || 0);
        state.offY = (ch - state.imgH * state.dispScale) / 2 + (state.panY || 0);
    }
    const c2d = (x, y) => ({ x: state.offX + x * state.dispScale, y: state.offY + y * state.dispScale });
    const d2c = (x, y) => ({ x: (x - state.offX) / state.dispScale, y: (y - state.offY) / state.dispScale });
    function localMouse(e) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (canvas.clientWidth  / r.width),
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

        if (state.bgImage && state.imgW) {
            const o = c2d(0, 0);
            ctx.drawImage(state.bgImage, o.x, o.y,
                state.imgW * state.dispScale, state.imgH * state.dispScale);
        }

        state.handles = [];
        if (!state.imgW) { drawInfo(); return; }

        const r = rectAtCurrent();
        const cx = r.x, cy = r.y;
        const cw = r.w, ch = r.h;
        const ang = (r.angle || 0) * Math.PI / 180;
        const cos = Math.cos(ang), sin = Math.sin(ang);
        const rectCx = cx + cw / 2, rectCy = cy + ch / 2;
        const cd = c2d(rectCx, rectCy);
        const dispW = cw * state.dispScale, dispH = ch * state.dispScale;

        const cornerDisp = (lx, ly) => {
            const rx = lx * cw / 2, ry = ly * ch / 2;
            const dx = (rx * cos - ry * sin) * state.dispScale;
            const dy = (rx * sin + ry * cos) * state.dispScale;
            return { x: cd.x + dx, y: cd.y + dy };
        };
        const cTL = cornerDisp(-1, -1);
        const cTR = cornerDisp( 1, -1);
        const cBR = cornerDisp( 1,  1);
        const cBL = cornerDisp(-1,  1);

        // Dim outside the rect.
        const o = c2d(0, 0);
        const iW = state.imgW * state.dispScale;
        const iH = state.imgH * state.dispScale;
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        const dim = new Path2D();
        dim.rect(o.x, o.y, iW, iH);
        dim.moveTo(cTL.x, cTL.y);
        dim.lineTo(cTR.x, cTR.y);
        dim.lineTo(cBR.x, cBR.y);
        dim.lineTo(cBL.x, cBL.y);
        dim.closePath();
        ctx.fill(dim, "evenodd");
        ctx.restore();

        // Rect border. Tint differently when this is an interpolated
        // (non-keyframed) frame vs a hard keyframe.
        const isOnKey = !!state.doc.keyframes[String(state.currentFrame)];
        ctx.strokeStyle = isOnKey ? "#f4b860" : "#7ec0ee";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cTL.x, cTL.y);
        ctx.lineTo(cTR.x, cTR.y);
        ctx.lineTo(cBR.x, cBR.y);
        ctx.lineTo(cBL.x, cBL.y);
        ctx.closePath();
        ctx.stroke();

        // Handles (8 sizing + rotate). Same names as bat_crop.js so the
        // drag math below can be lifted verbatim.
        const HR = 5;
        const defs = [
            { name: "tl", lx: -1, ly: -1 },
            { name: "tr", lx:  1, ly: -1 },
            { name: "br", lx:  1, ly:  1 },
            { name: "bl", lx: -1, ly:  1 },
            { name: "t",  lx:  0, ly: -1 },
            { name: "r",  lx:  1, ly:  0 },
            { name: "b",  lx:  0, ly:  1 },
            { name: "l",  lx: -1, ly:  0 },
        ];
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        for (const d of defs) {
            const p = cornerDisp(d.lx, d.ly);
            ctx.beginPath();
            ctx.rect(p.x - HR, p.y - HR, HR * 2, HR * 2);
            ctx.fill(); ctx.stroke();
            state.handles.push({ name: d.name, x: p.x, y: p.y, lx: d.lx, ly: d.ly });
        }

        // Rotation handle
        const topMid = cornerDisp(0, -1);
        const rotX = topMid.x + 30 * sin;
        const rotY = topMid.y + 30 * (-cos);
        ctx.strokeStyle = "#f4b860";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(topMid.x, topMid.y);
        ctx.lineTo(rotX, rotY);
        ctx.stroke();
        ctx.fillStyle = "#f4b860";
        ctx.strokeStyle = "#000";
        ctx.beginPath();
        ctx.arc(rotX, rotY, 6, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        state.handles.push({ name: "rot", x: rotX, y: rotY });

        drawInfo();
    }

    function drawInfo() {
        if (!state.imgW) { info.textContent = ""; return; }
        const r = rectAtCurrent();
        const snap = Math.max(1, (get("snap_to") | 0) || 8);
        const outW = Math.max(snap, Math.round(r.w / snap) * snap);
        const outH = Math.max(snap, Math.round(r.h / snap) * snap);
        const angTxt = Math.abs(r.angle) >= 0.05 ? `  · ${r.angle.toFixed(1)}°` : "";
        const onKey = !!state.doc.keyframes[String(state.currentFrame)];
        const freeTxt = constrained() ? "" : " · ⛶ free";
        info.textContent = `${Math.round(r.w)}×${Math.round(r.h)}${angTxt}  →  ${outW}×${outH}${onKey ? " · KEY" : ""}${freeTxt}`;
    }

    function handleAt(p) {
        const R = 10;
        for (const h of state.handles) {
            if (Math.abs(h.x - p.x) <= R && Math.abs(h.y - p.y) <= R) return h.name;
        }
        return null;
    }
    function insideBox(p) {
        if (!state.imgW) return false;
        const r = rectAtCurrent();
        const ang = (r.angle || 0) * Math.PI / 180;
        const cosA = Math.cos(ang), sinA = Math.sin(ang);
        const ip = d2c(p.x, p.y);
        const dx = ip.x - (r.x + r.w / 2), dy = ip.y - (r.y + r.h / 2);
        const u =  cosA * dx + sinA * dy;
        const v = -sinA * dx + cosA * dy;
        return Math.abs(u) <= r.w / 2 && Math.abs(v) <= r.h / 2;
    }
    function insideImage(p) {
        if (!state.imgW) return false;
        const o = c2d(0, 0);
        return p.x >= o.x && p.x <= o.x + state.imgW * state.dispScale
            && p.y >= o.y && p.y <= o.y + state.imgH * state.dispScale;
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!state.imgW) return;
        root.focus();
        const p = localMouse(e);
        const cP = d2c(p.x, p.y);
        const handleName = handleAt(p);
        let mode = null;
        if (handleName === "rot") mode = "rot";
        else if (handleName) mode = "resize:" + handleName;
        else if (insideBox(p)) mode = "move";
        else if (insideImage(p)) mode = "new";
        if (!mode) return;
        e.preventDefault();
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}

        const r = rectAtCurrent();
        const ratio = parseRatio(get("aspect_ratio"));
        const lock = !!get("aspect_lock") && ratio !== null;
        const angRad = (r.angle || 0) * Math.PI / 180;
        const cos = Math.cos(angRad), sin = Math.sin(angRad);
        const base = {
            startC: cP,
            x: r.x, y: r.y, w: r.w, h: r.h,
            cx: r.x + r.w / 2, cy: r.y + r.h / 2,
            angDeg: r.angle, cos, sin, lock, ratio,
        };
        if (mode.startsWith("resize:")) {
            const hSpec = state.handles.find(h => h.name === mode.slice(7));
            const lx = hSpec ? hSpec.lx : 0;
            const ly = hSpec ? hSpec.ly : 0;
            const ax_loc = -lx * r.w / 2;
            const ay_loc = -ly * r.h / 2;
            base.handleLx = lx; base.handleLy = ly;
            base.anchorImg = {
                x: base.cx + cos * ax_loc - sin * ay_loc,
                y: base.cy + sin * ax_loc + cos * ay_loc,
            };
        } else if (mode === "rot") {
            base.startMouseAng = Math.atan2(cP.y - base.cy, cP.x - base.cx);
            base.startAngDeg = r.angle;
        } else if (mode === "new") {
            // Brand-new rect — flatten rotation so the drag is straight.
            writeRect({ x: Math.round(cP.x), y: Math.round(cP.y), w: 1, h: 1, angle: 0 });
            base.angDeg = 0; base.cos = 1; base.sin = 0;
            base.x = Math.round(cP.x); base.y = Math.round(cP.y); base.w = 1; base.h = 1;
        }
        state.drag = { mode, base };
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!state.drag) return;
        const p = localMouse(e);
        const cP = d2c(p.x, p.y);
        const { mode, base } = state.drag;
        const Wimg = state.imgW, Himg = state.imgH;
        const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const rotated = Math.abs(base.angDeg) > 1e-3;
        // "constrain to canvas" off → axis-aligned rect may run off the edge
        // (backend zero-pads the overhang).
        const con = constrained();

        if (mode === "rot") {
            const curAng = Math.atan2(cP.y - base.cy, cP.x - base.cx);
            let deg = base.startAngDeg + (curAng - base.startMouseAng) * 180 / Math.PI;
            deg = ((deg + 180) % 360 + 360) % 360 - 180;
            writeRect({ x: base.x, y: base.y, w: base.w, h: base.h, angle: +deg.toFixed(1) });
            return;
        }

        let x = base.x, y = base.y, w = base.w, h = base.h;

        if (mode === "move") {
            const dx = cP.x - base.startC.x, dy = cP.y - base.startC.y;
            if (rotated || !con) {
                x = Math.round(base.x + dx);
                y = Math.round(base.y + dy);
            } else {
                x = cl(Math.round(base.x + dx), 0, Math.max(0, Wimg - base.w));
                y = cl(Math.round(base.y + dy), 0, Math.max(0, Himg - base.h));
            }
            w = base.w; h = base.h;
        } else if (mode === "new") {
            const anchX = base.startC.x, anchY = base.startC.y;
            const dirX = cP.x >= anchX ? 1 : -1;
            const dirY = cP.y >= anchY ? 1 : -1;
            let nw = Math.abs(cP.x - anchX), nh = Math.abs(cP.y - anchY);
            if (base.lock) {
                const wantH = nw / base.ratio;
                if (wantH > nh) nh = wantH; else nw = nh * base.ratio;
            }
            if (con && base.lock) {
                // Uniform shrink about the anchor so the locked ratio survives
                // the canvas clamp.
                let s = 1;
                if (dirX > 0) s = Math.min(s, (Wimg - anchX) / Math.max(1e-6, nw));
                else          s = Math.min(s, anchX / Math.max(1e-6, nw));
                if (dirY > 0) s = Math.min(s, (Himg - anchY) / Math.max(1e-6, nh));
                else          s = Math.min(s, anchY / Math.max(1e-6, nh));
                s = Math.max(0, Math.min(1, s));
                nw *= s; nh *= s;
                x = Math.round(dirX > 0 ? anchX : anchX - nw);
                y = Math.round(dirY > 0 ? anchY : anchY - nh);
            } else if (con) {
                const nx = Math.min(anchX, cP.x), ny = Math.min(anchY, cP.y);
                x = cl(Math.round(nx), 0, Math.max(0, Wimg - 1));
                y = cl(Math.round(ny), 0, Math.max(0, Himg - 1));
            } else {
                x = Math.round(dirX > 0 ? anchX : anchX - nw);
                y = Math.round(dirY > 0 ? anchY : anchY - nh);
            }
            w = Math.max(1, Math.round(nw));
            h = Math.max(1, Math.round(nh));
        } else if (rotated) {
            const { handleLx: lx, handleLy: ly, anchorImg, cos, sin } = base;
            const dx = cP.x - anchorImg.x, dy = cP.y - anchorImg.y;
            const u =  cos * dx + sin * dy;
            const v = -sin * dx + cos * dy;
            const isCorner = (lx !== 0 && ly !== 0);
            let new_w = base.w, new_h = base.h;
            if (isCorner) {
                new_w = Math.max(1, Math.abs(u));
                new_h = Math.max(1, Math.abs(v));
                if (base.lock) {
                    const wantH = new_w / base.ratio;
                    if (wantH > new_h) new_h = wantH;
                    else new_w = new_h * base.ratio;
                }
            } else if (lx === 0) {
                new_h = Math.max(1, Math.abs(v));
            } else {
                new_w = Math.max(1, Math.abs(u));
            }
            const offx = lx * new_w / 2;
            const offy = ly * new_h / 2;
            const cx = anchorImg.x + cos * offx - sin * offy;
            const cy = anchorImg.y + sin * offx + cos * offy;
            x = Math.round(cx - new_w / 2);
            y = Math.round(cy - new_h / 2);
            w = Math.max(1, Math.round(new_w));
            h = Math.max(1, Math.round(new_h));
        } else {
            const which = mode.slice(7);
            const r0 = base.x + base.w, b0 = base.y + base.h;
            let nx = base.x, ny = base.y, nr = r0, nb = b0;
            if (which.includes("l")) nx = cP.x;
            if (which.includes("r")) nr = cP.x;
            if (which.includes("t")) ny = cP.y;
            if (which.includes("b")) nb = cP.y;
            if (which === "t" || which === "b") { nx = base.x; nr = r0; }
            if (which === "l" || which === "r") { ny = base.y; nb = b0; }
            if (nr < nx) { const t = nr; nr = nx; nx = t; }
            if (nb < ny) { const t = nb; nb = ny; ny = t; }
            let nw = nr - nx, nh = nb - ny;
            const locked = base.lock && which.length === 2;
            if (locked) {
                const wantH = nw / base.ratio;
                if (wantH > nh) {
                    if (which.includes("t")) ny = nb - wantH; else nb = ny + wantH;
                    nh = wantH;
                } else {
                    const wantW = nh * base.ratio;
                    if (which.includes("l")) nx = nr - wantW; else nr = nx + wantW;
                    nw = wantW;
                }
            }
            if (con && locked) {
                // Aspect-locked AND constrained: shrink UNIFORMLY about the
                // fixed anchor corner to fit the canvas without deforming the
                // ratio (per-edge clamping would break it).
                const ax = which.includes("l") ? nr : nx;
                const ay = which.includes("t") ? nb : ny;
                const dx = which.includes("l") ? -nw : nw;
                const dy = which.includes("t") ? -nh : nh;
                let s = 1;
                if (dx > 0)      s = Math.min(s, (Wimg - ax) / dx);
                else if (dx < 0) s = Math.min(s, (0 - ax) / dx);
                if (dy > 0)      s = Math.min(s, (Himg - ay) / dy);
                else if (dy < 0) s = Math.min(s, (0 - ay) / dy);
                s = Math.max(0, Math.min(1, s));
                const fx = ax + dx * s, fy = ay + dy * s;
                x = Math.round(Math.min(ax, fx));
                y = Math.round(Math.min(ay, fy));
                w = Math.max(1, Math.round(Math.abs(fx - ax)));
                h = Math.max(1, Math.round(Math.abs(fy - ay)));
            } else if (con) {
                x = Math.round(Math.max(0, nx));
                y = Math.round(Math.max(0, ny));
                w = Math.max(1, Math.round(Math.min(nr, Wimg) - x));
                h = Math.max(1, Math.round(Math.min(nb, Himg) - y));
            } else {
                x = Math.round(nx);
                y = Math.round(ny);
                w = Math.max(1, Math.round(nr - nx));
                h = Math.max(1, Math.round(nb - ny));
            }
        }

        writeRect({ x, y, w, h, angle: base.angDeg });
    });

    const endDrag = (e) => {
        if (state.drag) {
            state.drag = null;
            try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        }
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // ── playback / frame stepping / keyframe management ──────────────
    // Viewport helpers (mirror bat_animated_grade / bat_roto).
    function viewSpan() { return Math.max(1, state.viewEnd - state.viewStart); }
    function frameToPct(f) { return ((f - state.viewStart) / viewSpan()) * 100; }
    function pxToFrame(dxPx, totalPx) {
        return clamp(
            state.viewStart + (dxPx / Math.max(1, totalPx)) * viewSpan(),
            0, Math.max(0, state.frameCount - 1),
        );
    }
    function syncRangeWindow() {
        if (state.frameCount <= 1) {
            rangeWindow.style.left = "0%";
            rangeWindow.style.width = "100%";
            return;
        }
        const maxF = state.frameCount - 1;
        rangeWindow.style.left  = `${(state.viewStart / maxF) * 100}%`;
        rangeWindow.style.width = `${Math.max(2, ((state.viewEnd - state.viewStart) / maxF) * 100)}%`;
    }
    function loopRange() {
        const start = Math.max(0, Math.min(state.viewStart, state.frameCount - 1));
        const end   = Math.max(start, Math.min(state.viewEnd, state.frameCount - 1));
        return [start, end];
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
            const [start, end] = loopRange();
            if (state.currentFrame < start || state.currentFrame > end) {
                setFrame(start);
            }
            state.playInterval = setInterval(() => {
                const [s, e] = loopRange();
                if (e <= s) return;
                const nxt = state.currentFrame + 1;
                setFrame(nxt > e ? s : nxt);
            }, 1000 / state.playFps);
        }
    }
    playBtn.onclick = togglePlay;
    prevBtn.onclick = () => setFrame(state.currentFrame - 1);
    nextBtn.onclick = () => setFrame(state.currentFrame + 1);
    addKeyBtn.onclick = () => {
        const cur = rectAtCurrent();
        state.doc.keyframes[String(state.currentFrame)] = cur;
        persist(); renderTimelineKeyframes(); render();
    };
    delKeyBtn.onclick = () => {
        delete state.doc.keyframes[String(state.currentFrame)];
        persist(); renderTimelineKeyframes(); render();
    };
    autoKeyInput.onchange = () => { state.autoKey = autoKeyInput.checked; };

    function setFrame(f) {
        if (state.frameCount <= 0) return;
        state.currentFrame = clamp(Math.round(f), 0, state.frameCount - 1);
        const previews = state.previewFrames || [];
        if (previews.length) {
            const stride = state.bgStride || 1;
            const idx = Math.min(previews.length - 1, Math.round(state.currentFrame / stride));
            const img = previews[idx];
            if (img instanceof Image) state.bgImage = img;
        }
        frameLabel.textContent = `${state.currentFrame + 1} / ${state.frameCount}`;
        // Position playhead within the current viewport so frame strip
        // tracks zoom.
        const pct = clamp(frameToPct(state.currentFrame), 0, 100);
        timelineProgress.style.width = `${pct}%`;
        timelinePlayhead.style.left  = `${pct}%`;
        renderTimelineKeyframes();
        render();
    }

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

    // Range strip pan/zoom (same pattern as bat_roto / bat_animated_grade).
    {
        let drag = null;
        const onDown = (ev, mode) => {
            ev.stopPropagation();
            ev.preventDefault();
            if (ev.button !== 0) return;
            const r = rangeStrip.getBoundingClientRect();
            drag = {
                mode, startX: ev.clientX,
                startVS: state.viewStart, startVE: state.viewEnd,
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
            const dxFrames = Math.round(((ev.clientX - drag.startX) / drag.totalPx) * maxF);
            if (drag.mode === "pan") {
                const span = drag.startVE - drag.startVS;
                const vs = clamp(drag.startVS + dxFrames, 0, maxF - span);
                state.viewStart = vs; state.viewEnd = vs + span;
            } else if (drag.mode === "left") {
                state.viewStart = clamp(drag.startVS + dxFrames, 0, drag.startVE - 1);
            } else if (drag.mode === "right") {
                state.viewEnd = clamp(drag.startVE + dxFrames, drag.startVS + 1, maxF);
            }
            syncRangeWindow();
            renderTimelineKeyframes();
            const pct = clamp(frameToPct(state.currentFrame), 0, 100);
            timelineProgress.style.width = `${pct}%`;
            timelinePlayhead.style.left  = `${pct}%`;
        });
        const onUp = (ev) => {
            if (!drag) return;
            try { rangeWindow.releasePointerCapture(ev.pointerId); } catch (_) {}
            drag = null;
        };
        rangeWindow.addEventListener("pointerup", onUp);
        rangeWindow.addEventListener("pointercancel", onUp);

        rangeStrip.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            if (ev.target !== rangeStrip) return;
            const r = rangeStrip.getBoundingClientRect();
            const maxF = Math.max(0, state.frameCount - 1);
            const target = ((ev.clientX - r.left) / Math.max(1, r.width)) * maxF;
            const span = state.viewEnd - state.viewStart;
            state.viewStart = clamp(Math.round(target - span / 2), 0, maxF - span);
            state.viewEnd   = state.viewStart + span;
            syncRangeWindow(); renderTimelineKeyframes();
        });
        rangeStrip.addEventListener("contextmenu", (ev) => {
            ev.preventDefault();
            const maxF = Math.max(0, state.frameCount - 1);
            state.viewStart = 0; state.viewEnd = maxF;
            syncRangeWindow(); renderTimelineKeyframes();
        });
    }

    // Keyframe markers live on `keyframeStrip` (the bottom strip).
    // Interactions:
    //   • click            — seek + select only this kf
    //   • shift+click      — toggle membership in multi-selection
    //   • drag             — move every selected kf by the same delta
    //   • right-click      — delete this kf
    function renderTimelineKeyframes() {
        keyframeStrip.querySelectorAll(".bat-kf").forEach(el => el.remove());
        rangeStrip.querySelectorAll(".bat-kf-mini").forEach(el => el.remove());
        syncRangeWindow();
        if (state.frameCount <= 1) return;
        const fullDenom = Math.max(1, state.frameCount - 1);

        // Drop stale kfSelection entries (e.g. after an inspector edit).
        for (const f of [...state.kfSelection]) {
            if (state.doc.keyframes[String(f)] === undefined) state.kfSelection.delete(f);
        }

        let drag = null;   // { startX, startFrame, basis: Map<frame,frame>, moved }

        for (const k of Object.keys(state.doc.keyframes)) {
            const f = Number(k);
            const inSel = state.kfSelection.has(f);

            // Mini marker on the range strip — always visible, gives the
            // artist context for what's outside the current viewport.
            const mini = document.createElement("div");
            mini.className = "bat-kf-mini";
            mini.style.cssText = `position:absolute; top:1px; bottom:1px; width:2px; background:${inSel ? "#7ab8ff" : "#f4b860"}; left:${(f / fullDenom) * 100}%; transform:translateX(-50%); pointer-events:none;`;
            rangeStrip.appendChild(mini);

            // Skip drawing the draggable kf on the main strip when it
            // sits outside the viewport.
            if (f < state.viewStart || f > state.viewEnd) continue;

            const dot = document.createElement("div");
            dot.className = "bat-kf";
            dot.dataset.frame = String(f);
            dot.title = `Keyframe @ frame ${f + 1} — drag to move, Shift+click to add to selection, right-click to delete`;
            dot.style.cssText = `
                position:absolute; top:0; bottom:0; width:8px;
                background:${inSel ? "#7ab8ff" : "#f4b860"};
                border:1px solid #000;
                left:${frameToPct(f)}%;
                transform:translateX(-50%); cursor:ew-resize; z-index:5;
            `;
            dot.addEventListener("contextmenu", (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                delete state.doc.keyframes[k];
                state.kfSelection.delete(f);
                persist(); renderTimelineKeyframes(); render();
            });
            dot.addEventListener("pointerdown", (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                if (ev.button !== 0) return;
                if (ev.shiftKey) {
                    if (state.kfSelection.has(f)) state.kfSelection.delete(f);
                    else                          state.kfSelection.add(f);
                    renderTimelineKeyframes();
                    return;
                }
                if (!state.kfSelection.has(f)) state.kfSelection = new Set([f]);
                const basis = new Map();
                for (const sf of state.kfSelection) basis.set(sf, sf);
                drag = {
                    startX: ev.clientX, startFrame: f, basis, moved: false,
                };
                dot.setPointerCapture(ev.pointerId);
                // NB: don't renderTimelineKeyframes() here — it
                // removes and re-creates the markers, detaching the
                // dot we just set pointer capture on, which would
                // kill the drag before the first pointermove fires.
            });
            dot.addEventListener("pointermove", (ev) => {
                if (!drag) return;
                if (Math.abs(ev.clientX - drag.startX) > 2) drag.moved = true;
                const r = keyframeStrip.getBoundingClientRect();
                const dxFrame = Math.round(((ev.clientX - drag.startX) / Math.max(1, r.width)) * viewSpan());
                keyframeStrip.querySelectorAll(".bat-kf").forEach((el) => {
                    const fromFrame = Number(el.dataset.frame ?? "-1");
                    if (drag.basis.has(fromFrame)) {
                        const nf = clamp(fromFrame + dxFrame, 0, state.frameCount - 1);
                        el.style.left = `${frameToPct(nf)}%`;
                        el.title = `Keyframe → frame ${nf + 1} (release to drop)`;
                    }
                });
            });
            dot.addEventListener("pointerup", (ev) => {
                if (!drag) return;
                try { dot.releasePointerCapture(ev.pointerId); } catch (_) {}
                if (!drag.moved) {
                    state.kfSelection = new Set([drag.startFrame]);
                    setFrame(drag.startFrame);
                    drag = null;
                    return;
                }
                const r = keyframeStrip.getBoundingClientRect();
                const dxFrame = Math.round(((ev.clientX - drag.startX) / Math.max(1, r.width)) * viewSpan());
                if (dxFrame === 0) { drag = null; renderTimelineKeyframes(); return; }
                // Build the move set, check for collisions.
                const moves = [];
                const newSet = new Set();
                let collision = false;
                for (const sf of state.kfSelection) {
                    const nf = clamp(sf + dxFrame, 0, state.frameCount - 1);
                    if (newSet.has(nf)) { collision = true; break; }
                    newSet.add(nf);
                    moves.push([sf, nf]);
                }
                if (!collision) {
                    for (const [, nf] of moves) {
                        if (state.kfSelection.has(nf)) continue;
                        if (state.doc.keyframes[String(nf)] !== undefined
                                && !state.kfSelection.has(nf)) { collision = true; break; }
                    }
                }
                if (collision) { drag = null; renderTimelineKeyframes(); return; }
                const stash = new Map();
                for (const [sf,] of moves) {
                    stash.set(sf, state.doc.keyframes[String(sf)]);
                    delete state.doc.keyframes[String(sf)];
                }
                state.kfSelection = new Set();
                for (const [sf, nf] of moves) {
                    state.doc.keyframes[String(nf)] = stash.get(sf);
                    state.kfSelection.add(nf);
                }
                drag = null;
                persist(); renderTimelineKeyframes(); render();
            });
            dot.addEventListener("pointercancel", () => {
                if (drag) { drag = null; renderTimelineKeyframes(); }
            });
            keyframeStrip.appendChild(dot);
        }
    }

    // ── keyboard ─────────────────────────────────────────────────────
    root.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        let handled = true;
        switch (e.key) {
            case " ": togglePlay(); break;
            case "ArrowLeft":  setFrame(state.currentFrame - 1); break;
            case "ArrowRight": setFrame(state.currentFrame + 1); break;
            case "k": case "K": addKeyBtn.click(); break;
            case "Delete": case "Backspace":
                if (e.target === root) {
                    // Multi-keyframe selection wins; otherwise fall
                    // back to "delete the keyframe at the current
                    // frame" (existing behaviour).
                    if (state.kfSelection.size > 0) {
                        for (const f of state.kfSelection) {
                            delete state.doc.keyframes[String(f)];
                        }
                        state.kfSelection = new Set();
                    } else {
                        delete state.doc.keyframes[String(state.currentFrame)];
                    }
                    persist(); renderTimelineKeyframes(); render();
                } else { handled = false; }
                break;
            default: handled = false;
        }
        if (handled) e.preventDefault();
    });

    // ── ingest from backend (input image batch) ──────────────────────
    node._batAnimCropIngest = async (frames, w, h, frameCount, stride) => {
        state.imgW = w; state.imgH = h;
        state.bgStride = Math.max(1, stride || 1);
        state.previewFrames = await Promise.all(frames.map(b64 => new Promise((res, rej) => {
            const im = new Image();
            im.onload = () => res(im);
            im.onerror = rej;
            im.src = `data:image/jpeg;base64,${b64}`;
        })));
        state.frameCount = Math.max(1, frameCount || state.previewFrames.length || 1);
        // Reset the viewport to cover the full clip on a fresh ingest,
        // unless the artist has already tightened it to a sub-range.
        if (state.viewEnd <= state.viewStart || state.viewEnd >= state.frameCount) {
            state.viewStart = 0;
            state.viewEnd = Math.max(0, state.frameCount - 1);
        }
        hint.style.display = state.previewFrames.length ? "none" : "block";
        // Persist image dimensions + frame count in state.doc so the
        // rect renders at the correct scale on reopen even before Run.
        state.doc.imgW = w; state.doc.imgH = h;
        state.doc.frameCount = state.frameCount;
        persist();
        // Cache the first frame as a bg preview for next reopen.
        if (frames.length > 0) {
            _saveCachedPreview(node, {
                firstFrame: frames[0],
                imgW: w, imgH: h, frameCount: state.frameCount,
            });
        }
        // If we have no keyframes yet AND no existing rect, seed with a
        // sensible default — centred half-size rect at frame 0 — so the
        // artist sees something to manipulate immediately.
        if (!Object.keys(state.doc.keyframes).length) {
            state.doc.keyframes["0"] = {
                x: Math.round(w * 0.25),
                y: Math.round(h * 0.25),
                w: Math.round(w * 0.5),
                h: Math.round(h * 0.5),
                angle: 0,
            };
            persist();
        }
        setFrame(Math.min(state.currentFrame, state.frameCount - 1));
    };

    // ── restore from cache on init ──────────────────────────────────
    // Two layers: dimensions from state.doc (ships with the workflow),
    // bg thumbnail from localStorage (per-machine).
    function _restoreCachedPreview() {
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW;
            state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
        }
        if (state.viewEnd <= state.viewStart) {
            state.viewStart = 0;
            state.viewEnd = Math.max(0, state.frameCount - 1);
        }
        const cached = _loadCachedPreview(node);
        if (cached?.firstFrame) {
            const im = new Image();
            im.onload = () => {
                state.previewFrames = [im];
                state.bgImage = im;
                state.bgStride = Math.max(1, state.frameCount);
                if (cached.imgW) state.imgW = cached.imgW;
                if (cached.imgH) state.imgH = cached.imgH;
                hint.style.display = "none";
                setFrame(state.currentFrame);
                render();
            };
            im.src = `data:image/jpeg;base64,${cached.firstFrame}`;
        }
        render();
    }

    // ── State JSON inspector ─────────────────────────────────────────
    // Collapsible panel under the canvas that exposes the entire node
    // state (keyframes, dimensions, frame count) as JSON. Useful for
    // confirming persistence and for copy/pasting a rect animation to
    // another Animated Crop node.
    const inspector = document.createElement("div");
    inspector.style.cssText = "border-top:1px solid #222; background:#0e1116; font:11px sans-serif;";
    const inspectorHeader = document.createElement("div");
    inspectorHeader.style.cssText = "display:flex; align-items:center; gap:6px; padding:4px 8px; cursor:pointer; user-select:none; color:#9ab;";
    const inspectorChevron = document.createElement("span");
    inspectorChevron.textContent = "▸";
    inspectorChevron.style.cssText = "font-size:9px; width:10px; display:inline-block;";
    const inspectorLabel = document.createElement("span");
    inspectorLabel.textContent = "State JSON (keyframes · dimensions)";
    inspectorLabel.style.flex = "1";
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy"; copyBtn.title = "Copy state JSON to the clipboard";
    copyBtn.style.cssText = "background:none; border:1px solid #2a2f37; color:#cdd; padding:2px 6px; border-radius:3px; cursor:pointer; font-size:10px;";
    const pasteBtn = document.createElement("button");
    pasteBtn.textContent = "Paste"; pasteBtn.title = "Replace state with the clipboard JSON";
    pasteBtn.style.cssText = copyBtn.style.cssText;
    inspectorHeader.append(inspectorChevron, inspectorLabel, copyBtn, pasteBtn);
    const inspectorBody = document.createElement("div");
    inspectorBody.style.cssText = "display:none; padding:6px 8px; gap:6px; flex-direction:column;";
    const inspectorArea = document.createElement("textarea");
    inspectorArea.spellcheck = false;
    inspectorArea.style.cssText = "width:100%; min-height:120px; max-height:280px; background:#0a0d12; color:#cde; border:1px solid #2a2f37; border-radius:3px; font:11px monospace; padding:4px 6px; resize:vertical; box-sizing:border-box; white-space:pre;";
    const inspectorMsg = document.createElement("div");
    inspectorMsg.style.cssText = "color:#9ab; font-size:10px;";
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply edits";
    applyBtn.style.cssText = "background:#1e2530; border:1px solid #2a3a55; color:#cde; padding:3px 8px; border-radius:3px; cursor:pointer; font-size:11px; align-self:flex-start;";
    inspectorBody.append(inspectorArea, inspectorMsg, applyBtn);
    inspector.append(inspectorHeader, inspectorBody);
    root.appendChild(inspector);

    function inspectorRefresh() {
        try { inspectorArea.value = JSON.stringify(state.doc, null, 2); } catch (_) {}
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
        try { await navigator.clipboard.writeText(text);
              copyBtn.textContent = "✓";
              setTimeout(() => copyBtn.textContent = "Copy", 900); }
        catch (_) { inspectorToggle(true); inspectorArea.value = text; inspectorArea.focus(); inspectorArea.select(); }
    };
    pasteBtn.onclick = async (ev) => {
        ev.stopPropagation();
        try { applyJsonText(await navigator.clipboard.readText()); }
        catch (_) {
            inspectorToggle(true);
            inspectorMsg.textContent = "Clipboard read blocked — paste into the textarea and click Apply.";
            inspectorMsg.style.color = "#f4b860";
        }
    };
    applyBtn.onclick = () => applyJsonText(inspectorArea.value);

    function applyJsonText(text) {
        let parsed;
        try { parsed = JSON.parse(text); }
        catch (e) {
            inspectorMsg.textContent = `Not valid JSON: ${e.message}`;
            inspectorMsg.style.color = "#ff8c8c";
            return;
        }
        if (typeof parsed !== "object" || parsed === null
                || typeof parsed.keyframes !== "object" || parsed.keyframes === null) {
            inspectorMsg.textContent = "Expected an object with a `keyframes` map.";
            inspectorMsg.style.color = "#ff8c8c";
            return;
        }
        state.doc = parsed;
        persist();
        inspectorRefresh();
        const n = Object.keys(state.doc.keyframes).length;
        inspectorMsg.textContent = `Applied ${n} keyframe${n === 1 ? "" : "s"}.`;
        inspectorMsg.style.color = "#7ed957";
        setFrame(state.currentFrame);
        render();
    }

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
    // pull back to see the out-of-frame area for an off-canvas crop.
    attachZoomControl({ wrap: canvasWrap, canvas, state, onChange: render, corner: "bl" });

    // Re-clamp every axis-aligned keyframe rect back inside the canvas when
    // the artist turns "constrain to canvas" on. Called from the widget's
    // callback (wired in onNodeCreated).
    node._batAnimCropReclamp = () => {
        if (!constrained() || !state.imgW || !state.imgH) return;
        let changed = false;
        for (const k of Object.keys(state.doc.keyframes)) {
            const r = state.doc.keyframes[k];
            if (Math.abs(r.angle || 0) > 1e-3) continue;   // rotated: leave alone
            const w = Math.min(r.w, state.imgW);
            const h = Math.min(r.h, state.imgH);
            const x = Math.min(Math.max(0, r.x), state.imgW - w);
            const y = Math.min(Math.max(0, r.y), state.imgH - h);
            if (x !== r.x || y !== r.y || w !== r.w || h !== r.h) {
                r.x = x; r.y = y; r.w = w; r.h = h; changed = true;
            }
        }
        if (changed) { persist(); }
        render();
    };

    _restoreCachedPreview();

    // Same fix as Bat_Roto: re-read the state widget after ComfyUI
    // restores widgets_values from the workflow JSON, so the canvas
    // reflects what's persisted. Without this hook, an F5 / workflow
    // reload would leave the canvas at its default keyframes even
    // though the persisted keyframes are sitting in the widget.
    node._batAnimCropReloadFromWidget = () => {
        try {
            const raw = W("state")?.value || "";
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed.keyframes !== "object") return;
            state.doc = parsed;
        } catch (_) { return; }
        state.kfSelection = new Set();
        state.drag = null;
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW;
            state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
        }
        setFrame(0);
        render();
        renderTimelineKeyframes();
        _restoreCachedPreview();
    };

    return root;
}

app.registerExtension({
    name: "Bat_AnimatedCrop",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Hide the raw `state` widget from the node body — it carries
            // the JSON keyframe map but we never want the artist editing
            // it as a string. `type:"hidden"` alone leaks the value as
            // raw text on some ComfyUI builds, so we also collapse the
            // layout slot AND no-op the draw.
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
            addBatDOMWidget(this, "bat_animcrop_editor", "bat_animcrop_editor", el, {
                minWidth: 560, height: 540, growable: true,
            });
            clampNodeSize(this, 560, 540);

            // When the artist toggles "constrain to canvas" back on, snap
            // any off-canvas keyframes back inside.
            const conW = this.widgets?.find(w => w.name === "constrain_to_canvas");
            if (conW) {
                const prev = conW.callback;
                const node = this;
                conW.callback = function () {
                    const rv = prev ? prev.apply(this, arguments) : undefined;
                    node._batAnimCropReclamp?.();
                    return rv;
                };
            }
            return r;
        };

        // onConfigure fires AFTER ComfyUI restores widgets_values
        // from the workflow JSON — re-read the state widget so the
        // canvas reflects the persisted keyframes. Without this hook
        // an F5 / workflow reload leaves the canvas at its default
        // even though the data is intact on disk.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const node = this;
            setTimeout(() => { node._batAnimCropReloadFromWidget?.(); }, 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batAnimCropIngest) return r;
            const frames = message.frames || [];
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const w  = one(message.w) || 0;
            const h  = one(message.h) || 0;
            const fc = one(message.frame_count) || frames.length || 1;
            const stride = one(message.stride) || 1;
            if (frames.length && w && h) this._batAnimCropIngest(frames, w, h, fc, stride);
            return r;
        };
    },
});
