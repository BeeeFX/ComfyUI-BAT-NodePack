/**
 * Bat_AnimatedGrade — per-frame keyframed grade.
 *
 * Lifts the live grade-preview canvas from Bat_Grade and the timeline /
 * keyframe machinery from Bat_AnimatedCrop. Each grade slider records a
 * keyframe (when Auto-key is on) at the current frame; between
 * keyframes the parameters lerp independently. The canvas previews the
 * grade at the CURRENT frame's interpolated values, applied to the
 * current frame's thumbnail.
 *
 * Persisted state JSON (shipped in the workflow + restored on reload):
 *
 *   { keyframes: { "<frame>": { blackpoint, whitepoint, lift, gain,
 *                                multiply, offset, gamma,
 *                                clamp_white, clamp_black } } }
 *
 * The Python sibling (bat_animated_grade.py) reads the same JSON and
 * applies the per-frame grade frame-by-frame.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { batTrack, batNodeCacheKey } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_AnimatedGrade";

// Per-shape defaults that match the Python `_PARAMS` table — used both
// as the "no-keyframe" pass-through and as the starting point for a
// fresh keyframe.
const DEFAULT_GRADE = {
    blackpoint: 0.0,
    whitepoint: 1.0,
    lift:       0.0,
    gain:       1.0,
    multiply:   1.0,
    offset:     0.0,
    gamma:      1.0,
    clamp_white: false,
    clamp_black: false,
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// localStorage preview cache, same idea as bat_roto / bat_animated_crop:
// stash the first frame's thumb keyed by node id so reopen restores the
// canvas immediately without a Run.
function _previewCacheKey(node) {
    // Workflow-scoped: keying on node.id ALONE collided across graphs — opening
    // another workflow whose node 14 is also this node type restored the wrong
    // shot's plate (at the wrong imgW/imgH, so shapes drew against a bogus
    // reference). batNodeCacheKey folds in the workflow identity.
    return batNodeCacheKey(app, "bat_animgrade_preview", node);
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

function buildEditor(node) {
    const root = document.createElement("div");
    root.tabIndex = 0;
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px;
        outline:none; min-height:520px; font:11px sans-serif; color:#cde;
    `;

    // ── top: canvas + sidebar ────────────────────────────────────────
    const topRow = document.createElement("div");
    topRow.style.cssText = "display:flex; flex:1; min-height:0;";
    root.appendChild(topRow);

    // canvasWrap centers the canvas in the available space; the canvas
    // itself scales DOWN to fit but never stretches past the image's
    // native aspect ratio (set as an inline `aspect-ratio` once we know
    // the input dimensions — see _applyCanvasAspect below).
    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative; flex:1; min-width:0; background:#000; display:flex; align-items:center; justify-content:center; overflow:hidden;";
    topRow.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    // Aspect-preserving: max-width/max-height pin the canvas inside the
    // wrap; aspect-ratio (set when imgW/imgH known) keeps the rendering
    // box from stretching when the node body is widened.
    canvas.style.cssText = "display:block; max-width:100%; max-height:100%; width:auto; height:auto;";
    canvasWrap.appendChild(canvas);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    function _applyCanvasAspect() {
        if (state.imgW > 0 && state.imgH > 0) {
            canvas.style.aspectRatio = `${state.imgW} / ${state.imgH}`;
        }
    }

    const hint = document.createElement("div");
    hint.style.cssText = "position:absolute; left:8px; top:6px; font:11px monospace; color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;";
    hint.textContent = "Run once to load the frames.";
    canvasWrap.appendChild(hint);

    // Sidebar with one slider per grade parameter. Each slider records
    // a keyframe at the current frame (when Auto-key is on) on commit.
    const sidebar = document.createElement("div");
    sidebar.style.cssText = "width:200px; background:#15181d; border-left:1px solid #222; display:flex; flex-direction:column; padding:6px; gap:6px; overflow-y:auto;";
    topRow.appendChild(sidebar);

    // ── transport: full-width timeline on top, controls row beneath ───
    // The timeline strips used to sit INLINE among the buttons in a single
    // flex row, so the scrub bar got whatever width was left over — often
    // ~200px for a 240-frame range (under 1px per frame, nearly impossible to
    // scrub or hit a keyframe). Stacking it gives the timeline the editor's
    // full width and matches a classic player (timeline above, transport
    // below).
    const transport = document.createElement("div");
    transport.style.cssText = "display:flex; flex-direction:column; gap:4px; padding:4px 6px; background:#15181d; border-top:1px solid #222;";
    root.appendChild(transport);

    // Row that holds the buttons / auto-key / frame counter, under the timeline.
    const controlsRow = document.createElement("div");
    controlsRow.style.cssText = "display:flex; align-items:center; gap:6px;";

    const btn = (txt, title) => {
        const b = document.createElement("button");
        b.textContent = txt; b.title = title;
        // white-space:nowrap + inline-flex keep multi-glyph icons (|◀, ▶|)
        // on a single line at any narrow width.
        b.style.cssText = "background:none; border:1px solid #2a2f37; color:#cdd; padding:2px 6px; font-size:11px; border-radius:3px; cursor:pointer; min-width:22px; white-space:nowrap; line-height:1; display:inline-flex; align-items:center; justify-content:center;";
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

    // Three timeline strips (frame / keyframes / range minimap).
    // Full width now that it owns its own row (was flex:1 competing with the
    // buttons). Strips are slightly taller too — easier to grab a keyframe.
    const timelineStack = document.createElement("div");
    timelineStack.style.cssText = "display:flex; flex-direction:column; width:100%; gap:2px; min-width:0;";
    const frameStrip = document.createElement("div");
    frameStrip.style.cssText = "position:relative; height:20px; background:#1a1d22; border:1px solid #222; border-radius:3px; cursor:pointer; user-select:none; overflow:hidden;";
    const frameProgress = document.createElement("div");
    frameProgress.style.cssText = "position:absolute; left:0; top:0; bottom:0; width:0; background:rgba(76,158,255,0.25);";
    frameStrip.appendChild(frameProgress);
    const framePlayhead = document.createElement("div");
    framePlayhead.style.cssText = "position:absolute; top:-2px; bottom:-2px; width:2px; background:#7ab8ff; pointer-events:none;";
    frameStrip.appendChild(framePlayhead);

    const keyStrip = document.createElement("div");
    keyStrip.style.cssText = "position:relative; height:16px; background:#11141a; border:1px solid #1c2128; border-radius:3px; user-select:none; overflow:hidden;";

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
    timelineStack.append(frameStrip, keyStrip, rangeStrip);

    const frameLabel = document.createElement("span");
    // margin-left:auto pins the counter to the right end of the controls row.
    frameLabel.style.cssText = "font-family:monospace; min-width:80px; text-align:right; color:#9ab; margin-left:auto;";
    frameLabel.textContent = "1 / 1";

    // Timeline first (full width), then the controls beneath it.
    controlsRow.append(prevBtn, playBtn, nextBtn, addKeyBtn, delKeyBtn,
                       autoKeyToggle, frameLabel);
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
        catch { return { keyframes: {} }; }
    };

    const state = {
        doc: readStateWidget() || { keyframes: {} },
        // image / mask thumbs
        imgW: 0, imgH: 0,
        sourceData: null,        // ImageData of the current frame thumb
        maskData:   null,        // ImageData of the mask thumb (grayscale in R channel)
        previewFrames: [],       // Image objects
        bgStride: 1,
        frameCount: 1,
        currentFrame: 0,
        // viewport
        viewStart: 0,
        viewEnd: 0,
        // keyframe multi-selection
        kfSelection: new Set(),
        // playback
        playing: false,
        playInterval: 0,
        playFps: 24,
        autoKey: true,
    };
    if (!state.doc.keyframes) state.doc.keyframes = {};
    node._batAnimGradeState = state;

    const persist = () => setStateWidget(state.doc);

    // ── grade math (CSS / Canvas2D pixel loop, mirrors Bat_Grade) ────
    function resolveGradeAtFrame(f) {
        const kfs = state.doc.keyframes || {};
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        if (!keys.length) return { ...DEFAULT_GRADE };
        if (f <= keys[0]) return { ...DEFAULT_GRADE, ...kfs[String(keys[0])] };
        if (f >= keys[keys.length - 1]) return { ...DEFAULT_GRADE, ...kfs[String(keys[keys.length - 1])] };
        let prev = keys[0], nxt = keys[keys.length - 1];
        for (const k of keys) {
            if (k <= f) prev = k;
            if (k >= f && k !== prev) { nxt = k; break; }
        }
        if (prev === nxt) return { ...DEFAULT_GRADE, ...kfs[String(prev)] };
        const a = { ...DEFAULT_GRADE, ...kfs[String(prev)] };
        const b = { ...DEFAULT_GRADE, ...kfs[String(nxt)] };
        const t = (f - prev) / (nxt - prev);
        return {
            blackpoint: a.blackpoint + (b.blackpoint - a.blackpoint) * t,
            whitepoint: a.whitepoint + (b.whitepoint - a.whitepoint) * t,
            lift:       a.lift       + (b.lift       - a.lift) * t,
            gain:       a.gain       + (b.gain       - a.gain) * t,
            multiply:   a.multiply   + (b.multiply   - a.multiply) * t,
            offset:     a.offset     + (b.offset     - a.offset) * t,
            gamma:      a.gamma      + (b.gamma      - a.gamma) * t,
            clamp_white: a.clamp_white,
            clamp_black: a.clamp_black,
        };
    }

    function paintCanvas() {
        if (!state.sourceData) return;
        const p = resolveGradeAtFrame(state.currentFrame);
        const bp = +p.blackpoint, wp = +p.whitepoint;
        const lift = +p.lift, gain = +p.gain, mult = +p.multiply, off = +p.offset;
        const gamma = Math.max(+p.gamma, 0.01);
        const wpMbp = Math.max(wp - bp, 1e-6), invG = 1 / gamma;
        const cw = !!p.clamp_white, cb = !!p.clamp_black;
        const src = state.sourceData.data;
        const w = state.sourceData.width, h = state.sourceData.height;
        const out = ctx.createImageData(w, h);
        const dst = out.data;
        // Only enable the mask gate when the mask thumb is exactly the
        // same size as the source thumb — otherwise the per-pixel byte
        // offset misaligns the gate (paints the right hue in the wrong
        // place). Set in ingest + restore via mimg.naturalWidth/Height.
        const mask = (state.maskData
                      && state.maskData.width === w
                      && state.maskData.height === h) ? state.maskData.data : null;
        for (let i = 0; i < src.length; i += 4) {
            let r = src[i] / 255, g = src[i + 1] / 255, b = src[i + 2] / 255;
            r = (r - bp) / wpMbp; g = (g - bp) / wpMbp; b = (b - bp) / wpMbp;
            r = r * (gain - lift) + lift;
            g = g * (gain - lift) + lift;
            b = b * (gain - lift) + lift;
            r = r * mult + off; g = g * mult + off; b = b * mult + off;
            r = r < 0 ? 0 : Math.pow(r, invG);
            g = g < 0 ? 0 : Math.pow(g, invG);
            b = b < 0 ? 0 : Math.pow(b, invG);
            if (cw) { if (r > 1) r = 1; if (g > 1) g = 1; if (b > 1) b = 1; }
            if (cb) { if (r < 0) r = 0; if (g < 0) g = 0; if (b < 0) b = 0; }
            if (mask) {
                const m = mask[i] / 255, inv = 1 - m;
                r = r * m + (src[i]     / 255) * inv;
                g = g * m + (src[i + 1] / 255) * inv;
                b = b * m + (src[i + 2] / 255) * inv;
            }
            dst[i]     = r > 1 ? 255 : (r < 0 ? 0 : Math.round(r * 255));
            dst[i + 1] = g > 1 ? 255 : (g < 0 ? 0 : Math.round(g * 255));
            dst[i + 2] = b > 1 ? 255 : (b < 0 ? 0 : Math.round(b * 255));
            dst[i + 3] = 255;
        }
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w; canvas.height = h;
        }
        ctx.putImageData(out, 0, 0);
        hint.style.display = "none";
    }
    let rafQ = 0;
    function schedulePaint() {
        if (rafQ) return;
        rafQ = requestAnimationFrame(() => { rafQ = 0; paintCanvas(); });
    }
    node._batAnimGradePaint = schedulePaint;

    // ── viewport helpers (same as bat_roto) ──────────────────────────
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

    // ── keyframe helpers ─────────────────────────────────────────────
    // The "editable" keyframe for the current frame: create one with
    // the interpolated values if auto-key is on and one doesn't exist;
    // else return the nearest earlier keyframe (or null if there are
    // none — in which case a brand-new {} is created and seeded).
    function editableKeyframe() {
        const kfs = state.doc.keyframes;
        const key = String(state.currentFrame);
        if (state.autoKey || !Object.keys(kfs).length) {
            if (!kfs[key]) kfs[key] = resolveGradeAtFrame(state.currentFrame);
            return kfs[key];
        }
        const keys = Object.keys(kfs).map(Number).sort((a, b) => a - b);
        let target = keys[0];
        for (const k of keys) if (k <= state.currentFrame) target = k;
        return kfs[String(target)];
    }

    function paramCommit(name, value) {
        const kf = editableKeyframe();
        kf[name] = value;
        persist();
        renderKeyStrip();
        schedulePaint();
    }

    // ── sidebar sliders ──────────────────────────────────────────────
    const sliderDefs = [
        { name: "blackpoint", min: -1, max: 1, step: 0.001 },
        { name: "whitepoint", min: 0.001, max: 4, step: 0.001 },
        { name: "lift",       min: -1, max: 1, step: 0.001 },
        { name: "gain",       min: -4, max: 4, step: 0.001 },
        { name: "multiply",   min: 0,  max: 4, step: 0.001 },
        { name: "offset",     min: -1, max: 1, step: 0.001 },
        { name: "gamma",      min: 0.01, max: 4, step: 0.001 },
    ];
    const sliderRefs = {};
    function buildSidebar() {
        sidebar.innerHTML = "";
        const title = document.createElement("div");
        title.textContent = "Grade";
        title.style.cssText = "font-weight:bold; color:#cde; padding:2px 0;";
        sidebar.appendChild(title);

        for (const d of sliderDefs) {
            const wrap = document.createElement("div");
            wrap.style.cssText = "display:flex; flex-direction:column; gap:1px;";
            const head = document.createElement("div");
            head.style.cssText = "display:flex; justify-content:space-between; color:#9ab; font-size:10px;";
            const lab = document.createElement("span");
            lab.textContent = d.name;
            const val = document.createElement("span");
            val.style.cssText = "font-family:monospace; color:#cde;";
            head.append(lab, val);
            const sl = document.createElement("input");
            sl.type = "range"; sl.min = String(d.min); sl.max = String(d.max); sl.step = String(d.step);
            sl.style.width = "100%";
            // Stop pointer events bubbling to LiteGraph so the node body
            // doesn't try to drag itself when the artist click-and-holds
            // a slider. Without these, the slider thumb stops following
            // the mouse the moment the cursor leaves the thumb's exact
            // pixel area because LiteGraph captures the pointer for
            // its own drag handler.
            sl.addEventListener("pointerdown", (ev) => ev.stopPropagation());
            sl.addEventListener("pointermove", (ev) => ev.stopPropagation());
            sl.addEventListener("pointerup",   (ev) => ev.stopPropagation());
            sl.addEventListener("click",       (ev) => ev.stopPropagation());
            const refresh = () => {
                const p = resolveGradeAtFrame(state.currentFrame);
                sl.value = String(p[d.name]);
                val.textContent = Number(p[d.name]).toFixed(3);
            };
            refresh();
            sl.oninput = () => {
                val.textContent = Number(sl.value).toFixed(3);
                // Mutate the editable keyframe live so the canvas
                // preview updates immediately. History recorded once
                // per slider release (onchange).
                const kf = editableKeyframe();
                kf[d.name] = parseFloat(sl.value);
                persist();
                schedulePaint();
            };
            sl.onchange = () => renderKeyStrip();
            wrap.append(head, sl);
            sidebar.appendChild(wrap);
            sliderRefs[d.name] = { slider: sl, val, refresh };
        }
        for (const t of ["clamp_white", "clamp_black"]) {
            const row = document.createElement("label");
            row.style.cssText = "display:flex; align-items:center; gap:5px; color:#cde; cursor:pointer; font-size:10px;";
            const cb = document.createElement("input");
            cb.type = "checkbox";
            const refresh = () => {
                cb.checked = !!resolveGradeAtFrame(state.currentFrame)[t];
            };
            refresh();
            cb.onchange = () => {
                const kf = editableKeyframe();
                kf[t] = cb.checked;
                persist(); schedulePaint(); renderKeyStrip();
            };
            row.append(cb, document.createTextNode(t));
            sidebar.appendChild(row);
            sliderRefs[t] = { checkbox: cb, refresh };
        }
    }

    function refreshSliders() {
        for (const name of Object.keys(sliderRefs)) {
            sliderRefs[name].refresh?.();
        }
    }

    // ── keyframe strip ───────────────────────────────────────────────
    function renderKeyStrip() {
        keyStrip.querySelectorAll(".bat-kf").forEach(el => el.remove());
        rangeStrip.querySelectorAll(".bat-kf-mini").forEach(el => el.remove());
        syncRangeWindow();
        if (state.frameCount <= 1) return;
        const fullDenom = Math.max(1, state.frameCount - 1);

        for (const f of [...state.kfSelection]) {
            if (state.doc.keyframes[String(f)] === undefined) state.kfSelection.delete(f);
        }

        let drag = null;
        for (const k of Object.keys(state.doc.keyframes)) {
            const f = Number(k);
            const inSel = state.kfSelection.has(f);

            const mini = document.createElement("div");
            mini.className = "bat-kf-mini";
            mini.style.cssText = `position:absolute; top:1px; bottom:1px; width:2px; background:${inSel ? "#7ab8ff" : "#f4b860"}; left:${(f / fullDenom) * 100}%; transform:translateX(-50%); pointer-events:none;`;
            rangeStrip.appendChild(mini);

            if (f < state.viewStart || f > state.viewEnd) continue;

            const dot = document.createElement("div");
            dot.className = "bat-kf";
            dot.dataset.frame = String(f);
            dot.title = `Keyframe @ frame ${f + 1} — drag to move, Shift+click to multi-select, right-click to delete`;
            dot.style.cssText = `position:absolute; top:0; bottom:0; width:8px; background:${inSel ? "#7ab8ff" : "#f4b860"}; border:1px solid #000; left:${frameToPct(f)}%; transform:translateX(-50%); cursor:ew-resize; z-index:5;`;
            dot.addEventListener("contextmenu", (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                delete state.doc.keyframes[k];
                state.kfSelection.delete(f);
                persist(); renderKeyStrip(); refreshSliders(); schedulePaint();
            });
            dot.addEventListener("pointerdown", (ev) => {
                ev.stopPropagation();
                ev.preventDefault();
                if (ev.button !== 0) return;
                if (ev.shiftKey) {
                    if (state.kfSelection.has(f)) state.kfSelection.delete(f);
                    else state.kfSelection.add(f);
                    renderKeyStrip();
                    return;
                }
                if (!state.kfSelection.has(f)) state.kfSelection = new Set([f]);
                const basis = new Map();
                for (const sf of state.kfSelection) basis.set(sf, sf);
                drag = { startX: ev.clientX, startFrame: f, basis, moved: false };
                dot.setPointerCapture(ev.pointerId);
            });
            dot.addEventListener("pointermove", (ev) => {
                if (!drag) return;
                if (Math.abs(ev.clientX - drag.startX) > 2) drag.moved = true;
                const r = keyStrip.getBoundingClientRect();
                const dxFrame = Math.round(((ev.clientX - drag.startX) / Math.max(1, r.width)) * viewSpan());
                keyStrip.querySelectorAll(".bat-kf").forEach((el) => {
                    const fromFrame = Number(el.dataset.frame ?? "-1");
                    if (drag.basis.has(fromFrame)) {
                        const nf = clamp(fromFrame + dxFrame, 0, state.frameCount - 1);
                        el.style.left = `${frameToPct(nf)}%`;
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
                const r = keyStrip.getBoundingClientRect();
                const dxFrame = Math.round(((ev.clientX - drag.startX) / Math.max(1, r.width)) * viewSpan());
                if (dxFrame === 0) { drag = null; renderKeyStrip(); return; }
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
                if (collision) { drag = null; renderKeyStrip(); return; }
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
                persist(); renderKeyStrip();
            });
            keyStrip.appendChild(dot);
        }
    }

    keyStrip.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        if (state.kfSelection.size > 0) {
            state.kfSelection = new Set();
            renderKeyStrip();
        }
    });

    // ── frame strip seek + range strip pan/zoom ──────────────────────
    frameStrip.addEventListener("pointerdown", (e) => {
        frameStrip.setPointerCapture(e.pointerId);
        seekFromMouse(e);
    });
    frameStrip.addEventListener("pointermove", (e) => {
        if (e.buttons & 1) seekFromMouse(e);
    });
    function seekFromMouse(e) {
        const r = frameStrip.getBoundingClientRect();
        setFrame(Math.round(pxToFrame(e.clientX - r.left, r.width)));
    }

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
            renderKeyStrip();
            const pct = clamp(frameToPct(state.currentFrame), 0, 100);
            frameProgress.style.width = `${pct}%`;
            framePlayhead.style.left  = `${pct}%`;
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
            const target = Math.round(((ev.clientX - r.left) / Math.max(1, r.width)) * maxF);
            const span = state.viewEnd - state.viewStart;
            state.viewStart = clamp(Math.round(target - span / 2), 0, maxF - span);
            state.viewEnd   = state.viewStart + span;
            syncRangeWindow(); renderKeyStrip();
        });
        rangeStrip.addEventListener("contextmenu", (ev) => {
            ev.preventDefault();
            const maxF = Math.max(0, state.frameCount - 1);
            state.viewStart = 0; state.viewEnd = maxF;
            syncRangeWindow(); renderKeyStrip();
        });
    }

    // ── playback / frame stepping ────────────────────────────────────
    // Playback loops inside the range strip's viewport so the artist
    // can preview a sub-range. The output mask still spans the full
    // batch — this only affects the canvas player.
    function loopRange() {
        const start = Math.max(0, Math.min(state.viewStart, state.frameCount - 1));
        const end   = Math.max(start, Math.min(state.viewEnd, state.frameCount - 1));
        return [start, end];
    }
    function togglePlay() {
        if (state.playing) {
            clearInterval(state.playInterval);
            state.playing = false; playBtn.textContent = "▶";
        } else {
            state.playing = true; playBtn.textContent = "⏸";
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
        state.doc.keyframes[String(state.currentFrame)] =
            resolveGradeAtFrame(state.currentFrame);
        persist(); renderKeyStrip(); refreshSliders();
    };
    delKeyBtn.onclick = () => {
        delete state.doc.keyframes[String(state.currentFrame)];
        persist(); renderKeyStrip(); refreshSliders(); schedulePaint();
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
            if (img instanceof Image) {
                // Stamp into a backing canvas so the grade preview has
                // pixel data to work with.
                const back = document.createElement("canvas");
                back.width = img.naturalWidth;
                back.height = img.naturalHeight;
                const bctx = back.getContext("2d");
                bctx.drawImage(img, 0, 0);
                state.sourceData = bctx.getImageData(0, 0, back.width, back.height);
                state.imgW = img.naturalWidth;
                state.imgH = img.naturalHeight;
                _applyCanvasAspect();
            }
        }
        frameLabel.textContent = `${state.currentFrame + 1} / ${state.frameCount}`;
        const pct = clamp(frameToPct(state.currentFrame), 0, 100);
        frameProgress.style.width = `${pct}%`;
        framePlayhead.style.left  = `${pct}%`;
        renderKeyStrip();
        refreshSliders();
        schedulePaint();
    }

    root.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        let handled = true;
        switch (e.key) {
            case " ":           togglePlay(); break;
            case "ArrowLeft":   setFrame(state.currentFrame - 1); break;
            case "ArrowRight":  setFrame(state.currentFrame + 1); break;
            case "k": case "K": addKeyBtn.click(); break;
            case "Delete": case "Backspace":
                if (e.target === root) {
                    if (state.kfSelection.size > 0) {
                        for (const f of state.kfSelection) delete state.doc.keyframes[String(f)];
                        state.kfSelection = new Set();
                    } else {
                        delete state.doc.keyframes[String(state.currentFrame)];
                    }
                    persist(); renderKeyStrip(); refreshSliders(); schedulePaint();
                } else { handled = false; }
                break;
            default: handled = false;
        }
        if (handled) e.preventDefault();
    });

    // ── ingest from backend ──────────────────────────────────────────
    node._batAnimGradeIngest = async (frames, w, h, frameCount, stride, b64Mask) => {
        state.imgW = w; state.imgH = h;
        state.bgStride = Math.max(1, stride || 1);
        state.previewFrames = await Promise.all(frames.map(b64 => new Promise((res, rej) => {
            const im = new Image();
            im.onload = () => res(im);
            im.onerror = rej;
            im.src = `data:image/jpeg;base64,${b64}`;
        })));
        state.frameCount = Math.max(1, frameCount || state.previewFrames.length || 1);
        state.viewStart = 0;
        state.viewEnd = Math.max(0, state.frameCount - 1);
        state.doc.imgW = w; state.doc.imgH = h;
        state.doc.frameCount = state.frameCount;
        persist();
        if (frames.length > 0) {
            _saveCachedPreview(node, {
                firstFrame: frames[0], imgW: w, imgH: h,
                frameCount: state.frameCount, mask: b64Mask || null,
            });
        }
        if (b64Mask) {
            const mimg = new Image();
            await new Promise((res, rej) => {
                mimg.onload = res; mimg.onerror = rej;
                mimg.src = `data:image/png;base64,${b64Mask}`;
            });
            // Use the THUMB's natural dimensions, not the original W/H.
            // Both the frame and mask thumbs are downsampled server-side
            // (same _b64 helper, max_dim=720), so they share natural
            // dimensions when the source mask matches the source image.
            // Storing the mask at thumb resolution keeps it pixel-
            // aligned with `state.sourceData` — the paint loop indexes
            // both ImageData buffers by the same byte offset, so a
            // mismatch silently misaligns the gate.
            const mb = document.createElement("canvas");
            mb.width = mimg.naturalWidth;
            mb.height = mimg.naturalHeight;
            const mctx = mb.getContext("2d");
            mctx.drawImage(mimg, 0, 0);
            state.maskData = mctx.getImageData(0, 0, mb.width, mb.height);
        } else {
            state.maskData = null;
        }
        setFrame(Math.min(state.currentFrame, state.frameCount - 1));
    };

    function _restoreCachedPreview() {
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW; state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
            state.viewStart = 0;
            state.viewEnd = Math.max(0, state.frameCount - 1);
        }
        const cached = _loadCachedPreview(node);
        if (cached?.firstFrame) {
            const im = new Image();
            im.onload = () => {
                state.previewFrames = [im];
                state.bgStride = Math.max(1, state.frameCount);
                if (cached.imgW) state.imgW = cached.imgW;
                if (cached.imgH) state.imgH = cached.imgH;
                setFrame(state.currentFrame);
                if (cached.mask) {
                    const mim = new Image();
                    mim.onload = () => {
                        // Same thumb-aligned reasoning as the ingest path.
                        const mb = document.createElement("canvas");
                        mb.width = mim.naturalWidth;
                        mb.height = mim.naturalHeight;
                        const mctx = mb.getContext("2d");
                        mctx.drawImage(mim, 0, 0);
                        state.maskData = mctx.getImageData(0, 0, mb.width, mb.height);
                        schedulePaint();
                    };
                    mim.src = `data:image/png;base64,${cached.mask}`;
                }
            };
            im.src = `data:image/jpeg;base64,${cached.firstFrame}`;
        }
        schedulePaint();
        renderKeyStrip();
    }

    node._batAnimGradeReloadFromWidget = () => {
        try {
            const raw = W("state")?.value || "";
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (!parsed || typeof parsed.keyframes !== "object") return;
            state.doc = parsed;
            if (!state.doc.keyframes) state.doc.keyframes = {};
        } catch (_) { return; }
        state.kfSelection = new Set();
        if (state.doc.imgW && state.doc.imgH) {
            state.imgW = state.doc.imgW; state.imgH = state.doc.imgH;
        }
        if (state.doc.frameCount) {
            state.frameCount = Math.max(1, state.doc.frameCount | 0);
            state.viewStart = 0;
            state.viewEnd = Math.max(0, state.frameCount - 1);
        }
        setFrame(0);
        refreshSliders();
        renderKeyStrip();
        _restoreCachedPreview();
    };

    buildSidebar();
    // Tracked so the observer, the playback interval and the retained preview
    // frames are all released when the node is deleted (see bat_lifecycle).
    const track = batTrack(node);
    track.observer(new ResizeObserver(schedulePaint), canvasWrap);
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
    _restoreCachedPreview();

    return root;
}

app.registerExtension({
    name: "Bat_AnimatedGrade",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
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
            addBatDOMWidget(this, "bat_animgrade_editor", "bat_animgrade_editor", el, {
                minWidth: 640, height: 580, growable: true,
            });
            clampNodeSize(this, 640, 580);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const node = this;
            setTimeout(() => { node._batAnimGradeReloadFromWidget?.(); }, 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batAnimGradeIngest) return r;
            const frames = message.frames || [];
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const w  = one(message.w) || 0;
            const h  = one(message.h) || 0;
            const fc = one(message.frame_count) || frames.length || 1;
            const stride = one(message.stride) || 1;
            const mask = one(message.input_mask) || null;
            if (frames.length && w && h) this._batAnimGradeIngest(frames, w, h, fc, stride, mask);
            return r;
        };
    },
});
