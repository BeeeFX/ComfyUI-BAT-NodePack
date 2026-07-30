/**
 * BAT — Bat_RefAligner front-end.
 *
 * Photoshop-style free-transform of the reference over a plate-sized canvas.
 * The transform widgets (translate_x/y, scale, rotation) are the source of
 * truth and serialise; the canvas reads + writes them. Plate and reference
 * pixels arrive as base64 previews in the node's onExecuted message (the
 * backend pushes them via the {"ui": {...}} return), so the editor populates
 * after a run — same pattern as the points editor.
 *
 * Transform values are expressed in CANVAS space (= plate resolution); the
 * canvas element shows a letterboxed, fit-to-widget view of that space.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { batTrack } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_RefAligner";
const HANDLE_R = 7;
const HIT_R = 12;

function makeEditor(node) {
    const root = document.createElement("div");
    root.style.cssText = "position:relative;width:100%;height:100%;background:#0a0a0a;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:100%;display:block;touch-action:none;";
    const hint = document.createElement("div");
    hint.style.cssText = "position:absolute;left:8px;bottom:6px;font:11px monospace;color:#9aa;pointer-events:none;text-shadow:0 1px 2px #000;";
    hint.textContent = "Run once to load plate + reference";

    // Display-only opacity for the reference overlay (does NOT affect output).
    const bar = document.createElement("div");
    bar.style.cssText = "position:absolute;top:6px;right:8px;display:flex;align-items:center;gap:6px;font:11px monospace;color:#cdd;background:rgba(0,0,0,0.55);padding:3px 8px;border-radius:4px;";
    const opSlider = document.createElement("input");
    opSlider.type = "range"; opSlider.min = "5"; opSlider.max = "100"; opSlider.step = "1";
    opSlider.value = localStorage.getItem("bat-ref-opacity") || "90";
    opSlider.style.width = "90px";
    bar.append(document.createTextNode("ref opacity"), opSlider);

    root.append(canvas, hint, bar);

    const ctx = canvas.getContext("2d");
    const state = {
        plateImg: null, refImg: null,
        plateW: 0, plateH: 0, refW: 0, refH: 0,
        // display transform (canvas-space → element px)
        dispScale: 1, offX: 0, offY: 0,
        handles: [],          // {name,x,y} in element px, recomputed each render
        drag: null,
        refOpacity: Math.max(0.05, Math.min(1, (parseInt(opSlider.value, 10) || 90) / 100)),
    };
    node._batRef = { root, canvas, ctx, state, render };

    opSlider.addEventListener("input", () => {
        state.refOpacity = Math.max(0.05, Math.min(1, parseInt(opSlider.value, 10) / 100));
        localStorage.setItem("bat-ref-opacity", opSlider.value);
        render();
    });

    // ── widget access ──
    const W = (n) => node.widgets.find(w => w.name === n);
    const get = (n) => (W(n)?.value) ?? 0;
    // Write a widget value. The callback is fired in a try/catch so a throw in
    // one widget's callback can't abort a multi-widget update (this was the
    // cause of Y-axis drags silently failing while X worked).
    const set = (n, v) => {
        const w = W(n);
        if (!w) return;
        w.value = v;
        try { w.callback?.call(w, v); } catch (e) { /* ignore */ }
        node.setDirtyCanvas?.(true, true);
    };

    function recomputeDisplay(cw, ch) {
        if (!state.plateW || !state.plateH) { state.dispScale = 1; state.offX = 0; state.offY = 0; return; }
        state.dispScale = Math.min(cw / state.plateW, ch / state.plateH);
        state.offX = (cw - state.plateW * state.dispScale) / 2;
        state.offY = (ch - state.plateH * state.dispScale) / 2;
    }
    const c2d = (x, y) => ({ x: state.offX + x * state.dispScale, y: state.offY + y * state.dispScale });
    const d2c = (x, y) => ({ x: (x - state.offX) / state.dispScale, y: (y - state.offY) / state.dispScale });

    function refGeom() {
        // Centre + half-size in canvas space, plus rotation.
        const cx = state.plateW / 2 + get("translate_x");
        const cy = state.plateH / 2 + get("translate_y");
        const s = Math.max(0.0001, get("scale"));
        const hw = state.refW * s / 2;
        const hh = state.refH * s / 2;
        const rot = get("rotation") * Math.PI / 180;
        return { cx, cy, hw, hh, rot };
    }

    function render() {
        // Zoom-stable sizing: offsetWidth is the layout-box size and is NOT
        // affected by LiteGraph's CSS scale transform (getBoundingClientRect
        // is — using it here would shrink the bitmap on zoom-out, then
        // pixelate on zoom-in). devicePixelRatio mixes in so the bitmap is
        // crisp on hi-DPI and after zoom-in. Capped at 2× to keep canvas
        // memory bounded. See also bat_crop.js and bat_points_editor for
        // the same pattern.
        const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        const cssW = Math.max(1, Math.floor(root.offsetWidth));
        const cssH = Math.max(1, Math.floor(root.offsetHeight));
        const bw = Math.floor(cssW * dpr);
        const bh = Math.floor(cssH * dpr);
        if (canvas.width !== bw || canvas.height !== bh) {
            canvas.width = bw;
            canvas.height = bh;
        }
        // Draw in CSS pixels; the transform fills the bitmap at DPR scale.
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        recomputeDisplay(cssW, cssH);
        ctx.clearRect(0, 0, cssW, cssH);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cssW, cssH);

        // Plate (fit, letterboxed).
        if (state.plateImg) {
            const o = c2d(0, 0);
            ctx.drawImage(state.plateImg, o.x, o.y, state.plateW * state.dispScale, state.plateH * state.dispScale);
            // canvas border
            ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
            ctx.strokeRect(o.x + 0.5, o.y + 0.5, state.plateW * state.dispScale - 1, state.plateH * state.dispScale - 1);
        }

        state.handles = [];
        if (state.refImg && state.plateW) {
            const g = refGeom();
            const cd = c2d(g.cx, g.cy);
            const dw = state.refW * get("scale") * state.dispScale;
            const dh = state.refH * get("scale") * state.dispScale;
            ctx.save();
            ctx.translate(cd.x, cd.y);
            ctx.rotate(g.rot);
            ctx.globalAlpha = state.refOpacity;
            ctx.drawImage(state.refImg, -dw / 2, -dh / 2, dw, dh);
            ctx.globalAlpha = 1;
            // transform box
            ctx.strokeStyle = "#7ec0ee"; ctx.lineWidth = 1.5;
            ctx.strokeRect(-dw / 2, -dh / 2, dw, dh);
            ctx.restore();

            // Handle positions (rotate corner offsets into element space).
            const cos = Math.cos(g.rot), sin = Math.sin(g.rot);
            const corner = (sx, sy, name) => {
                const lx = sx * dw / 2, ly = sy * dh / 2;
                state.handles.push({ name, x: cd.x + lx * cos - ly * sin, y: cd.y + lx * sin + ly * cos });
            };
            corner(-1, -1, "tl"); corner(1, -1, "tr"); corner(1, 1, "br"); corner(-1, 1, "bl");
            // rotate handle above the top edge
            const ry = -dh / 2 - 26;
            state.handles.push({ name: "rot", x: cd.x + (0) * cos - ry * sin, y: cd.y + (0) * sin + ry * cos });

            // draw handles
            for (const hd of state.handles) {
                ctx.beginPath();
                ctx.arc(hd.x, hd.y, HANDLE_R, 0, Math.PI * 2);
                ctx.fillStyle = hd.name === "rot" ? "#f4b860" : "#fff";
                ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5;
                ctx.fill(); ctx.stroke();
            }
        }
    }

    function localMouse(e) {
        const r = canvas.getBoundingClientRect();
        // Map viewport → CSS-pixel canvas coords. clientWidth/clientHeight
        // are the layout box (zoom-independent, DPR-independent), while
        // r.width/height include LiteGraph's CSS zoom. Quotient is the
        // per-CSS-pixel scale. State + draws live in CSS pixels after
        // ctx.setTransform(dpr,...) in render().
        return {
            x: (e.clientX - r.left) * (canvas.clientWidth / r.width),
            y: (e.clientY - r.top)  * (canvas.clientHeight / r.height),
        };
    }

    function handleAt(p) {
        for (const hd of state.handles) {
            if (Math.hypot(hd.x - p.x, hd.y - p.y) <= HIT_R) return hd.name;
        }
        return null;
    }

    function insideRefBox(p) {
        // Transform p into ref-local (unrotated) space; inside if within half-size.
        const g = refGeom();
        const cd = c2d(g.cx, g.cy);
        const dx = p.x - cd.x, dy = p.y - cd.y;
        const cos = Math.cos(-g.rot), sin = Math.sin(-g.rot);
        const lx = dx * cos - dy * sin, ly = dx * sin + dy * cos;
        const dw = state.refW * get("scale") * state.dispScale;
        const dh = state.refH * get("scale") * state.dispScale;
        return Math.abs(lx) <= dw / 2 && Math.abs(ly) <= dh / 2;
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!state.refImg) return;
        const p = localMouse(e);
        const hName = handleAt(p);
        const cStart = d2c(p.x, p.y);
        const g = refGeom();
        let mode = null;
        if (hName === "rot") mode = "rotate";
        else if (hName) mode = "scale";
        else if (insideRefBox(p)) mode = "move";
        if (!mode) return;
        e.preventDefault();
        try { canvas.setPointerCapture(e.pointerId); } catch {}
        node._userTransformed = true;
        // baseline values for the drag
        const base = {
            tx: get("translate_x"), ty: get("translate_y"),
            scale: get("scale"), rot: get("rotation"),
            startC: cStart, cx: g.cx, cy: g.cy,
            // distance from centre to pointer at drag start (canvas space)
            startDist: Math.hypot(cStart.x - g.cx, cStart.y - g.cy),
            startAng: Math.atan2(cStart.y - g.cy, cStart.x - g.cx),
        };
        state.drag = { mode, base };
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!state.drag) return;
        const p = localMouse(e);
        const cNow = d2c(p.x, p.y);
        const { mode, base } = state.drag;
        if (mode === "move") {
            set("translate_x", Math.round(base.tx + (cNow.x - base.startC.x)));
            set("translate_y", Math.round(base.ty + (cNow.y - base.startC.y)));
        } else if (mode === "scale") {
            const dist = Math.hypot(cNow.x - base.cx, cNow.y - base.cy);
            const ratio = base.startDist > 1e-3 ? dist / base.startDist : 1;
            set("scale", Math.max(0.01, +(base.scale * ratio).toFixed(4)));
        } else if (mode === "rotate") {
            const ang = Math.atan2(cNow.y - base.cy, cNow.x - base.cx);
            let deg = base.rot + (ang - base.startAng) * 180 / Math.PI;
            deg = ((deg + 180) % 360 + 360) % 360 - 180;
            set("rotation", +deg.toFixed(1));
        }
        render();
    });
    const endDrag = (e) => { if (state.drag) { state.drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch {} } };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // Re-render on resize.
    // Tracked so it's disconnected when the node is deleted (see bat_lifecycle).
    const track = batTrack(node);
    track.observer(new ResizeObserver(() => render()), root);

    function ingest(b64, which, w, h) {
        const img = new Image();
        img.onload = () => {
            if (which === "plate") { state.plateImg = img; state.plateW = w || img.naturalWidth; state.plateH = h || img.naturalHeight; }
            else { state.refImg = img; state.refW = w || img.naturalWidth; state.refH = h || img.naturalHeight; }
            maybeAutoFit();
            hint.textContent = state.plateImg && state.refImg ? "Drag = move · corners = scale · gold = rotate" : hint.textContent;
            render();
        };
        img.src = `data:image/jpeg;base64,${b64}`;
    }

    function maybeAutoFit() {
        if (node._userTransformed) return;
        if (!state.plateW || !state.refW) return;
        const fit = Math.min(state.plateW / state.refW, state.plateH / state.refH) * 0.9;
        set("scale", +fit.toFixed(4));
        set("translate_x", 0);
        set("translate_y", 0);
        set("rotation", 0);
    }

    node._batRefIngest = ingest;

    // Live-update watcher: ComfyUI's number widgets don't always fire their
    // callback while dragging, so poll the transform values each frame and
    // re-render on any change (cheap — only redraws when something moved).
    let lastSig = "";
    // track.rafLoop stops rescheduling once the node is removed. The old loop
    // re-armed itself forever when detached, leaking a per-frame callback (and
    // this closure, including the decoded reference image) for the session.
    track.rafLoop(() => {
        if (!root.isConnected) return;   // detached but node alive: idle
        const sig = `${get("translate_x")}|${get("translate_y")}|${get("scale")}|${get("rotation")}`;
        if (sig !== lastSig) { lastSig = sig; render(); }
    });

    render();
    return root;
}

app.registerExtension({
    name: "BAT.RefAligner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = makeEditor(this);
            // Dual-mode sizing — same collapse as bat_crop under Nodes 2.0.
            addBatDOMWidget(this, "ref_editor", "ref_editor", el, {
                minWidth: 460, height: 560, growable: true,
            });
            // re-render when transform widgets change by hand
            const self = this;
            for (const name of ["translate_x", "translate_y", "scale", "rotation"]) {
                const w = this.widgets.find(x => x.name === name);
                if (w) {
                    const orig = w.callback;
                    w.callback = function (v) {
                        try { orig?.call(this, v); } catch (e) { /* ignore */ }
                        self._batRef?.render();
                    };
                }
            }
            clampNodeSize(this, 460, 560);
            return r;
        };

        // When a saved workflow restores non-default transform values, treat
        // them as user-authored so the first run's ingest doesn't auto-fit
        // over them. Without this, reopening a project resets the alignment
        // the moment plate+ref previews arrive from the backend.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const defaults = { translate_x: 0, translate_y: 0, scale: 1.0, rotation: 0.0 };
            for (const [name, def] of Object.entries(defaults)) {
                const w = this.widgets?.find(x => x.name === name);
                if (w && w.value !== def) { this._userTransformed = true; break; }
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batRefIngest) return r;
            const one = (v) => Array.isArray(v) ? v[0] : v;
            const pw = one(message.plate_w), ph = one(message.plate_h);
            const rw = one(message.ref_w), rh = one(message.ref_h);
            if (message.plate) this._batRefIngest(one(message.plate), "plate", pw, ph);
            if (message.reference) this._batRefIngest(one(message.reference), "reference", rw, rh);
            return r;
        };
    },
});
