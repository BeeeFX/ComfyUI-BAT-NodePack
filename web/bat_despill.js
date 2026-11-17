/**
 * 🦇 Despill — on-node key viewer.
 *
 * Every picture is rendered by the server (/bat/despill/view) with the same
 * Python that makes the node's outputs (bat_despill.render_view), for exactly
 * the region on screen. Zoomed in you see real pixels, and the viewer cannot
 * drift from the render — there is no JS copy of the despill or comp maths.
 *
 *   views        comp · alpha · foreground (straight) · premultiplied ·
 *                spill removed (x4) · plate
 *   despill      slider + green/blue, written straight into the node's widgets,
 *                so what you settle on is what the next queue run outputs
 *   background   checker / black / grey / white (the comp output's too)
 *   compare      hold "B" (or the button) for despill 0
 *   navigate     wheel = zoom at the cursor, drag = pan, double-click = fit,
 *                ← / → = frame while the pointer is over the viewer, ⛶ fullscreen
 *
 * While a render is in flight the last picture is re-projected to the new view,
 * so pan and zoom move immediately and sharpen when the render lands. Frames
 * arrive as raw RGB rather than PNG — PNG encoding was 84 of ~100 ms a frame.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addBatDOMWidget } from "./bat_node_layout.js";
import { addBatFullscreen } from "./bat_fullscreen.js";
import { batTrack, registerCleanup } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_Despill";
const VIEWER_HEIGHT = 460;
const VIEWS = [["comp", "Comp"], ["alpha", "Alpha"], ["foreground", "FG"],
               ["premultiplied", "Premult"], ["spill", "Spill"], ["plate", "Plate"]];
const BACKGROUNDS = ["checker", "black", "grey", "white"];
const MAX_ZOOM = 16;

// ── pure view maths (tests/verify_despill.py runs these under quickjs) ────

/** Zoom that fits a w x h image in a sw x sh stage, centred. */
function fitView(w, h, sw, sh) {
    const zoom = Math.min(sw / w, sh / h);
    return { zoom, cx: w / 2, cy: h / 2 };
}

/** Zoom by `factor` keeping the image point under (px, py) (stage px) fixed. */
function zoomAt(view, factor, px, py, sw, sh, minZoom) {
    const zoom = Math.max(minZoom, Math.min(MAX_ZOOM, view.zoom * factor));
    const ix = view.cx + (px - sw / 2) / view.zoom;          // image point under the cursor
    const iy = view.cy + (py - sh / 2) / view.zoom;
    return { zoom, cx: ix - (px - sw / 2) / zoom, cy: iy - (py - sh / 2) / zoom };
}

/** The visible image rectangle [x, y, w, h], integer, clipped to the image. */
function visibleRoi(view, w, h, sw, sh) {
    const x0 = Math.max(0, Math.floor(view.cx - sw / 2 / view.zoom));
    const y0 = Math.max(0, Math.floor(view.cy - sh / 2 / view.zoom));
    const x1 = Math.min(w, Math.ceil(view.cx + sw / 2 / view.zoom));
    const y1 = Math.min(h, Math.ceil(view.cy + sh / 2 / view.zoom));
    return [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)];
}

/** Where an image rectangle lands on the stage: [x, y, w, h] in stage px. */
function roiToScreen(roi, view, sw, sh) {
    return [sw / 2 + (roi[0] - view.cx) * view.zoom, sh / 2 + (roi[1] - view.cy) * view.zoom,
            roi[2] * view.zoom, roi[3] * view.zoom];
}

/** Raw RGB bytes from /bat/despill/view -> ImageBitmap (lossless; PNG cost 84 ms a frame). */
async function rgbToBitmap(buf, w, h) {
    const rgb = new Uint8Array(buf);
    if (!w || !h || rgb.length !== w * h * 3) throw new Error(`bad frame (${rgb.length} bytes for ${w}x${h})`);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
        rgba[j] = rgb[i]; rgba[j + 1] = rgb[i + 1]; rgba[j + 2] = rgb[i + 2]; rgba[j + 3] = 255;
    }
    return createImageBitmap(new ImageData(rgba, w, h));
}

function ensureStyles() {
    if (document.getElementById("bat-despill-styles")) return;
    const s = document.createElement("style");
    s.id = "bat-despill-styles";
    s.textContent = `
.bat-despill-root { display:flex; flex-direction:column; gap:4px; width:100%; height:100%; box-sizing:border-box;
  font:12px/1.3 sans-serif; color:#ddd; }
.bat-despill-row { display:flex; align-items:center; gap:4px; flex:0 0 auto; flex-wrap:wrap; }
.bat-despill-row button, .bat-despill-row select { background:#333; color:#ddd; border:1px solid #555; border-radius:4px;
  padding:2px 7px; cursor:pointer; font:inherit; }
.bat-despill-row button:hover { background:#444; }
.bat-despill-row button.bat-despill-on { background:#1f5f8b; border-color:#2e86c1; }
.bat-despill-row button:disabled { opacity:.4; cursor:default; }
.bat-despill-row input[type=range] { flex:1 1 80px; min-width:60px; }
.bat-despill-stage { position:relative; flex:1 1 auto; min-height:120px; background:#111; border-radius:4px; overflow:hidden; }
.bat-despill-stage canvas { position:absolute; inset:0; width:100%; height:100%; cursor:grab; touch-action:none; }
.bat-despill-status { margin-left:auto; color:#8a8a8a; white-space:nowrap; }
.bat-despill-status.bat-despill-warn { color:#e0a040; }
`;
    document.head.appendChild(s);
}

// ── the viewer ──────────────────────────────────────────────────────────────

class DespillViewer {
    constructor(node) {
        this.node = node;
        this.track = batTrack(node);
        this.track.dispose(() => this.dispose());
        this.run = null;                 // {token, w, h, count, hasPlate}
        this.view = null;                // {zoom, cx, cy}
        this.mode = "comp";
        this.frame = 0;
        this.raw = false;
        this.shown = null;               // {img, roi}
        this.busy = false; this.pending = false;
        this.build();
    }

    w(name) { return this.node.widgets?.find((x) => x.name === name); }

    build() {
        ensureStyles();
        const root = (this.root = document.createElement("div"));
        root.className = "bat-despill-root";
        const row = () => { const r = document.createElement("div"); r.className = "bat-despill-row"; root.appendChild(r); return r; };
        const btn = (parent, label, title, fn) => {
            const b = document.createElement("button");
            b.textContent = label; b.title = title;
            b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
            parent.appendChild(b);
            return b;
        };

        const r1 = row();
        this.viewBtns = {};
        for (const [key, label] of VIEWS) {
            this.viewBtns[key] = btn(r1, label, `Show ${label.toLowerCase()}`, () => { this.mode = key; this.syncChrome(); this.request(); });
        }
        this.status = document.createElement("span");
        this.status.className = "bat-despill-status";
        r1.appendChild(this.status);

        const r2 = row();
        const lab = document.createElement("span"); lab.textContent = "Despill"; r2.appendChild(lab);
        this.slider = document.createElement("input");
        Object.assign(this.slider, { type: "range", min: 0, max: 1, step: 0.01 });
        this.slider.addEventListener("input", () => this.setWidget("despill_strength", Number(this.slider.value)));
        r2.appendChild(this.slider);
        this.sliderVal = document.createElement("span"); r2.appendChild(this.sliderVal);
        this.screenBtn = btn(r2, "G", "Screen colour (green / blue)", () =>
            this.setWidget("screen", this.w("screen")?.value === "blue" ? "green" : "blue"));
        this.bgSel = document.createElement("select");
        for (const b of BACKGROUNDS) { const o = document.createElement("option"); o.value = o.textContent = b; this.bgSel.appendChild(o); }
        this.bgSel.title = "Comp background";
        this.bgSel.addEventListener("change", () => this.setWidget("background", this.bgSel.value));
        r2.appendChild(this.bgSel);
        this.rawBtn = btn(r2, "B: no despill", "Hold to compare with despill 0 (or hold B over the viewer)", () => {});
        const hold = (on) => { if (this.raw !== on) { this.raw = on; this.rawBtn.classList.toggle("bat-despill-on", on); this.request(); } };
        this.rawBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); hold(true); });
        for (const ev of ["pointerup", "pointerleave"]) this.rawBtn.addEventListener(ev, () => hold(false));
        btn(r2, "Fit", "Fit the frame (or double-click)", () => this.fit());
        btn(r2, "1:1", "One image pixel per screen pixel", () => this.oneToOne());
        this.zoomLab = document.createElement("span"); r2.appendChild(this.zoomLab);

        const stage = (this.stage = document.createElement("div"));
        stage.className = "bat-despill-stage";
        stage.dataset.batFsMount = "1";
        this.canvas = document.createElement("canvas");
        stage.appendChild(this.canvas);
        root.appendChild(stage);

        const r3 = row();
        this.scrub = document.createElement("input");
        Object.assign(this.scrub, { type: "range", min: 0, max: 0, step: 1, value: 0 });
        this.scrub.addEventListener("input", () => { this.frame = Number(this.scrub.value); this.syncChrome(); this.request(); });
        r3.appendChild(this.scrub);
        this.frameLab = document.createElement("span"); r3.appendChild(this.frameLab);

        this.wireCanvas();
        this.track.observer(new ResizeObserver(() => { if (this.view && this.shown) this.draw(); else this.fit(); }), stage);
        this.syncChrome();
    }

    setWidget(name, value) {
        const wd = this.w(name);
        if (!wd) return;
        wd.value = value;
        wd.callback?.(value);
        this.node.graph?.setDirtyCanvas?.(true, true);
        this.syncChrome();
        this.request();
    }

    stageSize() { return [this.canvas.clientWidth || 1, this.canvas.clientHeight || 1]; }

    fit() {
        if (!this.run) { this.draw(); return; }
        const [sw, sh] = this.stageSize();
        this.view = fitView(this.run.w, this.run.h, sw, sh);
        this.request();
    }

    oneToOne() {
        if (!this.view) return;
        const dpr = window.devicePixelRatio || 1;
        this.view = { ...this.view, zoom: 1 / dpr };
        this.request();
    }

    wireCanvas() {
        const c = this.canvas;
        const local = (e) => {
            const r = c.getBoundingClientRect();
            return [(e.clientX - r.left) * (c.clientWidth / (r.width || 1)), (e.clientY - r.top) * (c.clientHeight / (r.height || 1))];
        };
        c.addEventListener("wheel", (e) => {
            if (!this.view || !this.run) return;
            e.preventDefault(); e.stopPropagation();
            const [px, py] = local(e);
            const [sw, sh] = this.stageSize();
            const minZoom = fitView(this.run.w, this.run.h, sw, sh).zoom / 4;
            this.view = zoomAt(this.view, e.deltaY < 0 ? 1.25 : 0.8, px, py, sw, sh, minZoom);
            this.draw(); this.request();
        }, { passive: false });
        c.addEventListener("pointerdown", (e) => {
            if (!this.view) return;
            e.stopPropagation();
            c.setPointerCapture?.(e.pointerId);
            this.drag = { p: local(e), cx: this.view.cx, cy: this.view.cy };
            c.style.cursor = "grabbing";
        });
        c.addEventListener("pointermove", (e) => {
            if (!this.drag) return;
            const [x, y] = local(e);
            this.view = { ...this.view, cx: this.drag.cx - (x - this.drag.p[0]) / this.view.zoom,
                                        cy: this.drag.cy - (y - this.drag.p[1]) / this.view.zoom };
            this.draw();
        });
        const end = () => { if (this.drag) { this.drag = null; c.style.cursor = "grab"; this.request(); } };
        c.addEventListener("pointerup", end);
        c.addEventListener("pointercancel", end);
        c.addEventListener("dblclick", (e) => { e.stopPropagation(); this.fit(); });
        c.addEventListener("pointerenter", () => { this.hover = true; });
        c.addEventListener("pointerleave", () => { this.hover = false; });
        this.onKey = (e) => {
            if (!this.hover || !this.run) return;
            if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                e.preventDefault(); e.stopPropagation();
                this.frame = Math.max(0, Math.min(this.run.count - 1, this.frame + (e.key === "ArrowRight" ? 1 : -1)));
                this.syncChrome(); this.request();
            } else if ((e.key === "b" || e.key === "B") && !e.repeat && e.type === "keydown") {
                this.raw = true; this.rawBtn.classList.add("bat-despill-on"); this.request();
            }
        };
        this.onKeyUp = (e) => {
            if ((e.key === "b" || e.key === "B") && this.raw) { this.raw = false; this.rawBtn.classList.remove("bat-despill-on"); this.request(); }
        };
        this.track.listener(window, "keydown", this.onKey, true);
        this.track.listener(window, "keyup", this.onKeyUp, true);
    }

    dispose() { this.disposed = true; }

    syncChrome() {
        const s = Number(this.w("despill_strength")?.value ?? 0.5);
        this.slider.value = s;
        this.sliderVal.textContent = s.toFixed(2);
        const blue = this.w("screen")?.value === "blue";
        this.screenBtn.textContent = blue ? "B screen" : "G screen";
        this.bgSel.value = this.w("background")?.value || "checker";
        for (const [k, b] of Object.entries(this.viewBtns)) b.classList.toggle("bat-despill-on", k === this.mode);
        this.viewBtns.plate.disabled = !this.run?.hasPlate;
        const n = this.run?.count || 0;
        this.scrub.max = Math.max(0, n - 1);
        this.scrub.disabled = n <= 1;
        this.scrub.value = this.frame;
        this.frameLab.textContent = n ? `${this.frame} / ${n - 1}` : "";
        const dpr = window.devicePixelRatio || 1;
        this.zoomLab.textContent = this.view ? `${Math.round(this.view.zoom * dpr * 100)}%` : "";
    }

    setStatus(t, warn = false) { this.status.textContent = t || ""; this.status.classList.toggle("bat-despill-warn", warn); }

    async request() {
        this.syncChrome();
        if (!this.run || !this.view || this.disposed) { this.draw(); return; }
        if (this.busy) { this.pending = true; return; }
        this.busy = true;
        const [sw, sh] = this.stageSize();
        const dpr = window.devicePixelRatio || 1;
        const roi = visibleRoi(this.view, this.run.w, this.run.h, sw, sh);
        const outW = Math.max(1, Math.min(4096, Math.round(roi[2] * this.view.zoom * dpr)));
        const outH = Math.max(1, Math.min(4096, Math.round(roi[3] * this.view.zoom * dpr)));
        try {
            const res = await api.fetchApi("/bat/despill/view", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    token: this.run.token, frame: this.frame, view: this.mode, roi, out_w: outW, out_h: outH,
                    strength: Number(this.w("despill_strength")?.value ?? 0.5), raw: this.raw,
                    screen: this.w("screen")?.value || "green", background: this.w("background")?.value || "checker",
                }),
            });
            if (!res.ok) {
                let msg = `render failed (${res.status})`;
                try { msg = (await res.json()).error || msg; } catch (_) {}
                this.setStatus(msg, true);
                return;
            }
            const ms = res.headers.get("X-Render-Ms");
            const img = await rgbToBitmap(await res.arrayBuffer(),
                                          Number(res.headers.get("X-Width")), Number(res.headers.get("X-Height")));
            this.shown = { img, roi };
            this.setStatus(`${this.raw ? "no despill · " : ""}${ms ? ms + " ms" : ""}`);
            this.draw();
        } catch (e) {
            this.setStatus(`render failed: ${e?.message || e}`, true);
        } finally {
            this.busy = false;
            if (this.pending) { this.pending = false; this.request(); }
        }
    }

    draw() {
        const c = this.canvas;
        const dpr = window.devicePixelRatio || 1;
        const [sw, sh] = this.stageSize();
        if (c.width !== Math.round(sw * dpr) || c.height !== Math.round(sh * dpr)) {
            c.width = Math.round(sw * dpr); c.height = Math.round(sh * dpr);
        }
        const x = c.getContext("2d");
        x.setTransform(dpr, 0, 0, dpr, 0, 0);
        x.fillStyle = "#111"; x.fillRect(0, 0, sw, sh);
        this.syncChrome();
        if (!this.run) {
            x.fillStyle = "#8a8a8a"; x.font = "13px sans-serif"; x.textAlign = "center";
            x.fillText("Run the node to review the key here", sw / 2, sh / 2);
            x.textAlign = "start";
            return;
        }
        if (!this.shown || !this.view) return;
        // Re-project the last render into the current view: pan/zoom move now, sharpen on arrival.
        const [dx, dy, dw, dh] = roiToScreen(this.shown.roi, this.view, sw, sh);
        x.imageSmoothingEnabled = this.view.zoom * dpr < 1;
        x.drawImage(this.shown.img, dx, dy, dw, dh);
    }

    async adopt(run) {
        this.run = run;
        this.frame = Math.min(this.frame, Math.max(0, run.count - 1));
        this.node.properties = this.node.properties || {};
        this.node.properties.bat_despill_token = run.token;
        if (!this.view) this.fit(); else this.request();
    }
}

app.registerExtension({
    name: "BATNodePack.Despill",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const r = onNodeCreated?.apply(this, args);
            const node = this;
            const v = (node._batDespill = new DespillViewer(node));
            const widget = addBatDOMWidget(node, "despill_viewer", "bat-despill-viewer", v.root, {
                height: VIEWER_HEIGHT, growable: true, minWidth: 360,
                onResize: () => { if (v.view && v.shown) v.draw(); },
            });
            if (widget) addBatFullscreen(node, widget, v.root, { title: "🦇 Despill" });
            for (const name of ["despill_strength", "screen", "background", "colorspace"]) {
                const wd = node.widgets?.find((x) => x.name === name);
                if (!wd) continue;
                const orig = wd.callback;
                wd.callback = function (...a) {
                    const out = orig?.apply(this, a);
                    v.request();          // syncs the viewer's controls too; repeats coalesce
                    return out;
                };
            }
            if (!node.size || node.size[0] < 520) node.setSize?.([560, node.size?.[1] || 640]);
            // Reopen a workflow with its viewer: the token survives in properties while
            // the server process (and ComfyUI's cache of the inputs) does.
            setTimeout(async () => {
                const token = node.properties?.bat_despill_token;
                if (!token || v.run) return;
                try {
                    const info = await (await fetch(api.apiURL(`/bat/despill/info?token=${token}`))).json();
                    if (info?.ok) v.adopt({ token, w: info.w, h: info.h, count: info.count, hasPlate: info.has_plate });
                } catch (_) { /* server restarted: wait for a run */ }
            }, 200);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...rest) {
            const r = onExecuted?.apply(this, [message, ...rest]);
            const one = (x) => (Array.isArray(x) ? x[0] : x);
            const token = one(message?.despill_token);
            if (token && this._batDespill) {
                this._batDespill.adopt({ token, w: +one(message.w), h: +one(message.h),
                                   count: +one(message.frame_count), hasPlate: !!one(message.has_plate) });
            }
            return r;
        };

        registerCleanup(nodeType);
    },
});
