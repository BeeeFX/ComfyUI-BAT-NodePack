/**
 * Bat SAM 3.1 Segmenter — front-end.
 *
 * A viewport built for SAM 3.1 rather than the shared points editor: every
 * click, box or text edit re-asks the server for the annotation frame's mask
 * (/bat/sam31/live, /bat/sam31/detect), which costs a decoder pass once the
 * frame is encoded, so the mask follows the artist's hand. A scrub bar walks
 * the clip; after a run (or ▶ Track) it shows the tracked mask on every frame.
 *
 * Built on addBatDOMWidget so it sizes correctly under both Nodes 1.0 and
 * Nodes 2.0 (bat_node_layout.js explains the contract). All editor state is
 * serialised into the hidden `prompt_state` widget as normalised 0-1
 * coordinates, so a proxy plate and the real frame agree.
 *
 * Where pixels come from, best first (the same ladder SeC uses, via the
 * shared bat_clip_source.js):
 *   1. a run's token     — exact frames + masks, served by /bat/sam31/frame
 *                          for as long as ComfyUI keeps the upstream batch;
 *   2. the upstream file — decoded server-side through /bat/frame-picker;
 *   3. the upstream node's own preview (LoadImage, a video element);
 *   4. a localStorage copy of the last annotation plate.
 * The live routes get the run's exact frame when the plate came from (1), and
 * an upload of the plate otherwise.
 *
 * ▶ Track queues a trimmed copy of the prompt — this node, everything feeding
 * it, and the internal Bat_Sam31TrackSink output node. It is an ordinary
 * queued job, so progress, interrupt and model management all behave, and
 * ComfyUI's cache hands the result to the next full queue run unchanged.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addBatDOMWidget, refreshBatLayout } from "./bat_node_layout.js";
import { addBatFullscreen } from "./bat_fullscreen.js";
import { chainCallback, resolveSourcePreview, captureVideoFrame } from "./bat_points_editor/utility.js";
import {
    batTrack, registerCleanup, isNodeAlive, batNodeCacheKey,
    batReplayLastExecution, batPreviewWillReplay,
} from "./bat_lifecycle.js";
import {
    findPathSource, widgetNum, sourceFrameIndex, sourceBatchLength, fileFrameCount, loadImage,
} from "./bat_clip_source.js";

const NODE_TYPE = "Bat_Sam31Segmenter";
const ADV_TYPE = "Bat_Sam31AdvancedParams";
const SINK_TYPE = "Bat_Sam31TrackSink";

// Mirrors bat_sam31_segmenter.DEFAULTS — used by the live routes when no
// advanced node is wired.
const ADV_DEFAULTS = { model_file: null, detection_threshold: 0.5, max_objects: 16, refine_iterations: 2 };

const EDITOR_HEIGHT = 440;
const PLATE_MAX = 1280;               // long edge of plates fetched for the viewport
const HIT_RADIUS = 9;                 // css px: clicking this close to a point removes it
const DRAG_THRESHOLD = 5;             // css px before a press becomes a box drag
const TEXT_DEBOUNCE_MS = 350;
const MASK_RGB = [64, 190, 255];      // click-mode overlay
const RUN_MASK_CACHE = 48;

// ── styles (prefixed: every pack shares one CSS namespace) ─────────────────

function ensureStyles() {
    if (document.getElementById("bat-sam31-styles")) return;
    const s = document.createElement("style");
    s.id = "bat-sam31-styles";
    s.textContent = `
.bat-sam31-root { display:flex; flex-direction:column; gap:4px; width:100%; height:100%;
  box-sizing:border-box; font:12px/1.3 sans-serif; color:#ddd; }
.bat-sam31-bar { display:flex; align-items:center; gap:4px; flex:0 0 auto; }
.bat-sam31-bar button { background:#333; color:#ddd; border:1px solid #555; border-radius:4px;
  padding:2px 8px; cursor:pointer; font:inherit; white-space:nowrap; }
.bat-sam31-bar button:hover { background:#444; }
.bat-sam31-bar button.bat-sam31-primary { background:#1f5f8b; border-color:#2e86c1; }
.bat-sam31-bar button.bat-sam31-off { opacity:.55; }
.bat-sam31-status { flex:1 1 auto; text-align:right; color:#9a9a9a; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
.bat-sam31-status.bat-sam31-warn { color:#e0a040; }
.bat-sam31-stage { position:relative; flex:1 1 auto; min-height:120px; background:#111;
  border-radius:4px; overflow:hidden; }
.bat-sam31-stage canvas { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair;
  touch-action:none; }
.bat-sam31-scrub { display:flex; align-items:center; gap:6px; flex:0 0 auto; }
.bat-sam31-scrub input[type=range] { flex:1 1 auto; }
.bat-sam31-hint { flex:0 0 auto; color:#8a8a8a; font-size:11px; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
`;
    document.head.appendChild(s);
}

// ── small helpers ───────────────────────────────────────────────────────────

const W = (node, name) => node.widgets?.find((w) => w.name === name);

function setWidgetValue(node, name, value, { fire = true } = {}) {
    const w = W(node, name);
    if (!w) return;
    w.value = value;
    if (fire) w.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
}

/** Hide a plain widget row in both renderers (the pack's batch_format recipe). */
function setRowHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget._batSamOrigType === undefined) {
            widget._batSamOrigType = widget.type;
            widget._batSamOrigComputeSize = widget.computeSize;
            widget._batSamOrigDraw = widget.draw;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.hidden = true;
    } else if (widget._batSamOrigType !== undefined) {
        widget.type = widget._batSamOrigType;
        widget.computeSize = widget._batSamOrigComputeSize;
        widget.draw = widget._batSamOrigDraw;
        widget.hidden = false;
    }
}

function emptyState() { return { points: [], box: null, instances: [] }; }

function readState(node) {
    try {
        const st = JSON.parse(W(node, "prompt_state")?.value || "{}");
        return {
            points: Array.isArray(st.points) ? st.points : [],
            box: Array.isArray(st.box) && st.box.length === 4 ? st.box : null,
            instances: Array.isArray(st.instances) ? st.instances : [],
        };
    } catch (_) {
        return emptyState();
    }
}

function writeState(node, st) {
    setWidgetValue(node, "prompt_state", JSON.stringify(st), { fire: false });
}

function boxIoU(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0;
}

/** Same rule as bat_sam31_runtime.match_instances: IoU >= .5, else kept. */
function keepFlagsFor(found, saved) {
    return found.map((inst) => {
        let best = null, bestIoU = 0;
        for (const s of saved || []) {
            const v = boxIoU(inst.box, s.box || [0, 0, 0, 0]);
            if (v > bestIoU) { best = s; bestIoU = v; }
        }
        return best && bestIoU >= 0.5 ? best.keep !== false : true;
    });
}

function instanceColour(i) {
    const h = (i * 137.508) % 360;
    // HSL -> RGB, s=.75 l=.55
    const s = 0.75, l = 0.55, c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - c / 2;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
        : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/** L-mode PNG -> {canvas tinted RGBA, lum bytes for hit tests, w, h, area}. */
function tintMask(img, rgb, alpha = 0.5) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(img, 0, 0);
    const d = x.getImageData(0, 0, w, h);
    const p = d.data;
    const lum = new Uint8Array(w * h);
    let on = 0;
    for (let i = 0, j = 0; i < p.length; i += 4, j++) {
        const a = p[i];
        lum[j] = a;
        if (a > 127) on++;
        p[i] = rgb[0]; p[i + 1] = rgb[1]; p[i + 2] = rgb[2]; p[i + 3] = a * alpha;
    }
    x.putImageData(d, 0, 0);
    return { canvas: c, lum, w, h, area: on / (w * h) };
}

async function apiJSON(path, body) {
    const res = await api.fetchApi(path, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    return { status: res.status, data };
}

/** The wired advanced node's live-relevant settings, else the defaults. */
function advancedSettings(node) {
    const out = { ...ADV_DEFAULTS };
    const input = node.inputs?.find((i) => i.name === "advanced");
    if (input?.link == null) return out;
    const link = node.graph?.getLink?.(input.link) ?? node.graph?.links?.get?.(input.link);
    const origin = link && node.graph.getNodeById(link.origin_id);
    if (!origin || origin.type !== ADV_TYPE) return out;
    for (const k of Object.keys(out)) {
        const v = W(origin, k)?.value;
        if (v !== undefined && v !== null) out[k] = v;
    }
    if (String(out.model_file || "").startsWith("(no SAM 3")) out.model_file = null;
    return out;
}

/**
 * The ▶ Track prompt: `ids` and everything upstream of them, plus one sink per
 * id. Links in an API prompt are [nodeId, outputSlot]; widget values never
 * look like that (a 2-array whose second item is a number is always a link —
 * ComfyUI's own executor uses the same test).
 */
function trimPrompt(output, ids) {
    const keep = {};
    const stack = [...ids];
    while (stack.length) {
        const id = stack.pop();
        if (keep[id] || !output[id]) continue;
        keep[id] = output[id];
        for (const v of Object.values(output[id].inputs || {})) {
            if (Array.isArray(v) && v.length === 2 && typeof v[1] === "number") stack.push(String(v[0]));
        }
    }
    ids.forEach((id, n) => { keep[`bat_sam31_track_${n}`] = { class_type: SINK_TYPE, inputs: { masks: [id, 0] } }; });
    return keep;
}

// ── per-node plate cache (localStorage, annotation frame only) ──────────────

function plateKey(node) { return batNodeCacheKey(app, "bat_sam31_plate", node); }
function savePlate(node, data) {
    try { localStorage.setItem(plateKey(node), JSON.stringify(data)); } catch (_) { /* quota */ }
}
function loadSavedPlate(node) {
    try { return JSON.parse(localStorage.getItem(plateKey(node)) || "null"); } catch (_) { return null; }
}

// ── the editor ──────────────────────────────────────────────────────────────

class Sam31Editor {
    constructor(node) {
        this.node = node;
        this.track = batTrack(node);
        this.track.dispose(() => clearTimeout(this._detTimer));
        this.view = 0;              // frame shown
        this.count = 0;             // batch length, 0 = unknown
        this.plate = null;          // {img, frame, kind, w, h, b64}
        this.live = null;           // {mask, sig} click-mode live mask on the annotation frame
        this.instances = [];        // text mode: [{box, score, keep, mask}]
        this.instSig = "";
        this.run = null;            // {token, w, h, count, annotation, liveFrames}
        this.runMasks = new Map();
        this.showMask = true;
        this.hover = -1;
        this.drag = null;
        this.plateGen = 0;
        this.liveBusy = false; this.livePending = false;
        this.detBusy = false; this.detPending = false;
        this.build();
    }

    // DOM ───────────────────────────────────────────────────────────────────
    build() {
        ensureStyles();
        const root = (this.root = document.createElement("div"));
        root.className = "bat-sam31-root";

        const bar = document.createElement("div");
        bar.className = "bat-sam31-bar";
        const btn = (label, title, fn, cls = "") => {
            const b = document.createElement("button");
            b.textContent = label; b.title = title; if (cls) b.className = cls;
            b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
            bar.appendChild(b);
            return b;
        };
        this.trackBtn = btn("▶ Track", "Track the whole clip now and fill the scrubber. Queueing "
            + "the workflow afterwards reuses this result.", () => this.queueTrack(), "bat-sam31-primary");
        this.clearBtn = btn("Clear", "Remove every point, the box and the text-mode choices.",
            () => this.clearPrompts());
        this.maskBtn = btn("◐ Mask", "Show / hide the mask overlay.", () => {
            this.showMask = !this.showMask;
            this.maskBtn.classList.toggle("bat-sam31-off", !this.showMask);
            this.draw();
        });
        this.annotateBtn = btn("◎ Annotate here", "Make the frame you are looking at the "
            + "annotation frame (frame_index_select).", () => this.annotateHere());
        this.status = document.createElement("div");
        this.status.className = "bat-sam31-status";
        bar.appendChild(this.status);
        root.appendChild(bar);

        const stage = (this.stage = document.createElement("div"));
        stage.className = "bat-sam31-stage";
        stage.dataset.batFsMount = "1";
        this.canvas = document.createElement("canvas");
        stage.appendChild(this.canvas);
        root.appendChild(stage);

        const scrub = document.createElement("div");
        scrub.className = "bat-sam31-scrub";
        this.slider = document.createElement("input");
        this.slider.type = "range"; this.slider.min = 0; this.slider.max = 0; this.slider.value = 0;
        this.slider.addEventListener("input", () => this.setView(Number(this.slider.value)));
        this.frameLabel = document.createElement("span");
        scrub.appendChild(this.slider);
        scrub.appendChild(this.frameLabel);
        root.appendChild(scrub);

        this.hint = document.createElement("div");
        this.hint.className = "bat-sam31-hint";
        root.appendChild(this.hint);

        const c = this.canvas;
        const L = (t, ty, fn, o) => this.track.listener(t, ty, fn, o);
        L(c, "pointerdown", (e) => this.onDown(e));
        L(c, "pointermove", (e) => this.onMove(e));
        L(c, "pointerup", (e) => this.onUp(e));
        L(c, "pointerleave", () => { if (this.hover !== -1) { this.hover = -1; this.draw(); } });
        L(c, "contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });
        this.track.observer(new ResizeObserver(() => this.draw()), stage);
        // A failed run would otherwise leave "Tracking queued…" on screen forever.
        L(api, "execution_error", (e) => {
            const d = e?.detail || {};
            const id = String(d.node_id ?? "");
            if (d.node_type !== NODE_TYPE) return;
            if (id !== String(this.node.id) && !id.endsWith(`:${this.node.id}`)) return;
            this.setStatus(`Run failed: ${d.exception_message || d.exception_type || "see the console"}`, true);
        });
    }

    get annotation() { return Math.max(0, Number(W(this.node, "frame_index_select")?.value) || 0); }
    get mode() { return W(this.node, "mode")?.value === "text" ? "text" : "click"; }
    get atAnnotation() { return this.view === this.annotation; }

    setStatus(text, warn = false) {
        this.status.textContent = text || "";
        this.status.classList.toggle("bat-sam31-warn", !!warn);
    }

    refreshChrome() {
        const n = this.count;
        this.slider.max = Math.max(0, n - 1);
        this.slider.disabled = n <= 1;
        this.slider.value = this.view;
        this.frameLabel.textContent = n ? `${this.view} / ${n - 1}` + (this.atAnnotation ? " ◎" : "")
                                        : `${this.view}`;
        this.annotateBtn.style.display = this.atAnnotation ? "none" : "";
        this.hint.textContent = !this.atAnnotation
            ? "Viewing a tracked frame — ◎ Annotate here to edit prompts on it."
            : this.mode === "text"
                ? "Type in text_prompt · click a match to keep / drop it · ▶ Track runs the clip"
                : "Click: object · right-click: not object · drag: box (right-click its edge to remove) · click a point to remove it";
    }

    // geometry ──────────────────────────────────────────────────────────────
    fitRect() {
        const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
        const fw = this.plate?.w || this.run?.w || 16, fh = this.plate?.h || this.run?.h || 9;
        const s = Math.min(cw / fw, ch / fh);
        const w = fw * s, h = fh * s;
        return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
    }

    /** Pointer -> canvas css px, dividing out the graph zoom (the element is CSS-scaled). */
    cssPoint(e) {
        const r = this.canvas.getBoundingClientRect();
        const sx = this.canvas.clientWidth / (r.width || 1), sy = this.canvas.clientHeight / (r.height || 1);
        return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
    }

    toNorm(p) {
        const f = this.fitRect();
        return { x: (p.x - f.x) / f.w, y: (p.y - f.y) / f.h };
    }

    fromNorm(nx, ny) {
        const f = this.fitRect();
        return { x: f.x + nx * f.w, y: f.y + ny * f.h };
    }

    // interaction ───────────────────────────────────────────────────────────
    pointNear(p) {
        const st = readState(this.node);
        for (let i = st.points.length - 1; i >= 0; i--) {
            const q = this.fromNorm(st.points[i].x, st.points[i].y);
            if (Math.hypot(q.x - p.x, q.y - p.y) <= HIT_RADIUS) return i;
        }
        return -1;
    }

    nearBoxEdge(p) {
        const b = readState(this.node).box;
        if (!b) return false;
        const a = this.fromNorm(b[0], b[1]), c = this.fromNorm(b[2], b[3]);
        const inX = p.x >= a.x - HIT_RADIUS && p.x <= c.x + HIT_RADIUS;
        const inY = p.y >= a.y - HIT_RADIUS && p.y <= c.y + HIT_RADIUS;
        const onV = Math.min(Math.abs(p.x - a.x), Math.abs(p.x - c.x)) <= HIT_RADIUS;
        const onH = Math.min(Math.abs(p.y - a.y), Math.abs(p.y - c.y)) <= HIT_RADIUS;
        return (onV && inY) || (onH && inX);
    }

    instanceAt(p) {
        const n = this.toNorm(p);
        if (n.x < 0 || n.y < 0 || n.x > 1 || n.y > 1) return -1;
        // Smallest first, so a person in front of a bigger one is reachable.
        const order = this.instances.map((_, i) => i)
            .sort((a, b) => (this.instances[a].mask?.area ?? 1) - (this.instances[b].mask?.area ?? 1));
        for (const i of order) {
            const m = this.instances[i].mask;
            if (!m) continue;
            const px = Math.min(m.w - 1, Math.floor(n.x * m.w)), py = Math.min(m.h - 1, Math.floor(n.y * m.h));
            if (m.lum[py * m.w + px] > 127) return i;
        }
        return -1;
    }

    onDown(e) {
        e.stopPropagation();
        if (!this.plate || !this.atAnnotation) return;
        this.canvas.setPointerCapture?.(e.pointerId);
        const p = this.cssPoint(e);
        this.drag = { start: p, now: p, button: e.button, moved: false };
    }

    onMove(e) {
        const p = this.cssPoint(e);
        if (this.drag) {
            this.drag.now = p;
            if (!this.drag.moved && Math.hypot(p.x - this.drag.start.x, p.y - this.drag.start.y) > DRAG_THRESHOLD
                && this.drag.button === 0 && this.mode === "click") {
                this.drag.moved = true;
            }
            if (this.drag.moved) this.draw();
            return;
        }
        if (this.mode === "text" && this.atAnnotation) {
            const h = this.instanceAt(p);
            if (h !== this.hover) { this.hover = h; this.draw(); }
        }
    }

    onUp(e) {
        e.stopPropagation();
        const d = this.drag;
        this.drag = null;
        if (!d || !this.plate || !this.atAnnotation) return;
        const st = readState(this.node);

        if (this.mode === "text") {
            if (d.button !== 0) return;
            const i = this.instanceAt(d.start);
            if (i < 0) return;
            this.instances[i].keep = !this.instances[i].keep;
            st.instances = this.instances.map((m) => ({ box: m.box, keep: m.keep }));
            writeState(this.node, st);
            this.draw();
            return;
        }

        if (d.moved) {
            const a = this.toNorm(d.start), b = this.toNorm(d.now);
            const c = (v) => Math.min(1, Math.max(0, v));
            st.box = [c(Math.min(a.x, b.x)), c(Math.min(a.y, b.y)), c(Math.max(a.x, b.x)), c(Math.max(a.y, b.y))];
        } else {
            const hit = this.pointNear(d.start);
            if (hit >= 0) {
                st.points.splice(hit, 1);
            } else if (d.button === 2 && this.nearBoxEdge(d.start)) {
                st.box = null;
            } else {
                const n = this.toNorm(d.start);
                if (n.x < 0 || n.y < 0 || n.x > 1 || n.y > 1) return;
                st.points.push({ x: n.x, y: n.y, label: d.button === 2 ? 0 : 1 });
            }
        }
        writeState(this.node, st);
        this.draw();
        this.requestLive();
    }

    clearPrompts() {
        const st = emptyState();
        this.instances.forEach((m) => { m.keep = true; });
        st.instances = this.instances.map((m) => ({ box: m.box, keep: true }));
        writeState(this.node, st);
        this.live = null;
        this.draw();
    }

    annotateHere() {
        // The widget's callback (chained in onNodeCreated) runs onAnnotationChanged.
        setWidgetValue(this.node, "frame_index_select", this.view);
    }

    onAnnotationChanged() {
        this.live = null;
        this.instances = []; this.instSig = "";
        this.setView(this.annotation);
    }

    // frames ────────────────────────────────────────────────────────────────
    setView(i) {
        const max = this.count ? this.count - 1 : Infinity;
        this.view = Math.max(0, Math.min(max, Math.round(i)));
        this.refreshChrome();
        this.loadPlate(this.view);
    }

    async loadPlate(i) {
        const gen = ++this.plateGen;
        const stale = () => gen !== this.plateGen || !isNodeAlive(this.node);
        let got = null;

        if (this.run?.liveFrames && i < this.run.count) {
            try {
                const img = await loadImage(api.apiURL(
                    `/bat/sam31/frame?token=${this.run.token}&i=${i}&kind=plate&max=${PLATE_MAX}`));
                got = { img, kind: "run", w: this.run.w, h: this.run.h };
            } catch (_) { this.run.liveFrames = false; }
        }
        if (!got && !stale()) got = await this.plateFromFile(i);
        if (!got && !stale() && i === 0) got = await this.plateFromNodePreview();
        if (!got && !stale() && i === this.annotation) {
            const saved = loadSavedPlate(this.node);
            if (saved?.src) {
                try { got = { img: await loadImage(saved.src), kind: "cache", w: saved.w, h: saved.h }; }
                catch (_) { /* stale cache */ }
            }
        }
        if (stale()) return;
        if (!got) {
            if (!this.plate) this.setStatus("No preview yet — connect a clip, or run once", true);
            this.draw();
            return;
        }
        this.plate = { ...got, frame: i, b64: null };
        if (i === this.annotation && got.kind !== "cache") this.cachePlate();
        if (this.run && i < this.run.count) this.loadRunMask(i);
        this.draw();
        if (i === this.annotation) {
            if (this.mode === "text") this.requestDetect(); else this.requestLive();
        }
    }

    async plateFromFile(i) {
        const source = findPathSource(this.node);
        if (!source) return null;
        const fileFrame = sourceFrameIndex(source.node, i, await fileFrameCount(source.path));
        const enc = encodeURIComponent(source.path);
        let w = 0, h = 0;
        try {
            const res = await fetch(api.apiURL(`/bat/video-info?path=${enc}`));
            if (res.ok) { const info = await res.json(); if (info?.ok) { w = +info.width || 0; h = +info.height || 0; } }
        } catch (_) { /* dimensions are an optimisation */ }
        try {
            const img = await loadImage(api.apiURL(
                `/bat/frame-picker/frame?path=${enc}&frame=${fileFrame}&max_w=${PLATE_MAX}`));
            return { img, kind: "file", w: w || img.naturalWidth, h: h || img.naturalHeight };
        } catch (_) { return null; }
    }

    async plateFromNodePreview() {
        const slot = (this.node.inputs ?? []).findIndex((inp) => inp.name === "frames");
        if (slot < 0) return null;
        const source = resolveSourcePreview(this.node, slot);
        if (!source) return null;
        if (source.isVideo && source.videoEl) {
            return await new Promise((resolve) => {
                captureVideoFrame(source.videoEl, (canvas) =>
                    resolve({ img: canvas, kind: "node", w: canvas.width, h: canvas.height }));
                setTimeout(() => resolve(null), 2000);
            });
        }
        if (source.url) {
            try {
                const img = await loadImage(source.url, { crossOrigin: true });
                return { img, kind: "node", w: img.naturalWidth, h: img.naturalHeight };
            } catch (_) { return null; }
        }
        return null;
    }

    cachePlate() {
        const p = this.plate;
        try {
            const c = document.createElement("canvas");
            const s = Math.min(1, 960 / Math.max(p.img.naturalWidth || p.img.width, p.img.naturalHeight || p.img.height));
            c.width = Math.round((p.img.naturalWidth || p.img.width) * s);
            c.height = Math.round((p.img.naturalHeight || p.img.height) * s);
            c.getContext("2d").drawImage(p.img, 0, 0, c.width, c.height);
            savePlate(this.node, { src: c.toDataURL("image/jpeg", 0.8), w: p.w, h: p.h });
        } catch (_) { /* tainted canvas or quota — a convenience only */ }
    }

    /** What the live routes should segment: the run's exact frame, else the plate. */
    liveSource() {
        const p = this.plate;
        if (!p || p.frame !== this.annotation) return null;
        if (p.kind === "run" && this.run?.liveFrames) return { token: this.run.token, frame: p.frame };
        if (!p.b64) {
            try {
                const c = document.createElement("canvas");
                c.width = p.img.naturalWidth || p.img.width;
                c.height = p.img.naturalHeight || p.img.height;
                c.getContext("2d").drawImage(p.img, 0, 0);
                p.b64 = c.toDataURL("image/jpeg", 0.92);
            } catch (_) { return null; }
        }
        return { image: p.b64 };
    }

    async loadRunMask(i) {
        if (!this.run || this.runMasks.has(i)) return;
        const token = this.run.token;
        this.runMasks.set(i, null);          // in flight
        try {
            const img = await loadImage(api.apiURL(
                `/bat/sam31/frame?token=${token}&i=${i}&kind=mask&max=${PLATE_MAX}`));
            if (this.run?.token !== token) return;
            this.runMasks.set(i, tintMask(img, MASK_RGB, 0.5));
            while (this.runMasks.size > RUN_MASK_CACHE) this.runMasks.delete(this.runMasks.keys().next().value);
            if (i === this.view) this.draw();
        } catch (_) { this.runMasks.delete(i); }
    }

    // live requests ─────────────────────────────────────────────────────────
    async requestLive() {
        if (this.mode !== "click" || !this.atAnnotation) return;
        if (this.liveBusy) { this.livePending = true; return; }
        const st = readState(this.node);
        if (!st.points.length && !st.box) { this.live = null; this.draw(); return; }
        const source = this.liveSource();
        if (!source) return;
        const sig = JSON.stringify([st.points, st.box, this.annotation, source.token || this.plate.frame]);
        if (this.live?.sig === sig) return;
        const adv = advancedSettings(this.node);
        this.liveBusy = true;
        try {
            const { status, data } = await apiJSON("/bat/sam31/live", {
                source, points: st.points, box: st.box, model: adv.model_file,
                refine_iterations: adv.refine_iterations,
            });
            if (!isNodeAlive(this.node)) return;
            if (data.busy) { this.setStatus("Job running — live preview resumes when it ends", true); return; }
            if (status !== 200 || !data.ok) { this.setStatus(data.error || `live preview failed (${status})`, true); return; }
            if (data.empty) { this.live = null; }
            else {
                const img = await loadImage(`data:image/png;base64,${data.mask}`);
                this.live = { mask: tintMask(img, MASK_RGB, 0.5), sig };
            }
            this.setStatus(data.encode_ms > 20 ? `encoded ${Math.round(data.encode_ms)} ms · mask ${Math.round(data.ms)} ms`
                                               : `mask ${Math.round(data.ms)} ms`);
            this.draw();
        } catch (e) {
            this.setStatus(`live preview failed: ${e?.message || e}`, true);
        } finally {
            this.liveBusy = false;
            if (this.livePending) { this.livePending = false; this.requestLive(); }
        }
    }

    scheduleDetect() {
        clearTimeout(this._detTimer);
        this._detTimer = setTimeout(() => this.requestDetect(), TEXT_DEBOUNCE_MS);
    }

    async requestDetect() {
        if (this.mode !== "text" || !this.atAnnotation) return;
        if (this.detBusy) { this.detPending = true; return; }
        const text = String(W(this.node, "text_prompt")?.value || "").trim();
        const source = this.liveSource();
        if (!source) return;
        const adv = advancedSettings(this.node);
        const sig = JSON.stringify([text, adv.detection_threshold, adv.max_objects, adv.model_file,
                                    this.annotation, source.token || this.plate.frame]);
        if (sig === this.instSig) return;
        if (!text) { this.instances = []; this.instSig = sig; this.setStatus("Type what to find in text_prompt"); this.draw(); return; }
        this.detBusy = true;
        this.setStatus(`finding “${text}”…`);
        try {
            const { status, data } = await apiJSON("/bat/sam31/detect", {
                source, text, threshold: adv.detection_threshold, max_objects: adv.max_objects,
                model: adv.model_file,
            });
            if (!isNodeAlive(this.node)) return;
            if (data.busy) { this.setStatus("Job running — live preview resumes when it ends", true); return; }
            if (status !== 200 || !data.ok) { this.setStatus(data.error || `detection failed (${status})`, true); return; }
            const saved = readState(this.node).instances;
            const keep = keepFlagsFor(data.instances, saved);
            this.instances = await Promise.all(data.instances.map(async (inst, i) => ({
                box: inst.box, score: inst.score, keep: keep[i],
                mask: tintMask(await loadImage(`data:image/png;base64,${inst.mask}`), instanceColour(i), 0.45),
            })));
            this.instSig = sig;
            const st = readState(this.node);
            st.instances = this.instances.map((m) => ({ box: m.box, keep: m.keep }));
            writeState(this.node, st);
            this.setStatus(`${this.instances.length} match${this.instances.length === 1 ? "" : "es"} · ${Math.round(data.ms)} ms`);
            this.draw();
        } catch (e) {
            this.setStatus(`detection failed: ${e?.message || e}`, true);
        } finally {
            this.detBusy = false;
            if (this.detPending) { this.detPending = false; this.requestDetect(); }
        }
    }

    // ▶ Track ───────────────────────────────────────────────────────────────
    /** This node's id(s) in the API prompt. Inside a subgraph the id is a path, so
     *  match on the node's own inputs instead; identical twins run together. */
    execIds(output) {
        const node = this.node;
        if (node.graph === app.graph && output[String(node.id)]?.class_type === NODE_TYPE) return [String(node.id)];
        const names = ["mode", "text_prompt", "frame_index_select", "detect_new_objects", "prompt_state"];
        const mine = JSON.stringify(names.map((n) => W(node, n)?.value));
        return Object.entries(output)
            .filter(([, v]) => v?.class_type === NODE_TYPE
                && JSON.stringify(names.map((n) => v.inputs?.[n])) === mine)
            .map(([k]) => k);
    }

    async queueTrack() {
        try {
            const { output, workflow } = await app.graphToPrompt();
            const ids = this.execIds(output);
            if (!ids.length) { this.setStatus("Couldn't find this node in the prompt", true); return; }
            await api.queuePrompt(0, { output: trimPrompt(output, ids), workflow });
            this.setStatus("Tracking queued…");
        } catch (e) {
            const msg = e?.response?.error?.message || e?.message || String(e);
            this.setStatus(`Track failed: ${msg}`, true);
            console.error("[Bat SAM3.1] track queue failed:", e);
        }
    }

    async adoptRun(run) {
        this.run = { ...run, liveFrames: true };
        this.runMasks.clear();
        this.live = null;
        this.node.properties = this.node.properties || {};
        this.node.properties.bat_sam31_token = run.token;
        try {
            const res = await fetch(api.apiURL(`/bat/sam31/info?token=${run.token}`));
            const info = await res.json();
            if (!info?.ok) { this.run = null; return false; }
            this.run.liveFrames = !!info.live_frames;
            this.run.count = info.count; this.run.w = info.w; this.run.h = info.h;
        } catch (_) { /* keep the optimistic flags */ }
        this.count = this.run.count;
        this.updateBounds();
        this.setView(this.view);
        return true;
    }

    // bounds ────────────────────────────────────────────────────────────────
    async updateBounds() {
        let count = this.run?.count || 0;
        if (!count) {
            const source = findPathSource(this.node);
            if (source) {
                const fileFrames = await fileFrameCount(source.path);
                if (fileFrames) count = sourceBatchLength(source.node, fileFrames);
            }
        }
        if (!isNodeAlive(this.node)) return;
        this.count = count;
        const w = W(this.node, "frame_index_select");
        if (w && count > 0) {
            w.options = w.options || {};
            w.options.min = 0;
            w.options.max = count - 1;
            const v = Number(w.value) || 0;
            if (v > count - 1) setWidgetValue(this.node, "frame_index_select", count - 1);   // fires onAnnotationChanged
        }
        if (this.view > count - 1 && count > 0) this.view = count - 1;
        this.refreshChrome();
    }

    // drawing ───────────────────────────────────────────────────────────────
    draw() {
        if (!isNodeAlive(this.node)) return;
        const c = this.canvas;
        const dpr = window.devicePixelRatio || 1;
        const cw = c.clientWidth, ch = c.clientHeight;
        if (!cw || !ch) return;
        if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) {
            c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr);
        }
        const x = c.getContext("2d");
        x.setTransform(dpr, 0, 0, dpr, 0, 0);
        x.fillStyle = "#111"; x.fillRect(0, 0, cw, ch);
        this.refreshChrome();
        const p = this.plate;
        if (!p) {
            // Say what to do rather than show a black box: the plate can only
            // come from a known loader, the upstream node's own preview, or a run.
            x.fillStyle = "#8a8a8a"; x.font = "13px sans-serif"; x.textAlign = "center";
            x.fillText("No picture yet", cw / 2, ch / 2 - 10);
            x.fillStyle = "#6d6d6d"; x.font = "12px sans-serif";
            x.fillText("Press ▶ Track (or queue the workflow) to load the clip, then click the object",
                       cw / 2, ch / 2 + 10);
            x.textAlign = "start";
            return;
        }
        const f = this.fitRect();
        x.imageSmoothingQuality = "high";
        if (p.frame === this.view) x.drawImage(p.img, f.x, f.y, f.w, f.h);

        const st = readState(this.node);
        if (this.showMask) {
            if (this.atAnnotation && this.mode === "text" && this.instances.length) {
                this.instances.forEach((m, i) => {
                    if (m.keep) x.drawImage(m.mask.canvas, f.x, f.y, f.w, f.h);
                });
            } else {
                // The live mask, while the artist is editing; otherwise the run's.
                const live = this.atAnnotation && this.mode === "click" ? this.live : null;
                const m = live ? live.mask : this.runMasks.get(this.view);
                if (m) x.drawImage(m.canvas, f.x, f.y, f.w, f.h);
            }
        }

        if (!this.atAnnotation) return;

        if (this.mode === "text") {
            x.font = "bold 12px sans-serif";
            this.instances.forEach((m, i) => {
                const a = this.fromNorm(m.box[0], m.box[1]), b = this.fromNorm(m.box[2], m.box[3]);
                const [r, g, bl] = instanceColour(i);
                x.setLineDash(m.keep ? [] : [4, 3]);
                x.lineWidth = i === this.hover ? 3 : 1.5;
                x.strokeStyle = m.keep ? `rgb(${r},${g},${bl})` : "rgba(160,160,160,.8)";
                x.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
                const tag = `${i + 1}${m.keep ? "" : " ✕"}`;
                x.fillStyle = "rgba(0,0,0,.65)";
                x.fillRect(a.x, a.y - 15, x.measureText(tag).width + 8, 15);
                x.fillStyle = m.keep ? "#fff" : "#aaa";
                x.fillText(tag, a.x + 4, a.y - 3);
            });
            x.setLineDash([]);
            return;
        }

        if (st.box) {
            const a = this.fromNorm(st.box[0], st.box[1]), b = this.fromNorm(st.box[2], st.box[3]);
            x.setLineDash([6, 4]); x.lineWidth = 1.5; x.strokeStyle = "#ffd24a";
            x.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
            x.setLineDash([]);
        }
        for (const pt of st.points) {
            const q = this.fromNorm(pt.x, pt.y);
            x.beginPath(); x.arc(q.x, q.y, 5.5, 0, Math.PI * 2);
            x.fillStyle = pt.label ? "#3ddc6a" : "#ff4d4d"; x.fill();
            x.lineWidth = 1.5; x.strokeStyle = "#fff"; x.stroke();
        }
        if (this.drag?.moved) {
            const a = this.drag.start, b = this.drag.now;
            x.setLineDash([4, 3]); x.strokeStyle = "#ffd24a"; x.lineWidth = 1;
            x.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
            x.setLineDash([]);
        }
    }
}

// ── node wiring ─────────────────────────────────────────────────────────────

function syncModeRows(node) {
    const text = W(node, "mode")?.value === "text";
    setRowHidden(W(node, "text_prompt"), !text);
    setRowHidden(W(node, "detect_new_objects"), !text);
    setRowHidden(W(node, "prompt_state"), true);
    refreshBatLayout(node);
}

function chainWidget(node, name, fn) {
    const w = W(node, name);
    if (!w) return;
    const orig = w.callback;
    w.callback = function (...args) {
        const r = orig?.apply(this, args);
        try { fn(...args); } catch (e) { console.error(`[Bat SAM3.1] ${name} handler failed:`, e); }
        return r;
    };
}

app.registerExtension({
    name: "BATNodePack.Sam31Segmenter",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        batReplayLastExecution(nodeType);
        registerCleanup(nodeType);

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const node = this;
            const ed = (node._sam31 = new Sam31Editor(node));
            const widget = addBatDOMWidget(node, "sam31_editor", "bat-sam31-editor", ed.root, {
                height: EDITOR_HEIGHT, growable: true, minWidth: 360, onResize: () => ed.draw(),
            });
            if (widget) addBatFullscreen(node, widget, ed.root, { title: "🦇 SAM 3.1 Segmenter" });
            if (!node.size || node.size[0] < 520) node.setSize?.([560, node.size?.[1] || 700]);

            syncModeRows(node);
            chainWidget(node, "mode", () => {
                syncModeRows(node);
                ed.live = null; ed.instSig = "";
                ed.refreshChrome();
                ed.draw();
                if (ed.mode === "text") ed.requestDetect(); else ed.requestLive();
            });
            chainWidget(node, "text_prompt", () => ed.scheduleDetect());
            chainWidget(node, "frame_index_select", () => ed.onAnnotationChanged());

            chainCallback(node, "onConnectionsChange", function (type) {
                if (type != null && type !== 1) return;
                clearTimeout(node._batSam31Conn);
                node._batSam31Conn = setTimeout(() => {
                    if (!isNodeAlive(node)) return;
                    ed.updateBounds().finally(() => ed.setView(ed.view));
                }, 150);
            });

            // Typing in text_prompt fires the widget callback only on commit in
            // Nodes 1.0; poll cheaply so detection follows the text as it changes.
            ed.track.interval(setInterval(() => {
                if (ed.mode !== "text") return;
                const t = String(W(node, "text_prompt")?.value || "");
                if (t !== ed._lastText) { ed._lastText = t; ed.scheduleDetect(); }
            }, 300));

            // Startup: after configure() has restored the widgets.
            setTimeout(async () => {
                if (!isNodeAlive(node)) return;
                syncModeRows(node);
                ed.view = ed.annotation;
                const token = node.properties?.bat_sam31_token;
                if (!batPreviewWillReplay(node) && token) {
                    if (await ed.adoptRun({ token, count: 0, w: 0, h: 0 })) return;
                    delete node.properties.bat_sam31_token;
                }
                await ed.updateBounds();
                ed.setView(ed.annotation);
            }, 150);
        });

        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            const ed = this._sam31;
            if (!ed || !message?.token) return;
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const note = one(message.note);
            ed.adoptRun({
                token: one(message.token), w: +one(message.w) || 0, h: +one(message.h) || 0,
                count: +one(message.frame_count) || 0, annotation: +one(message.annotation) || 0,
            }).then(() => ed.setStatus(note || `Tracked ${ed.count} frame${ed.count === 1 ? "" : "s"}`));
        });
    },
});
