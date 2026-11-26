/**
 * 🦇 Lines — per-filter widgets and the live preview.
 *
 * Widgets
 * -------
 * Every filter's settings are declared in Python, always, in a fixed order
 * (widgets_values are positional — saved workflows and the controlnet_aux
 * migrations depend on it). This file only HIDES the ones the current filter
 * ignores, with the pack's usual type/computeSize/draw/hidden swap (see
 * bat_batch_format.js). Hidden widgets still serialise. `visibleWidgets()` is
 * the single table of what shows when; tests/verify_lines_web.py runs it.
 *
 * Preview
 * -------
 * A run caches the `preview_frame` frame server-side under a token (bat_lines.py,
 * the Bat_Rescale pattern). Any settings change then re-runs the filter on that
 * frame through POST /bat/lines/render — the same run_filter() the node uses,
 * so the preview is the render, not an approximation. One request in flight at
 * a time; changes made meanwhile collapse into one follow-up with the latest
 * values. Views: the lines, the source, or a wipe between them (drag).
 * The last picture is kept per node in localStorage so a reopened workflow
 * shows something before the next run, and replayed after Ctrl+Z.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addBatDOMWidget, clampNodeSize, refreshBatLayout } from "./bat_node_layout.js";
import { addBatFullscreen } from "./bat_fullscreen.js";
import {
    batTrack, isNodeAlive, batNodeCacheKey, batCacheSet, batReplayLastExecution,
    batPreviewWillReplay,
} from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_Lines";
const DEBOUNCE_MS = 200;

/** Which widgets show for these values. Pure: tests/verify_lines_web.py runs it. */
function visibleWidgets(values) {
    const ALWAYS = ["filter", "resolution", "output_size", "invert", "preview_frame"];
    const MODEL_FILTERS = ["teed", "hed", "lineart realistic", "lineart anime", "lineart manga", "mlsd"];
    const PER_FILTER = {
        "canny": ["canny_low", "canny_high"],
        "xdog": ["xdog_style"],
        "lineart standard": ["standard_sigma", "standard_threshold"],
        "teed": ["teed_safe_steps"],
        "hed": ["hed_safe"],
        "lineart realistic": ["lineart_coarse"],
        "lineart anime": [],
        "lineart manga": [],
        "mlsd": ["mlsd_score", "mlsd_distance", "mlsd_line_width"],
    };
    const f = values.filter;
    let shown = ALWAYS.concat(PER_FILTER[f] || []);
    if (f === "xdog") {
        shown = shown.concat(values.xdog_style === "scribble"
            ? ["xdog_scribble_threshold"]
            : ["xdog_sigma", "xdog_k", "xdog_sharpen", "xdog_threshold", "xdog_softness"]);
    }
    if (MODEL_FILTERS.indexOf(f) >= 0) shown.push("auto_download");
    return shown;
}

// Every setting the preview depends on (all serialised widgets but the frame).
const PARAMS = [
    "filter", "resolution", "output_size", "invert", "canny_low", "canny_high",
    "xdog_style", "xdog_sigma", "xdog_k", "xdog_sharpen", "xdog_threshold",
    "xdog_softness", "xdog_scribble_threshold", "standard_sigma", "standard_threshold",
    "teed_safe_steps", "hed_safe", "lineart_coarse", "mlsd_score", "mlsd_distance",
    "mlsd_line_width", "auto_download",
];

const one = (v) => (Array.isArray(v) ? v[0] : v);

function setHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget._batOrigType === undefined) {
            widget._batOrigType = widget.type;
            widget._batOrigComputeSize = widget.computeSize;
            widget._batOrigDraw = widget.draw;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.hidden = true;
    } else if (widget._batOrigType !== undefined) {
        widget.type = widget._batOrigType;
        widget.computeSize = widget._batOrigComputeSize;
        widget.draw = widget._batOrigDraw;
        widget.hidden = false;
    }
}

function values(node) {
    const out = {};
    for (const w of node.widgets || []) {
        if (PARAMS.includes(w.name) || w.name === "preview_frame") out[w.name] = w.value;
    }
    return out;
}

function syncVisibility(node) {
    const shown = new Set(visibleWidgets(values(node)));
    for (const w of node.widgets || []) {
        if (PARAMS.includes(w.name) || w.name === "preview_frame") setHidden(w, !shown.has(w.name));
    }
    refreshBatLayout(node, null, { shrink: true });
}

/* ==========================================================================
 * Viewer
 * ========================================================================== */

function buildViewer(node) {
    const track = batTrack(node);
    const state = {
        token: null, frames: 1, anchor: 0,
        linesUrl: null, sourceUrl: null, sourceFrame: null,
        view: "lines", wipe: 0.5, status: "Run the node once to preview.",
        busy: false, pending: false, timer: 0, exact: true,
    };

    const root = document.createElement("div");
    root.className = "bat-lines-viewer";
    root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:200px;"
        + "background:#111;border-radius:4px;overflow:hidden;font:11px sans-serif;color:#ccc;";

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;gap:4px;align-items:center;padding:3px 6px;background:#1c1c1c;flex:0 0 auto;";
    const buttons = {};
    for (const [key, label] of [["lines", "Lines"], ["wipe", "Wipe"], ["source", "Source"]]) {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = "border:1px solid #444;background:#262626;color:#ccc;border-radius:3px;padding:1px 7px;cursor:pointer;font:11px sans-serif;";
        b.onclick = (e) => { e.stopPropagation(); state.view = key; paint(); if (key !== "lines") ensureSource(); };
        buttons[key] = b;
        bar.appendChild(b);
    }
    const status = document.createElement("span");
    status.style.cssText = "margin-left:auto;opacity:.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    bar.appendChild(status);

    const area = document.createElement("div");
    area.style.cssText = "position:relative;flex:1 1 auto;min-height:0;cursor:default;";
    const imgCss = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;";
    const srcImg = document.createElement("img");
    srcImg.style.cssText = imgCss;
    const linesImg = document.createElement("img");
    linesImg.style.cssText = imgCss;
    const handle = document.createElement("div");
    handle.style.cssText = "position:absolute;top:0;bottom:0;width:1px;background:#e6b800;pointer-events:none;";
    area.append(srcImg, linesImg, handle);
    root.append(bar, area);

    function paint() {
        for (const [k, b] of Object.entries(buttons)) {
            b.style.background = state.view === k ? "#3a3a3a" : "#262626";
            b.style.color = state.view === k ? "#fff" : "#ccc";
        }
        linesImg.src = state.linesUrl || "";
        srcImg.src = state.sourceUrl || "";
        linesImg.style.display = state.linesUrl && state.view !== "source" ? "" : "none";
        srcImg.style.display = state.sourceUrl && state.view !== "lines" ? "" : "none";
        const wiping = state.view === "wipe";
        linesImg.style.clipPath = wiping ? `inset(0 0 0 ${(state.wipe * 100).toFixed(2)}%)` : "";
        handle.style.display = wiping ? "" : "none";
        handle.style.left = `${(state.wipe * 100).toFixed(2)}%`;
        area.style.cursor = wiping ? "ew-resize" : "default";
        status.textContent = state.status + (state.exact ? "" : " *");
        status.title = state.exact ? state.status
            : "* The batch is no longer cached, so the frame the node last ran on stands in. Run the node to preview other frames.";
    }

    // Wipe: drag anywhere in the picture.
    let dragging = false;
    const setWipe = (e) => {
        const r = area.getBoundingClientRect();
        if (r.width > 0) state.wipe = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
        paint();
    };
    track.listener(area, "pointerdown", (e) => {
        if (state.view !== "wipe") return;
        dragging = true;
        area.setPointerCapture?.(e.pointerId);
        setWipe(e);
        e.stopPropagation();
    });
    track.listener(area, "pointermove", (e) => { if (dragging) setWipe(e); });
    track.listener(area, "pointerup", () => { dragging = false; });

    const frameIndex = () => {
        const pf = Number(node.widgets?.find((w) => w.name === "preview_frame")?.value) || 0;
        return Math.max(0, Math.min(pf, Math.max(0, state.frames - 1)));
    };

    function setLinesUrl(url) {
        if (state.linesUrl?.startsWith("blob:")) URL.revokeObjectURL(state.linesUrl);
        state.linesUrl = url;
    }
    function setSourceUrl(url) {
        if (state.sourceUrl?.startsWith("blob:")) URL.revokeObjectURL(state.sourceUrl);
        state.sourceUrl = url;
    }

    async function post(body) {
        // timeoutMs: null — frontend 1.58 aborts fetchApi after 60 s otherwise,
        // and a model filter's first preview can include loading its weights.
        return api.fetchApi("/bat/lines/render", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            timeoutMs: null,
        });
    }

    async function render() {
        if (!state.token || !isNodeAlive(node)) return;
        if (state.busy) { state.pending = true; return; }
        state.busy = true;
        state.pending = false;
        const frame = frameIndex();
        state.status = `Rendering frame ${frame}…`;
        paint();
        try {
            const r = await post({ token: state.token, frame, ...values(node) });
            if (!isNodeAlive(node)) return;
            if (r.status === 409) {
                state.token = null;
                state.status = "Run the node to preview (the cached frame is gone).";
            } else if (!r.ok) {
                let msg = `HTTP ${r.status}`;
                try { msg = (await r.json()).error || msg; } catch (_) {}
                state.status = `Preview failed: ${msg}`;
            } else {
                let info = {};
                try { info = JSON.parse(r.headers.get("X-Bat-Lines") || "{}"); } catch (_) {}
                setLinesUrl(URL.createObjectURL(await r.blob()));
                state.exact = info.exact !== false;
                state.status = `Frame ${frame}` + (info.w ? ` · ${info.w}×${info.h}` : "");
                if (state.view !== "lines") ensureSource();
            }
        } catch (e) {
            state.status = `Preview failed: ${e?.message || e}`;
        } finally {
            state.busy = false;
            paint();
            if (state.pending && isNodeAlive(node)) render();
        }
    }

    async function ensureSource() {
        const frame = frameIndex();
        if (!state.token || state.sourceFrame === frame) return;
        state.sourceFrame = frame;
        try {
            const r = await post({ token: state.token, frame, want: "source" });
            if (r.ok && isNodeAlive(node)) {
                setSourceUrl(URL.createObjectURL(await r.blob()));
                paint();
            } else {
                state.sourceFrame = null;
            }
        } catch (_) { state.sourceFrame = null; }
    }

    function schedule() {
        if (state.timer) clearTimeout(state.timer);
        state.timer = track.timeout(setTimeout(() => { state.timer = 0; render(); }, DEBOUNCE_MS));
    }

    node._batLinesChanged = (name) => {
        if (name === "preview_frame") state.sourceFrame = null;
        if (!state.token) return;
        schedule();
    };

    const cacheKey = () => batNodeCacheKey(app, "bat_lines", node);

    node._batLinesIngest = (msg) => {
        state.token = one(msg.token) || null;
        state.frames = Number(one(msg.frames)) || 1;
        state.anchor = Number(one(msg.preview_frame)) || 0;
        state.exact = true;
        const thumb = one(msg.thumb);
        const source = one(msg.source);
        if (thumb) setLinesUrl(`data:image/jpeg;base64,${thumb}`);
        if (source) { setSourceUrl(`data:image/jpeg;base64,${source}`); state.sourceFrame = state.anchor; }
        state.status = `Frame ${state.anchor} · as run`;
        paint();
        if (thumb && state.token) {
            batCacheSet(cacheKey(), JSON.stringify({
                token: state.token, frames: state.frames, anchor: state.anchor, thumb, source,
            }));
        }
    };

    node._batLinesRestore = async () => {
        if (batPreviewWillReplay(node)) return;
        let cached = null;
        try { cached = JSON.parse(localStorage.getItem(cacheKey()) || "null"); } catch (_) {}
        if (!cached) { paint(); return; }
        state.frames = Number(cached.frames) || 1;
        state.anchor = Number(cached.anchor) || 0;
        if (cached.thumb) setLinesUrl(`data:image/jpeg;base64,${cached.thumb}`);
        if (cached.source) { setSourceUrl(`data:image/jpeg;base64,${cached.source}`); state.sourceFrame = state.anchor; }
        state.status = "Last run (run again to preview changes).";
        paint();
        if (!cached.token) return;
        try {
            const r = await api.fetchApi(`/bat/lines/info?token=${encodeURIComponent(cached.token)}`);
            const j = await r.json();
            if (!isNodeAlive(node) || !j?.ok) return;
            state.token = cached.token;
            state.frames = Number(j.frames) || state.frames;
            state.exact = !!j.live_batch;
            render();
        } catch (_) { /* server restarted: the cached picture is what we have */ }
    };

    track.dispose(() => {
        if (state.timer) clearTimeout(state.timer);
        setLinesUrl(null);
        setSourceUrl(null);
    });

    paint();
    return root;
}

/* ==========================================================================
 * Registration
 * ========================================================================== */

function hookWidgets(node) {
    for (const w of node.widgets || []) {
        if (!(PARAMS.includes(w.name) || w.name === "preview_frame") || w._batLinesHooked) continue;
        w._batLinesHooked = true;
        const orig = w.callback;
        w.callback = function (...args) {
            const r = orig ? orig.apply(this, args) : undefined;
            if (w.name === "filter" || w.name === "xdog_style") syncVisibility(node);
            node._batLinesChanged?.(w.name);
            return r;
        };
    }
}

app.registerExtension({
    name: "Bat.Lines",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // A graph reload (Ctrl+Z is one) rebuilds every node; replay the last
        // run's preview into the new instance.
        batReplayLastExecution(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = buildViewer(this);
            const widget = addBatDOMWidget(this, "bat_lines_preview", "bat_lines_preview", el, {
                minWidth: 300, height: 280, growable: true,
            });
            addBatFullscreen(this, widget, el);
            hookWidgets(this);
            syncVisibility(this);
            clampNodeSize(this, 300, 0);
            // Deferred: node.id is final only once litegraph has finished
            // constructing, and the localStorage key is scoped by it.
            setTimeout(() => this._batLinesRestore?.(), 0);
            return r;
        };

        // configure() restores widget values after onNodeCreated: re-sync, or a
        // loaded workflow shows the default filter's settings.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            hookWidgets(this);
            syncVisibility(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && this._batLinesIngest) this._batLinesIngest(message);
            return r;
        };
    },
});
