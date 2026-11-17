/**
 * 🦇 Loader — node face.
 *
 * Two small jobs:
 *
 *  1. Swap the plain `path` string box for the pack's autocomplete widget, so a
 *     path is picked from the filesystem rather than typed from memory. Same
 *     widget 🦇 Video Loader and 🦇 Frame Picker use, pointed at the same
 *     `/bat/getpath` route.
 *
 *  2. Say what the path resolves to — "EXR sequence · 48 frames", "movie",
 *     "folder" — underneath it. A `####` pattern is the one input here whose
 *     meaning isn't obvious from looking at it: whether it matched 48 frames or
 *     none is the difference between a working graph and an error on execute,
 *     and waiting until execute to find out is the slow way to learn you typed
 *     `###` where the files have four digits.
 *
 * The count comes from `/bat/loader-scan`, which only lists a directory — it
 * never opens a frame, so it stays cheap enough to fire on every path change.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { installBatPathWidget, makeBatPathWidget } from "./bat_path_widget.js";

const NODE_TYPE = "Bat_Loader";
const PATH_ROUTE = "/bat/getpath";
const SCAN_ROUTE = "/bat/loader-scan";

/** Debounce, so holding a key down doesn't queue a scan per character. */
function debounce(fn, ms) {
    let t = null;
    return function (...args) {
        if (t) clearTimeout(t);
        t = setTimeout(() => { t = null; fn.apply(this, args); }, ms);
    };
}

async function scanPath(node) {
    const pathW = node.widgets?.find((w) => w.name === "path");
    if (!pathW) return;
    const value = (pathW.value || "").trim();
    if (!value) { node._batScan = null; redrawSummary(node, pathW); return; }

    // Same supersede guard the video loader uses: a slow answer for a path the
    // artist has already replaced must not overwrite the newer one's summary.
    const seq = (node._batScanSeq = (node._batScanSeq || 0) + 1);
    const pathAtRequest = value;
    if (node._batScanAbort) { try { node._batScanAbort.abort(); } catch (_) {} }
    const ac = (typeof AbortController === "function") ? new AbortController() : null;
    node._batScanAbort = ac;

    try {
        const r = await fetch(
            api.apiURL(`${SCAN_ROUTE}?path=${encodeURIComponent(pathAtRequest)}`),
            ac ? { signal: ac.signal } : undefined,
        );
        const info = await r.json();
        if (seq !== node._batScanSeq || (pathW.value || "").trim() !== pathAtRequest) return;
        node._batScan = info;
        redrawSummary(node, pathW);
    } catch (e) {
        if (e && e.name === "AbortError") return;      // expected on supersede
        console.warn("[Bat] loader-scan failed:", e);
    } finally {
        if (node._batScanAbort === ac) node._batScanAbort = null;
    }
}

const scanDebounced = debounce(scanPath, 220);

/** Repaint the summary in whichever renderer is drawing it. */
function redrawSummary(node, pathW) {
    node.setDirtyCanvas(true, true);
    // Nodes 2.0 draws it inside the path widget's own canvas, which repaints
    // only on triggerDraw (WidgetLegacy.vue) — setDirtyCanvas doesn't reach it.
    pathW?.triggerDraw?.();
}

/** One line of plain English about what the path points at. */
function summarise(info) {
    if (!info) return ["", "#8a93a0"];
    if (!info.ok) return [info.error || "not found", "#d98a8a"];
    const bits = [];
    if (info.kind) bits.push(info.kind);
    if (info.frames > 0) bits.push(`${info.frames} frame${info.frames === 1 ? "" : "s"}`);
    if (info.width && info.height) bits.push(`${info.width}x${info.height}`);
    if (info.layers > 0) bits.push(`${info.layers} layer${info.layers === 1 ? "" : "s"}`);
    if (info.has_audio) bits.push("audio");
    // A #### pattern that matched nothing is the failure this line exists to
    // catch, so it is coloured like an error even though the route said ok.
    const bad = !!info.pattern && !info.frames;
    return [bits.join(" · ") || "—", bad ? "#d98a8a" : "#8a93a0"];
}

app.registerExtension({
    name: "BAT.Loader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const pathIdx = this.widgets?.findIndex((w) => w.name === "path");
            if (pathIdx >= 0) {
                const orig = this.widgets[pathIdx];
                const opts = (nodeData.input?.required?.path || [])[1] || {};
                const node = this;
                const path = makeBatPathWidget({
                    name: "path", value: orig.value || "", options: opts,
                    route: PATH_ROUTE, title: "Media Path",
                    // Nodes 2.0's home for the summary line; 1.0 keeps
                    // painting it in onDrawForeground below.
                    subtitle: () => summarise(node._batScan),
                });
                path.callback = () => scanPath(this);
                installBatPathWidget(this, pathIdx, path);
            }

            return r;
        };

        // A workflow loads with a path already in it, and nothing has typed
        // anything — scan once so the summary is there before the first run.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => scanDebounced(this), 0);
            return r;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
            if (this.flags?.collapsed) return r;
            const pathW = this.widgets?.find((w) => w.name === "path");
            if (!pathW || pathW.last_y == null) return r;

            const [text, colour] = summarise(this._batScan);
            if (!text) return r;
            ctx.save();
            ctx.font = "10px monospace";
            ctx.fillStyle = colour;
            ctx.textAlign = "right";
            // Just above the path row, right-aligned, so it never collides with
            // the widget's own label or its value.
            ctx.fillText(text, this.size[0] - 14, pathW.last_y - 3);
            ctx.restore();
            return r;
        };
    },
});
