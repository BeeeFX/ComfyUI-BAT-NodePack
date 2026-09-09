/**
 * BAT — Bat_FramePicker front-end.
 *
 * Replaces the `path` STRING widget with a rounded path-autocomplete textbox
 * (the same popup the Volt/BAT loaders use, backed by
 * /bat/frame-picker/getpath) and adds a scrollable contact-sheet grid of every
 * frame in the chosen video or sequence. Clicking a cell sets `frame_index`;
 * the node then decodes only that one frame on execute.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { makeBatPathWidget } from "./bat_path_widget.js";
import { batTrack } from "./bat_lifecycle.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Bat_FramePicker";
const PATH_ROUTE = "/bat/frame-picker/getpath";
const INFO_ROUTE = "/bat/frame-picker/info";
const FRAME_ROUTE = "/bat/frame-picker/frame";

function debounce(fn, ms) {
    let t = null;
    return (...a) => { if (t) clearTimeout(t); t = setTimeout(() => { t = null; fn(...a); }, ms); };
}

// ─── Contact-sheet grid ───────────────────────────────────────────────────────

function makeGridWidget() {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;height:100%;width:100%;box-sizing:border-box;background:#0c0c0c;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden;";

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;align-items:center;gap:8px;padding:5px 8px;background:#161616;font:11px monospace;color:#bbb;";
    const label = document.createElement("span");
    label.textContent = "no source";
    const sizeWrap = document.createElement("label");
    sizeWrap.style.cssText = "display:flex;align-items:center;gap:6px;margin-left:auto;";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "48"; slider.max = "200";
    slider.value = localStorage.getItem("bat-frame-cell") || "96";
    sizeWrap.append(document.createTextNode("size"), slider);
    bar.append(label, sizeWrap);

    const grid = document.createElement("div");
    grid.style.cssText = "flex:1;overflow-y:auto;padding:8px;display:grid;gap:6px;align-content:start;";
    const applyCell = () => { grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${slider.value}px, 1fr))`; };
    applyCell();
    slider.addEventListener("input", () => { applyCell(); localStorage.setItem("bat-frame-cell", slider.value); });

    root.append(bar, grid);
    return { element: root, grid, label };
}

async function fetchInfo(node) {
    const st = node._batFrame;
    if (!st) return;
    const path = st.pathW.value;
    if (!path) { st.preview.label.textContent = "no source"; st.preview.grid.innerHTML = ""; return; }

    // Sequence guard + abort. Counting a large sequence on NFS can take seconds,
    // so two overlapping lookups could resolve out of order and leave the grid
    // showing the frame count of a path the artist already navigated away from.
    // Only the newest request is allowed to touch the UI; older ones are
    // aborted and their late responses ignored.
    const seq = ++st._seq;
    if (st._abort) { try { st._abort.abort(); } catch (_) {} }
    const ac = (typeof AbortController === "function") ? new AbortController() : null;
    st._abort = ac;

    try {
        const r = await fetch(
            api.apiURL(`${INFO_ROUTE}?path=${encodeURIComponent(path)}`),
            ac ? { signal: ac.signal } : undefined,
        );
        const info = await r.json();
        if (seq !== st._seq) return;                 // superseded — drop it
        if (!info.ok) { st.preview.label.textContent = info.error || "could not read source"; st.preview.grid.innerHTML = ""; st.frames = 0; return; }
        st.frames = info.frames || 0;
        st.type = info.type || "";
        buildGrid(node);
        node.setDirtyCanvas(true, true);
    } catch (e) {
        if (e && e.name === "AbortError") return;    // expected on supersede
        if (seq !== st._seq) return;
        st.preview.label.textContent = "lookup failed";
    } finally {
        if (st._abort === ac) st._abort = null;
    }
}

function buildGrid(node) {
    const st = node._batFrame;
    const grid = st.preview.grid;
    grid.innerHTML = "";
    if (st._io) { st._io.disconnect(); st._io = null; }

    const frames = st.frames || 0;
    st.preview.label.textContent = `${frames} frames · ${st.type || "—"}`;
    if (!frames) return;

    st._io = new IntersectionObserver((entries) => {
        for (const e of entries) {
            if (!e.isIntersecting) continue;
            const img = e.target.querySelector("img");
            if (img && !img.src) {
                img.src = api.apiURL(
                    `${FRAME_ROUTE}?path=${encodeURIComponent(st.pathW.value)}` +
                    `&frame=${e.target.dataset.frame}&max_w=256`
                );
            }
            st._io.unobserve(e.target);
        }
    }, { root: grid, rootMargin: "200px" });

    const selected = st.frameW.value | 0;
    for (let i = 0; i < frames; i++) {
        const cell = document.createElement("div");
        cell.dataset.frame = String(i);
        cell.style.cssText = "position:relative;width:100%;height:0;padding-bottom:75%;background:#1a1a1a;border:2px solid transparent;border-radius:3px;overflow:hidden;cursor:pointer;box-sizing:border-box;";
        if (i === selected) cell.style.borderColor = "#7ec0ee";
        const img = document.createElement("img");
        img.loading = "lazy";
        img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;";
        const num = document.createElement("span");
        num.textContent = i;
        num.style.cssText = "position:absolute;left:3px;bottom:3px;font:10px monospace;color:#fff;background:rgba(0,0,0,0.6);padding:0 3px;border-radius:2px;";
        cell.append(img, num);
        cell.onclick = () => selectFrame(node, i);
        grid.appendChild(cell);
        st._io.observe(cell);
    }
}

function selectFrame(node, i) {
    const st = node._batFrame;
    st.frameW.value = i;
    st.frameW.callback?.(i);
    highlightSelected(node);
    node.setDirtyCanvas(true, true);
}

function highlightSelected(node) {
    const st = node._batFrame;
    const sel = st.frameW.value | 0;
    st.preview.grid.querySelectorAll("[data-frame]").forEach(cell => {
        cell.style.borderColor = (Number(cell.dataset.frame) === sel) ? "#7ec0ee" : "transparent";
    });
    const selCell = st.preview.grid.querySelector(`[data-frame="${sel}"]`);
    if (selCell) selCell.scrollIntoView({ block: "nearest" });
}

app.registerExtension({
    name: "BAT.FramePicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!this.widgets) return r;

            const pathIdx = this.widgets.findIndex(w => w.name === "path");
            const frameW = this.widgets.find(w => w.name === "frame_index");
            if (pathIdx < 0 || !frameW) return r;

            // Swap the plain path STRING widget for the autocompleting one.
            const orig = this.widgets[pathIdx];
            const config = nodeData.input?.required?.path;
            const opts = (config && config[1]) || {};
            const pathW = makeBatPathWidget({
                name: "path", value: orig.value || "", options: opts,
                route: PATH_ROUTE, title: "Frame Source",
            });
            this.widgets[pathIdx] = pathW;

            const preview = makeGridWidget();
            // Dual-mode sizing — the contact-sheet grid is flex:1 inside a
            // height:100% root, so it had no height source under Nodes 2.0.
            addBatDOMWidget(this, "frame_grid", "frame_grid", preview.element, {
                minWidth: 420, height: 520, growable: true,
            });

            // `_seq` stamps each fetchInfo() call so a slow response for an old
            // path can't overwrite the grid for the path the artist is on now
            // (a 20k-file sequence on NFS can take seconds to count).
            this._batFrame = { pathW, frameW, preview, frames: 0, type: "",
                               _io: null, _seq: 0, _abort: null };

            const node = this;
            // Release the IntersectionObserver (and its observed cells) plus any
            // in-flight request when the node is deleted.
            const track = batTrack(this);
            track.dispose(() => {
                const st = node._batFrame;
                if (!st) return;
                if (st._io) { try { st._io.disconnect(); } catch (_) {} st._io = null; }
                if (st._abort) { try { st._abort.abort(); } catch (_) {} st._abort = null; }
            });
            const refetch = debounce(() => fetchInfo(node), 300);
            pathW.callback = () => refetch();
            const origFrameCb = frameW.callback;
            frameW.callback = function (v) { origFrameCb?.apply(this, arguments); highlightSelected(node); };

            clampNodeSize(this, 420, 520);
            queueMicrotask(() => fetchInfo(this));
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            queueMicrotask(() => fetchInfo(this));
            return r;
        };
    },
});
