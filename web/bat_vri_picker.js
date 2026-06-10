/**
 * BAT — Bat_VriPicker front-end.
 *
 * Turns the two component STRING widgets into dropdowns (filled from
 * /bat/vri-info) and adds a scrollable, resizable contact-sheet grid of every
 * frame of the *preview* component. Clicking a cell sets `frame_index`; the
 * node then decodes only that one frame from the *output* component.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Bat_VriPicker";
const INFO_ROUTE = "/bat/vri-info";
const FRAME_ROUTE = "/bat/vri-frame";

function debounce(fn, ms) {
    let t = null;
    return (...a) => { if (t) clearTimeout(t); t = setTimeout(() => { t = null; fn(...a); }, ms); };
}

const PLACEHOLDER = "(paste a VRI to see components)";

// Set a combo widget's option list + value (the widget is already a combo —
// see the type rewrite in beforeRegisterNodeDef).
function setComboValues(widget, values, preferred) {
    widget.options = widget.options || {};
    widget.options.values = values.length ? values : [PLACEHOLDER];
    if (!values.length) { widget.value = PLACEHOLDER; return; }
    if (!values.includes(widget.value) || widget.value === PLACEHOLDER) {
        widget.value = preferred && values.includes(preferred) ? preferred : values[0];
    }
}

function pickPreviewDefault(components) {
    // Prefer a lightweight image *sequence* (the actual per-frame proxy, e.g.
    // view_rec709.####.jpg) over single stills like thumbnail.jpg.
    const score = (c) => {
        const n = c.name.toLowerCase();
        let s = 0;
        if (c.type === "sequence") s += 100;   // multi-frame proxy beats a single still
        if (/\.(jpg|jpeg)$/.test(n)) s += 20;
        else if (/\.(png|webp)$/.test(n)) s += 10;
        if (n.includes("thumb")) s -= 200;      // de-prioritise thumbnail.*
        if (/\.exr$/.test(n)) s -= 500;          // never default preview to heavy exr
        if (c.type === "video") s -= 100;
        s += Math.min(30, (c.frames || 0) * 0.01);
        return s;
    };
    const best = [...components].sort((a, b) => score(b) - score(a))[0];
    return best ? best.name : "";
}
function pickOutputDefault(components) {
    const names = components.map(c => c.name);
    return names.find(n => /\.exr$/i.test(n)) || names[0] || "";
}

function makeGridWidget() {
    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;height:100%;width:100%;box-sizing:border-box;background:#0c0c0c;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden;";

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;align-items:center;gap:8px;padding:5px 8px;background:#161616;font:11px monospace;color:#bbb;";
    const label = document.createElement("span");
    label.textContent = "0 frames";
    const sizeWrap = document.createElement("label");
    sizeWrap.style.cssText = "display:flex;align-items:center;gap:6px;margin-left:auto;";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "48"; slider.max = "200";
    slider.value = localStorage.getItem("bat-vri-cell") || "96";
    sizeWrap.append(document.createTextNode("size"), slider);
    bar.append(label, sizeWrap);

    const grid = document.createElement("div");
    grid.style.cssText = "flex:1;overflow-y:auto;padding:8px;display:grid;gap:6px;align-content:start;";
    const applyCell = () => { grid.style.gridTemplateColumns = `repeat(auto-fill, minmax(${slider.value}px, 1fr))`; };
    applyCell();
    slider.addEventListener("input", () => { applyCell(); localStorage.setItem("bat-vri-cell", slider.value); });

    root.append(bar, grid);
    return { element: root, grid, label };
}

async function fetchVriInfo(node) {
    const st = node._batVri;
    const vri = st.vriW.value;
    if (!vri) return;
    try {
        const r = await fetch(api.apiURL(`${INFO_ROUTE}?vri=${encodeURIComponent(vri)}`));
        const info = await r.json();
        if (!info.ok) { st.preview.label.textContent = info.error || "VRI did not resolve"; st.preview.grid.innerHTML = ""; return; }
        st.components = info.components || [];
        const names = st.components.map(c => c.name);
        setComboValues(st.previewW, names, pickPreviewDefault(st.components));
        setComboValues(st.outputW, names, pickOutputDefault(st.components));
        buildGrid(node);
        node.setDirtyCanvas(true, true);
    } catch (e) {
        st.preview.label.textContent = "VRI lookup failed";
    }
}

function buildGrid(node) {
    const st = node._batVri;
    const grid = st.preview.grid;
    grid.innerHTML = "";
    if (st._io) { st._io.disconnect(); st._io = null; }

    const comp = st.components.find(c => c.name === st.previewW.value);
    const frames = comp ? (comp.frames || 0) : 0;
    st.preview.label.textContent = `${frames} frames · ${st.previewW.value || "—"}`;
    if (!frames) return;

    // Lazy-load thumbnails as cells scroll into view.
    st._io = new IntersectionObserver((entries) => {
        for (const e of entries) {
            if (!e.isIntersecting) continue;
            const img = e.target.querySelector("img");
            if (img && !img.src) {
                img.src = api.apiURL(
                    `${FRAME_ROUTE}?vri=${encodeURIComponent(st.vriW.value)}` +
                    `&component=${encodeURIComponent(st.previewW.value)}` +
                    `&frame=${e.target.dataset.frame}&max_w=256`
                );
            }
            st._io.unobserve(e.target);
        }
    }, { root: grid, rootMargin: "200px" });

    const selected = st.frameW.value | 0;
    for (let i = 0; i < frames; i++) {
        // Square cell via the padding-bottom hack (robust everywhere — the
        // aspect-ratio property was leaving rows with no height, so tiles
        // overlapped at larger sizes). 56.25% would be 16:9; 75% gives a
        // 4:3-ish tile that shows most of the frame without huge letterbox.
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
    const st = node._batVri;
    st.frameW.value = i;
    st.frameW.callback?.(i);
    highlightSelected(node);
    node.setDirtyCanvas(true, true);
}

function highlightSelected(node) {
    const st = node._batVri;
    const sel = st.frameW.value | 0;
    st.preview.grid.querySelectorAll("[data-frame]").forEach(cell => {
        cell.style.borderColor = (Number(cell.dataset.frame) === sel) ? "#7ec0ee" : "transparent";
    });
    const selCell = st.preview.grid.querySelector(`[data-frame="${sel}"]`);
    if (selCell) selCell.scrollIntoView({ block: "nearest" });
}

app.registerExtension({
    name: "BAT.VriPicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // Rewrite the component inputs from STRING → a string-list type so
        // LiteGraph builds real COMBO (dropdown) widgets. The server still
        // validates them as STRING, so the chosen value passes through fine.
        // (Same trick Volt Loader uses for its component dropdown.)
        const req = nodeData.input?.required || {};
        for (const key of ["preview_component", "output_component"]) {
            if (req[key] && req[key][0] === "STRING") {
                req[key] = [[PLACEHOLDER], req[key][1] || {}];
            }
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!this.widgets) return r;

            const vriW = this.widgets.find(w => w.name === "vri");
            const previewW = this.widgets.find(w => w.name === "preview_component");
            const outputW = this.widgets.find(w => w.name === "output_component");
            const frameW = this.widgets.find(w => w.name === "frame_index");
            if (!vriW || !previewW || !outputW || !frameW) return r;

            // Components are already COMBO widgets (type rewritten above);
            // show the placeholder until /bat/vri-info fills them.
            setComboValues(previewW, [], "");
            setComboValues(outputW, [], "");

            const preview = makeGridWidget();
            this.addDOMWidget("vri_grid", "vri_grid", preview.element, { serialize: false, hideOnZoom: false });

            this._batVri = { vriW, previewW, outputW, frameW, preview, components: [], _io: null };

            const node = this;
            const refetch = debounce(() => fetchVriInfo(node), 300);
            const origVriCb = vriW.callback;
            vriW.callback = function (v) { origVriCb?.apply(this, arguments); refetch(); };
            const origPrevCb = previewW.callback;
            previewW.callback = function (v) { origPrevCb?.apply(this, arguments); buildGrid(node); };
            const origFrameCb = frameW.callback;
            frameW.callback = function (v) { origFrameCb?.apply(this, arguments); highlightSelected(node); };

            this.size = [Math.max(420, this.size[0] || 0), Math.max(this.size[1] || 0, 520)];
            queueMicrotask(() => fetchVriInfo(this));
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            queueMicrotask(() => fetchVriInfo(this));
            return r;
        };
    },
});
