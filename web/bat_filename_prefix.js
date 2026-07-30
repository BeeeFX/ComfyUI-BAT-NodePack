/**
 * BAT — Bat_FilenamePrefix front-end.
 *
 * Builds a `filename_prefix` string from an ordered list of user-defined path
 * segments, so a workflow doesn't need its own String + Concatenate chain.
 *
 * Rather than one litegraph widget per field (which made the node a tower of
 * cluttered rows), the whole editor is a single DOM widget — a compact table,
 * one row per segment — mirroring the addDOMWidget pattern used by bat_grade.js
 * and the frame picker. Two buttons add Name / Version segments; each row has
 * its own separator, an editable label, the value field(s), and small
 * up / down / remove icons.
 *
 * Persistence: the segment list is serialised into the hidden `segments` STRING
 * widget (JSON). That value rides in the saved workflow, so on load
 * `onConfigure` rebuilds the table from it — before the first run — and the
 * state survives restarts. The Python side reads that same JSON.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "Bat_FilenamePrefix";
const VERSION_MODES = ["fixed", "increment", "decrement", "randomize"];

// ─── State model ─────────────────────────────────────────────────────────────

function defaultSegment(kind) {
    if (kind === "version") {
        return { kind: "version", label: "", value: 1, separator: "/",
                 prefix: "v", pad: 3, mode: "fixed", _userLabel: false };
    }
    return { kind: "name", label: "", value: "", separator: "/", _userLabel: false };
}

function relabel(node) {
    node._batSegs.forEach((seg, i) => {
        if (!seg._userLabel) seg.label = `segment_${i + 1}`;
    });
}

function serialize(node) {
    const w = node._batSegsWidget;
    if (!w) return;
    w.value = JSON.stringify(node._batSegs.map((s) => ({
        kind: s.kind, label: s.label, value: s.value, separator: s.separator,
        ...(s.kind === "version" ? { prefix: s.prefix, pad: s.pad, mode: s.mode } : {}),
        _userLabel: !!s._userLabel,
    })));
}

function loadSegments(node) {
    let data = [];
    try { data = JSON.parse(node._batSegsWidget?.value || "[]"); } catch { data = []; }
    if (!Array.isArray(data)) data = [];
    node._batSegs = data.map((s) => Object.assign(
        defaultSegment(s.kind === "version" ? "version" : "name"), s,
        { _userLabel: !!s._userLabel }));
}

// ─── Preview string ──────────────────────────────────────────────────────────

function formatVersion(seg) {
    const n = Number.parseInt(seg.value, 10);
    const num = Number.isFinite(n) ? n : 0;
    const pad = Math.max(0, Number.parseInt(seg.pad, 10) || 0);
    return `${seg.prefix ?? "v"}${String(num).padStart(pad, "0")}`;
}
function segmentText(seg) {
    return seg.kind === "version" ? formatVersion(seg) : String(seg.value ?? "");
}
function previewString(node) {
    return node._batSegs
        .map((s, i) => (i === 0 ? segmentText(s) : (s.separator ?? "/") + segmentText(s)))
        .join("");
}

// ─── Small DOM helpers ───────────────────────────────────────────────────────

const COL = {
    bg: "#1a1a1a", row: "#202020", rowAlt: "#242424", border: "#333",
    field: "#161616", fieldBorder: "#3a3a3a", text: "#ccc", dim: "#888",
    accent: "#7ec0ee", btn: "#2a2a2a", btnHover: "#383838",
};

function el(tag, css, props) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (props) Object.assign(e, props);
    return e;
}

function field(value, { width, placeholder, title } = {}) {
    const i = el("input", `
        background:${COL.field}; color:${COL.text};
        border:1px solid ${COL.fieldBorder}; border-radius:3px;
        font:11px monospace; padding:2px 5px; box-sizing:border-box;
        ${width ? `width:${width};` : "flex:1; min-width:0;"}
    `);
    i.type = "text";
    i.value = value ?? "";
    if (placeholder) i.placeholder = placeholder;
    if (title) i.title = title;
    return i;
}

function iconBtn(glyph, title, onClick) {
    const b = el("button", `
        background:${COL.btn}; color:${COL.text}; border:1px solid ${COL.border};
        border-radius:3px; font:11px monospace; width:22px; height:22px;
        cursor:pointer; padding:0; flex:0 0 auto; line-height:1;
    `, { title });
    b.textContent = glyph;
    b.onmouseenter = () => (b.style.background = COL.btnHover);
    b.onmouseleave = () => (b.style.background = COL.btn);
    b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onClick(); };
    // Keep canvas drag/select from stealing the pointer.
    b.onpointerdown = (e) => e.stopPropagation();
    return b;
}

function combo(value, values, onChange) {
    const s = el("select", `
        background:${COL.field}; color:${COL.text};
        border:1px solid ${COL.fieldBorder}; border-radius:3px;
        font:11px monospace; padding:1px 3px; cursor:pointer;
    `);
    for (const v of values) {
        const o = el("option", "", { value: v, textContent: v });
        if (v === value) o.selected = true;
        s.appendChild(o);
    }
    s.onchange = () => onChange(s.value);
    s.onpointerdown = (e) => e.stopPropagation();
    return s;
}

// ─── Table rendering ─────────────────────────────────────────────────────────

function render(node) {
    relabel(node);
    const root = node._batRoot;
    const body = node._batBody;
    body.innerHTML = "";

    node._batSegs.forEach((seg, i) => {
        const row = el("div", `
            display:flex; flex-direction:column; gap:3px;
            background:${i % 2 ? COL.rowAlt : COL.row};
            border:1px solid ${COL.border}; border-radius:4px;
            padding:5px 6px;
        `);

        // Line 1: separator (not for first) + kind tag + label + reorder/delete.
        const top = el("div", "display:flex; align-items:center; gap:4px;");

        if (i > 0) {
            const sep = field(seg.separator ?? "/",
                { width: "28px", title: "Separator before this segment" });
            sep.style.textAlign = "center";
            sep.oninput = () => { seg.separator = sep.value; commit(node, false); };
            sep.onpointerdown = (e) => e.stopPropagation();
            top.appendChild(sep);
        } else {
            top.appendChild(el("span", `width:28px; flex:0 0 auto; text-align:center;
                color:${COL.dim}; font:10px monospace;`, { textContent: "·" }));
        }

        const tag = el("span", `
            flex:0 0 auto; font:9px monospace; color:${COL.accent};
            border:1px solid ${COL.accent}; border-radius:3px; padding:1px 4px;
            opacity:0.8;`, { textContent: seg.kind === "version" ? "VER" : "NAME" });
        top.appendChild(tag);

        const label = field(seg.label, { placeholder: `segment_${i + 1}`, title: "Label" });
        label.oninput = () => {
            seg.label = label.value;
            seg._userLabel = label.value.trim() !== "" &&
                             label.value !== `segment_${i + 1}`;
            commit(node, false);
        };
        label.onpointerdown = (e) => e.stopPropagation();
        top.appendChild(label);

        top.appendChild(iconBtn("▲", "Move up", () => move(node, i, -1)));
        top.appendChild(iconBtn("▼", "Move down", () => move(node, i, +1)));
        top.appendChild(iconBtn("✕", "Remove segment", () => remove(node, i)));
        row.appendChild(top);

        // Line 2: the value editor.
        const val = el("div", "display:flex; align-items:center; gap:4px; padding-left:32px;");
        if (seg.kind === "version") {
            const prefix = field(seg.prefix ?? "v", { width: "52px", title: "Prefix" });
            prefix.oninput = () => { seg.prefix = prefix.value; commit(node, false); };
            prefix.onpointerdown = (e) => e.stopPropagation();

            const num = field(String(seg.value ?? 0), { width: "64px", title: "Version number" });
            num.oninput = () => {
                const n = Number.parseInt(num.value, 10);
                seg.value = Number.isFinite(n) ? Math.max(0, n) : 0;
                commit(node, false);
            };
            num.onpointerdown = (e) => e.stopPropagation();

            const down = iconBtn("−", "−1", () => { stepVersion(node, seg, -1); });
            const up = iconBtn("+", "+1", () => { stepVersion(node, seg, +1); });

            const padLbl = el("span", `color:${COL.dim}; font:10px monospace;`,
                { textContent: "pad" });
            const pad = field(String(seg.pad ?? 3), { width: "34px", title: "Zero-pad width" });
            pad.oninput = () => {
                const p = Number.parseInt(pad.value, 10);
                seg.pad = Number.isFinite(p) ? Math.max(0, p) : 0;
                commit(node, false);
            };
            pad.onpointerdown = (e) => e.stopPropagation();

            const mode = combo(seg.mode ?? "fixed", VERSION_MODES,
                (v) => { seg.mode = v; commit(node, false); });
            mode.title = "control_after_generate";

            val.append(prefix, down, num, up, padLbl, pad, mode);
        } else {
            const text = field(seg.value, { placeholder: "folder / name", title: "Value" });
            text.oninput = () => { seg.value = text.value; commit(node, false); };
            text.onpointerdown = (e) => e.stopPropagation();
            val.appendChild(text);
        }
        row.appendChild(val);

        body.appendChild(row);
    });

    if (node._batSegs.length === 0) {
        body.appendChild(el("div", `color:${COL.dim}; font:11px monospace;
            text-align:center; padding:10px;`,
            { textContent: "No segments — add a Name or Version below." }));
    }

    node._batPreview.textContent = previewString(node) || "(empty)";
    if (root) root.style.minHeight = "0";
    resize(node);
    // scrollHeight may be 0 until the browser lays the rows out; re-measure on
    // the next frame so the node settles to the real content height.
    requestAnimationFrame(() => resize(node));
}

// ─── Mutations ───────────────────────────────────────────────────────────────

function commit(node, rerender = true) {
    relabel(node);
    serialize(node);
    if (node._batPreview) node._batPreview.textContent = previewString(node) || "(empty)";
    if (rerender) render(node);
    node.setDirtyCanvas(true, true);
}

function move(node, i, delta) {
    const j = i + delta;
    if (j < 0 || j >= node._batSegs.length) return;
    const a = node._batSegs;
    [a[i], a[j]] = [a[j], a[i]];
    commit(node);
}
function remove(node, i) { node._batSegs.splice(i, 1); commit(node); }
function add(node, kind) { node._batSegs.push(defaultSegment(kind)); commit(node); }
function stepVersion(node, seg, delta) {
    const n = Number.parseInt(seg.value, 10);
    seg.value = Math.max(0, (Number.isFinite(n) ? n : 0) + delta);
    commit(node);
}

function applyAfterGenerate(node) {
    let changed = false;
    for (const seg of node._batSegs) {
        if (seg.kind !== "version") continue;
        const n = Number.parseInt(seg.value, 10) || 0;
        if (seg.mode === "increment") { seg.value = n + 1; changed = true; }
        else if (seg.mode === "decrement") { seg.value = Math.max(0, n - 1); changed = true; }
        else if (seg.mode === "randomize") { seg.value = Math.floor(Math.random() * 1e6); changed = true; }
    }
    if (changed) commit(node);
}

// Measured content height of the editor, used both to size the DOM widget's
// reserved slot (so litegraph doesn't leave a giant grey gap) and to grow the
// node. Falls back to a sane estimate before the DOM has laid out.
function contentHeight(node) {
    const h = node._batRoot ? node._batRoot.scrollHeight : 0;
    return h || 120;
}

// Snap the node to fit its content. computeSize() now reflects the DOM widget's
// pinned height (via getMin/MaxHeight), so we snap height to it — shrinking as
// well as growing, so removing segments doesn't leave a grey gap. Width keeps
// the user's wider value but never goes below the minimum.
function resize(node) {
    const min = node.computeSize();
    const w = Math.max(node.size[0] || 0, min[0], 340);
    node.setSize([w, min[1]]);
    node.setDirtyCanvas(true, true);
}

// ─── DOM root ────────────────────────────────────────────────────────────────

function buildRoot(node) {
    const root = el("div", `
        display:flex; flex-direction:column; gap:6px;
        background:${COL.bg}; border:1px solid ${COL.border};
        border-radius:5px; padding:7px; box-sizing:border-box; width:100%;
        font-family:monospace;
    `);

    // Add buttons.
    const bar = el("div", "display:flex; gap:6px;");
    const mkAdd = (text, kind) => {
        const b = el("button", `
            flex:1; background:${COL.btn}; color:${COL.text};
            border:1px solid ${COL.border}; border-radius:4px;
            font:12px monospace; padding:5px; cursor:pointer;`, { textContent: text });
        b.onmouseenter = () => (b.style.background = COL.btnHover);
        b.onmouseleave = () => (b.style.background = COL.btn);
        b.onpointerdown = (e) => e.stopPropagation();
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); add(node, kind); };
        return b;
    };
    bar.append(mkAdd("+ Name", "name"), mkAdd("+ Version", "version"));
    root.appendChild(bar);

    // Segment rows live here.
    const bodyWrap = el("div", "display:flex; flex-direction:column; gap:5px;");
    root.appendChild(bodyWrap);
    node._batBody = bodyWrap;

    // Live preview line.
    const prevWrap = el("div", `
        display:flex; align-items:center; gap:6px; margin-top:2px;
        background:${COL.field}; border:1px solid ${COL.fieldBorder};
        border-radius:4px; padding:4px 7px;`);
    prevWrap.appendChild(el("span", `color:${COL.dim}; font:11px monospace;
        flex:0 0 auto;`, { textContent: "→" }));
    const prev = el("span", `color:${COL.accent}; font:11px monospace;
        word-break:break-all;`, { textContent: "(empty)" });
    prevWrap.appendChild(prev);
    root.appendChild(prevWrap);
    node._batPreview = prev;

    node._batRoot = root;
    return root;
}

// ─── Extension registration ──────────────────────────────────────────────────

app.registerExtension({
    name: "BAT.FilenamePrefix",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        function initNode(node) {
            const segsWidget = node.widgets?.find((w) => w.name === "segments");
            if (segsWidget) {
                segsWidget.type = "hidden";
                segsWidget.computeSize = () => [0, -4];
            }
            node._batSegsWidget = segsWidget;
            loadSegments(node);

            // control_after_generate. ComfyUI runs `widget.afterQueued()` on
            // every widget right after a prompt is queued (the same hook the
            // seed's value-control widget uses). So the queued run captures the
            // value shown, then we advance version segments for the next run.
            if (segsWidget) {
                segsWidget.afterQueued = () => applyAfterGenerate(node);
            }

            const root = buildRoot(node);
            // DOM widgets size via computeLayoutSize, which reads getMinHeight /
            // getMaxHeight from the widget options. Pinning min === max to the
            // measured content height makes the widget reserve EXACTLY the
            // editor's height — otherwise it's treated as "growable" and expands
            // to fill the node, producing the grey overflow rectangle.
            const h = () => contentHeight(node) + 4;
            const dom = node.addDOMWidget("bat_prefix_editor", "bat_prefix_editor", root, {
                serialize: false, hideOnZoom: false,
                getMinHeight: h, getMaxHeight: h, getHeight: h,
            });
            node._batDom = dom;

            node.size = [Math.max(360, node.size[0] || 0), node.size[1] || 0];
            render(node);
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            initNode(this);
            return r;
        };

        // On workflow load the `segments` value is restored after onNodeCreated,
        // so re-read and re-render here — this restores the table before the
        // first run.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (!this._batSegsWidget) {
                this._batSegsWidget = this.widgets?.find((w) => w.name === "segments");
            }
            if (this._batBody) { loadSegments(this); render(this); }
            return r;
        };

    },
});
