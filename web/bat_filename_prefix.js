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
import { unpinWidgetWidth } from "./bat_node_layout.js";

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
    settle(node);
}

// The root is still detached during onNodeCreated, and even once attached the
// first animation frame can land before layout has run — so a single rAF
// re-measure sometimes kept a stale (too tall) height. Re-measure until the
// number stops changing, capped so a hidden/collapsed node can't spin forever.
function settle(node, tries = 5, last = -1) {
    requestAnimationFrame(() => {
        const h = contentHeight(node);
        resize(node);
        if (h !== last && tries > 0) settle(node, tries - 1, h);
    });
}

// Keep the reserved height honest when the editor reflows for reasons we don't
// drive — node resized narrower so a row wraps, font loaded, zoom changed.
function observe(node) {
    if (node._batRO || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => resize(node));
    ro.observe(node._batRoot);
    node._batRO = ro;

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        ro.disconnect();
        node._batRO = null;
        return onRemoved ? onRemoved.apply(this, arguments) : undefined;
    };
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
// node.
//
// scrollHeight is the *content* box — it omits the root's padding and border,
// and it reads 0 before the element is laid out. Both cases used to feed a
// too-tall number into the pinned widget height, which litegraph then painted
// as grey node background below the editor. So: sum the children's border boxes
// plus the flex gaps and the root's own padding/border, which is exact and works
// even while detached, and only fall back to scrollHeight if that yields 0.
function contentHeight(node) {
    const root = node._batRoot;
    if (!root) return 0;

    const cs = getComputedStyle(root);
    const px = (v) => Number.parseFloat(v) || 0;
    const chrome = px(cs.paddingTop) + px(cs.paddingBottom) +
                   px(cs.borderTopWidth) + px(cs.borderBottomWidth);
    const gap = px(cs.rowGap);

    // getBoundingClientRect() reports the box AFTER the canvas zoom transform,
    // in screen pixels, while everything else here — the padding and gaps from
    // getComputedStyle, offsetHeight, scrollHeight, and the node height this
    // feeds — is in unscaled layout units. Mixing the two made the measured
    // height track the zoom level, so the node grew and shrank as the canvas
    // zoomed. Divide the rect back out by the transform the element is actually
    // under, taken from the element itself rather than app.canvas.ds.scale, so
    // it stays right under whatever container the renderer mounts widgets in.
    const rootRect = root.getBoundingClientRect();
    const zoom = (rootRect.width > 0 && root.offsetWidth > 0)
        ? rootRect.width / root.offsetWidth
        : 1;

    let inner = 0, n = 0;
    for (const child of root.children) {
        const rect = child.getBoundingClientRect().height;
        const r = rect ? rect / zoom : (child.offsetHeight || child.scrollHeight);
        if (!r) continue;
        inner += r;
        n += 1;
    }
    if (n > 1) inner += gap * (n - 1);

    const measured = inner ? inner + chrome : root.scrollHeight + chrome;
    return Math.ceil(measured);
}

// Snap the node to fit its content. computeSize() now reflects the DOM widget's
// pinned height (via getMin/MaxHeight), so we snap height to it — shrinking as
// well as growing, so removing segments doesn't leave a grey gap. Width keeps
// the user's wider value but never goes below the minimum.
function resize(node) {
    // computeSize() reads the DOM widget's pinned height via getMinHeight /
    // getMaxHeight, but some frontend versions cache the widget's computed
    // layout. Clear that cache first or a height measured while the editor was
    // still taller (or not yet laid out) sticks, and the leftover space paints
    // as grey node background under the editor.
    const dom = node._batDom;
    if (dom) {
        dom.computedHeight = undefined;
        if (dom.options) dom.options.minHeight = undefined;
    }

    const min = node.computeSize();
    const w = Math.max(node.size[0] || 0, min[0], 340);
    // The ResizeObserver that calls this watches the element this resize moves,
    // so a no-op write is a redraw for nothing — and, if a measurement ever
    // wobbles by a pixel, a loop. Only write when the size actually changes.
    if (node.size[0] === w && node.size[1] === min[1]) return;
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
                // Hiding a widget takes all three legs — see the same trio in
                // bat_animated_grade.js / bat_video_combine.js.
                //
                // `type = "hidden"` is what the Nodes 2.0 DOMWidget keys its
                // zero-size computeLayoutSize() on, and computeSize [0,-4] is
                // the Nodes 1.0 leg that stops it reserving a row. Neither one
                // hides the ELEMENT: DomWidgets.vue mounts/shows a DOM widget on
                // `widget.isVisible()`, which reads `!widget.hidden` and never
                // looks at the type. `segments` was declared multiline, so the
                // frontend built it as a real <textarea> DOM widget — and with
                // `hidden` unset that textarea stayed live: laid out at
                // widget.y (the top of the node, behind our editor, since it is
                // registered first), width (widget.width ?? node.width) - 20,
                // and height computedHeight - 20 = -20. A negative height is an
                // invalid inline style, so the element fell back to its
                // `size-full` class — height:100% of the canvas layer. Result:
                // a borderless grey strip (var(--comfy-input-bg)) hidden behind
                // the editor and visible as a long box below the node.
                //
                // `segments` is also no longer multiline on the Python side, so
                // there is no textarea to strand any more; this stays correct
                // for whichever widget flavour the frontend hands us.
                segsWidget.type = "hidden";
                segsWidget.hidden = true;
                segsWidget.computeSize = () => [0, -4];
                segsWidget.draw = () => {};
                if (segsWidget.element) segsWidget.element.style.display = "none";
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
            const h = () => contentHeight(node);
            const dom = node.addDOMWidget("bat_prefix_editor", "bat_prefix_editor", root, {
                serialize: false, hideOnZoom: false,
                getMinHeight: h, getMaxHeight: h, getHeight: h,
            });
            // Never let a stamped pixel width govern the editor's slot — the
            // parameters panel brands unrecognised widget types with its own
            // width via WidgetLegacy. See unpinWidgetWidth() for the full story.
            unpinWidgetWidth(dom);
            node._batDom = dom;

            // Only seed the width; the height comes from the measured editor, so
            // seeding it here would just be another number to grow out of.
            node.size[0] = Math.max(360, node.size[0] || 0);
            render(node);
            observe(node);
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
