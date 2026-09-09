/**
 * Bat Bypass Switch — named toggles that bypass saved selections.
 *
 * The node
 * --------
 * A template author drops one on the canvas. It starts empty. They press
 * "＋ Add toggle…", which puts the canvas into *selection mode*: a banner
 * appears, they click/rubber-band whatever nodes, subgraph nodes and backdrops
 * belong together, press Enter, and name it. That becomes a toggle widget on
 * the node body. Clicking it walks the saved selection and sets every node to
 * mode 0 (Always) or mode 4 (Bypass).
 *
 * Nothing here is wired into the graph — the node has no inputs, no outputs
 * and never executes (see bat_bypass_switch.py). It edits the graph, it is not
 * part of it.
 *
 * Five decisions worth knowing about
 * ----------------------------------
 * 1. **On load the node rebuilds its UI but does NOT touch node modes.** The
 *    bypass state of every target is already saved in the workflow JSON, so
 *    re-asserting it on open would be redundant at best. At worst it silently
 *    undoes hand-bypassing someone did on purpose after the toggles were set.
 *    "Re-apply all toggles" is in the node's right-click menu for when you do
 *    want the switch to be authoritative.
 *
 * 2. **Backdrops are remembered as backdrops, not as their contents.** A
 *    preset stores the *group id*, and the nodes it controls are whatever sits
 *    inside that group at the moment you click the toggle
 *    (`recomputeInsideNodes()`). Drag a new node into the backdrop a week
 *    later and the toggle covers it, with no re-capture — which is the whole
 *    reason to use a backdrop in a template.
 *
 * 3. **Node targets are plain ids, and missing ones are reported, not
 *    repaired.** A toggle whose targets were deleted draws a ⚠ and the edit
 *    dialog says how many are gone, with a one-click clean-up. Ids are *not*
 *    remapped on paste, which is a real footgun: copy the switch together with
 *    its targets and the copy still points at the originals. That case is at
 *    least detected — two switches in one graph sharing preset ids can only be
 *    a copy — and the newer one gets a ⚠ and a warning telling you to
 *    re-select. (Stamping every target with a stable uid so paste could be
 *    followed is the alternative; it was scoped out.)
 *
 * 4. **Toggles are independent, unless you put them in an exclusive group.**
 *    Independent ones draw as real toggle widgets. An exclusive group draws as
 *    a single combo — "Mode: Video" — whose options are its member toggles
 *    plus "(none)". That is a better fit for canvas widgets than faking radio
 *    buttons out of toggles, and it makes the mutual exclusion visually
 *    obvious instead of something you discover by clicking.
 *
 * 5. **ON forces mode 0, unconditionally.** Not "restore whatever the mode was
 *    before", and not "leave muted nodes muted". A toggle that sometimes
 *    leaves a node switched off would be untrustworthy, and the point of this
 *    node is to be trustworthy in a template someone else opens.
 *
 * Canvas widgets, not a DOM panel: the toggle rows are litegraph widgets, so
 * they stay crisp at any zoom, need none of the Nodes-2.0 sizing contract in
 * bat_node_layout.js, and survive inside subgraphs. Renaming / reordering /
 * grouping happens in a modal instead of inline, which is the trade.
 */

import { app } from "/scripts/app.js";

const NODE_TYPE = "Bat_BypassSwitch";
const LOG = "[Bat_BypassSwitch]";

/** The one serialising widget: JSON state, declared in Python, hidden here. */
const STATE_WIDGET = "presets";

/** Marks every widget we build, so a rebuild can clear exactly ours. */
const DYN = "_batBswDynamic";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

/** Combo entry meaning "nothing in this exclusive group is on". */
const NONE_LABEL = "(none)";

const MIN_NODE_W = 260;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/**
 * Shape (v1):
 *   {
 *     v: 1,
 *     exgroups: [ { id, name } ],
 *     presets:  [ { id, name, on, ex, nodes: [nodeId], groups: [groupId] } ]
 *   }
 *
 * `presets` is an ordered list and that order is the draw order. `ex` is ""
 * for an independent toggle, or an exgroup id. An exclusive group draws at the
 * position of its first member.
 */
function emptyState() {
    return { v: 1, exgroups: [], presets: [] };
}

let _uidCounter = 0;
function uid(prefix) {
    _uidCounter += 1;
    return `${prefix}_${Date.now().toString(36)}${_uidCounter.toString(36)}` +
           Math.random().toString(36).slice(2, 6);
}

/**
 * Coerce anything into a valid state object.
 *
 * This is the only validator in the system, and it has to be total: the JSON
 * lives in a workflow file that people copy between machines, hand-edit, and
 * load into older or newer versions of the pack. A parse error or a bad field
 * must degrade to "this toggle is gone", never to a node that throws during
 * configure() and takes the whole graph load down with it.
 */
function normaliseState(raw) {
    const out = emptyState();
    if (!raw || typeof raw !== "object") return out;

    const seenGroupIds = new Set();
    for (const g of Array.isArray(raw.exgroups) ? raw.exgroups : []) {
        if (!g || typeof g !== "object") continue;
        const id = typeof g.id === "string" && g.id ? g.id : uid("g");
        if (seenGroupIds.has(id)) continue;
        seenGroupIds.add(id);
        out.exgroups.push({ id, name: String(g.name ?? "Group") });
    }

    const seenPresetIds = new Set();
    for (const p of Array.isArray(raw.presets) ? raw.presets : []) {
        if (!p || typeof p !== "object") continue;
        const id = typeof p.id === "string" && p.id ? p.id : uid("p");
        if (seenPresetIds.has(id)) continue;
        seenPresetIds.add(id);
        // An `ex` pointing at an exgroup that is not in the file would make a
        // toggle undrawable (it would wait forever for a group header that
        // never comes), so an unknown reference degrades to independent.
        const ex = typeof p.ex === "string" && seenGroupIds.has(p.ex) ? p.ex : "";
        out.presets.push({
            id,
            name: String(p.name ?? "Toggle"),
            on: p.on === true || p.on === "true",
            ex,
            nodes: (Array.isArray(p.nodes) ? p.nodes : []).filter(
                (n) => typeof n === "number" || typeof n === "string"),
            groups: (Array.isArray(p.groups) ? p.groups : []).filter(
                (n) => typeof n === "number" || typeof n === "string"),
        });
    }

    // Drop exgroups nothing points at any more. They are pure UI grouping, so
    // an empty one is dead weight that would draw an option-less combo.
    const used = new Set(out.presets.map((p) => p.ex).filter(Boolean));
    out.exgroups = out.exgroups.filter((g) => used.has(g.id));

    // An exclusive group with more than one member ON is a state we should
    // never write, but a hand-edited file can contain it. First one wins.
    for (const g of out.exgroups) {
        let taken = false;
        for (const p of out.presets) {
            if (p.ex !== g.id || !p.on) continue;
            if (taken) p.on = false;
            taken = true;
        }
    }
    return out;
}

function stateWidget(node) {
    return node.widgets?.find((w) => w.name === STATE_WIDGET);
}

function readState(node) {
    const w = stateWidget(node);
    if (!w) return emptyState();
    let raw = null;
    try {
        raw = typeof w.value === "string" ? JSON.parse(w.value) : w.value;
    } catch (e) {
        console.warn(LOG, "unreadable preset JSON, starting empty:", e);
        return emptyState();
    }
    return normaliseState(raw);
}

function writeState(node, state) {
    const w = stateWidget(node);
    if (!w) return;
    // Through the setter, not the backing field: the frontend keeps widget
    // values in a store keyed by (graph, node, widget name), and a direct
    // write to some internal slot would leave the store — and therefore
    // widgets_values, and therefore the saved workflow — holding the old JSON.
    w.value = JSON.stringify(state);
}

// ---------------------------------------------------------------------------
// Resolving a preset to live nodes
// ---------------------------------------------------------------------------

/** Is this item a graph node (as opposed to a group, a reroute, …)? */
function isNodeItem(it) {
    return !!it && typeof it === "object" && "mode" in it && "id" in it;
}

/** Is this item a group / backdrop? */
function isGroupItem(it) {
    return !!it && typeof it === "object" &&
           typeof it.recomputeInsideNodes === "function";
}

function findGroup(graph, gid) {
    const groups = graph?.groups || [];
    // Loose compare: group ids are numeric in the runtime but arrive from JSON
    // as whatever the file had.
    return groups.find((g) => String(g.id) === String(gid));
}

/**
 * Nodes a preset currently controls.
 *
 * Everything is resolved against `node.graph` — the graph the switch itself
 * lives in. That is what makes a switch inside a subgraph work: node ids are
 * scoped per-graph, and a subgraph's contents are invisible to the root
 * graph's getNodeById(), so asking the root would find nothing.
 *
 * Other Bypass Switches (including this one) are filtered out. They never
 * execute, so bypassing one does nothing except look broken.
 */
function resolveTargets(node, preset) {
    const graph = node.graph;
    const found = new Map();
    let missingNodes = 0;
    let missingGroups = 0;

    const keep = (t) => {
        if (!isNodeItem(t)) return;
        if (t === node || t.type === NODE_TYPE) return;
        found.set(t.id, t);
    };

    for (const id of preset.nodes || []) {
        const t = graph?.getNodeById?.(id);
        if (!t) { missingNodes += 1; continue; }
        keep(t);
    }

    for (const gid of preset.groups || []) {
        const g = findGroup(graph, gid);
        if (!g) { missingGroups += 1; continue; }
        // Live membership. _nodes is filled by centre-containment and already
        // includes the contents of nested groups, so no recursion needed here.
        try { g.recomputeInsideNodes(); } catch (e) {
            console.warn(LOG, "group recompute failed:", e);
        }
        for (const t of g._nodes || g.nodes || []) keep(t);
    }

    return {
        nodes: [...found.values()],
        missingNodes,
        missingGroups,
        missing: missingNodes + missingGroups,
    };
}

/** Force a preset's targets to Always (on) or Bypass (off). Returns a summary. */
function applyPreset(node, preset, on) {
    const info = resolveTargets(node, preset);
    const mode = on ? MODE_ALWAYS : MODE_BYPASS;
    let changed = 0;
    for (const t of info.nodes) {
        if (t.mode === mode) continue;
        t.mode = mode;
        changed += 1;
    }
    return { ...info, changed };
}

/** Run a graph mutation as one undo step. */
function asOneUndoStep(node, fn) {
    const canvas = app.canvas;
    try { canvas?.emitBeforeChange?.(); } catch (e) {}
    try { node.graph?.beforeChange?.(); } catch (e) {}
    try {
        return fn();
    } finally {
        try { node.graph?.afterChange?.(); } catch (e) {}
        try { canvas?.emitAfterChange?.(); } catch (e) {}
        node.graph?.setDirtyCanvas?.(true, true);
    }
}

/**
 * Flip one preset.
 *
 * Exclusive-group siblings are switched OFF *before* this one is switched on,
 * deliberately: two toggles in a group can legitimately share a target node
 * (a common loader feeding both an image and a video branch), and applying in
 * this order leaves a shared node enabled rather than bypassed by the loser.
 */
function setPresetOn(node, presetId, on) {
    asOneUndoStep(node, () => {
        const state = readState(node);
        const p = state.presets.find((x) => x.id === presetId);
        if (!p) return;

        if (on && p.ex) {
            for (const sib of state.presets) {
                if (sib.id === p.id || sib.ex !== p.ex || !sib.on) continue;
                sib.on = false;
                applyPreset(node, sib, false);
            }
        }
        p.on = !!on;
        const res = applyPreset(node, p, p.on);
        writeState(node, state);
        if (res.missing) {
            console.warn(LOG, `"${p.name}": ${res.missing} target(s) no longer`,
                         "exist and were skipped");
        }
    });
    refreshWidgets(node);
}

/** Pick the active member of an exclusive group (or null for "(none)"). */
function setExGroupActive(node, exId, presetId) {
    asOneUndoStep(node, () => {
        const state = readState(node);
        // Off first, on second — same shared-target reasoning as setPresetOn.
        for (const p of state.presets) {
            if (p.ex !== exId || p.id === presetId || !p.on) continue;
            p.on = false;
            applyPreset(node, p, false);
        }
        const chosen = state.presets.find((p) => p.id === presetId);
        if (chosen) {
            chosen.on = true;
            applyPreset(node, chosen, true);
        }
        writeState(node, state);
    });
    refreshWidgets(node);
}

/** Re-assert every toggle onto the graph. The menu's "Re-apply" action. */
function reapplyAll(node) {
    let touched = 0;
    let missing = 0;
    asOneUndoStep(node, () => {
        const state = readState(node);
        // Off before on, globally: overlapping presets then resolve in favour
        // of "some toggle wants this node enabled", which is the intuitive
        // reading of two switches disagreeing.
        for (const p of state.presets) {
            if (p.on) continue;
            const r = applyPreset(node, p, false);
            touched += r.changed; missing += r.missing;
        }
        for (const p of state.presets) {
            if (!p.on) continue;
            const r = applyPreset(node, p, true);
            touched += r.changed; missing += r.missing;
        }
    });
    refreshWidgets(node);
    toast(`Re-applied ${readState(node).presets.length} toggle(s)` +
          (touched ? ` — ${touched} node(s) changed` : " — nothing to change") +
          (missing ? `, ${missing} missing target(s) skipped` : ""),
          missing ? "warn" : "info");
}

// ---------------------------------------------------------------------------
// Small shared bits
// ---------------------------------------------------------------------------

/**
 * Is this node still in its graph?
 *
 * Cheaper than importing the pack's lifecycle tracker (which needs per-node
 * registration) and enough for what we use it for: guarding the deferred
 * label refresh so a node deleted mid-timeout does not get poked.
 */
function nodeIsLive(node) {
    return !!node?.graph && node.graph.getNodeById?.(node.id) === node;
}

function toast(text, severity) {
    const map = { info: "info", warn: "warn", error: "error" };
    try {
        app.extensionManager?.toast?.add({
            severity: map[severity] || "info",
            summary: "Bypass Switch",
            detail: text,
            life: severity === "error" ? 8000 : 4000,
        });
        return;
    } catch (e) { /* fall through */ }
    console.info(LOG, text);
}

/** Everything we inject is prefixed bat-bsw- : the packs share one CSS namespace. */
const CSS_ID = "bat-bsw-style";
const CSS = `
.bat-bsw-banner {
    position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
    z-index: 10000; pointer-events: auto;
    display: flex; flex-direction: column; gap: 6px;
    min-width: 380px; max-width: 620px;
    padding: 12px 16px;
    font: 13px/1.45 var(--comfy-font, system-ui, sans-serif);
    color: var(--input-text, #e6e6e6);
    background: var(--comfy-menu-bg, #232323);
    border: 1px solid #7b5cff;
    border-radius: 8px;
    box-shadow: 0 8px 28px rgba(0,0,0,.55);
}
.bat-bsw-banner-title { font-weight: 600; }
.bat-bsw-banner-title .bat-bsw-for { color: #b9a6ff; }
.bat-bsw-banner-hint { opacity: .78; }
.bat-bsw-banner-count { font-variant-numeric: tabular-nums; }
.bat-bsw-banner-count.bat-bsw-empty { color: #e0a44a; }
.bat-bsw-banner-row { display: flex; gap: 8px; align-items: center;
    justify-content: space-between; }
.bat-bsw-kbd {
    display: inline-block; padding: 1px 5px; margin: 0 1px;
    font-size: 11px; font-family: inherit;
    border: 1px solid var(--border-color, #555); border-radius: 3px;
    background: rgba(255,255,255,.07);
}
.bat-bsw-overlay {
    position: fixed; inset: 0; z-index: 10001;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,.55);
    font: 13px/1.45 var(--comfy-font, system-ui, sans-serif);
    color: var(--input-text, #e6e6e6);
}
.bat-bsw-panel {
    display: flex; flex-direction: column; gap: 12px;
    width: min(760px, 92vw); max-height: 84vh;
    padding: 18px 20px;
    background: var(--comfy-menu-bg, #232323);
    border: 1px solid var(--border-color, #555);
    border-radius: 10px;
    box-shadow: 0 16px 48px rgba(0,0,0,.6);
}
.bat-bsw-panel h3 { margin: 0; font-size: 15px; font-weight: 600; }
.bat-bsw-panel h4 { margin: 10px 0 0; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .04em; opacity: .6; }
.bat-bsw-sub { margin: -6px 0 0; opacity: .7; }
.bat-bsw-rows { display: flex; flex-direction: column; gap: 6px;
    overflow-y: auto; min-height: 40px; }
.bat-bsw-row {
    display: grid; gap: 8px; align-items: center;
    grid-template-columns: auto 1fr auto auto auto;
    padding: 7px 8px;
    background: rgba(255,255,255,.04);
    border: 1px solid var(--border-color, #4a4a4a);
    border-radius: 6px;
}
.bat-bsw-arrows { display: flex; flex-direction: column; gap: 2px; }
.bat-bsw-meta { font-size: 11px; opacity: .68; white-space: nowrap;
    font-variant-numeric: tabular-nums; }
.bat-bsw-meta .bat-bsw-missing { color: #e0a44a; opacity: 1; }
.bat-bsw-actions { display: flex; gap: 8px; justify-content: flex-end;
    align-items: center; }
.bat-bsw-actions .bat-bsw-spacer { flex: 1; }
.bat-bsw-panel input[type="text"], .bat-bsw-panel select {
    box-sizing: border-box; width: 100%; padding: 5px 7px;
    font: inherit; color: var(--input-text, #e6e6e6);
    background: var(--comfy-input-bg, #1a1a1a);
    border: 1px solid var(--border-color, #555); border-radius: 4px;
}
.bat-bsw-panel select { width: auto; min-width: 130px; }
.bat-bsw-panel button {
    padding: 5px 11px; font: inherit; cursor: pointer;
    color: var(--input-text, #e6e6e6);
    background: var(--comfy-input-bg, #333);
    border: 1px solid var(--border-color, #555); border-radius: 4px;
}
.bat-bsw-panel button:hover:not(:disabled) { border-color: #7b5cff; }
.bat-bsw-panel button:disabled { opacity: .4; cursor: default; }
.bat-bsw-panel button.bat-bsw-primary { background: #4a3aa8; border-color: #7b5cff; }
.bat-bsw-panel button.bat-bsw-tiny { padding: 0 6px; line-height: 1.3; font-size: 10px; }
.bat-bsw-panel button.bat-bsw-danger:hover { border-color: #c05a5a; color: #ffb3b3; }
.bat-bsw-warn { color: #e0a44a; }
.bat-bsw-empty-note { padding: 14px; text-align: center; opacity: .6; }
`;

function ensureStyles() {
    if (document.getElementById(CSS_ID)) return;
    const el = document.createElement("style");
    el.id = CSS_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
}

function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
}

function button(text, cls, onClick) {
    const b = el("button", cls, text);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
}

// ---------------------------------------------------------------------------
// Widgets
// ---------------------------------------------------------------------------

function relayout(node) {
    try {
        const computed = node.computeSize?.();
        if (computed) {
            const w = Math.max(node.size?.[0] || 0, computed[0], MIN_NODE_W);
            // Height is taken from computeSize rather than clamped upwards, so
            // deleting a toggle actually shrinks the node instead of leaving a
            // dead band where its row used to be.
            const h = Math.max(computed[1] || 0, 0);
            if (typeof node.setSize === "function") node.setSize([w, h]);
            else node.size = [w, h];
        } else {
            node.expandToFitContent?.();
        }
    } catch (e) {
        console.warn(LOG, "relayout failed:", e);
    }
    node.setDirtyCanvas?.(true, true);
}

/** Hide the JSON state widget: three legs, one per renderer generation. */
function hideStateWidget(node) {
    const w = stateWidget(node);
    if (!w || w.type === "hidden") return;
    w.type = "hidden";          // Nodes 2.0 collapses on the type
    w.hidden = true;            // current litegraph filters layout on this
    w.computeSize = () => [0, -4];  // older builds size by height
}

function removeDynamicWidgets(node) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (!w || !w[DYN]) continue;
        if (typeof node.removeWidget === "function") {
            try { node.removeWidget(w); }
            catch (e) { node.widgets.splice(i, 1); }
        } else {
            node.widgets.splice(i, 1);
        }
        try { w.onRemove?.(); } catch (e) {}
    }
    node._widgetSlotsDirty = true;
}

/**
 * Append one widget from a plain spec.
 *
 * Two non-obvious requirements, both learned the hard way elsewhere in this
 * pack:
 *
 * • It must go through `addCustomWidget`, never `node.widgets.push`.
 *   addCustomWidget wraps the spec in a concrete widget class, marks the slot
 *   layout dirty and binds the node id; a bare push skips all three and the
 *   widget silently never renders on 1.4x frontends.
 *
 * • `value` must be re-assigned *after* the add. The frontend's widget-value
 *   store is keyed by (graph, node, widget name) and outlives the widget
 *   itself — removeWidget splices the widget off but never unregisters its
 *   state — so a rebuilt widget with the same name and type adopts the
 *   *previous* value and throws away the one in the spec. Ctrl+Z, which is a
 *   full graph reload, hits this every time.
 */
function addWidget(node, spec) {
    const intended = spec.value;
    const plain = { ...spec, [DYN]: true, serialize: false, node };
    const w = (typeof node.addCustomWidget === "function")
        ? node.addCustomWidget(plain)
        : (node.widgets.push(plain), plain);
    w[DYN] = true;
    // Top-level, not options.serialize: litegraph's serialiser tests
    // `widget.serialize !== false`, and it writes widgets_values at each
    // widget's *full array index*, so a serialising dynamic widget would also
    // punch holes into the array around the state widget at index 0.
    w.serialize = false;
    if (w.value !== intended) w.value = intended;
    return w;
}

/** " ⚠" suffix when a preset's targets are missing, or the node is a copy. */
function presetLabel(node, preset, withWarning) {
    if (!withWarning) return preset.name;
    if (node._bswDuplicate) return `${preset.name}  ⚠`;
    const info = resolveTargets(node, preset);
    return info.missing ? `${preset.name}  ⚠` : preset.name;
}

/**
 * Option labels for an exclusive group's combo, and the label→preset map.
 *
 * Labels have to be unique — a combo keys its selection off the string — so
 * duplicates (and anything colliding with the "(none)" entry) get a counter
 * suffix. The map is what the callback reads, so a suffixed label still
 * resolves to the right preset.
 */
function exComboOptions(node, members, withWarning) {
    const values = [NONE_LABEL];
    const map = new Map();
    const labelFor = new Map();
    for (const p of members) {
        let base = presetLabel(node, p, withWarning) || "Toggle";
        let label = base;
        let n = 2;
        while (label === NONE_LABEL || map.has(label)) label = `${base} (${n++})`;
        values.push(label);
        map.set(label, p.id);
        labelFor.set(p.id, label);
    }
    return { values, map, labelFor };
}

function addPresetToggle(node, preset, withWarning) {
    const w = addWidget(node, {
        name: `bsw_${preset.id}`,
        label: presetLabel(node, preset, withWarning),
        type: "toggle",
        value: !!preset.on,
        options: { on: "on", off: "bypassed" },
        callback(v) {
            if (node._bswSyncing) return;
            setPresetOn(node, preset.id, !!v);
        },
    });
    w._bswPresetId = preset.id;
    return w;
}

function addExGroupCombo(node, state, exId, withWarning) {
    const g = state.exgroups.find((x) => x.id === exId);
    const members = state.presets.filter((p) => p.ex === exId);
    const { values, map, labelFor } = exComboOptions(node, members, withWarning);
    const active = members.find((p) => p.on);
    const w = addWidget(node, {
        name: `bswex_${exId}`,
        label: g?.name || "Group",
        type: "combo",
        value: active ? labelFor.get(active.id) : NONE_LABEL,
        options: { values },
        callback(v) {
            if (node._bswSyncing) return;
            setExGroupActive(node, exId, map.get(v) ?? null);
        },
    });
    w._bswExId = exId;
    w._bswMap = map;
    w._bswLabelFor = labelFor;
    return w;
}

function addActionButtons(node, state) {
    const first = state.presets.length === 0;
    addWidget(node, {
        name: "bsw_add",
        label: first ? "＋ Add first toggle…" : "＋ Add toggle…",
        type: "button",
        value: null,
        callback() { enterSelectionMode(node, { mode: "create" }); },
    });
    if (!first) {
        addWidget(node, {
            name: "bsw_edit",
            label: "✎ Edit toggles…",
            type: "button",
            value: null,
            callback() { openEditDialog(node); },
        });
    }
}

/**
 * Rebuild every dynamic widget from state.
 *
 * `withWarning` is false during a graph load: configure() runs for every node
 * in one synchronous pass, so a switch configured early would resolve its
 * targets against a half-built graph, find nothing, and paint a ⚠ on every
 * toggle. The deferred pass in scheduleWarningRefresh() puts the markers back
 * once the graph is whole.
 */
function buildWidgets(node, withWarning = true) {
    hideStateWidget(node);
    removeDynamicWidgets(node);
    const state = readState(node);
    const seenEx = new Set();
    node._bswSyncing = true;
    try {
        for (const p of state.presets) {
            if (!p.ex) { addPresetToggle(node, p, withWarning); continue; }
            if (seenEx.has(p.ex)) continue;   // group draws at its first member
            seenEx.add(p.ex);
            addExGroupCombo(node, state, p.ex, withWarning);
        }
        addActionButtons(node, state);
    } finally {
        node._bswSyncing = false;
    }
    relayout(node);
}

/**
 * Push state into the existing widgets without rebuilding them.
 *
 * Used after a toggle click, which changes values but never the widget set.
 * Rebuilding from inside a widget callback would splice the array litegraph is
 * mid-way through handling; this avoids the question entirely.
 */
function refreshWidgets(node) {
    if (!node.widgets) return;
    const state = readState(node);
    node._bswSyncing = true;
    try {
        for (const w of node.widgets) {
            if (w._bswPresetId) {
                const p = state.presets.find((x) => x.id === w._bswPresetId);
                if (!p) continue;
                w.label = presetLabel(node, p, true);
                if (w.value !== !!p.on) w.value = !!p.on;
            } else if (w._bswExId) {
                const members = state.presets.filter((x) => x.ex === w._bswExId);
                const active = members.find((x) => x.on);
                const want = active ? (w._bswLabelFor?.get(active.id) ?? NONE_LABEL)
                                    : NONE_LABEL;
                if (w.value !== want) w.value = want;
            }
        }
    } finally {
        node._bswSyncing = false;
    }
    node.setDirtyCanvas?.(true, true);
}

/**
 * Find another switch in this graph that shares preset ids with this one.
 *
 * Preset ids are random per creation, so two switches holding the same id can
 * only be one switch copied — and a copied switch still points at the
 * *original* nodes, quietly controlling a part of the graph nobody expects it
 * to. There is no way to tell which of the two is the copy from the JSON, so
 * the tie is broken on node id: litegraph hands out ids monotonically and a
 * paste always gets fresh ones, so the higher id is the newer node.
 *
 * Detection only — nothing is repaired. Re-selecting the copy's toggles is a
 * decision about what the copy is *for*, and it is not ours to guess.
 */
function duplicateSwitch(node) {
    const graph = node.graph;
    if (!graph) return null;
    const mine = new Set(readState(node).presets.map((p) => p.id));
    if (!mine.size) return null;
    for (const other of graph.nodes || graph._nodes || []) {
        if (!other || other === node || other.type !== NODE_TYPE) continue;
        if (!(Number(node.id) > Number(other.id))) continue;
        const theirs = readState(other).presets;
        if (theirs.some((p) => mine.has(p.id))) return other;
    }
    return null;
}

/**
 * Repaint the ⚠ markers once the graph has finished loading.
 *
 * A single macrotask is enough: ComfyUI configures every node in one
 * synchronous pass, so anything queued here runs after the last one.
 */
function scheduleWarningRefresh(node) {
    setTimeout(() => {
        if (!nodeIsLive(node)) return;
        const twin = duplicateSwitch(node);
        if (twin) {
            node._bswDuplicate = true;
            toast(`This switch shares its toggles with node #${twin.id}, so it is `
                  + `a copy — and it still bypasses that node's targets, not `
                  + `copies of them. Re-select its toggles (right-click → Edit `
                  + `toggles… → Re-select…).`, "warn");
        }
        refreshWidgets(node);
        // Combo option labels carry the ⚠ too, and those live in the widget's
        // options array rather than in `label`, so they need the full rebuild.
        const hasEx = readState(node).presets.some((p) => p.ex);
        if (hasEx) buildWidgets(node, true);
    }, 0);
}

// ---------------------------------------------------------------------------
// Selection mode
// ---------------------------------------------------------------------------

/**
 * One selection session at a time, tracked module-wide.
 *
 * Module-level rather than per-node on purpose: the banner, the key handler
 * and the canvas selection are all singletons, so two concurrent sessions
 * would fight over them. A second request is refused rather than queued.
 */
let _session = null;

/**
 * Split the canvas selection into things we can bypass.
 *
 * Reroutes and anything else positionable are dropped silently — there is no
 * meaningful "bypass" for them. Bypass Switches are dropped too: they never
 * execute, so bypassing one changes nothing while looking like it should.
 */
function partitionSelection(node, items) {
    const nodes = [];
    const groups = [];
    let skippedSwitches = 0;
    for (const it of items || []) {
        if (isGroupItem(it)) { groups.push(it); continue; }
        if (!isNodeItem(it)) continue;
        if (it === node || it.type === NODE_TYPE) { skippedSwitches += 1; continue; }
        nodes.push(it);
    }
    return { nodes, groups, skippedSwitches };
}

/** How many distinct nodes a captured selection actually covers. */
function effectiveNodeCount(node, nodes, groups) {
    return resolveTargets(node, {
        nodes: nodes.map((n) => n.id),
        groups: groups.map((g) => g.id),
    }).nodes.length;
}

function buildBanner(session) {
    const b = el("div", "bat-bsw-banner");
    const title = el("div", "bat-bsw-banner-title");
    title.append(document.createTextNode("🦇 Selection mode — "));
    const forName = el("span", "bat-bsw-for",
        session.mode === "recapture"
            ? `re-selecting “${session.presetName}”`
            : "picking nodes for a new toggle");
    title.append(forName);
    b.append(title);

    const hint = el("div", "bat-bsw-banner-hint");
    hint.innerHTML =
        "Click nodes, subgraphs and backdrops. Shift-click or drag a box to add " +
        "more. A backdrop is remembered as a backdrop, so anything you put in " +
        "it later is covered too.";
    b.append(hint);

    const count = el("div", "bat-bsw-banner-count");
    b.append(count);

    const row = el("div", "bat-bsw-banner-row");
    const keys = el("div");
    keys.innerHTML =
        '<span class="bat-bsw-kbd">Enter</span> confirm &nbsp;·&nbsp; ' +
        '<span class="bat-bsw-kbd">Esc</span> cancel';
    row.append(keys);
    const btns = el("div", "bat-bsw-actions");
    btns.append(button("Cancel", "", () => cancelSelection()));
    const ok = button("Confirm", "bat-bsw-primary", () => confirmSelection());
    btns.append(ok);
    row.append(btns);
    b.append(row);

    session.countEl = count;
    session.confirmBtn = ok;
    document.body.appendChild(b);
    return b;
}

function updateBannerCount() {
    const s = _session;
    if (!s || !s.countEl) return;
    const items = [...(app.canvas?.selectedItems || [])];
    const { nodes, groups } = partitionSelection(s.node, items);
    const total = effectiveNodeCount(s.node, nodes, groups);
    const parts = [];
    parts.push(`${nodes.length} node${nodes.length === 1 ? "" : "s"}`);
    if (groups.length) {
        parts.push(`${groups.length} backdrop${groups.length === 1 ? "" : "s"}`);
    }
    const empty = total === 0;
    s.countEl.textContent = empty
        ? "Nothing selected yet."
        : `Selected: ${parts.join(" · ")}  →  ${total} node${total === 1 ? "" : "s"} controlled`;
    s.countEl.classList.toggle("bat-bsw-empty", empty);
    if (s.confirmBtn) s.confirmBtn.disabled = empty;
}

/**
 * Enter / Escape, in capture phase on window.
 *
 * Capture phase and stopImmediatePropagation, because the core keybinding
 * handler listens on the same events and we must be sure the graph does not
 * also act on the keypress. The listener only exists while a session is
 * active, so normal Enter/Escape behaviour is untouched the rest of the time.
 *
 * The typing guard matters even though the banner has no text inputs: a
 * property-editor prompt or the search box can be open over the canvas, and
 * swallowing Enter there would be maddening.
 */
function onSelectionKey(e) {
    if (!_session) return;
    const t = e.target;
    const tag = t?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
        t?.isContentEditable) {
        return;
    }
    if (e.key === "Enter" || e.key === "NumpadEnter") {
        e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
        confirmSelection();
    } else if (e.key === "Escape") {
        e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
        cancelSelection();
    }
}

function endSession(restorePrevious) {
    const s = _session;
    if (!s) return null;
    _session = null;

    try { s.banner?.remove(); } catch (e) {}
    window.removeEventListener("keydown", onSelectionKey, true);
    if (s.poll) clearInterval(s.poll);
    // Put the canvas hook back exactly as we found it, including the case
    // where there wasn't one.
    const canvas = app.canvas;
    if (canvas) {
        if (s.prevOnSelectionChange === undefined) delete canvas.onSelectionChange;
        else canvas.onSelectionChange = s.prevOnSelectionChange;
    }
    if (restorePrevious && canvas) {
        try {
            canvas.deselectAll?.();
            const alive = s.prevSelection.filter(
                (it) => isGroupItem(it) || nodeIsLive(it));
            if (alive.length) canvas.selectItems?.(alive, true);
        } catch (e) { /* a restore failure is cosmetic */ }
    }
    app.canvas?.setDirty?.(true, true);
    return s;
}

function cancelSelection() {
    const s = endSession(true);
    if (s) s.onDone?.(null);
}

function confirmSelection() {
    const s = _session;
    if (!s) return;
    const items = [...(app.canvas?.selectedItems || [])];
    const { nodes, groups, skippedSwitches } = partitionSelection(s.node, items);
    if (effectiveNodeCount(s.node, nodes, groups) === 0) {
        // Staying in the session beats silently creating an empty toggle. To
        // clear a toggle's targets, delete the toggle.
        updateBannerCount();
        if (s.countEl) {
            s.countEl.textContent = skippedSwitches
                ? "Only Bypass Switches selected — those can't be bypassed."
                : "Select at least one node or backdrop, then press Enter.";
            s.countEl.classList.add("bat-bsw-empty");
        }
        return;
    }
    endSession(false);
    s.onDone?.({
        nodes: nodes.map((n) => n.id),
        groups: groups.map((g) => g.id),
        groupTitles: groups.map((g) => g.title || ""),
        nodeCount: nodes.length,
        groupCount: groups.length,
    });
}

/**
 * Put the canvas into selection mode.
 *
 * @param node   the switch that will own the result
 * @param opts   { mode: "create" | "recapture", preset?, onDone? }
 * @returns      true if a session started; false if one was already running
 *               (callers that closed their own dialog first need to know, or
 *               an unsaved draft vanishes with nothing to show for it)
 *
 * The current selection is cleared on entry so nothing the artist happened to
 * have selected gets swept into the toggle by accident. It is restored if they
 * cancel. In "recapture" mode the preset's existing targets are pre-selected
 * instead, so adding one node to a toggle is one shift-click rather than a
 * full re-pick.
 */
function enterSelectionMode(node, opts = {}) {
    const canvas = app.canvas;
    if (!canvas) { toast("No canvas available.", "error"); return false; }
    if (_session) {
        toast("Already in selection mode — finish or cancel that one first.",
              "warn");
        return false;
    }
    ensureStyles();

    const preset = opts.preset || null;
    const session = {
        node,
        mode: opts.mode === "recapture" ? "recapture" : "create",
        preset,
        presetName: preset?.name || "",
        prevSelection: [...(canvas.selectedItems || [])],
        prevOnSelectionChange: Object.prototype.hasOwnProperty.call(
            canvas, "onSelectionChange") ? canvas.onSelectionChange : undefined,
        onDone: opts.onDone || ((captured) => defaultCreate(node, captured)),
    };
    _session = session;

    try { canvas.deselectAll?.(); } catch (e) {}
    if (preset) {
        const pre = [];
        for (const id of preset.nodes || []) {
            const t = node.graph?.getNodeById?.(id);
            if (t) pre.push(t);
        }
        for (const gid of preset.groups || []) {
            const g = findGroup(node.graph, gid);
            if (g) pre.push(g);
        }
        if (pre.length) {
            try { canvas.selectItems?.(pre, true); } catch (e) {}
        }
    }

    session.banner = buildBanner(session);
    window.addEventListener("keydown", onSelectionKey, true);

    // Chain the canvas hook for instant updates, and poll as a backstop: the
    // hook is not fired by every path that can change the selection (deleting
    // a selected node, for one), and a stale count in the banner is exactly
    // the kind of thing that makes someone confirm the wrong selection.
    const prev = session.prevOnSelectionChange;
    canvas.onSelectionChange = function (...args) {
        const r = typeof prev === "function" ? prev.apply(this, args) : undefined;
        updateBannerCount();
        return r;
    };
    session.poll = setInterval(updateBannerCount, 200);
    updateBannerCount();
    return true;
}

// ---------------------------------------------------------------------------
// Dialogs
// ---------------------------------------------------------------------------

/**
 * A modal overlay, built by hand rather than with <dialog>.
 *
 * Deliberately not a <dialog>: ComfyUI's core keybinding handler bails out
 * without preventDefault whenever anything is in its dialog stack, so a
 * <dialog> that outlives its moment — or one the core happens to see — takes
 * every core hotkey down with it. A plain absolutely-positioned div cannot do
 * that.
 *
 * Key routing has two legs. Escape is caught on window in capture phase so it
 * closes the dialog no matter where focus sits, and never reaches the graph.
 * Every other key is stopped at the overlay, in *bubble* phase, so our inputs
 * receive their keystrokes normally but the document-level core handler does
 * not also see them — otherwise typing a toggle name would trigger whatever
 * single-letter shortcuts happen to be bound.
 */
function openOverlay(onEscape) {
    ensureStyles();
    const overlay = el("div", "bat-bsw-overlay");
    const panel = el("div", "bat-bsw-panel");
    panel.tabIndex = -1;
    overlay.append(panel);

    const onWindowKey = (e) => {
        if (e.key !== "Escape") return;
        e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
        close();
        onEscape?.();
    };
    const close = () => {
        window.removeEventListener("keydown", onWindowKey, true);
        try { overlay.remove(); } catch (e) {}
    };

    overlay.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") e.stopPropagation();
    });
    window.addEventListener("keydown", onWindowKey, true);
    document.body.appendChild(overlay);
    panel.focus();
    return { overlay, panel, close };
}

/** "6 nodes · 1 backdrop · 2 missing" for one preset. */
function metaFor(node, preset) {
    const info = resolveTargets(node, preset);
    const bits = [];
    const gc = (preset.groups || []).length;
    bits.push(`${info.nodes.length} node${info.nodes.length === 1 ? "" : "s"}`);
    if (gc) bits.push(`${gc} backdrop${gc === 1 ? "" : "s"}`);
    const frag = el("span", "bat-bsw-meta", bits.join(" · "));
    if (info.missing) {
        frag.append(document.createTextNode(" · "));
        frag.append(el("span", "bat-bsw-missing", `${info.missing} missing`));
    }
    return frag;
}

/** <select> for a preset's exclusive-group membership. */
function exSelect(draft, current, onChange) {
    const sel = document.createElement("select");
    const add = (value, text) => {
        const o = document.createElement("option");
        o.value = value; o.textContent = text;
        sel.append(o);
    };
    add("", "— independent —");
    for (const g of draft.exgroups) add(g.id, `group: ${g.name}`);
    add("__new__", "＋ new group…");
    sel.value = draft.exgroups.some((g) => g.id === current) ? current : "";
    sel.addEventListener("change", () => onChange(sel.value, sel));
    return sel;
}

/**
 * Name a freshly captured selection and add it as a toggle.
 *
 * The toggle's starting state is read off the graph rather than defaulted:
 * if every captured node is already bypassed the toggle starts off, otherwise
 * on. Creating a toggle therefore never changes a single node's mode, which is
 * what you want when you are describing a workflow that already works.
 */
function defaultCreate(node, captured) {
    if (!captured) return;
    const draft = readState(node);

    // A backdrop's title is almost always the name you would have typed.
    let suggested = "";
    if (captured.groupCount === 1 && captured.nodeCount === 0) {
        suggested = captured.groupTitles[0] || "";
    }
    if (!suggested) suggested = `Toggle ${draft.presets.length + 1}`;

    const { panel, close } = openOverlay();
    panel.append(el("h3", null, "New toggle"));

    const probe = { nodes: captured.nodes, groups: captured.groups };
    const resolved = resolveTargets(node, probe);
    const startsOn = resolved.nodes.some((n) => n.mode !== MODE_BYPASS);
    panel.append(el("p", "bat-bsw-sub",
        `Captured ${captured.nodeCount} node(s)` +
        (captured.groupCount ? ` and ${captured.groupCount} backdrop(s)` : "") +
        ` — ${resolved.nodes.length} node(s) controlled. ` +
        `The toggle will start ${startsOn ? "on" : "off (bypassed)"}, ` +
        `matching how the graph is right now.`));

    const nameRow = el("div");
    nameRow.append(el("h4", null, "Name"));
    const name = document.createElement("input");
    name.type = "text";
    name.value = suggested;
    nameRow.append(name);
    panel.append(nameRow);

    const grpRow = el("div");
    grpRow.append(el("h4", null, "Exclusive group"));
    grpRow.append(el("p", "bat-bsw-sub",
        "Toggles in the same group draw as one dropdown and only one can be on."));
    let exId = "";
    let newGroupName = null;
    const newName = document.createElement("input");
    newName.type = "text";
    newName.placeholder = "New group name, e.g. Mode";
    newName.style.display = "none";
    const sel = exSelect(draft, "", (v) => {
        exId = v;
        newName.style.display = v === "__new__" ? "" : "none";
        if (v === "__new__") newName.focus();
    });
    grpRow.append(sel);
    grpRow.append(newName);
    panel.append(grpRow);

    const commit = () => {
        const finalName = name.value.trim() || suggested;
        let ex = "";
        if (exId === "__new__") {
            newGroupName = newName.value.trim();
            if (!newGroupName) { newName.focus(); return; }
            ex = uid("g");
            draft.exgroups.push({ id: ex, name: newGroupName });
        } else if (exId) {
            ex = exId;
        }

        const preset = {
            id: uid("p"),
            name: finalName,
            on: startsOn,
            ex,
            nodes: captured.nodes.slice(),
            groups: captured.groups.slice(),
        };
        draft.presets.push(preset);
        close();

        asOneUndoStep(node, () => {
            // Joining a group that already has an active member is the one
            // case where creating a toggle *does* touch the graph: two ON
            // members is not a state the group can be in, and the toggle the
            // artist just made is the one they meant.
            if (preset.on && ex) {
                for (const sib of draft.presets) {
                    if (sib.id === preset.id || sib.ex !== ex || !sib.on) continue;
                    sib.on = false;
                    applyPreset(node, sib, false);
                }
                applyPreset(node, preset, true);
            }
            writeState(node, draft);
        });
        buildWidgets(node, true);
        toast(`Added “${finalName}” — ${resolved.nodes.length} node(s).`);
    };

    name.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); commit(); }
    });

    const actions = el("div", "bat-bsw-actions");
    actions.append(el("div", "bat-bsw-spacer"));
    actions.append(button("Cancel", "", close));
    actions.append(button("Add toggle", "bat-bsw-primary", commit));
    panel.append(actions);

    name.focus();
    name.select();
}

/**
 * Rename / reorder / regroup / re-capture / delete.
 *
 * Everything is edited on a deep copy and only written on Save, so Cancel is
 * a real cancel. `draft` is passed back in when the dialog re-opens after a
 * detour through selection mode, so a re-capture does not lose the renames
 * someone had already typed.
 */
function openEditDialog(node, draftIn) {
    const before = readState(node);
    let draft = draftIn || normaliseState(JSON.parse(JSON.stringify(before)));
    const { panel, close } = openOverlay();

    panel.append(el("h3", null, "🦇 Bypass Switch — toggles"));
    panel.append(el("p", "bat-bsw-sub",
        "Deleting a toggle leaves its nodes exactly as they are — it removes " +
        "the switch, not the bypass."));

    const rows = el("div", "bat-bsw-rows");
    panel.append(rows);

    const groupsSection = el("div");
    panel.append(groupsSection);

    const actions = el("div", "bat-bsw-actions");
    panel.append(actions);

    const totalMissing = () => draft.presets.reduce(
        (n, p) => n + resolveTargets(node, p).missing, 0);

    const render = () => {
        rows.textContent = "";
        if (!draft.presets.length) {
            rows.append(el("div", "bat-bsw-empty-note",
                "No toggles. Close this and press “＋ Add toggle…”."));
        }
        draft.presets.forEach((p, i) => {
            const row = el("div", "bat-bsw-row");

            const arrows = el("div", "bat-bsw-arrows");
            const up = button("▲", "bat-bsw-tiny", () => {
                [draft.presets[i - 1], draft.presets[i]] =
                    [draft.presets[i], draft.presets[i - 1]];
                render();
            });
            const down = button("▼", "bat-bsw-tiny", () => {
                [draft.presets[i + 1], draft.presets[i]] =
                    [draft.presets[i], draft.presets[i + 1]];
                render();
            });
            up.disabled = i === 0;
            down.disabled = i === draft.presets.length - 1;
            arrows.append(up, down);
            row.append(arrows);

            const name = document.createElement("input");
            name.type = "text";
            name.value = p.name;
            name.addEventListener("input", () => { p.name = name.value; });
            row.append(name);

            row.append(metaFor(node, p));

            row.append(exSelect(draft, p.ex, (v, sel) => {
                if (v !== "__new__") { p.ex = v; render(); return; }
                // Inline rather than a nested prompt: a second modal over this
                // one is a focus-management problem nobody needs.
                const gname = (window.prompt("New exclusive group name:",
                                             "Mode") || "").trim();
                if (!gname) { sel.value = p.ex || ""; return; }
                const gid = uid("g");
                draft.exgroups.push({ id: gid, name: gname });
                p.ex = gid;
                render();
            }));

            const rowActions = el("div", "bat-bsw-actions");
            rowActions.append(button("Re-select…", "", () => {
                close();
                const started = enterSelectionMode(node, {
                    mode: "recapture",
                    preset: p,
                    onDone: (captured) => {
                        if (captured) {
                            p.nodes = captured.nodes.slice();
                            p.groups = captured.groups.slice();
                            // Re-selecting is the documented cure for a pasted
                            // copy, so stop calling it one.
                            delete node._bswDuplicate;
                        }
                        openEditDialog(node, draft);
                    },
                });
                if (!started) openEditDialog(node, draft);
            }));
            rowActions.append(button("✕", "bat-bsw-danger", () => {
                draft.presets.splice(i, 1);
                render();
            }));
            row.append(rowActions);

            rows.append(row);
        });

        // Group renames live in their own section: they are shared by several
        // rows, so editing one inside a row would be a lie about its scope.
        groupsSection.textContent = "";
        if (draft.exgroups.length) {
            groupsSection.append(el("h4", null, "Exclusive groups"));
            for (const g of draft.exgroups) {
                const row = el("div", "bat-bsw-row");
                row.style.gridTemplateColumns = "1fr auto auto";
                const gname = document.createElement("input");
                gname.type = "text";
                gname.value = g.name;
                gname.addEventListener("input", () => { g.name = gname.value; });
                row.append(gname);
                const members = draft.presets.filter((p) => p.ex === g.id).length;
                row.append(el("span", "bat-bsw-meta",
                    `${members} toggle${members === 1 ? "" : "s"}`));
                row.append(button("Ungroup", "bat-bsw-danger", () => {
                    for (const p of draft.presets) if (p.ex === g.id) p.ex = "";
                    draft.exgroups = draft.exgroups.filter((x) => x.id !== g.id);
                    render();
                }));
                groupsSection.append(row);
            }
        }

        actions.textContent = "";
        const missing = totalMissing();
        if (missing) {
            actions.append(button(
                `Clean up ${missing} missing target${missing === 1 ? "" : "s"}`,
                "bat-bsw-warn",
                () => {
                    for (const p of draft.presets) {
                        p.nodes = (p.nodes || []).filter(
                            (id) => !!node.graph?.getNodeById?.(id));
                        p.groups = (p.groups || []).filter(
                            (gid) => !!findGroup(node.graph, gid));
                    }
                    render();
                }));
        }
        actions.append(el("div", "bat-bsw-spacer"));
        actions.append(button("Cancel", "", close));
        actions.append(button("Save", "bat-bsw-primary", () => {
            close();
            const saved = normaliseState(draft);
            asOneUndoStep(node, () => {
                writeState(node, saved);
                // Normalisation can switch a toggle off — moving a second
                // active toggle into an exclusive group, for one — and a
                // toggle that says "off" while its nodes are still enabled is
                // exactly the inconsistency this node exists to prevent. So
                // any on-state that changed as a result of saving gets pushed
                // onto the graph.
                const wasOn = new Map(before.presets.map((p) => [p.id, p.on]));
                for (const p of saved.presets) {
                    if (!wasOn.has(p.id) || wasOn.get(p.id) === p.on) continue;
                    applyPreset(node, p, p.on);
                }
            });
            buildWidgets(node, true);
        }));
    };

    render();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Bat.BypassSwitch",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments)
                                    : undefined;
            ensureStyles();
            // A management node should not look like a processing node. Set
            // here rather than in a style hook so a recolour by hand still
            // wins: configure() restores the saved colour after this runs.
            this.color = "#372a5c";
            this.bgcolor = "#241b3d";
            buildWidgets(this, true);
            return r;
        };

        // Widget values are restored inside configure(), before onConfigure is
        // called, so this is the first point at which the saved JSON is
        // readable. `false` suppresses the ⚠ markers: mid-load the rest of the
        // graph may not exist yet, and every target would look missing.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            buildWidgets(this, false);
            scheduleWarningRefresh(this);
            return r;
        };

        // A node deleted mid-selection would leave the banner up and the
        // session pointing at a corpse.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (_session?.node === this) cancelSelection();
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = getExtraMenuOptions
                ? getExtraMenuOptions.apply(this, arguments) : undefined;
            const node = this;
            const state = readState(node);
            options.push(
                null,
                {
                    content: "🦇 Add toggle…",
                    callback: () => enterSelectionMode(node, { mode: "create" }),
                },
                {
                    content: "🦇 Edit toggles…",
                    disabled: !state.presets.length,
                    callback: () => openEditDialog(node),
                },
                {
                    // The deliberate escape hatch from "on load we do not
                    // touch node modes": this is how you make the switch
                    // authoritative again after hand-bypassing something.
                    content: "🦇 Re-apply all toggles to the graph",
                    disabled: !state.presets.length,
                    callback: () => reapplyAll(node),
                },
            );
            return r;
        };
    },
});
