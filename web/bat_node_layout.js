/**
 * BAT — dual-mode DOM-widget sizing (ComfyUI Nodes 1.0 + Nodes 2.0).
 *
 * The problem
 * -----------
 * Nodes 1.0 (classic litegraph canvas) sizes a node from `node.size`, so the
 * editors set `this.size = [w, h]` right after addDOMWidget and everything
 * lines up.
 *
 * Nodes 2.0 (`Comfy.VueNodes.Enabled`, DOM/Vue rendering) computes node height
 * BOTTOM-UP from each widget's `computeLayoutSize()`, which returns
 * `{ minHeight, maxHeight, minWidth }`. The built-in DOMWidgetImpl reads
 * `options.getMinHeight?.()` / `options.getMaxHeight?.()`, falling back to the
 * CSS custom properties `--comfy-widget-min-height` / `--comfy-widget-max-height`.
 * A direct `node.size = [...]` assignment is simply ignored there (and actively
 * recomputed away when `Comfy.VueNodes.AutoScaleLayout` is on).
 *
 * None of the canvas editors implemented any of that, which produced two
 * distinct failures under 2.0:
 *
 *   • Roots with `min-height` (roto / animated_crop / animated_grade) rendered,
 *     but the node height and the widget height disagreed — a grey band below
 *     the editor, and dragging the node didn't resize the canvas.
 *   • Roots with only `height:100%` and NO height source (bat_crop,
 *     bat_ref_aligner, bat_frame_picker) had nothing to derive a height from in
 *     a 2.0 flex slot, so the editor collapsed to ~1px or vanished.
 *
 * bat_filename_prefix.js already solved this correctly for its small editor and
 * documents the key insight: if min !== max the widget is treated as "growable"
 * and expands to fill, producing a grey overflow rectangle. This module
 * generalises that fix.
 *
 * Usage
 * -----
 *   import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
 *
 *   const w = addBatDOMWidget(node, "bat_roto_editor", "bat_roto_editor", el, {
 *       minWidth: 640,
 *       height: 540,          // number, or () => number for measured content
 *       growable: true,       // canvas editors: absorb extra node height
 *   });
 *   clampNodeSize(node, 640, 540);   // no-op under Nodes 2.0
 */

import { app } from "../../scripts/app.js";

/**
 * Is the Vue (Nodes 2.0) renderer active?
 *
 * Read live rather than cached — the artist can toggle it in Settings without
 * reloading, and we want the next layout pass to be correct.
 */
export function vueNodesEnabled() {
    try {
        const v = app?.extensionManager?.setting?.get("Comfy.VueNodes.Enabled");
        if (typeof v === "boolean") return v;
    } catch (_) { /* setting store not ready */ }
    try {
        const v = app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled");
        if (typeof v === "boolean") return v;
    } catch (_) { /* older frontends */ }
    return false;
}

/**
 * addDOMWidget + the sizing contract BOTH renderers need.
 *
 * @param {object} node
 * @param {string} name        widget name
 * @param {string} type        widget type string
 * @param {HTMLElement} el     editor root element
 * @param {object} opts
 *   minWidth  {number}            hard floor on width (default 320)
 *   height    {number|Function}   design height, or () => measured height
 *   growable  {boolean}           true  → editor absorbs extra node height
 *                                 false → pin min===max (avoids the grey
 *                                         overflow rectangle; use for content
 *                                         whose height is exactly known)
 *   maxHeight {number}            optional cap when growable
 *   onResize  {Function}          (w, h) called on every DOM box change —
 *                                 replaces the Nodes-1.0-only `onResize` node
 *                                 hook, and fires in BOTH renderers.
 *   ...rest                       forwarded to addDOMWidget options
 * @returns the created widget (or null if addDOMWidget threw)
 */
/**
 * Headroom for a growable widget with no explicit `maxHeight`: the artist can
 * drag an editor to this multiple of its design height, but the layout won't
 * expand it there on its own. Pass an explicit `maxHeight` to override.
 */
const GROW_FACTOR = 3;

export function addBatDOMWidget(node, name, type, el, opts = {}) {
    const {
        minWidth = 320,
        height,
        growable = false,
        maxHeight = null,
        onResize = null,
        ...rest
    } = opts;

    const measure = typeof height === "function"
        ? height
        : () => (typeof height === "number" ? height : 240);

    const getMin = () => Math.max(1, Math.round(measure()));
    // A growable widget with no explicit cap used to advertise
    // Number.MAX_SAFE_INTEGER. Under Nodes 1.0 that was harmless — clampNodeSize
    // pinned node.size straight after. Under Nodes 2.0 clampNodeSize is a no-op
    // and height is derived from this ceiling, so "unbounded" reads as "absorb
    // all available height" and the node grew past the bottom of the viewport
    // (and resisted manual resize, since the layout re-derived every frame).
    // Default to a multiple of the design height instead: the node opens at its
    // design size, the artist can still drag it taller, and it never self-expands
    // to fill the canvas. An explicit maxHeight still wins.
    const getMax = growable
        ? () => (maxHeight != null
            ? Math.round(maxHeight)
            : Math.round(getMin() * GROW_FACTOR))
        : getMin;                       // pinned: min === max

    let w = null;
    try {
        w = node.addDOMWidget(name, type, el, {
            serialize: false,
            hideOnZoom: false,
            // Nodes 1.0 DOM widgets and the 2.0 DOMWidgetImpl default both read
            // these option callbacks.
            getMinHeight: getMin,
            getMaxHeight: getMax,
            getHeight: getMin,
            ...rest,
        });
    } catch (e) {
        console.error(`[BAT.layout] addDOMWidget failed for ${name}:`, e);
        return null;
    }

    // Set computeLayoutSize explicitly so we never depend on the DOMWidgetImpl
    // default happening to consult our options. This mirrors what the frontend's
    // own video widget does.
    try {
        // maxHeight goes through getMax() so the growable-with-no-cap case
        // reports the same real ceiling as the option callbacks above. Returning
        // `undefined` here previously left the 2.0 layout with no ceiling at all,
        // which is the other half of the runaway-height bug.
        w.computeLayoutSize = () => ({
            minHeight: getMin(),
            maxHeight: getMax(),
            minWidth,
        });
    } catch (_) { /* frozen widget object on some builds — options still apply */ }

    // Publish the CSS-var fallback the frontend reads when no getMinHeight /
    // getMaxHeight is supplied. Also gives the root an intrinsic height, which
    // is what the `height:100%`-only editors were missing.
    try {
        el.style.setProperty("--comfy-widget-min-height", `${getMin()}px`);
        // Publish the ceiling too. Previously only the pinned case set this, so a
        // growable editor gave the frontend a floor and no ceiling — the CSS-var
        // half of the runaway-height bug.
        // (getMax() === getMin() in the pinned case, so this covers both.)
        el.style.setProperty("--comfy-widget-max-height", `${getMax()}px`);
        // A plain min-height so the element has a height source in ANY flex
        // context — this alone stops bat_crop / bat_ref_aligner collapsing.
        if (!el.style.minHeight) el.style.minHeight = `${getMin()}px`;
    } catch (_) {}

    // Replaces the Nodes-1.0-only `onResize` node callback (which never fires
    // under 2.0). Measuring the element works in both renderers and needs none
    // of the hardcoded chrome offsets (`node.size[0] - 45`) the editors used.
    if (onResize) {
        try {
            const ro = new ResizeObserver(() => {
                try { onResize(el.clientWidth, el.clientHeight); }
                catch (e) { console.error("[BAT.layout] onResize handler failed:", e); }
            });
            ro.observe(el);
            (node._batLayoutObservers ||= []).push(ro);
        } catch (_) { /* no ResizeObserver: sizes stay at their initial values */ }
    }

    return w;
}

/**
 * Clamp the node's size — a NO-OP under Nodes 2.0, where height is derived from
 * computeLayoutSize and writing node.size fights the layout.
 *
 * Use this in place of every post-addDOMWidget `this.size = [...]`.
 */
export function clampNodeSize(node, minW, minH) {
    if (vueNodesEnabled()) return;
    try {
        const w = Math.max(minW || 0, node.size?.[0] || 0);
        const h = Math.max(minH || 0, node.size?.[1] || 0);
        if (typeof node.setSize === "function") node.setSize([w, h]);
        else node.size = [w, h];
        node.setDirtyCanvas?.(true, true);
    } catch (e) {
        console.error("[BAT.layout] clampNodeSize failed:", e);
    }
}

/** Disconnect layout observers (called from the lifecycle teardown). */
export function disposeBatLayout(node) {
    for (const ro of node?._batLayoutObservers || []) {
        try { ro.disconnect(); } catch (_) {}
    }
    if (node) node._batLayoutObservers = [];
}
