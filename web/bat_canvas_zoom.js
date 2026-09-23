/**
 * BAT — unlock zooming further out on the graph canvas.
 *
 * The limit
 * ---------
 * litegraph's `DragAndScale` (one instance per LGraphCanvas, created in its
 * constructor as `canvas.ds`) carries `min_scale = 0.1` / `max_scale = 10`,
 * and *every* zoom path in the frontend funnels through the same clamp in
 * `DragAndScale.changeScale()`:
 *
 *     if (e < this.min_scale) e = this.min_scale
 *
 * • wheel / trackpad zoom → `changeDeltaScale()` → `changeScale()`
 * • Alt+- / Alt+= and the View menu's Zoom In / Zoom Out commands
 *   (`Comfy.Canvas.ZoomIn` / `ZoomOut`) → `ds.changeScale(scale * 1.1)`
 * • the zoom-percentage box in the canvas Zoom Controls popover
 *   → `setAppZoomFromPercentage()` → `changeScale()`
 * • two-finger pinch on a touch device, which clamps against
 *   `canvas.ds.min_scale` by hand rather than calling `changeScale()`
 *
 * So there is exactly one number to move, and moving it covers every gesture
 * and command — including the pinch path, because that reads the live field
 * too rather than a copy of the default.
 *
 * On a big comp graph 10% is not far enough out to see the whole thing at
 * once. This extension lowers that floor. The default here is 5% — twice as
 * far out as stock — on the classic canvas (Nodes 2.0: see below), and the
 * setting goes down to 1% (10× further) for the graphs that need it.
 *
 * Why an instance write and not a prototype patch
 * -----------------------------------------------
 * `min_scale` is an own property assigned in the `DragAndScale` constructor,
 * so a prototype value would be shadowed; intercepting it would mean defining
 * an accessor on the prototype, which would also silently re-floor any *other*
 * DragAndScale in the page. There is only ever one — `new DragAndScale(...)`
 * appears once in the bundle, in the LGraphCanvas constructor, and ComfyUI
 * builds a single canvas that outlives workflow switches (a tab change swaps
 * the *graph*, via `litegraph:set-graph`, and keeps the canvas and its `ds`).
 * A one-shot write on the instance is therefore both sufficient and the least
 * invasive thing that works. `apply()` is idempotent anyway, and runs again on
 * every settings change.
 *
 * Nothing about max_scale changes: zooming *in* past 10× was never the
 * complaint, and the deep-zoom end is where the renderer's text and DOM-widget
 * work gets expensive.
 *
 * What it looks like down there
 * -----------------------------
 * Core's level-of-detail rendering (`MinFontSizeForLOD`) already draws nodes
 * as plain boxes once the title text would be sub-legible, and the DOM-widget
 * layer stops laying out widgets it cannot show. That is the desired
 * behaviour for an overview — and it is also why going further out stays
 * cheap: the extra graph area costs box fills, not text and HTML.
 *
 * Nodes 2.0 is different
 * ----------------------
 * The Vue renderer has none of that. Every node is a full DOM subtree
 * (GraphCanvas.vue renders all of them, with no culling and no LOD), and the
 * transform pane simply scales them, so a zoomed-out overview of a big graph
 * paints every widget of every node. The floor itself works there — the pane
 * mirrors `ds.scale` and has no clamp of its own — but lowering it by default
 * would quietly make the overview heavier for everyone on 2.0.
 *
 * So the *default* follows the renderer: 5% under the classic canvas, stock
 * 10% under Nodes 2.0 — i.e. out of the box this extension changes nothing
 * there. A value the artist sets explicitly is honoured as-is in both, rather
 * than clamped: a slider that silently stops working in one renderer is the
 * more surprising failure. `defaultValue` is a function, which the settings
 * store re-resolves on every read of an unset setting, so this needs no
 * "has the user touched it" bookkeeping; switching renderer re-applies it.
 */

import { app } from "../../scripts/app.js";
import { vueNodesEnabled } from "./bat_node_layout.js";

/** Stock litegraph floor, as a percentage: the tooltip, and the 2.0 default. */
const CORE_MIN_PERCENT = 10;

/** Default: 5% — half of core's floor, i.e. twice as far out. */
const DEFAULT_MIN_PERCENT = 5;

const SETTING_ID = "BAT.Canvas.MinZoom";

/** The default for whichever renderer is active (see the header). */
function defaultPercent() {
    return vueNodesEnabled() ? CORE_MIN_PERCENT : DEFAULT_MIN_PERCENT;
}

/** The artist's value if they set one, else the renderer's default. */
function currentPercent() {
    return app.ui?.settings?.getSettingValue?.(SETTING_ID) ?? defaultPercent();
}

/**
 * Push the floor down to `percent` and, if the view is currently sitting
 * *below* a newly-raised floor, pull it back to the legal minimum.
 *
 * Safe to call before the canvas exists — the setting's onChange fires at
 * registration time, which is well before `setup()`.
 */
function apply(percent) {
    const ds = app?.canvas?.ds;
    if (!ds) return;

    // A frontend too old to resolve a function default hands it over as-is.
    if (typeof percent === "function") percent = percent();
    const scale = Math.max(0.001, Number(percent) / 100);
    if (!Number.isFinite(scale)) return;

    ds.min_scale = scale;

    // Raising the floor while parked underneath it would otherwise leave the
    // view at an out-of-range zoom until the next wheel notch.
    if (ds.scale < scale) {
        ds.changeScale(scale);
        app.canvas.setDirty(true, true);
    }
}

app.registerExtension({
    name: "BAT.CanvasZoom",

    settings: [
        {
            id: SETTING_ID,
            category: ["🦇 BAT", "Canvas", "MinZoom"],
            name: "Minimum canvas zoom (%)",
            tooltip:
                `How far out the graph canvas can zoom. ComfyUI stops at ` +
                `${CORE_MIN_PERCENT}%; ${DEFAULT_MIN_PERCENT}% is twice as ` +
                `far out and is the default on the classic canvas. Under ` +
                `Nodes 2.0 the default stays at ${CORE_MIN_PERCENT}%, because ` +
                `every Vue node is still drawn in full when zoomed out — set ` +
                `a value here to go further out there too. Applies to the ` +
                `wheel, the pinch gesture, the Zoom Out command and the ` +
                `zoom-percentage box.`,
            type: "slider",
            attrs: { min: 1, max: CORE_MIN_PERCENT, step: 0.5 },
            // A function: re-read on every lookup of an unset value, so it
            // follows the renderer. An explicit value is stored and wins.
            defaultValue: defaultPercent,
            onChange: apply,
        },
    ],

    setup() {
        // The canvas is built after extensions register, so the onChange that
        // fired during registration found no `ds`. This is the write that
        // actually lands.
        apply(currentPercent());

        // Switching renderer changes the default, and our own onChange does
        // not fire for that. The legacy settings dialog re-dispatches every
        // setting change as "<id>.change", after the store has the new value.
        try {
            app.ui?.settings?.addEventListener?.(
                "Comfy.VueNodes.Enabled.change", () => apply(currentPercent()));
        } catch (e) { /* older frontend: applies on next reload instead */ }
    },
});
