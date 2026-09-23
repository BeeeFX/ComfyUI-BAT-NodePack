/**
 * 🦇 EXR Layer / 🦇 Cryptomatte Matte — pick a layer by name instead of spelling it.
 *
 * Both nodes take a `layer_name` string, and the names are whatever the render
 * happened to be written with (`diffuse`, `N`, `CryptoObject00`, `refract01`…).
 * Nobody remembers those, and a typo is only discovered on execute. So the
 * backend sends the names it actually found over the `bat-layers` message and
 * this adds a **Pick layer…** button that lists them.
 *
 * Why a separate button rather than turning `layer_name` into a combo: a combo
 * is only as good as its `options.values`, which are empty until the node has
 * run once — so before the first run the field would be a dropdown of nothing,
 * with no way to type the name you already know. And the frontend's widget-value
 * store hands a rebuilt widget the value the LAST widget of that name held,
 * which is how a combo swap silently resurrects a stale layer name on a
 * different file. The text field stays the single source of truth and the
 * button only writes into it.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPES = new Set(["Bat_ExrLayer", "Bat_CryptomatteMatte"]);

/**
 * Layer names last reported, keyed by the id the backend saw — the node's
 * EXECUTION id, which for a node inside a subgraph is its path ("12:5"), not
 * its local `node.id`. Insertion order is kept most-recent-last.
 */
const known = new Map();

/** Names for `node`: an exact id match, else the latest subgraph path ending in it. */
function namesFor(node) {
    const id = String(node.id);
    if (known.has(id)) return known.get(id);
    const suffix = `:${id}`;
    for (const [key, names] of [...known].reverse()) {
        if (key.endsWith(suffix)) return names;
    }
    return [];
}

// Where the last press landed. Under Nodes 2.0 the button widget's callback
// gets no event (WidgetButton.vue calls callback(undefined)), and a ContextMenu
// without one opens at the window's top-left corner. Capture phase, so nothing
// that stops propagation further down can hide the press from us.
let lastPointerDown = null;
document.addEventListener("pointerdown", (e) => { lastPointerDown = e; }, true);

function setLayerName(node, value) {
    const w = node.widgets?.find((x) => x.name === "layer_name");
    if (!w) return;
    const old = w.value;
    w.value = value;
    // Assigning .value fires nothing. Anything listening the way the rest of
    // the frontend does never hears a value picked from a menu, so announce it.
    try { w.callback?.(value, app.canvas, node); } catch (_) {}
    try { node.onWidgetChanged?.(w.name, value, old, w); } catch (_) {}
    node.setDirtyCanvas(true, true);
}

function pickLayer(node, event) {
    const names = namesFor(node);
    if (!names.length) {
        // Nothing to offer yet. Say why, rather than opening an empty menu.
        app.extensionManager?.toast?.add?.({
            severity: "info",
            summary: "No layers yet",
            detail: "Run the graph once with this node connected to a 🦇 Loader "
                  + "and the layers in the file will be listed here.",
            life: 5000,
        });
        return;
    }
    new LiteGraph.ContextMenu(names, {
        event: event || lastPointerDown || undefined,
        title: "Layer",
        scale: Math.max(1, app.canvas?.ds?.scale || 1),
        callback: (value) => setLayerName(node, value),
    });
}

app.registerExtension({
    name: "BAT.ExrLayers",

    setup() {
        api.addEventListener("bat-layers", ({ detail }) => {
            if (!detail || detail.node == null) return;
            const key = String(detail.node);
            known.delete(key);                      // re-insert as most recent
            known.set(key, Array.isArray(detail.layers) ? detail.layers : []);
            const node = app.graph?.getNodeById?.(detail.node);
            node?.setDirtyCanvas(true, true);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            // serialize:false — this is a control, not state. Saving it would
            // put an extra entry in widgets_values and shift every widget after
            // it by one on load, which is how a button silently corrupts the
            // values of the widgets below it.
            const btn = this.addWidget("button", "Pick layer…", null,
                                       (_v, _c, _n, _pos, event) => pickLayer(node, event));
            btn.serialize = false;
            return r;
        };

        // The names belong to a node id, and a deleted node's id can be reused.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            known.delete(String(this.id));
            // A node inside a subgraph was reported under its path.
            if (this.graph && this.graph !== app.graph) {
                for (const key of [...known.keys()]) {
                    if (key.endsWith(`:${this.id}`)) known.delete(key);
                }
            }
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
