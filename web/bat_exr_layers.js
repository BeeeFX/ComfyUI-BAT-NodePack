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

/** Layer names last reported per node id. */
const known = new Map();

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
    const names = known.get(String(node.id)) || [];
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
        event,
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
            known.set(String(detail.node), Array.isArray(detail.layers) ? detail.layers : []);
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
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
