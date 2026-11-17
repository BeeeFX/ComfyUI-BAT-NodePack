import { app } from "../../scripts/app.js";

const NODE_TYPE = "Bat_VaceBatchTool";
const ADD_BTN_NAME = "+ Keyframe";
const RM_BTN_NAME = "\u2212 Keyframe";

app.registerExtension({
    name: "BAT.VaceBatchTool",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            ensureProps(this);
            addControlButtons(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            ensureProps(this);

            const savedCount = (info && info.properties && info.properties.kf_count) || 0;
            const savedIndices = (info && info.properties && info.properties.kf_indices) || [];

            // Reset and rebuild
            this.properties.kf_count = 0;
            this.properties.kf_indices = [];

            for (let i = 0; i < savedCount; i++) {
                const v = typeof savedIndices[i] === "number" ? savedIndices[i] : 0;
                addKeyframeRow(this, v, /*restoring=*/true);
            }

            this.setSize(this.computeSize());
            this.setDirtyCanvas(true, true);
            return r;
        };
    },
});

function ensureProps(node) {
    node.properties = node.properties || {};
    if (typeof node.properties.kf_count !== "number") node.properties.kf_count = 0;
    if (!Array.isArray(node.properties.kf_indices)) node.properties.kf_indices = [];
}

/**
 * Remove a widget the way the frontend expects. `removeWidget` also drops the
 * widget's entry from the widget value store; splicing `node.widgets` alone
 * left that entry behind, so re-adding `index_N` re-bound to the stale state
 * and came back showing the removed row's old index instead of 0.
 */
function dropWidget(node, widget) {
    if (!widget) return;
    if (typeof node.removeWidget === "function") {
        node.removeWidget(widget);
    } else {
        const i = node.widgets.indexOf(widget);
        if (i !== -1) node.widgets.splice(i, 1);
    }
}

function addControlButtons(node) {
    const addBtn = node.addWidget("button", ADD_BTN_NAME, null, () => {
        addKeyframeRow(node, 0, false);
    });
    addBtn.serialize = false;

    const rmBtn = node.addWidget("button", RM_BTN_NAME, null, () => {
        removeKeyframeRow(node);
    });
    rmBtn.serialize = false;
}

function addKeyframeRow(node, defaultIndex, restoring) {
    const n = (node.properties.kf_count || 0) + 1;
    node.properties.kf_count = n;
    while (node.properties.kf_indices.length < n) node.properties.kf_indices.push(0);
    node.properties.kf_indices[n - 1] = defaultIndex;

    if (!restoring) {
        node.addInput(`image_${n}`, "IMAGE");
        node.addInput(`mask_${n}`, "MASK");
    }

    const slot = n - 1;
    const w = node.addWidget(
        "number",
        `index_${n}`,
        defaultIndex,
        (v) => {
            const intv = Math.max(0, Math.floor(v));
            node.properties.kf_indices[slot] = intv;
            w.value = intv;
        },
        { min: 0, max: 100000, step: 10, precision: 0 }
    );

    keepButtonsLast(node);

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

/**
 * Put the +/− buttons back after the newest index row.
 *
 * Splicing the new widget in front of the buttons only reorders
 * `node.widgets`, which is all Nodes 1.0 draws from. Nodes 2.0 renders rows in
 * the widget store's order — registration order — so the index rows showed up
 * under the buttons there (and after every reload, since onConfigure adds
 * them once the buttons are registered). Removing and re-adding the buttons
 * appends them last in both. They hold no value, so nothing is lost.
 */
function keepButtonsLast(node) {
    const btns = node.widgets.filter((w) => w.name === ADD_BTN_NAME || w.name === RM_BTN_NAME);
    if (!btns.length) return;
    for (const b of btns) dropWidget(node, b);
    addControlButtons(node);
}

function removeKeyframeRow(node) {
    const n = node.properties.kf_count || 0;
    if (n <= 0) return;

    const imgIdx = node.findInputSlot(`image_${n}`);
    if (imgIdx !== -1) node.removeInput(imgIdx);
    const maskIdx = node.findInputSlot(`mask_${n}`);
    if (maskIdx !== -1) node.removeInput(maskIdx);

    dropWidget(node, node.widgets.find((w) => w.name === `index_${n}`));

    node.properties.kf_count = n - 1;
    node.properties.kf_indices.pop();

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}
