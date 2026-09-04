/**
 * Bat Batch Format — widget visibility.
 *
 * `target_num_frames` is only meaningful in one combination: mode is
 * "specific_num_frames" AND auto_target_frames is off. In every other state
 * it is dead UI that invites people to type a number that does nothing, so
 * it is collapsed out of the node.
 *
 * `auto_target_frames` itself is only meaningful in "specific_num_frames"
 * mode, so it hides in the other two modes for the same reason.
 *
 * Hiding uses the pack's usual approach (type/computeSize/draw/hidden), which
 * removes the row from layout rather than just blanking it — see the `state`
 * widget in bat_animated_grade.js.
 */

import { app } from "/scripts/app.js";

const NODE_TYPE = "Bat_BatchFormat";

function setHidden(node, widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget._batOrigType === undefined) {
            widget._batOrigType = widget.type;
            widget._batOrigComputeSize = widget.computeSize;
            widget._batOrigDraw = widget.draw;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.hidden = true;
    } else if (widget._batOrigType !== undefined) {
        widget.type = widget._batOrigType;
        widget.computeSize = widget._batOrigComputeSize;
        widget.draw = widget._batOrigDraw;
        widget.hidden = false;
    }
}

function syncVisibility(node) {
    const mode = node.widgets?.find(w => w.name === "mode");
    const auto = node.widgets?.find(w => w.name === "auto_target_frames");
    const target = node.widgets?.find(w => w.name === "target_num_frames");
    if (!mode || !auto || !target) return;

    const isSpecific = mode.value === "specific_num_frames";
    setHidden(node, auto, !isSpecific);
    setHidden(node, target, !isSpecific || auto.value === true);

    // Re-fit: the node keeps its old height otherwise, leaving a dead band
    // where the collapsed rows used to be.
    const min = node.computeSize?.();
    if (min) node.setSize([Math.max(node.size[0], min[0]), min[1]]);
    node.setDirtyCanvas?.(true, true);
}

/** Wrap a widget callback so ours runs after the original, never instead. */
function chain(widget, node) {
    const orig = widget.callback;
    widget.callback = function (...args) {
        const r = orig ? orig.apply(this, args) : undefined;
        syncVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "Bat.BatchFormat",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const mode = this.widgets?.find(w => w.name === "mode");
            const auto = this.widgets?.find(w => w.name === "auto_target_frames");
            if (mode) chain(mode, this);
            if (auto) chain(auto, this);
            syncVisibility(this);
            return r;
        };

        // configure() restores serialised widget values after onNodeCreated,
        // so a workflow loaded with auto off would otherwise come back with
        // target_num_frames still collapsed.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            syncVisibility(this);
            return r;
        };
    },
});
