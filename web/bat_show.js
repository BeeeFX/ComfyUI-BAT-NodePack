/**
 * BAT — 🦇 Show Any / 🦇 Show Tensor Shape front-end.
 *
 * One read-only <pre> DOM widget per node, filled from the `ui.text` the
 * Python side returns. Both nodes share this file because the only difference
 * between them is what the backend puts in that string.
 *
 * Why a DOM widget and not a multiline STRING widget: a STRING widget
 * serialises into `widgets_values`, so a 4000-character tensor dump would be
 * written into the saved workflow and reloaded forever after. This readout is
 * transient by nature — it belongs to the last run, not to the document.
 * `addBatDOMWidget` also pins the pack's layout/teardown behaviour, so the
 * panel resizes with the node and is disposed with it.
 *
 * Surviving undo: Ctrl+Z in ComfyUI is a full `loadGraphData`, which throws
 * away every node object and builds new ones — so the text from the last run
 * would vanish even though the run is still valid. `batReplayLastExecution`
 * (bat_lifecycle.js) replays the last payload once the graph is configured,
 * keyed per graph — same as the other BAT previews. (This file used to keep
 * its own node.id-keyed map and replay it from onNodeCreated, but the id is
 * not assigned until after onNodeCreated, so that lookup never hit.)
 *
 * 🦇 WAN Context Calculator reuses the panel for its report.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, refreshBatLayout } from "./bat_node_layout.js";
import { batReplayLastExecution } from "./bat_lifecycle.js";

const NODE_TYPES = ["Bat_ShowAny", "Bat_ShowTensorShape", "Bat_WanContextCalculator"];
const PLACEHOLDER = "— not run yet —";

function buildPanel(node) {
    const el = document.createElement("pre");
    Object.assign(el.style, {
        margin: "0",
        padding: "6px 8px",
        overflow: "auto",
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        font: "11px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        color: "var(--input-text, #ddd)",
        background: "var(--comfy-input-bg, #1a1a1a)",
        border: "1px solid var(--border-color, #333)",
        borderRadius: "4px",
        userSelect: "text",
        cursor: "text",
    });
    el.textContent = PLACEHOLDER;

    const widget = addBatDOMWidget(node, "bat_show_text", "bat_show", el, {
        minWidth: 240,
        height: 120,
        growable: true,
        maxHeight: 520,
    });

    node._batShowEl = el;
    node._batShowWidget = widget;
    return widget;
}

function render(node, text) {
    const el = node?._batShowEl;
    if (!el) return;
    const str = (text == null || text === "") ? PLACEHOLDER : String(text);
    el.textContent = str;
    // Scroll back to the top: on a re-run you want the start of the new value,
    // not wherever you happened to have scrolled for the previous one.
    el.scrollTop = 0;
    try {
        refreshBatLayout(node, node._batShowWidget);
    } catch (e) {
        // Layout is cosmetic — never let it break the readout itself.
        console.warn("[BAT.show] layout refresh failed:", e);
    }
}

/** ComfyUI hands `ui.text` back as an array of strings. */
function textFromMessage(message) {
    const t = message?.text;
    if (t == null) return null;
    return Array.isArray(t) ? t.join("\n") : String(t);
}

app.registerExtension({
    name: "BAT.ShowNodes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.includes(nodeData?.name)) return;

        // Replays the last run's payload after an undo/redo rebuilds the graph.
        batReplayLastExecution(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            try {
                buildPanel(this);
            } catch (e) {
                console.error("[BAT.show] could not build panel:", e);
            }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            try {
                const text = textFromMessage(message);
                if (text != null) render(this, text);
            } catch (e) {
                console.error("[BAT.show] could not render payload:", e);
            }
            return r;
        };
    },
});

console.log("[BAT.show] module loaded");
