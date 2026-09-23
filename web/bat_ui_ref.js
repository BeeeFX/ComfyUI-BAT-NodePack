/**
 * BAT — resolve preview payloads that the server kept out of the history.
 *
 * The editors' `ui` dicts (JPEG strips, masks, float tiles) used to travel in
 * the "executed" message, and ComfyUI keeps every one of those in its prompt
 * history for the life of the server. bat_ui_ref.py now writes the payload to
 * a temp file and sends `{bat_ui: [token]}` instead; this extension fetches
 * `/bat/ui/<token>` and calls the node's own onExecuted with the dict it has
 * always received — so no editor had to learn about tokens.
 *
 * It wraps each BAT node INSTANCE's onExecuted, one microtask after
 * `nodeCreated`. `nodeCreated` fires inside the constructor, before LiteGraph
 * calls onNodeCreated — where some editors (the Points Editor) chain their own
 * instance-level onExecuted. Deferring puts this wrapper outside every hook,
 * prototype- or instance-level, so all of them see the resolved message, and
 * the lifecycle stash keeps that for a later undo/paste replay without another
 * fetch. No "executed" event can land in between: those arrive in later tasks.
 *
 * A message with no token (an older server, or a node whose payload was kept
 * inline because the sidecar could not be written) passes straight through.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const UI_KEY = "bat_ui";

/** Fetch a stashed payload; null when it is gone (server restarted, pruned). */
export async function batFetchUi(token) {
    try {
        const r = await api.fetchApi(`/bat/ui/${encodeURIComponent(token)}`);
        if (!r.ok) return null;
        return await r.json();
    } catch (e) {
        console.warn("[BAT] preview payload fetch failed:", e);
        return null;
    }
}

/** The message with its stashed payload merged back in (or null if gone). */
export async function batResolveUi(message) {
    const token = message?.[UI_KEY]?.[0];
    if (typeof token !== "string") return message;
    const payload = await batFetchUi(token);
    if (!payload) return null;
    const out = { ...message, ...payload };
    delete out[UI_KEY];
    return out;
}

function wrapInstance(node) {
    if (!node || node._batUiRefWrapped) return;
    node._batUiRefWrapped = true;
    const inner = node.onExecuted;
    if (typeof inner !== "function") return;
    node.onExecuted = function (message) {
        if (typeof message?.[UI_KEY]?.[0] !== "string") {
            return inner.apply(this, arguments);
        }
        // Two runs can resolve out of order (a big strip, then a small one);
        // only the newest may reach the editor.
        const gen = (this._batUiGen = (this._batUiGen || 0) + 1);
        batResolveUi(message).then((resolved) => {
            if (!resolved || gen !== this._batUiGen || !this.graph) return;
            try { inner.call(this, resolved); }
            catch (e) { console.error("[BAT] onExecuted failed after resolving the preview:", e); }
        });
        return undefined;
    };
}

app.registerExtension({
    name: "BAT.UiRef",
    nodeCreated(node) {
        const cls = String(node?.comfyClass ?? node?.type ?? "");
        if (cls.startsWith("Bat_")) queueMicrotask(() => wrapInstance(node));
    },
});
