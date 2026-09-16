/**
 * BAT — a hotkey for "run only the selected output nodes".
 *
 * What already exists, and what does not
 * -------------------------------------
 * Running a single branch is *not* something this file implements. The
 * frontend has shipped the command since 1.19.6:
 *
 *     Comfy.QueueSelectedOutputNodes
 *
 * It is what the ▶ button in the selection toolbox calls, and what the node
 * context menu's "Run branch" calls. Internally it resolves the execution
 * path of the selected output nodes and queues the prompt with
 * `partialExecutionTargets`, which the server reads as
 * `partial_execution_targets` in `/prompt` and turns into the output set in
 * `execution.validate_prompt()` — so only those outputs, and only the nodes
 * feeding them, are executed.
 *
 * What core does *not* ship is a keybinding for it. Its default list binds
 * Ctrl+Enter to `Comfy.QueuePrompt`, Ctrl+Shift+Enter to `QueuePromptFront`
 * and Ctrl+Alt+Enter to `Interrupt`, and stops there — the branch command is
 * registered but unbound, reachable only with the mouse. That is the whole
 * gap this file fills.
 *
 * Why a default keybinding and not our own key handler
 * ----------------------------------------------------
 * `registerExtension({ keybindings: [...] })` hands the combo to core's
 * keybinding store as a *default* binding. That buys three things a
 * hand-rolled `keydown` listener would have to reimplement, and would
 * reimplement slightly differently:
 *
 *   • the bailouts in `keybindHandler` — it ignores combos typed into a
 *     TEXTAREA/INPUT/contentEditable when the combo is "reserved by text
 *     input", and ignores everything while a dialog is on `dialogStack`;
 *   • conflict detection — the capture dialog tells you if a combo is
 *     already taken, or reserved by the browser;
 *   • rebinding and unsetting, persisted per user in
 *     `Comfy.Keybinding.NewBindings` / `Comfy.Keybinding.UnsetBindings`.
 *
 * So the binding below is a *default*, not a lock: Settings → Keybinding
 * lists it against this command (tagged as coming from an extension), and
 * changing or clearing it there overrides us. The setting registered further
 * down mirrors the result back into the BAT panel so it is discoverable from
 * where the rest of the pack's options live.
 *
 * Alt+Enter
 * ---------
 * Chosen because it is the one free member of the Enter family — Ctrl,
 * Ctrl+Shift and Ctrl+Alt are all taken by core (queue, queue-front,
 * interrupt), which puts "queue this branch" alongside its siblings rather
 * than somewhere unrelated. It carries a modifier, so it is not
 * "reserved by text input" and will fire from inside a text widget exactly
 * as Ctrl+Enter does.
 *
 * With no output node in the selection the command raises core's own
 * "please select output nodes" toast. That is deliberate: guessing at
 * downstream outputs, or falling back to a full queue, both risk rendering
 * something the artist never looked at.
 */

import { app } from "../../scripts/app.js";

/** Core's branch-execution command. Present since frontend 1.19.6. */
const COMMAND_ID = "Comfy.QueueSelectedOutputNodes";

/** The default this pack ships. Overridable in Settings → Keybinding. */
const DEFAULT_COMBO = { alt: true, key: "Enter" };

const SETTING_ID = "BAT.Hotkeys.RunSelectedOutputs";

/** Where core persists the user's rebinds and unsets. */
const USER_BINDINGS_SETTING = "Comfy.Keybinding.NewBindings";
const UNSET_BINDINGS_SETTING = "Comfy.Keybinding.UnsetBindings";

/**
 * Same labelling core uses in the Keybinding panel: modifiers in
 * ctrl → alt → shift order, joined with " + ".
 */
function formatCombo(combo) {
    const parts = [];
    if (combo.ctrl) parts.push("Ctrl");
    if (combo.alt) parts.push("Alt");
    if (combo.shift) parts.push("Shift");
    parts.push(combo.key);
    return parts.join(" + ");
}

/** Identity of a combo, for matching an unset entry against our default. */
function comboKey(combo) {
    return [!!combo.ctrl, !!combo.alt, !!combo.shift, combo.key].join("|");
}

function readSetting(id) {
    try {
        const value = app.ui?.settings?.getSettingValue?.(id);
        return Array.isArray(value) ? value : [];
    } catch {
        return [];
    }
}

/**
 * The combos that actually reach the command right now: our default unless
 * the user has unset it, plus anything they have bound themselves.
 *
 * Read from the two settings rather than from the keybinding store, because
 * the store is internal to the frontend bundle while the settings are public
 * — and they are the same data, since `persistUserKeybindings()` writes the
 * store straight into them.
 */
function effectiveCombos() {
    const unset = new Set(
        readSetting(UNSET_BINDINGS_SETTING)
            .filter((k) => k?.commandId === COMMAND_ID && k?.combo)
            .map((k) => comboKey(k.combo)),
    );

    const combos = [];
    if (!unset.has(comboKey(DEFAULT_COMBO))) combos.push(DEFAULT_COMBO);

    for (const k of readSetting(USER_BINDINGS_SETTING)) {
        if (k?.commandId === COMMAND_ID && k?.combo) combos.push(k.combo);
    }
    return combos;
}

/** A keycap-ish chip, styled off the theme's own variables. */
function chip(text) {
    const el = document.createElement("kbd");
    el.textContent = text;
    el.style.cssText =
        "padding:2px 8px;border-radius:4px;font:inherit;font-size:12px;" +
        "border:1px solid var(--border-color, #444);" +
        "background:var(--comfy-input-bg, rgba(127,127,127,.15));" +
        "white-space:nowrap;";
    return el;
}

/**
 * The BAT-panel mirror. Read-only by design — the combo lives in core's
 * keybinding store, and a second editable copy here would be a second source
 * of truth that could disagree with it.
 *
 * Rendered through a custom setting `type`, which core calls with
 * (name, setValue, value, attrs) and expects a DOM element back. We never
 * call setValue: nothing about this row is stored.
 */
function renderBinding() {
    const row = document.createElement("div");
    row.style.cssText =
        "display:flex;align-items:center;gap:6px;justify-content:flex-end;" +
        "flex-wrap:wrap;";

    let combos = [];
    try {
        combos = effectiveCombos();
    } catch {
        // Never let a settings row throw — it would blank the panel.
    }

    if (combos.length === 0) {
        const none = document.createElement("span");
        none.textContent = "Not bound";
        none.style.opacity = "0.6";
        row.appendChild(none);
    } else {
        combos.forEach((combo, i) => {
            if (i > 0) {
                const or = document.createElement("span");
                or.textContent = "or";
                or.style.opacity = "0.6";
                row.appendChild(or);
            }
            row.appendChild(chip(formatCombo(combo)));
        });
    }
    return row;
}

app.registerExtension({
    name: "BAT.Hotkeys",

    keybindings: [{ combo: DEFAULT_COMBO, commandId: COMMAND_ID }],

    settings: [
        {
            id: SETTING_ID,
            category: ["🦇 BAT", "Hotkeys", "RunSelectedOutputs"],
            name: "Run selected output nodes",
            tooltip:
                "Queues only the selected output nodes and the nodes " +
                "feeding them — the same thing as the ▶ button in the " +
                "selection toolbox. BAT supplies the default combo; change " +
                "or clear it in Settings → Keybinding, under the command " +
                '"Queue Selected Output Nodes".',
            type: renderBinding,
            defaultValue: null,
        },
    ],
});
