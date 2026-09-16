#!/usr/bin/env python
"""
Behavioural checks for the run-selected-outputs hotkey (``web/bat_hotkeys.js``).

The extension is small — one default keybinding and one read-only settings row
— and both halves fail quietly. A combo that collides with a core default gets
shadowed by whichever binding the store resolves first, and the artist just
sees the wrong thing happen on Alt+Enter. A settings row whose render function
throws does not log an error next to the row; it blanks the panel. So both are
driven here through quickjs, against core's own default keybinding list
(transcribed from comfyui_frontend_package 1.49.6) and a DOM stub.

Six claims:

1. **The keybinding names the command core actually ships.**
   ``Comfy.QueueSelectedOutputNodes`` is what the selection toolbox's ▶ button
   calls; a typo here registers a binding for a command that does not exist
   and the key does nothing at all.

2. **The combo does not collide with a core default.** Ctrl+Enter,
   Ctrl+Shift+Enter and Ctrl+Alt+Enter are queue, queue-front and interrupt.
   Landing on one of those would be worse than no binding.

3. **The combo carries a modifier,** so ``isReservedByTextInput`` does not
   swallow it while a text widget has focus — the same reason Ctrl+Enter works
   from inside a prompt box.

4. **The settings row is shaped the way the dialog expects** — a custom render
   function under the 🦇 BAT panel, with a tooltip saying where the combo is
   really configured.

5. **The mirror reflects the live keybinding state**: the shipped default when
   the user has not touched it, the user's combo after a rebind, and
   "Not bound" once the default is unset. This is read out of
   ``Comfy.Keybinding.NewBindings`` / ``UnsetBindings``, which is where
   ``persistUserKeybindings()`` writes the keybinding store.

6. **The render never throws** — not on missing settings, not on a garbage
   value, not with no settings API at all. A throw here takes the panel with
   it.

    pip install quickjs
    python tests/verify_hotkeys.py
"""

import json
import os
import re
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_failures = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        _failures.append(label)
        if detail:
            print("      ", detail)


# ---------------------------------------------------------------------------
# Core's default keybindings, verbatim from keybindingService in
# comfyui_frontend_package 1.49.6. Only the Enter family and the modifier
# combos matter to the collision check, but the list is kept whole so a future
# frontend bump can be diffed against it rather than re-derived.
# ---------------------------------------------------------------------------

CORE_DEFAULTS = [
    ({"ctrl": True, "key": "Enter"}, "Comfy.QueuePrompt"),
    ({"ctrl": True, "shift": True, "key": "Enter"}, "Comfy.QueuePromptFront"),
    ({"ctrl": True, "alt": True, "key": "Enter"}, "Comfy.Interrupt"),
    ({"key": "r"}, "Comfy.RefreshNodeDefinitions"),
    ({"key": "w"}, "Workspace.ToggleSidebarTab.workflows"),
    ({"key": "n"}, "Workspace.ToggleSidebarTab.node-library"),
    ({"key": "m"}, "Workspace.ToggleSidebarTab.model-library"),
    ({"key": "a"}, "Workspace.ToggleSidebarTab.assets"),
    ({"alt": True, "key": "m"}, "Comfy.ToggleLinear"),
    ({"ctrl": True, "key": "s"}, "Comfy.SaveWorkflow"),
    ({"ctrl": True, "key": "o"}, "Comfy.OpenWorkflow"),
    ({"ctrl": True, "key": "g"}, "Comfy.Graph.GroupSelectedNodes"),
    ({"ctrl": True, "key": ","}, "Comfy.ShowSettingsDialog"),
    ({"alt": True, "key": "="}, "Comfy.Canvas.ZoomIn"),
    ({"alt": True, "shift": True, "key": "+"}, "Comfy.Canvas.ZoomIn"),
    ({"alt": True, "key": "+"}, "Comfy.Canvas.ZoomIn"),
    ({"alt": True, "key": "-"}, "Comfy.Canvas.ZoomOut"),
    ({"key": "."}, "Comfy.Canvas.FitView"),
    ({"key": "p"}, "Comfy.Canvas.ToggleSelected.Pin"),
    ({"alt": True, "key": "c"}, "Comfy.Canvas.ToggleSelectedNodes.Collapse"),
    ({"ctrl": True, "key": "b"}, "Comfy.Canvas.ToggleSelectedNodes.Bypass"),
    ({"ctrl": True, "key": "m"}, "Comfy.Canvas.ToggleSelectedNodes.Mute"),
    ({"ctrl": True, "key": "`"}, "Workspace.ToggleBottomPanelTab.logs-terminal"),
    ({"ctrl": True, "shift": True, "key": "e"}, "Comfy.Graph.ConvertToSubgraph"),
    ({"alt": True, "shift": True, "key": "m"}, "Comfy.Canvas.ToggleMinimap"),
    ({"ctrl": True, "shift": True, "key": "k"}, "Workspace.ToggleBottomPanel.Shortcuts"),
    ({"key": "v"}, "Comfy.Canvas.Unlock"),
    ({"key": "h"}, "Comfy.Canvas.Lock"),
    ({"key": "Escape"}, "Comfy.Graph.ExitSubgraph"),
    ({"ctrl": True, "key": "a"}, "Comfy.Canvas.SelectAll"),
    ({"ctrl": True, "shift": True, "key": "v"}, "Comfy.Canvas.PasteFromClipboardWithConnect"),
    ({"key": "Delete"}, "Comfy.Canvas.DeleteSelectedItems"),
    ({"key": "Backspace"}, "Comfy.Canvas.DeleteSelectedItems"),
]

COMMAND_ID = "Comfy.QueueSelectedOutputNodes"


def combo_key(c):
    return "|".join([
        str(bool(c.get("ctrl"))),
        str(bool(c.get("alt"))),
        str(bool(c.get("shift"))),
        str(c.get("key")),
    ])


# ---------------------------------------------------------------------------
# The frontend, as much of it as this file touches.
#
# createElement returns something the extension can set cssText/textContent on
# and append to; __text() flattens a tree back to the visible string, which is
# what the assertions are actually about.
# ---------------------------------------------------------------------------

STUBS = r"""
    globalThis.__settingValues = {};

    function makeEl(tag) {
        return {
            tagName: tag.toUpperCase(),
            style: { cssText: "" },
            textContent: "",
            children: [],
            appendChild: function (c) { this.children.push(c); return c; },
        };
    }

    var document = { createElement: makeEl };

    globalThis.__text = function (el) {
        if (!el) return "";
        if (el.children && el.children.length) {
            return el.children.map(globalThis.__text).join(" ");
        }
        return el.textContent;
    };

    var app = {
        ui: { settings: { getSettingValue: function (id) {
                  return globalThis.__settingValues[id];
              } } },
        registerExtension: function (ext) { globalThis.__ext = ext; },
    };
    globalThis.app = app;

    // Drop the settings API entirely, the way an older frontend would.
    globalThis.__breakSettings = function () { app.ui = undefined; };

    // Render the mirror row the way FormItem does: call the type function
    // with (name, setValue, value, attrs) and keep whatever element it hands
    // back. setValue is recorded so the test can prove it is never called.
    globalThis.__setterCalls = 0;
    globalThis.__render = function () {
        var s = (globalThis.__ext.settings || [])[0];
        return s.type(
            s.name,
            function () { globalThis.__setterCalls++; },
            globalThis.__settingValues[s.id],
            s.attrs,
        );
    };
"""


def js_context():
    """Evaluate the extension in quickjs with ComfyUI stubbed out.

    Evaluated whole, so a syntax error or stale identifier anywhere in the
    file fails here rather than surfacing as a key that quietly does nothing.
    """
    import quickjs

    path = os.path.join(PACK, "web", "bat_hotkeys.js")
    src = open(path, encoding="utf-8").read()
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(src)
    ctx.eval("if (!globalThis.__ext) throw new Error('no extension registered');")
    return ctx


def jrun(ctx, body):
    ctx.eval("globalThis.__ret = (function () {" + body + "})();")
    return json.loads(ctx.eval("JSON.stringify(globalThis.__ret === undefined "
                               "? null : globalThis.__ret)"))


def set_bindings(ctx, new_bindings=None, unset_bindings=None):
    ctx.eval(
        "globalThis.__settingValues['Comfy.Keybinding.NewBindings'] = "
        + json.dumps(new_bindings or [])
        + "; globalThis.__settingValues['Comfy.Keybinding.UnsetBindings'] = "
        + json.dumps(unset_bindings or [])
        + ";"
    )


# ---------------------------------------------------------------------------
# 1-3. The keybinding itself
# ---------------------------------------------------------------------------

def test_keybinding():
    ctx = js_context()
    got = jrun(ctx, """
        const kbs = globalThis.__ext.keybindings || [];
        return { name: globalThis.__ext.name, n: kbs.length, kb: kbs[0] || null };
    """)

    check("extension registers under a BAT name",
          isinstance(got["name"], str) and got["name"].startswith("BAT."), got)
    check("exactly one keybinding is shipped", got["n"] == 1, got)

    kb = got["kb"] or {}
    check("it binds core's Queue Selected Output Nodes command",
          kb.get("commandId") == COMMAND_ID, got)

    combo = kb.get("combo") or {}
    taken = {combo_key(c): cid for c, cid in CORE_DEFAULTS}
    clash = taken.get(combo_key(combo))
    check("the combo is free in core's default list",
          clash is None, f"collides with {clash}: {combo}")

    check("the combo carries a modifier, so text inputs do not swallow it",
          bool(combo.get("ctrl") or combo.get("alt") or combo.get("shift")),
          combo)
    check("shift is not the only modifier (isShiftOnly is text-reserved too)",
          not (combo.get("shift") and not combo.get("ctrl")
               and not combo.get("alt")),
          combo)


# ---------------------------------------------------------------------------
# 4. The settings row is shaped for the dialog
# ---------------------------------------------------------------------------

def test_setting_shape():
    ctx = js_context()
    got = jrun(ctx, """
        const s = (globalThis.__ext.settings || [])[0] || {};
        return {
            n: (globalThis.__ext.settings || []).length,
            id: s.id,
            typeIsFn: typeof s.type === "function",
            cat: s.category,
            name: s.name,
            tooltip: s.tooltip || "",
            hasOnChange: typeof s.onChange === "function",
        };
    """)

    check("exactly one setting, id BAT.Hotkeys.RunSelectedOutputs",
          got["n"] == 1 and got["id"] == "BAT.Hotkeys.RunSelectedOutputs", got)
    check("type is a render function — FormItem's custom-value path",
          got["typeIsFn"], got)
    check("it nests under the same 🦇 BAT panel as the rest of the pack",
          isinstance(got["cat"], list) and len(got["cat"]) >= 2
          and got["cat"][0] == "🦇 BAT", got)
    check("the row has a name",
          isinstance(got["name"], str) and len(got["name"]) > 0, got)
    check("the tooltip points at Settings → Keybinding, where it is editable",
          "Keybinding" in got["tooltip"], got)
    check("no onChange — nothing about this row is stored",
          not got["hasOnChange"], got)


# ---------------------------------------------------------------------------
# 5. The mirror tracks the live keybinding state
# ---------------------------------------------------------------------------

def test_mirror_default():
    ctx = js_context()
    set_bindings(ctx)
    got = jrun(ctx, """
        const el = globalThis.__render();
        return { text: globalThis.__text(el), setter: globalThis.__setterCalls };
    """)

    shipped = jrun(ctx, "return (globalThis.__ext.keybindings || [])[0].combo;")
    expected = " + ".join(
        [p for p in ("Ctrl" if shipped.get("ctrl") else None,
                     "Alt" if shipped.get("alt") else None,
                     "Shift" if shipped.get("shift") else None,
                     shipped.get("key")) if p]
    )

    check("with no user overrides the mirror shows the shipped default",
          got["text"] == expected, f"{got['text']!r} != {expected!r}")
    check("rendering never writes the setting value",
          got["setter"] == 0, got)


def test_mirror_after_rebind():
    ctx = js_context()
    shipped = jrun(ctx, "return (globalThis.__ext.keybindings || [])[0].combo;")

    # What core persists when the capture dialog rebinds a default: the
    # default goes into UnsetBindings, the new combo into NewBindings.
    set_bindings(
        ctx,
        new_bindings=[{"commandId": COMMAND_ID,
                       "combo": {"ctrl": True, "shift": True, "alt": True,
                                 "key": "Enter"}}],
        unset_bindings=[{"commandId": COMMAND_ID, "combo": shipped}],
    )
    got = jrun(ctx, "return globalThis.__text(globalThis.__render());")
    check("after a rebind the mirror shows only the user's combo",
          got == "Ctrl + Alt + Shift + Enter", repr(got))


def test_mirror_unset():
    ctx = js_context()
    shipped = jrun(ctx, "return (globalThis.__ext.keybindings || [])[0].combo;")
    set_bindings(ctx, unset_bindings=[{"commandId": COMMAND_ID,
                                       "combo": shipped}])
    got = jrun(ctx, "return globalThis.__text(globalThis.__render());")
    check("clearing the binding reads as Not bound, not as a blank row",
          got.strip() == "Not bound", repr(got))


def test_mirror_ignores_other_commands():
    ctx = js_context()
    # Someone rebinding an unrelated command onto a combo must not show up on
    # this row, and unsetting an unrelated command must not blank it.
    set_bindings(
        ctx,
        new_bindings=[{"commandId": "Comfy.QueuePrompt",
                       "combo": {"ctrl": True, "key": "Enter"}}],
        unset_bindings=[{"commandId": "Comfy.Interrupt",
                         "combo": {"alt": True, "key": "Enter"}}],
    )
    got = jrun(ctx, "return globalThis.__text(globalThis.__render());")
    check("other commands' bindings are filtered out by commandId",
          got != "Not bound" and "Ctrl" not in got, repr(got))


# ---------------------------------------------------------------------------
# 6. The render is unkillable
# ---------------------------------------------------------------------------

def test_render_survives_bad_input():
    cases = {
        "settings never loaded": None,
        "value is not a list": "not-a-list",
        "entries are junk": [None, 7, {"commandId": COMMAND_ID}],
    }
    for label, value in cases.items():
        ctx = js_context()
        if value is not None:
            ctx.eval(
                "globalThis.__settingValues['Comfy.Keybinding.NewBindings'] = "
                + json.dumps(value)
                + "; globalThis.__settingValues['Comfy.Keybinding.UnsetBindings'] = "
                + json.dumps(value) + ";"
            )
        got = jrun(ctx, """
            try {
                const el = globalThis.__render();
                return { threw: false, text: globalThis.__text(el) };
            } catch (e) { return { threw: true, msg: String(e) }; }
        """)
        check(f"render survives: {label}", not got["threw"], got)

    ctx = js_context()
    got = jrun(ctx, """
        globalThis.__breakSettings();
        try {
            const el = globalThis.__render();
            return { threw: false, text: globalThis.__text(el) };
        } catch (e) { return { threw: true, msg: String(e) }; }
    """)
    check("render survives: no settings API at all (older frontend)",
          not got["threw"], got)


def main():
    for fn in (test_keybinding, test_setting_shape, test_mirror_default,
               test_mirror_after_rebind, test_mirror_unset,
               test_mirror_ignores_other_commands,
               test_render_survives_bad_input):
        print()
        fn()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED")
        for f in _failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
