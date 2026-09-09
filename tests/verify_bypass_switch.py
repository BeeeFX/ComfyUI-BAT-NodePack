#!/usr/bin/env python
"""
Behavioural checks for Bat_BypassSwitch.

This node has no Python to test — it is a stub, by design — and no pixels to
compare. What it has is a pile of graph-mutating logic in
``web/bat_bypass_switch.js`` whose failure modes are all silent and all
destructive to somebody's template: a toggle that bypasses the wrong nodes, a
preset list that loses its names on reload, a shared node left bypassed by the
loser of an exclusive group.

There is no node binary on the render machines, so the extension is evaluated
through quickjs against a stubbed canvas, DOM and graph, and driven the way the
browser drives it. Seven claims are load-bearing enough to be worth a test
rather than an argument:

1. **The state normaliser is total.** The preset JSON lives in a workflow file
   that gets hand-edited, copied between machines and loaded by older and newer
   versions of the pack. Anything it is handed must come back as a usable state
   object, because the alternative is a throw inside configure() that takes the
   whole graph load down with it.

2. **A backdrop is resolved live.** The documented promise is that dragging a
   node into a captured backdrop puts it under that toggle with no re-capture.

3. **Missing targets are skipped, not fatal.** Deleting a node a toggle points
   at must leave the toggle working on the rest.

4. **Exclusive groups apply off-before-on.** Two toggles in one group can share
   a target node; the order the modes are written in decides whether that
   shared node ends up enabled or bypassed, and only one of those answers is
   defensible.

5. **The widget layout follows the preset order,** with an exclusive group
   drawing once at its first member's position — the rule the edit dialog's
   reorder arrows are documented against.

6. **Rebuilt widgets do not inherit stale values.** The frontend's widget-value
   store is keyed by (graph, node, widget name) and outlives the widget, so a
   rebuild adopts the *previous* value unless the code re-asserts it. Ctrl+Z is
   a full graph reload, so this fires constantly in real use. The stub store
   here reproduces the trap deliberately.

7. **Only the state widget serialises.** litegraph writes widgets_values at
   each widget's full array index, so a dynamic widget that serialised would
   punch a hole into the array and corrupt the saved JSON.

    pip install quickjs
    python tests/verify_bypass_switch.py
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
# The browser, as much of it as this file touches
# ---------------------------------------------------------------------------

STUBS = r"""
    globalThis.__registered = null;
    globalThis.__log = [];
    var console = {
        log:   function () { globalThis.__log.push(["log", arguments[0]]); },
        info:  function () { globalThis.__log.push(["info", arguments[0]]); },
        warn:  function () { globalThis.__log.push(["warn", arguments[0]]); },
        error: function () { globalThis.__log.push(["error", arguments[0]]); },
    };

    function stubEl(tag) {
        var e = {
            tagName: (tag || "div").toUpperCase(),
            className: "", innerHTML: "", tabIndex: 0,
            type: "", value: "", placeholder: "", disabled: false,
            children: [], _listeners: {}, _removed: false,
            style: { display: "", setProperty: function () {}, cssText: "" },
            classList: { add: function () {}, remove: function () {},
                         toggle: function () {}, contains: function () { return false; } },
            append: function () {
                for (var i = 0; i < arguments.length; i++) this.children.push(arguments[i]);
            },
            appendChild: function (c) { this.children.push(c); return c; },
            remove: function () { this._removed = true; },
            addEventListener: function (t, fn) {
                (this._listeners[t] = this._listeners[t] || []).push(fn);
            },
            removeEventListener: function () {},
            focus: function () {}, select: function () {}, blur: function () {},
            setAttribute: function () {}, contains: function () { return false; },
            // Test seam: fire the handlers a real click would.
            _fire: function (t, ev) {
                var l = this._listeners[t] || [];
                for (var i = 0; i < l.length; i++) l[i](ev || {
                    preventDefault: function () {}, stopPropagation: function () {},
                    stopImmediatePropagation: function () {},
                });
            },
        };
        // textContent has to be a real accessor, not a field: the edit dialog
        // re-renders by assigning "" to a container, and a stub that kept the
        // old children would leave the test clicking buttons from the previous
        // render — which is exactly the bug that hid a real one here.
        var _text = "";
        Object.defineProperty(e, "textContent", {
            get: function () { return _text; },
            set: function (v) { _text = v; e.children.length = 0; },
            configurable: true, enumerable: true,
        });
        return e;
    }
    // Depth-first search for a live element by its text, so a test can press
    // the button an artist would press instead of calling the handler direct.
    globalThis.findByText = function (root, text) {
        if (!root || root._removed) return null;
        if (root.textContent === text) return root;
        for (var i = 0; i < (root.children || []).length; i++) {
            var hit = globalThis.findByText(root.children[i], text);
            if (hit) return hit;
        }
        return null;
    };
    globalThis.findInput = function (root, out) {
        out = out || [];
        if (root && !root._removed) {
            if (root.tagName === "INPUT") out.push(root);
            for (var i = 0; i < (root.children || []).length; i++)
                globalThis.findInput(root.children[i], out);
        }
        return out;
    };
    globalThis.click = function (root, text) {
        var b = globalThis.findByText(root, text);
        if (!b) throw new Error("no element labelled " + JSON.stringify(text));
        b._fire("click");
        return b;
    };
    var document = {
        getElementById: function () { return null; },
        createElement: function (t) { return stubEl(t); },
        createTextNode: function (t) { var e = stubEl("span"); e.textContent = t; return e; },
        head: stubEl(), body: stubEl(),
    };
    globalThis.__winKeys = [];
    var window = {
        addEventListener: function (t, fn) {
            if (t === "keydown") globalThis.__winKeys.push(fn);
        },
        removeEventListener: function (t, fn) {
            var i = globalThis.__winKeys.indexOf(fn);
            if (i !== -1) globalThis.__winKeys.splice(i, 1);
        },
        prompt: function () { return globalThis.__promptAnswer || null; },
    };
    // Deliver a key the way the browser would, to every capture listener.
    globalThis.pressKey = function (key) {
        var ev = { key: key, target: { tagName: "CANVAS" },
                   preventDefault: function () {},
                   stopPropagation: function () {},
                   stopImmediatePropagation: function () {} };
        var l = globalThis.__winKeys.slice();
        for (var i = 0; i < l.length; i++) l[i](ev);
    };
    globalThis.__timeouts = [];
    var setTimeout = function (fn) { globalThis.__timeouts.push(fn); return 1; };
    var clearTimeout = function () {};
    var setInterval = function () { return 2; };
    var clearInterval = function () {};

    // --- graph -------------------------------------------------------------
    globalThis.select = function (items) { app.canvas.selectedItems = items; };
    var app = {
        registerExtension: function (e) { globalThis.__registered = e; },
        extensionManager: null,
        canvas: {
            selectedItems: [],
            deselectAll: function () {}, selectItems: function () {},
            setDirty: function () {}, setDirtyCanvas: function () {},
            emitBeforeChange: function () {}, emitAfterChange: function () {},
        },
    };

    globalThis.makeGraph = function () {
        var byId = {};
        var g = {
            _byId: byId, groups: [], nodes: [],
            getNodeById: function (id) { return byId[String(id)] || null; },
            beforeChange: function () {}, afterChange: function () {},
            setDirtyCanvas: function () {}, change: function () {},
        };
        return g;
    };
    globalThis.addNode = function (graph, id, mode, type) {
        var n = { id: id, type: type || "SomeNode", mode: mode || 0,
                  graph: graph, setDirtyCanvas: function () {} };
        graph._byId[String(id)] = n;
        graph.nodes.push(n);
        return n;
    };
    // A backdrop. recomputeInsideNodes() re-reads a `members` thunk, which is
    // how the "resolved live" claim is exercised: the test changes what the
    // thunk returns and expects the toggle to follow.
    globalThis.addGroup = function (graph, id, title, membersFn) {
        var g = {
            id: id, title: title, _nodes: [],
            recomputeInsideNodes: function () { this._nodes = membersFn(); },
        };
        graph.groups.push(g);
        return g;
    };

    // The frontend's widget-value store, reproduced with its actual defect:
    // registration is keyed by name+type and returns whatever the last widget
    // of that name+type held, even after the widget itself was removed.
    globalThis.__store = {};
    globalThis.makeSwitch = function (graph, id, presetsJson) {
        var node = {
            id: id, type: "Bat_BypassSwitch", graph: graph, mode: 0,
            size: [260, 100], widgets: [],
            addCustomWidget: function (plain) {
                var key = plain.name + "|" + plain.type;
                var backing = (key in globalThis.__store)
                    ? globalThis.__store[key] : plain.value;
                globalThis.__store[key] = backing;
                delete plain.value;
                Object.defineProperty(plain, "value", {
                    get: function () { return globalThis.__store[key]; },
                    set: function (v) { globalThis.__store[key] = v; },
                    configurable: true, enumerable: true,
                });
                this.widgets.push(plain);
                return plain;
            },
            removeWidget: function (w) {
                var i = this.widgets.indexOf(w);
                if (i === -1) throw new Error("Widget not found on this node");
                this.widgets.splice(i, 1);
            },
            computeSize: function () { return [260, 26 * this.widgets.length + 30]; },
            setSize: function (s) { this.size = s; },
            setDirtyCanvas: function () {},
        };
        // Stands in for the Python-declared STRING widget: a plain data
        // property, present before anything dynamic is added, at index 0.
        node.widgets.push({ name: "presets", type: "text",
                            value: presetsJson || '{"v":1,"exgroups":[],"presets":[]}',
                            computeSize: function () { return [0, 20]; } });
        graph._byId[String(id)] = node;
        graph.nodes.push(node);
        return node;
    };

    // Readable dump of a node's widget layout, for the ordering assertions.
    globalThis.layoutOf = function (node) {
        return node.widgets.map(function (w) {
            return { name: w.name, label: w.label || null, type: w.type,
                     value: w.value === null ? null : w.value,
                     serialize: w.serialize === false ? false : true,
                     options: w.options && w.options.values
                        ? w.options.values : null };
        });
    };
    globalThis.modesOf = function (graph, ids) {
        return ids.map(function (i) {
            var n = graph.getNodeById(i);
            return n ? n.mode : -1;
        });
    };
"""


def js_context():
    """Evaluate the extension in quickjs with ComfyUI stubbed out.

    Evaluated whole rather than sliced: a syntax error or a stale identifier
    anywhere in the file — including the dialog code this test does not drive —
    fails here instead of showing up as a dead button in a browser.
    """
    import quickjs

    path = os.path.join(PACK, "web", "bat_bypass_switch.js")
    src = open(path, encoding="utf-8").read()
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(src)
    # The pack's other extensions are registered as a side effect of loading;
    # confirm ours was too, since a typo in registerExtension is otherwise a
    # node that simply never gains its widgets.
    ctx.eval("if (!globalThis.__registered) throw new Error('no extension registered');")
    return ctx


def jrun(ctx, body):
    """Run a JS body that returns a JSON-serialisable value."""
    ctx.eval("globalThis.__ret = (function () {" + body + "})();")
    return json.loads(ctx.eval("JSON.stringify(globalThis.__ret === undefined "
                               "? null : globalThis.__ret)"))


# ---------------------------------------------------------------------------
# 1. normaliseState is total
# ---------------------------------------------------------------------------

def test_normalise():
    ctx = js_context()

    cases = [
        ("null", 0, 0),
        ("undefined", 0, 0),
        ('"a string"', 0, 0),
        ("42", 0, 0),
        ("[]", 0, 0),
        ('{"presets": "not a list"}', 0, 0),
        ('{"presets": [null, 5, "x"]}', 0, 0),
        ('{"presets": [{}]}', 1, 0),
    ]
    bad = []
    for expr, npresets, ngroups in cases:
        got = jrun(ctx, f"""
            const s = normaliseState({expr});
            return [s.presets.length, s.exgroups.length, s.v];
        """)
        if got[:2] != [npresets, ngroups] or got[2] != 1:
            bad.append(f"{expr} -> {got}, wanted [{npresets}, {ngroups}, 1]")
    check(f"normaliseState survives junk input  [{len(cases)} cases]",
          not bad, "; ".join(bad))

    # An `ex` pointing nowhere must degrade to independent, not to a toggle
    # that never draws because it is waiting for a group header.
    got = jrun(ctx, """
        const s = normaliseState({presets: [{id: "p1", name: "A", ex: "ghost"}]});
        return s.presets[0].ex;
    """)
    check("unknown exclusive-group reference degrades to independent",
          got == "", f"got {got!r}")

    # Duplicate ids would give two widgets the same name, and the widget-value
    # store is keyed by name.
    got = jrun(ctx, """
        const s = normaliseState({presets: [{id: "p1", name: "A"},
                                            {id: "p1", name: "B"}]});
        return [s.presets.length, s.presets[0].name];
    """)
    check("duplicate preset ids are dropped", got == [1, "A"], f"got {got}")

    # A hand-edited file can say two members of one group are on. It is not a
    # state the group can be in, so it has to be resolved on read.
    got = jrun(ctx, """
        const s = normaliseState({
            exgroups: [{id: "g1", name: "Mode"}],
            presets: [{id: "p1", name: "A", ex: "g1", on: true},
                      {id: "p2", name: "B", ex: "g1", on: true}]});
        return s.presets.map(p => p.on);
    """)
    check("two active members of one exclusive group: first wins",
          got == [True, False], f"got {got}")

    got = jrun(ctx, """
        const s = normaliseState({exgroups: [{id: "g1", name: "Orphan"}],
                                  presets: [{id: "p1", name: "A"}]});
        return s.exgroups.length;
    """)
    check("an exclusive group nothing belongs to is dropped", got == 0,
          f"got {got}")

    # Round-trip through the widget, which is what save/load does.
    got = jrun(ctx, """
        const g = makeGraph();
        const n = makeSwitch(g, 1);
        writeState(n, {v: 1, exgroups: [{id: "g1", name: "Mode"}],
                       presets: [{id: "p1", name: "Vidéo — 2x", on: true,
                                  ex: "g1", nodes: [7, 8], groups: [3]}]});
        const back = readState(n);
        return [back.presets[0].name, back.presets[0].on,
                back.presets[0].nodes, back.presets[0].groups,
                back.exgroups[0].name,
                typeof n.widgets[0].value];
    """)
    check("state round-trips through the serialising widget",
          got == ["Vidéo — 2x", True, [7, 8], [3], "Mode", "string"],
          f"got {got}")


# ---------------------------------------------------------------------------
# 2 & 3. Resolution: live backdrops, missing targets, self-exclusion
# ---------------------------------------------------------------------------

def test_resolution():
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10), b = addNode(g, 11), c = addNode(g, 12);
        // A second switch, and this switch itself, must both be filtered out:
        // neither ever executes, so bypassing one only looks like it worked.
        const other = makeSwitch(g, 2);
        const info = resolveTargets(sw, {nodes: [10, 11, 12, 1, 2], groups: []});
        return [info.nodes.map(n => n.id).sort(), info.missing];
    """)
    check("resolve skips the switch itself and other switches",
          got == [[10, 11, 12], 0], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10);
        const info = resolveTargets(sw, {nodes: [10, 99, 98], groups: [77]});
        return [info.nodes.length, info.missingNodes, info.missingGroups,
                info.missing];
    """)
    check("deleted targets are counted, not fatal", got == [1, 2, 1, 3],
          f"got {got}")

    # The documented backdrop promise.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10), b = addNode(g, 11);
        let members = [a];
        addGroup(g, 5, "Face detail", () => members);
        const preset = {nodes: [], groups: [5]};
        const before = resolveTargets(sw, preset).nodes.map(n => n.id);
        members = [a, b];                       // artist drags b into the backdrop
        const after = resolveTargets(sw, preset).nodes.map(n => n.id);
        return [before, after];
    """)
    check("a backdrop's membership is resolved live, not snapshotted",
          got == [[10], [10, 11]], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10);
        addGroup(g, 5, "G", () => [a]);
        // 10 named twice, once directly and once via the backdrop.
        const info = resolveTargets(sw, {nodes: [10], groups: [5]});
        return info.nodes.length;
    """)
    check("a node named both directly and via a backdrop counts once",
          got == 1, f"got {got}")


# ---------------------------------------------------------------------------
# 4. Apply order
# ---------------------------------------------------------------------------

def test_apply():
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0); addNode(g, 11, 4); addNode(g, 12, 2);
        const p = {nodes: [10, 11, 12], groups: []};
        applyPreset(sw, p, false);
        const off = modesOf(g, [10, 11, 12]);
        applyPreset(sw, p, true);
        const on = modesOf(g, [10, 11, 12]);
        return [off, on];
    """)
    # A muted (2) node is forced to Always by ON, deliberately: a toggle that
    # sometimes leaves a node switched off would be untrustworthy.
    check("off writes mode 4 everywhere, on writes mode 0 everywhere",
          got == [[4, 4, 4], [0, 0, 0]], f"got {got}")

    # The claim: exclusive-group siblings are switched off BEFORE the winner is
    # switched on, so a node both of them control ends up enabled.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0);   // only in A
        addNode(g, 11, 0);   // shared by A and B  <- the interesting one
        addNode(g, 12, 4);   // only in B
        writeState(sw, {v: 1, exgroups: [{id: "g1", name: "Mode"}], presets: [
            {id: "pA", name: "A", on: true,  ex: "g1", nodes: [10, 11], groups: []},
            {id: "pB", name: "B", on: false, ex: "g1", nodes: [11, 12], groups: []},
        ]});
        buildWidgets(sw, true);
        setPresetOn(sw, "pB", true);
        const st = readState(sw);
        return [modesOf(g, [10, 11, 12]),
                st.presets.map(p => p.on)];
    """)
    check("exclusive switch leaves a shared target enabled (off applied first)",
          got == [[4, 0, 0], [False, True]], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0); addNode(g, 11, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "pA", name: "A", on: true, ex: "", nodes: [10], groups: []},
            {id: "pB", name: "B", on: true, ex: "", nodes: [11], groups: []},
        ]});
        buildWidgets(sw, true);
        setPresetOn(sw, "pA", false);
        return [modesOf(g, [10, 11]), readState(sw).presets.map(p => p.on)];
    """)
    check("independent toggles do not disturb each other",
          got == [[4, 0], [False, True]], f"got {got}")

    # reapplyAll resolves an overlap the same way, globally.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 4);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "pOff", name: "Off", on: false, ex: "", nodes: [10], groups: []},
            {id: "pOn",  name: "On",  on: true,  ex: "", nodes: [10], groups: []},
        ]});
        buildWidgets(sw, true);
        reapplyAll(sw);
        return modesOf(g, [10]);
    """)
    check("re-apply resolves an overlap in favour of enabled", got == [0],
          f"got {got}")


# ---------------------------------------------------------------------------
# 5, 6, 7. Widget layout, stale values, serialisation
# ---------------------------------------------------------------------------

STATE_MIXED = """
    {v: 1, exgroups: [{id: "g1", name: "Mode"}], presets: [
        {id: "p1", name: "Face detail", on: true,  ex: "",   nodes: [10], groups: []},
        {id: "p2", name: "Image",       on: false, ex: "g1", nodes: [11], groups: []},
        {id: "p3", name: "Upscale",     on: false, ex: "",   nodes: [12], groups: []},
        {id: "p4", name: "Video",       on: true,  ex: "g1", nodes: [13], groups: []},
    ]}
"""


def test_layout():
    ctx = js_context()

    got = jrun(ctx, f"""
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12, 13]) addNode(g, i);
        writeState(sw, {STATE_MIXED});
        buildWidgets(sw, true);
        return layoutOf(sw).map(w => [w.name, w.type, w.label]);
    """)
    want = [
        ["presets", "hidden", None],
        ["bsw_p1", "toggle", "Face detail"],
        # The exclusive group draws once, at its FIRST member's position, so it
        # lands between p1 and p3 rather than at the tail.
        ["bswex_g1", "combo", "Mode"],
        ["bsw_p3", "toggle", "Upscale"],
        ["bsw_add", "button", "＋ Add toggle…"],
        ["bsw_edit", "button", "✎ Edit toggles…"],
    ]
    check("widget order follows preset order; a group draws at its first member",
          got == want, f"got {got}")

    got = jrun(ctx, f"""
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12, 13]) addNode(g, i);
        writeState(sw, {STATE_MIXED});
        buildWidgets(sw, true);
        const combo = sw.widgets.find(w => w.name === "bswex_g1");
        return [combo.options.values, combo.value,
                sw.widgets.find(w => w.name === "bsw_p1").value];
    """)
    check("exclusive combo offers (none) plus its members, showing the active one",
          got == [["(none)", "Image", "Video"], "Video", True], f"got {got}")

    # Only the state widget may serialise: litegraph writes widgets_values at
    # each widget's full array index.
    got = jrun(ctx, f"""
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12, 13]) addNode(g, i);
        writeState(sw, {STATE_MIXED});
        buildWidgets(sw, true);
        return layoutOf(sw).map(w => w.serialize);
    """)
    check("only the state widget serialises",
          got and got[0] is True and all(s is False for s in got[1:]),
          f"got {got}")

    # The store trap, reproduced: the stub store hands a rebuilt widget the
    # previous widget's value. Without the post-add re-assignment in
    # addWidget(), these toggles would come back with the pre-flip values.
    got = jrun(ctx, f"""
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12, 13]) addNode(g, i);
        writeState(sw, {STATE_MIXED});
        buildWidgets(sw, true);
        const first = [sw.widgets.find(w => w.name === "bsw_p1").value,
                       sw.widgets.find(w => w.name === "bswex_g1").value];
        // Flip the saved state behind the widgets' backs — exactly what a
        // Ctrl+Z (a full loadGraphData) does — then rebuild.
        const st = readState(sw);
        st.presets[0].on = false;      // Face detail off
        st.presets[1].on = true;       // Mode -> Image
        st.presets[3].on = false;
        writeState(sw, st);
        buildWidgets(sw, true);
        const second = [sw.widgets.find(w => w.name === "bsw_p1").value,
                        sw.widgets.find(w => w.name === "bswex_g1").value];
        return [first, second];
    """)
    check("a rebuilt widget takes the new state, not the store's stale value",
          got == [[True, "Video"], [False, "Image"]], f"got {got}")

    # Duplicate member names inside one group, and a member literally called
    # "(none)": the combo keys its selection off the label string, so the
    # labels have to be made unique or one option becomes unreachable.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12]) addNode(g, i);
        writeState(sw, {v: 1, exgroups: [{id: "g1", name: "Mode"}], presets: [
            {id: "p1", name: "Pass",   on: false, ex: "g1", nodes: [10], groups: []},
            {id: "p2", name: "Pass",   on: true,  ex: "g1", nodes: [11], groups: []},
            {id: "p3", name: "(none)", on: false, ex: "g1", nodes: [12], groups: []},
        ]});
        buildWidgets(sw, true);
        const combo = sw.widgets.find(w => w.name === "bswex_g1");
        const vals = combo.options.values;
        const unique = new Set(vals).size === vals.length;
        // The active one is the SECOND "Pass"; its label must resolve back to p2.
        return [vals, combo.value, unique, combo._bswMap.get(combo.value)];
    """)
    check("colliding option labels are uniquified and still map to their preset",
          got == [["(none)", "Pass", "Pass (2)", "(none) (2)"], "Pass (2)",
                  True, "p2"], f"got {got}")

    # An empty switch offers only the add button, and says so.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        buildWidgets(sw, true);
        return layoutOf(sw).map(w => [w.name, w.label]);
    """)
    check("an empty switch shows one inviting button",
          got == [["presets", None], ["bsw_add", "＋ Add first toggle…"]],
          f"got {got}")


# ---------------------------------------------------------------------------
# The load path: no false ⚠, and no node modes touched
# ---------------------------------------------------------------------------

def test_load_path():
    ctx = js_context()

    # Mid-load the rest of the graph does not exist yet. Painting ⚠ on every
    # toggle at that moment would make every loaded template look broken.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "Face detail", on: true, ex: "", nodes: [10], groups: []}]});
        buildWidgets(sw, false);                    // node 10 not added yet
        const during = sw.widgets.find(w => w.name === "bsw_p1").label;
        addNode(g, 10);                             // …the graph finishes loading
        refreshWidgets(sw);
        const after = sw.widgets.find(w => w.name === "bsw_p1").label;
        return [during, after];
    """)
    check("no false ⚠ while the graph is still loading",
          got == ["Face detail", "Face detail"], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "Gone", on: true, ex: "", nodes: [10, 99], groups: []}]});
        addNode(g, 10);
        buildWidgets(sw, true);
        return sw.widgets.find(w => w.name === "bsw_p1").label;
    """)
    check("a toggle with a deleted target is marked ⚠", got == "Gone  ⚠",
          f"got {got!r}")

    # The load-behaviour decision, asserted: opening a workflow must not
    # rewrite node modes, however much they disagree with the toggles.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 4);      // hand-bypassed after the toggle was set
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "A", on: true, ex: "", nodes: [10], groups: []}]});
        buildWidgets(sw, false);
        refreshWidgets(sw);
        const untouched = modesOf(g, [10]);
        reapplyAll(sw);         // …until you ask for it
        return [untouched, modesOf(g, [10])];
    """)
    check("loading does not touch node modes; re-apply does",
          got == [[4], [0]], f"got {got}")


def test_duplicate_detection():
    """A pasted switch is the one silent failure left; it must not stay silent."""
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const orig = makeSwitch(g, 1);
        const copy = makeSwitch(g, 5);       // higher id = the pasted one
        addNode(g, 10);
        const state = {v: 1, exgroups: [], presets: [
            {id: "p1", name: "A", on: true, ex: "", nodes: [10], groups: []}]};
        writeState(orig, state);
        writeState(copy, state);             // paste keeps the same preset ids
        return [duplicateSwitch(copy) ? duplicateSwitch(copy).id : null,
                duplicateSwitch(orig)];
    """)
    check("the newer of two switches sharing preset ids is flagged, not the older",
          got == [1, None], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const a = makeSwitch(g, 1);
        const b = makeSwitch(g, 5);
        addNode(g, 10); addNode(g, 11);
        writeState(a, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "A", on: true, ex: "", nodes: [10], groups: []}]});
        writeState(b, {v: 1, exgroups: [], presets: [
            {id: "p2", name: "B", on: true, ex: "", nodes: [11], groups: []}]});
        return [duplicateSwitch(a), duplicateSwitch(b)];
    """)
    check("two independent switches in one graph are not mistaken for copies",
          got == [None, None], f"got {got}")

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "A", on: true, ex: "", nodes: [10], groups: []}]});
        sw._bswDuplicate = true;
        buildWidgets(sw, true);
        return sw.widgets.find(w => w.name === "bsw_p1").label;
    """)
    check("a flagged copy marks every toggle ⚠", got == "A  ⚠", f"got {got!r}")


# ---------------------------------------------------------------------------
# The flows an artist actually performs, end to end
# ---------------------------------------------------------------------------

def test_capture_flow():
    """＋ Add toggle… → select → Enter → name → Add toggle.

    Driven through the real buttons and the real keypress rather than by
    calling the handlers, because the wiring between them is most of what can
    break: a banner button bound to nothing, an Enter that never reaches
    confirmSelection, a naming dialog whose Add does not commit.
    """
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10, 0), b = addNode(g, 11, 0);
        let members = [b];
        const grp = addGroup(g, 5, "Face detail", () => members);
        buildWidgets(sw, true);

        // Press the node's own button.
        sw.widgets.find(w => w.name === "bsw_add").callback();
        const banner = document.body.children[document.body.children.length - 1];
        const inSelection = !!banner && banner.className === "bat-bsw-banner";

        // Pick a node and a backdrop, then confirm with the keyboard.
        select([a, grp]);
        pressKey("Enter");

        // The naming dialog is now up; take its default name.
        const overlay = document.body.children[document.body.children.length - 1];
        const isOverlay = overlay.className === "bat-bsw-overlay";
        click(overlay, "Add toggle");

        const st = readState(sw);
        return [inSelection, isOverlay, st.presets.length,
                st.presets[0].nodes, st.presets[0].groups, st.presets[0].on,
                sw.widgets.map(w => w.name)];
    """)
    check("add-toggle flow: button → select → Enter → dialog → committed preset",
          got[:6] == [True, True, 1, [10], [5], True], f"got {got[:6]}")
    check("the captured toggle draws as a widget and unlocks Edit",
          len(got[6]) == 4 and got[6][0] == "presets"
          and got[6][1].startswith("bsw_")
          and got[6][2:] == ["bsw_add", "bsw_edit"], f"got {got[6]}")

    # Capturing must not move a single node: the toggle describes the graph as
    # it already is.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10, 4), b = addNode(g, 11, 4);   // both bypassed
        buildWidgets(sw, true);
        sw.widgets.find(w => w.name === "bsw_add").callback();
        select([a, b]);
        pressKey("Enter");
        const overlay = document.body.children[document.body.children.length - 1];
        click(overlay, "Add toggle");
        return [modesOf(g, [10, 11]), readState(sw).presets[0].on];
    """)
    check("capturing an all-bypassed selection starts the toggle off, changing nothing",
          got == [[4, 4], False], f"got {got}")

    # Esc must abandon cleanly and leave no preset behind.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10, 0);
        buildWidgets(sw, true);
        sw.widgets.find(w => w.name === "bsw_add").callback();
        select([a]);
        pressKey("Escape");
        // The key listener must be gone, or the next Enter anywhere would
        // confirm a session that no longer exists.
        return [readState(sw).presets.length, globalThis.__winKeys.length,
                modesOf(g, [10])];
    """)
    check("Escape cancels selection mode and unhooks its key listener",
          got == [0, 0, [0]], f"got {got}")

    # Confirming an empty selection must not create an empty toggle.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        buildWidgets(sw, true);
        sw.widgets.find(w => w.name === "bsw_add").callback();
        select([]);
        pressKey("Enter");
        const stillSelecting = globalThis.__winKeys.length === 1;
        pressKey("Escape");
        return [readState(sw).presets.length, stillSelecting];
    """)
    check("confirming nothing keeps you in selection mode instead of making an empty toggle",
          got == [0, True], f"got {got}")

    # A selection of nothing but switches is the same non-answer.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const other = makeSwitch(g, 2);
        buildWidgets(sw, true);
        sw.widgets.find(w => w.name === "bsw_add").callback();
        select([sw, other]);
        pressKey("Enter");
        const stillSelecting = globalThis.__winKeys.length === 1;
        pressKey("Escape");
        return [readState(sw).presets.length, stillSelecting];
    """)
    check("a selection of only Bypass Switches is refused", got == [0, True],
          f"got {got}")


def test_edit_dialog():
    """Rename, reorder, group, delete, clean up, Save and Cancel."""
    ctx = js_context()

    setup = """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        for (const i of [10, 11, 12]) addNode(g, i, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "One",   on: true,  ex: "", nodes: [10], groups: []},
            {id: "p2", name: "Two",   on: false, ex: "", nodes: [11], groups: []},
            {id: "p3", name: "Three", on: true,  ex: "", nodes: [12], groups: []},
        ]});
        buildWidgets(sw, true);
        openEditDialog(sw);
        const panel = document.body.children[document.body.children.length - 1]
                        .children[0];
    """

    got = jrun(ctx, setup + """
        // Rename the first row, then Save.
        const inputs = findInput(panel);
        inputs[0].value = "Renamed";
        inputs[0]._fire("input");
        click(panel, "Save");
        const st = readState(sw);
        return [st.presets.map(p => p.name),
                sw.widgets.filter(w => w.label).map(w => w.label)];
    """)
    check("rename in the dialog reaches the state and the widget label",
          got == [["Renamed", "Two", "Three"],
                  ["Renamed", "Two", "Three", "＋ Add toggle…", "✎ Edit toggles…"]],
          f"got {got}")

    got = jrun(ctx, setup + """
        const inputs = findInput(panel);
        inputs[0].value = "Renamed";
        inputs[0]._fire("input");
        click(panel, "Cancel");
        return readState(sw).presets.map(p => p.name);
    """)
    check("Cancel discards the edits", got == ["One", "Two", "Three"],
          f"got {got}")

    got = jrun(ctx, setup + """
        // ▼ on the first row, then Save. Order is draw order.
        click(panel, "▼");
        const panel2 = document.body.children[document.body.children.length - 1]
                         .children[0];
        click(panel2, "Save");
        return readState(sw).presets.map(p => p.name);
    """)
    check("the ▼ arrow reorders the toggles", got == ["Two", "One", "Three"],
          f"got {got}")

    got = jrun(ctx, setup + """
        click(panel, "✕");
        const panel2 = document.body.children[document.body.children.length - 1]
                         .children[0];
        click(panel2, "Save");
        // Deleting a toggle must not un-bypass or bypass anything.
        return [readState(sw).presets.map(p => p.name), modesOf(g, [10, 11, 12])];
    """)
    check("✕ deletes the toggle and leaves its nodes exactly as they were",
          got == [["Two", "Three"], [0, 0, 0]], f"got {got}")

    # Grouping two live toggles: normalisation has to resolve the two-on state,
    # and the graph has to be brought in line with whichever one lost.
    got = jrun(ctx, setup + """
        globalThis.__promptAnswer = "Mode";
        const sels = [];
        (function walk(n) {
            if (!n || n._removed) return;
            if (n.tagName === "SELECT") sels.push(n);
            (n.children || []).forEach(walk);
        })(panel);
        // Row 1 and row 3 are both ON; put both into a new exclusive group.
        sels[0].value = "__new__"; sels[0]._fire("change");
        let p2 = document.body.children[document.body.children.length - 1].children[0];
        const sels2 = [];
        (function walk(n) {
            if (!n || n._removed) return;
            if (n.tagName === "SELECT") sels2.push(n);
            (n.children || []).forEach(walk);
        })(p2);
        const gid = sels2[2].children.map(o => o.value).find(v => v && v !== "__new__");
        sels2[2].value = gid; sels2[2]._fire("change");
        let p3 = document.body.children[document.body.children.length - 1].children[0];
        click(p3, "Save");
        const st = readState(sw);
        return [st.exgroups.map(x => x.name),
                st.presets.map(p => [p.name, p.ex ? "grouped" : "solo", p.on]),
                modesOf(g, [10, 11, 12]),
                sw.widgets.map(w => w.name)];
    """)
    check("grouping two active toggles resolves to one winner and bypasses the loser",
          got[0] == ["Mode"]
          and got[1] == [["One", "grouped", True], ["Two", "solo", False],
                         ["Three", "grouped", False]]
          and got[2] == [0, 0, 4], f"got {got[:3]}")
    check("the new group draws as one combo where its first member was",
          got[3][0] == "presets" and got[3][1].startswith("bswex_")
          and got[3][2] == "bsw_p2"
          and got[3][3:] == ["bsw_add", "bsw_edit"], f"got {got[3]}")

    # Clean-up of dead ids.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "One", on: true, ex: "",
             nodes: [10, 98, 99], groups: [77]}]});
        buildWidgets(sw, true);
        openEditDialog(sw);
        let panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Clean up 3 missing targets");
        panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Save");
        const p = readState(sw).presets[0];
        return [p.nodes, p.groups,
                sw.widgets.find(w => w.name === "bsw_p1").label];
    """)
    check("clean-up drops dead ids and clears the ⚠",
          got == [[10], [], "One"], f"got {got}")

    # Ungrouping puts the members back to independent toggles.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0); addNode(g, 11, 4);
        writeState(sw, {v: 1, exgroups: [{id: "g1", name: "Mode"}], presets: [
            {id: "p1", name: "A", on: true,  ex: "g1", nodes: [10], groups: []},
            {id: "p2", name: "B", on: false, ex: "g1", nodes: [11], groups: []}]});
        buildWidgets(sw, true);
        openEditDialog(sw);
        let panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Ungroup");
        panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Save");
        const st = readState(sw);
        return [st.exgroups.length, st.presets.map(p => p.ex),
                sw.widgets.map(w => w.name), modesOf(g, [10, 11])];
    """)
    check("Ungroup returns the members to independent toggles, graph untouched",
          got == [0, ["", ""], ["presets", "bsw_p1", "bsw_p2", "bsw_add",
                                "bsw_edit"], [0, 4]], f"got {got}")


def test_recapture_flow():
    """Re-select… must detour through selection mode and come back with the draft."""
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const a = addNode(g, 10, 0), b = addNode(g, 11, 0), c = addNode(g, 12, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "One", on: true, ex: "", nodes: [10], groups: []}]});
        buildWidgets(sw, true);
        openEditDialog(sw);
        let panel = document.body.children[document.body.children.length - 1].children[0];

        // Type a rename FIRST, then detour: the rename must survive the trip.
        const inputs = findInput(panel);
        inputs[0].value = "Kept";
        inputs[0]._fire("input");
        click(panel, "Re-select…");

        const banner = document.body.children[document.body.children.length - 1];
        const inSelection = banner.className === "bat-bsw-banner";
        select([b, c]);
        pressKey("Enter");

        // Back in the dialog, on the same draft.
        panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Save");
        const p = readState(sw).presets[0];
        return [inSelection, p.name, p.nodes];
    """)
    check("Re-select… captures new targets and keeps unsaved renames",
          got == [True, "Kept", [11, 12]], f"got {got}")

    # Cancelling the detour must leave the targets alone but still come back.
    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 0); addNode(g, 11, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "One", on: true, ex: "", nodes: [10], groups: []}]});
        buildWidgets(sw, true);
        openEditDialog(sw);
        let panel = document.body.children[document.body.children.length - 1].children[0];
        click(panel, "Re-select…");
        pressKey("Escape");
        panel = document.body.children[document.body.children.length - 1].children[0];
        const backInDialog = panel.className === "bat-bsw-panel";
        click(panel, "Save");
        return [backInDialog, readState(sw).presets[0].nodes];
    """)
    check("cancelling a re-select returns to the dialog with the old targets",
          got == [True, [10]], f"got {got}")


def test_refused_recapture():
    """A Re-select… that cannot start must not swallow the unsaved draft."""
    ctx = js_context()

    got = jrun(ctx, """
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        const other = makeSwitch(g, 2);
        addNode(g, 10, 0);
        writeState(sw, {v: 1, exgroups: [], presets: [
            {id: "p1", name: "One", on: true, ex: "", nodes: [10], groups: []}]});
        buildWidgets(sw, true);
        // Something else is already selecting.
        buildWidgets(other, true);
        other.widgets.find(w => w.name === "bsw_add").callback();

        openEditDialog(sw);
        let panel = document.body.children[document.body.children.length - 1].children[0];
        const inputs = findInput(panel);
        inputs[0].value = "Kept";
        inputs[0]._fire("input");
        click(panel, "Re-select…");

        panel = document.body.children[document.body.children.length - 1].children[0];
        const backInDialog = panel.className === "bat-bsw-panel";
        click(panel, "Save");
        pressKey("Escape");
        return [backInDialog, readState(sw).presets[0].name];
    """)
    check("a refused re-select re-opens the dialog with the draft intact",
          got == [True, "Kept"], f"got {got}")


def test_menu_and_hooks():
    """The prototype hooks the extension installs, driven as the frontend does."""
    ctx = js_context()

    got = jrun(ctx, """
        const proto = {};
        const nodeType = { prototype: proto };
        // beforeRegisterNodeDef must ignore every other node in the pack.
        globalThis.__registered.beforeRegisterNodeDef(
            { prototype: {} }, { name: "Bat_Crop" });
        globalThis.__registered.beforeRegisterNodeDef(nodeType,
            { name: "Bat_BypassSwitch" });
        const installed = ["onNodeCreated", "onConfigure", "onRemoved",
                           "getExtraMenuOptions"].filter(k => typeof proto[k] === "function");

        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        addNode(g, 10, 4);
        Object.assign(sw, proto);
        sw.onNodeCreated();
        const freshLayout = sw.widgets.map(w => w.name);

        // Now the frontend restores the saved value and calls onConfigure.
        sw.widgets[0].value = JSON.stringify({v: 1, exgroups: [], presets: [
            {id: "p1", name: "Loaded", on: true, ex: "", nodes: [10], groups: []}]});
        sw.onConfigure({});
        const loadedLayout = sw.widgets.map(w => w.name);
        const modeAfterLoad = modesOf(g, [10]);

        // The deferred pass the ⚠ markers and copy-detection ride on.
        globalThis.__timeouts.splice(0).forEach(fn => fn());

        const opts = [];
        sw.getExtraMenuOptions({}, opts);
        const entries = opts.filter(o => o).map(o => [o.content, !!o.disabled]);

        // Re-apply from the menu.
        opts.filter(o => o && o.content.indexOf("Re-apply") !== -1)[0].callback();
        return [installed, freshLayout, loadedLayout, modeAfterLoad,
                entries, modesOf(g, [10]), sw.color !== undefined];
    """)
    check("beforeRegisterNodeDef installs all four hooks and skips other nodes",
          got[0] == ["onNodeCreated", "onConfigure", "onRemoved",
                     "getExtraMenuOptions"], f"got {got[0]}")
    check("a fresh node shows only the add button; onConfigure builds the saved toggle",
          got[1] == ["presets", "bsw_add"]
          and got[2] == ["presets", "bsw_p1", "bsw_add", "bsw_edit"],
          f"got {got[1]} then {got[2]}")
    check("onConfigure leaves the hand-bypassed target alone", got[3] == [4],
          f"got {got[3]}")
    check("the right-click menu offers add / edit / re-apply, all enabled",
          got[4] == [["🦇 Add toggle…", False], ["🦇 Edit toggles…", False],
                     ["🦇 Re-apply all toggles to the graph", False]],
          f"got {got[4]}")
    check("the menu's re-apply is what finally asserts the toggle", got[5] == [0],
          f"got {got[5]}")

    # A node deleted mid-selection must not leave the banner up.
    got = jrun(ctx, """
        const proto = {};
        globalThis.__registered.beforeRegisterNodeDef({ prototype: proto },
            { name: "Bat_BypassSwitch" });
        const g = makeGraph();
        const sw = makeSwitch(g, 1);
        Object.assign(sw, proto);
        sw.onNodeCreated();
        sw.widgets.find(w => w.name === "bsw_add").callback();
        const during = globalThis.__winKeys.length;
        sw.onRemoved();
        return [during, globalThis.__winKeys.length];
    """)
    check("deleting the node mid-selection tears the session down", got == [1, 0],
          f"got {got}")

    # Two sessions at once would fight over one banner and one canvas selection.
    got = jrun(ctx, """
        const g = makeGraph();
        const a = makeSwitch(g, 1), b = makeSwitch(g, 2);
        buildWidgets(a, true); buildWidgets(b, true);
        a.widgets.find(w => w.name === "bsw_add").callback();
        b.widgets.find(w => w.name === "bsw_add").callback();
        const keys = globalThis.__winKeys.length;
        pressKey("Escape");
        return [keys, globalThis.__winKeys.length];
    """)
    check("a second selection request is refused, not queued", got == [1, 0],
          f"got {got}")


def main():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        print("SKIP  every check needs quickjs — pip install quickjs")
        return 0

    test_normalise()
    test_resolution()
    test_apply()
    test_layout()
    test_load_path()
    test_duplicate_detection()
    test_capture_flow()
    test_edit_dialog()
    test_recapture_flow()
    test_refused_recapture()
    test_menu_and_hooks()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
