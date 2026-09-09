#!/usr/bin/env python
"""
Checks for web/bat_path_widget.js — the path field shared by 🦇 Video Loader
and 🦇 Frame Picker.

Everything here is a bug that leaves the node *looking* fine, which is why it
gets a test:

1. **The pill has to span the node.** A canvas widget is drawn (and hit-tested)
   at `widget.width || node.size[0]`, and the Vue legacy widget component
   stamps `widget.width` with whatever container last drew it. Stamped, the
   pill freezes at that width and clicks land beside it. `unpinWidgetWidth`
   has to keep the property unset no matter who writes it.

2. **The value is fitted by measurement.** A fixed character cut wastes a wide
   node and overflows a narrow one, and it has to trim from the front — the
   filename is the end you need to read.

3. **Picking a path notifies the node.** Dependent state (the thumbnail fetch,
   the contact sheet) hangs off the change, and assigning `widget.value`
   raises nothing on its own.

The JS runs under quickjs — there is no node binary on these machines.

    pip install quickjs
    python tests/verify_path_widget.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


STUBS = r"""
var LiteGraph = {
    WIDGET_OUTLINE_COLOR: "#111",
    WIDGET_BGCOLOR: "#222",
    WIDGET_TEXT_COLOR: "#ddd",
    WIDGET_SECONDARY_TEXT_COLOR: "#999",
    NODE_WIDGET_HEIGHT: 20,
};
var app = { canvas: { ds: { scale: 1 }, canvas: { getBoundingClientRect() { return {left:0, top:0}; } } } };
var api = { apiURL(u) { return u; } };
var markBatWidget = function (w) { return w; };
var fetch = function () { return Promise.reject(new Error("no network")); };
var document = { createElement() { return { style: {}, classList: { add(){} }, append(){}, appendChild(){}, querySelector(){ return null; }, addEventListener(){}, remove(){} }; },
                 body: { append(){} } };

// A canvas 2D context whose text metrics are a simple 7px-per-character model:
// enough to drive the fitting maths deterministically.
globalThis.__drawn = [];
function makeCtx(isGraph) {
    return {
        canvas: isGraph ? app.canvas.canvas : {},
        textAlign: "left", strokeStyle: "", fillStyle: "",
        measureText(t) { return { width: t.length * 7 }; },
        beginPath(){}, roundRect(){}, fill(){}, stroke(){}, save(){}, restore(){},
        rect(){}, clip(){},
        fillText(t, x, y) { globalThis.__drawn.push({ t, x, y }); },
    };
}
"""


def _ctx():
    import quickjs

    ctx = quickjs.Context()
    ctx.eval(STUBS)

    # The real unpinWidgetWidth, not a stub: whether the property actually
    # stays unset is claim 1, so a faithful copy in the test would be testing
    # the copy.
    for name in ("bat_node_layout.js", "bat_path_widget.js"):
        src = open(os.path.join(PACK, "web", name), encoding="utf-8").read()
        src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
        src = re.sub(r"^export ", "", src, flags=re.M)
        ctx.eval(src)
    return ctx


def test_width_unpinned():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        print("SKIP  path widget JS — pip install quickjs to run it")
        return

    ctx = _ctx()
    ctx.eval("""
        globalThis.w = makeBatPathWidget({
            name: "path", value: "/jobs/sh010/plate.mov",
            options: { bat_path_extensions: "mov,mp4" },
            route: "/bat/getpath", title: "Video Path",
        });
        // WidgetLegacy.vue does exactly this on every draw.
        globalThis.w.width = 490;
    """)
    check("widget.width stays unset after a stamp",
          ctx.eval("typeof globalThis.w.width === 'undefined'"),
          str(ctx.eval("String(globalThis.w.width)")))
    check("the stamp does not throw", ctx.eval("globalThis.w.value") ==
          "/jobs/sh010/plate.mov")
    check("widget type is BAT.PATH", ctx.eval("globalThis.w.type") == "BAT.PATH")

    # Drawn with a real width, the pill spans it; handed nothing, it falls back
    # to the node's own width.
    ctx.eval("""
        globalThis.node = { size: [640, 200], _changed: [], _dirty: 0,
            onWidgetChanged(n, v, o, wd) { this._changed.push([n, v, o]); },
            setDirtyCanvas() { this._dirty++; } };
        globalThis.__drawn = [];
        globalThis.w.draw(makeCtx(true), globalThis.node, 640, 10, 20);
        globalThis.right = globalThis.__drawn[globalThis.__drawn.length - 1].x;
    """)
    check("value is right-aligned to the handed-in width",
          ctx.eval("globalThis.right") == 640 - 15 * 2 - 5,
          str(ctx.eval("globalThis.right")))
    ctx.eval("""
        globalThis.__drawn = [];
        globalThis.w.draw(makeCtx(true), globalThis.node, 0, 10, 20);
        globalThis.right2 = globalThis.__drawn[globalThis.__drawn.length - 1].x;
    """)
    check("falls back to node.size[0] when handed no width",
          ctx.eval("globalThis.right2") == 640 - 15 * 2 - 5,
          str(ctx.eval("globalThis.right2")))


def test_value_fitting():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        return

    ctx = _ctx()
    ctx.eval("""
        globalThis.node = { size: [0, 0] };
        function drawAt(width, value) {
            const w = makeBatPathWidget({ name: "path", value });
            globalThis.__drawn = [];
            w.draw(makeCtx(true), globalThis.node, width, 0, 20);
            return globalThis.__drawn;
        }
    """)
    long_path = "/jobs/proj/shots/sh010/comp/renders/v012/sh010_comp_v012.####.exr"

    wide = ctx.eval(f"JSON.stringify(drawAt(1200, {long_path!r}).map(d => d.t))")
    narrow = ctx.eval(f"JSON.stringify(drawAt(300, {long_path!r}).map(d => d.t))")
    import json
    wide, narrow = json.loads(wide), json.loads(narrow)

    check("a wide node shows the whole value",
          long_path in wide, str(wide))
    check("a wide node still shows the label", "path" in wide, str(wide))
    check("a narrow node trims from the FRONT",
          any(t.startswith("…") and long_path.endswith(t[1:]) for t in narrow),
          str(narrow))
    # The label is kept while there is still a usable amount of room for the
    # value (a 300px node has ~190px left, which is plenty); it steps aside
    # only once the value would be squeezed.
    tight = json.loads(ctx.eval(
        f"JSON.stringify(drawAt(200, {long_path!r}).map(d => d.t))"))
    check("the label is kept at a middling width", "path" in narrow, str(narrow))
    check("the label steps aside when the row is tight",
          "path" not in tight and len(tight) == 1, str(tight))

    # The fit is by measurement, so widening shows strictly more.
    shown = {}
    for width in (300, 500, 900):
        vals = json.loads(ctx.eval(
            f"JSON.stringify(drawAt({width}, {long_path!r}).map(d => d.t))"))
        shown[width] = max((len(t) for t in vals if t != "path"), default=0)
    check("widening the node shows more of the path",
          shown[300] < shown[500] < shown[900], str(shown))

    # An empty value draws the placeholder colour, not a stray "…".
    empty = json.loads(ctx.eval('JSON.stringify(drawAt(400, "").map(d => d.t))'))
    check("an empty value draws nothing but the label",
          [t for t in empty if t != "path"] == [""], str(empty))


def test_commit_notifies():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        return

    ctx = _ctx()
    # commitValue is module-private, so drive it the way the dialog does. The
    # dialog itself needs a DOM; reach the function through the same path by
    # calling it directly now that `export` has been stripped.
    ctx.eval("""
        globalThis.w = makeBatPathWidget({ name: "path", value: "old.mov" });
        globalThis.seen = [];
        globalThis.w.callback = (v) => globalThis.seen.push(["callback", v]);
        globalThis.node = {
            _dirty: 0,
            onWidgetChanged(n, v, o) { globalThis.seen.push(["changed", n, v, o]); },
            setDirtyCanvas() { this._dirty++; },
        };
        commitValue(globalThis.w, globalThis.node, "new.mov");
    """)
    import json
    seen = json.loads(ctx.eval("JSON.stringify(globalThis.seen)"))
    check("commit fires the widget callback",
          ["callback", "new.mov"] in seen, str(seen))
    check("commit raises onWidgetChanged with the old value",
          ["changed", "path", "new.mov", "old.mov"] in seen, str(seen))
    check("commit marks the canvas dirty",
          ctx.eval("globalThis.node._dirty") >= 1)
    check("commit sets the value",
          ctx.eval("globalThis.w.value") == "new.mov")


def test_mouse_guard():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        return

    ctx = _ctx()
    ctx.eval("""
        globalThis.w = makeBatPathWidget({ name: "path" });
        globalThis.moved = globalThis.w.mouse({ type: "pointermove" }, [0,0], {});
        globalThis.wheeled = globalThis.w.mouse({ type: "wheel" }, [0,0], {});
    """)
    check("mouse ignores pointermove", ctx.eval("globalThis.moved") is False)
    check("mouse ignores wheel", ctx.eval("globalThis.wheeled") is False)


def test_consumers_use_the_shared_widget():
    """The point of the module is that there is only one copy."""
    stale = []
    for name in sorted(os.listdir(os.path.join(PACK, "web"))):
        if not name.endswith(".js") or name == "bat_path_widget.js":
            continue
        src = open(os.path.join(PACK, "web", name), encoding="utf-8").read()
        if "litesearchbox" in src or re.search(r"function\s+drawPathWidget", src):
            stale.append(name)
    check("no second copy of the path widget in the pack",
          not stale, ", ".join(stale))

    for name in ("bat_video_loader.js", "bat_frame_picker.js"):
        src = open(os.path.join(PACK, "web", name), encoding="utf-8").read()
        check(f"{name} imports the shared widget",
              "makeBatPathWidget" in src and "bat_path_widget.js" in src)


if __name__ == "__main__":
    test_width_unpinned()
    test_value_fitting()
    test_commit_notifies()
    test_mouse_guard()
    test_consumers_use_the_shared_widget()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
