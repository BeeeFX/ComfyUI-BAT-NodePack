"""BAT — regression test for web/bat_node_layout.js, the shared DOM-widget
sizing contract.

Ten editors depend on this file, and its whole job is to hand the two renderers
numbers they will believe. A mistake here does not throw — the node just comes
out the wrong height, or grows to fill the viewport. So the numbers get pinned.

    env/bin/python tools/test_node_layout.py

Runs the real module under quickjs with the ComfyUI globals stubbed.
"""

import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

STUB = r"""
var __vue = false;                       // Comfy.VueNodes.Enabled
var app = { extensionManager: { setting: { get: function(k){ return __vue; } } } };
function ResizeObserver(f){ this.observe = function(){}; this.disconnect = function(){}; }

function mkEl(cssMinHeight) {
    var props = {};
    return {
        style: {
            minHeight: cssMinHeight || "",
            setProperty: function(k, v){ props[k] = v; },
        },
        _props: props,
        clientWidth: 400, clientHeight: 300,
    };
}
function mkNode(el) {
    var node = {
        size: [400, 500], properties: {},
        graph: { _v: 0, incrementVersion: function(){ this._v++; } },
        widgets: [],
        addDOMWidget: function(name, type, element, options) {
            var w = { name: name, type: type, element: element, options: options };
            node.widgets.push(w);
            return w;
        },
        computeSize: function(){ return node._natural || [400, 100]; },
        setSize: function(s){ node.size = [s[0], s[1]]; },
        setDirtyCanvas: function(){},
    };
    return node;
}
"""

CHECKS = []


def check(label, got, want):
    ok = got == want
    CHECKS.append(ok)
    print(f"{'ok ' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  expected {want!r}"))


def load():
    import quickjs

    src = open(os.path.join(PACK, "web", "bat_node_layout.js"), encoding="utf-8").read()
    src = re.sub(r"^import\s[\s\S]*?from\s+['\"][^'\"]+['\"];", "", src, flags=re.M)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(STUB)
    ctx.eval(src)
    return ctx


def main():
    ctx = load()

    # ── pinned (growable: false) ────────────────────────────────────────
    ctx.eval("""
        var el = mkEl(), node = mkNode(el);
        var w = addBatDOMWidget(node, "a", "a", el, { height: 240, minWidth: 300 });
        function size(){ return JSON.stringify(w.computeLayoutSize()); }
    """)
    check("pinned widget reports min === max",
          json.loads(ctx.eval("size()")),
          {"minHeight": 240, "maxHeight": 240, "minWidth": 300})
    check("...and publishes both CSS vars",
          json.loads(ctx.eval("JSON.stringify(el._props)")),
          {"--comfy-widget-min-height": "240px", "--comfy-widget-max-height": "240px"})

    # ── growable, no explicit cap ───────────────────────────────────────
    ctx.eval("""
        var el2 = mkEl(), node2 = mkNode(el2);
        var w2 = addBatDOMWidget(node2, "b", "b", el2, { height: 200, growable: true });
    """)
    check("growable with no cap falls back to GROW_FACTOR, not infinity",
          json.loads(ctx.eval("JSON.stringify(w2.computeLayoutSize())"))["maxHeight"], 600)

    # ── growable, numeric cap (every existing caller's shape) ───────────
    ctx.eval("""
        var el3 = mkEl(), node3 = mkNode(el3);
        var w3 = addBatDOMWidget(node3, "c", "c", el3,
                                 { height: 200, growable: true, maxHeight: 450 });
    """)
    check("a numeric maxHeight is still honoured",
          json.loads(ctx.eval("JSON.stringify(w3.computeLayoutSize())"))["maxHeight"], 450)

    # ── growable, callable cap + callable height (the collapse case) ────
    ctx.eval("""
        var collapsed = true;
        var el4 = mkEl(), node4 = mkNode(el4);
        var w4 = addBatDOMWidget(node4, "d", "d", el4, {
            height: function(){ return collapsed ? 26 : 346; },
            growable: true,
            maxHeight: function(){ return collapsed ? 26 : null; },
        });
    """)
    check("a collapsed editor is pinned to its header height",
          json.loads(ctx.eval("JSON.stringify(w4.computeLayoutSize())")),
          {"minHeight": 26, "maxHeight": 26, "minWidth": 320})
    ctx.eval("collapsed = false;")
    check("...and the SAME widget re-reads both on the next layout pass",
          json.loads(ctx.eval("JSON.stringify(w4.computeLayoutSize())")),
          {"minHeight": 346, "maxHeight": 1038, "minWidth": 320})

    # ── hidden ──────────────────────────────────────────────────────────
    ctx.eval("w4.type = 'hidden';")
    check("a hidden widget claims no height, as the frontend's own impl does",
          json.loads(ctx.eval("JSON.stringify(w4.computeLayoutSize())")),
          {"minHeight": 0, "maxHeight": 0, "minWidth": 0})
    ctx.eval("w4.type = 'd';")

    # ── an editor's own cssText min-height is not stomped on first publish ──
    ctx.eval("""
        var el5 = mkEl("540px"), node5 = mkNode(el5);
        var w5 = addBatDOMWidget(node5, "e", "e", el5, { height: 200 });
    """)
    check("an editor's own cssText min-height survives (roto / animated_crop)",
          ctx.eval("el5.style.minHeight"), "540px")
    check("...but a re-measure overrides it with the widget's contract",
          ctx.eval("w5._batPublishVars(); el5.style.minHeight"), "200px")

    # ── refreshBatLayout, Nodes 1.0 ─────────────────────────────────────
    ctx.eval("__vue = false; node4._natural = [400, 120]; node4.size = [400, 500];")
    ctx.eval("refreshBatLayout(node4, w4)")
    check("without shrink the node only ever grows", json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 500])
    ctx.eval("refreshBatLayout(node4, w4, { shrink: true })")
    check("with shrink it drops to the natural size",
          json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 120])
    check("...and bumps the graph version, which is what wakes Nodes 2.0",
          ctx.eval("node4.graph._v") > 0, True)

    # ── refreshBatLayout, Nodes 2.0 ─────────────────────────────────────
    ctx.eval("__vue = true; node4.size = [400, 500];")
    ctx.eval("refreshBatLayout(node4, w4, { shrink: true })")
    check("under Nodes 2.0 node.size is left alone (the layout derives it)",
          json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 500])
    ctx.eval("__vue = false;")

    # ── setBatWidgetHidden ──────────────────────────────────────────────
    ctx.eval("setBatWidgetHidden(node4, w4, true)")
    check("hiding sets all three legs (2.0 type, 1.0 hidden, legacy computeSize)",
          json.loads(ctx.eval(
              "JSON.stringify([w4.type, w4.hidden, w4.computeSize(), el4.style.display])")),
          ["hidden", True, [0, -4], "none"])
    ctx.eval("setBatWidgetHidden(node4, w4, false)")
    check("...and showing restores the original type and drops computeSize",
          json.loads(ctx.eval(
              "JSON.stringify([w4.type, w4.hidden, w4.computeSize === undefined, el4.style.display])")),
          ["d", False, True, ""])

    failed = CHECKS.count(False)
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
