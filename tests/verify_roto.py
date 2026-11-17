#!/usr/bin/env python
"""
Regression checks for Bat_Roto's round-2 changes (2026-09-23).

1. **Preview sidecar.** The frame strip goes through bat_ui_ref.stash_ui, so the
   prompt history holds a token, and load_ui gives back the exact dict the
   editor used to receive.
2. **Keyframe easing, preview == render.** A shape's `keyframe_ease` map
   ({"<frame>": name}, the curve on the way OUT of that key) is applied by both
   `_resolve_shape_at_frame` (Python) and `resolveShapeAtFrame` (the JS
   preview). They are run side by side here on random shapes, including point
   counts that differ between keys. A shape with no map renders exactly as it
   did before easing existed.
3. **No re-run after the first run.** The plate's size/length moved out of the
   `state` JSON (a prompt input, so part of the cache key) into node.properties.
4. **One undo, not two.** On the node Ctrl+Z belongs to core's ChangeTracker;
   the editor's own history answers only in fullscreen, and every committed
   edit is handed to core as an undo step.
5. **Middle-drag pans the picture under Nodes 2.0.** A window capture-phase
   router claims the press before TransformPane can forward it to the graph —
   inside a Vue node only; Nodes 1.0 keeps the canvas's own listeners.

(3) and (4) are source-level pins: the editor can't run headless here.

    python tests/verify_roto.py
"""

import importlib
import json
import os
import random
import re
import sys
import types

import quickjs
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _harness import auto_stub_js, strip_modules  # noqa: E402

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def expect(label, got, want):
    check(label, got == want, f"got {got!r}, want {want!r}")


def read(rel):
    with open(os.path.join(PACK, rel), encoding="utf-8") as f:
        return f.read()


def load_pack_modules():
    """Import the nodes without the pack __init__ (which needs a live server)."""
    pkg = types.ModuleType("batpkg")
    pkg.__path__ = [PACK]
    sys.modules["batpkg"] = pkg
    return (importlib.import_module("batpkg.bat_roto"),
            importlib.import_module("batpkg.bat_ui_ref"))


def js_function(src, name):
    """A module-level `function name(...) { ... }` block, by brace matching."""
    start = src.index(f"\nfunction {name}(") + 1
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


# ─── 1. sidecar ──────────────────────────────────────────────────────────────

def test_sidecar(roto, ui_ref):
    print("\npreview payload goes to a sidecar")
    out = roto.BatRoto().render(torch.rand(3, 16, 24, 3), '{"shapes":[]}')
    ui = out["ui"]
    check("the history holds a token, not the strip",
          set(ui) == {"bat_ui"} and "frames" not in ui, str(list(ui)))
    full = ui_ref.load_ui(ui)
    check("load_ui gives back the dict the editor reads",
          full is not None and len(full["frames"]) == 3 and full["w"] == [24]
          and full["h"] == [16] and full["stride"] == [1] and full["frame_count"] == [3],
          str(full and {k: (v if k != "frames" else len(v)) for k, v in full.items()}))


# ─── 2. easing parity ────────────────────────────────────────────────────────

EASES = ("linear", "ease_in", "ease_out", "ease_in_out", "hold")


def rand_points(n):
    return [[round(random.uniform(0, 400), 3) for _ in range(6)] for _ in range(n)]


def rand_shape():
    keys = sorted(random.sample(range(0, 40), random.randint(1, 5)))
    base = random.randint(3, 6)
    shape = {"keyframes": {str(k): rand_points(base + random.choice((0, 0, 0, 1, -1)))
                           for k in keys}}
    if random.random() < 0.8:
        shape["keyframe_ease"] = {str(k): random.choice(EASES + ("bogus",))
                                  for k in keys if random.random() < 0.7}
    return shape


def test_easing_parity(roto):
    print("\nkeyframe easing: preview == render")
    src = read("web/bat_roto.js")
    ctx = quickjs.Context()
    ctx.eval(strip_modules(read("web/bat_easing.js")))
    ctx.eval(js_function(src, "lerpKeyframes"))
    ctx.eval(js_function(src, "resolveShapeAtFrame"))

    random.seed(7)
    worst, cases, eased_moves = 0.0, 0, 0
    for _ in range(150):
        shape = rand_shape()
        for f in range(-2, 43, 3):
            js = json.loads(ctx.eval(f"JSON.stringify(resolveShapeAtFrame({json.dumps(shape)}, {f}))"))
            py = roto._resolve_shape_at_frame(shape, f)
            if js is None or py is None:
                check("both sides agree there is nothing to draw", js is None and py is None)
                continue
            if len(js) != len(py) or any(len(p) != len(q) for p, q in zip(js, py)):
                check(f"same point layout at frame {f}", False, f"{js} vs {py}")
                continue
            worst = max(worst, max((abs(a - b) for p, q in zip(js, py) for a, b in zip(p, q)),
                                   default=0.0))
            cases += 1
            lin = dict(shape, keyframe_ease={})
            if roto._resolve_shape_at_frame(lin, f) != py:
                eased_moves += 1
    check(f"JS and Python agree on {cases} frames (max |d| {worst:.2g}, py is float32)",
          worst < 1e-3)
    check("easing actually changes in-between frames", eased_moves > 50, str(eased_moves))

    # Legacy: no map, an empty map and an all-linear map render identically.
    shape = {"keyframes": {"0": rand_points(4), "20": rand_points(4)}}
    same = all(roto._resolve_shape_at_frame(shape, f)
               == roto._resolve_shape_at_frame(dict(shape, keyframe_ease={"0": "linear"}), f)
               == roto._resolve_shape_at_frame(dict(shape, keyframe_ease={}), f)
               for f in range(21))
    check("a shape with no ease (every saved workflow) is still linear", same)

    held = roto._resolve_shape_at_frame(dict(shape, keyframe_ease={"0": "hold"}), 15)
    check("hold keeps the earlier key until the next one",
          all(abs(a - b) < 1e-3 for p, q in zip(held, shape["keyframes"]["0"]) for a, b in zip(p, q)))

    node = roto.BatRoto()
    sq = lambda o: [[10 + o, 10, 10 + o, 10, 10 + o, 10], [30 + o, 10, 30 + o, 10, 30 + o, 10],
                    [30 + o, 30, 30 + o, 30, 30 + o, 30], [10 + o, 30, 10 + o, 30, 10 + o, 30]]
    doc = {"shapes": [{"id": "a", "closed": True, "keyframes": {"0": sq(0), "4": sq(16)}}]}
    plain = node.render(torch.zeros(5, 40, 64, 3), json.dumps(doc))["result"][0]
    doc["shapes"][0]["keyframe_ease"] = {"0": "ease_in"}
    eased = node.render(torch.zeros(5, 40, 64, 3), json.dumps(doc))["result"][0]
    check("render: end keys unchanged by an ease",
          torch.equal(plain[0], eased[0]) and torch.equal(plain[4], eased[4]))
    # Mid-segment (t = 0.5) linear has moved 8 px of the 16; ease_in only 4.
    cx = lambda m: float((m * torch.arange(m.shape[1])).sum() / m.sum())
    lag = cx(plain[2]) - cx(eased[2])
    check("render: ease_in lags the linear move mid-segment by t - t^2 = 4 px",
          abs(lag - 4.0) < 0.1, f"{lag:.3f}px")


# ─── 3 + 4. source pins ──────────────────────────────────────────────────────

def test_source_pins():
    print("\nplate metadata stays out of the cache key")
    src = read("web/bat_roto.js")
    check("the ingest no longer stamps imgW/imgH/frameCount into `state`",
          not re.search(r"state\.doc\.(imgW|imgH|frameCount)\s*=", src))
    check("...it records them in node.properties",
          "(node.properties ||= {})[PLATE_PROP] = { w, h, frameCount: state.frameCount }" in src)
    check("...and a legacy state's fields are still read as the fallback",
          "return { w: d.imgW || 0, h: d.imgH || 0, frameCount: d.frameCount || 0 };" in src)

    print("\none undo, not two")
    check("the editor's Ctrl+Z/Y answers only in fullscreen",
          "if ((e.ctrlKey || e.metaKey) && !e.altKey && isBatFullscreen(node)) {" in src)
    rec = src[src.index("    function recordHistory() {"):]
    rec = rec[:rec.index("\n    }\n")]
    check("every committed edit is handed to core's ChangeTracker", "coreCapture();" in rec)
    check("...but not while a reload rebuilds the history",
          src.count("quietHistory = true;") == 2)
    check("the ease picker is in the transport row",
          "controlsRow.append(prevBtn, playBtn, nextBtn, addKeyBtn, delKeyBtn, easeSel," in src)


# ─── 5. middle-drag router ───────────────────────────────────────────────────

STUB = r"""
var __win = {};
var window = {
    addEventListener: function (t, fn, cap) { (__win[t] = __win[t] || []).push({ fn: fn, cap: !!cap }); },
};
function mkEl(cls, parent) {
    var el = {
        className: cls || "", parentNode: parent || null, attrs: {}, children: [],
        style: {}, _l: {}, clientWidth: 400, clientHeight: 200,
        appendChild: function (c) { c.parentNode = this; this.children.push(c); return c; },
        append: function () { for (var i = 0; i < arguments.length; i++) this.appendChild(arguments[i]); },
        setAttribute: function (k, v) { this.attrs[k] = String(v); },
        getAttribute: function (k) { return k in this.attrs ? this.attrs[k] : null; },
        addEventListener: function (t, fn) { (this._l[t] = this._l[t] || []).push(fn); },
        getBoundingClientRect: function () { return { left: 0, top: 0, width: 400, height: 200 }; },
        setPointerCapture: function () {}, releasePointerCapture: function () {},
        matches: function (sel) {
            if (sel === ".lg-node") return (" " + this.className + " ").indexOf(" lg-node ") !== -1;
            var m = /^\[([\w-]+)\]$/.exec(sel);
            return !!m && (m[1] in this.attrs);
        },
        closest: function (sel) {
            for (var n = this; n; n = n.parentNode) if (n.matches && n.matches(sel)) return n;
            return null;
        },
    };
    if (parent) parent.appendChild(el);
    return el;
}
var document = { createElement: function () { return mkEl(""); } };
function ev(target, type, over) {
    var e = { target: target, type: type, button: 1, pointerId: 7, clientX: 0, clientY: 0,
              stopped: false, prevented: false,
              stopPropagation: function () { this.stopped = true; },
              preventDefault: function () { this.prevented = true; } };
    for (var k in (over || {})) e[k] = over[k];
    return e;
}
/** Deliver to window capture listeners only — what runs before TransformPane. */
function winCapture(e) {
    var l = (__win[e.type] || []).filter(function (x) { return x.cap; });
    for (var i = 0; i < l.length; i++) l[i].fn(e);
    return e;
}
function make(inVueNode) {
    var host = inVueNode ? mkEl("group/node lg-node absolute") : mkEl("dom-widget");
    var wrap = mkEl("", host);
    var canvas = mkEl("", wrap);
    var state = { imgW: 100, imgH: 50, dispScale: 1, offX: 0, offY: 0 };
    var changes = 0;
    attachZoomControl({ wrap: wrap, canvas: canvas, state: state, onChange: function () { changes++; } });
    return { canvas: canvas, state: state, changes: function () { return changes; } };
}
"""


def test_pan_router():
    print("\nmiddle-drag pans the picture under Nodes 2.0")
    src = read("web/bat_zoom_control.js")
    ctx = quickjs.Context()
    ctx.eval(auto_stub_js(src))
    ctx.eval(STUB)
    ctx.eval(strip_modules(src))

    ctx.eval("var v = make(true);")
    ctx.eval("var d = winCapture(ev(v.canvas, 'pointerdown', { clientX: 10, clientY: 10 }));")
    expect("a press inside a Vue node is claimed before TransformPane sees it",
          ctx.eval("d.stopped && d.prevented"), True)
    ctx.eval("var m = winCapture(ev(v.canvas, 'pointermove', { clientX: 40, clientY: 30, button: -1 }));")
    expect("the drag pans the editor view", ctx.eval("[v.state.panX, v.state.panY].join()"), "30,20")
    expect("...and each move is claimed too", ctx.eval("m.stopped"), True)
    ctx.eval("var u = winCapture(ev(v.canvas, 'pointerup', { clientX: 40, clientY: 30 }));")
    ctx.eval("var after = winCapture(ev(v.canvas, 'pointermove', { clientX: 90, clientY: 90, button: -1 }));")
    expect("release ends it: later moves pass through untouched",
          ctx.eval("u.stopped && !after.stopped && v.state.panX === 30"), True)
    ctx.eval("var left = winCapture(ev(v.canvas, 'pointerdown', { button: 0 }));")
    expect("a left press is never claimed", ctx.eval("left.stopped"), False)

    ctx.eval("var o = make(false);")
    ctx.eval("var d1 = winCapture(ev(o.canvas, 'pointerdown', { clientX: 10, clientY: 10 }));")
    expect("Nodes 1.0 / fullscreen: the router leaves the press to the canvas",
          ctx.eval("!d1.stopped && o.state.panX === 0"), True)
    ctx.eval("o.canvas._l.pointerdown.forEach(function (f) { f(d1); });"
             "o.canvas._l.pointermove.forEach(function (f) {"
             "  f(ev(o.canvas, 'pointermove', { clientX: 15, clientY: 12, button: -1 })); });")
    expect("...whose own listeners still pan exactly as before",
          ctx.eval("[o.state.panX, o.state.panY].join()"), "5,2")
    expect("the canvas is marked for the router",
          ctx.eval("v.canvas.getAttribute('data-bat-zoom-canvas') !== null"), True)


def main():
    print("Bat_Roto verification")
    roto, ui_ref = load_pack_modules()
    test_sidecar(roto, ui_ref)
    test_easing_parity(roto)
    test_source_pins()
    test_pan_router()
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
