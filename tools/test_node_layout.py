"""BAT — regression test for web/bat_node_layout.js, the shared DOM-widget
sizing contract.

Ten editors depend on this file, and its whole job is to hand the two renderers
numbers they will believe. A mistake here does not throw — the node just comes
out the wrong height, or grows to fill the viewport. So the numbers get pinned.

    env/bin/python tools/test_node_layout.py

Runs the real module under quickjs with the ComfyUI globals stubbed.
"""

import os
import sys
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PACK, "tests"))
from _harness import auto_stub_js, strip_modules  # noqa: E402

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

    ctx = quickjs.Context()
    # Derived no-op stubs first (markBatWidget from bat_paste_guard.js, and any
    # helper imported later), then the curated ones, which must win.
    ctx.eval(auto_stub_js(src))
    ctx.eval(STUB)
    ctx.eval(strip_modules(src))
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
    # node.size is the stored height FLOOR under the Vue node's DOM content
    # (frontend 1.55, graphLayoutAttachment.ts), so a collapse has to lower it —
    # this used to assert it was left alone, which kept the old height.
    ctx.eval("__vue = true; node4.size = [400, 500];")
    ctx.eval("refreshBatLayout(node4, w4)")
    check("under Nodes 2.0 a plain refresh writes nothing (the DOM grows the node)",
          json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 500])
    ctx.eval("refreshBatLayout(node4, w4, { shrink: true })")
    check("...but a shrink lowers the stored height floor, width untouched",
          json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 120])
    ctx.eval("node4.size = [400, 80]; refreshBatLayout(node4, w4, { shrink: true })")
    check("...and never raises it",
          json.loads(ctx.eval("JSON.stringify(node4.size)")), [400, 80])
    ctx.eval("__vue = false;")

    # ── clampNodeSize ───────────────────────────────────────────────────
    ctx.eval("var n6 = mkNode(mkEl()); n6.size = [250, 90]; clampNodeSize(n6, 640, 540);")
    check("Nodes 1.0: both floors apply",
          json.loads(ctx.eval("JSON.stringify(n6.size)")), [640, 540])
    ctx.eval("__vue = true; var n7 = mkNode(mkEl()); n7.size = [250, 90]; clampNodeSize(n7, 640, 540);")
    check("Nodes 2.0: a fresh node gets the width floor (its height stays DOM-derived)",
          json.loads(ctx.eval("JSON.stringify(n7.size)")), [640, 90])
    ctx.eval("var n8 = mkNode(mkEl()); n8.size = [700, 90]; clampNodeSize(n8, 640, 540);")
    check("...and a node already wider is left alone",
          json.loads(ctx.eval("JSON.stringify(n8.size)")), [700, 90])
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

    # ── addBatDOMWidget keeps its widget out of widgets_values ──────────
    ctx.eval("var el9 = mkEl(), node9 = mkNode(el9); var w9 = addBatDOMWidget(node9, 'f', 'f', el9, { height: 100 });")
    check("the DOM widget is serialize:false (workflow), not just options.serialize (prompt)",
          json.loads(ctx.eval("JSON.stringify([w9.serialize, w9.options.serialize])")), [False, False])

    test_dom_widget_serialisation()

    failed = CHECKS.count(False)
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0

# ════════════════════════════════════════════════════════════════════════
# DOM widgets and widgets_values
# ════════════════════════════════════════════════════════════════════════
#
# addBatDOMWidget sets `widget.serialize = false`, so a BAT editor no longer
# writes a "" into widgets_values after its real values. Saves made before
# that still carry the slot, which is only harmless while every DOM widget is
# the LAST widget on its node: positional restore walks the widgets, so a
# trailing value past the last serialised one is never read. This drives
# every editor for real under quickjs — Python widgets built from INPUT_TYPES
# the way the frontend builds them, the extension's own hooks, a stand-in for
# the frontend's serialise / positional configure — and pins:
#   • each DOM widget is serialize:false and nothing serialised follows it;
#   • a pre-change save (trailing "") and a new save restore identically;
#   • the three nodes that read widgets_values themselves still infer right
#     (Grade / Animated Grade lift_mode, Rescale ringing, Video Combine's
#     codec tail).
# Needs torch + ComfyUI importable (PYTHONPATH=<ComfyUI>) for INPUT_TYPES;
# skipped otherwise.

EDITORS = {
    "bat_advanced_blend.js": ["Bat_AdvancedBlend"],
    "bat_animated_crop.js": ["Bat_AnimatedCrop"],
    "bat_animated_grade.js": ["Bat_AnimatedGrade"],
    "bat_crop.js": ["Bat_Crop"],
    "bat_exposure_bracket.js": ["Bat_ExposureBracket", "Bat_ExposureMerge"],
    "bat_filename_prefix.js": ["Bat_FilenamePrefix"],
    "bat_frame_picker.js": ["Bat_FramePicker"],
    "bat_framehold.js": ["Bat_Framehold"],
    "bat_grade.js": ["Bat_Grade"],
    "bat_hdr_tonal_composite.js": ["Bat_HDRTonalComposite"],
    "bat_layered_images.js": ["Bat_LayeredImages"],
    "bat_ref_aligner.js": ["Bat_RefAligner"],
    "bat_rescale.js": ["Bat_Rescale"],
    "bat_roto.js": ["Bat_Roto"],
    "bat_show.js": ["Bat_ShowAny", "Bat_ShowTensorShape", "Bat_WanContextCalculator"],
    "bat_video_combine.js": ["Bat_VideoCombine"],
    "bat_video_loader.js": ["Bat_VideoLoader"],
}


def python_node_defs():
    """name -> {name, input, input_order} from the real INPUT_TYPES, or None."""
    import importlib.util
    import types
    try:
        srv = types.ModuleType("server")

        class _Routes:
            def __getattr__(self, _k):
                return lambda *a, **k: (lambda f: f)

        class _Instance:
            routes = _Routes()

            def __getattr__(self, _k):
                return lambda *a, **k: None

        srv.PromptServer = type("PromptServer", (), {"instance": _Instance()})
        sys.modules.setdefault("server", srv)
        spec = importlib.util.spec_from_file_location(
            "batpack", os.path.join(PACK, "__init__.py"),
            submodule_search_locations=[PACK])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["batpack"] = pkg
        spec.loader.exec_module(pkg)
    except Exception as e:  # noqa: BLE001 - any missing dependency means skip
        print(f"skip  DOM-widget serialisation checks (cannot import the pack: {e!r})")
        return None

    def plain(o):
        return {k: v for k, v in o.items()
                if isinstance(v, (str, int, float, bool, list, type(None)))}

    out = {}
    for name, cls in pkg.NODE_CLASS_MAPPINGS.items():
        it = cls.INPUT_TYPES()
        inp, order = {}, {}
        for sec in ("required", "optional"):
            inp[sec] = {}
            order[sec] = list((it.get(sec) or {}).keys())
            for k, v in (it.get(sec) or {}).items():
                t = v[0] if isinstance(v[0], (str, list)) else str(v[0])
                opts = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
                inp[sec][k] = [t, plain(opts)]
        out[name] = {"name": name, "input": inp, "input_order": order}
    return out


# A browser that says yes to everything: any property is another such object,
# calling it returns one, it coerces to 0 / "", iterates as empty. The editors
# build a lot of DOM and canvas in onNodeCreated; none of it matters here, only
# that they get far enough to add their widgets.
PERMISSIVE_JS = r"""
var __timers = [], __warn = [];
function __P() {
  var store = {};
  return new Proxy(function(){}, {
    get: function(t, k) {
      if (k in store) return store[k];
      if (k === Symbol.toPrimitive) return function(h){ return h === "string" ? "" : 0; };
      if (k === Symbol.iterator) return function(){ return [][Symbol.iterator](); };
      if (k === "then") return undefined;
      if (["length", "width", "height", "clientWidth", "clientHeight",
           "offsetWidth", "offsetHeight"].indexOf(k) !== -1) return 0;
      if (k === "children" || k === "childNodes") return (store[k] = []);
      if (k === "dataset") return (store[k] = {});
      if (k === "style") return (store[k] = { setProperty: function(){}, removeProperty: function(){} });
      if (k === "getBoundingClientRect") return function(){ return {width:400,height:300,left:0,top:0,right:400,bottom:300}; };
      if (k === "getImageData" || k === "createImageData") return function(){ return {data: new Uint8ClampedArray(16), width:2, height:2}; };
      if (k === "toString") return function(){ return ""; };
      return (store[k] = __P());
    },
    set: function(t, k, v) { store[k] = v; return true; },
    has: function(){ return true; },
    apply: function(){ return __P(); },
    construct: function(){ return __P(); },
  });
}
var window = __P(), document = __P(), navigator = __P();
var localStorage = { getItem: function(){ return null; }, setItem: function(){}, removeItem: function(){} };
var sessionStorage = localStorage;
var console = { log: function(){}, info: function(){}, debug: function(){},
                warn: function(){ __warn.push(Array.prototype.join.call(arguments, " ")); },
                error: function(){ __warn.push("ERROR " + Array.prototype.join.call(arguments, " ")); } };
var setTimeout = function(fn){ __timers.push(fn); return __timers.length; };
var clearTimeout = function(){}, setInterval = function(){ return 0; }, clearInterval = function(){};
var requestAnimationFrame = function(fn){ __timers.push(fn); return 1; }, cancelAnimationFrame = function(){};
var queueMicrotask = function(fn){ __timers.push(fn); };
var ResizeObserver = function(){ this.observe = function(){}; this.disconnect = function(){}; this.unobserve = function(){}; };
var IntersectionObserver = ResizeObserver, MutationObserver = ResizeObserver;
var Image = function(){ return __P(); }, OffscreenCanvas = function(){ return __P(); };
var URL = { createObjectURL: function(){ return ""; }, revokeObjectURL: function(){} };
var URLSearchParams = function(){ this.set = function(){}; this.toString = function(){ return ""; }; };
var getComputedStyle = function(){ return __P(); };
var devicePixelRatio = 1, performance = { now: function(){ return 0; } };
var LiteGraph = __P();
var __ext = [], __SPECS = null;
var app = { registerExtension: function(e){ __ext.push(e); }, graph: { extra: {} }, canvas: __P(),
            extensionManager: { setting: { get: function(){ return false; } } }, ui: __P() };
var api = { apiURL: function(p){ return p; }, addEventListener: function(){}, removeEventListener: function(){},
            fetchApi: function(p){
              if (__SPECS && String(p).indexOf("/bat/video/formats") === 0)
                return Promise.resolve({ ok: true, json: function(){ return Promise.resolve(__SPECS); } });
              return new Promise(function(){});
            } };
function __drain(){ var q = __timers; __timers = []; for (var i = 0; i < q.length; i++) { try { q[i](); } catch (e) { __warn.push("timer threw: " + e); } } }

// The node as the frontend hands it to onNodeCreated: Python widgets in
// INPUT_TYPES order (required, then optional), a value-control combo right
// after any INT that asks for one.
function __mkNode(def) {
  var node = { id: -1, type: def.name, size: [400, 300], pos: [0, 0], properties: {}, flags: {},
    inputs: [], outputs: [], widgets: [], constructor: { nodeData: def },
    graph: { extra: {}, incrementVersion: function(){}, setDirtyCanvas: function(){}, links: {}, getLink: function(){ return null; } },
    addWidget: function(type, name, value, cb, opts) {
      var w = { type: String(type).toLowerCase(), name: name, value: value,
                callback: typeof cb === "function" ? cb : undefined,
                options: (opts && typeof opts === "object") ? opts : {} };
      node.widgets.push(w); return w; },
    addCustomWidget: function(w) { node.widgets.push(w); return w; },
    // As domWidget.ts: the value is options.getValue() ?? "", and nothing but
    // the caller sets widget.serialize.
    addDOMWidget: function(name, type, element, options) {
      var w = { name: name, type: type, element: element, options: options || {}, __dom: true,
                get value(){ return (this.options.getValue && this.options.getValue()) || ""; },
                set value(v){ if (this.options.setValue) this.options.setValue(v); } };
      node.widgets.push(w); return w; },
    removeWidget: function(w) { var i = node.widgets.indexOf(w); if (i >= 0) node.widgets.splice(i, 1); },
    addInput: function(n, t) { node.inputs.push({ name: n, type: t }); }, removeInput: function(i) { node.inputs.splice(i, 1); },
    addOutput: function(n, t) { node.outputs.push({ name: n, type: t, links: [] }); }, removeOutput: function(i) { node.outputs.splice(i, 1); },
    findInputSlot: function(n) { for (var i = 0; i < node.inputs.length; i++) if (node.inputs[i].name === n) return i; return -1; },
    findOutputSlot: function() { return -1; },
    setSize: function(s) { node.size = [s[0], s[1]]; }, computeSize: function() { return [400, 300]; },
    setDirtyCanvas: function(){}, expandToFitContent: function(){},
    getInputLink: function(){ return null; }, getInputNode: function(){ return null; }, isInputConnected: function(){ return false; },
    addProperty: function(n, v) { node.properties[n] = v; },
  };
  ["required", "optional"].forEach(function (sec) {
    (def.input_order[sec] || []).forEach(function (name) {
      var spec = def.input[sec][name], t = spec[0], o = spec[1] || {};
      var isCombo = Array.isArray(t) || t === "COMBO";
      var wt = o.forceInput ? null : isCombo ? "combo"
             : ({ INT: "number", FLOAT: "number", STRING: o.multiline ? "customtext" : "text", BOOLEAN: "toggle" })[t];
      if (!wt) { node.inputs.push({ name: name, type: t }); return; }
      var dflt = ("default" in o) ? o["default"]
               : isCombo ? (Array.isArray(t) ? t[0] : (o.options || [])[0])
               : t === "STRING" ? "" : t === "BOOLEAN" ? false : 0;
      node.widgets.push({ type: wt, name: name, value: dflt,
                          options: isCombo ? { values: Array.isArray(t) ? t : (o.options || []) } : {} });
      if (o.control_after_generate || (t === "INT" && (name === "seed" || name === "noise_seed")))
        node.widgets.push({ type: "combo", name: "control_after_generate", value: "fixed",
                            options: { values: ["fixed", "increment", "decrement", "randomize"] } });
    });
  });
  return node;
}

// LGraphNode.configure's widget half (frontend 1.55): positional restore over
// the widgets whose `serialize` is not false, then onConfigure.
function __baseConfigure(node, info) {
  var vals = info.widgets_values || [], i = 0;
  for (var k = 0; k < node.widgets.length; k++) {
    var w = node.widgets[k];
    if (w.serialize === false) continue;
    if (i >= vals.length) break;
    w.value = vals[i++];
  }
  if (node.onConfigure) node.onConfigure(Object.assign({}, info));
}

// serialiseWidgetValues (frontend 1.55): skip serialize === false only.
function __serialise(node) {
  var pos = [], named = {};
  node.widgets.forEach(function (w) {
    if (w.serialize === false) return;
    var v = w.value, s = (v != null && typeof v === "object") ? JSON.parse(JSON.stringify(v)) : (v == null ? null : v);
    pos.push(s); named[w.name] = s;
  });
  return { widgets_values: pos, widgets_values_named: named };
}

// The same save as the frontend wrote it before DOM widgets were
// serialize:false: each DOM widget's "" in its own place in the list.
function __serialiseAsBefore(node) {
  var pos = [];
  node.widgets.forEach(function (w) {
    if (w.element || w.__dom) { pos.push(""); return; }
    if (w.serialize === false) return;
    var v = w.value;
    pos.push((v != null && typeof v === "object") ? JSON.parse(JSON.stringify(v)) : (v == null ? null : v));
  });
  return pos;
}

function __create(def) {
  var nodeType = { prototype: { configure: function (info) { return __baseConfigure(this, info); } } };
  __ext.forEach(function (e) {
    try { if (e.beforeRegisterNodeDef) e.beforeRegisterNodeDef(nodeType, def, app); }
    catch (err) { __warn.push("beforeRegisterNodeDef threw: " + err); }
  });
  var node = __mkNode(def);
  for (var k in nodeType.prototype) node[k] = nodeType.prototype[k];
  try { if (node.onNodeCreated) node.onNodeCreated(); }
  catch (err) { __warn.push("onNodeCreated threw: " + err); }
  __ext.forEach(function (e) { try { if (e.nodeCreated) e.nodeCreated(node, app); } catch (err) {} });
  node.id = 7;
  return node;
}

function __describe(node) {
  return JSON.stringify(node.widgets.map(function (w) {
    return { name: w.name, dom: !!(w.element || w.__dom), ser: w.serialize !== false };
  }));
}
"""


# Sibling modules whose REAL code an editor's load path runs: Animated Grade's
# onConfigure calls restoreLegacyLiftMode from bat_grade.js.
REAL_IMPORTS = {"bat_animated_grade.js": ["bat_grade.js"]}


def editor_context(js_file):
    import quickjs
    from _harness import imported_names

    read = lambda f: open(os.path.join(PACK, "web", f), encoding="utf-8").read()  # noqa: E731
    src = read(js_file)
    extra = [read(f) for f in REAL_IMPORTS.get(js_file, [])]
    ctx = quickjs.Context()
    ctx.eval(PERMISSIVE_JS)
    # Imported helpers exist and return a permissive object; imported CONSTANTS
    # (EASES, …) are permissive values so `for … of` still works.
    names = sorted(imported_names(src, *extra) - {"app", "api"})
    ctx.eval("".join(f"var {n} = __P();\n" if n.isupper()
                     else f"function {n}() {{ return __P(); }}\n" for n in names))
    ctx.eval("function batPreviewWillReplay(){ return false; }"
             "function isNodeAlive(){ return true; } function markBatWidget(){}")
    ctx.eval(strip_modules(read("bat_node_layout.js")))   # the REAL helper under test
    fix = lambda code: strip_modules(code).replace(  # noqa: E731
        "import.meta.url", '"http://x/extensions/x/web/x.js"')
    for code in extra:
        # Its own scope (sibling editors share top-level names like NODE_TYPE),
        # publishing only what it exports.
        exported = re.findall(r"^export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)",
                              code, flags=re.M)
        ctx.eval("(function(){\n" + fix(code) + "\n"
                 + "".join(f"globalThis.{n} = {n};\n" for n in exported) + "})();")
    ctx.eval(fix(src))
    return ctx


def settle(ctx):
    for _ in range(40):
        while ctx.execute_pending_job():
            pass
        ctx.eval("__drain()")


def load_node(ctx, defs, name, info=None):
    """Create a node; with `info`, load it the way a workflow does."""
    ctx.eval(f"var __n = __create({json.dumps(defs[name])});")
    if info is not None:
        ctx.eval(f"__n.configure({json.dumps(info)});")
    settle(ctx)
    return ctx


def values(ctx):
    return json.loads(ctx.eval("JSON.stringify(__serialise(__n))"))


def widget_value(ctx, name):
    return json.loads(ctx.eval(
        f"JSON.stringify((__n.widgets.find(function(w){{ return w.name === {json.dumps(name)}; }}) || {{}}).value)"))


def test_dom_widget_serialisation():
    print("\nDOM widgets and widgets_values")
    defs = python_node_defs()
    if defs is None:
        return
    sys.path.insert(0, os.path.join(PACK, "tests"))
    import verify_video_combine_migration as vcm  # the real /bat/video/formats payload

    for js_file, nodes in EDITORS.items():
        ctx = editor_context(js_file)
        ctx.eval(f"__SPECS = {json.dumps(vcm.SPECS)};")
        for name in nodes:
            if name not in defs:        # its module failed to import here
                print(f"skip  {name} (not importable in this environment)")
                continue
            load_node(ctx, defs, name)
            ws = json.loads(ctx.eval("__describe(__n)"))
            doms = [i for i, w in enumerate(ws) if w["dom"]]
            order = " ".join(w["name"] + ("[DOM]" if w["dom"] else "") + ("" if w["ser"] else "(ns)")
                             for w in ws)
            print(f"      {name}: {order}")
            check(f"{name}: has its DOM widget", len(doms) >= 1, True)
            if not doms:
                print("      ", json.loads(ctx.eval("JSON.stringify(__warn)"))[:3])
                continue
            check(f"{name}: DOM widget is serialize:false",
                  all(not ws[i]["ser"] for i in doms), True)
            check(f"{name}: nothing serialised after the DOM widget",
                  [w["name"] for w in ws[doms[0] + 1:] if w["ser"]], [])

            # Round trip. `new` is what a save writes now; `old` is the same
            # save made before the change, the editor's "" in its own slot.
            new = values(ctx)["widgets_values"]
            old = json.loads(ctx.eval("JSON.stringify(__serialiseAsBefore(__n))"))
            got_new = values(load_node(ctx, defs, name, {"widgets_values": new}))["widgets_values"]
            got_old = values(load_node(ctx, defs, name, {"widgets_values": old}))["widgets_values"]
            check(f"{name}: a new save round-trips", got_new, new)
            check(f"{name}: a pre-change save (DOM slot present) loads the same", got_old, new)
            ctx.eval("__warn = [];")

    # ── the nodes that read widgets_values themselves ──────────────────
    missing = [n for n in ("Bat_Grade", "Bat_AnimatedGrade", "Bat_Rescale", "Bat_VideoCombine")
               if n not in defs]
    if missing:
        print(f"skip  legacy-load checks ({', '.join(missing)} not importable here)")
        return
    ctx = editor_context("bat_grade.js")
    ten = [0.05, 0.95, 0.1, 1.1, 1.2, 0.02, 1.0, False, False, 3]
    named_ten = dict(zip(["blackpoint", "whitepoint", "lift", "gain", "multiply", "offset",
                          "gamma", "clamp_white", "clamp_black", "preview_frame"], ten))
    for label, info, want in [
        ("pre-lift_mode save, trailing \"\"", {"widgets_values": ten + [""]}, "legacy"),
        ("pre-lift_mode save, named", {"widgets_values": ten + [""],
                                       "widgets_values_named": {**named_ten, "bat_grade_preview": ""}}, "legacy"),
        ("lift_mode save, trailing \"\"", {"widgets_values": ten + ["nuke", ""]}, "nuke"),
        ("new save (no DOM slot)", {"widgets_values": ten + ["nuke"]}, "nuke"),
        ("new save holding legacy", {"widgets_values": ten + ["legacy"]}, "legacy"),
        ("new save, named", {"widgets_values": ten + ["nuke"],
                             "widgets_values_named": {**named_ten, "lift_mode": "nuke"}}, "nuke"),
    ]:
        load_node(ctx, defs, "Bat_Grade", info)
        check(f"Bat_Grade lift_mode: {label}", widget_value(ctx, "lift_mode"), want)
        check(f"Bat_Grade {label}: values land on the right widgets",
              [widget_value(ctx, "gain"), widget_value(ctx, "preview_frame")], [1.1, 3])

    ctx = editor_context("bat_animated_grade.js")
    state = '{"keyframes":{}}'
    for label, vals, want in [
        ("pre-lift_mode save, trailing \"\"", [state, ""], "legacy"),
        ("lift_mode save, trailing \"\"", [state, "nuke", ""], "nuke"),
        ("new save", [state, "nuke"], "nuke"),
        ("new save holding legacy", [state, "legacy"], "legacy"),
    ]:
        load_node(ctx, defs, "Bat_AnimatedGrade", {"widgets_values": vals})
        check(f"Bat_AnimatedGrade lift_mode: {label}", widget_value(ctx, "lift_mode"), want)

    ctx = editor_context("bat_rescale.js")
    base = ["scale", 0.5, 1024, 1.0, "lanczos", 8, True, 2]      # ... through preview_frame
    for label, vals, want in [
        ("pre-ringing save, trailing \"\"", base + [""], "off"),
        ("ringing save, trailing \"\"", base + ["local", ""], "local"),
        ("new save", base + ["negative"], "negative"),
        ("new save, off", base + ["off"], "off"),
    ]:
        load_node(ctx, defs, "Bat_Rescale", {"widgets_values": vals})
        check(f"Bat_Rescale ringing: {label}", widget_value(ctx, "ringing"), want)
        check(f"Bat_Rescale {label}: multiple_of stays put", widget_value(ctx, "multiple_of"), 8)

    ctx = editor_context("bat_video_combine.js")
    ctx.eval(f"__SPECS = {json.dumps(vcm.SPECS)};")
    for label, vals, want in [
        # Stable layout, h264 saved before bit_depth existed, 10-bit.
        ("Stable-era 10-bit h264, trailing \"\"",
         [24.0, 0, "B", "video/h264-mp4", False, True, 16, "yuv420p10le", "fast", ""],
         {"bit_depth": "10", "pix_fmt": "yuv420p10le", "crf": 16}),
        ("current save, trailing \"\"",
         [24.0, "B", "video/prores-mov", True, "4444", "yuva444p10le", "10", ""],
         {"profile": "4444", "pix_fmt": "yuva444p10le"}),
        ("new save (no DOM slot)",
         [24.0, "B", "video/h265-mp4", True, 20, "yuv420p", "slow", "8"],
         {"crf": 20, "bit_depth": "8", "preset": "slow"}),
    ]:
        load_node(ctx, defs, "Bat_VideoCombine", {"widgets_values": vals})
        got = {k: widget_value(ctx, k) for k in want}
        check(f"Bat_VideoCombine: {label}", got, want)


if __name__ == "__main__":
    sys.exit(main())
