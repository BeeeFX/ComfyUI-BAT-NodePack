"""Prove Bat_VideoCombine loads Stable-era workflows with the right values.

`loop_count` and `pingpong` were removed from the node on 2026-08-06. ComfyUI
restores widgets_values positionally, so every workflow saved before that lands
two slots out of alignment: filename_prefix takes loop_count's value, format
takes the prefix, save_output takes the format string, and the codec tail is
read two slots early. web/bat_video_combine.js migrates those arrays by name
(STATIC_LAYOUTS / matchStaticLayout / applyStaticLayout); this checks it works.

There is no node binary on this box, so the extension is driven under quickjs
(see the js-verification note): imports are stubbed, the real /bat/video/formats
payload is built from bat_video_formats/*.json, and the loader's exact sequence
is replayed — positional restore first, then onConfigure — for a workflow saved
by each historical layout. Run it with the RND venv python:

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_video_combine_migration.py
"""
import ast
import json
import os
import re
import sys

import quickjs

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
JS = os.path.join(PACK, "web", "bat_video_combine.js")


def format_specs():
    """Rebuild the /bat/video/formats payload straight from the JSONs."""
    out = {}
    for fn in sorted(os.listdir(os.path.join(PACK, "bat_video_formats"))):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PACK, "bat_video_formats", fn), encoding="utf-8") as fh:
            fmt = json.load(fh)
        widgets = []
        for name, spec in (fmt.get("widgets") or {}).items():
            w = {"name": name, "type": spec.get("type", "STRING"),
                 "label": spec.get("label", name), "default": spec.get("default"),
                 "hidden": bool(spec.get("hidden", False))}
            for k in ("options", "min", "max"):
                if k in spec:
                    w[k] = spec[k]
            widgets.append(w)
        out[fmt["label"]] = {"widgets": widgets, "derived": fmt.get("derived") or {}}
    return out


SPECS = format_specs()
FORMAT_LABELS = sorted(SPECS)

# Enough of a browser/ComfyUI to get the extension registered. buildPlayer's DOM
# work is guarded by a try/catch in onNodeCreated, so a thin document stub is
# fine — none of it touches the widget values this test is about.
STUBS = """
var __log = [];
var console = {
    log:   function () { __log.push(["log", Array.prototype.join.call(arguments, " ")]); },
    info:  function () { __log.push(["info", Array.prototype.join.call(arguments, " ")]); },
    warn:  function () { __log.push(["warn", Array.prototype.join.call(arguments, " ")]); },
    error: function () { __log.push(["error", Array.prototype.join.call(arguments, " ")]); },
};
var setTimeout = function (fn) { try { fn(); } catch (e) {} return 0; };
var clearTimeout = function () {};
var requestAnimationFrame = function () { return 0; };
var cancelAnimationFrame = function () {};
var document = undefined;
var window = undefined;
var localStorage = undefined;

var __ext = null;
var app = { registerExtension: function (ext) { __ext = ext; }, graph: null, canvas: null };
var api = {
    fetchApi: function (path) {
        if (path === "/bat/video/formats") {
            return Promise.resolve({ json: function () { return Promise.resolve(__SPECS); } });
        }
        return Promise.reject(new Error("unstubbed fetch " + path));
    },
    addEventListener: function () {},
};
var batTrack = function () {};
var batNodeCacheKey = function () { return "test-key"; };
var addBatDOMWidget = function (node, name, type, el, opts) {
    var w = { name: name, type: type, element: el, options: { serialize: false } };
    node.widgets.push(w);
    return w;
};
var clampNodeSize = function () {};
"""

# The node's serialisable widgets as the CURRENT Python INPUT_TYPES declares
# them, in order, with their defaults — this is what ComfyUI builds before it
# restores anything.
CURRENT_STATICS = [
    ("frame_rate", "number", 24.0, None),
    ("filename_prefix", "string", "BatVideo", None),
    ("format", "combo", "video/h264-mp4", FORMAT_LABELS),
    ("save_output", "toggle", True, None),
]

HARNESS = """
// Build a fresh node the way ComfyUI does, run the extension's hooks over it,
// then replay the loader: positional widgets_values restore, then onConfigure.
function loadWorkflowNode(savedVals) {
    var node = {
        widgets: [], size: [400, 400], inputs: [], outputs: [], id: 1,
        addCustomWidget: function (w) { this.widgets.push(w); return w; },
        removeWidget: function (w) {
            var i = this.widgets.indexOf(w);
            if (i !== -1) this.widgets.splice(i, 1);
        },
        setDirtyCanvas: function () {},
        computeSize: function () { return [400, 400]; },
        onResize: function () {},
        graph: { _nodes: [], getLink: function () { return null; } },
    };
    __STATIC_WIDGETS__.forEach(function (s) {
        node.widgets.push({
            name: s.name, type: s.type, value: s.def,
            options: s.values ? { values: s.values } : {},
            callback: null,
        });
    });

    // beforeRegisterNodeDef patches nodeType.PROTOTYPE, then ComfyUI news up a
    // node off it — flattened here onto the plain object above.
    var nodeType = { prototype: {
        configure: function (info) { return this.__baseConfigure(info); },
    } };
    __ext.beforeRegisterNodeDef(nodeType, __NODE_DATA__, app);
    for (var k in nodeType.prototype) node[k] = nodeType.prototype[k];
    if (node.onNodeCreated) node.onNodeCreated();

    // LiteGraph's own configure(): deal widgets_values out positionally to the
    // serialisable widgets, then fire onConfigure with the same info object.
    // The extension patches configure() ahead of this, which is the whole
    // point of the test — the migration has to land before the deal.
    node.__baseConfigure = function (info) {
        var vals = info.widgets_values || [];
        var serialisable = this.widgets.filter(function (w) {
            return w.name && (!w.options || w.options.serialize !== false);
        });
        for (var i = 0; i < serialisable.length && i < vals.length; i++) {
            serialisable[i].value = vals[i];
        }
        if (this.onConfigure) this.onConfigure(info);
    };

    node.configure({ widgets_values: savedVals });
    return node;
}

// name -> value for every serialisable widget, i.e. what would be sent to the
// backend if the artist hit Queue right after opening the workflow.
function widgetValues(node) {
    var out = {};
    node.widgets.forEach(function (w) {
        if (!w.name) return;
        if (w.options && w.options.serialize === false) return;
        out[w.name] = w.value;
    });
    return out;
}

var __nodes = {};
function run(key, savedVals) { __nodes[key] = loadWorkflowNode(savedVals); }
function values(key) { return JSON.stringify(widgetValues(__nodes[key])); }
function logs() { return JSON.stringify(__log); }
"""


def build_context():
    src = open(JS, encoding="utf-8").read()
    body = re.sub(r"(?ms)^[ \t]*import\s.*?;[ \t]*$", "", src)
    body = re.sub(r"(?m)^\s*export\s+(default\s+)?", "", body)

    statics = [{"name": n, "type": t, "def": d, "values": v}
               for n, t, d, v in CURRENT_STATICS]
    # nodeData as ComfyUI serves it from INPUT_TYPES — the migration reads the
    # declared defaults off it to recover from a retired combo value.
    node_data = {"name": "Bat_VideoCombine", "input": {"required": {
        "images": ["IMAGE"],
        "frame_rate": ["FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}],
        "filename_prefix": ["STRING", {"default": "BatVideo"}],
        "format": [FORMAT_LABELS, {"default": "video/h264-mp4"}],
        "save_output": ["BOOLEAN", {"default": True}],
    }}}
    harness = (HARNESS.replace("__STATIC_WIDGETS__", json.dumps(statics))
                      .replace("__NODE_DATA__", json.dumps(node_data)))
    stubs = STUBS.replace("__SPECS", json.dumps(SPECS))

    ctx = quickjs.Context()
    ctx.eval(stubs + "\n" + body + "\n" + harness)
    return ctx


def pump(ctx, rounds=200):
    """quickjs doesn't drain the microtask queue on its own."""
    for _ in range(rounds):
        try:
            if not ctx.execute_pending_job():
                return
        except Exception:
            return


def python_static_widgets():
    """The node's current widget order, read out of INPUT_TYPES with ast.

    Parsed rather than imported: importing the module drags in torch and a
    running PromptServer, and all we need is the shape of one dict literal.
    A required input counts as a widget when its type is a primitive or a
    COMBO (a list of choices); anything else — IMAGE here — is a link slot and
    never appears in widgets_values.
    """
    tree = ast.parse(open(os.path.join(PACK, "bat_video_combine.py"), encoding="utf-8").read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES"):
            continue
        ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
        for key, val in zip(ret.value.keys, ret.value.values):
            if key.value != "required":
                continue
            names = []
            for k, v in zip(val.keys, val.values):
                spec = v.elts[0]
                is_combo = not (isinstance(spec, ast.Constant) and isinstance(spec.value, str))
                if is_combo or spec.value in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                    names.append(k.value)
            return names
    raise AssertionError("could not find INPUT_TYPES in bat_video_combine.py")


FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}\n         got  {got!r}\n         want {want!r}")
        FAILURES.append(label)


def load(ctx, key, saved):
    ctx.eval(f"run({json.dumps(key)}, {json.dumps(saved)})")
    pump(ctx)
    return json.loads(ctx.eval(f"values({json.dumps(key)})"))


def main():
    ctx = build_context()

    # The whole migration hinges on STATIC_LAYOUTS[0] describing the node as it
    # is TODAY. If someone changes INPUT_TYPES without pushing a new layout on
    # the front of that list, every check below still passes while real
    # workflows silently shift again — so assert the two agree first.
    print("STATIC_LAYOUTS[0] matches the Python INPUT_TYPES:")
    check("layout head", json.loads(ctx.eval("JSON.stringify(STATIC_LAYOUTS[0])")),
          python_static_widgets())

    print()

    # ── A Stable-era ProRes workflow. Layout:
    #    frame_rate, loop_count, filename_prefix, format, pingpong, save_output
    #    then the codec tail in prores-mov.json order: profile, pix_fmt.
    print("Stable-era workflow (loop_count + pingpong present), ProRes:")
    v = load(ctx, "prores", [25.0, 0, "SHOT_010_comp", "video/prores-mov",
                             False, False, "4444", "yuva444p10le"])
    check("frame_rate", v.get("frame_rate"), 25.0)
    check("filename_prefix", v.get("filename_prefix"), "SHOT_010_comp")
    check("format", v.get("format"), "video/prores-mov")
    check("save_output", v.get("save_output"), False)
    check("profile (codec tail)", v.get("profile"), "4444")
    check("loop_count dropped", "loop_count" in v, False)
    check("pingpong dropped", "pingpong" in v, False)
    # bit_depth was appended to prores-mov.json after this workflow was saved,
    # so it has no tail entry; it must be inferred back off the saved pix_fmt
    # rather than silently defaulting a 10-bit deliverable down to 8.
    check("bit_depth inferred from saved pix_fmt", v.get("bit_depth"), "10")

    # ── Same vintage, h264, non-default loop_count/pingpong and save_output on.
    print("\nStable-era workflow, h264, pingpong on:")
    v = load(ctx, "h264", [24.0, 3, "BatVideo", "video/h264-mp4",
                           True, True, 12, "yuv420p", "slow"])
    check("frame_rate", v.get("frame_rate"), 24.0)
    check("filename_prefix", v.get("filename_prefix"), "BatVideo")
    check("format", v.get("format"), "video/h264-mp4")
    check("save_output", v.get("save_output"), True)
    check("crf (codec tail)", v.get("crf"), 12)
    check("preset (codec tail)", v.get("preset"), "slow")

    # ── A workflow saved by the CURRENT layout must be untouched.
    print("\nCurrent-layout workflow, EXR sequence:")
    v = load(ctx, "exr", [48.0, "SHOT_020_lin", "image/exr-sequence", True,
                          "gbrapf32le", "dwaa", "true", "32f", "FLOAT", 45, "dwaa"])
    check("frame_rate", v.get("frame_rate"), 48.0)
    check("filename_prefix", v.get("filename_prefix"), "SHOT_020_lin")
    check("format", v.get("format"), "image/exr-sequence")
    check("save_output", v.get("save_output"), True)
    check("bit_depth (codec tail)", v.get("bit_depth"), "32f")

    # ── A format that no longer exists must not stick on the combo, but the
    #    widgets around it must still land on the right names.
    print("\nStable-era workflow naming a dropped format (av1-webm):")
    v = load(ctx, "gone", [24.0, 0, "OLD_SHOT", "video/av1-webm", False, True, 30])
    check("filename_prefix still migrated", v.get("filename_prefix"), "OLD_SHOT")
    check("format falls back to the declared default", v.get("format"), "video/h264-mp4")

    # ── An array from no known layout must be left alone, not scrambled.
    print("\nUnrecognisable widgets_values:")
    v = load(ctx, "junk", ["nonsense", "nonsense", "nonsense"])
    logs = json.loads(ctx.eval("logs()"))
    warned = any(lvl == "warn" and "no known static" in msg for lvl, msg in logs)
    check("warns instead of guessing", warned, True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
