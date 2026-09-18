#!/usr/bin/env python
"""
Behavioural checks for the ten utility nodes added on 2026-09-17.

These exist so ``ComfyUI-Easy-Use`` and ``comfyui-art-venture`` can be removed
from the studio installs. That makes the migration path, not the nodes, the
risky part: a workflow someone opens next month gets rewritten in place by
ETC_Core's popup, and a wrong widget index or output slot silently changes
what that workflow computes. Six claims are load-bearing enough to test:

1. **Migration is lossless for `easy compare`.** Its operator list is a widget
   value that carries straight across. If BAT's list is missing an entry the
   migrated node comes back set to something the artist never chose — and
   four of easy-use's ten operators ("a > 0" and friends) are ones no sane
   person would design in today. They are kept for exactly this reason.

2. **`Bat_IndexSwitch` has as many branches as were reachable before.**
   easy-use built 20 input slots and clamped its index widget to 0-9. Ten is
   therefore the real number, and fewer would drop live links.

3. **`StringToInt` lands on the INT output.** Core's `ComfyNumberConvert`
   emits FLOAT on slot 0 and INT on slot 1. Migrating without moving the
   output link turns every frame count in the graph into a float.

4. **The display nodes' `enabled` toggle actually gates evaluation.**
   `OUTPUT_NODE` is read off the class by execution.py and cannot vary per
   instance, so laziness is the only mechanism that can stop a parked readout
   pinning its upstream branch into every queue. If `check_lazy_status`ceases
   to return `[]` when disabled, the toggle becomes decorative.

5. **Passthrough is identity.** These nodes get inserted mid-wire. Returning a
   copy, or a rendering, instead of the object itself would be an invisible
   change to whatever is downstream.

6. **`Bat_ListIndex` keeps the batch dimension.** `t[i]` drops it and the
   result stops being a valid IMAGE; `t[i:i+1]` does not.

    pip install quickjs
    python tests/verify_utility_nodes.py
"""

import importlib
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load_pack_modules():
    """Import the new modules without the pack __init__.

    __init__ pulls in bat_video_loader, which registers aiohttp routes at
    import time and therefore needs a live PromptServer. Nothing here does.
    """
    pkg = types.ModuleType("batpkg")
    pkg.__path__ = [PACK]
    sys.modules["batpkg"] = pkg
    if PACK not in sys.path:
        sys.path.insert(0, PACK)
    return (importlib.import_module("batpkg.bat_convert"),
            importlib.import_module("batpkg.bat_logic"),
            importlib.import_module("batpkg.bat_show"))


# ─── 1. registration ─────────────────────────────────────────────────────────

NEW_NODES = [
    "Bat_ShowAny", "Bat_ShowTensorShape", "Bat_ConvertAny", "Bat_AnyToString",
    "Bat_NumberToString", "Bat_Compare", "Bat_IndexSwitch", "Bat_ListLength",
    "Bat_ListIndex", "Bat_ListBatch",
]


def test_registration():
    print("\nregistration")
    src = open(os.path.join(PACK, "__init__.py"), encoding="utf-8").read()
    classes = set(re.findall(r'"(Bat_[A-Za-z0-9_]+)":\s+[A-Z]', src))
    names = set(re.findall(r'"(Bat_[A-Za-z0-9_]+)":\s+"', src))
    for n in NEW_NODES:
        check(f"{n} in NODE_CLASS_MAPPINGS", n in classes)
        check(f"{n} has a display name", n in names)
    check("no node without a display name", not (classes - names),
          str(sorted(classes - names)))


# ─── 2. conversion ───────────────────────────────────────────────────────────

def test_convert(bc):
    print("\nconversion")
    c = bc.BatConvertAny()
    check("'3.7' -> int truncates", c.convert("3.7", "int") == (3,))
    check("'3.7' -> float parses", c.convert("3.7", "float") == (3.7,))
    # bool("False") is True in Python; the whole point of the falsey table.
    check("'false' -> boolean is False", c.convert("false", "boolean") == (False,))
    check("'0' -> boolean is False", c.convert("0", "boolean") == (False,))
    check("'yes' -> boolean is True", c.convert("yes", "boolean") == (True,))
    check("ConvertAny is not an OUTPUT_NODE",
          getattr(bc.BatConvertAny, "OUTPUT_NODE", False) is False,
          "an output node is an unconditional execution root")

    n = bc.BatNumberToString()
    check("42 pad4 -> '0042'", n.to_string(42, -1, 4) == ("0042",))
    check("3 pad3 prefix v -> 'v003'", n.to_string(3, -1, 3, "v") == ("v003",))
    check("3.14159 dec2 -> '3.14'", n.to_string(3.14159, 2, 0) == ("3.14",))
    check("-7 pad4 keeps sign -> '-0007'", n.to_string(-7, -1, 4) == ("-0007",))
    check("2.5 dec1 pad4 pads whole part only",
          n.to_string(2.5, 1, 4) == ("0002.5",))
    check("int stays int-shaped", n.to_string(5, -1, 0) == ("5",))


# ─── 3. logic ────────────────────────────────────────────────────────────────

# Verbatim from comfyui-easy-use py/nodes/logic.py COMPARE_FUNCTIONS.
EASY_OPS = ["a == b", "a != b", "a < b", "a > b", "a <= b", "a >= b",
            "a > 0", "a <= 0", "b > 0", "b <= 0"]


def test_logic(bl):
    print("\nlogic")
    missing = [o for o in EASY_OPS if o not in bl.COMPARE_OPS]
    check("every easy-use compare operator survives migration",
          not missing, f"missing {missing}")

    c = bl.BatCompare()
    check("5 > 3", c.compare(5, 3, "a > b") == (True,))
    check("a > 0 ignores b", c.compare(5, None, "a > 0") == (True,))

    check("Bat_IndexSwitch exposes 10 branches", bl.MAX_BRANCHES == 10,
          "easy-use's index widget allowed 0-9")
    s = bl.BatIndexSwitch()
    opt = bl.BatIndexSwitch.INPUT_TYPES()["optional"]
    check("every branch input is lazy",
          all(v[1].get("lazy") for v in opt.values()),
          "without lazy, all branches execute and all but one are discarded")
    check("only the selected branch is requested",
          s.check_lazy_status(index=3) == ["value3"])
    check("branch 9 is selectable", s.switch(index=9, value9="nine") == ("nine",))
    try:
        s.switch(index=2, value2=None)
        check("unwired branch raises", False, "no error raised")
    except ValueError as e:
        check("unwired branch names the input", "value2" in str(e))


def test_lists(bl):
    print("\nlists")
    L, I, B = bl.BatListLength(), bl.BatListIndex(), bl.BatListBatch()
    check("length of a list", L.length([1, 2, 3]) == (3,))
    check("length of None is 0", L.length(None) == (0,))
    check("negative index", I.pick([10, 20, 30], -1) == (30,))
    try:
        I.pick([1, 2], 9)
        check("out-of-range index raises", False)
    except IndexError as e:
        check("out-of-range index explains itself", "out of range" in str(e))
    check("join lists", B.join([1, 2], [3]) == ([1, 2, 3],))
    check("join passes through a missing side", B.join(None, [9]) == ([9],))

    try:
        import torch
    except ImportError:
        print("  skip  tensor checks (no torch)")
        return
    batch = torch.rand(10, 64, 64, 3)
    out = I.pick(batch, -1)[0]
    check("ListIndex keeps the batch dimension", tuple(out.shape) == (1, 64, 64, 3),
          f"got {tuple(out.shape)} — t[i] would give (64, 64, 3)")
    check("ListLength of a batch is its frame count", L.length(batch) == (10,))
    joined = B.join(torch.rand(3, 64, 64, 3), torch.rand(2, 64, 64, 3))[0]
    check("join concatenates along frames", tuple(joined.shape) == (5, 64, 64, 3))
    try:
        B.join(torch.rand(3, 64, 64, 3), torch.rand(2, 32, 32, 3))
        check("mismatched batches raise", False)
    except ValueError as e:
        check("mismatched batches say which shapes", "64" in str(e) and "32" in str(e))


# ─── 4. display nodes ────────────────────────────────────────────────────────

def test_show(bs):
    print("\ndisplay nodes")
    for cls in (bs.BatShowAny, bs.BatShowTensorShape):
        name = cls.__name__
        check(f"{name} is an OUTPUT_NODE", cls.OUTPUT_NODE is True,
              "a dead-end readout never executes otherwise")
        inst = cls()
        check(f"{name} enabled -> requests its input",
              inst.check_lazy_status(enabled=True) == ["value"])
        check(f"{name} disabled -> requests nothing",
              inst.check_lazy_status(enabled=False) == [],
              "this is the only thing that keeps a parked readout from "
              "pinning its upstream branch into every queue")
        opt = cls.INPUT_TYPES()["optional"]["value"][1]
        check(f"{name} value input is lazy", opt.get("lazy") is True)

    a = bs.BatShowAny()
    sentinel = {"frames": 48}
    out = a.show(True, sentinel)
    check("ShowAny passes the same object through",
          out["result"][0] is sentinel, "a copy would be an invisible change")
    check("ShowAny renders text", "48" in out["ui"]["text"][0])
    off = a.show(False, None)
    check("ShowAny says it is disabled", "disabled" in off["ui"]["text"][0])
    check("ShowAny outputs None while disabled", off["result"] == (None,))

    t = bs.BatShowTensorShape()
    try:
        import torch
    except ImportError:
        print("  skip  tensor report (no torch)")
        return
    x = torch.rand(48, 1080, 1920, 3)
    r = t.show(True, x)
    text = r["ui"]["text"][0]
    check("ShowTensorShape passes the tensor through", r["result"][0] is x)
    check("report names the shape", "[48, 1080, 1920, 3]" in text)
    check("report labels the layout", "1920x1080" in text and "3 ch" in text)
    check("report gives dtype and device", "float32" in text and "cpu" in text)
    check("report gives memory size", "MiB" in text)
    check("info output mirrors the panel", r["result"][1] == text)


# ─── 5. migrations ───────────────────────────────────────────────────────────

EXPECTED_MIGRATIONS = {
    "easy showAnything": "Bat_ShowAny",
    "easy showTensorShape": "Bat_ShowTensorShape",
    "easy convertAnything": "Bat_ConvertAny",
    "easy compare": "Bat_Compare",
    "easy lengthAnything": "Bat_ListLength",
    "easy indexAnything": "Bat_ListIndex",
    "easy batchAnything": "Bat_ListBatch",
    "easy anythingIndexSwitch": "Bat_IndexSwitch",
    "StringToInt": "ComfyNumberConvert",
    "StringToNumber": "ComfyNumberConvert",
    "BooleanPrimitive": "PrimitiveBoolean",
    "easy string": "PrimitiveString",
    "easy boolean": "PrimitiveBoolean",
    "easy int": "PrimitiveInt",
    "easy float": "PrimitiveFloat",
    "Qwen2.5VL": "AILab_QwenVL_Advanced",
}


def test_migrations():
    print("\nmigrations")
    try:
        import quickjs
    except ImportError:
        print("  skip  (pip install quickjs)")
        return

    src = open(os.path.join(PACK, "web", "bat-migrations.js"), encoding="utf-8").read()
    ctx = quickjs.Context()
    ctx.eval("""
        var __got = {};
        var console = {log:function(){}, warn:function(){}};
        var window = {ETC:{registerNodeMigration:function(m){ __got[m.from] = m; }}};
        var queueMicrotask = function(f){ f(); };
        var setTimeout = function(){};
    """)
    ctx.eval(src)

    for legacy, target in EXPECTED_MIGRATIONS.items():
        got = ctx.eval(f"__got[{legacy!r}] ? __got[{legacy!r}].to : null")
        check(f"{legacy} -> {target}", got == target, f"got {got!r}")

    # The one that silently corrupts a graph if it regresses.
    slot = ctx.eval("__got['StringToInt'].mapOutputs ? "
                    "__got['StringToInt'].mapOutputs(0) : -1")
    check("StringToInt output moves to the INT slot", slot == 1,
          "ComfyNumberConvert emits FLOAT on 0, INT on 1")

    seeded = ctx.eval("JSON.stringify(__got['easy showAnything'].mapWidgetValues([]))")
    check("showAnything migrates as enabled", seeded == "[true]",
          "otherwise the readout comes back switched off")

    # BAT targets must actually be registered, or the popup offers a
    # replacement that cannot be placed.
    init = open(os.path.join(PACK, "__init__.py"), encoding="utf-8").read()
    registered = set(re.findall(r'"(Bat_[A-Za-z0-9_]+)":\s+[A-Z]', init))
    for legacy, target in EXPECTED_MIGRATIONS.items():
        if target.startswith("Bat_"):
            check(f"target {target} is registered", target in registered)


# ─── 6. the front-end panel ──────────────────────────────────────────────────

def test_show_js():
    print("\nweb/bat_show.js")
    try:
        import quickjs
    except ImportError:
        print("  skip  (pip install quickjs)")
        return
    sys.path.insert(0, HERE)
    from _harness import auto_stub_js, strip_modules

    src = open(os.path.join(PACK, "web", "bat_show.js"), encoding="utf-8").read()
    ctx = quickjs.Context()
    ctx.eval(auto_stub_js(src))
    ctx.eval("""
        var __el = null;
        var document = {createElement: function(){
            __el = {style:{}, textContent:"", scrollTop: 0};
            return __el;
        }};
        var console = {log:function(){}, warn:function(){}, error:function(){}};
        var __ext = null;
        var app = {registerExtension: function(o){ __ext = o; }};
        function addBatDOMWidget(node, name){ return {name: name}; }
        function refreshBatLayout(){}
    """)
    ctx.eval(strip_modules(src))

    check("extension registers", ctx.eval("__ext && __ext.name") == "BAT.ShowNodes")

    ctx.eval("""
        var proto = {};
        var nodeType = {prototype: proto};
        __ext.beforeRegisterNodeDef(nodeType, {name: "Bat_ShowAny"});
        var node = Object.create(proto);
        node.id = 7;
        node.onNodeCreated();
    """)
    check("panel starts with a placeholder",
          "not run" in ctx.eval("__el.textContent"))

    ctx.eval("node.onExecuted({text: ['shape [48, 1080, 1920, 3]']})")
    check("panel shows the payload", "1920" in ctx.eval("__el.textContent"))

    # Undo is a full loadGraphData: the node object is thrown away and rebuilt.
    ctx.eval("""
        var node2 = Object.create(proto);
        node2.id = 7;
        node2.onNodeCreated();
    """)
    check("payload survives an undo rebuild",
          "1920" in ctx.eval("__el.textContent"),
          "a rebuilt node with the same id must replay its last text")

    ctx.eval("""
        var other = Object.create(proto);
        other.id = 99;
        other.onNodeCreated();
    """)
    check("an unrelated node does not inherit it",
          "not run" in ctx.eval("__el.textContent"))


def _find_qwen_config():
    """ComfyUI-QwenVL's config.json, if that pack sits beside this one.

    The two packs are deployed independently, so this check is best-effort:
    it upgrades the assertion where the data is available rather than failing
    on an install that simply does not carry the pack.
    """
    import json as _json
    custom_nodes = os.path.dirname(PACK)
    candidate = os.path.join(custom_nodes, "ComfyUI-QwenVL", "config.json")
    if not os.path.isfile(candidate):
        return None
    try:
        with open(candidate, encoding="utf-8") as fh:
            return _json.load(fh)
    except Exception:
        return None


# ─── 7. the QwenVL consolidation ─────────────────────────────────────────────

def test_qwen_migration():
    """alexcong Qwen2.5VL -> 1038lab AILab_QwenVL_Advanced.

    Nine widgets become sixteen in a different order, so everything here is a
    hand-written map and every entry is a chance to put a value in the wrong
    slot. The input map matters even more: a workflow that converted `seed` to
    an input puts it at slot 1, which is where the target's `video` socket
    lives — a positional map would wire a seed into the video input and nobody
    would notice until the captions came back wrong.
    """
    print("\nQwenVL consolidation")
    try:
        import quickjs
    except ImportError:
        print("  skip  (pip install quickjs)")
        return

    src = open(os.path.join(PACK, "web", "bat-migrations.js"), encoding="utf-8").read()
    ctx = quickjs.Context()
    ctx.eval("""
        var __got = {}; var __warn = [];
        var console = {log:function(){}, warn:function(m){ __warn.push(String(m)); }};
        var window = {ETC:{registerNodeMigration:function(m){ __got[m.from] = m; }}};
        var queueMicrotask = function(f){ f(); };
        var setTimeout = function(){};
    """)
    ctx.eval(src)

    import json as _json

    # The exact widgets_values from the two studio workflows.
    studio = ["PROMPT TEXT", "Qwen2.5-VL-3B-Instruct", "none", False,
              0.7, 1024, 1, "fixed", ""]
    ctx.eval(f"var __o = {_json.dumps(studio)};")
    out = _json.loads(ctx.eval("JSON.stringify(__got['Qwen2.5VL'].mapWidgetValues(__o))"))

    check("widget count matches the target node", len(out) == 16, f"got {len(out)}")
    check("prompt lands in custom_prompt (index 6)", out[6] == "PROMPT TEXT",
          "custom_prompt overrides preset_prompt in AILab_QwenVL.py:368-370")
    check("model carries over", out[0] == "Qwen2.5-VL-3B-Instruct")
    check("quantization 'none' -> 'None (FP16)'", out[1] == "None (FP16)")
    check("max_new_tokens -> max_tokens (index 7)", out[7] == 1024)
    check("temperature preserved (index 8)", out[8] == 0.7)
    check("keep_model_loaded preserved (index 13)", out[13] is False)
    # preset_prompt is inert once custom_prompt is set, but it still has to be a
    # value the combo accepts or the node fails validation. Check it against the
    # pack's real prompt list where that pack is installed alongside this one —
    # a hand-typed emoji literal is exactly the kind of thing that drifts (the
    # first version of this test lost the U+FE0F variation selector).
    preset = out[5]
    cfg = _find_qwen_config()
    if cfg:
        presets = cfg.get("_preset_prompts", [])
        check("preset_prompt is one the target actually offers",
              preset in presets, f"{preset!r} not in {len(presets)} presets")
    else:
        print("  skip  preset_prompt list check (ComfyUI-QwenVL not installed here)")
        check("preset_prompt is a non-empty string",
              isinstance(preset, str) and preset.strip() != "")

    # Edge cases that would otherwise produce an invalid node.
    ctx.eval("__warn = [];")
    ctx.eval('var __sky = ["t","SkyCaptioner-V1","8bit",true,0.0,128,-1,"fixed","/tmp/x.mp4"];')
    sky = _json.loads(ctx.eval("JSON.stringify(__got['Qwen2.5VL'].mapWidgetValues(__sky))"))
    warns = ctx.eval("__warn.join(' | ')")
    check("unknown model falls back", sky[0] == "Qwen2.5-VL-7B-Instruct")
    check("unknown model warns", "SkyCaptioner" in warns)
    check("quantization '8bit' maps", sky[1] == "8-bit (Balanced)")
    check("temperature 0 clamped to the target's floor", sky[8] == 0.1,
          "alexcong allowed 0; 1038lab's minimum is 0.1")
    check("seed -1 clamped to 1", sky[14] == 1,
          "alexcong used -1 for random; 1038lab's minimum is 1")
    check("a set video_path warns", "video_path" in warns,
          "the replacement takes a piped IMAGE, not a path")

    # Input remapping is by name, never by index.
    ctx.eval("""
        var newNode = {inputs: [{name:"image"}, {name:"video"}]};
        var mi = __got['Qwen2.5VL'].mapInputs;
        var __image = mi(0, "image", null, newNode);
        var __seed  = mi(1, "seed",  null, newNode);
    """)
    check("image maps to the image slot", ctx.eval("__image") == 0)
    check("a converted seed does NOT land on video", ctx.eval("__seed") != 1,
          "slot 1 on the target is the video input")
    check("an unmatched input is refused, not guessed", ctx.eval("__seed") == -1)

    # And when the target does expose the widget as a socket, use it.
    ctx.eval("""
        var withSeed = {inputs: [{name:"image"}, {name:"video"}, {name:"seed"}]};
        var __seed2 = __got['Qwen2.5VL'].mapInputs(1, "seed", null, withSeed);
    """)
    check("a seed socket is found by name when present", ctx.eval("__seed2") == 2)


def main():
    bc, bl, bs = load_pack_modules()
    test_registration()
    test_convert(bc)
    test_logic(bl)
    test_lists(bl)
    test_show(bs)
    test_migrations()
    test_show_js()
    test_qwen_migration()

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
