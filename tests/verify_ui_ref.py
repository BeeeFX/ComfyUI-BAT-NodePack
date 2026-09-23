#!/usr/bin/env python
"""
Checks for the preview-payload sidecar: bat_ui_ref.py + web/bat_ui_ref.js.

The editors' `ui` dicts are multi-MB, and ComfyUI keeps every executed `ui` in
its prompt history for the life of the server. The sidecar moves the payload
to a temp file and sends a token instead. What must hold:

Python
1. stash → load round-trips the payload exactly, and `keep` keys stay inline.
2. The stashed ui is tiny (it is what the history stores).
3. A payload that cannot be written falls back to inline, never raises.
4. The folder is pruned oldest-first past its cap, never the file just written.
5. Tokens are validated — no path can be smuggled through load_ui.

JS
6. The wrapper hands the editor the resolved dict, minus the token.
7. A message without a token passes straight through, synchronously.
8. Of two runs resolving out of order, only the newest reaches the editor.
9. An instance-level hook installed in onNodeCreated (the Points Editor's) is
   inside the wrapper, so it sees the resolved message too.

    python tests/verify_ui_ref.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load_module(tmp):
    # Point folder_paths at a scratch temp dir so nothing lands in a real install.
    import types
    fp = types.ModuleType("folder_paths")
    fp.get_temp_directory = lambda: tmp
    sys.modules["folder_paths"] = fp
    spec = importlib.util.spec_from_file_location("bat_ui_ref", os.path.join(PACK, "bat_ui_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_python():
    print("\nbat_ui_ref.py")
    tmp = tempfile.mkdtemp(prefix="bat_ui_ref_")
    try:
        m = load_module(tmp)
        payload = {"frames": ["x" * 50_000] * 4, "meta": [{"w": 1920, "h": 1080}],
                   "images": [{"filename": "a.png", "type": "temp", "subfolder": ""}]}
        ui = m.stash_ui(payload, keep=("images",))
        check("stashed ui carries a token", isinstance(ui.get("bat_ui", [None])[0], str), ui)
        check("keep keys stay inline", ui.get("images") == payload["images"])
        check("the heavy keys are not inline", "frames" not in ui)
        check("the stashed ui is tiny", len(json.dumps(ui)) < 300, len(json.dumps(ui)))
        check("load_ui round-trips exactly", m.load_ui(ui) == payload)
        check("load_ui passes an inline ui through", m.load_ui({"text": ["a"]}) == {"text": ["a"]})

        for bad in ("../../etc/passwd", "zz" * 16, "", 5):
            check(f"token {bad!r} is refused", m.load_ui({"bat_ui": [bad]}) is None)

        unjsonable = {"t": [object()]}
        check("an unserialisable payload falls back to inline",
              m.stash_ui(unjsonable) is unjsonable)
        check("an empty payload is left alone", m.stash_ui({}) == {})

        # Pruning: shrink the cap, write several, the newest must survive.
        m.MAX_BYTES = 120_000
        tokens = []
        for i in range(6):
            u = m.stash_ui({"frames": [str(i) * 40_000]})
            tokens.append(u["bat_ui"][0])
        folder = os.path.join(tmp, "bat_ui")
        left = [f for f in os.listdir(folder) if f.endswith(".json")]
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in left)
        check("the folder is pruned under its cap", size <= m.MAX_BYTES, size)
        check("the payload just written survives the prune",
              os.path.isfile(os.path.join(folder, tokens[-1] + ".json")))
        check("the oldest went first",
              not os.path.isfile(os.path.join(folder, tokens[0] + ".json")))
        check("no .part files are left behind",
              not [f for f in os.listdir(folder) if f.endswith(".part")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.modules.pop("folder_paths", None)


JS_STUBS = r"""
var __ext = null;
var app = { registerExtension(e) { __ext = e; } };
var __store = {};
var __pending = [];
var api = {
    fetchApi(url) {
        const token = decodeURIComponent(url.split("/").pop());
        return new Promise((resolve) => __pending.push(() => resolve({
            ok: token in __store,
            json: async () => __store[token],
        })));
    },
};
var console = { warn() {}, error() {}, log() {} };
var queueMicrotask = (f) => { Promise.resolve().then(f); };   // browsers have it; quickjs does not
function __flushOne(i) { const f = __pending.splice(i, 1)[0]; f && f(); }
"""


def test_js():
    print("\nweb/bat_ui_ref.js")
    try:
        import quickjs
    except ImportError:
        print("  skip  (pip install quickjs)")
        return
    from _harness import strip_modules

    src = open(os.path.join(PACK, "web", "bat_ui_ref.js"), encoding="utf-8").read()
    ctx = quickjs.Context()
    ctx.eval(JS_STUBS)
    ctx.eval(strip_modules(src))

    def settle():
        # Run pending jobs (promise reactions / microtasks) to a fixed point.
        for _ in range(50):
            ctx.execute_pending_job()

    ctx.eval("""
        var got = [];
        function Proto() {}
        Proto.prototype.onExecuted = function (m) { got.push(JSON.stringify(m)); };
        var node = new Proto();
        node.comfyClass = "Bat_Roto";
        node.graph = {};
        __ext.nodeCreated(node);
        // What onNodeCreated does in the Points Editor: an instance-level chain,
        // installed after nodeCreated but before the microtask.
        var instanceSaw = [];
        (function () {
            const orig = node.onExecuted;
            node.onExecuted = function (m) { const r = orig.apply(this, arguments); instanceSaw.push(JSON.stringify(m)); return r; };
        })();
        __store["a".repeat(32)] = { frames: ["F1"], meta: [1] };
        __store["b".repeat(32)] = { frames: ["F2"], meta: [2] };
    """)
    settle()                                     # the deferred wrap happens now

    ctx.eval("""node.onExecuted({ text: ["plain"] });""")
    check("a message without a token passes through synchronously",
          ctx.eval("got.length") == 1 and "plain" in ctx.eval("got[0]"))

    ctx.eval("""
        got = []; instanceSaw = [];
        node.onExecuted({ bat_ui: ["a".repeat(32)], extra: [7] });
        node.onExecuted({ bat_ui: ["b".repeat(32)] });
        __flushOne(1);   // the SECOND run's fetch resolves first
    """)
    settle()
    ctx.eval("__flushOne(0);")                  # then the stale first one
    settle()
    got = json.loads("[" + ",".join(json.loads(ctx.eval("JSON.stringify(got)"))) + "]")
    check("only the newest run reaches the editor", len(got) == 1 and got[0].get("frames") == ["F2"], got)
    check("the resolved message has no token", got and "bat_ui" not in got[0], got)

    ctx.eval("""got = []; node.onExecuted({ bat_ui: ["a".repeat(32)], extra: [7] }); __flushOne(0);""")
    settle()
    got = json.loads("[" + ",".join(json.loads(ctx.eval("JSON.stringify(got)"))) + "]")
    check("other ui keys are kept alongside the payload",
          got and got[0].get("extra") == [7] and got[0].get("frames") == ["F1"], got)
    saw = json.loads(ctx.eval("JSON.stringify(instanceSaw)"))
    check("an instance-level hook from onNodeCreated sees the resolved message",
          saw and "F1" in saw[-1] and "bat_ui" not in saw[-1], saw)

    ctx.eval("""got = []; node.onExecuted({ bat_ui: ["c".repeat(32)] }); __flushOne(0);""")
    settle()
    check("a payload that is gone (server restart) is dropped quietly",
          ctx.eval("got.length") == 0)

    ctx.eval("""
        var other = new Proto(); other.comfyClass = "KSampler"; other.graph = {};
        var before = other.onExecuted;
        __ext.nodeCreated(other);
    """)
    settle()
    check("non-BAT nodes are not wrapped", ctx.eval("other.onExecuted === before"))


def main():
    test_python()
    test_js()
    print()
    if _failures:
        print(f"FAILURES: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
