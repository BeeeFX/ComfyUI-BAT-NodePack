"""Bat_Framehold — prove the JS frame-spec parser matches the Python one.

The on-node scrubber highlights the selection and reports the output length
BEFORE you run, which is only worth anything if its parser agrees with the one
that actually gates the batch. The two live in different languages (see the
NOTE in bat_framehold._parse_frames), so this runs both over the same cases.

    env/bin/python tools/test_framehold_parity.py

Needs `quickjs` (pip install quickjs); there is no node binary on the boxes.
The JS module's imports are stripped and its ComfyUI globals stubbed — the
parser itself touches none of them.
"""

import os
import re
import sys
import types
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)


def load_python():
    """Import bat_framehold with `server` stubbed (no PromptServer here)."""
    srv = types.ModuleType("server")

    class _Routes:
        def get(self, _path):
            return lambda f: f

    srv.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=_Routes()))
    srv.web = types.SimpleNamespace(json_response=None, Response=None)
    sys.modules.setdefault("server", srv)

    spec = importlib.util.spec_from_file_location(
        "bat_framehold", os.path.join(PACK, "bat_framehold.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_js():
    import quickjs

    src = open(os.path.join(PACK, "web", "bat_framehold.js"), encoding="utf-8").read()
    # Matches multi-line import blocks too, not just one-liners.
    src = re.sub(r"^import\s[\s\S]*?from\s+['\"][^'\"]+['\"];", "", src, flags=re.M)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval("""
        var window = { devicePixelRatio: 1 };
        var localStorage = { getItem: function(){return null;}, setItem: function(){} };
        var document = { createElement: function(){ return { style:{}, append:function(){},
                          appendChild:function(){}, addEventListener:function(){},
                          setAttribute:function(){}, removeAttribute:function(){} }; } };
        var app = { registerExtension: function(){} };
        var api = { apiURL: function(u){ return u; } };
        function addBatDOMWidget(){ return null; }
        function clampNodeSize(){}
        function batTrack(){ return { interval:function(){}, dispose:function(){} }; }
        function isNodeAlive(){ return true; }
        function batNodeCacheKey(){ return "k"; }
    """)
    ctx.eval(src)                                  # also a syntax check
    ctx.eval("function _parse(s, n){ var r = parseFrameSpec(s, n);"
             "return JSON.stringify(r.error ? {e:r.error} : {l:r.list}); }")
    ctx.eval("function _anchor(t, n){ return JSON.stringify(tokenStart(t, n)); }")
    return ctx


CASES = [
    ("0", 50), ("8", 50), ("0-10", 50), ("0, 5", 50), ("0-2, 7, 10-12", 50),
    ("10-5", 50), ("-1", 50), ("-5--1", 50), ("5, 0, 5", 50),
    ("", 50), ("   ", 50), (",,", 50),
    ("999", 50), ("-999", 50), ("40-999", 50), ("0-0", 50),
    (" 3 - 7 ", 50), ("3,", 50), ("0-10", 1), ("-1", 1),
    ("2", 3), ("-2", 3), ("-3--1", 3),
    # malformed — both sides must refuse, not guess
    ("5-", 50), ("--1", 50), ("abc", 50), ("1.5", 50), ("1-2-3", 50), ("1 2", 50),
]


# The R (range) key rewrites the last token as anchor→playhead, so the anchor
# it picks out of a token is the whole behaviour. JS-only — there is no Python
# counterpart to compare against, just the intended semantics.
ANCHOR_CASES = [
    ("3", 50, 3), ("3-7", 50, 3), ("7-3", 50, 7), (" 12 ", 50, 12),
    ("-1", 50, 49), ("-5--1", 50, 45), ("abc", 50, None),
]


def main():
    py = load_python()
    ctx = load_js()
    import json

    failures = 0
    for spec, n in CASES:
        try:
            expect = {"l": py._parse_frames(spec, n)}
        except ValueError as e:
            expect = {"e": str(e)}
        got = json.loads(ctx.eval(f"_parse({json.dumps(spec)}, {n})"))

        ok = ("e" in expect) == ("e" in got) and (
            "e" in expect or expect["l"] == got["l"])
        if not ok:
            failures += 1
            print(f"MISMATCH  spec={spec!r} n={n}\n   python={expect}\n   js    ={got}")
        else:
            shown = "raises" if "e" in expect else expect["l"]
            print(f"ok  {spec!r:16} n={n:<4} -> {shown}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} parser cases agree")

    for tok, n, want in ANCHOR_CASES:
        got = json.loads(ctx.eval(f"_anchor({json.dumps(tok)}, {n})"))
        if got != want:
            failures += 1
            print(f"MISMATCH  tokenStart({tok!r}, {n}) = {got}, expected {want}")
        else:
            print(f"ok  tokenStart({tok!r:10}, {n}) -> {got}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
