#!/usr/bin/env python
"""
🦇 Lines' JavaScript against its Python (quickjs, CPU, no weights).

1. `visibleWidgets()` (web/bat_lines.js) shows each filter's own settings and
   nothing else, and every Python widget is reachable from some filter.
2. The controlnet_aux migrations (web/bat-migrations.js) build arrays that line
   up with the node's widgets — same length, same order, same defaults as the
   Python — and every value they produce is one the node accepts.
3. End to end: an old controlnet_aux node's widgets_values, migrated, make
   🦇 Lines produce byte-identical lines (the no-model filters, CPU).

    pip install quickjs
    python tests/verify_lines_web.py
"""

import json
import os
import sys

import numpy as np

try:
    import quickjs
except ImportError:
    print("SKIP  quickjs not installed (pip install quickjs)")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import verify_lines as vl  # noqa: E402  (loads the pack modules + the CA reference)

lines = vl.lines
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def grab(src, name):
    """One top-level `function name(...) {...}` out of an ES module."""
    start = src.index(f"function {name}(")
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


lines_js = open(os.path.join(PACK, "web", "bat_lines.js"), encoding="utf-8").read()
mig_js = open(os.path.join(PACK, "web", "bat-migrations.js"), encoding="utf-8").read()

ctx = quickjs.Context()
ctx.eval("var WARNINGS = []; var console = {warn: function(m){WARNINGS.push(String(m));}, log: function(){}};")
ctx.eval(grab(lines_js, "visibleWidgets"))
for fn in ("linesValues", "caResolution", "caNumber", "linesLookChanged", "mapToLines", "mapFromAIO"):
    ctx.eval(grab(mig_js, fn))


def js(expr):
    return json.loads(ctx.eval(f"JSON.stringify({expr})"))


SPEC = lines.BatLines.INPUT_TYPES()["required"]
WIDGETS = [n for n in SPEC if n != "image"]
FILTERS = SPEC["filter"][0]


# ---------------------------------------------------------------------------
# 1. Visibility
# ---------------------------------------------------------------------------

def test_visibility():
    expect = {
        "canny": {"canny_low", "canny_high"},
        "lineart standard": {"standard_sigma", "standard_threshold"},
        "teed": {"teed_safe_steps", "auto_download"},
        "hed": {"hed_safe", "auto_download"},
        "lineart realistic": {"lineart_coarse", "auto_download"},
        "lineart anime": {"auto_download"},
        "lineart manga": {"auto_download"},
        "mlsd": {"mlsd_score", "mlsd_distance", "mlsd_line_width", "auto_download"},
    }
    always = {"filter", "resolution", "output_size", "invert", "preview_frame"}
    reachable = set()
    for f in FILTERS:
        styles = ["soft", "scribble"] if f == "xdog" else ["soft"]
        for st in styles:
            shown = set(js(f"visibleWidgets({json.dumps({'filter': f, 'xdog_style': st})})"))
            reachable |= shown
            check(f"{f}{' / ' + st if f == 'xdog' else ''}: only real widgets", shown <= set(WIDGETS),
                  str(shown - set(WIDGETS)))
            if f == "xdog":
                want = always | {"xdog_style"} | (
                    {"xdog_scribble_threshold"} if st == "scribble"
                    else {"xdog_sigma", "xdog_k", "xdog_sharpen", "xdog_threshold", "xdog_softness"})
            else:
                want = always | expect[f]
            check(f"{f}{' / ' + st if f == 'xdog' else ''}: shows its own settings only", shown == want,
                  f"extra {sorted(shown - want)} missing {sorted(want - shown)}")
    check("every widget is visible for some filter", reachable == set(WIDGETS), str(set(WIDGETS) - reachable))


# ---------------------------------------------------------------------------
# 2. Migration arrays line up with the node
# ---------------------------------------------------------------------------

def as_params(arr):
    return dict(zip(WIDGETS, arr))


def test_migration_arrays():
    d = js("linesValues({})")
    check("linesValues has one value per widget", len(d) == len(WIDGETS), f"{len(d)} vs {len(WIDGETS)}")
    py = lines._defaults()
    got = as_params(d)
    diffs = {k: (got[k], py[k]) for k in WIDGETS if got[k] != py[k] and k != "output_size"}
    check("migration defaults == the node's defaults (bar output_size)", not diffs, str(diffs))
    check("migrated nodes keep processing size", got["output_size"] == "processing")

    samples = {
        "CannyEdgePreprocessor": [30, 90, 768],
        "Scribble_XDoG_Preprocessor": [5, 1024],
        "LineartStandardPreprocessor": [2.0, 3, 640],
        "LineArtPreprocessor": ["enable", 1024],
        "AnimeLineArtPreprocessor": [512],
        "Manga2Anime_LineArt_Preprocessor": [768],
        "M-LSDPreprocessor": [0.2, 5.0, 512],
        "HEDPreprocessor": ["disable", 512],
        "TEEDPreprocessor": [3, 640],
        "PiDiNetPreprocessor": ["enable", 512],
        "AnyLineArtPreprocessor_aux": ["lineart_standard", 1280, 36, 255, 36, 1],
    }
    want = {
        "CannyEdgePreprocessor": dict(filter="canny", canny_low=30, canny_high=90, resolution=768),
        "Scribble_XDoG_Preprocessor": dict(filter="xdog", xdog_style="scribble", xdog_scribble_threshold=5, resolution=1024),
        "LineartStandardPreprocessor": dict(filter="lineart standard", standard_sigma=2.0, standard_threshold=3, resolution=640),
        "LineArtPreprocessor": dict(filter="lineart realistic", lineart_coarse=True, resolution=1024),
        "AnimeLineArtPreprocessor": dict(filter="lineart anime", resolution=512),
        "Manga2Anime_LineArt_Preprocessor": dict(filter="lineart manga", resolution=768),
        "M-LSDPreprocessor": dict(filter="mlsd", mlsd_score=0.2, mlsd_distance=5.0, resolution=512),
        "HEDPreprocessor": dict(filter="hed", hed_safe=False, resolution=512),
        "TEEDPreprocessor": dict(filter="teed", teed_safe_steps=3, resolution=640),
        "PiDiNetPreprocessor": dict(filter="teed", teed_safe_steps=2, resolution=512),
        "AnyLineArtPreprocessor_aux": dict(filter="teed", resolution=1280),
    }
    for cls, old in samples.items():
        ctx.eval("WARNINGS = []")
        arr = js(f"mapToLines({json.dumps(cls)}, {json.dumps(old)})")
        p = as_params(arr)
        bad = {k: (p.get(k), v) for k, v in want[cls].items() if p.get(k) != v}
        check(f"{cls} -> {want[cls]['filter']}: values carried over", len(arr) == len(WIDGETS) and not bad, str(bad))
        accepted = lines.params_from_request(p)
        changed = {k: (p[k], accepted[k]) for k in WIDGETS if p[k] != accepted[k]}
        check(f"{cls}: every migrated value is one the node accepts", not changed, str(changed))
        warned = js("WARNINGS")
        look_changes = cls in ("TEEDPreprocessor", "PiDiNetPreprocessor", "AnyLineArtPreprocessor_aux")
        check(f"{cls}: warns {'that the look changes' if look_changes else 'nothing'}",
              bool(warned) == look_changes, str(warned))

    check("resolution under 64 becomes 512, as in controlnet_aux",
          as_params(js('mapToLines("CannyEdgePreprocessor", [100, 200, 32])'))["resolution"] == 512)
    check("missing values fall back to controlnet_aux's defaults",
          as_params(js('mapToLines("CannyEdgePreprocessor", [])'))
          == as_params(js('linesValues({filter: "canny"})')))
    aio = as_params(js('mapFromAIO(["M-LSDPreprocessor", 640])'))
    check("AIO_Preprocessor dispatches on its preprocessor name", aio["filter"] == "mlsd" and aio["resolution"] == 640, str(aio))
    ctx.eval("WARNINGS = []")
    aio = as_params(js('mapFromAIO(["DepthAnythingPreprocessor", 512])'))
    check("AIO with no line equivalent falls back to canny, with a warning",
          aio["filter"] == "canny" and len(js("WARNINGS")) == 1)
    check("an unknown class is not mapped", js('mapToLines("SomethingElse", [1, 2])') is None)

    for name in ("CannyEdgePreprocessor", "TEEDPreprocessor", "AIO_Preprocessor", "M-LSDPreprocessor"):
        check(f"{name} is registered as a migration to Bat_Lines",
              (f'"{name}"' in mig_js) and ('to: "Bat_Lines"' in mig_js))


# ---------------------------------------------------------------------------
# 3. Old node -> migrated node -> same bytes
# ---------------------------------------------------------------------------

def test_end_to_end():
    f = vl.plate(300, 420, seed=5)
    u8 = np.asarray(f.numpy() * 255.0, dtype=np.uint8)
    cases = [
        ("CannyEdgePreprocessor", [30, 90, 768], lambda: vl.ref_canny(u8, 30, 90, 768)),
        ("Scribble_XDoG_Preprocessor", [12, 640], lambda: vl.ref_xdog(u8, 12, 640)),
        ("LineartStandardPreprocessor", [2.0, 3, 512], lambda: vl.ref_lineart_standard(u8, 2.0, 3, 512)),
    ]
    for cls, old, ref in cases:
        p = lines.params_from_request(as_params(js(f"mapToLines({json.dumps(cls)}, {json.dumps(old)})")))
        out = lines.run_filter(f, p, None if p["output_size"] == "processing" else (300, 420))
        check(f"{cls} {old}: migrated node gives the same bytes", np.array_equal(out, ref()))


if __name__ == "__main__":
    test_visibility()
    test_migration_arrays()
    test_end_to_end()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all passed")
