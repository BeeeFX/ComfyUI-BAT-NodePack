#!/usr/bin/env python
"""
bat_easing.py and web/bat_easing.js must be the same curves: the editors'
live previews use the JS, the render uses the Python, and any drift shows up
as the preview and the output disagreeing mid-segment.

Also pins the contract that keeps old workflows unchanged: a keyframe with no
`ease` (or an unknown one) is linear.

    python tests/verify_easing.py
"""

import importlib.util
import os
import sys

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


def main():
    spec = importlib.util.spec_from_file_location("bat_easing", os.path.join(PACK, "bat_easing.py"))
    py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py)

    print("\ncontract")
    check("no ease is linear", py.ease_name({}) == "linear")
    check("unknown ease is linear", py.ease_name({"ease": "bounce"}) == "linear")
    check("non-dict key is linear", py.ease_name(None) == "linear")
    for name in py.EASES:
        if name == "hold":
            continue
        check(f"{name}: ends pinned", py.apply_ease(0, name) == 0 and py.apply_ease(1, name) == 1)
    check("hold stays on the earlier key", py.apply_ease(0.99, "hold") == 0)
    check("t is clamped", py.apply_ease(-1, "linear") == 0 and py.apply_ease(2, "linear") == 1)

    try:
        import quickjs
    except ImportError:
        print("  skip  JS parity (pip install quickjs)")
    else:
        from _harness import strip_modules
        ctx = quickjs.Context()
        ctx.eval(strip_modules(open(os.path.join(PACK, "web", "bat_easing.js"), encoding="utf-8").read()))
        print("\nPython vs JS")
        check("same curve names", list(py.EASES) == list(ctx.eval("JSON.stringify(EASES)")[2:-2].split('","')))
        worst = 0.0
        for name in list(py.EASES) + ["nonsense"]:
            for i in range(0, 101):
                t = i / 100
                d = abs(py.apply_ease(t, name) - ctx.eval(f"applyEase({t}, {name!r})"))
                worst = max(worst, d)
        check("every curve agrees at 101 points", worst < 1e-12, worst)
        for key in ("{}", "{ease: 'hold'}", "{ease: 'x'}", "null"):
            js = ctx.eval(f"easeName({key})")
            pyk = {"{}": {}, "{ease: 'hold'}": {"ease": "hold"}, "{ease: 'x'}": {"ease": "x"}, "null": None}[key]
            check(f"easeName({key}) agrees", js == py.ease_name(pyk), (js, py.ease_name(pyk)))

    print()
    if _failures:
        print(f"FAILURES: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
