#!/usr/bin/env python
"""
Behavioural checks for the canvas zoom-out unlock (``web/bat_canvas_zoom.js``).

The extension is four lines of real work — it lowers ``canvas.ds.min_scale`` —
but every one of its failure modes is silent. A typo in the setting id, an
``onChange`` that throws on the pre-canvas call it is guaranteed to get, or a
percentage written where a scale was wanted all end the same way: the wheel
still stops at 10% and nothing anywhere says why. So it is driven here through
quickjs against a ``DragAndScale`` stub whose ``changeScale()`` is a transcript
of the frontend's own clamp (comfyui_frontend_package 1.49.6), which is the
thing the whole extension is aimed at.

Five claims:

1. **The extension registers, and its setting is shaped the way the settings
   dialog expects** — a ``slider`` with ``attrs.min``/``max``/``step``. A bad
   type renders as a text box; a missing ``attrs`` renders as a slider with
   PrimeVue's own 0-100 range, which would let someone pick 0.

2. **onChange survives being called before the canvas exists.** ``addSetting``
   invokes it at registration time, which is before ComfyUI builds the canvas.
   A throw there aborts the rest of the extension's registration.

3. **setup() is what actually lands the write,** and the default takes the
   floor to half of core's — the wheel then zooms out to 5% and stops.

4. **Changing the setting applies live,** without a reload.

5. **Raising the floor while parked underneath it pulls the view back up.**
   Otherwise the view sits at an illegal zoom until the next wheel notch, and
   `changeScale`'s early-out on an unchanged value means a nudge in the wrong
   direction does nothing at all.

Also asserted: ``max_scale`` is left alone. Zooming in was never the ask, and
the deep-zoom end is where rendering gets expensive.

    pip install quickjs
    python tests/verify_canvas_zoom.py
"""

import json
import os
import re
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_failures = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        _failures.append(label)
        if detail:
            print("      ", detail)


# ---------------------------------------------------------------------------
# The frontend, as much of it as this file touches.
#
# changeScale() is transcribed from DragAndScale in
# comfyui_frontend_package 1.49.6 (settingStore-*.js) — the clamp, the
# early-out on an unchanged value and the snap-to-1 all matter to the
# assertions below, so none of them are paraphrased away.
# ---------------------------------------------------------------------------

STUBS = r"""
    globalThis.__log = [];
    var console = {
        log:   function () { globalThis.__log.push(["log", arguments[0]]); },
        warn:  function () { globalThis.__log.push(["warn", arguments[0]]); },
        error: function () { globalThis.__log.push(["error", arguments[0]]); },
    };

    function makeDS() {
        return {
            scale: 1,
            offset: [0, 0],
            min_scale: 0.1,          // litegraph defaults
            max_scale: 10,
            element: { width: 1920, height: 1080,
                       getBoundingClientRect: function () {
                           return { x: 0, y: 0, width: 1920, height: 1080 };
                       } },
            changeScale: function (v, center, snap) {
                if (snap === undefined) snap = true;
                if (v < this.min_scale) v = this.min_scale;
                else if (v > this.max_scale) v = this.max_scale;
                if (v === this.scale) return;
                this.scale = v;
                if (snap && Math.abs(this.scale - 1) < 0.01) this.scale = 1;
            },
            changeDeltaScale: function (d, center) {
                this.changeScale(this.scale * d, center);
            },
        };
    }

    globalThis.__settingValues = {};
    globalThis.__dirty = 0;

    var app = {
        canvas: null,                 // as it is when extensions register
        ui: { settings: { getSettingValue: function (id) {
                  return globalThis.__settingValues[id];
              } } },
        registerExtension: function (ext) { globalThis.__ext = ext; },
    };

    // Bring the canvas up the way ComfyUI does: after registration, before
    // setup(). setDirty is on the canvas, not the ds.
    globalThis.__makeCanvas = function () {
        app.canvas = {
            ds: makeDS(),
            setDirty: function () { globalThis.__dirty++; },
        };
        return app.canvas;
    };

    // What addSetting() does at registration time.
    globalThis.__registerSettings = function () {
        (globalThis.__ext.settings || []).forEach(function (s) {
            var v = globalThis.__settingValues[s.id];
            if (v === undefined) v = s.defaultValue;
            globalThis.__settingValues[s.id] = v;
            if (s.onChange) s.onChange(v, undefined);
        });
    };

    globalThis.__setSetting = function (id, v) {
        var s = (globalThis.__ext.settings || []).filter(function (x) {
            return x.id === id;
        })[0];
        var old = globalThis.__settingValues[id];
        globalThis.__settingValues[id] = v;
        if (s && s.onChange) s.onChange(v, old);
    };
"""


def js_context():
    """Evaluate the extension in quickjs with ComfyUI stubbed out.

    Evaluated whole, so a syntax error or stale identifier anywhere in the
    file fails here rather than showing up as a zoom limit that never moved.
    """
    import quickjs

    path = os.path.join(PACK, "web", "bat_canvas_zoom.js")
    src = open(path, encoding="utf-8").read()
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(src)
    ctx.eval("if (!globalThis.__ext) throw new Error('no extension registered');")
    return ctx


def jrun(ctx, body):
    ctx.eval("globalThis.__ret = (function () {" + body + "})();")
    return json.loads(ctx.eval("JSON.stringify(globalThis.__ret === undefined "
                               "? null : globalThis.__ret)"))


# ---------------------------------------------------------------------------
# 1. The setting is shaped the way the settings dialog expects
# ---------------------------------------------------------------------------

def test_setting_shape():
    ctx = js_context()
    got = jrun(ctx, """
        const s = (globalThis.__ext.settings || [])[0] || {};
        return {
            name: globalThis.__ext.name,
            n: (globalThis.__ext.settings || []).length,
            id: s.id, type: s.type,
            cat: s.category, def: s.defaultValue,
            attrs: s.attrs || null,
            hasOnChange: typeof s.onChange === "function",
            hasSetup: typeof globalThis.__ext.setup === "function",
            hasTooltip: typeof s.tooltip === "string" && s.tooltip.length > 0,
        };
    """)

    check("extension has a name and a setup()",
          bool(got["name"]) and got["hasSetup"], got)
    check("exactly one setting, id BAT.Canvas.MinZoom",
          got["n"] == 1 and got["id"] == "BAT.Canvas.MinZoom", got)
    check("type is slider with a bounded range and an onChange",
          got["type"] == "slider" and got["hasOnChange"]
          and isinstance(got["attrs"], dict)
          and got["attrs"].get("min", 0) > 0
          and got["attrs"].get("max") == 10
          and got["attrs"].get("step", 0) > 0, got)
    check("category nests under a top-level panel and a sub-group",
          isinstance(got["cat"], list) and len(got["cat"]) >= 2, got)
    check("default is 5% — half of core's floor",
          got["def"] == 5, got)
    check("setting carries a tooltip", got["hasTooltip"], got)


# ---------------------------------------------------------------------------
# 2. onChange tolerates the pre-canvas call it is guaranteed to get
# ---------------------------------------------------------------------------

def test_pre_canvas_onchange():
    ctx = js_context()
    got = jrun(ctx, """
        try {
            globalThis.__registerSettings();      // app.canvas is still null
            return { threw: false, canvas: app.canvas };
        } catch (e) {
            return { threw: true, msg: String(e) };
        }
    """)
    check("registration-time onChange does not throw with no canvas",
          got["threw"] is False, got)


# ---------------------------------------------------------------------------
# 3. setup() lands the write, and the wheel then reaches 5%
# ---------------------------------------------------------------------------

def test_setup_applies_default():
    ctx = js_context()
    got = jrun(ctx, """
        globalThis.__registerSettings();
        globalThis.__makeCanvas();
        const before = app.canvas.ds.min_scale;
        globalThis.__ext.setup();
        const ds = app.canvas.ds;
        // Wheel the view out until it refuses to go further.
        for (let i = 0; i < 200; i++) ds.changeDeltaScale(1 / 1.1);
        return { before, min: ds.min_scale, max: ds.max_scale,
                 floor: ds.scale };
    """)
    check("the stub starts at litegraph's own 0.1 floor",
          got["before"] == 0.1, got)
    check("setup() lowers min_scale to 0.05 (2x further out)",
          abs(got["min"] - 0.05) < 1e-9, got)
    check("wheel-zoom-out now bottoms out at 5%, not 10%",
          abs(got["floor"] - 0.05) < 1e-9, got)
    check("max_scale is left at litegraph's default",
          got["max"] == 10, got)


# ---------------------------------------------------------------------------
# 4. Changing the setting applies live
# ---------------------------------------------------------------------------

def test_live_change():
    ctx = js_context()
    got = jrun(ctx, """
        globalThis.__registerSettings();
        globalThis.__makeCanvas();
        globalThis.__ext.setup();
        const ds = app.canvas.ds;
        globalThis.__setSetting("BAT.Canvas.MinZoom", 1);
        const min1 = ds.min_scale;
        for (let i = 0; i < 400; i++) ds.changeDeltaScale(1 / 1.1);
        return { min1, floor: ds.scale };
    """)
    check("a new setting value re-floors the live canvas",
          abs(got["min1"] - 0.01) < 1e-9, got)
    check("the wheel follows it down to 1%",
          abs(got["floor"] - 0.01) < 1e-9, got)


# ---------------------------------------------------------------------------
# 5. Raising the floor while parked underneath it recovers the view
# ---------------------------------------------------------------------------

def test_raise_floor_recovers_view():
    ctx = js_context()
    got = jrun(ctx, """
        globalThis.__registerSettings();
        globalThis.__makeCanvas();
        globalThis.__ext.setup();
        const ds = app.canvas.ds;
        globalThis.__setSetting("BAT.Canvas.MinZoom", 1);
        for (let i = 0; i < 400; i++) ds.changeDeltaScale(1 / 1.1);
        const parked = ds.scale;                  // 0.01, below the new floor
        const dirty0 = globalThis.__dirty;
        globalThis.__setSetting("BAT.Canvas.MinZoom", 5);
        return { parked, scale: ds.scale, min: ds.min_scale,
                 redrawn: globalThis.__dirty > dirty0 };
    """)
    check("view was parked below the raised floor",
          abs(got["parked"] - 0.01) < 1e-9, got)
    check("raising the floor pulls the view back to it",
          abs(got["scale"] - 0.05) < 1e-9 and abs(got["min"] - 0.05) < 1e-9, got)
    check("and asks for a redraw", got["redrawn"] is True, got)


# ---------------------------------------------------------------------------

def main():
    try:
        import quickjs  # noqa: F401
    except ImportError:
        print("quickjs not installed — pip install quickjs")
        return 1

    test_setting_shape()
    test_pre_canvas_onchange()
    test_setup_applies_default()
    test_live_change()
    test_raise_floor_recovers_view()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for f in _failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
