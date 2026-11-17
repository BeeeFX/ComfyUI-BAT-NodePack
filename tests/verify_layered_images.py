#!/usr/bin/env python
"""
Prove that Bat_LayeredImages' live preview computes the same thing the render
does.

Same contract as the other two verify scripts: the on-node canvas is a second
implementation of the render, in another language, and two hand-written copies of
one algorithm drift the first time someone edits either. Here there are
*nineteen* blend modes duplicated that way, four of them non-separable colour
operations with clipping behaviour that is easy to get subtly wrong — which is
the only reason such a duplication is defensible at all.

What is covered
---------------
* Every blend mode, separable and not, over a grid that deliberately includes
  the values the formulas branch on (0, 0.25, 0.5, 1) and values outside [0,1],
  since nothing in the pack clamps.
* The full composite: 1..5 layers, masked and unmasked, varying opacity, and
  disabled layers — against the real `composite()`.
* The accumulated alpha, which is a node output and not just an internal.
* `render_region()` against the same composite at full frame, to prove a region
  is independent of its surroundings (it should be exactly, since there is no
  blur here — unlike Advanced Blend, which needs a context margin).
* `parse_layers()` against malformed state, because that string is stored in a
  workflow and reaches the network.
* `bat_scrub.js`'s value logic — snapping without float dust, clamping, garbage
  input, and independence from the graph zoom.

    $ ../../../env/bin/python tests/verify_layered_images.py

Tolerance
---------
Relative, as in verify_exposure_bracket.py: the arithmetic modes carry values
well outside [0,1] and an absolute bound is meaningless there.

For the composite it is also DEPTH-SCALED, and that is a measurement rather than
a convenience. Python accumulates in float32 and JS in float64, so error compounds
per layer — measured 0, 3.7e-6, 1.6e-5, 1.1e-4, 1.75e-4 relative at depths
1/2/3/5/8, affecting one pixel in 391. `saturation` is the sensitive one because
it divides by the *backdrop's* colour span, which is the value doing the
accumulating.

A single layer is required to be near-EXACT (see DEPTH1_TOL), and that is the
assertion carrying the weight: it proves the nineteen formulas are identical,
because a real difference in any of them shows up at depth 1 as a residual
orders of magnitude larger. The depth-scaled bound then only has to accommodate
arithmetic.
"""

import importlib.util
import json
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import auto_stub_js, strip_modules

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

ATOL = 1e-5
RTOL = 1e-4

# A single layer involves no accumulation, so the two implementations should
# agree to float32 representation and nothing worse. Measured: exactly 0.0 for
# every mode. Held tight on purpose — this is the check that would catch a
# formula divergence.
DEPTH1_TOL = 1e-6


def _load(name):
    pkg = sys.modules.get("batpack")
    if pkg is None:
        pkg = types.ModuleType("batpack")
        pkg.__path__ = [PACK]
        sys.modules["batpack"] = pkg
    spec = importlib.util.spec_from_file_location(
        "batpack." + name, os.path.join(PACK, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["batpack." + name] = mod
    spec.loader.exec_module(mod)
    return mod


def _raw(name):
    return open(os.path.join(PACK, "web", name), encoding="utf-8").read()


def _plain(name):
    return strip_modules(_raw(name))


DRIVER = """
function _modes(payload) {
    const j = JSON.parse(payload);
    const out = [];
    if (isSeparable(j.mode)) {
        const f = SEPARABLE[j.mode];
        for (let i = 0; i < j.cb.length; i++) out.push(f(j.cb[i], j.cs[i]));
    } else {
        const f = NON_SEPARABLE[j.mode];
        const t = [0, 0, 0];
        for (let i = 0; i < j.cb.length; i += 3) {
            f(t, j.cb[i], j.cb[i+1], j.cb[i+2], j.cs[i], j.cs[i+1], j.cs[i+2]);
            out.push(t[0], t[1], t[2]);
        }
    }
    return JSON.stringify(out);
}

function _composite(payload) {
    const j = JSON.parse(payload);
    const layers = j.layers.map(l => ({
        data: Float32Array.from(l.data),
        mask: l.mask ? Float32Array.from(l.mask) : null,
    }));
    const r = compositeStack(layers, j.settings, j.w, j.h, j.clamp);
    return JSON.stringify({out: Array.from(r.out), alpha: Array.from(r.alpha)});
}

function _paint(payload) {
    const j = JSON.parse(payload);
    const layers = j.layers.map(l => ({
        data: Float32Array.from(l.data),
        mask: l.mask ? Float32Array.from(l.mask) : null,
    }));
    const r = compositeStack(layers, j.settings, j.w, j.h, j.clamp);
    const px = paintLayered({view: j.view, layers, out: r.out, alpha: r.alpha,
                             w: j.w, h: j.h});
    const rgb = [];
    for (let i = 0; i < px.length; i += 4) rgb.push(px[i], px[i + 1], px[i + 2]);
    return JSON.stringify(rgb);
}
"""


SCRUB_SMOKE = r"""
function _scrubSmoke() {
    // ── snapToStep ───────────────────────────────────────────────────────
    // No float dust. Math.round(0.3/0.01)*0.01 is 0.30000000000000004, and that
    // value would be written into the workflow JSON and shown as "30%" while
    // comparing unequal to 0.3 for ever.
    if (snapToStep(0.3, 0, 1, 0.01) !== 0.3)
        throw new Error("float dust: " + snapToStep(0.3, 0, 1, 0.01));
    if (snapToStep(0.07, 0, 1, 0.01) !== 0.07) throw new Error("0.07 not exact");
    if (snapToStep(0.123, 0, 1, 0.01) !== 0.12) throw new Error("did not snap");
    if (snapToStep(0.129, 0, 1, 0.01) !== 0.13) throw new Error("did not round up");
    // Clamping, both ends.
    if (snapToStep(-5, 0, 1, 0.01) !== 0) throw new Error("no low clamp");
    if (snapToStep(99, 0, 1, 0.01) !== 1) throw new Error("no high clamp");
    // Garbage must land on `min`, not NaN — this is reachable from a saved
    // workflow's layer JSON.
    for (const bad of [NaN, Infinity, -Infinity, undefined, null, "abc", {}]) {
        const v = snapToStep(bad, 0, 1, 0.01);
        if (!Number.isFinite(v)) throw new Error("non-finite out for " + String(bad));
    }
    // A non-unit range and a coarse step.
    if (snapToStep(7.3, 0, 10, 0.5) !== 7.5) throw new Error("coarse step wrong");
    if (snapToStep(-3.2, -8, 8, 0.25) !== -3.25) throw new Error("negative min wrong");

    // ── valueFromPosition ────────────────────────────────────────────────
    const rect = {left: 100, width: 200};
    if (valueFromPosition(100, rect, 0, 1, 0.01) !== 0) throw new Error("left edge");
    if (valueFromPosition(300, rect, 0, 1, 0.01) !== 1) throw new Error("right edge");
    if (valueFromPosition(200, rect, 0, 1, 0.01) !== 0.5) throw new Error("midpoint");
    // Outside the bar clamps rather than running away — this is what makes a
    // captured drag that leaves the row behave sanely.
    if (valueFromPosition(-999, rect, 0, 1, 0.01) !== 0) throw new Error("no clamp left");
    if (valueFromPosition(9999, rect, 0, 1, 0.01) !== 1) throw new Error("no clamp right");

    // ZOOM INDEPENDENCE, which is the property that matters most: Nodes 1.0
    // CSS-transforms DOM widgets by the canvas scale, so the same normalised
    // position under a different measured width must give the same value. A
    // control doing its arithmetic in raw pixels would drift with the zoom.
    const zoomed = {left: 40, width: 80};      // same bar at 0.4x
    for (const t of [0, 0.13, 0.5, 0.77, 1]) {
        const a = valueFromPosition(rect.left + t * rect.width, rect, 0, 1, 0.01);
        const b = valueFromPosition(zoomed.left + t * zoomed.width, zoomed, 0, 1, 0.01);
        if (a !== b) throw new Error(`zoom dependent at t=${t}: ${a} vs ${b}`);
    }
    // A degenerate rect (a collapsed or not-yet-laid-out node) must not divide
    // by zero.
    const v0 = valueFromPosition(50, {left: 50, width: 0}, 0, 1, 0.01);
    if (!Number.isFinite(v0)) throw new Error("zero-width rect gave " + v0);
    return "ok";
}
"""


def scrub_smoke():
    """Exercise bat_scrub.js's pure value logic.

    New shared module, and nine other files in the pack still use a native
    `<input type="range">` that it is meant to replace — so it is worth pinning
    before they adopt it.
    """
    import quickjs
    ctx = quickjs.Context()
    ctx.eval(_plain("bat_scrub.js"))
    ctx.eval(SCRUB_SMOKE)
    ctx.eval("_scrubSmoke()")


def make_ctx():
    import quickjs
    ctx = quickjs.Context()
    ctx.eval(_plain("bat_blend_modes.js"))
    ctx.eval(_plain("bat_layered_core.js"))
    ctx.eval(DRIVER)
    return ctx


def worker_parses():
    """Parse the layered worker with its imports stubbed."""
    import quickjs
    ctx = quickjs.Context()
    ctx.eval("function WorkerGlobalScope() {}"
             "var self = new WorkerGlobalScope(); self.postMessage = function(){};"
             "var performance = {now: function(){ return 0; }};")
    ctx.eval(_plain("bat_blend_modes.js"))
    ctx.eval(_plain("bat_layered_core.js"))
    ctx.eval(_plain("bat_layered_worker.js"))
    ctx.eval("if (typeof self.onmessage !== 'function') "
             "throw new Error('worker did not install onmessage');")

    # The main page imports this file too (ComfyUI's /extensions route globs
    # every .js), and there `self` is `window`: it must leave onmessage alone.
    page = quickjs.Context()
    page.eval("var self = {postMessage: function(){}, onmessage: 'page handler'};"
              "var performance = {now: function(){ return 0; }};")
    page.eval(_plain("bat_blend_modes.js"))
    page.eval(_plain("bat_layered_core.js"))
    page.eval(_plain("bat_layered_worker.js"))
    page.eval("if (self.onmessage !== 'page handler') "
              "throw new Error('worker clobbered the page onmessage');")


def extension_parses():
    """Evaluate the whole editor with its imports stubbed, and register it."""
    import quickjs
    src = _plain("bat_layered_images.js").replace("import.meta.url", '"file:///bat/"')
    ctx = quickjs.Context()
    # First, so the curated stubs below win — see tests/_harness.py.
    ctx.eval(auto_stub_js(_raw("bat_layered_images.js")))
    ctx.eval("""
    var __registered = null;
    var app = {
        registerExtension: function (e) { __registered = e; },
        graph: null,
        extensionManager: { setting: { get: function () { return false; } } },
    };
    function addBatDOMWidget() { return {}; }
    function clampNodeSize() {}
    function hdrSupported() { return false; }
    function decodeHdrTile() {}
    function imageDataToSource() {}
    function batTrack() { return { listener: function(){}, dispose: function(){},
        timeout: function(i){ return i; }, observer: function(){} }; }
    function registerCleanup() {}
    function batNodeCacheKey() { return "k"; }
    function isNodeAlive() { return true; }
    function attachZoomControl() { return {setZoom:function(){},getZoom:function(){return 1;},
        resetView:function(){},refresh:function(){},destroy:function(){}}; }
    function compositeStack() { return {out: new Float32Array(3), alpha: new Float32Array(1)}; }
    function paintLayered() { return new Uint8ClampedArray(4); }
    function makeScrubber() { return {el: {}, get: function(){ return 1; }, set: function(){}}; }
    var SEPARABLE = {normal: function(a,b){ return b; }};
    var NON_SEPARABLE = {};
    var MODES = ["normal"]; var MODE_LABELS = {normal: "Normal"};
    function isSeparable() { return true; }
    function fetch() { return Promise.reject(new Error("no network")); }
    function AbortController() { this.signal = {aborted:false}; this.abort = function(){}; }
    function createImageBitmap() { return Promise.reject(new Error("no bitmaps")); }
    function Worker() { throw new Error("no workers"); }
    var URL = { createObjectURL: function(){ return ""; } };
    var window = { devicePixelRatio: 1, addEventListener: function(){} };
    var document = { createElement: function(){ throw new Error("no DOM"); } };
    var localStorage = { getItem: function(){ return null; }, setItem: function(){} };
    """)
    ctx.eval(src)
    ctx.eval("if (!__registered) throw new Error('extension did not register');")
    ctx.eval("""
    (function () {
        const nt = function () {}; nt.prototype = {};
        __registered.beforeRegisterNodeDef(nt, {name: 'Bat_LayeredImages'}, app);
        for (const h of ['onNodeCreated', 'onExecuted']) {
            if (typeof nt.prototype[h] !== 'function') throw new Error('missing ' + h);
        }
    })();
    """)
    return ctx


LAYER_STATE_DRIVER = r"""
function _readState(text, count, slotsJson) {
    const node = {widgets: [{name: "layers", value: text}]};
    return JSON.stringify(readLayerState(node, count, JSON.parse(slotsJson)));
}
function _writeState(text, settingsJson, slotsJson) {
    const node = {widgets: [{name: "layers", value: text}]};
    writeLayerState(node, JSON.parse(settingsJson), JSON.parse(slotsJson));
    return node.widgets[0].value;
}
"""


def layer_state_checks(ctx, m):
    """Slot-keyed layer settings, and the older positional format.

    Settings used to be matched to layers by position among the CONNECTED ones,
    so unwiring image_2 of three handed layer 2's mode to image_3. Entries now
    carry their slot; one without is the old format and must load exactly as it
    always did, which with no gap is also what the slot lookup gives.
    """
    ctx.eval(LAYER_STATE_DRIVER)
    # The editor harness stubs MODES down to ["normal"]; widen it in place to
    # the real list, or every mode below would read back as the default.
    ctx.eval(f"MODES.splice(0, MODES.length, ...{json.dumps(list(m.MODES))});")
    _r, _w = ctx.get("_readState"), ctx.get("_writeState")
    # quickjs takes scalars only, so the lists cross as JSON.
    def js_read(text, count, slots):
        return _r(text, count, json.dumps(slots))

    def js_write(text, settings, slots):
        return _w(text, json.dumps(settings), json.dumps(slots))

    def both(text, count, slots):
        py = m.parse_layers(text, count, slots)
        js = json.loads(js_read(text, count, slots))
        assert py == js, (text, slots, py, js)
        return py

    legacy = json.dumps({"layers": [{"mode": "normal"}, {"mode": "multiply"},
                                    {"mode": "screen", "opacity": 0.5}]})
    # Old format, no gap: unchanged, whatever the slots.
    got = both(legacy, 3, [1, 2, 3])
    assert [g["mode"] for g in got] == ["normal", "multiply", "screen"], got
    # Old format with a gap still reads by position — that is what it meant.
    got = both(legacy, 2, [1, 3])
    assert [g["mode"] for g in got] == ["normal", "multiply"], got

    keyed = json.dumps({"layers": [{"slot": 1, "mode": "normal"},
                                   {"slot": 2, "mode": "multiply"},
                                   {"slot": 3, "mode": "screen", "opacity": 0.5}]})
    # The bug: image_2 unwired. image_3 must keep screen, not inherit multiply.
    got = both(keyed, 2, [1, 3])
    assert [g["mode"] for g in got] == ["normal", "screen"], got
    assert got[1]["opacity"] == 0.5, got
    # A slot with no entry gets the default; slots omitted -> 1..n.
    got = both(keyed, 2, [1, 5])
    assert got[1] == dict(m.DEFAULT_LAYER), got
    both(keyed, 3, None)

    # Writing: slots recorded, entries for unwired slots kept, legacy dropped.
    text = js_write(legacy, [{"mode": "normal", "opacity": 1, "enabled": True},
                             {"mode": "screen", "opacity": 0.5, "enabled": True}],
                    [1, 3])
    doc = json.loads(text)["layers"]
    assert [e["slot"] for e in doc] == [1, 3], doc
    text = js_write(text, [{"mode": "darken", "opacity": 1, "enabled": False}], [1])
    doc = json.loads(text)["layers"]
    assert [(e["slot"], e["mode"]) for e in doc] == [(1, "darken"), (3, "screen")], doc
    got = both(text, 2, [1, 3])
    assert [g["mode"] for g in got] == ["darken", "screen"], got
    assert got[0]["enabled"] is False, got


def main():
    import numpy as np
    import torch

    modes_py = _load("bat_blend_modes")
    _load("bat_hdr_preview")
    _load("bat_hdr_tonal_composite")
    _load("bat_advanced_blend")
    m = _load("bat_layered_images")

    ext_ctx = extension_parses()
    print("whole-file JS parse + registration: OK")
    layer_state_checks(ext_ctx, m)
    print("layer settings: slot-keyed, old positional format loads unchanged: OK")
    worker_parses()
    print("bat_layered_worker.js parses and installs its handler: OK")
    scrub_smoke()
    print("bat_scrub.js: snapping, clamping, garbage input, zoom independence: OK")

    ctx = make_ctx()
    js_modes = ctx.get("_modes")
    js_comp = ctx.get("_composite")

    failures = []
    worst = {}

    def check(label, py, js, rtol=RTOL):
        py = np.asarray(py, dtype=np.float64).reshape(-1)
        js = np.asarray(js, dtype=np.float64).reshape(-1)
        if py.shape != js.shape:
            failures.append((label, f"shape {py.shape} vs {js.shape}"))
            return
        if not py.size:
            return
        allow = ATOL + rtol * np.abs(py)
        resid = np.abs(py - js) / allow
        i = int(np.argmax(resid))
        worst[label] = max(worst.get(label, 0.0), float(resid[i]))
        if resid[i] > 1.0:
            failures.append((label, f"resid {resid[i]:.2f}x tolerance "
                                    f"(|d| {abs(py[i]-js[i]):.3e} at {py[i]:.6g})"))

    # ── 1. every blend mode ──────────────────────────────────────────────
    # The grid deliberately includes the branch points (0, 0.25, 0.5, 1) and
    # values outside [0,1], because nothing clamps and the modes still have to
    # agree there.
    vals = [-0.4, 0.0, 1e-7, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.999, 1.0, 1.6, 4.0]
    cb, cs = [], []
    for a in vals:
        for b in vals:
            cb.append(a)
            cs.append(b)
    # Pad to a multiple of 3 so the non-separable pass can read RGB triples.
    while len(cb) % 3:
        cb.append(0.5)
        cs.append(0.5)

    t_cb = torch.tensor(cb, dtype=torch.float32).reshape(1, 1, -1, 1).repeat(1, 1, 1, 3)
    t_cs = torch.tensor(cs, dtype=torch.float32).reshape(1, 1, -1, 1).repeat(1, 1, 1, 3)
    # For the non-separable modes the three channels must differ, or hue and
    # saturation are trivially neutral and the test proves nothing.
    t_cb_rgb = torch.tensor(cb, dtype=torch.float32).reshape(1, 1, -1, 3)
    t_cs_rgb = torch.tensor(cs, dtype=torch.float32).reshape(1, 1, -1, 3)

    for mode in modes_py.MODES:
        if modes_py.is_separable(mode):
            py = modes_py.blend(t_cb, t_cs, mode)[0, 0, :, 0].numpy()
            js = json.loads(js_modes(json.dumps({"mode": mode, "cb": cb, "cs": cs})))
        else:
            py = modes_py.blend(t_cb_rgb, t_cs_rgb, mode).reshape(-1).numpy()
            js = json.loads(js_modes(json.dumps({"mode": mode, "cb": cb, "cs": cs})))
        check(f"mode[{mode}]", py, js)

    # Divide by a black source: Photoshop's answer (white where there is a
    # backdrop, black where there is none), not cb / EPS — a 5e5 firefly.
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    got = modes_py.blend(t([0.5, 0.0, -0.2, 3.0, 0.5]), t([0.0, 0.0, 0.0, 1e-7, 0.25]),
                         "divide").tolist()
    assert got == [1.0, 0.0, 0.0, 1.0, 2.0], got
    js = json.loads(js_modes(json.dumps({"mode": "divide", "cb": [0.5, 0.0, -0.2, 3.0, 0.5],
                                         "cs": [0.0, 0.0, 0.0, 1e-7, 0.25]})))
    assert js == got, js

    # ── 2. the composite ─────────────────────────────────────────────────
    torch.manual_seed(19)
    H, W = 17, 23
    N = H * W

    def layer(seed, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        return (torch.rand(1, H, W, 3, generator=g) * scale)

    def mask(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.rand(1, H, W, 1, generator=g)

    all_layers = [(layer(100 + i, 1.0 if i % 3 else 2.5), mask(200 + i))
                  for i in range(5)]

    n_cases = 0
    for depth in (1, 2, 3, 5):
        for mode_set in ([m2] * depth for m2 in modes_py.MODES):
            settings = [{"mode": md, "opacity": 1.0, "enabled": True}
                        for md in mode_set]
            for variant in ("plain", "masked", "opacity", "disabled"):
                use = []
                cfg = [dict(s) for s in settings]
                for i in range(depth):
                    img, mk = all_layers[i]
                    use.append((img, mk if variant == "masked" else None))
                    if variant == "opacity":
                        cfg[i]["opacity"] = [1.0, 0.35, 0.8, 0.0, 0.6][i % 5]
                    if variant == "disabled" and i == depth // 2:
                        cfg[i]["enabled"] = False

                # Depth-scaled: float32 accumulation compounds per layer, and
                # a single layer must still be near-exact. See the docstring.
                tol = DEPTH1_TOL if depth == 1 else RTOL * depth

                py_out, py_alpha = m.composite(use, cfg, False)
                payload = json.dumps({
                    "layers": [{"data": im[0].numpy().reshape(-1).tolist(),
                                "mask": (None if mk is None
                                         else mk[0, ..., 0].numpy().reshape(-1).tolist())}
                               for (im, mk) in use],
                    "settings": cfg, "w": W, "h": H, "clamp": False,
                })
                got = json.loads(js_comp(payload))
                check(f"composite[d{depth}/{variant}]", py_out[0].numpy(),
                      got["out"], tol)
                check(f"alpha[d{depth}/{variant}]", py_alpha[0, ..., 0].numpy(),
                      got["alpha"], tol)
                n_cases += 1

    print(f"{n_cases} composite configurations "
          f"({len(modes_py.MODES)} modes x depths 1/2/3/5 x 4 variants)")

    # ── 3. regions must be independent (no blur, so exactly) ─────────────
    entry = {"layers": [(im, mk) for (im, mk) in all_layers[:3]],
             "settings": None, "meta": {}}
    cfg = [{"mode": md, "opacity": 0.8, "enabled": True}
           for md in ("normal", "screen", "color")]
    full = m.render_region(entry, cfg, (0, 0, W, H), W, H, "result").astype(np.int32)
    roi_worst = 0
    for (x, y, w, h) in [(0, 0, W, H), (5, 3, 9, 7), (0, 0, 4, 3),
                         (W - 4, H - 3, 4, 3), (11, 8, 6, 5)]:
        got = m.render_region(entry, cfg, (x, y, w, h), w, h, "result").astype(np.int32)
        roi_worst = max(roi_worst, int(np.abs(got - full[y:y + h, x:x + w]).max()))
    print(f"region renders vs full frame: worst delta = {roi_worst} code value(s)")
    if roi_worst:
        failures.append(("render_region", f"delta {roi_worst}"))

    # ── 3b. the full layer must show the view the draft shows ────────────
    # It is drawn OVER the draft once it lands. Alpha had no server branch and
    # came back as the RGB result; Solo came back un-premultiplied, so a soft
    # matte showed full colour wherever it was above zero.
    js_paint = ctx.get("_paint")
    use = [all_layers[0], (all_layers[1][0], None), all_layers[2]]
    payload = {
        "layers": [{"data": im[0].numpy().reshape(-1).tolist(),
                    "mask": (None if mk is None
                             else mk[0, ..., 0].numpy().reshape(-1).tolist())}
                   for (im, mk) in use],
        "settings": cfg, "w": W, "h": H, "clamp": False,
    }
    view_entry = {"layers": use, "settings": None, "meta": {}}
    view_worst = 0
    for view in ("result", "alpha", "layer:0", "layer:1", "layer:2"):
        py = m.render_region(view_entry, cfg, (0, 0, W, H), W, H, view).astype(np.int32)
        js = np.asarray(json.loads(js_paint(json.dumps(dict(payload, view=view)))),
                        dtype=np.int32).reshape(H, W, 3)
        d = int(np.abs(py - js).max())
        view_worst = max(view_worst, d)
        # One code value: float32 against float64 at a rounding boundary.
        if d > 1:
            failures.append((f"view[{view}]", f"server vs draft delta {d}"))
    print(f"server views vs draft (result/alpha/solo): worst delta = {view_worst} "
          "code value(s)")

    # ── 4. layer-state parsing must never throw ──────────────────────────
    for bad in ["", "{", "null", "[]", '{"layers":"nope"}',
                '{"layers":[{"mode":"nonsense","opacity":"x"}]}',
                '{"layers":[{"opacity":1e999},{"opacity":-5}]}',
                '{"layers":[1,2,3]}']:
        got = m.parse_layers(bad, 3)
        assert len(got) == 3, (bad, got)
        for g in got:
            assert g["mode"] in modes_py.MODES, (bad, g)
            assert 0.0 <= g["opacity"] <= 1.0 and np.isfinite(g["opacity"]), (bad, g)
            assert isinstance(g["enabled"], bool), (bad, g)
    # A good one must survive intact.
    ok = m.parse_layers('{"layers":[{"mode":"screen","opacity":0.25,"enabled":false}]}', 1)
    assert ok[0] == {"mode": "screen", "opacity": 0.25, "enabled": False}, ok
    print("layer-state parsing survives malformed input and preserves good input: OK")

    # ── 5. the node end to end: a gap, a short mask, several chunks ───────
    # image_2 unwired, so slot 3 must get ITS settings (screen), not slot 2's.
    # Six frames crosses a CHUNK_FRAMES boundary, and a two-frame mask has to
    # hold its last frame for frames 2..5 — the per-chunk mask path must give
    # exactly what the old whole-batch alignment did.
    n_fr = m.CHUNK_FRAMES + 2
    base = torch.rand(n_fr, H, W, 3, generator=torch.Generator().manual_seed(5))
    top = torch.rand(1, H, W, 3, generator=torch.Generator().manual_seed(6))
    mk = torch.rand(2, H, W, generator=torch.Generator().manual_seed(7))
    state = json.dumps({"layers": [{"slot": 1, "mode": "normal"},
                                   {"slot": 2, "mode": "multiply"},
                                   {"slot": 3, "mode": "screen", "opacity": 0.7}]})
    res = m.BatLayeredImages().run(layers=state, image_1=base, image_3=top,
                                   mask_3=mk, resize_filter="area",
                                   unique_id="12:5")
    cfg = [{"mode": "normal", "opacity": 1.0, "enabled": True},
           {"mode": "screen", "opacity": 0.7, "enabled": True}]
    for f in range(n_fr):
        a = mk[min(f, 1)][None, ..., None]
        want_rgb, want_a = m.composite([(base[f:f + 1], None), (top, a)], cfg)
        assert torch.allclose(res["result"][0][f:f + 1], want_rgb, atol=1e-6), f
        assert torch.allclose(res["result"][1][f:f + 1], want_a[..., 0], atol=1e-6), f
    # The payload now lives in a sidecar (bat_ui_ref.py): the history keeps only
    # the token, and the JS resolves it back to exactly this dict.
    assert set(res["ui"]) == {"bat_ui"}, sorted(res["ui"])
    ui = sys.modules["batpack.bat_ui_ref"].load_ui(res["ui"])
    assert ui["settings"][0][1]["mode"] == "screen", ui["settings"]
    assert ui["slots"] == [[1, 3]] and len(ui["tiles"][0]) == 2, sorted(ui)
    # The subgraph execution id rides along for the full-resolution request,
    # and the thumbnail a reopened workflow shows is the COMPOSITE.
    assert ui["node_id"] == ["12:5"], ui.get("node_id")
    # The full layer renders only for the run whose tiles the client holds; the
    # id alone is shared by same-numbered nodes in other open workflows, and a
    # restored thumbnail has no token at all.
    entry = m._cache_get("12:5")
    assert m._run_matches(entry, {"node_id": "12:5", "run": ui["run"][0]})
    for stale in ({"run": "0" * 32}, {}, {"run": ""}):
        assert not m._run_matches(entry, dict(stale, node_id="12:5")), stale
    import base64, io
    from PIL import Image
    thumb = np.asarray(Image.open(io.BytesIO(base64.b64decode(ui["jpeg_result"][0]))),
                       dtype=np.float32) / 255.0
    want = res["result"][0][0].clamp(0, 1).numpy()
    # The plates are noise, which JPEG cannot hold per pixel — so compare the
    # means: the composite's, not the bottom layer's (what it used to ship).
    d_comp = abs(float(thumb.mean()) - float(want.mean()))
    d_base = abs(float(thumb.mean()) - float(base[0].numpy().mean()))
    assert thumb.shape == want.shape and d_comp < 0.02 and d_comp < d_base, (d_comp, d_base)
    print("node run: slot-keyed settings across a gap, short mask held, chunked: OK")

    # Chunks run on ComfyUI's device and fall back to the host for anything it
    # can't do. "meta" stands in for a failing GPU (the resampler reads a value
    # back, which meta cannot); the result must be exactly the CPU one.
    kw = dict(layers=state, image_1=base, image_3=top[:, ::2, ::2], mask_3=mk,
              resize_filter="bilinear")
    ref = m.BatLayeredImages().run(**kw)
    real = m._compute_devices
    m._compute_devices = lambda fb: (torch.device("meta"), torch.device("cpu"))
    try:
        got = m.BatLayeredImages().run(**kw)
    finally:
        m._compute_devices = real
    assert all(torch.equal(g, r) for g, r in zip(got["result"], ref["result"]))
    print("device fallback reproduces the CPU composite: OK")

    for label in sorted(worst):
        if worst[label] > 0.05:
            print(f"  {label:34s} residual {worst[label]:6.3f}x")
    print(f"worst residual across all checks: {max(worst.values()):.3f}x tolerance")

    if failures:
        print(f"\nFAIL — {len(failures)} check(s) over tolerance:")
        for label, why in failures[:12]:
            print(f"  {label}: {why}")
        return 1
    print("OK — the live preview and the render agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
