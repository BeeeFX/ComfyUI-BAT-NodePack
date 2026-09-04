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


def _plain(name):
    src = open(os.path.join(PACK, "web", name), encoding="utf-8").read()
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)
    return re.sub(r"^export ", "", src, flags=re.M)


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
    ctx.eval("var self = {postMessage: function(){}};"
             "var performance = {now: function(){ return 0; }};")
    ctx.eval(_plain("bat_blend_modes.js"))
    ctx.eval(_plain("bat_layered_core.js"))
    ctx.eval(_plain("bat_layered_worker.js"))
    ctx.eval("if (typeof self.onmessage !== 'function') "
             "throw new Error('worker did not install onmessage');")


def extension_parses():
    """Evaluate the whole editor with its imports stubbed, and register it."""
    import quickjs
    src = _plain("bat_layered_images.js").replace("import.meta.url", '"file:///bat/"')
    ctx = quickjs.Context()
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


def main():
    import numpy as np
    import torch

    modes_py = _load("bat_blend_modes")
    _load("bat_hdr_preview")
    _load("bat_hdr_tonal_composite")
    _load("bat_advanced_blend")
    m = _load("bat_layered_images")

    extension_parses()
    print("whole-file JS parse + registration: OK")
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
