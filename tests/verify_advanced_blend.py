#!/usr/bin/env python
"""
Prove that the Bat_AdvancedBlend live preview computes the same thing the
render does.

Why this exists
---------------
The node's whole value is that an artist can trust the on-node canvas while
dialling in a detail transfer, and then hit Run and get what they were looking
at. That contract is enforced by two independent implementations of the same
algorithm — ``_core`` in ``bat_advanced_blend.py`` and ``blendTile`` in
``web/bat_advanced_blend.js`` — which is exactly the kind of arrangement that
drifts silently the first time someone edits one side.

So run both and diff them. There is no Node.js on the render machines, but the
``quickjs`` wheel embeds a full ES2020 engine in-process, which is plenty: the
part of the JS being tested is pure numerics with no DOM and no imports.

    $ ../../../env/bin/python tests/verify_advanced_blend.py

Exits non-zero on any mismatch above tolerance.

Tolerance
---------
1e-4 for the JS/Python comparison. The two sides are not bit-identical by construction and shouldn't be:
Python accumulates the Gaussian in float32 tensors, JS accumulates in float64
and stores to Float32Array. That is a ~1e-7 relative difference per tap, and a
65-tap separable blur compounds it a little. Anything at 1e-4 is real drift in
the maths, not arithmetic noise.
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
JS_PATH = os.path.join(PACK, "web", "bat_advanced_blend.js")


# ---------------------------------------------------------------------------
# Load the Python side without importing the pack's __init__ (which pulls in
# ComfyUI's `server` module and a lot else besides).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Load the JS side.
# ---------------------------------------------------------------------------

def js_prelude():
    """The blend itself, as a plain script.

    `web/bat_blend_core.js` is exactly this and nothing else — no DOM, no
    ComfyUI, no editor. It was extracted from the editor file precisely so this
    test could load it whole instead of slicing the top off a 1000-line module
    and hoping the cut stayed in the right place.
    """
    src = open(os.path.join(PACK, "web", "bat_blend_core.js"), encoding="utf-8").read()
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)   # no module loader
    src = re.sub(r"^export ", "", src, flags=re.M)            # plain script scope
    return src


def js_worker_parses():
    """Parse the blend worker, with its one import stubbed.

    It has no DOM to fake and never runs on the main thread, so nothing else
    would catch a syntax error in it — the symptom would be a preview that
    silently falls back to main-thread blending, i.e. exactly the stutter this
    worker exists to remove, with no error anyone would notice.
    """
    import quickjs
    path = os.path.join(PACK, "web", "bat_blend_worker.js")
    src = re.sub(r"^import .*?;\s*$", "", open(path, encoding="utf-8").read(), flags=re.M)
    ctx = quickjs.Context()
    ctx.eval("var self = {postMessage: function(){}};"
             "var performance = {now: function(){ return 0; }};")
    ctx.eval(js_prelude())
    ctx.eval(src)
    ctx.eval("if (typeof self.onmessage !== 'function') "
             "throw new Error('worker did not install onmessage');")


def js_parses_whole_file():
    """Evaluate the ENTIRE extension, not just the numeric prelude.

    The slice above only proves the top of the file parses. A syntax error or a
    stale identifier in the editor half would sail past it and only surface as a
    blank node in the browser, so run the whole thing here with the four imports
    stubbed out. `registerExtension` is captured rather than executed, which is
    enough to catch anything the parser or top-level scope can catch.
    """
    import quickjs
    raw = open(JS_PATH, encoding="utf-8").read()
    src = strip_modules(raw)
    # `import.meta.url` is how the extension locates the worker beside itself,
    # and it is valid in the browser because ComfyUI serves extensions as ES
    # modules. quickjs evaluates this as a plain script, where it is a syntax
    # error, so neutralise it — the Worker constructor is stubbed anyway.
    src = src.replace("import.meta.url", '"file:///bat/"')
    stubs = """
    var __registered = null;
    var app = {
        registerExtension: function (e) { __registered = e; },
        extensionManager: { setting: { get: function () { return false; } } },
    };
    function addBatDOMWidget() { return {}; }
    function clampNodeSize() {}
    function hdrSupported() { return false; }
    function decodeHdrTile() {}
    function imageDataToSource() {}
    function batTrack() { return { listener: function(){}, dispose: function(){} }; }
    function registerCleanup() {}
    function batNodeCacheKey() { return "k"; }
    function isNodeAlive() { return true; }
    function attachZoomControl() {
        return { setZoom: function(){}, getZoom: function(){ return 1; },
                 resetView: function(){}, refresh: function(){} };
    }
    function fetch() { return Promise.reject(new Error("no network in this harness")); }
    function AbortController() { this.signal = {aborted: false}; this.abort = function(){}; }
    function createImageBitmap() { return Promise.reject(new Error("no bitmaps here")); }
    var URL = { createObjectURL: function(){ return ""; } };
    var window = { devicePixelRatio: 1, addEventListener: function(){} };
    function blendTile() { return {out: new Float32Array(3), high: new Float32Array(3)}; }
    function makeBlurCache() { return {get: function(){}, clear: function(){}}; }
    function paintView() { return new Uint8ClampedArray(4); }
    function neutralHigh() { return 0; }
    function Worker() { throw new Error("no workers in this harness"); }
    """
    ctx = quickjs.Context()
    # Auto-stubs FIRST so the curated ones above win — see _harness.py for why
    # a helper the editor imports but this harness has never heard of must
    # still exist (it is what silently broke four suites at once).
    ctx.eval(auto_stub_js(raw) + "\n" + stubs + "\n" + src)
    ctx.eval("if (!__registered || __registered.name !== 'Bat_AdvancedBlend') "
             "throw new Error('extension did not register');")
    # Exercise the registration path the frontend takes, so a typo inside
    # beforeRegisterNodeDef is caught here rather than in the browser.
    ctx.eval("""
    (function () {
        const nt = function () {}; nt.prototype = {};
        const r = __registered.beforeRegisterNodeDef(nt, {name: 'Bat_AdvancedBlend'}, app);
        if (typeof nt.prototype.onNodeCreated !== 'function') throw new Error('no onNodeCreated');
        if (typeof nt.prototype.onExecuted !== 'function') throw new Error('no onExecuted');
    })();
    """)

    # The full-render request key. `runId` must be part of it: without it,
    # changing `preview_frame` and re-running produced a key identical to the
    # render already on screen, the request was skipped as redundant, and the
    # PREVIOUS frame's render stayed pinned on top of the correct new draft for
    # ever. The symptom was "preview_frame doesn't work" and nothing about it
    # pointed at a cache key, which is exactly why it is pinned here.
    ctx.eval("""
    (function () {
        const K = fullRequestKey;
        const region = {world: {x: 0, y: 0, w: 100, h: 80}, outW: 100, outH: 80};
        const base = {params: {mix: 1}, view: "result", amp: 1, region, runId: 1};
        const same = K(base);
        if (K({...base}) !== same) throw new Error('key is not stable');

        // A new execution must produce a different key even when nothing else
        // changed — this is the preview_frame bug.
        if (K({...base, runId: 2}) === same)
            throw new Error('runId is not part of the key: a re-run would be skipped');

        // And the things that describe what to draw must still matter.
        if (K({...base, view: "detail"}) === same) throw new Error('view missing from key');
        if (K({...base, amp: 4}) === same) throw new Error('amp missing from key');
        if (K({...base, params: {mix: 0.5}}) === same) throw new Error('params missing from key');
        if (K({...base, region: {...region, outW: 200}}) === same)
            throw new Error('output size missing from key');
        if (K({...base, region: {...region, world: {x: 5, y: 0, w: 100, h: 80}}}) === same)
            throw new Error('region missing from key');
    })();
    """)

    # The rule that stops the canvas flickering between two sharpnesses mid-drag,
    # and its one exception. Pinned because the exception is the fragile half: if
    # a later edit lets a pan hold the old render, it gets drawn over the wrong
    # part of the picture, which looks like a rendering bug rather than a policy
    # mistake.
    ctx.eval("""
    (function () {
        const H = shouldHoldFullRender;
        // Full + a widget change + something to hold -> hold it.
        if (!H({quality: 'full', hasFull: true, viewChanged: false}))
            throw new Error('Full must hold the render through a widget change');
        // A pan or zoom moved the region: it must be dropped, on every setting.
        if (H({quality: 'full', hasFull: true, viewChanged: true}))
            throw new Error('a view change must drop the render');
        // Half and Quarter chose responsiveness; they keep drop-to-draft.
        if (H({quality: 'half', hasFull: true, viewChanged: false}))
            throw new Error('Half must not hold the render');
        if (H({quality: 'quarter', hasFull: true, viewChanged: false}))
            throw new Error('Quarter must not hold the render');
        // Nothing on screen yet.
        if (H({quality: 'full', hasFull: false, viewChanged: false}))
            throw new Error('nothing to hold on the first pass');
    })();
    """)


REPAIR_SMOKE = r"""
// Reproduce the exact corruption a mid-list serialize=false widget caused on
// copy/paste, then prove repairWidgetValues() undoes it.
//
// The mechanism: the toggle button consumed the slot belonging to `resize_mode`,
// so every value behind it shifted up by one. That is what put `true` in
// frequency_separation, `NaN` in split_radius and a blank in detail_mode.
function _repairSmoke() {
    var RESIZE = ["match_a", "match_b", "match_larger", "match_smaller"];
    var FILTER = ["lanczos", "bicubic", "bilinear", "area", "nearest"];
    var MODES  = ["over", "add", "multiply", "screen", "overlay",
                  "soft_light", "difference", "min", "max"];
    var DETAIL = ["subtract", "divide"];

    function combo(name, values, v) {
        return {name: name, value: v, options: {values: values}};
    }
    function num(name, v) { return {name: name, value: v, options: {}}; }
    function bool(name, v) { return {name: name, value: v, options: {}}; }

    // Pristine node, exactly as the definition creates it.
    function pristine() {
        return [
            combo("blend_mode", MODES, "over"), num("mix", 1.0),
            combo("resize_mode", RESIZE, "match_a"),
            combo("resize_filter", FILTER, "lanczos"),
            bool("frequency_separation", false), num("split_radius", 4.0),
            combo("detail_mode", DETAIL, "subtract"),
            num("low_mix", 1.0), num("high_mix", 1.0), num("detail_gain", 1.0),
            num("detail_limit", 0.0), num("soften_a", 0.0), num("soften_b", 0.0),
            bool("clamp_output", false), num("preview_frame", 0),
        ];
    }
    var defaults = {};
    pristine().forEach(function (w) { defaults[w.name] = w.value; });

    // The artist's real settings, then the same list shifted by one — what
    // paste actually did.
    var saved = ["over", 1.0, "match_a", "lanczos", true, 4.0, "subtract",
                 0.0, 1.0, 1.0, 0.0, 3.0, 0.0, false, 0];
    var widgets = pristine();
    for (var i = 0; i < widgets.length; i++) {
        // Off by one: widget i receives the value meant for widget i-1.
        widgets[i].value = (i === 0) ? saved[0] : saved[i - 1];
    }

    // Sanity: the shift really does produce the symptoms from the bug report,
    // otherwise this test is checking something else.
    var by = {};
    widgets.forEach(function (w) { by[w.name] = w.value; });
    if (RESIZE.includes(by.resize_mode)) throw new Error("expected a broken resize_mode");
    if (typeof by.frequency_separation === "boolean") {
        throw new Error("expected frequency_separation to receive a non-boolean");
    }
    if (typeof by.split_radius === "number") {
        throw new Error("expected split_radius to receive a non-number");
    }

    var fixed = repairWidgetValues(widgets, defaults);
    if (!fixed.length) throw new Error("repair found nothing to fix");

    // Every value must now be legitimate for its own widget.
    for (var j = 0; j < widgets.length; j++) {
        var w = widgets[j];
        var opts = w.options && w.options.values;
        if (opts) {
            if (!opts.includes(w.value)) {
                throw new Error(w.name + " still invalid: " + w.value);
            }
        } else if (typeof defaults[w.name] === "number") {
            if (typeof w.value !== "number" || !isFinite(w.value)) {
                throw new Error(w.name + " still not a finite number: " + w.value);
            }
        } else if (typeof defaults[w.name] === "boolean") {
            if (typeof w.value !== "boolean") {
                throw new Error(w.name + " still not a boolean: " + w.value);
            }
        }
    }

    // A healthy node must be left completely alone — a repair that "corrects"
    // valid values would be worse than the bug it fixes.
    var healthy = pristine();
    healthy[5].value = 12.5;          // split_radius, unusual but legal
    healthy[7].value = 0.0;           // low_mix at an extreme, legal
    healthy[2].value = "match_b";     // a real option, just not the default
    healthy[13].value = true;         // clamp_output on
    var touched = repairWidgetValues(healthy, defaults);
    if (touched.length) throw new Error("repair damaged a healthy node: " + touched.join(","));
    if (healthy[5].value !== 12.5 || healthy[2].value !== "match_b") {
        throw new Error("repair overwrote legitimate values");
    }

    // Unknown widgets (a future addition, or another pack's) are ignored.
    var extra = pristine();
    extra.push({name: "something_new", value: NaN, options: {}});
    repairWidgetValues(extra, defaults);
    if (!isNaN(extra[extra.length - 1].value)) {
        throw new Error("repair touched a widget it has no default for");
    }
    return "ok";
}
"""


STUB_DOM = """
var __registered = null;
var app = {
    registerExtension: function (e) { __registered = e; },
    extensionManager: { setting: { get: function () { return false; } } },
};
function addBatDOMWidget() { return {}; }
function clampNodeSize() {}
function hdrSupported() { return false; }
function decodeHdrTile() {}
function imageDataToSource() {}
function batTrack() { return { listener: function(){}, dispose: function(){}, timeout: function(i){ return i; }, observer: function(){} }; }
function registerCleanup() {}
function batNodeCacheKey() { return "k"; }
function isNodeAlive() { return true; }
function attachZoomControl() { return {setZoom:function(){},getZoom:function(){return 1;},resetView:function(){},refresh:function(){},destroy:function(){}}; }
function fetch() { return Promise.reject(new Error("no network")); }
function AbortController() { this.signal = {aborted:false}; this.abort = function(){}; }
function createImageBitmap() { return Promise.reject(new Error("no bitmaps")); }
var URL = { createObjectURL: function(){ return ""; } };
var window = { devicePixelRatio: 1, addEventListener: function(){} };
var document = { createElement: function(){ throw new Error("no DOM here"); } };
var localStorage = { getItem: function(){ return null; }, setItem: function(){} };
function blendTile() { return {out: new Float32Array(3), high: new Float32Array(3)}; }
function makeBlurCache() { return {get: function(){}, clear: function(){}}; }
function paintView() { return new Uint8ClampedArray(4); }
function neutralHigh() { return 0; }
function Worker() { throw new Error("no workers in this harness"); }
"""


ZOOM_SMOKE = r"""
// A DOM thin enough to run bat_zoom_control.js + bat_paste_guard.js headlessly.
// Not a browser emulation — it exists so a syntax error or stale identifier in
// modules FOUR shipped editors depend on (Crop, Animated Crop, Roto, Advanced
// Blend, plus every DOM widget via bat_node_layout) fails here rather than in
// front of an artist.
var __doc = [];
var __now = 0;
function __el(marks) {
    var attrs = {};
    var self = {
        style: { cssText: "", setProperty: function(){} },
        clientWidth: 400, clientHeight: 300,
        appendChild: function(c){ return c; },
        append: function(){},
        setAttribute: function(k, v){ attrs[k] = v; },
        getAttribute: function(k){ return attrs[k]; },
        hasAttr: function(k){ return k in attrs; },
        closest: function(sel){
            var k = sel.replace(/^\[|\]$/g, "");
            return (k in attrs) ? self : (marks ? marks.closest(sel) : null);
        },
        addEventListener: function(){}, removeEventListener: function(){},
        getBoundingClientRect: function(){ return {left:0, top:0, width:400, height:300}; },
        setPointerCapture: function(){}, releasePointerCapture: function(){},
    };
    return self;
}
var document = {
    createElement: function(){ return __el(); },
    addEventListener: function(t, h, o){ __doc.push({type: t, fn: h, capture: o === true}); },
    removeEventListener: function(t, h, o){
        for (var i = 0; i < __doc.length; i++) {
            if (__doc[i].type === t && __doc[i].fn === h) { __doc.splice(i, 1); return; }
        }
    },
};
var performance = { now: function(){ return __now; } };

function __fire(type, ev) {
    var handled = 0;
    for (var i = 0; i < __doc.length; i++) {
        if (__doc[i].type === type) { __doc[i].fn(ev); handled++; }
    }
    return handled;
}

function _zoomSmoke() {
    var wrap = __el(), canvas = __el();
    var changes = 0;
    // imgW/imgH are part of the contract — see the module docstring.
    var state = { imgW: 200, imgH: 100, dispScale: 1, offX: 0, offY: 0 };
    var ctl = attachZoomControl({
        wrap: wrap, canvas: canvas, state: state,
        onChange: function(){ changes++; },
    });
    var api = ["setZoom", "getZoom", "resetView", "refresh", "destroy"];
    for (var i = 0; i < api.length; i++) {
        if (typeof ctl[api[i]] !== "function") throw new Error("missing " + api[i]);
    }
    if (state.dispZoom !== 1) throw new Error("dispZoom not seeded");
    if (state.panX !== 0 || state.panY !== 0) throw new Error("pan not seeded");

    // The control must mark its canvas, or the guard has nothing to match on.
    if (!canvas.hasAttr(BAT_WIDGET_ATTR)) throw new Error("canvas not marked");

    // Exactly one capture-phase paste listener on the document, however many
    // controls exist: a per-instance one is the accumulating leak, and three
    // shipped editors never call destroy().
    var second = attachZoomControl({
        wrap: __el(), canvas: __el(),
        state: { imgW: 10, imgH: 10, dispScale: 1, offX: 0, offY: 0 },
        onChange: function(){},
    });
    var pastes = 0;
    for (var j = 0; j < __doc.length; j++) {
        if (__doc[j].type === "paste") {
            pastes++;
            if (!__doc[j].capture) throw new Error("paste guard is not capture-phase");
        }
    }
    if (pastes !== 1) throw new Error("paste guard is not a singleton: " + pastes);
    second.destroy();

    // Zoom range is clamped to 0.2x - 4x.
    ctl.setZoom(2);
    if (Math.abs(state.dispZoom - 2) > 1e-9) throw new Error("setZoom did nothing");
    ctl.setZoom(999);
    if (state.dispZoom > 4.000001) throw new Error("zoom not clamped high: " + state.dispZoom);
    ctl.setZoom(0);
    if (state.dispZoom < 0.199999) throw new Error("zoom not clamped low: " + state.dispZoom);

    state.panX = 37; state.panY = -12;
    ctl.resetView();
    if (state.dispZoom !== 1 || state.panX !== 0 || state.panY !== 0) {
        throw new Error("resetView left the view at " + state.dispZoom + "/" + state.panX);
    }
    if (changes === 0) throw new Error("onChange never fired");

    // destroy() must be safe to call repeatedly.
    ctl.destroy(); ctl.destroy();
    return "ok";
}

function _guardSmoke() {
    var swallowed = 0;
    var ev = {
        preventDefault: function(){ swallowed++; },
        stopPropagation: function(){}, stopImmediatePropagation: function(){},
    };

    // Idle: a real Ctrl+V must pass straight through. Swallowing `paste`
    // unconditionally would break paste for the entire application.
    __now = 10000;
    __fire("paste", ev);
    if (swallowed !== 0) throw new Error("guard swallowed an unrelated paste");

    // A middle press OUTSIDE any marked widget must not arm it either — the
    // graph canvas and ordinary text fields have to keep working.
    var bare = __el();
    __fire("pointerdown", {button: 1, target: bare});
    __fire("paste", ev);
    if (swallowed !== 0) throw new Error("guard armed from an unmarked element");

    // A middle press INSIDE a marked widget arms it.
    var marked = __el(); markBatWidget(marked);
    __fire("pointerdown", {button: 1, target: marked});
    __fire("paste", ev);
    if (swallowed !== 1) throw new Error("guard did not swallow the X11 paste");

    // Left button must never arm it.
    __now += 10000;
    __fire("pointerdown", {button: 0, target: marked});
    __fire("paste", ev);
    if (swallowed !== 1) throw new Error("left button armed the guard");

    // It expires: a paste well after the middle interaction is the user's own.
    __fire("pointerup", {button: 1, target: marked});
    __now += 5000;
    __fire("paste", ev);
    if (swallowed !== 1) throw new Error("guard never expires");
    return "ok";
}
"""


def zoom_control_smoke():
    """Load and exercise bat_paste_guard.js + bat_zoom_control.js headlessly."""
    import quickjs

    def plain(name):
        src = open(os.path.join(PACK, "web", name), encoding="utf-8").read()
        src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)
        return re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(ZOOM_SMOKE)
    ctx.eval(plain("bat_paste_guard.js"))
    ctx.eval(plain("bat_zoom_control.js"))
    ctx.eval("_zoomSmoke()")
    ctx.eval("_guardSmoke()")


def repair_smoke():
    """Reproduce the copy/paste value shift and prove the repair undoes it."""
    import quickjs
    raw = open(os.path.join(PACK, "web", "bat_advanced_blend.js"), encoding="utf-8").read()
    src = strip_modules(raw).replace("import.meta.url", '"file:///bat/"')
    ctx = quickjs.Context()
    ctx.eval(auto_stub_js(raw))      # first, so the curated stubs below win
    ctx.eval(STUB_DOM)
    ctx.eval(REPAIR_SMOKE)
    ctx.eval(src)
    ctx.eval("_repairSmoke()")


DRIVER = """
function _run(payload) {
    const j = JSON.parse(payload);
    const a = Float32Array.from(j.a);
    const b = Float32Array.from(j.b);
    const mask = j.mask ? Float32Array.from(j.mask) : null;
    const r = blendTile(a, b, j.w, j.h, mask, j.p, null);
    return JSON.stringify({out: Array.from(r.out), high: Array.from(r.high)});
}
"""


def main():
    import numpy as np
    import torch
    import quickjs

    _load("bat_hdr_preview")
    m = _load("bat_advanced_blend")
    node = m.BatAdvancedBlend()

    js_parses_whole_file()
    print("whole-file JS parse + registration: OK")

    js_worker_parses()
    print("bat_blend_worker.js parses and installs its handler: OK")

    repair_smoke()
    print("copy/paste value-shift repair: reproduces the corruption and fixes it: OK")

    zoom_control_smoke()
    print("bat_zoom_control.js + bat_paste_guard.js: API, zoom clamp, resetView, "
          "singleton capture-phase guard, arming rules, expiry: OK")

    ctx = quickjs.Context()
    ctx.eval(js_prelude())
    ctx.eval(DRIVER)
    run_js = ctx.get("_run")

    torch.manual_seed(7)
    # Deliberately odd and small: exercises the kernel-truncation path at the
    # frame edge, where a radius wider than the tile has to be clipped and the
    # Gaussian renormalised — the most likely place for the two sides to differ.
    W, H = 37, 29

    # A structured pair rather than pure noise, so a mistake in the band split
    # shows up as a visible bias rather than hiding inside random values.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W),
                            indexing="ij")
    base = (xx * 0.55 + yy * 0.3).unsqueeze(-1).repeat(1, 1, 3)
    base[8:16, 10:24] = 0.93
    base[20:26, 4:30] = 0.04
    plate_b = (base + torch.rand(H, W, 3) * 0.06).clamp(0, 1)
    # A is "the upscale": same picture, more high-frequency energy, slightly
    # lifted — which is what an over-sharpening refiner actually hands back.
    plate_a = (plate_b * 1.04 + torch.rand(H, W, 3) * 0.22 - 0.08).clamp(0, 1)

    # Scene-linear versions with real highlight range — a small blown region
    # at 40x white, which is what an EXR off a render actually looks like and
    # what clamp_output=False exists to protect.
    hdr_b = plate_b * 3.0
    hdr_b[6:12, 6:14] = 40.0
    hdr_a = plate_a * 3.0
    hdr_a[6:12, 6:14] = 46.5
    hdr_a[18:22, 8:20] = 12.0

    mask = torch.rand(1, H, W)

    cases = []
    for freq in (False, True):
        for mode in m.BLEND_MODES:
            cases.append(dict(blend_mode=mode, frequency_separation=freq))
    cases += [
        dict(frequency_separation=True, detail_mode="divide"),
        dict(frequency_separation=True, detail_mode="divide", detail_limit=1.8),
        dict(frequency_separation=True, detail_mode="divide", detail_limit=0.5),
        dict(frequency_separation=True, detail_limit=0.05),
        dict(frequency_separation=True, low_mix=0.0, high_mix=1.0),
        dict(frequency_separation=True, low_mix=1.0, high_mix=0.35),
        dict(frequency_separation=True, detail_gain=2.4),
        dict(frequency_separation=True, detail_gain=0.0),
        dict(frequency_separation=True, split_radius=1.0),
        dict(frequency_separation=True, split_radius=11.0),
        dict(frequency_separation=True, split_radius=60.0),   # wider than the tile
        dict(frequency_separation=True, soften_a=3.0),
        dict(frequency_separation=True, soften_a=2.0, soften_b=5.0),
        dict(soften_a=4.0, soften_b=1.0),
        dict(mix=0.0), dict(mix=0.42), dict(clamp_output=False),
        dict(frequency_separation=True, clamp_output=False, detail_gain=3.0),
        dict(frequency_separation=True, blend_mode="screen", mix=0.63,
             low_mix=0.7, high_mix=0.2, detail_gain=1.5, soften_a=2.0,
             split_radius=6.0, use_mask=True),
        dict(frequency_separation=True, detail_mode="divide", use_mask=True,
             mix=0.8, low_mix=0.3, high_mix=0.9, detail_limit=3.0),
        # HDR: clamp_output off is the default, so these are the ordinary path.
        dict(hdr=True, clamp_output=False),
        dict(hdr=True, clamp_output=False, blend_mode="add"),
        dict(hdr=True, clamp_output=False, frequency_separation=True),
        dict(hdr=True, clamp_output=False, frequency_separation=True,
             detail_mode="divide", detail_limit=4.0),
        dict(hdr=True, clamp_output=False, frequency_separation=True,
             low_mix=0.0, high_mix=1.0, soften_a=3.0),
    ]

    worst = 0.0
    worst_case = None
    failures = []

    for i, over in enumerate(cases):
        p = dict(blend_mode="over", mix=1.0, frequency_separation=False,
                 split_radius=4.0, detail_mode="subtract", low_mix=1.0,
                 high_mix=1.0, detail_gain=1.0, detail_limit=0.0,
                 soften_a=0.0, soften_b=0.0, clamp_output=True)
        use_mask = over.pop("use_mask", False)
        # `hdr` swaps in plates carrying values well above white, to prove the
        # whole chain is range-agnostic and that the two sides agree there too.
        hdr = over.pop("hdr", False)
        p.update(over)
        src_a, src_b = (hdr_a, hdr_b) if hdr else (plate_a, plate_b)
        in_a, in_b = src_a.unsqueeze(0), src_b.unsqueeze(0)

        # Python. resize_mode/filter are irrelevant here — both plates are
        # already the same size, which is the state the JS always sees.
        res = node.blend(image_a=in_a, image_b=in_b, resize_mode="match_a",
                         resize_filter="area", preview_frame=0,
                         mask=(mask if use_mask else None), **p)
        py_out = res["result"][0][0].numpy().reshape(-1)
        # Re-derive the high band the same way _core does, so the `detail`
        # output's 0.5 offset doesn't have to be undone by hand here.
        neutral = 1.0 if p["detail_mode"] == "divide" else 0.0
        py_high = (res["result"][1][0].numpy() - 0.5 + neutral).reshape(-1)

        # JS
        payload = json.dumps({
            "a": src_a.numpy().reshape(-1).tolist(),
            "b": src_b.numpy().reshape(-1).tolist(),
            "w": W, "h": H,
            "mask": (mask[0].numpy().reshape(-1).tolist() if use_mask else None),
            "p": p,
        })
        js = json.loads(run_js(payload))
        js_out = np.asarray(js["out"], dtype=np.float64)
        js_high = np.asarray(js["high"], dtype=np.float64)

        d_out = float(np.abs(js_out - py_out).max())
        d_high = float(np.abs(js_high - py_high).max())
        d = max(d_out, d_high)
        if d > worst:
            worst, worst_case = d, dict(p, use_mask=use_mask)
        if d > 1e-4:
            failures.append((i, dict(p, use_mask=use_mask), d_out, d_high))

    label = ", ".join(f"{k}={v}" for k, v in sorted((worst_case or {}).items()))
    # ── the full-resolution preview service ──────────────────────────────
    #
    # The endpoint renders a REGION rather than a frame, which means slicing the
    # plates and relying on a context margin so the blurs see the same
    # neighbourhood they would have at full frame. Get that margin wrong and the
    # region is correct in the middle and wrong in a band around the edge — a
    # faint seam that only shows up once the artist pans, i.e. exactly the kind
    # of bug that ships. So assert the strong property directly: every region,
    # under every parameter set, is BIT-IDENTICAL to the same crop of the
    # full-frame render.
    roi_worst, roi_bad = 0, []
    entry = {"a": plate_a.unsqueeze(0), "b": plate_b.unsqueeze(0),
             "mask": mask, "meta": {}}

    def rp(**kw):
        base = dict(blend_mode="over", mix=1.0, frequency_separation=True,
                    split_radius=4.0, detail_mode="subtract", low_mix=1.0,
                    high_mix=1.0, detail_gain=1.0, detail_limit=0.0,
                    soften_a=0.0, soften_b=0.0, clamp_output=False,
                    mask_chunk=None)
        base.update(kw)
        return base

    roi_cases = [
        ("defaults", rp()),
        ("wide split", rp(split_radius=13.0)),
        ("soften both", rp(soften_a=6.0, soften_b=9.0, split_radius=11.0)),
        ("detail only", rp(low_mix=0.0, high_mix=1.0, soften_a=3.0)),
        ("divide", rp(detail_mode="divide", detail_limit=3.0, split_radius=7.0)),
        ("freq off", rp(frequency_separation=False, soften_a=5.0, soften_b=2.0)),
        ("screen+mix", rp(blend_mode="screen", mix=0.6, split_radius=6.0, soften_a=4.0)),
    ]
    # Corners and an off-grid interior region, because the margin has to clamp
    # at the frame edge and not at an interior one.
    regions = [(0, 0, W, H), (11, 9, 17, 13), (0, 0, 9, 7),
               (W - 9, H - 7, 9, 7), (W // 3, H // 3, 12, 11)]
    for name, prm in roi_cases:
        for view in ("result", "detail", "diff"):
            ref = m.render_region(entry, prm, (0, 0, W, H), W, H, view, 4.0)
            ref = ref.astype(np.int32)
            for (rx, ry, rw, rh) in regions:
                got = m.render_region(entry, prm, (rx, ry, rw, rh), rw, rh,
                                      view, 4.0).astype(np.int32)
                d = int(np.abs(got - ref[ry:ry + rh, rx:rx + rw]).max())
                roi_worst = max(roi_worst, d)
                if d > 0:
                    roi_bad.append((name, view, (rx, ry, rw, rh), d))

    n_roi = len(roi_cases) * 3 * len(regions)
    print(f"{n_roi} region renders vs full-frame: worst delta = {roi_worst} code value(s)")
    if roi_bad:
        failures.extend((f"render_region[{n}/{v}]{r}", f"delta {d}")
                        for n, v, r, d in roi_bad[:8])

    # The request parser is reachable from the network, so it has to refuse
    # nonsense rather than pass a NaN radius into an allocation.
    hostile = m._params_from_request({
        "blend_mode": "'; DROP", "detail_mode": "nope",
        "split_radius": float("nan"), "soften_a": 1e9, "mix": "abc",
        "detail_limit": float("-inf"), "high_mix": None,
    })
    assert hostile["blend_mode"] == "over", hostile["blend_mode"]
    assert hostile["detail_mode"] == "subtract", hostile["detail_mode"]
    assert hostile["split_radius"] == 4.0, hostile["split_radius"]
    assert hostile["soften_a"] == 64.0, hostile["soften_a"]
    assert hostile["mix"] == 1.0, hostile["mix"]
    assert hostile["detail_limit"] == 0.0, hostile["detail_limit"]
    assert np.isfinite(list(v for v in hostile.values()
                            if isinstance(v, float))).all()
    print("request parser rejects malformed input: OK")

    print(f"{len(cases)} cases | worst |JS - Python| = {worst:.3e}")
    print(f"  at: {label}")

    if failures:
        print(f"\nFAIL — {len(failures)} case(s) above 1e-4:")
        for i, p, d_out, d_high in failures:
            print(f"  [{i}] out {d_out:.3e}  high {d_high:.3e}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(p.items())))
        return 1

    print("OK — the live preview and the render agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
