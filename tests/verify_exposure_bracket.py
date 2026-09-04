#!/usr/bin/env python
"""
Prove that the Bat_ExposureBracket / Bat_ExposureMerge live previews compute
the same thing the render does.

Same contract, and the same reason, as `verify_advanced_blend.py`: the on-node
canvas is a second implementation of the render, written in another language,
and two hand-written implementations of one algorithm drift the first time
somebody edits either side. So run both and diff them.

What is covered
---------------
* `bat_transfer.js` against the Python transfer functions — every mode, both
  directions, including the deliberate asymmetry where `linear` decodes as the
  identity but *encodes* through sRGB.
* `exposeToSdr` against `_expose_to_sdr`, across the gamma modes and a range of
  stops, on scene-linear input with values well above white (which is the whole
  point of the strip).
* The merge: `passWeights` / `computeAlignment` / `mergeTiles` against
  `_well_exposed`, `_alignment` and `_weighted_merge`, over every `align` mode,
  both `reference` modes, several sigmas, and 1..5 passes.
* The whole extension file, evaluated with its imports stubbed, so a syntax
  error or a stale identifier in the editor half fails here rather than showing
  up as a blank node in the browser.

    $ ../../../env/bin/python tests/verify_exposure_bracket.py

Exits non-zero on any mismatch above tolerance.

Tolerance
---------
Relative, not absolute — see the ATOL/RTOL comment below for why an absolute
bound is the wrong instrument on unbounded HDR data.

One honest exclusion
--------------------
`align="auto"` measures the ratio between two passes over the whole frame in
the render and over a 256px tile in the preview, so the two cannot agree by
construction. The test therefore drives the JS with the *same* buffer the
Python gets, which isolates the arithmetic — that is what can drift. The
tile-vs-frame difference is a known approximation, is why the render's real
scales are shipped down to the preview, and is why the preview labels the
number as an estimate the moment a widget makes it recompute.
"""

import importlib.util
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
JS_EXT = os.path.join(PACK, "web", "bat_exposure_bracket.js")
JS_TRANSFER = os.path.join(PACK, "web", "bat_transfer.js")

# Tolerance is relative-plus-absolute, not absolute: |py - js| <= ATOL + RTOL*|py|.
#
# An absolute bound is meaningless on the data these nodes carry. A scene-linear
# plate decoded through the sRGB curve reaches 1.5e6 on this test's inputs, where
# float32's own epsilon is 0.18 — so an absolute 1e-4 fails on arithmetic that is
# perfectly correct. And the merge divides an accumulated sum by a weight total
# that legitimately gets down to ~0.03, which multiplies float32's accumulation
# error by 30 before it is compared.
#
# Both were measured: the transfer functions agree to a flat ~1.1e-7 RELATIVE
# across six decades of magnitude, and the merge to ~2e-5 relative. Both are
# float32 eps and nothing else. RTOL 1e-4 sits comfortably above them while
# still catching any real algorithmic drift, which shows up as a relative error
# of 1e-2 or worse — a wrong constant or a mis-placed branch is never a
# 0.01%-of-value mistake.
ATOL = 1e-5
RTOL = 1e-4


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


def _strip_modules(src):
    """Turn an ES module into a plain script quickjs can evaluate."""
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)
    src = re.sub(r"^export ", "", src, flags=re.M)
    return src


STUBS = """
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
function batTrack() { return { listener: function(){}, dispose: function(){} }; }
function registerCleanup() {}
function batNodeCacheKey() { return "k"; }
function isNodeAlive() { return true; }
var document = { createElement: function () { throw new Error("no DOM in this harness"); } };
var localStorage = { getItem: function () { return null; }, setItem: function () {} };
"""

DRIVER = """
function _transfer(payload) {
    const j = JSON.parse(payload);
    const out = [];
    for (const v of j.values) {
        out.push(j.dir === "encode" ? encodeFromLinear(v, j.mode) : toLinear(v, j.mode));
    }
    return JSON.stringify(out);
}

function _expose(payload) {
    const j = JSON.parse(payload);
    const scale = Math.pow(2, j.ev);
    const out = [];
    for (const v of j.values) out.push(exposeToSdr(v, scale, j.mode));
    return JSON.stringify(out);
}

function _weights(payload) {
    const j = JSON.parse(payload);
    const plate = Float32Array.from(j.plate);
    return JSON.stringify(Array.from(
        passWeights(plate, j.w, j.h, j.ev, j.mode, j.sigma)));
}

function _align(payload) {
    const j = JSON.parse(payload);
    const plate = Float32Array.from(j.plate);
    const passes = j.passes.map(p => ({ev: p.ev, lin: Float32Array.from(p.lin)}));
    const a = computeAlignment(passes, plate, j.w, j.h, j.mode,
                               j.align, j.reference, j.sigma);
    return JSON.stringify(a.scales);
}

function _merge(payload) {
    const j = JSON.parse(payload);
    const plate = Float32Array.from(j.plate);
    const passes = j.passes.map(p => ({ev: p.ev, lin: Float32Array.from(p.lin)}));
    const r = mergeTiles(passes, plate, j.w, j.h, j.mode, j.scales, j.sigma);
    return JSON.stringify({out: Array.from(r.out), wsum: Array.from(r.wsum)});
}
"""


def js_parses_whole_file():
    """Evaluate the entire extension, imports stubbed, and drive registration."""
    import quickjs
    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(_strip_modules(open(JS_TRANSFER, encoding="utf-8").read()))
    ctx.eval(_strip_modules(open(JS_EXT, encoding="utf-8").read()))
    ctx.eval("if (!__registered) throw new Error('extension did not register');")
    for name in ("Bat_ExposureBracket", "Bat_ExposureMerge"):
        ctx.eval("""
        (function () {
            const nt = function () {}; nt.prototype = {};
            __registered.beforeRegisterNodeDef(nt, {name: %r}, app);
            for (const h of ['onNodeCreated', 'onAfterGraphConfigured', 'onExecuted']) {
                if (typeof nt.prototype[h] !== 'function')
                    throw new Error('%s: missing ' + h);
            }
        })();
        """ % (name, name))


def make_ctx():
    import quickjs
    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(_strip_modules(open(JS_TRANSFER, encoding="utf-8").read()))
    ctx.eval(_strip_modules(open(JS_EXT, encoding="utf-8").read()))
    ctx.eval(DRIVER)
    return ctx


def main():
    import numpy as np
    import torch

    hp = _load("bat_hdr_preview")          # noqa: F841 - import side effect
    comp = _load("bat_hdr_tonal_composite")
    m = _load("bat_exposure_bracket")

    js_parses_whole_file()
    print("whole-file JS parse + registration (both node types): OK")

    ctx = make_ctx()
    js_transfer = ctx.get("_transfer")
    js_expose = ctx.get("_expose")
    js_weights = ctx.get("_weights")
    js_align = ctx.get("_align")
    js_merge = ctx.get("_merge")

    MODES = ["srgb", "rec709", "gamma_2_2", "gamma_2_4", "linear"]
    worst = {}

    def note(label, d):
        worst[label] = max(worst.get(label, 0.0), float(d))

    failures = []

    def check(label, py, js):
        """Compare, scoring each element against its own allowance.

        `resid` is the residual as a fraction of what is permitted, so 1.0 means
        "exactly at tolerance" and the numbers stay comparable across checks
        whose values differ by six orders of magnitude.
        """
        py = np.asarray(py, dtype=np.float64).reshape(-1)
        js = np.asarray(js, dtype=np.float64).reshape(-1)
        if py.shape != js.shape:
            failures.append((label, f"shape {py.shape} vs {js.shape}"))
            return
        if not py.size:
            return
        allow = ATOL + RTOL * np.abs(py)
        resid = np.abs(py - js) / allow
        i = int(np.argmax(resid))
        note(label, resid[i])
        if resid[i] > 1.0:
            failures.append((label, f"resid {resid[i]:.2f}x tolerance "
                                    f"(|d| {abs(py[i] - js[i]):.3e} at value {py[i]:.6g})"))

    # ── 1. transfer functions ────────────────────────────────────────────
    # Includes negatives (both sides floor at 0), the piecewise knees, and
    # values far above white.
    vals = [-0.5, -1e-9, 0.0, 1e-6, 0.003, 0.0031308, 0.004, 0.04, 0.04045,
            0.05, 0.081, 0.1, 0.25, 0.5, 0.9, 1.0, 1.0001, 2.0, 12.0, 47.5, 400.0]
    t = torch.tensor(vals, dtype=torch.float32)
    for mode in MODES:
        py = comp._to_linear(t, mode).numpy()
        js = json.loads(js_transfer(json.dumps({"values": vals, "mode": mode, "dir": "decode"})))
        check(f"toLinear[{mode}]", py, js)

        py = comp._encode_from_linear(t, mode).numpy()
        js = json.loads(js_transfer(json.dumps({"values": vals, "mode": mode, "dir": "encode"})))
        check(f"encodeFromLinear[{mode}]", py, js)

    # ── 2. exposeToSdr ───────────────────────────────────────────────────
    for mode in MODES:
        for ev in [0.0, -1.0, -1.5, -3.0, -4.5, -6.0, 1.5, 3.0]:
            py = m._expose_to_sdr(t.reshape(1, 1, -1, 1).repeat(1, 1, 1, 3),
                                  ev, mode)[0, 0, :, 0].numpy()
            js = json.loads(js_expose(json.dumps({"values": vals, "ev": ev, "mode": mode})))
            check(f"exposeToSdr[{mode}]", py, js)

    # ── 3. the merge ─────────────────────────────────────────────────────
    torch.manual_seed(11)
    H, W = 23, 31
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W),
                            indexing="ij")
    # A scene-linear plate with a genuinely blown region, which is the case the
    # bracket exists for: exposing down has somewhere to go.
    plate_lin = (xx * 2.2 + yy * 0.6).unsqueeze(-1).repeat(1, 1, 3)
    plate_lin[4:10, 6:18] = 38.0
    plate_lin[16:20, 3:26] = 0.002
    plate_lin = (plate_lin + torch.rand(H, W, 3) * 0.05).clamp(min=0.0)
    plate4 = plate_lin.unsqueeze(0)

    ALL_EVS = [0.0, -1.5, -3.0, -4.5, -6.0]

    def make_pass(ev, seed):
        """A plausible LTX return: roughly the plate at that stop's scale, with
        its own hallucination on top, so the measured alignment has something
        real to find and the per-pass weights actually differ."""
        g = torch.Generator().manual_seed(seed)
        k = 2.0 ** (-ev)
        return ((plate_lin * (1.0 / max(k, 1e-6)) * 0.94
                 + torch.rand(H, W, 3, generator=g) * 0.04)
                * k).clamp(min=0.0).unsqueeze(0)

    for mode in ["srgb", "rec709", "linear"]:
        for sigma in [0.1, 0.2, 0.35]:
            for ev in ALL_EVS:
                py = m._well_exposed(
                    comp._luminance(m._expose_to_sdr(plate4, ev, mode)), sigma
                )[0].numpy()
                js = json.loads(js_weights(json.dumps({
                    "plate": plate_lin.numpy().reshape(-1).tolist(),
                    "w": W, "h": H, "ev": ev, "mode": mode, "sigma": sigma,
                })))
                check(f"passWeights[{mode}]", py, js)

    node = m.BatExposureMerge()
    n_align = 0
    for mode in ["srgb", "rec709", "linear"]:
        for n_pass in (1, 2, 3, 5):
            evs = ALL_EVS[:n_pass]
            got = [(ev, make_pass(ev, 100 + i)) for i, ev in enumerate(evs)]
            js_passes = [{"ev": ev, "lin": img[0].numpy().reshape(-1).tolist()}
                         for ev, img in got]
            for align in ("off", "nominal", "auto"):
                for reference in ("auto", "first"):
                    for sigma in (0.15, 0.2, 0.3):
                        py_scales = node._alignment(got, plate4, mode, align,
                                                    reference, sigma)
                        payload = json.dumps({
                            "plate": plate_lin.numpy().reshape(-1).tolist(),
                            "passes": js_passes, "w": W, "h": H, "mode": mode,
                            "align": align, "reference": reference, "sigma": sigma,
                        })
                        js_scales = json.loads(js_align(payload))
                        check(f"computeAlignment[{align}/{reference}]",
                              py_scales, js_scales)
                        n_align += 1

                        py_out = node._weighted_merge(
                            got, plate4, mode, py_scales, sigma)[0].numpy()
                        js_out = json.loads(js_merge(json.dumps({
                            "plate": plate_lin.numpy().reshape(-1).tolist(),
                            "passes": js_passes, "w": W, "h": H, "mode": mode,
                            "scales": [float(k) for k in py_scales], "sigma": sigma,
                        })))["out"]
                        check("mergeTiles", py_out, js_out)

    print(f"{n_align} merge configurations x {len(MODES)} transfer modes")
    print("  (residual = worst error as a fraction of its allowance; "
          "1.00 would be exactly at tolerance)")
    for label in sorted(worst):
        print(f"  {label:34s} residual {worst[label]:6.3f}x")

    if failures:
        print(f"\nFAIL — {len(failures)} check(s) over tolerance:")
        for label, why in failures:
            print(f"  {label}: {why}")
        return 1
    print("OK — both live previews agree with the render.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
