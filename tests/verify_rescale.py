#!/usr/bin/env python
"""
Equivalence checks for Bat_Rescale.

Three claims are load-bearing enough to be worth a test rather than an
argument:

1. **A region render is the render.** Asking the preview endpoint for
   destination pixels [dx0,dx1) x [dy0,dy1) must return exactly what cropping
   that rect out of a full-frame resize would. Get the kernel's context margin
   wrong and the result is right in the middle of a region and wrong in a band
   around its edge — a seam that only appears once the artist pans, on the very
   tool they are using to judge sharpness.

2. **This module's resampler is the pack's resampler.** `_map_matrix` is a
   phase-explicit generalisation of `bat_advanced_blend._resample_matrix`; for a
   whole frame the two must agree to float noise, or the pack has two lanczos
   implementations that will drift.

3. **The size readout is the size rendered.** `web/bat_rescale.js` mirrors
   `plan_size()` so the node can show the output resolution before you run it.
   The JS is evaluated here through quickjs (there is no node binary on the
   render machines) and driven against the Python over every mode, including
   the rounding edges where Python's banker's rounding and JS's Math.round
   disagree if either side uses its language's default.

    pip install quickjs
    python tests/verify_rescale.py
"""

import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The pack imports as a package elsewhere; here the modules are loaded directly
# so the test can run without ComfyUI on the path.
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

# bat_rescale imports `_kernel` / `_num` from bat_advanced_blend as
# `from .bat_advanced_blend import ...`, which needs a package context. Fake the
# minimum: load the blend module under the package name the import expects.
import types

pkg = types.ModuleType("batpack")
pkg.__path__ = [PACK]
sys.modules["batpack"] = pkg
blend = _load("batpack.bat_advanced_blend",
              os.path.join(PACK, "bat_advanced_blend.py"))
rescale = _load("batpack.bat_rescale", os.path.join(PACK, "bat_rescale.py"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# 1. Region renders are bit-identical to a crop of the full render
# ---------------------------------------------------------------------------

def _plate(h, w, seed=0):
    """A plate with real high-frequency content.

    Grain plus hard edges plus a zone plate: the three things a resample can get
    wrong in different ways (aliasing, ringing, phase).
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(-1, 1, h).view(h, 1)
    x = torch.linspace(-1, 1, w).view(1, w)
    r2 = x * x + y * y
    zone = 0.5 + 0.5 * torch.cos(r2 * 220.0)
    edges = ((x * 7).floor() % 2 == 0).float() * ((y * 5).floor() % 2 == 0).float()
    grain = torch.rand((h, w), generator=g) * 0.25
    img = (zone * 0.6 + edges * 0.25 + grain).clamp(0, 1.4)
    return img.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)


def test_region_exact():
    src = _plate(360, 480, seed=3)
    fh, fw = src.shape[1], src.shape[2]
    for filt in rescale.FILTERS:
        for out_h, out_w in ((180, 240), (137, 211), (540, 720)):
            full = rescale.resample_region(src, out_h, out_w, filt)
            sx, sy = out_w / fw, out_h / fh
            worst = 0.0
            for (dx0, dy0, dw, dh) in ((0, 0, 31, 23),
                                       (out_w - 40, out_h - 30, 40, 30),
                                       (out_w // 3, out_h // 4,
                                        max(1, out_w // 5), max(1, out_h // 6))):
                dx1, dy1 = dx0 + dw, dy0 + dh
                sl_x0, sl_x1 = rescale._pad_interval(dx0 / sx, dx1 / sx, dw, filt, 0, fw)
                sl_y0, sl_y1 = rescale._pad_interval(dy0 / sy, dy1 / sy, dh, filt, 0, fh)
                slab = src[:, sl_y0:sl_y1, sl_x0:sl_x1]
                got = rescale.resample_region(
                    slab, dh, dw, filt,
                    src_x=dx0 / sx, src_y=dy0 / sy,
                    src_w=dw / sx, src_h=dh / sy,
                    off_x=sl_x0, off_y=sl_y0)
                want = full[:, dy0:dy1, dx0:dx1]
                worst = max(worst, float((got - want).abs().max()))
            # 1e-4 absolute, the same bound verify_advanced_blend uses. The
            # observed worst case is ~2.5e-5 on a lanczos enlargement, which is
            # float32 matmul noise between two different matrix shapes; the
            # box and nearest kernels come out exactly zero, as they must.
            check(f"region == full-frame crop  [{filt} {fw}x{fh}->{out_w}x{out_h}]",
                  worst < 1e-4, f"max |Δ| = {worst:.2e}")


# ---------------------------------------------------------------------------
# 2. Whole-frame agreement with the pack's existing resampler
# ---------------------------------------------------------------------------

def test_blend_parity():
    """...with one documented exception.

    `area` while ENLARGING is the one case the two kernels disagree, and the
    disagreement is Advanced Blend's: its box is closed on both sides
    (`|x| < 0.5`), so an output centre landing exactly on an input pixel
    boundary zeroes both taps and falls through to a nearest fallback whose
    rounding is not shift-invariant. Every third sample of a 2:3 resize lands
    there. This module's box is half-open, which removes the case. The
    divergence is asserted rather than skipped, so if that node is ever fixed
    this test says so instead of quietly passing.
    """
    src = _plate(256, 320, seed=7)
    for filt in rescale.FILTERS:
        for out_h, out_w in ((128, 160), (200, 480), (256, 320), (512, 640)):
            a = rescale.resample_region(src, out_h, out_w, filt)
            b = blend._resize(src, out_h, out_w, filt)
            d = float((a - b).abs().max())
            boundary_case = (filt == "area" and out_w == 480)
            name = f"agrees with bat_advanced_blend._resize  [{filt} ->{out_w}x{out_h}]"
            if boundary_case:
                check(name + "  (expected to differ: closed-box hole)",
                      d > 1e-3, f"max |Δ| = {d:.2e}")
            else:
                check(name, d < 1e-5, f"max |Δ| = {d:.2e}")


# ---------------------------------------------------------------------------
# 3. plan_size() parity with the JS mirror
# ---------------------------------------------------------------------------

CASES = []
for (h, w) in ((1080, 1920), (2160, 3840), (1000, 1000), (721, 1281), (13, 4096)):
    for mode in rescale.MODES:
        for scale in (0.25, 0.5, 0.6667, 1.0, 1.5, 2.0):
            for target in (512, 1024, 1920):
                for mp in (0.25, 1.0, 4.0):
                    for mult in (1, 8, 64):
                        CASES.append((h, w, mode, scale, target, mp, mult))
# Rounding edges: a factor that lands exactly on .5 on one or both axes. This is
# the case Python's banker's rounding and JS's Math.round disagree on, and the
# reason both sides spell the rounding out as floor(x + 0.5).
CASES += [(1921, 1921, "factor", 0.5, 1024, 1.0, 1),
          (1921, 1080, "factor", 0.5, 1024, 1.0, 1),
          (3, 3, "factor", 0.5, 1024, 1.0, 1),
          (1080, 1920, "width", 1.0, 961, 1.0, 1),
          (1080, 1920, "long_edge", 1.0, 961, 1.0, 8)]


STUBS = """
    globalThis.__registered = null;
    var app = { registerExtension: function (e) { globalThis.__registered = e; },
                graph: { extra: {} }, extensionManager: {}, ui: {} };
    function addBatDOMWidget() { return { computeLayoutSize: function () {} }; }
    function clampNodeSize() {}
    function batTrack() {
        return { interval: function (x) { return x; }, timeout: function (x) { return x; },
                 observer: function (o) { return o; }, listener: function () {},
                 rafLoop: function () {}, dispose: function () {}, dead: false };
    }
    function isNodeAlive() { return true; }
    function batNodeCacheKey(a, p, n) { return p + "_" + (n && n.id); }
    var localStorage = { getItem: function () { return null; },
                         setItem: function () {}, removeItem: function () {} };
    var window = { devicePixelRatio: 1, addEventListener: function () {} };
    var ResizeObserver = function () { this.observe = function () {};
                                       this.disconnect = function () {}; };
    var Image = function () {};
    var document = { createElement: function () {
        return { style: { setProperty: function () {}, cssText: "" },
                 appendChild: function () {}, append: function () {},
                 addEventListener: function () {}, classList: { add: function () {} },
                 getContext: function () { return null; },
                 focus: function () {}, setAttribute: function () {} };
    } };
    var requestAnimationFrame = function () { return 0; };
    var fetch = function () { return Promise.reject(new Error("no server in the test")); };
    var createImageBitmap = function () { return Promise.reject(new Error("no bitmaps")); };
"""


def _js_context():
    """Evaluate the extension in quickjs with ComfyUI stubbed out.

    The file is evaluated whole rather than sliced: everything this test drives
    is self-contained, and evaluating all of it means a syntax error or a stale
    identifier anywhere in the viewer fails here rather than showing up as a
    blank node in a browser. Copying the functions out instead would drift
    exactly the way this test exists to prevent.
    """
    import quickjs

    path = os.path.join(PACK, "web", "bat_rescale.js")
    src = open(path, encoding="utf-8").read()
    # Drop the ES imports — the stubs above stand in for what they provide.
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
    # quickjs has no module loader, so drop the export keyword wherever it
    # appears (functions and consts alike) and let everything land as globals.
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(src)
    return ctx


def test_plan_size_parity():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        print("SKIP  plan_size JS parity — pip install quickjs to run it")
        return

    ctx = _js_context()
    ctx.eval("""
        globalThis.__plan = function (h, w, mode, scale, target, mp, mult, rh, rw) {
            const r = planSize(h, w, {
                mode: mode, scale: scale, target: target, megapixels: mp,
                multiple_of: mult, ref_h: rh || null, ref_w: rw || null,
            });
            return r.h + "x" + r.w;
        };
    """)
    plan = ctx.get("__plan")

    bad = []
    for (h, w, mode, scale, target, mp, mult) in CASES:
        ref_h, ref_w = 720, 1280
        ph, pw = rescale.plan_size(h, w, mode, scale, target, mp, mult,
                                   ref_h, ref_w)
        js = plan(h, w, mode, float(scale), int(target), float(mp), int(mult),
                  ref_h, ref_w)
        if js != f"{ph}x{pw}":
            bad.append(f"{mode} {w}x{h} s={scale} t={target} mp={mp} m={mult}: "
                       f"py {ph}x{pw} != js {js}")
    check(f"planSize JS mirrors plan_size  [{len(CASES)} cases]", not bad)
    for line in bad[:8]:
        print("      ", line)


# ---------------------------------------------------------------------------
# 4. The wipe divider's grab zone is where the divider is drawn
# ---------------------------------------------------------------------------

def test_divider_hit_test():
    """A pointer landing on the divider must hit it at ANY graph zoom.

    This is a regression test, not a hypothetical: the first version converted
    pointer events with devicePixelRatio alone, which is exactly right at graph
    zoom 1.0 and progressively offset either side of it — litegraph scales the
    node with a CSS transform that `clientWidth` knows nothing about. The
    symptom was a handle you had to aim next to rather than at, by an amount
    that depended on the graph zoom, and at far-out zooms could not hit at all.

    So each case computes where the divider APPEARS on screen, dispatches a
    click there, and asserts two things: the fixed conversion hits, and the old
    devicePixelRatio-only conversion would have missed. The second half is what
    stops this test passing vacuously if the offset is ever reintroduced.
    """
    try:
        import quickjs           # noqa: F401
    except ImportError:
        print("SKIP  divider hit-test — pip install quickjs to run it")
        return

    ctx = _js_context()
    ctx.eval("""
        globalThis.__hit = function (layoutW, layoutH, rectW, dpr, wipe, wantMiss) {
            // The canvas box, in device pixels, exactly as computeView() builds it.
            const view = { drawX: 0, drawY: 0,
                           outW: Math.round(layoutW * dpr),
                           outH: Math.round(layoutH * dpr) };
            const { toDevice } = pointerToDevice(rectW, layoutW, dpr);
            const geom = wipeGeom(view, wipe, toDevice);

            // Where that divider lands on screen: device px -> displayed px.
            const shownX = geom.x / toDevice;
            const shownY = (view.drawY + view.outH / 2) / toDevice;

            // The conversion the viewer uses.
            const good = isOnDivider(geom, view, shownX * toDevice, shownY * toDevice);
            // The conversion it used to use: devicePixelRatio only. Its error
            // grows with distance from the origin, so near the left edge it can
            // still land inside the tolerance by luck — the test accounts for
            // that rather than asserting it always missed.
            const naive = isOnDivider(geom, view, shownX * dpr, shownY * dpr);
            const offset = Math.abs(shownX * toDevice - shownX * dpr);
            return [good ? "hit" : "MISS", naive ? "hit" : "miss",
                    offset.toFixed(1), geom.tol.toFixed(1)].join("/");
        };
    """)
    hit = ctx.get("__hit")

    # (layout px, displayed px, dpr) — a graph zoom of 0.5 halves the displayed
    # width against the layout width, 2.0 doubles it.
    cases = [
        ("graph 100%, dpr 1",   400, 300, 400, 1.0),
        ("graph 100%, dpr 2",   400, 300, 400, 2.0),
        ("graph 50%,  dpr 1",   400, 300, 200, 1.0),
        ("graph 50%,  dpr 2",   400, 300, 200, 2.0),
        ("graph 150%, dpr 1",   400, 300, 600, 1.0),
        ("graph 25%,  dpr 1",   800, 450, 200, 1.0),
    ]
    for (name, lw, lh, rw, dpr) in cases:
        for wipe in (0.03, 0.5, 0.97):
            got = hit(lw, lh, rw, dpr, wipe, True)
            good, naive, offset, tol = got.split("/")
            # The old conversion is only WRONG where its error exceeds the grab
            # tolerance; within it, hitting was never the bug. So: the fixed
            # conversion must always hit, and the old one must miss exactly
            # where it was off by more than the tolerance.
            drifted = float(offset) > float(tol)
            ok = good == "hit" and (naive == "miss" if drifted else naive == "hit")
            check(f"divider grab zone follows the divider  [{name}, wipe {wipe}]",
                  ok, f"fixed={good} legacy={naive} "
                      f"legacy offset {offset}px vs tolerance {tol}px")


# ---------------------------------------------------------------------------
# 5. The extension parses, registers, and keeps its hands off other nodes
# ---------------------------------------------------------------------------

def test_extension_loads():
    try:
        import quickjs           # noqa: F401
    except ImportError:
        print("SKIP  extension load — pip install quickjs to run it")
        return

    ctx = _js_context()
    ok = ctx.eval("!!globalThis.__registered && "
                  "typeof globalThis.__registered.beforeRegisterNodeDef === 'function'")
    check("extension evaluates and registers", bool(ok))

    ctx.eval("""
        globalThis.__ran = false;
        globalThis.__touched = false;
        (async function () {
            const ext = globalThis.__registered;
            const other = { prototype: {} };
            await ext.beforeRegisterNodeDef(other, { name: "SomeOtherNode" }, app);
            globalThis.__touched = !!other.prototype.onNodeCreated;
            const mine = { prototype: {} };
            await ext.beforeRegisterNodeDef(mine, { name: "Bat_Rescale" }, app);
            globalThis.__ran = typeof mine.prototype.onNodeCreated === "function"
                            && typeof mine.prototype.onExecuted === "function";
        })();
    """)
    # beforeRegisterNodeDef is async, so the promise above is still a pending
    # job at this point — quickjs does not pump its own microtask queue.
    for _ in range(64):
        try:
            if not ctx.execute_pending_job():
                break
        except AttributeError:
            break

    check("hooks onNodeCreated / onExecuted for Bat_Rescale",
          bool(ctx.eval("globalThis.__ran")))
    check("leaves other node types alone",
          not bool(ctx.eval("globalThis.__touched")))


if __name__ == "__main__":
    test_region_exact()
    test_blend_parity()
    test_plan_size_parity()
    test_divider_hit_test()
    test_extension_loads()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")
