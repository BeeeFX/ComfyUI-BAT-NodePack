#!/usr/bin/env python
"""
Animated Crop: the round-2 fixes of 2026-09-23.

1. **Batched crop == the frame-at-a-time crop, bit for bit.** The node used
   to allocate a full-canvas mask, slice, resize and grid_sample one frame at
   a time and torch.cat three lists at the end (~25x Bat_Crop on a static
   rect). It now resolves every rect first and batches. `_reference_crop`
   below is the old loop verbatim, run against the new node over static /
   panned / zoomed / rotated / mixed clips, every fill, with and without a
   mask, constrained or not, and with chunks forced down to a few frames.
   Also times both on a static rect.
2. **Preview sidecar.** The strip leaves the `ui` (bat_ui_ref) and resolves
   back to the same payload, stride included.
3. **First run matches the editor.** A new node's empty state renders the
   editor's centred half-size seed; a state saved without the `seed` marker
   keeps the old 512x512-at-origin rect. The editor keeps the clip size in
   node.properties, not in `state` (source pins).
4. **Keyframe easing.** A key's `ease` shapes the segment after it; keys
   without one interpolate exactly as before; the editor's preview
   interpolation (quickjs) matches Python's at every frame.

    python tests/verify_animated_crop.py
"""

import importlib
import json
import math
import os
import re
import sys
import time
import types

import torch
import torch.nn.functional as F

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


def load_pack_modules():
    """Import the nodes without the pack __init__ (which needs a live server)."""
    pkg = types.ModuleType("batpkg")
    pkg.__path__ = [PACK]
    sys.modules["batpkg"] = pkg
    return tuple(importlib.import_module(f"batpkg.{m}") for m in (
        "bat_animated_crop", "bat_crop", "bat_ui_ref"))


# ─── the pre-batching implementation, for the equality check ─────────────────

def _old_resize_nhwc(t, h, w):
    if t.shape[1] == h and t.shape[2] == w:
        return t
    out = F.interpolate(t.permute(0, 3, 1, 2), size=(h, w), mode="bicubic",
                        align_corners=False)
    return out.permute(0, 2, 3, 1).clamp(
        t.amin().item(), t.amax().item()).contiguous()


def _reference_crop(bac, bc, image, state, snap_to, constrain_to_canvas=True,
                    outside_fill="black", mask=None):
    """Bat_AnimatedCrop.crop's image/mask/rect_mask/frames as of 2026-09-22:
    one frame at a time. The rect resolution is the module's own (so eased
    keys exercise the same pipeline); everything after it is the old code."""
    n, H, W, c = image.shape
    device, dtype = image.device, image.dtype
    doc = json.loads(state) if state else {}
    kfs = doc.get("keyframes") or {}
    empty = bac._seed_rect(doc, W, H)
    in_mask = None
    if mask is not None:
        in_mask = bc._broadcast_mask_to_n(mask.to(torch.float32), n)
        if in_mask.shape[-2:] != (H, W):
            in_mask = F.interpolate(in_mask.unsqueeze(1), size=(H, W), mode="bilinear",
                                    align_corners=False).squeeze(1).clamp(0, 1)
    first_rect = bac._resolve_rect_at_frame(kfs, 0, empty)
    ref_w = max(1, int(round(first_rect["w"])))
    ref_h = max(1, int(round(first_rect["h"])))
    out_w = bc._snap(ref_w, snap_to)
    out_h = bc._snap(ref_h, snap_to)
    scale_w, scale_h = out_w / ref_w, out_h / ref_h
    out_frames, out_masks, rect_masks, rects = [], [], [], []
    for f in range(n):
        rect = bac._resolve_rect_at_frame(kfs, f, empty)
        w = max(1, int(round(float(rect["w"]) * scale_w)))
        h = max(1, int(round(float(rect["h"]) * scale_h)))
        x = float(rect["x"]) + (float(rect["w"]) - w) / 2.0
        y = float(rect["y"]) + (float(rect["h"]) - h) / 2.0
        angle = float(rect["angle"])
        if constrain_to_canvas:
            if abs(angle) < 1e-3:
                x = min(max(0.0, x), float(max(0, W - w)))
                y = min(max(0.0, y), float(max(0, H - h)))
            else:
                cx_, cy_ = bc.constrain_rotated_rect(int(round(x)), int(round(y)),
                                                     w, h, angle, W, H)
                x, y = float(cx_), float(cy_)
        rects.append({"x": x, "y": y, "w": w, "h": h, "angle": angle})
        frame_img = image[f:f + 1]
        frame_mask_in = in_mask[f:f + 1] if in_mask is not None else None
        if abs(angle) < 1e-3:
            ix, iy = int(round(x)), int(round(y))
            src_x0, src_y0 = max(0, ix), max(0, iy)
            src_x1, src_y1 = min(W, ix + w), min(H, iy + h)
            dst_x0, dst_y0 = src_x0 - ix, src_y0 - iy
            cropped = bc._extract_axis_aligned(frame_img, ix, iy, w, h, outside_fill)
            rect_m = torch.zeros((1, H, W), dtype=dtype, device=device)
            if src_x1 > src_x0 and src_y1 > src_y0:
                rect_m[:, src_y0:src_y1, src_x0:src_x1] = 1.0
            if frame_mask_in is not None:
                cropped_m = torch.zeros((1, h, w), device=device, dtype=dtype)
                if src_x1 > src_x0 and src_y1 > src_y0:
                    cropped_m[:, dst_y0:dst_y0 + (src_y1 - src_y0),
                                 dst_x0:dst_x0 + (src_x1 - src_x0)] = \
                        frame_mask_in[:, src_y0:src_y1, src_x0:src_x1]
            else:
                cropped_m = torch.ones((1, h, w), device=device, dtype=dtype)
        else:
            cropped = bc._rotated_crop(frame_img, x, y, w, h, angle, outside_fill)
            rect_m = bc._rotated_rect_mask(H, W, x + (w - 1) / 2.0, y + (h - 1) / 2.0,
                                           w, h, angle, 1, device, dtype)
            if frame_mask_in is not None:
                cropped_m = bc._rotated_crop_mask(frame_mask_in, x, y, w, h, angle,
                                                  outside_fill)
            else:
                cropped_m = torch.ones((1, h, w), device=device, dtype=dtype)
        out_frames.append(_old_resize_nhwc(cropped, out_h, out_w))
        out_masks.append(bac._resize_nhw(cropped_m, out_h, out_w))
        rect_masks.append(rect_m)
    return (torch.cat(out_frames).contiguous(), torch.cat(out_masks).contiguous(),
            torch.cat(rect_masks).contiguous(), rects)


def ramp(H, W, n):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    img = torch.stack([xs / W, ys / H, (xs + ys) / (W + H)], -1).unsqueeze(0)
    return img.repeat(n, 1, 1, 1) + torch.linspace(0, 0.2, n).view(n, 1, 1, 1)


def kf(x, y, w, h, angle=0.0, ease=None):
    k = {"x": x, "y": y, "w": w, "h": h, "angle": angle}
    if ease:
        k["ease"] = ease
    return k


# ─── 1. batched == frame-at-a-time ───────────────────────────────────────────

def _same(label, bac, bc, image, st, snap=8, con=True, fill="black", mask=None):
    state = json.dumps(st)
    got = bac.BatAnimatedCrop().crop(image, state, snap, False, "free", con, fill,
                                     mask=mask)["result"]
    o, info, m, rm = got
    ro, rmask, rrm, rrects = _reference_crop(bac, bc, image, state, snap, con, fill, mask)
    ok = (o.dtype == ro.dtype and m.dtype == rmask.dtype and rm.dtype == rrm.dtype
          and torch.equal(o, ro) and torch.equal(m, rmask) and torch.equal(rm, rrm)
          and info["frames"] == rrects and o.is_contiguous() and m.is_contiguous())
    detail = ""
    if not ok:
        def d(a, b):
            return (a.float() - b.float()).abs().max().item() if a.shape == b.shape \
                else f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        detail = (f"image {d(o, ro)} mask {d(m, rmask)} rect {d(rm, rrm)} "
                  f"dtypes {o.dtype}/{ro.dtype} {m.dtype}/{rmask.dtype} "
                  f"frames {'same' if info['frames'] == rrects else 'differ'}")
    check(label, ok, detail)


def test_equality(bac, bc):
    print("\nbatched crop == frame-at-a-time crop (bit-exact)")
    H, W, n = 48, 64, 9
    img = ramp(H, W, n)
    torch.manual_seed(3)
    noise = torch.rand(n, H, W, 3) * 1.5 - 0.2     # HDR-ish, exercises the clamp
    mask_full = torch.rand(n, H, W)
    mask_one = torch.rand(1, H, W)
    mask_small = torch.rand(n, 24, 32)
    cases = {
        "static axis": {"keyframes": {"0": kf(10, 6, 30, 20)}},
        "static axis, off-canvas": {"keyframes": {"0": kf(-8, 30, 30, 24)}},
        "axis pan": {"keyframes": {"0": kf(2, 3, 32, 24), "8": kf(30, 20, 32, 24)}},
        "axis zoom": {"keyframes": {"0": kf(10, 6, 24, 16), "8": kf(4, 2, 50, 40)}},
        "axis zoom out and back (split size groups)": {"keyframes": {
            "0": kf(10, 6, 24, 16), "4": kf(4, 2, 44, 36), "8": kf(10, 6, 24, 16)}},
        "static rotated": {"keyframes": {"0": kf(12, 8, 30, 20, 17.0)}},
        "rotated animated": {"keyframes": {"0": kf(12, 8, 30, 20, 0.0),
                                           "8": kf(20, 14, 34, 22, 40.0)}},
        "rotated, identical runs": {"keyframes": {
            "0": kf(12, 8, 30, 20, 12.0), "3": kf(12, 8, 30, 20, 12.0),
            "5": kf(14, 8, 30, 20, 12.0, ease="hold"), "8": kf(18, 9, 30, 20, 12.0)}},
        "mixed axis/rotated, eased": {"keyframes": {
            "0": kf(5, 5, 28, 20, 0.0, ease="ease_in_out"), "4": kf(12, 9, 28, 20, 0.0),
            "5": kf(12, 9, 28, 20, 25.0, ease="ease_out"), "8": kf(20, 12, 36, 26, -30.0)}},
        "empty (seed)": {"keyframes": {}, "seed": "centre"},
    }
    for name, st in cases.items():
        _same(name, bac, bc, img, st)
    for fill in ("black", "gray", "edge", "reflect"):
        for con in (True, False):
            _same(f"mixed, fill={fill}, constrain={con}, mask", bac, bc, noise,
                  cases["mixed axis/rotated, eased"], 8, con, fill, mask_full)
            _same(f"off-canvas pan, fill={fill}, constrain={con}", bac, bc, noise,
                  {"keyframes": {"0": kf(-12, -6, 30, 24), "8": kf(44, 30, 30, 24, 0.0)}},
                  8, con, fill)
    _same("rotated animated, 1-frame mask batch", bac, bc, noise,
          cases["rotated animated"], mask=mask_one)
    _same("axis zoom, mask at another size", bac, bc, noise, cases["axis zoom"],
          mask=mask_small)
    _same("snap_to=1", bac, bc, noise, cases["axis zoom"], snap=1)
    _same("float64 plate, mixed", bac, bc, noise.double(), cases["mixed axis/rotated, eased"],
          mask=mask_full)

    # Force tiny chunks: split runs, gathered (non-contiguous) frame sets.
    saved = bac._CHUNK_BYTES
    try:
        bac._CHUNK_BYTES = 1
        _same("1-frame chunks, zoom out and back", bac, bc, noise,
              cases["axis zoom out and back (split size groups)"], mask=mask_full)
        _same("1-frame chunks, mixed, gray", bac, bc, noise,
              cases["mixed axis/rotated, eased"], fill="gray", mask=mask_full)
        per = 4 * (H * W * 4 + 3 * 30 * 20 * 5 + 24 * 32 * 4)
        bac._CHUNK_BYTES = per * 3
        rot_back = {"keyframes": {"0": kf(12, 8, 30, 20, 10.0), "4": kf(8, 6, 40, 28, 30.0),
                                  "8": kf(12, 8, 30, 20, 10.0)}}
        _same("3-frame chunks, rotated zoom out and back", bac, bc, noise, rot_back,
              fill="edge", mask=mask_full)
    finally:
        bac._CHUNK_BYTES = saved


def _best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_timing(bac, bc):
    print("\ntiming (static rect, 60 frames at 270x480, preview strip excluded, best of 3)")
    torch.manual_seed(0)
    img = torch.rand(60, 270, 480, 3)
    saved = bac._preview_strip
    bac._preview_strip = lambda image: ([], 1)
    try:
        for label, key in (("axis", kf(120, 67, 240, 135)),
                           ("rotated", kf(120, 67, 240, 135, 10.0))):
            st = json.dumps({"keyframes": {"0": key}})
            new = _best_of(lambda: bac.BatAnimatedCrop().crop(
                img, st, 8, False, "free", True, "black"))
            old = _best_of(lambda: _reference_crop(bac, bc, img, st, 8))
            print(f"        {label:8s} batched {new * 1000:7.1f} ms   frame-at-a-time "
                  f"{old * 1000:7.1f} ms   ({old / max(new, 1e-9):.1f}x)")
            # Loose on purpose: a shared CI box is noisy, and equality above is
            # the real contract. This only catches the batching regressing.
            check(f"{label}: batched is not slower", new <= old * 1.25,
                  f"{new:.3f}s vs {old:.3f}s")
    finally:
        bac._preview_strip = saved


# ─── 2. preview sidecar ──────────────────────────────────────────────────────

def test_ui(bac, bur):
    print("\npreview sidecar")
    tiny = torch.rand(479, 4, 4, 3)
    ui = bac.BatAnimatedCrop().crop(tiny, "{}", 1, False, "free", True, "black")["ui"]
    check("ui is a token, not the strip", set(ui) == {"bat_ui"}, str(list(ui)))
    full = bur.load_ui(ui)
    check("token resolves", isinstance(full, dict) and "frames" in full)
    if isinstance(full, dict) and "frames" in full:
        check("479 frames -> <=240 thumbnails", len(full["frames"]) <= 240,
              f"{len(full['frames'])} (stride {full['stride'][0]})")
        check("payload keys the editor reads",
              full["w"] == [4] and full["h"] == [4] and full["frame_count"] == [479]
              and full["stride"] == [2], str({k: full[k] for k in ("w", "h", "stride", "frame_count")}))


# ─── 3. first run matches the editor ─────────────────────────────────────────

def test_seed(bac):
    print("\nempty state: seed rect")
    img = torch.rand(2, 54, 97, 3)          # odd width: Math.round vs banker's
    o, info, _, _ = bac.BatAnimatedCrop().crop(
        img, json.dumps({"keyframes": {}, "seed": "centre"}), 1, False, "free", True,
        "black")["result"]
    f0 = info["frames"][0]
    # Editor seed: x=Math.round(W*.25), w=Math.round(W*.5) → 24, 49 for W=97.
    check("marked empty state renders the editor's centred seed",
          (f0["x"], f0["y"], f0["w"], f0["h"]) == (24, 14, 49, 27), str(f0))
    check("output is the seed size", tuple(o.shape[1:3]) == (27, 49), str(tuple(o.shape)))
    big = torch.rand(1, 600, 700, 3)
    _, info, _, _ = bac.BatAnimatedCrop().crop(
        big, '{"keyframes":{}}', 8, False, "free", True, "black")["result"]
    f0 = info["frames"][0]
    check("unmarked (saved) empty state keeps 512x512 at the origin",
          (f0["x"], f0["y"], f0["w"], f0["h"]) == (0, 0, 512, 512), str(f0))
    default = bac.BatAnimatedCrop.INPUT_TYPES()["required"]["state"][1]["default"]
    check("new nodes default to the marked state",
          json.loads(default) == {"keyframes": {}, "seed": "centre"}, default)
    legacy = {"keyframes": {"0": kf(1, 2, 30, 20)}, "imgW": 64, "imgH": 48, "frameCount": 3}
    a = bac.BatAnimatedCrop().crop(torch.rand(3, 48, 64, 3), json.dumps(legacy), 8, False,
                                   "free", True, "black")["result"][2]
    check("legacy imgW/imgH/frameCount in state are still accepted", a.shape[0] == 3)


# ─── 4. easing ───────────────────────────────────────────────────────────────

def _old_resolve(kfs, frame):
    """The linear-only resolver as it was."""
    keys = sorted(int(k) for k in kfs.keys())
    if frame <= keys[0]:
        return dict(kfs[str(keys[0])])
    if frame >= keys[-1]:
        return dict(kfs[str(keys[-1])])
    prev, nxt = keys[0], keys[-1]
    for k in keys:
        if k <= frame:
            prev = k
        if k >= frame and k != prev:
            nxt = k
            break
    a, b = kfs[str(prev)], kfs[str(nxt)]
    t = (frame - prev) / (nxt - prev)
    return {f: a[f] + (b[f] - a[f]) * t for f in ("x", "y", "w", "h", "angle")}


EASED = {
    "0": kf(0, 0, 100, 60, 0.0, ease="ease_in"),
    "10": kf(40, 20, 60, 40, 15.0, ease="hold"),
    "17": kf(80, 10, 120, 90, -10.0, ease="ease_in_out"),
    "30": kf(10.5, 33.25, 90, 70, 5.0, ease="ease_out"),
    "41": kf(12, 30, 91, 71, 5.5),
    "50": kf(3, 3, 20, 20, 0.0, ease="bogus"),
    "57": kf(9, 9, 30, 25, 3.0),
}
FIELDS = ("x", "y", "w", "h", "angle")


def test_easing(bac):
    print("\nkeyframe easing")
    plain = {k: {f: v[f] for f in FIELDS} for k, v in EASED.items()}
    same = all(bac._resolve_rect_at_frame(plain, f) == _old_resolve(plain, f)
               for f in range(-2, 62))
    check("keys without `ease` interpolate exactly as before", same)
    r = bac._resolve_rect_at_frame(EASED, 5)
    check("ease_in: halfway in time is a quarter of the way", abs(r["x"] - 10.0) < 1e-12, r)
    r = bac._resolve_rect_at_frame(EASED, 16)
    check("hold: stays on the earlier key until the next", r["x"] == 40 and r["angle"] == 15.0, r)
    check("hold: next key lands on its frame", bac._resolve_rect_at_frame(EASED, 17)["x"] == 80)
    r = bac._resolve_rect_at_frame(EASED, 55)
    check("unknown ease is linear", abs(r["x"] - (3 + 6 * 5 / 7)) < 1e-12, r)

    try:
        import quickjs
    except ImportError:
        print("  skip  JS parity (pip install quickjs)")
        return
    from _harness import strip_modules
    src = open(os.path.join(PACK, "web", "bat_animated_crop.js"), encoding="utf-8").read()
    fns = [re.search(rf"^function {name}\(.*?^\}}", src, re.M | re.S)
           for name in ("seedRect", "interpolateRect")]
    check("editor has module-level seedRect / interpolateRect", all(fns))
    if not all(fns):
        return
    ctx = quickjs.Context()
    ctx.eval(strip_modules(open(os.path.join(PACK, "web", "bat_easing.js"),
                                encoding="utf-8").read()))
    for m in fns:
        ctx.eval(m.group(0))
    worst = 0.0
    for kfs in (EASED, {k: {f: v[f] for f in FIELDS} for k, v in EASED.items()}):
        js_all = json.loads(ctx.eval(
            f"JSON.stringify(Array.from({{length: 64}}, (_, i) => "
            f"interpolateRect({json.dumps(kfs)}, i - 2, null)))"))
        for i, js in enumerate(js_all):
            py = bac._resolve_rect_at_frame(kfs, i - 2)
            worst = max(worst, max(abs(py[f] - js[f]) for f in FIELDS))
    check("JS preview == Python render at every frame (eased and plain)", worst < 1e-9, worst)
    seed_js = json.loads(ctx.eval(
        'JSON.stringify(interpolateRect({}, 0, {imgW: 97, imgH: 54, seed: "centre"}))'))
    seed_py = bac._seed_rect({"seed": "centre"}, 97, 54)
    check("JS empty-map seed == Python seed",
          all(seed_js[f] == seed_py[f] for f in FIELDS), (seed_js, seed_py))
    legacy_js = json.loads(ctx.eval('JSON.stringify(interpolateRect({}, 0, {imgW: 97, imgH: 54}))'))
    check("JS unmarked empty map is the legacy 512 rect",
          (legacy_js["x"], legacy_js["w"]) == (0, 512), legacy_js)
    key_copy = json.loads(ctx.eval(
        'JSON.stringify(interpolateRect({"3": {x:1,y:2,w:3,h:4,angle:0,ease:"hold"}}, 9, null))'))
    check("a held copy of a key doesn't carry its ease", "ease" not in key_copy, key_copy)


# ─── 3 + 4. the editor itself, under quickjs ─────────────────────────────────

_DOM_STUB = r"""
var console = { log() {}, warn() {}, error() {} };
var __timers = [];
function setTimeout(fn) { __timers.push(fn); return __timers.length; }
function clearTimeout() {}
function setInterval() { return 1; }
function clearInterval() {}
var window = { devicePixelRatio: 1 };
var localStorage = { getItem() { return null; }, setItem() {}, removeItem() {} };
var navigator = {};
class Path2D { rect() {} moveTo() {} lineTo() {} closePath() {} }
class Image {}
class ResizeObserver { observe() {} disconnect() {} }
function __el(tag) {
    const base = {
        tagName: String(tag || "div").toUpperCase(), style: {}, dataset: {},
        children: [], listeners: {},
        appendChild(c) { this.children.push(c); return c; },
        append(...cs) { this.children.push(...cs); },
        addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
        querySelectorAll() { return []; },
        getContext() { return new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}) }); },
        getBoundingClientRect() { return { left: 0, top: 0, width: 560, height: 400 }; },
        clientWidth: 560, clientHeight: 400, offsetWidth: 560, offsetHeight: 400,
    };
    return new Proxy(base, { get: (t, k) => (k in t ? t[k] : () => {}) });
}
var document = { createElement: __el, createTextNode: __el };
var __ext = null;
var app = { registerExtension(ext) { __ext = ext; } };
function batTrack() { return { observer() {}, dispose() {} }; }
function __makeNode(stateValue) {
    function NodeType() {}
    __ext.beforeRegisterNodeDef(NodeType, { name: "Bat_AnimatedCrop" }, app);
    const node = new NodeType();
    node.widgets = [{ name: "state", value: stateValue },
                    { name: "snap_to", value: 8 },
                    { name: "constrain_to_canvas", value: true }];
    node.properties = {};
    node.setDirtyCanvas = () => {};
    NodeType.prototype.onNodeCreated.call(node);
    return node;
}
function __select(node) {
    // The ease <select> is the only element with a "change" listener.
    const seen = new Set();
    const walk = (el) => {
        if (!el || typeof el !== "object" || seen.has(el)) return null;
        seen.add(el);
        if (el.tagName === "SELECT") return el;
        for (const c of el.children || []) { const r = walk(c); if (r) return r; }
        return null;
    };
    return walk(node.__root);
}
"""


def _editor(quickjs):
    from _harness import imported_names, strip_modules
    src = open(os.path.join(PACK, "web", "bat_animated_crop.js"), encoding="utf-8").read()
    easing = open(os.path.join(PACK, "web", "bat_easing.js"), encoding="utf-8").read()
    curated = {"app", "batTrack"} | imported_names(easing) | {
        "easeName", "applyEase", "EASES", "EASE_LABELS"}
    stubs = "".join(f"function {n}() {{}}\n"
                    for n in sorted(imported_names(src) - curated))
    ctx = quickjs.Context()
    ctx.eval(stubs + _DOM_STUB + strip_modules(easing))
    # Keep a handle on each editor's root so the test can find the <select>.
    src = src.replace("function buildEditor(node) {",
                      "function buildEditor(node) { const __r = __buildEditor(node); node.__root = __r; return __r; }\n"
                      "function __buildEditor(node) {", 1)
    ctx.eval(strip_modules(src))
    return ctx


def test_editor(quickjs):
    print("\neditor (quickjs, stubbed DOM)")
    ctx = _editor(quickjs)

    def run(js):
        out = ctx.eval(js)
        while ctx.execute_pending_job():
            pass
        return out

    # A new node: its first run writes nothing into `state`.
    run(f"var a = __makeNode({json.dumps(json.dumps({'keyframes': {}, 'seed': 'centre'}))});"
        "var a0 = a.widgets[0].value; a._batAnimCropIngest([], 64, 48, 10, 1);")
    check("new node: ingest leaves `state` untouched (no re-run)",
          run("a.widgets[0].value === a0"), run("a.widgets[0].value"))
    check("new node: clip size lands in node.properties",
          json.loads(run("JSON.stringify(a.properties.bat_clip)"))
          == {"imgW": 64, "imgH": 48, "frameCount": 10}, run("JSON.stringify(a.properties)"))
    # rectAtFrame's own arguments, taken from the live editor state.
    r = json.loads(run("var sa = a._batAnimCropState; JSON.stringify(interpolateRect("
                       "sa.doc.keyframes, 0, {imgW: sa.imgW, imgH: sa.imgH, seed: sa.doc.seed}))"))
    check("new node: editor shows the seed the backend renders",
          (r["x"], r["y"], r["w"], r["h"]) == (16, 12, 32, 24), r)
    run("a._batAnimCropIngest([], 64, 48, 10, 1);")
    check("new node: a second run with the same clip still writes nothing",
          run("a.widgets[0].value === a0"))

    # A saved-before state with no keys: seeded on ingest exactly as before
    # (its output sequence is unchanged), minus the clip fields.
    run("var b = __makeNode('{\"keyframes\":{}}'); b._batAnimCropIngest([], 64, 48, 10, 1);")
    st = json.loads(run("b.widgets[0].value"))
    check("legacy empty state: seeded on first ingest, as before",
          st.get("keyframes", {}).get("0") == {"x": 16, "y": 12, "w": 32, "h": 24, "angle": 0}, st)
    check("legacy empty state: no clip fields written into it",
          not ({"imgW", "imgH", "frameCount"} & set(st)), st)

    # A saved-before state carrying the clip fields: read on reload, left
    # alone by a run, dropped (into properties) on the next real edit.
    legacy = {"keyframes": {"0": kf(4, 4, 20, 16), "9": kf(30, 20, 20, 16)},
              "imgW": 64, "imgH": 48, "frameCount": 10}
    run(f"var c = __makeNode({json.dumps(json.dumps(legacy))}); c._batAnimCropReloadFromWidget();"
        "var c0 = c.widgets[0].value;")
    check("legacy clip fields are still read on reload",
          run("c._batAnimCropState.imgW === 64 && c._batAnimCropState.frameCount === 10"))
    run("c._batAnimCropIngest([], 64, 48, 10, 1);")
    check("legacy state: a run doesn't rewrite it", run("c.widgets[0].value === c0"))
    run("var sel = __select(c); sel.value = 'ease_in_out';"
        "sel.listeners.change.forEach(fn => fn());")
    st = json.loads(run("c.widgets[0].value"))
    check("ease picker writes the key under the playhead",
          st["keyframes"]["0"].get("ease") == "ease_in_out", st)
    check("the edit drops the legacy clip fields from the state",
          not ({"imgW", "imgH", "frameCount"} & set(st)), st)
    check("... and they are kept in node.properties",
          json.loads(run("JSON.stringify(c.properties.bat_clip)"))
          == {"imgW": 64, "imgH": 48, "frameCount": 10})
    mid = json.loads(run("JSON.stringify(interpolateRect(c._batAnimCropState.doc.keyframes, 3, null))"))
    py = _resolve_py(st["keyframes"], 3)
    check("editor preview of the eased segment == backend",
          all(abs(mid[f] - py[f]) < 1e-9 for f in FIELDS), (mid, py))
    run("sel.value = 'linear'; sel.listeners.change.forEach(fn => fn());")
    st = json.loads(run("c.widgets[0].value"))
    check("linear removes `ease` (serialises as before easing existed)",
          "ease" not in st["keyframes"]["0"], st)
    run("c._batAnimCropState.kfSelection = new Set([9]); sel.value = 'hold';"
        "sel.listeners.change.forEach(fn => fn());")
    st = json.loads(run("c.widgets[0].value"))
    check("with a timeline selection the picker writes the selected key",
          st["keyframes"]["9"].get("ease") == "hold" and "ease" not in st["keyframes"]["0"], st)


_resolve_py = None


# ─── JS source pins ──────────────────────────────────────────────────────────

def test_js_sources():
    print("\njs sources")
    s = open(os.path.join(PACK, "web", "bat_animated_crop.js"), encoding="utf-8").read()
    ingest = re.search(r"node\._batAnimCropIngest = async.*?\n    \};", s, re.S)
    body = ingest.group(0) if ingest else ""
    check("ingest no longer writes the clip size into state",
          "state.doc.imgW" not in body and "state.doc.frameCount" not in body)
    check("clip size goes to node.properties", "node.properties.bat_clip" in s)
    check("persist() strips legacy clip fields from state",
          re.search(r"const persist = .*?delete state\.doc\.imgW", s, re.S) is not None)
    check("ease picker applies to keyframe objects",
          'easeSelect' in s and re.search(r"delete k\.ease", s) is not None)
    check("keydown ignores the <select>", '"SELECT"' in s)


def main():
    global _resolve_py
    bac, bc, bur = load_pack_modules()
    _resolve_py = bac._resolve_rect_at_frame
    test_equality(bac, bc)
    test_timing(bac, bc)
    test_ui(bac, bur)
    test_seed(bac)
    test_easing(bac)
    try:
        import quickjs
    except ImportError:
        print("  skip  editor checks (pip install quickjs)")
    else:
        test_editor(quickjs)
    test_js_sources()
    if _failures:
        print(f"\n{len(_failures)} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
