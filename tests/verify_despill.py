"""Checks for 🦇 Despill (bat_despill.py + web/bat_despill.js). CPU only.

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_despill.py

Formulas are checked against plain numpy written out from their definitions;
CorridorKey 2 parity (same pixels as its key node) is checked from the other
side, in ComfyUI-BAT-CorridorKey2/tests/test_bat_despill_parity.py.
"""
import io
import json
import os
import re
import sys
import types

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
pkg = types.ModuleType("batpack")
pkg.__path__ = [PACK]
sys.modules["batpack"] = pkg
from batpack import bat_despill as bd  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok or not detail else f"\n         {detail}"))
    if not ok:
        FAILS.append(label)


rng = np.random.default_rng(5)

print("colour + despill formulas")
x = rng.uniform(-0.5, 4.0, 5000)
ref_dec = np.sign(x) * np.where(np.abs(x) <= 0.04045, np.abs(x) / 12.92, ((np.abs(x) + 0.055) / 1.055) ** 2.4)
ref_enc = np.sign(x) * np.where(np.abs(x) <= 0.0031308, np.abs(x) * 12.92, 1.055 * np.abs(x) ** (1 / 2.4) - 0.055)
check("sRGB decode == IEC 61966-2-1 (sign-mirrored)",
      np.abs(bd.srgb_to_linear(torch.from_numpy(x)).numpy() - ref_dec).max() < 1e-9)
check("sRGB encode == IEC 61966-2-1 (sign-mirrored)",
      np.abs(bd.linear_to_srgb(torch.from_numpy(x)).numpy() - ref_enc).max() < 1e-9)
check("encode(decode(x)) round-trips", np.abs(bd.linear_to_srgb(bd.srgb_to_linear(torch.from_numpy(x))).numpy() - x).max() < 1e-9)
px = rng.uniform(0, 1, (50, 3))
for screen, (a, b, c) in (("green", (1, 0, 2)), ("blue", (2, 0, 1))):
    for s in (0.0, 0.4, 1.0):
        spill = np.maximum(px[:, a] - (px[:, b] + px[:, c]) / 2, 0)
        full = px.copy(); full[:, a] -= spill; full[:, b] += spill / 2; full[:, c] += spill / 2
        want = px * (1 - s) + full * s
        got = bd.despill(torch.from_numpy(px), s, screen).numpy()
        check(f"{screen} despill {s} == average despill", np.abs(got - want).max() < 1e-12)
ck = bd.checker_srgb(130, 200).numpy()
yy, xx = np.mgrid[0:130, 0:200]
check("checker: 64 px squares, 0.75 / 0.5",
      np.array_equal(ck, np.where(((yy // 64) + (xx // 64)) % 2 == 0, 0.75, 0.5).astype(np.float32)))

print("node outputs")
T, H, W = 3, 150, 230
fg = torch.from_numpy(rng.uniform(0, 1, (T, H, W, 3)).astype(np.float32))
fg[..., 1] = (fg[..., 1] + 0.2).clamp(0, 1)
alpha = torch.from_numpy(rng.uniform(0, 1, (T, H, W)).astype(np.float32))
node = bd.BatDespill()
out = node.run(fg, alpha, "sRGB", "green", 0.5, "checker")
clean = bd.despill(fg[1], 0.5, "green")
check("foreground == despilled", torch.equal(out["result"][0][1], clean))
check("premultiplied == fg x alpha", torch.allclose(out["result"][1][1], clean * alpha[1][..., None]))
a = alpha[1].numpy()[..., None]
comp_ref = bd.linear_to_srgb(torch.from_numpy(bd.srgb_to_linear(clean).numpy() * a
                                              + bd.srgb_to_linear(bd.checker_srgb(H, W)).numpy()[..., None] * (1 - a)).clamp(min=0)).clamp(0, 1)
check("comp == fg over checker in linear light", torch.allclose(out["result"][2][1], comp_ref.float(), atol=1e-6))

print("viewer regions == crops of the full render")
for view in bd.VIEWS:
    full = bd.render_view(fg[0], alpha[0], fg[0], view, 0.5, "green", "checker", False)
    y0, x0, h, w = 37, 71, 90, 120
    part = bd.render_view(fg[0][y0:y0 + h, x0:x0 + w], alpha[0][y0:y0 + h, x0:x0 + w],
                          fg[0][y0:y0 + h, x0:x0 + w], view, 0.5, "green", "checker", False, y0=y0, x0=x0)
    d = (full[y0:y0 + h, x0:x0 + w] - part).abs().max().item()
    check(f"{view}", d < 1e-6, f"max |d| {d:.1e}")    # comp: pow on other shapes, float noise only

print("linear")
lin = fg[:1] * 3.0
res = node.run(lin, alpha[:1], "linear", "green", 0.0, "black")["result"]
check("values above 1 survive", res[0].max().item() > 1.5)
check("comp over black == fg x alpha", torch.allclose(res[2][0], lin[0] * alpha[0][..., None], atol=1e-6))

print("viewer route")
token = node.run(fg, alpha, "sRGB", "green", 0.5, "checker")["ui"]["despill_token"][0]
u8, err = bd.view_rgb({"token": token, "frame": 2, "roi": [10, 20, 200, 100], "out_w": 100, "out_h": 50})
check("raw RGB at the requested size", u8 is not None and u8.shape == (50, 100, 3) and u8.dtype == np.uint8, err)
png, _ = bd.view_png({"token": token, "frame": 0, "roi": [0, 0, 8, 8], "out_w": 32, "out_h": 32})
check("zoomed in -> real pixels (nearest)", Image.open(io.BytesIO(png)).size == (32, 32))
t_fg, t_a = fg.clone(), alpha.clone()
tok2 = node.run(t_fg, t_a, "sRGB", "green", 0.5, "checker")["ui"]["despill_token"][0]
del t_fg, t_a
check("dropped frames -> clear message", "dropped" in (bd.view_rgb({"token": tok2})[1] or ""))
check("unknown token -> clear message", "unknown" in (bd.view_rgb({"token": "nope"})[1] or ""))
try:
    node.run(fg, alpha[:2], "sRGB", "green", 0.5, "checker")
    check("frame-count mismatch refused", False)
except ValueError:
    check("frame-count mismatch refused", True)

print("viewer maths (web/bat_despill.js under quickjs)")
import quickjs  # noqa: E402
SRC = open(os.path.join(PACK, "web", "bat_despill.js"), encoding="utf-8").read()


def grab(name):
    start = SRC.index(f"function {name}(")
    depth, i = 0, SRC.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(SRC[i], 0)
        if depth == 0:
            return SRC[start:i + 1]
        i += 1


MAXZ = re.search(r"const MAX_ZOOM = (\d+);", SRC).group(1)
ctx = quickjs.Context()
ctx.eval(f"const MAX_ZOOM = {MAXZ};\n" + "\n".join(grab(n) for n in ("fitView", "zoomAt", "visibleRoi", "roiToScreen")))
J = lambda e: json.loads(ctx.eval(f"JSON.stringify({e})"))
SW, SH, FW, FH = 600, 400, 1920, 1318
v = J(f"fitView({FW},{FH},{SW},{SH})")
check("fit shows the whole frame", J(f"visibleRoi({json.dumps(v)},{FW},{FH},{SW},{SH})") == [0, 0, FW, FH])
for px_, py_, f in ((100, 50, 1.25), (580, 390, 4.0), (300, 200, 0.8)):
    z = J(f"zoomAt({json.dumps(v)},{f},{px_},{py_},{SW},{SH},0.01)")
    before = (v["cx"] + (px_ - SW / 2) / v["zoom"], v["cy"] + (py_ - SH / 2) / v["zoom"])
    after = (z["cx"] + (px_ - SW / 2) / z["zoom"], z["cy"] + (py_ - SH / 2) / z["zoom"])
    check(f"zoom x{f} keeps the pixel under the cursor", max(abs(p - q) for p, q in zip(before, after)) < 1e-9)
check("zoom clamps at MAX_ZOOM", J(f"zoomAt({json.dumps(v)},1e6,0,0,{SW},{SH},0.01)")["zoom"] == int(MAXZ))
z = {"zoom": 2.0, "cx": 500.25, "cy": 300.75}
roi = J(f"visibleRoi({json.dumps(z)},{FW},{FH},{SW},{SH})")
scr = J(f"roiToScreen({json.dumps(roi)},{json.dumps(z)},{SW},{SH})")
check("zoomed region covers the stage", scr[0] <= 0 and scr[1] <= 0 and scr[0] + scr[2] >= SW and scr[1] + scr[3] >= SH, scr)
body = grab("rgbToBitmap").replace("async function", "function").replace(
    "return createImageBitmap(new ImageData(rgba, w, h));", "return Array.from(rgba);")
ctx.eval(body)
check("raw RGB -> opaque RGBA", J("rgbToBitmap(new Uint8Array([1,2,3,250,251,252]).buffer,2,1)") == [1, 2, 3, 255, 250, 251, 252, 255])

print()
print("All checks passed." if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
