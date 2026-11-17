"""
Checks for 🦇 Video Grid Split + 🦇 Video Grid Merge.

* Split's output is unchanged by moving its tile maths into
  ``grid_tile_rects`` (compared against a verbatim copy of the old loop).
* split -> merge reproduces the input, with and without overlap, at sizes that
  do not divide evenly, for full and partial tile ranges.
* Overlaps are feathered, not hard-cut; uniformly scaled tiles land on a
  scaled canvas.

Run from the pack root:  python tests/verify_grid_merge.py
"""

import importlib
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load():
    pkg = types.ModuleType("batpkg")
    pkg.__path__ = [PACK]
    sys.modules["batpkg"] = pkg
    return (importlib.import_module("batpkg.bat_video_grid_split"),
            importlib.import_module("batpkg.bat_grid_merge"))


def old_split(images, columns, rows, overlap, start_index, end_index):
    """The pre-refactor VideoGridSplit.split_video loop, verbatim."""
    b, h, w, c = images.shape
    base_tile_h = h / rows
    base_tile_w = w / columns
    tile_h = min(h, math.ceil(base_tile_h * (1 + overlap)))
    tile_w = min(w, math.ceil(base_tile_w * (1 + overlap)))
    output_list = []
    total_tiles = rows * columns
    if end_index == -1 or end_index > total_tiles:
        limit_index = total_tiles
    else:
        limit_index = end_index
    current_tile_index = 0
    for r in range(rows):
        for col in range(columns):
            if start_index <= current_tile_index < limit_index:
                center_y = (r + 0.5) * base_tile_h
                center_x = (col + 0.5) * base_tile_w
                y1 = int(center_y - (tile_h / 2))
                x1 = int(center_x - (tile_w / 2))
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                y2 = y1 + tile_h
                x2 = x1 + tile_w
                if y2 > h:
                    y2 = h
                    y1 = max(0, h - tile_h)
                if x2 > w:
                    x2 = w
                    x1 = max(0, w - tile_w)
                output_list.append(images[:, y1:y2, x1:x2, :])
            current_tile_index += 1
            if current_tile_index >= limit_index:
                break
        if current_tile_index >= limit_index:
            break
    return output_list


SIZES = [(37, 53), (64, 64), (45, 101), (1081, 33), (9, 250)]
GRIDS = [(1, 1), (2, 3), (3, 5), (5, 3), (4, 4)]
OVERLAPS = [0.0, 0.05, 0.25, 0.5]


def test_split_unchanged(gs):
    import torch
    print("\nsplit output unchanged")
    split = gs.VideoGridSplit()
    bad = n = 0
    g = torch.Generator().manual_seed(0)
    for h, w in SIZES:
        img = torch.rand((2, h, w, 3), generator=g)
        for cols, rows in GRIDS:
            if rows > h or cols > w:
                continue
            for ov in OVERLAPS:
                for s, e in ((0, -1), (2, 7), (5, 3), (0, 999), (40, -1)):
                    n += 1
                    new = split.split_video(img, cols, rows, ov, s, e)[0]
                    old = old_split(img, cols, rows, ov, s, e)
                    same = len(new) == len(old) and all(
                        a.shape == b.shape and torch.equal(a, b) for a, b in zip(new, old))
                    bad += not same
    check(f"{n} split configurations match the old loop", bad == 0, f"{bad} differ")


def test_round_trip(gs, gm):
    import torch
    print("\nsplit -> merge round trip")
    split, merge = gs.VideoGridSplit(), gm.BatGridMerge()
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    n = 0
    for h, w in SIZES:
        img = torch.rand((2, h, w, 3), generator=g)
        for cols, rows in GRIDS:
            if rows > h or cols > w:
                continue
            for ov in OVERLAPS:
                tiles = split.split_video(img, cols, rows, ov, 0, -1)[0]
                out = merge.merge(tiles, [cols], [rows], [ov], [0], [-1], [0], [0],
                                  reference=[img])[0]
                n += 1
                if out.shape != img.shape:
                    worst = float("inf")
                    continue
                worst = max(worst, float((out - img).abs().max()))
    check(f"{n} grids reassemble to the input", worst < 1e-6, f"max error {worst}")

    img = torch.rand((1, 45, 101, 3), generator=g)
    tiles = split.split_video(img, 5, 3, 0.25, 0, -1)[0]
    out = merge.merge(tiles, [5], [3], [0.25], [0], [-1], [101], [45])[0]
    check("width/height widgets work without a reference",
          out.shape == img.shape and float((out - img).abs().max()) < 1e-6)

    tiles = split.split_video(img, 5, 3, 0.25, 3, 9)[0]
    out = merge.merge(tiles, [5], [3], [0.25], [3], [9], [101], [45], reference=[img])[0]
    covered = out.abs().sum(dim=-1) > 0
    check("a partial tile range reassembles its part exactly",
          float((out - img)[covered[..., None].expand_as(out)].abs().max()) < 1e-6
          and not bool(covered.all()))


def test_blend_and_scale(gs, gm):
    import torch
    import torch.nn.functional as F
    print("\nfeathering and scale")
    split, merge = gs.VideoGridSplit(), gm.BatGridMerge()
    img = torch.zeros((1, 40, 80, 3))
    tiles = split.split_video(img, 2, 1, 0.5, 0, -1)[0]
    # Left tile processed to 0, right tile to 1: the shared band must ramp.
    tiles = [tiles[0], tiles[1] + 1.0]
    out = merge.merge(tiles, [2], [1], [0.5], [0], [-1], [80], [40])[0][0, 0, :, 0]
    band = out[20:60]
    check("the overlap ramps monotonically instead of a hard seam",
          bool((band[1:] >= band[:-1] - 1e-7).all()) and float(band.min()) < 0.1
          and float(band.max()) > 0.9 and len(set(band.tolist())) > 10)
    check("outside the overlap each tile is untouched",
          float(out[:10].abs().max()) == 0.0 and float((out[-10:] - 1).abs().max()) == 0.0)

    img = torch.rand((1, 30, 48, 3))
    tiles = split.split_video(img, 4, 3, 0.25, 0, -1)[0]
    up = [F.interpolate(t.permute(0, 3, 1, 2), scale_factor=2, mode="nearest").permute(0, 2, 3, 1)
          for t in tiles]
    out = merge.merge(up, [4], [3], [0.25], [0], [-1], [0], [0], reference=[img])[0]
    want = F.interpolate(img.permute(0, 3, 1, 2), scale_factor=2, mode="nearest").permute(0, 2, 3, 1)
    check("2x-processed tiles merge onto a 2x canvas",
          out.shape == want.shape and float((out - want).abs().max()) < 1e-6,
          f"{tuple(out.shape)}")

    for label, call in (
        ("missing frame size is refused",
         lambda: merge.merge(tiles, [4], [3], [0.25], [0], [-1], [0], [0])),
        ("a tile count that does not match the grid is refused",
         lambda: merge.merge(tiles[:-1], [4], [3], [0.25], [0], [-1], [48], [30])),
    ):
        try:
            call()
            check(label, False, "no error")
        except ValueError:
            check(label, True)


def main():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip: no torch")
        return 0
    gs, gm = load()
    test_split_unchanged(gs)
    test_round_trip(gs, gm)
    test_blend_and_scale(gs, gm)
    print()
    if _failures:
        print(f"{len(_failures)} FAILED:")
        for f in _failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
