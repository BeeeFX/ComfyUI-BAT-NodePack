"""
🦇 Video Grid Merge — put 🦇 Video Grid Split's tiles back together.

Split cuts a frame batch into a rows x columns grid of (optionally overlapping)
sub-clips so each can be processed on its own; this node is the other half. It
takes the tile LIST that Split emits (``INPUT_IS_LIST``), the same grid
settings, and the original frame size, and rebuilds the frame.

Geometry comes from ``bat_video_grid_split.grid_tile_rects`` — the function
Split cuts with — so the two cannot disagree about where a tile belongs.

Overlaps are blended with a linear feather: across the band two neighbours
share, one tile's weight ramps down as the other's ramps up, and the pair sums
to one. Tiles that came back unchanged therefore reassemble to the input
exactly (to float rounding); tiles that were processed separately meet without
a hard seam. The weights are normalised per pixel, so the edge tiles that Split
shifts inward (to stay inside the frame) blend correctly too.

Tiles that were processed at a uniform scale (every tile upscaled 2x, say) are
placed on a canvas scaled the same way.
"""

import logging

import torch

from .bat_video_grid_split import grid_tile_indices, grid_tile_rects

logger = logging.getLogger("[Bat_GridMerge]")


def _profile(spans, i, length):
    """1-D feather weights for tile `i` along one axis.

    `spans` are the (start, end) of every tile index along that axis on the
    output canvas. Ramps cover the overlap with the previous / next tile; the
    +0.5 keeps every weight strictly positive, so no covered pixel divides by
    zero, and makes the two ramps of a shared band sum to exactly one.
    """
    a, b = spans[i]
    pos = torch.arange(a, b, dtype=torch.float32) + 0.5
    w = torch.ones(length, dtype=torch.float32)
    if i > 0:
        pb = spans[i - 1][1]
        n = min(pb, b) - a
        if n > 0:
            w = w * torch.clamp((pos - a) / n, max=1.0)
    if i < len(spans) - 1:
        na = spans[i + 1][0]
        n = b - max(na, a)
        if n > 0:
            w = w * torch.clamp((b - pos) / n, max=1.0)
    return w


class BatGridMerge:
    """Reassemble 🦇 Video Grid Split tiles into full frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {
                    "tooltip": "The tile list from 🦇 Video Grid Split (processed "
                               "or not), in the order Split produced it.",
                }),
                "columns": ("INT", {"default": 5, "min": 1, "max": 64,
                                    "tooltip": "Same as on Grid Split."}),
                "rows": ("INT", {"default": 3, "min": 1, "max": 64,
                                 "tooltip": "Same as on Grid Split."}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5,
                                      "step": 0.05, "display": "slider",
                                      "tooltip": "Same as on Grid Split. The "
                                                 "overlap is feathered linearly."}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                                        "tooltip": "Same as on Grid Split — which "
                                                   "tiles the list holds."}),
                "end_index": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1,
                                      "tooltip": "Same as on Grid Split. -1 = to "
                                                 "the last tile."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 1,
                                  "tooltip": "Width of the frames that went INTO "
                                             "Grid Split. 0 = take it from "
                                             "`reference`."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 32768, "step": 1,
                                   "tooltip": "Height of the frames that went INTO "
                                              "Grid Split. 0 = take it from "
                                              "`reference`."}),
            },
            "optional": {
                "reference": ("IMAGE", {
                    "tooltip": "The frames Grid Split was given — only their size "
                               "is used. Wire this or set width/height: a tile's "
                               "size alone does not pin down the frame size "
                               "(ceil rounding), and a 1-px guess would misplace "
                               "every tile.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "merge"
    CATEGORY = "BAT/Video"
    DESCRIPTION = (
        "Reassemble the tile list from 🦇 Video Grid Split into full frames, "
        "feathering the overlaps linearly. Use the same grid settings as the "
        "Split, and wire its input frames to `reference` (or type the size)."
    )

    def merge(self, tiles, columns, rows, overlap, start_index, end_index,
              width, height, reference=None):
        # INPUT_IS_LIST hands every input over as a list; the settings are
        # widgets, so each list holds one value.
        columns, rows, overlap = int(columns[0]), int(rows[0]), float(overlap[0])
        start_index, end_index = int(start_index[0]), int(end_index[0])
        width, height = int(width[0]), int(height[0])

        if (width <= 0 or height <= 0) and reference:
            ref = reference[0]
            height = height if height > 0 else int(ref.shape[1])
            width = width if width > 0 else int(ref.shape[2])
        if width <= 0 or height <= 0:
            raise ValueError(
                "🦇 Video Grid Merge: the original frame size is unknown. Wire "
                "the frames Grid Split was given into `reference`, or set "
                "width and height."
            )
        if rows > height or columns > width:
            raise ValueError(
                f"🦇 Video Grid Merge: grid {columns}x{rows} is finer than the "
                f"frame ({width}x{height}px)."
            )

        indices = list(grid_tile_indices(rows, columns, start_index, end_index))
        if not tiles:
            raise ValueError("🦇 Video Grid Merge: no tiles.")
        if len(tiles) != len(indices):
            raise ValueError(
                f"🦇 Video Grid Merge: got {len(tiles)} tiles but a {columns}x{rows} "
                f"grid with start {start_index} / end {end_index} selects "
                f"{len(indices)}. Use the same settings as the Grid Split."
            )

        rects = grid_tile_rects(height, width, rows, columns, overlap)
        tile_h, tile_w = rects[0][1] - rects[0][0], rects[0][3] - rects[0][2]
        first = tiles[0]
        b, th, tw, c = (int(v) for v in first.shape)
        for t in tiles:
            if tuple(t.shape) != (b, th, tw, c):
                raise ValueError(
                    f"🦇 Video Grid Merge: tiles differ in shape ({tuple(first.shape)} "
                    f"vs {tuple(t.shape)}); they must all be processed alike."
                )

        # Tiles processed at a uniform scale go on a canvas scaled to match.
        sy, sx = th / tile_h, tw / tile_w
        out_h, out_w = max(th, round(height * sy)), max(tw, round(width * sx))
        if (sy, sx) != (1.0, 1.0):
            logger.info("tiles are %.3gx / %.3gx the split size; merging onto "
                        "a %dx%d canvas.", sx, sy, out_w, out_h)

        # Spans per grid row / column on the output canvas. A row's tiles all
        # share one vertical span (and a column's one horizontal span), so the
        # feather profiles are per row and per column.
        def _span(p, scale, size, canvas):
            s = min(max(0, round(p * scale)), canvas - size)
            return (s, s + size)

        y_spans = [_span(rects[r * columns][0], sy, th, out_h) for r in range(rows)]
        x_spans = [_span(rects[col][2], sx, tw, out_w) for col in range(columns)]

        device = first.device
        acc = torch.zeros((b, out_h, out_w, c), dtype=torch.float32, device=device)
        wsum = torch.zeros((out_h, out_w), dtype=torch.float32, device=device)
        for tile, i in zip(tiles, indices):
            r, col = divmod(i, columns)
            (y1, y2), (x1, x2) = y_spans[r], x_spans[col]
            wgt = torch.outer(_profile(y_spans, r, th), _profile(x_spans, col, tw)).to(device)
            acc[:, y1:y2, x1:x2, :] += tile.to(device=device, dtype=torch.float32) * wgt[None, :, :, None]
            wsum[y1:y2, x1:x2] += wgt

        covered = wsum > 0
        if not bool(covered.all()):
            logger.warning("the tile list does not cover the whole frame "
                           "(start/end index); the gaps are left black.")
        out = acc / torch.where(covered, wsum, torch.ones_like(wsum))[None, :, :, None]
        return (out.to(first.dtype),)
