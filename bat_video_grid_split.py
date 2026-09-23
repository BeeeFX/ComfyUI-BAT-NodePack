import math
import torch


def grid_tile_rects(h, w, rows, columns, overlap):
    """(y1, y2, x1, x2) of every tile of the grid, row-major.

    The one place the tile geometry lives: 🦇 Video Grid Split cuts with it and
    🦇 Grid Merge (bat_grid_merge.py) reassembles with it, so the two can never
    disagree about where a tile came from.
    """
    # 1. Calculate the Base Tile Size
    base_tile_h = h / rows
    base_tile_w = w / columns

    # 2. Calculate the Actual Tile Size (Base + Overlap)
    #    ceil, not int(): truncating dropped the last pixel row/column on
    #    any size that doesn't divide evenly (e.g. h=1081, rows=3 lost a
    #    row), leaving a seam when the tiles were recombined.
    tile_h = min(h, math.ceil(base_tile_h * (1 + overlap)))
    tile_w = min(w, math.ceil(base_tile_w * (1 + overlap)))

    rects = []
    for r in range(rows):
        for col in range(columns):
            # --- Coordinate Logic ---
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

            rects.append((y1, y2, x1, x2))
    return rects


def grid_tile_indices(rows, columns, start_index, end_index):
    """Tile indices a start/end pair selects: [start_index, end), where an
    end of -1 (or past the grid) means "to the last tile"."""
    total_tiles = rows * columns
    if end_index == -1 or end_index > total_tiles:
        limit_index = total_tiles
    else:
        limit_index = end_index
    return range(start_index, limit_index)


class VideoGridSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), 
                "columns": ("INT", {"default": 5, "min": 1, "max": 64}), 
                "rows": ("INT", {"default": 3, "min": 1, "max": 64}),
                "overlap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05, "display": "slider"}),
                # Added start and end index controls
                "start_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "end_index": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,) 
    FUNCTION = "split_video"
    CATEGORY = "BAT/Video"
    DESCRIPTION = "Split a video frame batch into a grid of sub-clips with optional overlap. Useful for tiled rendering where one input video maps to N output regions."

    def split_video(self, images, columns, rows, overlap, start_index, end_index):
        # Input shape: [Batch, Height, Width, Channels]
        b, h, w, c = images.shape

        # Reject grids finer than the pixel grid. Without this, tile_h/tile_w
        # truncated to 0 and every returned tile was an EMPTY tensor
        # ([N,0,W,C]), which blows up somewhere downstream with an opaque error
        # instead of naming the real problem here.
        if rows > h or columns > w:
            raise ValueError(
                f"Bat_VideoGridSplit: grid {columns}x{rows} is finer than the "
                f"image ({w}x{h}px) — each tile would be zero pixels. Reduce "
                f"rows/columns."
            )

        # Tiles are views into the batch — slicing, no copy.
        rects = grid_tile_rects(h, w, rows, columns, overlap)
        output_list = []
        for i in grid_tile_indices(rows, columns, start_index, end_index):
            y1, y2, x1, x2 = rects[i]
            output_list.append(images[:, y1:y2, x1:x2, :])

        return (output_list,)
