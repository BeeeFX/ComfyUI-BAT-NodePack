"""
Bat Points Editor — forked copy of KJNodes' PointsEditor with a
fix on the JS side: Reset canvas no longer collapses the coord
space back to the downscaled (≤1024) cached-image dimensions on
the first execution. The Python class matches upstream apart from
the fixes noted inline, so the node behaves identically once correct
width and height widgets reach it.
"""

import base64
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .bat_ui_ref import stash_ui

# The editor never draws or caches the plate above 1024 px on the long edge
# (maxDisplayDim in web/bat_points_editor/editor_base.js), so shipping more is
# only history bloat.
BG_PREVIEW_MAX_DIM = 1024


class BatPointsEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_store": ("STRING", {"multiline": False, "advanced": True}),
                "coordinates": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "neg_coordinates": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "bbox_store": ("STRING", {"multiline": False, "advanced": True}),
                "bboxes": ("STRING", {"multiline": False, "socketless": True, "advanced": True}),
                "bbox_format": (
                    [
                        'xyxy',
                        'xywh',
                    ],
                ),
                # 16384, not upstream's 4096: this fork keeps the plate's
                # native resolution as the coord space, so a 5K+ plate would
                # otherwise write a width the prompt validator rejects.
                "width": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 8}),
                "normalize": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "bg_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX", "MASK", "IMAGE")
    RETURN_NAMES = ("positive_coords", "negative_coords", "bbox", "bbox_mask", "cropped_image")
    FUNCTION = "pointdata"
    CATEGORY = "BAT/Utility"
    DESCRIPTION = """
## Graphical editor to create coordinates

**Shift + click** to add a positive (green) point.
**Shift + right click** to add a negative (red) point.
**Right click on a point** to delete it.
**Ctrl + click** to draw a bounding box.
**Drag bbox corners** to resize, **drag inside** to move.
**Right click on bbox** to delete it.

To add an image select the node and copy/paste or drag in the image.
Or from the bg_image input on queue (first frame of the batch).

Fork of KJNodes' Points Editor with the Reset-canvas resolution
bug fixed — clicking Reset no longer collapses the coord space
to 1024 on the longest side; original image dimensions are kept.
"""

    def pointdata(self, points_store, bbox_store, width, height, coordinates, neg_coordinates, normalize, bboxes, bbox_format="xyxy", bg_image=None):
        coordinates = json.loads(coordinates)
        if not coordinates:
            raise ValueError("No points on the canvas. Use Shift+click to add positive points or Shift+right-click to add negative points before executing.")
        pos_coordinates = []
        for coord in coordinates:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
            if normalize:
                norm_x = coord['x'] / width
                norm_y = coord['y'] / height
                pos_coordinates.append({'x': norm_x, 'y': norm_y})
            else:
                pos_coordinates.append({'x': coord['x'], 'y': coord['y']})

        # Parse into a fresh list ALWAYS. Previously `neg_coordinates` stayed a
        # raw string when falsy (""), so the json.dumps() at the end emitted
        # '""' instead of '[]' and downstream nodes got a string where they
        # expected a list.
        raw_neg = neg_coordinates
        neg_coordinates = []
        if raw_neg:
            coordinates = json.loads(raw_neg)
            for coord in coordinates:
                coord['x'] = int(round(coord['x']))
                coord['y'] = int(round(coord['y']))
                if normalize:
                    norm_x = coord['x'] / width
                    norm_y = coord['y'] / height
                    neg_coordinates.append({'x': norm_x, 'y': norm_y})
                else:
                    neg_coordinates.append({'x': coord['x'], 'y': coord['y']})

        mask = np.zeros((height, width), dtype=np.uint8)
        bboxes = json.loads(bboxes)
        valid_bboxes = []
        for bbox in bboxes:
            if (bbox.get("startX") is None or
                bbox.get("startY") is None or
                bbox.get("endX") is None or
                bbox.get("endY") is None):
                continue
            else:
                x_min = min(int(bbox["startX"]), int(bbox["endX"]))
                y_min = min(int(bbox["startY"]), int(bbox["endY"]))
                x_max = max(int(bbox["startX"]), int(bbox["endX"]))
                y_max = max(int(bbox["startY"]), int(bbox["endY"]))

                valid_bboxes.append((x_min, y_min, x_max, y_max))

        # NOTE: this block used to be indented INSIDE the loop above, which
        # (a) re-painted the mask for every accumulated bbox on each iteration
        # (1+2+3=6 paints for 3 bboxes) and (b) reassigned `bboxes` — the very
        # list being iterated — to a list of tuples, so the next iteration hit
        # `tuple.get(...)` and raised AttributeError. The node therefore crashed
        # with more than one bbox. Dedented to run once after the loop.
        bboxes_xyxy = []
        for bbox in valid_bboxes:
            x_min, y_min, x_max, y_max = bbox
            bboxes_xyxy.append((x_min, y_min, x_max, y_max))
            mask[y_min:y_max, x_min:x_max] = 1

        if bbox_format == "xywh":
            bboxes_xywh = []
            for bbox in valid_bboxes:
                x_min, y_min, x_max, y_max = bbox
                w = x_max - x_min
                h = y_max - y_min
                bboxes_xywh.append((x_min, y_min, w, h))
            bboxes = bboxes_xywh
        else:
            bboxes = bboxes_xyxy

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0).float().cpu()

        if bg_image is not None and len(valid_bboxes) > 0:
            x_min, y_min, x_max, y_max = bboxes[0]
            cropped_image = bg_image[:, y_min:y_max, x_min:x_max, :]
        elif bg_image is not None:
            cropped_image = bg_image
        else:
            cropped_image = torch.zeros(1, height, width, 3, dtype=torch.float32)

        if bg_image is None:
            return (json.dumps(pos_coordinates), json.dumps(neg_coordinates), bboxes, mask_tensor, cropped_image)
        else:
            # JPEG is RGB-only: an RGBA / grey+alpha plate raised OSError here.
            frame = bg_image[0]
            if frame.shape[-1] < 3:
                frame = frame[..., :1].repeat(1, 1, 3)
            elif frame.shape[-1] > 3:
                frame = frame[..., :3]
            transform = transforms.ToPILImage()
            image = transform(frame.permute(2, 0, 1))
            # Downscale for the editor, but send the true size alongside: the
            # coord space (the width/height widgets, and so every point) must
            # stay in plate pixels, not preview pixels.
            full_w, full_h = image.size
            if max(full_w, full_h) > BG_PREVIEW_MAX_DIM:
                r = BG_PREVIEW_MAX_DIM / max(full_w, full_h)
                image = image.resize((max(1, round(full_w * r)), max(1, round(full_h * r))), Image.BILINEAR)
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=75)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                # Sidecar'd so the plate stays out of the prompt history;
                # web/bat_ui_ref.js hands the editor the resolved dict.
                "ui": stash_ui({"bg_image": [img_base64], "bg_w": [full_w], "bg_h": [full_h]}),
                "result": (json.dumps(pos_coordinates), json.dumps(neg_coordinates), bboxes, mask_tensor, cropped_image)
            }
