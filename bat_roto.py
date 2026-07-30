"""
Bat_Roto — Nuke-style roto with multi-shape bezier editor, per-frame
keyframes, and an animated MASK output.

Shape model (matches what the JS editor serialises into the `state` widget):

    state = {
        "shapes": [
            {
                "id": "abc12345",
                "name": "shape 1",
                "color": "#7ab8ff",
                "opacity": 1.0,
                "invert": false,
                "feather": 0.0,                     # pixels, applied at render time
                "keyframes": {
                    "0":  [point, point, ...],      # frame index → point list
                    "12": [point, point, ...],
                    "24": [point, point, ...]
                }
            },
            ...
        ]
    }

    point = [x, y, hx_in, hy_in, hx_out, hy_out]    # absolute pixel coords

Between keyframes each point lerps independently (position + both
tangent handles). Before the first keyframe / after the last we hold —
no extrapolation.

The output MASK is a (N, H, W) float tensor. For each frame we
rasterise every shape's interpolated point list as a closed cubic
bezier path (sampled to a polyline), fill it with cv2.fillPoly, apply
per-shape opacity / feather / invert, and union-add them all
(saturating at 1.0).
"""

import base64
import hashlib
import json
import math
import logging
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("[Bat_Roto]")

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("cv2 not installed — Bat_Roto will fall back to PIL "
                   "for shape rasterisation (slower).")
    from PIL import Image, ImageDraw, ImageFilter  # type: ignore


# ─── Bezier sampling ────────────────────────────────────────────────────────


def _sample_bezier_segment(p0, p1, p2, p3, n: int = 24):
    """Sample a cubic bezier segment at n evenly-spaced parameter values."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    mt = 1.0 - t
    return (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3


def _shape_to_polyline(points, samples_per_seg: int = 24) -> np.ndarray:
    """Turn a closed list of [x, y, hxi, hyi, hxo, hyo] points into a
    polyline (N, 2) approximating the cubic bezier path that goes
    point[i] → handle_out → handle_in → point[i+1] for every edge."""
    if len(points) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    out = []
    for i in range(n):
        p0 = pts[i, 0:2]
        p3 = pts[(i + 1) % n, 0:2]
        # Out-handle of p0, in-handle of p3 — convention: stored as
        # ABSOLUTE coordinates of the handle endpoints.
        p1 = pts[i, 4:6]
        p2 = pts[(i + 1) % n, 2:4]
        # Drop the last sample of each segment so adjacent segments don't
        # produce duplicate vertices.
        seg = _sample_bezier_segment(p0, p1, p2, p3, samples_per_seg)
        out.append(seg[:-1])
    out.append(pts[:1, 0:2])  # close
    return np.concatenate(out, axis=0)


# ─── Keyframe interpolation ─────────────────────────────────────────────────


def _interp_points(kf_a: list, kf_b: list, t: float) -> list:
    """Lerp every value in kf_a → kf_b at parameter t ∈ [0,1].

    Point counts can differ when the artist adds or removes an anchor at one
    keyframe but not the other. Returning kf_a in that case (the old behaviour)
    froze every in-between frame: add a point at frame 40 and frames 12-39 quietly
    stopped animating, with nothing in the UI to say why.

    Instead we pad the shorter list by repeating its LAST point up to the longer
    length, so the shared prefix keeps animating normally and only the extra
    anchors are static. Not a perfect correspondence — a true fix needs the
    editor to keep counts in sync (its deCasteljauSubdivide path does exactly
    that on insert) — but it degrades smoothly instead of silently freezing.
    """
    if not kf_a:
        return kf_b or []
    if not kf_b:
        return kf_a

    a_list, b_list = list(kf_a), list(kf_b)
    if len(a_list) != len(b_list):
        if len(a_list) < len(b_list):
            a_list = a_list + [a_list[-1]] * (len(b_list) - len(a_list))
        else:
            b_list = b_list + [b_list[-1]] * (len(a_list) - len(b_list))

    # Per-point component counts must also match ([x,y,hxi,hyi,hxo,hyo]); a
    # malformed/hand-edited point is the one case we still can't lerp.
    if any(len(p) != len(q) for p, q in zip(a_list, b_list)):
        return kf_a

    a = np.asarray(a_list, dtype=np.float32)
    b = np.asarray(b_list, dtype=np.float32)
    return (a + (b - a) * float(t)).tolist()


def _resolve_shape_at_frame(shape: dict, frame: int) -> Optional[list]:
    """Get the (possibly interpolated) point list of `shape` at `frame`."""
    kfs = shape.get("keyframes") or {}
    if not kfs:
        return None
    # Convert key strings to ints, sorted.
    sorted_keys = sorted((int(k) for k in kfs.keys()))
    if frame <= sorted_keys[0]:
        return kfs[str(sorted_keys[0])]
    if frame >= sorted_keys[-1]:
        return kfs[str(sorted_keys[-1])]
    # Bracket the frame.
    prev = sorted_keys[0]
    nxt = sorted_keys[-1]
    for k in sorted_keys:
        if k <= frame:
            prev = k
        if k >= frame and nxt == sorted_keys[-1] and k != prev:
            nxt = k
            break
    if prev == nxt:
        return kfs[str(prev)]
    span = nxt - prev
    t = (frame - prev) / span if span else 0.0
    return _interp_points(kfs[str(prev)], kfs[str(nxt)], t)


# ─── Rasterisation ──────────────────────────────────────────────────────────


def _rasterise_shape(
    mask: np.ndarray,           # (H, W) float32 accumulator, modified in place
    points: list,
    value: float,
    opacity: float,
    invert: bool,
    feather: float,
) -> None:
    """Draw `points` (closed) onto `mask` with the given options.

    The shape paints with RGB = `value` (a single grey level) and
    alpha = polygon_coverage × `opacity`. Coverage is the post-feather,
    optionally inverted polygon coverage (so the soft edge applies
    BEFORE the value/opacity gate). Composited into `mask` via
    Porter-Duff src-over: a fully-opaque shape replaces what's under
    it, while a partially-opaque shape blends. This is what lets the
    artist set a mid-grey shape that still occludes the shape below
    it — separate from `lighten/max`, which couldn't ever block an
    underlying contribution.
    """
    h, w = mask.shape
    polyline = _shape_to_polyline(points)
    if polyline.shape[0] < 3:
        return

    coverage = np.zeros((h, w), dtype=np.float32)
    if _HAS_CV2:
        cv2.fillPoly(coverage, [polyline.astype(np.int32)], 1.0)
        if feather > 0:
            # cv2.GaussianBlur kernel must be odd, and must be WIDE ENOUGH for
            # the sigma or the blur is a no-op. `int(round(feather))*2+1` gave
            # k=1 for any feather < 0.5 — a 1x1 kernel, i.e. the artist typed a
            # sub-pixel feather and saw nothing happen. Size the kernel from the
            # sigma the way OpenCV itself does (~3 sigma each side) and floor it
            # at 3 so a small feather still softens the edge.
            k = max(3, int(2 * math.ceil(3.0 * float(feather)) + 1))
            coverage = cv2.GaussianBlur(coverage, (k, k), float(feather))
    else:
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).polygon(
            [(float(p[0]), float(p[1])) for p in polyline], fill=255,
        )
        if feather > 0:
            im = im.filter(ImageFilter.GaussianBlur(radius=float(feather)))
        coverage = np.asarray(im, dtype=np.float32) / 255.0

    if invert:
        coverage = 1.0 - coverage

    src_alpha = coverage * float(opacity)
    src_value = float(value)

    # Src-over alpha composite in the mask domain:
    #   mask = src_value * src_alpha + mask * (1 - src_alpha)
    # Doing it in place to avoid a fresh allocation per shape.
    np.multiply(mask, 1.0 - src_alpha, out=mask)
    mask += src_value * src_alpha
    np.clip(mask, 0.0, 1.0, out=mask)


# ─── Node ───────────────────────────────────────────────────────────────────


class BatRoto:
    """Nuke-style roto. Output is a MASK animated across the input batch.

    State (shape list + keyframes) is persisted in a hidden STRING widget
    that the JS editor mutates on every edit. ComfyUI serialises widget
    values into the workflow JSON so the roto survives page refresh and
    follows the workflow across sessions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # Hidden-from-the-eye but visible-to-serialiser: we want
                # the workflow JSON to carry the state, but the artist
                # shouldn't see a giant blob of JSON on the node body.
                # ComfyUI's "hidden" key serialises but doesn't draw —
                # except hidden widgets aren't editable from JS in some
                # versions. Use STRING + multiline=False and let the JS
                # widget hide it visually instead.
                "state": ("STRING", {
                    "default": '{"shapes":[]}',
                    "multiline": False,
                    "_bat_hidden": True,
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "render"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Draw animated bezier roto masks on top of an image (sequence). "
        "Multi-shape stacking, per-shape opacity / invert / feather, "
        "per-frame keyframes. State persists into the workflow JSON."
    )

    @classmethod
    def IS_CHANGED(cls, images, state):
        # State changes drive re-render; image identity is handled by the
        # upstream image hash already.
        #
        # sha1, NOT the built-in hash(): hash() of a str is salted per process
        # (PYTHONHASHSEED), so identical state produced a different key after
        # every ComfyUI restart — forcing a needless full re-render on the first
        # run — and its 64-bit truncation could collide across different states
        # within one session, serving a stale cached result.
        try:
            return "state:" + hashlib.sha1(
                str(state).encode("utf-8", "surrogatepass")).hexdigest()
        except Exception:
            return "state:err"

    def render(self, images, state):
        n, h, w, _ = images.shape
        try:
            doc = json.loads(state) if state else {}
        except (TypeError, ValueError):
            logger.warning("Bat_Roto: state JSON invalid; outputting empty mask.")
            doc = {}

        shapes = doc.get("shapes") or []
        global_feather = float(doc.get("global_feather", 0.0))
        if shapes:
            out = np.zeros((n, h, w), dtype=np.float32)
            for f in range(n):
                for shape in shapes:
                    # Skip hidden shapes — the canvas eye toggle hides
                    # them from rendering AND drops them from the output.
                    if shape.get("visible") is False:
                        continue
                    # Open shapes (still being drawn) are skipped too —
                    # rasterising an open polygon usually looks like
                    # garbage and the artist hasn't committed yet.
                    # Older saved shapes default to closed=True via the
                    # JS hydration step, so this only skips shapes the
                    # artist is mid-creating.
                    if shape.get("closed") is False:
                        continue
                    # `value` defaults to 1.0 so legacy state JSON
                    # (which only had `opacity`) keeps producing the
                    # same final intensity = opacity × 1.0.
                    value = float(shape.get("value", 1.0))
                    opacity = float(shape.get("opacity", 1.0))
                    if value <= 0 or opacity <= 0:
                        continue
                    points = _resolve_shape_at_frame(shape, f)
                    if not points or len(points) < 2:
                        continue
                    _rasterise_shape(
                        mask=out[f],
                        points=points,
                        value=value,
                        opacity=opacity,
                        invert=bool(shape.get("invert", False)),
                        feather=float(shape.get("feather", 0.0)),
                    )
                # Global feather applied AFTER all shapes are unioned
                # on this frame so it softens the silhouette of the
                # combined mask rather than each shape individually
                # (different from per-shape feather, which softens the
                # shape's own edge before union).
                if global_feather > 0:
                    if _HAS_CV2:
                        # Kernel sized from sigma (see _rasterise_shape) so a
                        # sub-pixel feather isn't silently a no-op.
                        k = max(3, int(2 * math.ceil(3.0 * float(global_feather)) + 1))
                        out[f] = cv2.GaussianBlur(out[f], (k, k), float(global_feather))
                    else:
                        im = Image.fromarray(
                            (out[f] * 255.0).clip(0, 255).astype(np.uint8), "L",
                        ).filter(ImageFilter.GaussianBlur(radius=float(global_feather)))
                        out[f] = np.asarray(im, dtype=np.float32) / 255.0
            mask_tensor = torch.from_numpy(out)
        else:
            mask_tensor = torch.zeros((n, h, w), dtype=torch.float32)

        # Push input frames as base64 JPEGs so the JS editor can scrub
        # through them in the canvas. We downscale aggressively (max
        # 720px on the long edge) and cap the batch at 240 frames so
        # the {"ui":...} payload stays manageable even for long videos.
        # Coordinates the user clicks on the canvas are still in the
        # FULL-RES image space (we report the original w/h alongside the
        # thumbs) — the JS scales between display and image-space
        # coordinates itself.
        frames_b64 = []
        max_preview_frames = 240
        stride = max(1, n // max_preview_frames) if n > max_preview_frames else 1
        for i in range(0, n, stride):
            arr = (images[i].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            im = Image.fromarray(arr, "RGB")
            if max(im.size) > 720:
                r = 720 / max(im.size)
                im = im.resize(
                    (max(1, int(im.width * r)), max(1, int(im.height * r))),
                    Image.BILINEAR,
                )
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=78)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        return {
            "ui": {
                "frames": frames_b64,
                "w": [int(w)],
                "h": [int(h)],
                "stride": [int(stride)],
                "frame_count": [int(n)],
            },
            "result": (mask_tensor,),
        }
