"""
Bat_AnimatedGrade — Nuke-style colour grade with per-frame keyframes.

Same grade formula as Bat_Grade (blackpoint / whitepoint / lift / gain /
multiply / offset / gamma + clamp toggles), but every parameter can be
keyframed across the input image batch. Between keyframes the parameters
lerp independently.

State JSON (kept in a hidden STRING widget; same persistence model as
Bat_Roto and Bat_AnimatedCrop):

    {
        "keyframes": {
            "0":  { blackpoint, whitepoint, lift, gain, multiply, offset,
                    gamma, clamp_white, clamp_black },
            "24": { ... },
            ...
        }
    }

Anything not present in `keyframes` falls back to "no change" defaults
(lift=0, gain=1, gamma=1, etc) so a freshly-placed node passes the
input through untouched.

The canvas editor pushes the input image batch as base-64 JPEG thumbs
(same payload shape as the Roto / AnimatedCrop nodes) so the artist can
scrub the timeline and see the grade applied live without re-running.
"""

import base64
import hashlib
import json
import logging
from io import BytesIO

import numpy as np
import torch
from PIL import Image

# Re-use the grade math from the static node so the two stay in lockstep.
from .bat_grade import _apply_grade

logger = logging.getLogger("[Bat_AnimatedGrade]")

# All numeric grade params that can be keyframed, with their pass-through
# defaults. clamp_* toggles are tracked separately so booleans don't try
# to interpolate.
_PARAMS = {
    "blackpoint": 0.0,
    "whitepoint": 1.0,
    "lift":       0.0,
    "gain":       1.0,
    "multiply":   1.0,
    "offset":     0.0,
    "gamma":      1.0,
}
_TOGGLES = ("clamp_white", "clamp_black")


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = 720, quality: int = 78) -> str:
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize(
            (max(1, int(im.width * r)), max(1, int(im.height * r))),
            Image.BILINEAR,
        )
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _b64_png_l(arr_hw: np.ndarray, max_dim: int = 720) -> str:
    im = Image.fromarray(arr_hw, "L")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize(
            (max(1, int(im.width * r)), max(1, int(im.height * r))),
            Image.BILINEAR,
        )
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resolve_grade_at_frame(kfs: dict, frame: int) -> dict:
    """Interpolated grade params at `frame`. Holds at the ends.

    Numeric params lerp; toggles take the surrounding keyframe's
    value (or False if there are no keyframes). Falls back to
    pass-through defaults for any param a keyframe omits, so a
    partially-edited shape doesn't NaN out the math.
    """
    if not kfs:
        out = dict(_PARAMS)
        for t in _TOGGLES:
            out[t] = False
        return out
    keys = sorted(int(k) for k in kfs.keys())
    if frame <= keys[0]:
        return _merge_defaults(kfs[str(keys[0])])
    if frame >= keys[-1]:
        return _merge_defaults(kfs[str(keys[-1])])
    prev = keys[0]
    nxt = keys[-1]
    for k in keys:
        if k <= frame:
            prev = k
        if k >= frame and k != prev:
            nxt = k
            break
    if prev == nxt:
        return _merge_defaults(kfs[str(prev)])
    a = _merge_defaults(kfs[str(prev)])
    b = _merge_defaults(kfs[str(nxt)])
    t = (frame - prev) / (nxt - prev)
    out = {}
    for name in _PARAMS:
        out[name] = a[name] + (b[name] - a[name]) * t
    for tname in _TOGGLES:
        # No sensible lerp for a bool; take the EARLIER keyframe's
        # value so toggle changes step on the keyframe that introduced
        # them (NLE convention).
        out[tname] = a[tname]
    return out


def _merge_defaults(kf: dict) -> dict:
    out = dict(_PARAMS)
    for t in _TOGGLES:
        out[t] = False
    if isinstance(kf, dict):
        for k, v in kf.items():
            if k in _PARAMS:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    pass
            elif k in _TOGGLES:
                out[k] = bool(v)
    return out


class BatAnimatedGrade:
    """Per-frame keyframed grade. Output dimensions = input dimensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Hidden state widget — the JS editor mutates it on
                # every keyframe edit; ComfyUI ships it inside the
                # workflow JSON.
                "state": ("STRING", {
                    "default": '{"keyframes":{}}',
                    "multiline": False,
                }),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grade"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Nuke-style grade with per-frame keyframes. Same math as Bat_Grade "
        "but every parameter is animatable across the input batch via the "
        "canvas timeline."
    )

    @classmethod
    def IS_CHANGED(cls, image, state, mask=None):
        # sha1 rather than hash() — see the note in bat_roto.IS_CHANGED: hash()
        # is salted per process, so the key changed on every restart.
        try:
            return "state:" + hashlib.sha1(
                str(state).encode("utf-8", "surrogatepass")).hexdigest()
        except Exception:
            return "state:err"

    def grade(self, image, state, mask=None):
        n, H, W, _ = image.shape
        try:
            doc = json.loads(state) if state else {}
        except (TypeError, ValueError):
            logger.warning("Bat_AnimatedGrade: state JSON invalid; using pass-through.")
            doc = {}
        kfs = doc.get("keyframes") or {}

        # Apply the grade in RUNS of consecutive frames that share the same
        # interpolated params, instead of one _apply_grade call per frame.
        #
        # _apply_grade is fully vectorised over a batch, so the per-frame loop
        # paid N times the kernel-launch overhead of a single batched call for
        # no benefit. In the overwhelmingly common cases — no keyframes, or one
        # keyframe (constant grade) — every frame resolves to identical params
        # and this collapses to ONE call over the whole batch. Between two
        # keyframes the params genuinely change each frame, so those degrade
        # gracefully back to per-frame runs.
        #
        # A per-frame mask forces a run break too: the mask varies per frame, so
        # frames sharing params can only be batched when they share a mask slice.
        def _param_key(p):
            # Round so float noise in the lerp doesn't split otherwise-equal runs.
            return (
                tuple(round(float(p[k]), 6) for k in
                      ("blackpoint", "whitepoint", "lift", "gain",
                       "multiply", "offset", "gamma")),
                bool(p["clamp_white"]), bool(p["clamp_black"]),
            )

        mask_is_per_frame = mask is not None and mask.shape[0] > 1

        runs = []            # [(start, end_exclusive, params)]
        for f in range(n):
            params = _resolve_grade_at_frame(kfs, f)
            key = _param_key(params)
            if (runs and runs[-1][3] == key and not mask_is_per_frame):
                runs[-1][1] = f + 1          # extend the current run
            else:
                runs.append([f, f + 1, params, key])

        out_frames = []
        for start, stop, params, _key in runs:
            batch_img = image[start:stop]
            batch_mask = None
            if mask is not None:
                if mask_is_per_frame:
                    # Runs are single frames in this case (see the break above).
                    mi = min(start, mask.shape[0] - 1)
                    batch_mask = mask[mi:mi + 1]
                else:
                    # Constant gate — _apply_grade broadcasts it over the run.
                    batch_mask = mask[0:1]
            graded = _apply_grade(
                image=batch_img,
                mask=batch_mask,
                blackpoint=float(params["blackpoint"]),
                whitepoint=float(params["whitepoint"]),
                lift=float(params["lift"]),
                gain=float(params["gain"]),
                multiply=float(params["multiply"]),
                offset=float(params["offset"]),
                gamma=float(params["gamma"]),
                clamp_white=bool(params["clamp_white"]),
                clamp_black=bool(params["clamp_black"]),
            )
            out_frames.append(graded)
        out = (out_frames[0] if len(out_frames) == 1
               else torch.cat(out_frames, dim=0)).contiguous()

        # Push input frames as base64 JPEGs so the JS editor can scrub
        # through them in the canvas (same payload contract as
        # Bat_Roto / Bat_AnimatedCrop).
        frames_b64 = []
        max_preview_frames = 240
        stride = max(1, n // max_preview_frames) if n > max_preview_frames else 1
        for i in range(0, n, stride):
            arr = (image[i].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            frames_b64.append(_b64_jpeg(arr))

        ui = {
            "frames": frames_b64,
            "w": [int(W)],
            "h": [int(H)],
            "stride": [int(stride)],
            "frame_count": [int(n)],
        }
        if mask is not None:
            # Send the first mask frame as a thumbnail so the JS preview
            # can replicate the gate too.
            mask_u8 = (mask[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            ui["input_mask"] = [_b64_png_l(mask_u8)]

        # No .cpu(): forcing the result to host memory cost a full device->host
        # copy (~750MB for a 300-frame 1080p batch) that the next node has to
        # copy straight back. ComfyUI handles device placement.
        return {"ui": ui, "result": (out,)}
