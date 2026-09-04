"""
High-precision preview transport shared by the BAT grading nodes.

The live canvas previews on Bat_Grade / Bat_AnimatedGrade normally ship the
input frame as a downscaled 8-bit JPEG, which is fine for judging a look but
destroys the one thing a bit-depth check needs: the source's quantisation
steps. Four separate stages were flattening it — a clamp to [0,1], a cast to
uint8, JPEG's lossy blocking, and a bilinear downscale that averages banding
away. A 16-bit render and an 8-bit render arrived at the browser byte-identical.

This module builds a second, small payload alongside that JPEG: a 16-bit
buffer of the same frame, point-sampled (never averaged) and zlib-compressed.
The JS side inflates it, grades in float, and only quantises to 8-bit at the
final putImageData — so the display is 8-bit (as every monitor is) but the
*maths* runs on the real data, and stepping a view exposure over a gradient
shows banding on an 8-bit source and smooth ramp on a 16-bit one.

Three deliberate choices:

* **Point sampling by default, not resize.** Any averaging filter is a
  lowpass, and quantisation banding is exactly the high-frequency detail it
  removes — a bilinear downscale would manufacture intermediate levels and
  hide the artefact we are trying to see. Nearest-neighbour keeps every
  surviving pixel's exact value. Callers that *display* the tile rather than
  inspecting it for bit depth want the opposite trade and can pass
  ``sample="area"``; Bat_HDRTonalComposite does, because it composites from
  the tile directly and an aliased tile just looks broken.
* **Range-normalised, not clamped.** Values are mapped through the frame's own
  [lo, hi] rather than clamped to [0,1], so scene-referred data above white
  survives to the viewer instead of being flattened before it leaves Python.
  Normalising cannot invent or destroy levels: an 8-bit source still lands on
  256 distinct codes, so the banding test stays honest.
* **Small.** 256px max on the long edge by default, so the extra payload is
  tens to a few hundred KB after zlib rather than the megabytes a
  full-resolution float buffer would push through the UI channel on every
  execution. `max_dim` is a parameter, though — a consumer that shows the
  tile full-width on the node body needs more than a bit-depth probe does.
"""

import base64
import logging
import zlib

import numpy as np

logger = logging.getLogger("[Bat_HDRPreview]")

# Long-edge cap for the high-precision tile. Small on purpose — see the module
# docstring. Banding is visible in a few hundred pixels of gradient; what it is
# not visible through is an averaging downscale, which is why the size cap is
# cheap but the sampling method is not negotiable.
HDR_MAX_DIM = 256


def _point_sample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Nearest-neighbour decimate an (H, W, C) array to fit `max_dim`.

    Deliberately not PIL/cv2 resize: every resampling filter they offer
    averages neighbouring pixels, which smooths out the quantisation steps
    this tile exists to reveal."""
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return arr
    nh = max(1, int(round(h * max_dim / longest)))
    nw = max(1, int(round(w * max_dim / longest)))
    ys = np.minimum((np.arange(nh) * (h / nh)).astype(np.int64), h - 1)
    xs = np.minimum((np.arange(nw) * (w / nw)).astype(np.int64), w - 1)
    return arr[ys][:, xs]


def _area_sample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Box-average decimate an (H, W, C) array to fit `max_dim`.

    The opposite trade to `_point_sample`, and the right one for any consumer
    that is judging tone or detail rather than bit depth. Nearest-neighbour
    from 1920 to 256 keeps one pixel in 56 and aliases everything else, which
    both looks crunchy and can miss a small highlight entirely or promote a
    stray one; an area average is what the eye expects from a scaled-down
    frame, and it estimates a region's real level rather than sampling one
    arbitrary pixel from it.

    `np.add.reduceat` gives an exact non-overlapping box filter for arbitrary
    (non-integer) ratios, which a simple reshape-and-mean cannot.
    """
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return arr
    nh = max(1, int(round(h * max_dim / longest)))
    nw = max(1, int(round(w * max_dim / longest)))
    ys = (np.arange(nh) * (h / nh)).astype(np.int64)
    xs = (np.arange(nw) * (w / nw)).astype(np.int64)
    # reduceat needs strictly increasing offsets. Guaranteed while we are
    # downscaling (h/nh >= 1), but a degenerate frame shouldn't raise.
    if np.any(np.diff(ys) < 1) or np.any(np.diff(xs) < 1):
        return _point_sample(arr, max_dim)
    cy = np.diff(np.append(ys, h)).astype(np.float32)
    cx = np.diff(np.append(xs, w)).astype(np.float32)
    out = np.add.reduceat(arr, ys, axis=0) / cy[:, None, None]
    out = np.add.reduceat(out, xs, axis=1) / cx[None, :, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def hdr_tile(frame, max_dim: int = HDR_MAX_DIM, sample: str = "point"):
    """Pack one (H, W, C>=3) float frame into the JS-side preview payload.

    Returns a dict of {data, w, h, lo, hi} — `data` being base64'd zlib'd
    little-endian uint16, RGB interleaved — or None if anything goes wrong.
    A preview is never worth failing an artist's run over, so every failure
    path here degrades to the existing 8-bit JPEG rather than raising.

    `sample` selects the decimation filter, and the default stays "point"
    because Bat_Grade's Inspect mode depends on it: averaging would erase the
    quantisation steps that whole feature exists to reveal. Pass "area" when
    the tile is going to be *displayed* rather than inspected for bit depth —
    Bat_HDRTonalComposite does, since it composites from the tile directly and
    a nearest-neighbour tile just looks aliased.
    """
    try:
        arr = frame.detach().cpu().numpy()
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = np.ascontiguousarray(arr[:, :, :3], dtype=np.float32)
        arr = (_area_sample(arr, max_dim) if sample == "area"
               else _point_sample(arr, max_dim))

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        lo = float(finite.min())
        hi = float(finite.max())
        # A flat frame (or a single-value one) would divide by zero on the JS
        # side; give it a unit span so it decodes back to a flat frame.
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-9:
            hi = lo + 1.0

        quantised = np.clip((arr - lo) / (hi - lo) * 65535.0 + 0.5, 0, 65535)
        packed = quantised.astype("<u2").tobytes()
        # Level 6 is the knee of the curve here: smooth CG frames compress to
        # a fraction of the raw size, and pushing to 9 costs noticeably more
        # CPU on every execution for a few percent.
        blob = zlib.compress(packed, 6)
        return {
            "data": base64.b64encode(blob).decode("ascii"),
            "w": int(arr.shape[1]),
            "h": int(arr.shape[0]),
            "lo": lo,
            "hi": hi,
        }
    except Exception as exc:  # pragma: no cover - preview must never break a run
        logger.warning("high-precision preview tile failed, falling back to JPEG: %s", exc)
        return None
