"""
Bat_Framehold — select specific frames from an IMAGE batch by index.

The `frames` string is a comma-separated list of tokens; each token is
either a single index or an inclusive range:

    "0"        → frame index 0 (the first frame)
    "0-10"     → frames 0,1,2,…,10 (inclusive on both ends)
    "0, 5"     → frames 0 and 5
    "0-2, 7, 10-12"  → 0,1,2,7,10,11,12

Indices are 0-based. Negative indices count from the end (-1 = last),
matching Python slicing intuition. A descending range like "10-5" is
read as 10,9,…,5 so you can reverse a run. Out-of-range indices are
clamped to the valid batch range (and a warning is logged) rather than
crashing a render mid-graph. Order and duplicates are preserved exactly
as written — "5, 0, 5" yields three frames in that order — so this
doubles as a simple reorder/hold tool.

Selection is a single ``index_select`` on the batch dimension: one
vectorised gather, no per-frame Python copy.
"""

import logging
import re

import torch

logger = logging.getLogger("[Bat_Framehold]")


def _parse_frames(spec: str, n: int) -> list:
    """Parse the frame-spec string into an explicit ordered index list,
    resolving negatives and clamping to [0, n-1]."""
    out = []
    if not spec or not spec.strip():
        return out

    def _resolve(v: int) -> int:
        # Keep the value as written for the log message — `v` gets reassigned by
        # the negative-index conversion, so logging it after that printed the
        # resolved number and made the warning confusing ("index -3 below
        # range" when the artist typed -13).
        requested = v
        if v < 0:
            v = n + v
        if v < 0:
            logger.warning("Bat_Framehold: index %d below range; clamped to 0.", requested)
            return 0
        if v > n - 1:
            logger.warning("Bat_Framehold: index %d past last frame %d; clamped.",
                           requested, n - 1)
            return n - 1
        return v

    # Explicit grammar instead of hunting for "the first '-' after position 0",
    # which mis-parsed several real inputs: "5-" silently produced an empty
    # bound (skipped with only a log line), and "--1" split into an empty low
    # bound. A typo in a shot submission then rendered the wrong frames with no
    # hard error. Ranges and singles are now matched exactly, and anything that
    # doesn't match raises so the mistake surfaces immediately.
    range_re = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")
    single_re = re.compile(r"^-?\d+$")

    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        m = range_re.match(tok)
        if m:
            lo, hi = _resolve(int(m.group(1))), _resolve(int(m.group(2)))
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        elif single_re.match(tok):
            out.append(_resolve(int(tok)))
        else:
            raise ValueError(
                f"Bat_Framehold: cannot parse frame spec token {tok!r}. "
                f"Expected an index (e.g. 5, -1) or a range (e.g. 0-10, -5--1)."
            )
    return out


class BatFramehold:
    """Pick / reorder frames of an IMAGE batch by an index spec string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frames": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "placeholder": "e.g.  0   |   0-10   |   0, 5, 8-12",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "run"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Select frames from an image batch by index. Single: `0`. "
        "Inclusive range: `0-10`. List: `0, 5`. Mix: `0-2, 7, 10-12`. "
        "Negatives count from the end; order and duplicates are preserved."
    )

    def run(self, images, frames):
        n = images.shape[0]
        idxs = _parse_frames(frames, n)
        if not idxs:
            logger.warning("Bat_Framehold: no valid frames parsed from %r; "
                           "passing the batch through unchanged.", frames)
            return (images, int(n))
        index = torch.tensor(idxs, dtype=torch.long, device=images.device)
        picked = images.index_select(0, index)
        return (picked, int(picked.shape[0]))
