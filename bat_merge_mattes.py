"""
Bat_MergeMattes — combine a primary matte with a secondary one, properly.
========================================================================

Built for CorridorKey 2 (primary) + MEMatte (secondary) on 2026-11-05: the
primary is the better matte overall, but the secondary sometimes holds extra
hair strands just outside its edge, or is solid where the primary has holes —
while the primary also finds subject the secondary missed entirely. A plain
add (or screen) over-brightens wherever both are semi-transparent; a plain max
also imports every stray blob of the secondary's, anywhere in frame.

The rule here: the secondary may only RAISE the matte, and only inside a trust
zone around the primary's subject.

    zone   = primary > threshold, closed by `fill_holes` px (so holes and
             narrow gaps inside the subject count as inside), then extended
             outwards with a linear falloff to 0 at `edge_reach` px
    merged = max(primary, secondary * zone * strength)

Properties that follow, and that tests/verify_merge_mattes.py checks:
  * never below the primary — nothing the primary found is lost;
  * never above max(primary, secondary) — no add/screen over-brightening;
  * the secondary is ignored beyond edge_reach px of the primary's subject;
  * a gap is only filled where the SECONDARY also sees subject: closing the
    zone over a real gap (between arm and body, between fingers) changes
    nothing unless the secondary claims that gap too.

Optional cleanup — the one case where the secondary may LOWER the matte. On
the shot this was built for, part of what the primary had and the secondary
did not was a mistake: a semi-transparent ghost over the wall in the gap
between chin and shoulder. `cleanup` fades out primary pixels that are
semi-transparent (below 0.9), where the secondary says clear background, and
more than `cleanup_distance` px from anything the secondary calls subject.
Solid primary pixels are never touched. It is OFF by default because, per
pixel, that ghost and real hair wisps only the primary found look the same:
measured on that frame, cleanup from 6 px removed most of the ghost and ~9 %
of the hair-wisp alpha; from 16 px it kept 99 % of the wisps and only thinned
the ghost. For one known region, a garbage matte is the precise tool.

Measured there (frame 40, 1920x1318): the secondary had more than the primary
on 12k pixels — ~3k in a 15 px rim outside the primary's edge, ~0.3k in holes,
32 px farther out. Over the 83 frames the merged matte flickers less than
either input (one-frame blink pixels 273/frame vs 326 primary, 463 secondary).
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:          # scipy fallback: slower, same result
    _HAS_CV2 = False
    from scipy import ndimage


def _as_batch(m):
    if m.dim() == 2:
        m = m.unsqueeze(0)
    elif m.dim() == 4:
        m = m[..., 0] if m.shape[-1] == 1 else m[:, 0]
    return m.float().cpu()


def _close(region, radius):
    """Binary closing with a disc of `radius` px (fills holes and gaps narrower than ~2r)."""
    if radius <= 0:
        return region
    if _HAS_CV2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        # Pad so the closing isn't clipped by the frame border.
        p = radius + 1
        padded = cv2.copyMakeBorder(region.astype(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        return cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k)[p:-p, p:-p].astype(bool)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = xx * xx + yy * yy <= radius * radius
    p = radius + 1
    padded = np.pad(region, p)
    return ndimage.binary_closing(padded, structure=disc)[p:-p, p:-p]


def _distance_outside(region):
    """Euclidean distance (px) from each pixel to the region; 0 inside it."""
    if not region.any():
        return np.full(region.shape, np.inf, np.float32)
    if _HAS_CV2:
        return cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)
    return ndimage.distance_transform_edt(~region).astype(np.float32)


def trust_zone(primary, threshold=0.5, fill_holes=16, edge_reach=12):
    """[H,W] float 0..1: how much the secondary is allowed to contribute at each pixel."""
    region = _close(primary > threshold, int(fill_holes))
    if edge_reach <= 0:
        return region.astype(np.float32)
    d = _distance_outside(region)
    return np.clip(1.0 - d / float(edge_reach), 0.0, 1.0).astype(np.float32)


CLEANUP_SOLID = 0.9      # primary at or above this is never cleaned up


def cleanup_weight(primary, secondary, distance):
    """[H,W] 0..1: how much of the primary to remove (see the module docstring)."""
    sec_subject = secondary > 0.5
    d = _distance_outside(sec_subject)
    ramp = max(4.0, float(distance))
    far = np.clip((d - float(distance)) / ramp, 0.0, 1.0)
    semi = np.clip((CLEANUP_SOLID - primary) / 0.2, 0.0, 1.0)
    return (far * semi * (secondary < 0.02)).astype(np.float32)


def merge_mattes(primary, secondary, threshold=0.5, fill_holes=16, edge_reach=12, strength=1.0,
                 cleanup=0.0, cleanup_distance=12, on_frame=None):
    """Merge two MASK batches -> (merged, added, removed, zone), each [T,H,W]."""
    p, s = _as_batch(primary), _as_batch(secondary)
    if p.shape[1:] != s.shape[1:]:
        raise ValueError(f"primary {tuple(p.shape[1:])} and secondary {tuple(s.shape[1:])} masks "
                         f"must be the same size.")
    if p.shape[0] != s.shape[0]:
        if s.shape[0] == 1:
            s = s.expand(p.shape[0], -1, -1)
        elif p.shape[0] == 1:
            p = p.expand(s.shape[0], -1, -1)
        else:
            raise ValueError(f"primary has {p.shape[0]} frames, secondary {s.shape[0]}: give the "
                             f"same number, or one mask for the whole clip.")
    T = p.shape[0]
    merged = torch.empty(T, *p.shape[1:])
    zone = torch.empty_like(merged)
    for t in range(T):
        pn = p[t].numpy()
        z = trust_zone(pn, threshold, fill_holes, edge_reach)
        zone[t] = torch.from_numpy(z)
        keep = p[t]
        if cleanup > 0:
            keep = keep * (1.0 - float(cleanup) * torch.from_numpy(cleanup_weight(pn, s[t].numpy(), cleanup_distance)))
        merged[t] = torch.maximum(keep, s[t] * zone[t] * float(strength))
        if on_frame is not None:
            on_frame(t)
    return merged, (merged - p).clamp(min=0.0), (p - merged).clamp(min=0.0), zone


class BatMergeMattes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary": ("MASK", {"tooltip": "The matte you trust most (e.g. CorridorKey 2's alpha). "
                                                "Everything in it is kept."}),
                "secondary": ("MASK", {"tooltip": "The matte that fills in (e.g. MEMatte's). It can only "
                                                  "raise the primary, and only near the primary's subject."}),
                "edge_reach": ("INT", {
                    "default": 12, "min": 0, "max": 256,
                    "tooltip": "How far (px) outside the primary's edge the secondary may add detail — "
                               "hair strands, a fuller edge. Its influence fades to nothing at this "
                               "distance, so stray blobs farther out are ignored.",
                }),
                "fill_holes": ("INT", {
                    "default": 16, "min": 0, "max": 256,
                    "tooltip": "Holes and gaps in the primary up to about twice this size (px) count as "
                               "inside the subject, so the secondary can fill them fully — but only where "
                               "the secondary itself sees subject. 0 = off (edge_reach still reaches a "
                               "little way into any hole from its rim).",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Where the primary counts as subject when working out its edge.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale on everything the secondary contributes. 0 = the primary unchanged.",
                }),
                "cleanup": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Let the secondary REMOVE semi-transparent primary ghosts where it sees "
                               "clear background (e.g. haze over a wall). Off by default: it can also "
                               "remove real hair wisps only the primary found. Solid primary is never "
                               "touched.",
                }),
                "cleanup_distance": ("INT", {
                    "default": 12, "min": 0, "max": 256,
                    "tooltip": "Cleanup only acts this far (px) or more from anything the secondary "
                               "calls subject. Larger keeps more wisps and removes less ghost.",
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("merged", "added", "removed", "trust_zone")
    OUTPUT_TOOLTIPS = (
        "max(primary, secondary x trust zone): never brighter than either matte, and never less "
        "than the primary unless cleanup is on.",
        "What the secondary added on top of the primary — for checking.",
        "What cleanup removed from the primary (empty when cleanup is 0).",
        "Where the secondary was allowed to contribute (1 = fully, fading to 0 at edge_reach).",
    )
    FUNCTION = "merge"
    CATEGORY = "BAT/mask"
    DESCRIPTION = (
        "Combine two mattes without the problems of add or max: the primary is kept whole, and the "
        "secondary only raises it inside a zone around the primary's subject — filling its holes "
        "and adding edge detail like hair strands, while ignoring the secondary's stray blobs "
        "elsewhere. Built for CorridorKey 2 (primary) + MEMatte (secondary).")

    def merge(self, primary, secondary, edge_reach, fill_holes, threshold, strength,
              cleanup=0.0, cleanup_distance=12):
        import comfy.model_management as mm
        import comfy.utils
        pbar = comfy.utils.ProgressBar(int(_as_batch(primary).shape[0]))

        def on_frame(_t):
            mm.throw_exception_if_processing_interrupted()
            pbar.update(1)

        return merge_mattes(primary, secondary, threshold, fill_holes, edge_reach, strength,
                            cleanup, cleanup_distance, on_frame)
