"""
BAT — keyframe easing shared by the animated editors.

Animated Crop, Animated Grade and Roto interpolate between keyframes. Until now
that was linear only. Each keyframe may now carry an ``ease`` naming the curve
used on the way OUT of it, towards the next key (After Effects' convention: the
outgoing segment belongs to the earlier key). A key without one is ``linear``,
so every saved workflow interpolates exactly as before.

``web/bat_easing.js`` is the same table for the editors' live previews; the two
must stay identical, or the preview and the render disagree mid-segment.
tests/verify_easing.py pins that.
"""

EASES = ("linear", "ease_in", "ease_out", "ease_in_out", "hold")
DEFAULT_EASE = "linear"


def ease_name(key):
    """The ease of a keyframe dict (or the default for anything else)."""
    name = key.get("ease") if isinstance(key, dict) else None
    return name if name in EASES else DEFAULT_EASE


def apply_ease(t, name):
    """Map a segment fraction t in [0, 1] through the named curve.

    ``hold`` keeps the earlier key's value until the next key (a step);
    the in/out curves are quadratic, ``ease_in_out`` is smoothstep.
    """
    t = 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else float(t)
    if name == "ease_in":
        return t * t
    if name == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if name == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    if name == "hold":
        return 0.0
    return t
