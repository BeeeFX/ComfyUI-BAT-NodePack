"""
Display nodes — 🦇 Show Any / 🦇 Show Tensor Shape.
==================================================

Two debug readouts that print a value onto the node face.

Why these exist when core has ``PreviewAny``
--------------------------------------------
Core's "Preview as Text" outputs the *string rendering* of its input, not the
input. So it cannot sit inline on a wire — dropping it between two nodes
replaces an IMAGE with a description of an IMAGE. These pass the original
value straight through, so you can insert one mid-graph to look at what is
flowing without rebuilding the connection afterwards.

The ``enabled`` toggle, and why it is the point
-----------------------------------------------
Both nodes are ``OUTPUT_NODE``s, because a debug readout is normally a
dead end — nothing is wired to its output — and a node that is not an output
node and has no consumer never executes at all.

Being an output node is not free. Output nodes are unconditional execution
roots: the executor adds every one of them to the run, so a branch that feeds
only a readout can never be skipped. Leave eight of them in a production
template — which is what a survey of nine studio workflows actually found —
and eight branches are pinned into every queue whether you are looking at
them or not.

``OUTPUT_NODE`` is read off the *class* by ``execution.py``, so it cannot be
switched per node instance. What *can* be switched per instance is whether the
input is ever requested. The ``value`` input is declared ``lazy``, and
``check_lazy_status`` asks for it only when ``enabled`` is true. Turn a readout
off and its upstream branch is genuinely not evaluated — the node still runs,
costs nothing, and reports that it is off.

The trade: while disabled the passthrough output emits ``None``, because the
value was never computed. That is harmless for the dead-end case these nodes
are usually in, and is the reason the toggle defaults to *on*. Only switch it
off on a readout whose output feeds nothing.

Inspired by ``ComfyUI-Easy-Use``'s ``easy showAnything`` and
``easy showTensorShape``, and by core's ``PreviewAny`` (itself from
rgthree-comfy's ``display_any``); see CREDITS in README.md.
"""

from .bat_types import ANY, describe

_DISABLED_TEXT = "(disabled — upstream branch not evaluated)"


def _tensor_report(value, _depth=0):
    """Multi-line shape report for a tensor, or a list/dict of tensors."""
    pad = "  " * _depth

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
        lines = [f"{pad}shape   {list(shape)}"]

        # ComfyUI's IMAGE is BHWC and MASK is BHW. Labelling the dimensions is
        # the whole reason to look at this node, so guess the layout rather
        # than printing bare numbers.
        if len(shape) == 4:
            lines.append(f"{pad}layout  batch {shape[0]} · {shape[2]}x{shape[1]} · {shape[3]} ch")
        elif len(shape) == 3:
            lines.append(f"{pad}layout  batch {shape[0]} · {shape[2]}x{shape[1]} (mask-like)")

        lines.append(f"{pad}dtype   {str(value.dtype).replace('torch.', '')}")
        lines.append(f"{pad}device  {value.device}")
        try:
            if value.numel():
                if value.is_floating_point():
                    lines.append(f"{pad}range   {value.min().item():.6g} → {value.max().item():.6g}")
                    lines.append(f"{pad}mean    {value.mean().item():.6g}")
                else:
                    lines.append(f"{pad}range   {value.min().item()} → {value.max().item()}")
        except Exception:
            lines.append(f"{pad}range   <unreadable>")
        try:
            mb = value.element_size() * value.numel() / (1024 ** 2)
            lines.append(f"{pad}memory  {mb:,.1f} MiB")
        except Exception:
            pass
        return "\n".join(lines)

    if isinstance(value, (list, tuple)):
        if not value:
            return f"{pad}(empty {type(value).__name__})"
        out = [f"{pad}{type(value).__name__} of {len(value)}"]
        for i, item in enumerate(value[:4]):
            out.append(f"{pad}[{i}]")
            out.append(_tensor_report(item, _depth + 1))
        if len(value) > 4:
            out.append(f"{pad}… (+{len(value) - 4} more)")
        return "\n".join(out)

    if isinstance(value, dict):
        if not value:
            return f"{pad}(empty dict)"
        out = [f"{pad}dict of {len(value)}"]
        for k in list(value)[:6]:
            out.append(f"{pad}{k}:")
            out.append(_tensor_report(value[k], _depth + 1))
        if len(value) > 6:
            out.append(f"{pad}… (+{len(value) - 6} more keys)")
        return "\n".join(out)

    return f"{pad}{type(value).__name__}  {describe(value)}"


class _ShowBase:
    """Shared laziness and passthrough plumbing for the two readouts."""

    OUTPUT_NODE = True
    CATEGORY = "BAT/Logic"

    def check_lazy_status(self, enabled=True, **kwargs):
        # The load-bearing line: returning [] means the executor never
        # evaluates whatever is upstream of `value`.
        #
        # Only request `value` when it is wired: an unwired optional input is
        # absent from kwargs (a wired-but-unevaluated one arrives as None), and
        # asking for an absent input makes the executor raise NodeInputError —
        # which failed every queue that had an unwired readout parked in it.
        if enabled and "value" in kwargs and kwargs["value"] is None:
            return ["value"]
        return []


class BatShowAny:
    """Print any value on the node, and pass it through unchanged."""

    OUTPUT_NODE = True
    CATEGORY = "BAT/Logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "showing",
                    "label_off": "off",
                    "tooltip":
                        "When off, the upstream branch is not evaluated at all "
                        "and the output is None. Use it to park a readout in a "
                        "template without paying for it every queue.",
                }),
            },
            "optional": {
                "value": (ANY, {"lazy": True, "tooltip": "Anything."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "show"
    DESCRIPTION = (
        "Show any value as text on the node and pass it through unchanged. "
        "Switch it off to skip its upstream branch entirely."
    )

    check_lazy_status = _ShowBase.check_lazy_status

    def show(self, enabled=True, value=None):
        if not enabled:
            return {"ui": {"text": [_DISABLED_TEXT]}, "result": (None,)}
        text = describe(value)
        return {"ui": {"text": [text]}, "result": (value,)}


class BatShowTensorShape:
    """Report a tensor's shape, dtype, device, range and size."""

    OUTPUT_NODE = True
    CATEGORY = "BAT/Logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "showing",
                    "label_off": "off",
                    "tooltip":
                        "When off, the upstream branch is not evaluated at all "
                        "and the outputs are None/empty.",
                }),
            },
            "optional": {
                "value": (ANY, {
                    "lazy": True,
                    "tooltip": "A tensor, or a list/dict containing tensors.",
                }),
            },
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("value", "info")
    FUNCTION = "show"
    DESCRIPTION = (
        "Shape, dtype, device, value range and memory size of a tensor — or of "
        "every tensor inside a list or dict. Passes the value through."
    )

    check_lazy_status = _ShowBase.check_lazy_status

    def show(self, enabled=True, value=None):
        if not enabled:
            return {"ui": {"text": [_DISABLED_TEXT]}, "result": (None, "")}
        text = _tensor_report(value)
        return {"ui": {"text": [text]}, "result": (value, text)}
