"""Shared wildcard type for the BAT utility nodes.

ComfyUI decides whether two sockets may connect by comparing their type
strings. A ``str`` subclass whose ``__ne__`` always returns False therefore
reads as "equal to everything", which is how every pack in the ecosystem
spells a wildcard socket.

``bat_vace_batch`` carries its own private copy of this from before there was
a second user; it is left alone deliberately rather than refactored, so this
change touches no node that already works.
"""


class _AnyType(str):
    def __ne__(self, other):
        return False


#: Wildcard socket type — connects to any other type.
ANY = _AnyType("*")


def describe(value) -> str:
    """Short, human-readable rendering of any value.

    Used by the display nodes and by ``Bat_AnyToString``. Tensors are
    summarised rather than dumped: a 1080p float batch printed in full is
    several megabytes of text and would wedge the browser.
    """
    import json

    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)

    # torch is imported lazily so these nodes stay importable in the unit-test
    # harness, which has no torch.
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return _describe_tensor(value)
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        inner = ", ".join(describe(v) for v in value[:8])
        more = f", … (+{len(value) - 8})" if len(value) > 8 else ""
        return f"[{inner}{more}]"

    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return "<unprintable value>"


def _describe_tensor(t) -> str:
    """One-line summary of a tensor: shape, dtype, device, value range."""
    shape = "x".join(str(d) for d in tuple(t.shape))
    bits = [f"Tensor({shape})", str(t.dtype).replace("torch.", ""), str(t.device)]
    try:
        if t.numel() and t.is_floating_point():
            bits.append(f"min {t.min().item():.4g}  max {t.max().item():.4g}")
        elif t.numel():
            bits.append(f"min {t.min().item()}  max {t.max().item()}")
    except Exception:
        # A meta/quantised tensor has no readable values; the shape still does.
        pass
    return "  ·  ".join(bits)
