"""
Type conversion helpers — 🦇 Convert Any / Any to String / Number to String.
===========================================================================

Core ComfyUI already converts numbers well: ``ComfyNumberConvert``
("Convert Number") takes INT / FLOAT / STRING / BOOL and emits FLOAT and INT.
Use it where it fits — these three nodes exist only for what it does not do.

* It never emits STRING or BOOL, so there is no way to turn a computed value
  back into text for a filename, a label or a prompt. ``Bat_ConvertAny`` and
  ``Bat_AnyToString`` cover that direction.
* It formats nothing. A frame number wants ``0042``, a version wants ``v003``,
  and ``str(42)`` gives neither. ``Bat_NumberToString`` is the padding and
  precision node, which is the form these values actually take everywhere
  else in the pipeline (see the ``####`` / ``%04d`` handling in ETC Media
  Manager and Farm).

None of these is an ``OUTPUT_NODE``. That matters: easy-use's
``convertAnything`` declares ``OUTPUT_NODE = True`` for no reason that its
behaviour needs, and an output node is an unconditional execution root — one
sitting in a branch makes that branch permanently unskippable. A pure
conversion has no business forcing execution of anything.

Inspired by ``ComfyUI-Easy-Use``'s ``easy convertAnything`` and
``comfyui-art-venture``'s ``StringToInt`` / ``StringToNumber``; see
CREDITS in README.md.
"""

from .bat_types import ANY, describe

# Strings that mean False. Anything else non-empty is True — note that plain
# `bool("False")` is True, which is the trap this table exists to avoid.
_FALSEY = {"", "0", "false", "no", "off", "none", "null"}


def _to_float(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Cannot convert an empty string to a number.")
        return float(text)
    raise TypeError(f"Cannot convert {type(value).__name__} to a number.")


def _to_int(value) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Cannot convert an empty string to a number.")
        try:
            return int(text)
        except ValueError:
            # "3.7" is a perfectly ordinary thing to find in a string field;
            # int() rejects it outright, so go via float and truncate.
            return int(float(text))
    raise TypeError(f"Cannot convert {type(value).__name__} to a number.")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in _FALSEY
    return value is not None


class BatConvertAny:
    """any → STRING / INT / FLOAT / BOOLEAN, picked on the node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {
                    "tooltip": "Any value. Strings are parsed; booleans count "
                               "as 1/0; numbers convert directly.",
                }),
                "output_type": (["string", "int", "float", "boolean"], {
                    "default": "string",
                    "tooltip": "What to convert to. The output socket is a "
                               "wildcard, so it connects to whatever you wire "
                               "it into.",
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"
    CATEGORY = "BAT/Convert"
    DESCRIPTION = (
        "Convert any value to a string, int, float or boolean. Use core's "
        "Convert Number when you only need FLOAT/INT from a number."
    )

    def convert(self, value, output_type="string"):
        if output_type == "string":
            return (describe(value) if not isinstance(value, str) else value,)
        if output_type == "int":
            return (_to_int(value),)
        if output_type == "float":
            return (_to_float(value),)
        if output_type == "boolean":
            return (_to_bool(value),)
        raise ValueError(f"Unknown output_type {output_type!r}")


class BatAnyToString:
    """any → STRING, with control over how structured values are rendered."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"tooltip": "Any value."}),
                "mode": (["auto", "json", "repr"], {
                    "default": "auto",
                    "tooltip":
                        "auto — readable summary; tensors become "
                        "'Tensor(1x1080x1920x3) float32 cuda:0'. "
                        "json — pretty JSON where the value allows it. "
                        "repr — Python repr(), for debugging exact types.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "to_string"
    CATEGORY = "BAT/Convert"
    DESCRIPTION = (
        "Render any value as text. Tensors are summarised by shape and range "
        "rather than dumped, so a frame batch cannot flood the graph."
    )

    def to_string(self, value, mode="auto"):
        if mode == "repr":
            return (repr(value),)
        if mode == "json":
            import json
            try:
                return (json.dumps(value, indent=2, ensure_ascii=False, default=str),)
            except Exception:
                # Fall through rather than fail the prompt — the artist wanted
                # to look at a value, not to be told it is unserialisable.
                return (describe(value),)
        return (describe(value),)


class BatNumberToString:
    """INT / FLOAT → STRING with zero-padding, precision and affixes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"tooltip": "A number, or a string holding one."}),
                "decimals": ("INT", {
                    "default": -1, "min": -1, "max": 12, "step": 1,
                    "tooltip":
                        "Digits after the point. -1 keeps the value as it is: "
                        "an int prints with no point, a float prints its "
                        "shortest exact form.",
                }),
                "pad_to": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip":
                        "Zero-pad the integer part to this width. 4 turns 42 "
                        "into 0042 — frame numbers, version numbers.",
                }),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "tooltip": "Prepended, e.g. 'v'."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Appended, e.g. '.exr'."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "to_string"
    CATEGORY = "BAT/Convert"
    DESCRIPTION = (
        "Format a number as text — zero-padded frame numbers, fixed-precision "
        "values, 'v003' style version strings."
    )

    def to_string(self, value, decimals=-1, pad_to=0, prefix="", suffix=""):
        number = _to_float(value)

        if decimals < 0:
            # Preserve intent: an integer-valued input should not sprout a
            # decimal point just by passing through this node.
            if isinstance(value, bool) or float(number).is_integer():
                body = str(int(number))
            else:
                body = repr(float(number))
        else:
            body = f"{number:.{decimals}f}"

        if pad_to > 0:
            negative = body.startswith("-")
            if negative:
                body = body[1:]
            # Pad the integer part only — padding the whole string would count
            # the decimal point and the fraction digits toward the width.
            if "." in body:
                whole, frac = body.split(".", 1)
                body = f"{whole.zfill(pad_to)}.{frac}"
            else:
                body = body.zfill(pad_to)
            if negative:
                body = "-" + body

        return (f"{prefix}{body}{suffix}",)
