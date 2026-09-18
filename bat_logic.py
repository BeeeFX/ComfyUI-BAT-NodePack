"""
Logic and list helpers — 🦇 Compare / Index Switch / list nodes.
===============================================================

Core ComfyUI covers most of what a graph needs to branch: ``ComfySwitchNode``
(If/Else), ``ComfyNotNode`` / ``AndNode`` / ``OrNode``, ``StringCompare`` and
``ComfyMathExpression``. These five fill the gaps that are left.

* **Compare** — core compares *strings*. There is no generic a/b comparison
  producing a BOOLEAN, which is what you need to drive a switch from a frame
  count, a resolution or a computed value.
* **Index Switch** — core's switch is binary. Choosing between five loaders by
  an integer takes four chained If/Else nodes; this is one node.
* **List Length / Index / Batch** — core has ``ImageFromBatch`` and
  ``RepeatImageBatch`` for IMAGE specifically. These work on anything, which
  matters for the outputs that are not images: masks, metadata lists, layer
  dicts from 🦇 Loader.

Laziness
--------
``Bat_IndexSwitch`` declares its value inputs ``lazy``. Without that, wiring
five branches into a switch executes all five and throws four away — the exact
cost the node is meant to avoid. ``check_lazy_status`` asks only for the
branch ``index`` selects, so the others are never evaluated. Core's
``ComfySwitchNode`` uses the same mechanism.

Inspired by ``ComfyUI-Easy-Use``'s ``easy compare``, ``easy anythingIndexSwitch``,
``easy lengthAnything``, ``easy indexAnything`` and ``easy batchAnything``;
see CREDITS in README.md.
"""

from .bat_types import ANY

#: Number of branches on 🦇 Index Switch. Fixed rather than grown on demand —
#: the inputs are optional, so unused ones sit as empty dots and cost nothing.
#: 10 matches the usable range of easy-use's ``anythingIndexSwitch`` (it built
#: 20 input slots but clamped its index widget to 0–9), so every workflow that
#: migrates off it keeps all of its branches.
MAX_BRANCHES = 10

COMPARE_OPS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b":  lambda a, b: a < b,
    "a > b":  lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
    "a is None": lambda a, b: a is None,
    "a is not None": lambda a, b: a is not None,
    # The last four are carried over verbatim from easy-use's `easy compare`
    # so a migrated node keeps whatever operator it was set to. They read
    # oddly next to the others, but dropping them would silently change the
    # meaning of any workflow that used one.
    "a > 0":  lambda a, b: a > 0,
    "a <= 0": lambda a, b: a <= 0,
    "b > 0":  lambda a, b: b > 0,
    "b <= 0": lambda a, b: b <= 0,
}


def _is_tensor(value) -> bool:
    try:
        import torch
        return isinstance(value, torch.Tensor)
    except Exception:
        return False


class BatCompare:
    """Compare two values → BOOLEAN."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (ANY, {"tooltip": "Left-hand value."}),
                "b": (ANY, {"tooltip": "Right-hand value. Ignored by the None tests."}),
                "operation": (list(COMPARE_OPS.keys()), {
                    "default": "a == b",
                    "tooltip": "Comparison to apply.",
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "BAT/Logic"
    DESCRIPTION = "Compare two values and output a boolean, for driving switches."

    def compare(self, a=None, b=None, operation="a == b"):
        fn = COMPARE_OPS.get(operation)
        if fn is None:
            raise ValueError(f"Unknown operation {operation!r}")
        try:
            return (bool(fn(a, b)),)
        except TypeError as e:
            # Ordering two values of different types raises in Python 3. Say
            # which types, because "'<' not supported" alone is unhelpful when
            # the values came down a wildcard wire.
            raise TypeError(
                f"Cannot apply '{operation}' to {type(a).__name__} and "
                f"{type(b).__name__}: {e}"
            ) from None


class BatIndexSwitch:
    """Pick one of N inputs by integer index. Only the chosen branch executes."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(MAX_BRANCHES):
            optional[f"value{i}"] = (ANY, {
                "lazy": True,
                "tooltip": f"Returned when index == {i}.",
            })
        return {
            "required": {
                "index": ("INT", {
                    "default": 0, "min": 0, "max": MAX_BRANCHES - 1, "step": 1,
                    "tooltip": "Which input to pass through.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "switch"
    CATEGORY = "BAT/Logic"
    DESCRIPTION = (
        f"Pass through one of {MAX_BRANCHES} inputs, chosen by index. Inputs "
        "are lazy — the branches you did not select are never executed."
    )

    def check_lazy_status(self, index=0, **kwargs):
        """Ask the executor only for the branch we are about to return."""
        name = f"value{int(index)}"
        if kwargs.get(name) is None:
            return [name]
        return []

    def switch(self, index=0, **kwargs):
        name = f"value{int(index)}"
        if name not in kwargs:
            raise ValueError(
                f"index {index} is out of range (0–{MAX_BRANCHES - 1})."
            )
        value = kwargs[name]
        if value is None:
            raise ValueError(
                f"🦇 Index Switch: index is {index} but nothing is connected to "
                f"'{name}'. Wire that input, or change the index."
            )
        return (value,)


class BatListLength:
    """How many items are in a batch, list or string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY, {"tooltip": "Batch, list or string."})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "length"
    CATEGORY = "BAT/Logic"
    DESCRIPTION = (
        "Length of a batch, list or string. An image batch reports its frame "
        "count — the first dimension, not the pixel count."
    )

    def length(self, value=None):
        if value is None:
            return (0,)
        if _is_tensor(value):
            return (int(value.shape[0]) if value.dim() else 1,)
        if isinstance(value, (list, tuple, dict, str)):
            return (len(value),)
        return (1,)


class BatListIndex:
    """Pick item N out of a batch or list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {"tooltip": "Batch, list or string."}),
                "index": ("INT", {
                    "default": 0, "min": -1000000, "max": 1000000, "step": 1,
                    "tooltip": "0-based. Negative counts from the end (-1 is last).",
                }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "pick"
    CATEGORY = "BAT/Logic"
    DESCRIPTION = (
        "Take one item from a batch or list. An image batch keeps its batch "
        "dimension, so the result is still a valid IMAGE."
    )

    def pick(self, value=None, index=0):
        if value is None:
            raise ValueError("🦇 List Index: nothing connected to 'value'.")

        if _is_tensor(value):
            count = int(value.shape[0]) if value.dim() else 1
            i = self._resolve(index, count)
            # Slice rather than subscript: value[i] drops the batch dimension
            # and the result stops being an IMAGE.
            return (value[i:i + 1],)

        if isinstance(value, (list, tuple, str)):
            i = self._resolve(index, len(value))
            return (value[i],)

        # A scalar is a batch of one; index 0 is the only valid request.
        if self._resolve(index, 1) == 0:
            return (value,)
        raise IndexError(f"index {index} out of range for a single value.")

    @staticmethod
    def _resolve(index, count):
        if count == 0:
            raise IndexError("🦇 List Index: the input is empty.")
        i = index if index >= 0 else count + index
        if not 0 <= i < count:
            raise IndexError(
                f"🦇 List Index: index {index} is out of range for {count} item(s)."
            )
        return i


class BatListBatch:
    """Join two batches or lists into one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_a": (ANY, {"tooltip": "First batch or list."}),
                "value_b": (ANY, {"tooltip": "Second batch or list."}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("batch",)
    FUNCTION = "join"
    CATEGORY = "BAT/Logic"
    DESCRIPTION = (
        "Concatenate two batches or lists. Image batches join along the frame "
        "dimension and must agree on resolution and channel count."
    )

    def join(self, value_a=None, value_b=None):
        if value_a is None:
            return (value_b,)
        if value_b is None:
            return (value_a,)

        if _is_tensor(value_a) and _is_tensor(value_b):
            import torch
            if value_a.shape[1:] != value_b.shape[1:]:
                raise ValueError(
                    f"🦇 List Batch: cannot join {tuple(value_a.shape)} with "
                    f"{tuple(value_b.shape)} — everything after the frame "
                    f"dimension has to match. Rescale one of them first."
                )
            return (torch.cat((value_a, value_b), dim=0),)

        if isinstance(value_a, (list, tuple)) and isinstance(value_b, (list, tuple)):
            return (list(value_a) + list(value_b),)
        if isinstance(value_a, str) and isinstance(value_b, str):
            return (value_a + value_b,)

        return ([value_a, value_b],)
