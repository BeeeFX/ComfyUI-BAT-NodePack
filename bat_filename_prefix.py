"""
Bat_FilenamePrefix — build a `filename_prefix` string from an ordered list of
user-defined path segments, so each workflow doesn't need its own pile of
String + String-Concatenate nodes feeding the save node.

The node grows two kinds of segments on demand (front-end, see
``web/bat_filename_prefix.js``):

  * **name**    — a freely-typed folder/filename level.
  * **version** — a ``vNNN``-style value with a configurable prefix and pad
                  width plus a seed-style control_after_generate, rendered as
                  e.g. ``v001`` / ``take0007``.

Each segment carries its own *preceding* separator (default ``/``), so a path
like ``MyShow/SH010/v001_render`` is just four segments where ``render`` uses
``_`` as its separator. The first segment's separator is always ignored.

The whole segment list (kind, label, value, separator, version settings, order)
is serialised to a single hidden ``segments`` STRING widget. That JSON rides in
the saved workflow, so the front-end can rebuild every widget on load — before
the first run — and the value survives restarts. This node just reads that JSON
and joins it.

Philosophy: **never silently rewrite the user's text.** Empty segments,
filename-illegal characters, or a stray path separator inside a segment raise a
clear error naming the offending segment rather than being cleaned up.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Cross-platform-strict: the Windows-forbidden set. These are illegal (or
# dangerous) in a path component on at least one OS / network share, so we
# reject them everywhere for portable outputs. The path separators `/` and `\`
# are handled separately — structure must come from the per-segment separator
# field, not from inside a segment's value.
_ILLEGAL_CHARS = set('<>:"|?*')
_PATH_SEPARATORS = set("/\\")


def _format_version(seg):
    """Render a version segment as ``{prefix}{number:0{pad}d}``."""
    try:
        number = int(seg.get("value", 0))
    except (TypeError, ValueError):
        raise ValueError(
            f"version segment {seg.get('label', '?')!r} has a non-integer "
            f"value {seg.get('value')!r}"
        )
    prefix = str(seg.get("prefix", "v"))
    try:
        pad = int(seg.get("pad", 3))
    except (TypeError, ValueError):
        pad = 3
    pad = max(0, pad)
    return f"{prefix}{number:0{pad}d}"


def _segment_text(seg, index):
    """Resolve one segment to its literal string, validating as we go."""
    label = seg.get("label") or f"segment_{index + 1}"
    kind = seg.get("kind", "name")

    if kind == "version":
        text = _format_version(seg)
    else:
        text = str(seg.get("value", ""))

    if text.strip() == "":
        raise ValueError(f"segment {index + 1} ({label!r}) is empty")
    if text != text.strip():
        raise ValueError(
            f"segment {index + 1} ({label!r}) has leading/trailing whitespace: "
            f"{text!r}"
        )

    bad = sorted(set(c for c in text if c in _ILLEGAL_CHARS))
    if bad:
        raise ValueError(
            f"segment {index + 1} ({label!r}) contains illegal character(s) "
            f"{''.join(bad)!r} — forbidden: < > : \" | ? *"
        )
    if any(c in _PATH_SEPARATORS for c in text):
        raise ValueError(
            f"segment {index + 1} ({label!r}) contains a path separator (/ or \\). "
            f"Use the segment's separator field to add folder levels instead."
        )
    return text


def build_prefix(segments):
    """Join an ordered segment list into a relative filename_prefix string."""
    if not segments:
        raise ValueError(
            "Bat_FilenamePrefix: no segments defined — add at least one "
            "Name or Version segment."
        )

    out = []
    for i, seg in enumerate(segments):
        text = _segment_text(seg, i)
        if i == 0:
            out.append(text)
        else:
            sep = str(seg.get("separator", "/"))
            out.append(sep + text)
    return "".join(out)


class BatFilenamePrefix:
    """Assemble a `filename_prefix` string from user-defined path segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hidden widget, populated/serialised entirely by the front-end.
                # Holds a JSON list of segment dicts. Kept "required" + STRING so
                # it serialises into the workflow and rides along on load.
                "segments": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    CATEGORY = "BAT/io"
    DESCRIPTION = (
        "Build a `filename_prefix` string from an ordered list of path "
        "segments (names and auto-incrementing version numbers), replacing a "
        "pile of String + Concatenate nodes feeding the save node."
    )
    FUNCTION = "build"

    @classmethod
    def IS_CHANGED(cls, segments):
        # The serialised JSON fully describes the output, so it is its own hash.
        return segments

    def build(self, segments):
        try:
            data = json.loads(segments) if segments else []
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bat_FilenamePrefix: corrupt segment data: {e}")
        if not isinstance(data, list):
            raise ValueError("Bat_FilenamePrefix: segment data is not a list")

        prefix = build_prefix(data)
        return (prefix,)
