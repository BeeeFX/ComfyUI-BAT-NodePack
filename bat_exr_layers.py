"""
The two nodes that make 🦇 Loader's `layers` and `cryptomatte` outputs useful.

A multi-layer EXR arrives from the loader as a dict of named tensors. These pull
one out:

  🦇 EXR Layer        a named layer (diffuse, N, depth, …) as image + mask
  🦇 Cryptomatte Matte  a coverage mask for the objects you clicked on

Both publish the names they can see back to the node face over the
``bat-layers`` websocket message, so the name field becomes a dropdown of what
is actually in the file rather than something you have to spell correctly — see
web/bat_exr_layers.js.
"""

import json
import logging
import re

import torch

import server

logger = logging.getLogger(__name__)

# Layer keys that are bookkeeping rather than picture, hidden from the dropdown.
_HIDDEN_PREFIXES = ("raw_",)


def _publish_names(unique_id, names):
    """Send the available layer names to this node's face."""
    if not unique_id:
        return
    try:
        server.PromptServer.instance.send_sync(
            "bat-layers", {"node": unique_id, "layers": list(names)})
    except Exception as exc:                       # headless / API run
        logger.debug("[Bat_ExrLayer] could not publish layer names: %s", exc)


def _parse_points(raw, width, height):
    """A coordinate string from 🦇 Points Editor as [(x, y)] pixel indices.

    Accepts what that node actually emits in either of its modes — a JSON list
    of ``{"x": .., "y": ..}`` — and works out which it is: `normalize=True`
    gives 0..1 fractions, `normalize=False` gives pixels. A normalised
    coordinate is never above 1, so anything over 1.5 anywhere in the list means
    the whole list is in pixels. Guessing beats making the artist keep a toggle
    on the other node in sync with this one.

    Also accepts ``{"positive": [...], "negative": [...]}``, which is the shape
    a single combined field would carry.
    """
    if not raw or not str(raw).strip():
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("[Bat_Cryptomatte] could not parse points: %r", raw[:80])
        return []
    if isinstance(data, dict):
        data = data.get("positive") or []
    if not isinstance(data, list):
        return []

    pairs = []
    for p in data:
        try:
            if isinstance(p, dict):
                x, y = float(p["x"]), float(p["y"])
            else:
                x, y = float(p[0]), float(p[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        pairs.append((x, y))

    normalised = all(abs(x) <= 1.5 and abs(y) <= 1.5 for x, y in pairs) if pairs else True
    out = []
    for x, y in pairs:
        px = int(x * width) if normalised else int(x)
        py = int(y * height) if normalised else int(y)
        out.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
    return out


class BatExrLayer:
    """Pull one named layer out of a multi-layer EXR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layers": ("LAYERS",),
                "layer_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the layer to extract. Run once with "
                               "this node connected and the field becomes a "
                               "dropdown of what the file actually contains.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "BAT/Loader"
    DESCRIPTION = (
        "Extract a named layer (diffuse, normal, depth, position, …) from a "
        "multi-layer EXR loaded by 🦇 Loader. Single-channel layers come back "
        "as both a greyscale image and a mask."
    )
    FUNCTION = "get_layer"

    def get_layer(self, layers, layer_name, unique_id=None):
        layers = layers or {}
        visible = [k for k in layers
                   if not k.startswith(_HIDDEN_PREFIXES) and not k.startswith("crypto")]
        _publish_names(unique_id, visible)

        name = (layer_name or "").strip()
        if not name:
            if not visible:
                raise ValueError(
                    "Bat_ExrLayer: this EXR carries no extractable layers.")
            # A freshly connected node has no name yet; the first real layer is
            # a better answer than an error the artist can only fix by running
            # again.
            name = visible[0]
            logger.info("[Bat_ExrLayer] no layer picked — using %r", name)

        if name not in layers:
            raise KeyError(
                f"Bat_ExrLayer: no layer named {name!r}. This file has: "
                f"{', '.join(sorted(visible)) or '(none)'}")

        layer = layers[name]
        if not isinstance(layer, torch.Tensor):
            raise TypeError(
                f"Bat_ExrLayer: {name!r} is a {type(layer).__name__}, not a "
                "single layer — pick one of its parts instead.")

        # [N,H,W] is a single-channel layer (a mask, depth, an AOV); [N,H,W,C]
        # is a picture.
        if layer.dim() == 3:
            return (layer.unsqueeze(-1).repeat(1, 1, 1, 3), layer)

        # An explicit `<name>_alpha` companion wins over the layer's own fourth
        # channel: bat_exr stores it separately whenever the EXR named it that
        # way, and it is the more specific answer.
        alpha = layers.get(f"{name}_alpha")
        if isinstance(alpha, torch.Tensor):
            mask = alpha if alpha.dim() == 3 else alpha[..., 0]
        elif layer.shape[-1] >= 4:
            mask = layer[..., 3]
        else:
            mask = torch.zeros(layer.shape[:3], dtype=layer.dtype)

        image = layer[..., :3] if layer.shape[-1] >= 3 else layer
        # .contiguous(): both of those are strided VIEWS over the whole layer,
        # so without a copy the 3-channel output keeps the full N-channel buffer
        # alive for as long as anything downstream holds it.
        return (image.contiguous(), mask.contiguous())


class BatCryptomatteMatte:
    """Build a coverage mask from the cryptomatte IDs under some points."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cryptomatte": ("CRYPTOMATTE",),
                "layer_name": ("STRING", {
                    "default": "",
                    "tooltip": "Which cryptomatte to read (CryptoObject, "
                               "CryptoMaterial, …). Becomes a dropdown once "
                               "the node has run with a file connected.",
                }),
            },
            "optional": {
                "positive_coords": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Points on objects to KEEP. Wire 🦇 Points "
                               "Editor's positive_coords here.",
                }),
                "negative_coords": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Points on objects to DROP from the selection.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rank preview", "mask")
    CATEGORY = "BAT/Loader"
    DESCRIPTION = (
        "Turn clicked points into a cryptomatte coverage mask. A cryptomatte "
        "stores object IDs and their coverage in channel pairs; this reads the "
        "IDs under your positive points, drops the ones under your negative "
        "points, and sums the coverage of what's left across every rank. Feed "
        "it 🦇 Points Editor (with the loader's images as bg_image)."
    )
    FUNCTION = "get_matte"

    def get_matte(self, cryptomatte, layer_name, positive_coords="",
                  negative_coords="", unique_id=None):
        cryptomatte = cryptomatte or {}
        names = list(cryptomatte.keys())
        _publish_names(unique_id, names)
        if not names:
            raise ValueError("Bat_Cryptomatte: this EXR carries no layers.")

        name = (layer_name or "").strip()
        if name.startswith("crypto:"):
            name = name[len("crypto:"):]
        if not name:
            name = names[0]
            logger.info("[Bat_Cryptomatte] no layer picked — using %r", name)
        if name not in cryptomatte:
            raise KeyError(
                f"Bat_Cryptomatte: no layer named {name!r}. This file has: "
                f"{', '.join(sorted(names))}")

        primary = cryptomatte[name]
        if not isinstance(primary, torch.Tensor) or primary.dim() != 4:
            raise TypeError(
                f"Bat_Cryptomatte: {name!r} is not a cryptomatte rank stack.")

        frames, height, width = primary.shape[0], primary.shape[1], primary.shape[2]

        group = self._rank_group(cryptomatte, name, height, width) or [primary]

        mask = torch.zeros((frames, height, width), dtype=torch.float32)

        positives = _parse_points(positive_coords, width, height)
        negatives = _parse_points(negative_coords, width, height)
        if not positives:
            # No selection yet: an empty mask plus the rank preview, so the
            # artist can see what they are clicking on before they click.
            return (self._preview(primary), mask)

        # IDs are picked off the FIRST frame. A cryptomatte ID is stable for the
        # life of an object, so reading it once and applying it to every frame
        # is what makes the matte hold through a shot — re-picking per frame
        # would instead follow whatever happened to be under the cursor.
        def ids_at(points):
            found = set()
            for x, y in points:
                for rank in group:
                    for pair in range(0, rank.shape[-1] - 1, 2):
                        ident = rank[0, y, x, pair].item()
                        cover = rank[0, y, x, pair + 1].item()
                        if ident != 0.0 and cover > 0.0:
                            found.add(ident)
            return found

        target_ids = ids_at(positives) - ids_at(negatives)
        if not target_ids:
            logger.info("[Bat_Cryptomatte] no object IDs under those points "
                        "(clicked on empty background?) — the mask is empty")
            return (self._preview(primary), mask)

        # Sum coverage wherever a rank's id channel matches one of the targets.
        # Vectorised over the whole batch: the ids are the same every frame but
        # the pixels they cover are not, so the mask has to be built per frame.
        for tid in target_ids:
            for rank in group:
                for pair in range(0, rank.shape[-1] - 1, 2):
                    ident = rank[..., pair]
                    cover = rank[..., pair + 1]
                    # Exact float equality would be right in principle — these
                    # are bit patterns reinterpreted as floats — but a 16-bit
                    # half EXR quantises them, so compare with a tolerance
                    # relative to the magnitude rather than an absolute one.
                    hit = torch.isclose(ident, torch.full_like(ident, tid),
                                        rtol=1e-5, atol=0.0)
                    mask += torch.where(hit, cover, torch.zeros_like(cover))

        mask.clamp_(0.0, 1.0)
        return (self._preview(primary), mask)

    @staticmethod
    def _rank_group(cryptomatte, name, height, width):
        """Every rank belonging to the same cryptomatte as `name`.

        A cryptomatte is written as several RANKS — CryptoObject00,
        CryptoObject01, … — each holding two (id, coverage) channel pairs. A
        pixel covered by several objects spreads across them, so a selection has
        to be read from and summed over the whole group rather than one rank.

        Two details that a plain `startswith(base)` gets wrong on real files:

        * **The un-numbered alias.** Renders carry both ``object`` and
          ``object00``, holding the SAME rank. Summing over both double-counts
          every coverage value. So when numbered ranks exist, only those are
          used; the bare name is the group only when nothing is numbered.

        * **The split fourth channel.** bat_exr stores a 4-channel layer it did
          not recognise as a cryptomatte (this file's ranks are named
          ``object00``, not ``CryptoObject00``) as a 3-channel picture plus a
          separate ``*_alpha``. That drops the second (id, coverage) pair out of
          reach, and with it every object whose rank-1 coverage lives there.
          Rejoining them costs one concat per rank and makes the deeper ranks
          selectable again.
        """
        base = re.match(r"^(.*?)\d*$", name).group(1) or name
        numbered, bare = [], []
        for key in sorted(cryptomatte):
            hit = re.fullmatch(re.escape(base) + r"(\d*)", key)
            if not hit:
                continue
            value = cryptomatte[key]
            if not isinstance(value, torch.Tensor) or value.dim() != 4:
                continue
            if value.shape[1] != height or value.shape[2] != width:
                continue
            if value.shape[-1] == 3:
                alpha = cryptomatte.get(f"{key}_alpha")
                if isinstance(alpha, torch.Tensor) and alpha.dim() == 3:
                    value = torch.cat([value, alpha.unsqueeze(-1)], dim=-1)
            if value.shape[-1] < 2:
                continue
            (numbered if hit.group(1) else bare).append(value)
        return numbered or bare

    @staticmethod
    def _preview(rank):
        """The rank stack as something displayable — its first three channels.

        These are raw ID/coverage values, not colour, so it looks like noise.
        That is the point: it is a map of which object is where, which is what
        you aim at.
        """
        if rank.shape[-1] >= 3:
            return rank[..., :3].contiguous()
        return rank[..., :1].repeat(1, 1, 1, 3)
