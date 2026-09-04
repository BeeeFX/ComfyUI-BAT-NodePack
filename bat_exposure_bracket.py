"""
Bat_ExposureBracket / Bat_ExposureMerge — bracket a plate, run each stop
through LTX's SDR->HDR separately, merge the results back into one HDR.

Why bracket at all
------------------
LTX's HDR IC-LoRA reconstructs from a display-referred plate, and what it
invents depends on where in the tonal range it is looking. Show it the same
sky at 0 EV and at -3 EV and it is solving two different problems: in the
first the sky is near clip and the model is extrapolating; in the second it
sits in the midtones where the model has the most to go on. Running several
exposures and merging buys two things:

* **Coverage.** Each stop is well-exposed over a different band, so the merge
  can take each region from the pass that had the most to work with.
* **Variance.** The reconstruction is generative, so each pass hallucinates
  slightly differently. Averaging N passes over a region where several are
  well-exposed suppresses that disagreement roughly as sqrt(N), which a single
  pass cannot do at any setting.

One honest caveat: exposing DOWN a plate that is already clipped at 1.0
recovers nothing by itself — a flat region stays flat, just darker. The gain
there is purely that LTX reconstructs it differently from a new tonal
position. If the plate is scene-linear with real values above 1.0 (an ACEScg
plate, say), exposing down genuinely reveals data, and the bracket earns its
keep twice over.

Wiring
------
    Bat_ExposureBracket ─┬─ bracket ──────────────────────────┐
                         ├─ ev 0.0  → LTX HDR → hdr_linear ─┐ │
                         ├─ ev -1.5 → LTX HDR → hdr_linear ─┤ │
                         └─ ev -3.0 → LTX HDR → hdr_linear ─┤ │
                                                            ↓ ↓
                                                   Bat_ExposureMerge
                                                            ↓
                                                        hdr_out
                                             → Bat_HDRTonalComposite.hdr_ai

Two details that are load-bearing
---------------------------------
**The pipe is output slot 0.** ComfyUI maps outputs to ``RETURN_TYPES`` by
INDEX, and litegraph's ``removeOutput`` renumbers the links on every slot
after the one removed. Put the pipe last and trimming the stop count would
silently repoint the pipe link at an IMAGE. With the pipe first, trimming only
ever removes from the end and every surviving index still means what it did.
(Inputs are keyed by NAME in the prompt, so the merge node's dynamic inputs
have no such constraint.)

**The merge's inputs follow the bracket, not the wiring.** The pipe's
*contents* only exist at execution time, but the bracket NODE is in the graph,
so the frontend reads its stop list straight off its widgets and gives the
merge exactly that many inputs, each labelled with the EV it expects. Change
the bracket's count and the merge follows. It resolves through Reroutes, falls
back to growing one spare slot at a time when no bracket is connected, and
refuses to trim a slot that has a live link.

**The merge recomputes each pass's SDR input rather than being handed it.**
The pipe carries the base plate in linear and the stop list; the weights are
derived from those. Passing the eight exposed images back in would work, but
it doubles the wiring and lets them drift out of step with the stops they are
supposed to correspond to.
"""

import base64
import logging
import math
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from .bat_hdr_preview import hdr_tile
from .bat_hdr_tonal_composite import (
    _EPS, _K_MIN, _K_MAX, _encode_from_linear, _luminance, _to_linear,
)

logger = logging.getLogger("[Bat_ExposureBracket]")

# Hard ceiling on the stop count. Eight is already far more passes than the
# variance argument pays for (sqrt(8) is 2.8x) and each one is a full LTX
# generation, so the cost is linear while the benefit is square-root.
MAX_STOPS = 8

# Width of the well-exposedness weight, in SDR code values around 0.5. The
# Mertens fusion default is 0.2 and it holds up here: at 0.2 a pixel sitting
# at clip still carries ~4% weight rather than none, so a region that clipped
# in EVERY pass still merges to something rather than dividing by zero.
_WELL_EXPOSED_SIGMA = 0.2

# Floor on the summed weight before it is used as a divisor.
_W_FLOOR = 1e-4

# Long edge of the on-node preview tile. Smaller than Bat_AdvancedBlend's 512
# because the strip derives up to eight exposures from it on every repaint and
# each one is a full per-pixel pass; 384 keeps a slider drag comfortable while
# still being big enough to judge whether a highlight has structure in it.
PREVIEW_TILE_DIM = 384

# Long edge for the MERGE preview's tiles. Smaller, because a single payload
# carries the plate plus one per connected pass — up to nine zlib'd 16-bit
# buffers — and that goes through the UI channel on every execution. At 256 the
# whole set lands around half a megabyte; at 384 it would be over one.
MERGE_TILE_DIM = 256

# Minimum overlap (summed pairwise weight, as a fraction of the frame) before
# an auto-alignment ratio is trusted. Two passes four stops apart may share
# almost no well-exposed pixels, and a ratio measured on a handful of them is
# noise.
_MIN_OVERLAP = 0.002

# Same chunking rationale as the composite node: hold the working set flat
# regardless of clip length. Scaled by the pass count, since the merge holds
# every pass live at once.
_CHUNK_BYTES = 128 << 20


# ---------------------------------------------------------------------------
# Stop layout
# ---------------------------------------------------------------------------

def parse_stops(text: str):
    """Parse a comma/space separated EV list. Returns None if unusable.

    Deliberately forgiving — artists type "0, -1.5, -3" and also "0 -1.5 -3"
    and also "0,-1.5,-3," — but it refuses silently-wrong input rather than
    guessing, so a typo falls back to the generated layout instead of
    producing a bracket nobody asked for.
    """
    if not text or not text.strip():
        return None
    out = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            logger.warning("could not parse %r in custom_stops; falling back "
                           "to the generated layout", tok)
            return None
    if not out:
        return None
    return out[:MAX_STOPS]


def build_stops(count: int, spacing: float, direction: str, custom: str = ""):
    """Resolve the EV list. `custom` wins when it parses."""
    parsed = parse_stops(custom)
    if parsed:
        return parsed
    count = max(1, min(int(count), MAX_STOPS))
    if direction == "up":
        return [i * spacing for i in range(count)]
    if direction == "symmetric":
        # Centre on 0. Even counts straddle it rather than favouring a side.
        start = -(count - 1) / 2.0
        return [(start + i) * spacing for i in range(count)]
    # "down" — the default. Clipped highlights are the usual failure and going
    # down is what gives LTX room in them; going up an already-clipped plate
    # mostly just clips more of it.
    return [-i * spacing for i in range(count)]


# ---------------------------------------------------------------------------
# Shared maths
# ---------------------------------------------------------------------------

def _expose_to_sdr(plate_lin: torch.Tensor, ev: float, gamma_mode: str):
    """Scene-linear -> one exposure, display-encoded and clamped for LTX.

    The clamp is the point of the operation, not an afterthought: LTX wants a
    display-referred image, and it is the clamp that turns "expose down" into
    "bring what was above white into the visible range".
    """
    exposed = (plate_lin * (2.0 ** float(ev))).clamp(0.0, 1.0)
    return _encode_from_linear(exposed, gamma_mode).clamp(0.0, 1.0)


def _b64_jpeg(arr_hwc: np.ndarray, max_dim: int = PREVIEW_TILE_DIM,
              quality: int = 82) -> str:
    """Downscaled base64 JPEG of an (H,W,3) uint8 array — the preview's 8-bit
    fallback, same shape the other BAT previews ship."""
    im = Image.fromarray(arr_hwc, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.BILINEAR)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _well_exposed(sdr_luma: torch.Tensor, sigma: float = _WELL_EXPOSED_SIGMA):
    """Mertens-style well-exposedness on the SDR input's luminance.

    Peaks at mid-grey and falls off toward both clip points, so a pixel is
    weighted by how much the model could actually see of it in that pass.
    """
    d = sdr_luma - 0.5
    return torch.exp(-(d * d) / (2.0 * sigma * sigma))


# ---------------------------------------------------------------------------
# Node 1 — the splitter
# ---------------------------------------------------------------------------

_GAMMA_MODES = ["srgb", "rec709", "gamma_2_2", "gamma_2_4", "linear"]


class BatExposureBracket:
    """Split one plate into N exposures, each ready for an LTX SDR->HDR pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate": ("IMAGE", {
                    "tooltip": "The plate to bracket. If it carries scene-linear values "
                               "above 1.0, exposing down genuinely recovers them."}),
            },
            "optional": {
                "plate_gamma_mode": (_GAMMA_MODES, {
                    "default": "srgb",
                    "tooltip": "How `plate` is encoded. Exposure is a multiply in LINEAR "
                               "light, so the plate is decoded, scaled, then re-encoded — "
                               "the same curve you would set on Bat_HDRTonalComposite."}),
                "count": ("INT", {
                    "default": 3, "min": 1, "max": MAX_STOPS,
                    "tooltip": "How many exposures. Each one is a separate LTX generation, "
                               "so cost is linear while the noise benefit is only sqrt(N) — "
                               "3 to 5 is the sweet spot."}),
                "spacing": ("FLOAT", {
                    "default": 1.5, "min": 0.25, "max": 6.0, "step": 0.25,
                    "tooltip": "Stops between exposures. Wide spacing covers more range but "
                               "leaves less overlap for the merge to align and average on."}),
                "direction": (["down", "symmetric", "up"], {
                    "default": "down",
                    "tooltip": "down: 0, -1.5, -3 ... biased at highlights, the usual "
                               "failure. symmetric: straddles 0, for plates with real range "
                               "at both ends. up: 0, +1.5, +3 for shadow-dominated plates."}),
                "custom_stops": ("STRING", {
                    "default": "",
                    "tooltip": "Type your own EV list and it overrides count/spacing/"
                               "direction entirely — e.g. '0, -1, -2.5, -4'. Leave empty to "
                               "use the generated layout. Unparseable text is ignored (with "
                               "a log warning) rather than guessed at."}),
                # Appended LAST deliberately. ComfyUI restores widgets_values
                # positionally, so adding a widget anywhere but the end would
                # shift every value in every saved workflow by one. A shorter
                # saved array simply leaves this at its default.
                "preview_frame": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which frame of the batch the on-node preview strip "
                               "shows. Display only — the bracket still applies to "
                               "every frame."}),
            },
        }

    # The pipe is FIRST on purpose — see the module docstring. Trimming the
    # dynamic slots only ever pops from the end, so every surviving output
    # index keeps meaning the same thing.
    RETURN_TYPES = ("BAT_BRACKET",) + ("IMAGE",) * MAX_STOPS
    RETURN_NAMES = ("bracket",) + tuple(f"ev_{i + 1}" for i in range(MAX_STOPS))
    OUTPUT_TOOLTIPS = (
        "Wire to Bat_ExposureMerge. Carries the stop list and the base plate so "
        "the merge can derive its own weights.",
    ) + tuple(f"Exposure {i + 1}, display-encoded and clamped — feed to an LTX "
              f"SDR->HDR pass." for i in range(MAX_STOPS))
    FUNCTION = "split"
    CATEGORY = "BAT/Colour"
    DESCRIPTION = (
        "Splits a plate into several exposures for separate LTX SDR->HDR passes, "
        "then Bat_ExposureMerge recombines them. Output slots appear and "
        "disappear with `count`. Live on-node strip previews every stop, so you "
        "can see whether exposing down actually recovers anything on THIS plate "
        "before spending an LTX generation per stop finding out."
    )

    def split(self, plate, plate_gamma_mode="srgb", count=3, spacing=1.5,
              direction="down", custom_stops="", preview_frame=0):
        if plate.ndim != 4 or plate.shape[3] < 3:
            raise ValueError(
                f"Bat_ExposureBracket expects a [B,H,W,C>=3] IMAGE, got "
                f"{tuple(plate.shape)}.")
        plate = plate[..., :3].float()
        stops = build_stops(count, spacing, direction, custom_stops)
        plate_lin = _to_linear(plate, str(plate_gamma_mode))

        images = [_expose_to_sdr(plate_lin, ev, str(plate_gamma_mode)) for ev in stops]

        pipe = {
            "version": 1,
            "stops": [float(e) for e in stops],
            "gamma_mode": str(plate_gamma_mode),
            # Kept by reference, not copied — the merge re-derives each pass's
            # SDR input from this so the weights cannot drift out of step with
            # the stops they belong to.
            "plate_lin": plate_lin,
        }

        # Every declared slot must be returned even when unused, or the
        # executor's index mapping breaks. The spares repeat the last real
        # exposure rather than being black, so an accidental connection to a
        # trimmed slot produces a sane picture instead of a mystery.
        padded = images + [images[-1]] * (MAX_STOPS - len(images))
        ui = {"stops": [[round(float(e), 3) for e in stops]],
              "count": [len(stops)]}
        ui.update(self._preview_payload(plate, str(plate_gamma_mode),
                                        preview_frame))
        return {"ui": ui, "result": (pipe, *padded)}

    # ------------------------------------------------------------------
    @staticmethod
    def _preview_payload(plate, gamma_mode, preview_frame):
        """Ship what the JS strip needs to render every stop for itself.

        The tile goes down **as received**, not decoded to linear, for one
        reason: `plate_gamma_mode` is a widget the artist flips, and it changes
        both the decode and the re-encode. Shipping a pre-decoded buffer would
        bake one guess in and leave the strip stale until the next run, whereas
        the JS can apply `toLinear` itself and update instantly. It mirrors the
        Python's own order — `_to_linear`, then `_expose_to_sdr` — rather than
        starting halfway through it.

        What the tile must NOT do is clamp. "Does exposing down actually
        recover anything, or is this plate clipped flat at 1.0?" is the single
        question the strip exists to answer, and a clamped tile answers it
        wrongly, always cheerfully, in the direction of "yes it does".
        `hdr_tile` range-normalises instead, so a plate peaking at 40.0 arrives
        intact. Area-sampled: the strip is displayed, and a nearest-neighbour
        decimate of a 4K frame just reads as broken.

        Degrades to the 8-bit JPEG if the tile can't be built — a preview is
        never worth failing an artist's run over.
        """
        idx = max(0, min(int(preview_frame), plate.shape[0] - 1))
        ui = {
            "gamma_mode": [gamma_mode],
            "frames": [int(plate.shape[0])],
            "preview_frame": [int(idx)],
            "w": [int(plate.shape[2])],
            "h": [int(plate.shape[1])],
        }

        tile = hdr_tile(plate[idx], PREVIEW_TILE_DIM, sample="area")
        if tile is not None:
            ui["plate_tile"] = [tile]
        else:
            logger.warning("could not build the preview tile; the stop strip "
                           "will fall back to the 8-bit plate and cannot show "
                           "recovery from above white.")

        # 8-bit fallback of the plate as received. Enough to frame the shot and
        # to survive in localStorage for the next reopen, but it is clamped, so
        # every stop derived from it below 0 EV will look flat in exactly the
        # region the artist is checking. The JS labels it when it is in use.
        u8 = (plate[idx].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        ui["plate_jpeg"] = [_b64_jpeg(u8)]
        return ui


# ---------------------------------------------------------------------------
# Node 2 — the merge
# ---------------------------------------------------------------------------

class BatExposureMerge:
    """Merge the LTX HDR results of a bracket back into one linear HDR."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {
            "align": (["auto", "nominal", "off"], {
                "default": "auto",
                "tooltip": "How to put the passes on a common scale before averaging. "
                           "auto: measure the real ratio between passes where both are "
                           "well exposed, falling back to nominal if they barely overlap. "
                           "nominal: assume LTX preserved the input exposure and divide by "
                           "2^ev. off: no scaling, for when the passes already match."}),
            "well_exposed_sigma": ("FLOAT", {
                "default": _WELL_EXPOSED_SIGMA, "min": 0.05, "max": 1.0, "step": 0.01,
                "tooltip": "Width of the well-exposedness weight around mid-grey. Smaller "
                           "= each region comes from fewer passes (sharper, noisier); "
                           "larger = more averaging (smoother, less per-pass detail)."}),
            "reference": (["auto", "first"], {
                "default": "auto",
                "tooltip": "Which pass the others are aligned to. auto picks the stop "
                           "closest to 0 EV, which is the one whose exposure you trust."}),
        }
        for i in range(MAX_STOPS):
            opt[f"hdr_{i + 1}"] = ("IMAGE", {
                "tooltip": f"hdr_linear from the LTX pass fed by the bracket's ev_{i + 1}."})
        return {"required": {"bracket": ("BAT_BRACKET",)}, "optional": opt}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_out",)
    OUTPUT_TOOLTIPS = (
        "The merged scene-linear HDR. Feed to Bat_HDRTonalComposite's hdr_ai.",)
    FUNCTION = "merge"
    CATEGORY = "BAT/Colour"
    DESCRIPTION = (
        "Merges the LTX HDR results of a Bat_ExposureBracket into one linear HDR, "
        "weighting each pass by how well exposed that pixel was in its own input. "
        "Input slots appear and disappear with `count`. Live on-node canvas with "
        "a viewer exposure — the output is unbounded HDR, so exposing down is the "
        "only way to see whether the highlights merged sensibly — plus a "
        "false-coloured weight view showing which pass owns which region."
    )

    def merge(self, bracket, align="auto", well_exposed_sigma=_WELL_EXPOSED_SIGMA,
              reference="auto", **passes):
        if not isinstance(bracket, dict) or "stops" not in bracket:
            raise ValueError(
                "Bat_ExposureMerge: `bracket` is not a BAT_BRACKET payload. Wire it "
                "from Bat_ExposureBracket's first output.")
        stops = bracket["stops"]
        gamma_mode = bracket.get("gamma_mode", "srgb")
        plate_lin = bracket["plate_lin"]

        # Collect the connected passes, keeping each one paired with the stop
        # it was generated at. An unconnected optional input simply is not in
        # kwargs, so a gap in the middle is handled rather than mis-indexed.
        got = []
        for i, ev in enumerate(stops):
            img = passes.get(f"hdr_{i + 1}")
            if img is None:
                continue
            got.append((ev, img[..., :3].float()))
        if not got:
            raise ValueError(
                f"Bat_ExposureMerge: no hdr_* inputs connected. The bracket declares "
                f"{len(stops)} stop(s); wire each LTX pass's hdr_linear to the matching "
                f"hdr_1..hdr_{len(stops)}.")
        missing = len(stops) - len(got)
        if missing:
            logger.warning("%d of %d bracket stops are not connected; merging the %d "
                           "that are", missing, len(stops), len(got))

        ref_shape = got[0][1].shape
        for ev, img in got:
            if img.shape != ref_shape:
                raise ValueError(
                    f"Bat_ExposureMerge: the passes disagree in shape — {tuple(ref_shape)} "
                    f"vs {tuple(img.shape)} (stop {ev:+.2f} EV). Every LTX pass has to run "
                    f"at the same resolution and frame count.")
        if plate_lin.shape[1:3] != ref_shape[1:3]:
            raise ValueError(
                f"Bat_ExposureMerge: the bracket's plate is "
                f"{plate_lin.shape[2]}x{plate_lin.shape[1]} but the passes are "
                f"{ref_shape[2]}x{ref_shape[1]}. The LTX graph resized somewhere; feed "
                f"Bat_ExposureBracket the same resolution the sampler saw.")
        if plate_lin.shape[0] != ref_shape[0]:
            if plate_lin.shape[0] == 1:
                plate_lin = plate_lin.expand(ref_shape[0], -1, -1, -1)
            else:
                raise ValueError(
                    f"Bat_ExposureMerge: the bracket's plate has {plate_lin.shape[0]} "
                    f"frames but the passes have {ref_shape[0]}.")

        scales = self._alignment(got, plate_lin, gamma_mode, align,
                                 reference, float(well_exposed_sigma))
        out = self._weighted_merge(got, plate_lin, gamma_mode, scales,
                                   float(well_exposed_sigma))
        ui = self._preview_payload(got, plate_lin, gamma_mode, scales, stops,
                                   align, reference, float(well_exposed_sigma))
        return {"ui": ui, "result": (out,)}

    # ------------------------------------------------------------------
    @staticmethod
    def _preview_payload(got, plate_lin, gamma_mode, scales, stops,
                         align, reference, sigma):
        """Everything the JS canvas needs to re-run the merge locally.

        Tiles: the plate (scene-linear, from the pipe) plus one per connected
        pass, all at the same long edge off the same area-sampler, so they
        arrive pixel-aligned and the JS loop can index them with one counter.
        MERGE_TILE_DIM is smaller than the bracket's because there are up to
        eight of them in a single payload and each is a zlib'd 16-bit buffer.

        The alignment scales are shipped rather than left to the JS. `auto`
        measures the ratio between two passes over the WHOLE frame, and a tile
        is 1/300th of the pixels — a tile-derived k is close but not the number
        the render used, and the merge's most common failure is a pass sitting
        at the wrong level, which is exactly what k controls. So the real ones
        go down, along with the widget values they were computed from; the JS
        uses them while those still match and recomputes from the tile (clearly
        labelled as an estimate) once the artist moves a slider.

        Note the plate arrives already linear here, unlike the bracket's
        payload: `gamma_mode` comes from the pipe, not from a widget on this
        node, so there is nothing for the artist to flip and nothing to keep
        live.
        """
        n_frames = int(plate_lin.shape[0])
        ui = {
            "gamma_mode": [gamma_mode],
            "stops": [[round(float(e), 3) for e in stops]],
            "pass_evs": [[round(float(ev), 3) for ev, _ in got]],
            "scales": [[float(k) for k in scales]],
            # The widget values `scales` was computed from. The JS compares
            # these against the live widgets and only trusts the shipped
            # numbers while they agree.
            "k_align": [str(align)],
            "k_reference": [str(reference)],
            "k_sigma": [float(sigma)],
            "frames": [n_frames],
            "w": [int(plate_lin.shape[2])],
            "h": [int(plate_lin.shape[1])],
        }

        plate_tile = hdr_tile(plate_lin[0], MERGE_TILE_DIM, sample="area")
        if plate_tile is None:
            logger.warning("could not build the merge preview's plate tile; "
                           "the live canvas will stay empty this run.")
            return ui
        ui["plate_lin_tile"] = [plate_tile]

        tiles = []
        for ev, img in got:
            t = hdr_tile(img[0], MERGE_TILE_DIM, sample="area")
            if t is None:
                logger.warning("could not build a preview tile for the %+.2f EV "
                               "pass; the live canvas will skip it.", ev)
                tiles = []
                break
            tiles.append(t)
        if tiles:
            ui["pass_tiles"] = [tiles]

        # No localStorage fallback for this node, unlike the bracket's. The
        # merge preview is meaningless without the pass tiles, and those are
        # hundreds of KB each against a ~5MB origin-wide budget shared with
        # every other BAT node's cache. It repopulates on the next run instead.
        return ui

    # ------------------------------------------------------------------
    @staticmethod
    def _ref_index(got, mode):
        if mode == "first":
            return 0
        return min(range(len(got)), key=lambda i: abs(got[i][0]))

    def _alignment(self, got, plate_lin, gamma_mode, mode, reference, sigma):
        """One scalar per pass, bringing them all onto the reference's scale."""
        n = len(got)
        if mode == "off":
            return [1.0] * n
        nominal = [2.0 ** (-ev) for ev, _ in got]
        if mode == "nominal":
            r = self._ref_index(got, reference)
            return [k / nominal[r] for k in nominal]
        if n == 1:
            return [1.0]

        r = self._ref_index(got, reference)
        w_ref = _well_exposed(_luminance(_expose_to_sdr(
            plate_lin, got[r][0], gamma_mode)), sigma)
        y_ref = _luminance(got[r][1])
        total = float(w_ref.numel())

        scales = []
        for i, (ev, img) in enumerate(got):
            if i == r:
                scales.append(1.0)
                continue
            w_i = _well_exposed(_luminance(_expose_to_sdr(
                plate_lin, ev, gamma_mode)), sigma)
            overlap = w_ref * w_i
            frac = float(overlap.sum()) / max(total, 1.0)
            y_i = _luminance(img)
            den = float((overlap * y_i).sum())
            num = float((overlap * y_ref).sum())
            if frac < _MIN_OVERLAP or den <= _EPS or num <= 0.0:
                k = nominal[i] / nominal[r]
                logger.info("stop %+.2f EV: overlap with the reference is %.3f%% — "
                            "too little to measure, using the nominal ratio %.4f",
                            ev, frac * 100.0, k)
            else:
                k = num / den
                if not math.isfinite(k) or not (_K_MIN <= k <= _K_MAX):
                    k = nominal[i] / nominal[r]
                    logger.warning("stop %+.2f EV: measured ratio was out of range; "
                                   "using the nominal %.4f", ev, k)
            scales.append(float(k))
        return scales

    @staticmethod
    def _weighted_merge(got, plate_lin, gamma_mode, scales, sigma):
        """sum(w_i * k_i * pass_i) / sum(w_i), chunked over frames."""
        b, h, w, _ = got[0][1].shape
        per_chunk = max(1, int(_CHUNK_BYTES //
                               max(h * w * 3 * 4 * max(len(got), 1), 1)))
        out = torch.empty((b, h, w, 3), dtype=torch.float32,
                          device=got[0][1].device)
        for s in range(0, b, per_chunk):
            e = min(s + per_chunk, b)
            acc = torch.zeros((e - s, h, w, 3), dtype=torch.float32,
                              device=out.device)
            wsum = torch.zeros((e - s, h, w, 1), dtype=torch.float32,
                               device=out.device)
            pl = plate_lin[s:e]
            for (ev, img), k in zip(got, scales):
                sdr = _expose_to_sdr(pl, ev, gamma_mode)
                wt = _well_exposed(_luminance(sdr), sigma).unsqueeze(-1)
                acc += img[s:e] * k * wt
                wsum += wt
            out[s:e] = acc / wsum.clamp(min=_W_FLOOR)
        return out.clamp(min=0.0)
