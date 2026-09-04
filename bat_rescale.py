"""
Bat_Rescale — resolution change with a live preview that does NOT change size
on screen.

What it is for
--------------
The judgement this node exists to serve is "how far down can I take this plate
before I lose something I care about?", which comes up every time a shot is
going through an upscaler. An upscale done on a smaller, perfectly sharp source
beats one done on a bigger, slightly soft source — so the useful move is to
scale a plate *down* to the point just before detail loss becomes visible, and
hand that to the refiner.

Nothing in ComfyUI answers that question, because every preview scales with the
image: halve the resolution and the picture on the node gets smaller, which
tells you nothing about the pixels. This preview holds the picture at a
**constant size on screen** and changes only its resolution underneath — the
image stays the same size in your eye while its detail visibly goes away. That
is the entire point of the node, and it is why the viewer is region-based
rather than fit-based.

The preview
-----------
Two layers, the same shape as ``Bat_AdvancedBlend``'s (see that module for the
reasoning at length):

* **draft** — ``web/bat_rescale.js`` scaling the tile it already has, in the
  browser, on the same frame as your mouse. Approximate: canvas resampling is
  not this module's resampler.
* **truth** — a POST to ``/bat/rescale/render``, which runs *this* file's
  resampler on the cached frame at full resolution and returns a PNG of exactly
  the region on screen. Not a mirror of the render, but the render itself:
  region renders are asserted bit-identical to the same crop of a full-frame
  resize (``tests/verify_rescale.py``).

Both layers are returned as ONE stacked PNG holding the unscaled region on top
and the rescaled-and-magnified region below. That is not a transport
micro-optimisation — it is what makes the wipe trustworthy. The two halves have
to cover exactly the same source area or a wipe shows a subpixel jump at the
divider that reads as detail; rendering them in one call from one snapped
region makes that impossible to get wrong, and the wipe then costs nothing
because it never leaves the browser.

PNG, never JPEG: the subject here is per-pixel sharpness, and a DCT codec puts
its own texture exactly where we are looking.

Why the magnify filter is a viewer control
------------------------------------------
Going back up to screen size is a display decision, not part of the render, so
it lives in the viewer bar rather than in a widget:

* ``pixels`` (default) — nearest, so one output pixel is one visible block and
  you are looking at the real pixel grid rather than at a resampler's opinion
  of it. The default because that is the thing being decided: at a given scale,
  are the pixels still carrying the detail?
* ``smooth`` — resample back up with lanczos. The round-trip test of
  *information*: original versus round-tripped, same size, same subject, so
  what looks missing is what the downscale actually threw away. Reach for it
  when nearest's blockiness is answering a question you did not ask.

The readout carries an RMS and a PSNR of the round trip over the visible region,
because "noticeable" is a judgement and a number is a useful second opinion —
in practice the eye goes before ~48 dB on grain and long after it on flat CG.

Range
-----
The resampler is separable, runs in float32 and is unclamped, so a scene-linear
plate keeps its values above white; ``clamp_output`` is OFF by default. It is
implemented here rather than borrowed from ``comfy.utils.common_upscale`` for
the same reason ``bat_advanced_blend`` gives: that module's lanczos path
round-trips through PIL, which clamps to [0,1] and quantises to 8-bit.

The kernels themselves ARE borrowed from ``bat_advanced_blend`` — one
definition of lanczos in the pack, not two. What is local is the phase-explicit
matrix builder ``_map_matrix``, which resamples an arbitrary *source interval*
to N samples. That generalisation is what buys region-exactness: asking for
destination pixels ``[dx0, dx1)`` of a full-frame resize is the same call as
asking for the whole axis, so the preview cannot drift from the render by
construction. ``tests/verify_rescale.py`` still checks it against
``bat_advanced_blend._resize`` for whole frames, because "by construction" is
an argument and a test is evidence.

The preview is display-only: it is clamped to [0,1] and 8-bit on the way to the
browser, while the node's own output passes through at full float precision.
"""

import asyncio
import io
import json
import logging
import math
import threading
import uuid
import weakref
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

# The kernels, and the request-value coercion, come from the blend node so the
# pack has one lanczos rather than two that can drift apart.
from .bat_advanced_blend import _kernel, _num

logger = logging.getLogger("[Bat_Rescale]")

FILTERS = ["lanczos", "bicubic", "bilinear", "area", "nearest"]
MODES = ["factor", "long_edge", "short_edge", "width", "height",
         "megapixels", "match_reference"]

# Frames per chunk through the resampler. A 4K frame is ~100MB of float32, and
# the resample holds an input chunk, an intermediate and an output at once, so a
# 300-frame batch done whole would need tens of GB. Every operation here is
# per-frame, so chunking changes nothing but the peak.
CHUNK_FRAMES = 8

# Largest picture the preview endpoint will return per axis. The client asks for
# its own canvas size, so this is a backstop against a malformed request turning
# into a gigapixel allocation, not a limit anyone should meet.
MAX_OUT_DIM = 4096

# One cached 4K preview frame is 3840*2160*3*4 = 99MB. Four of them is a
# plausible working set for a shot graph with several of these nodes in it;
# past that the LRU evicts and the oldest node's viewer asks for a re-run.
CACHE_MAX_BYTES = 400 << 20


# ---------------------------------------------------------------------------
# Size planning
# ---------------------------------------------------------------------------
#
# NOTE: web/bat_rescale.js mirrors plan_size() so the node can show the output
# resolution before you run it. Change one, change the other —
# tests/verify_rescale.py runs the JS against this and asserts they agree.
#
# Rounding is `floor(x + 0.5)` rather than `round()` on purpose: Python's round()
# is banker's rounding and JS's Math.round() is not, so a factor landing exactly
# on .5 (0.5 on a 1921-wide plate) would disagree between the readout and the
# render.

def _round_half_up(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def _snap(v: int, multiple: int) -> int:
    """Round `v` to the nearest multiple of `multiple`, never below one."""
    if multiple <= 1:
        return max(1, int(v))
    return max(multiple, int(_round_half_up(v / multiple) * multiple))


def plan_size(h, w, mode, scale, target, megapixels, multiple_of,
              ref_h=None, ref_w=None):
    """Output (height, width) for a source of `h`x`w`.

    Aspect ratio is preserved by every mode except ``match_reference``, whose
    whole job is to land on someone else's exact frame size. ``multiple_of``
    then snaps both axes, which can shift the ratio by a fraction of a percent
    — that is the trade for handing a model a size it accepts, and it is why
    the snap is opt-in at 1.
    """
    h, w = max(1, int(h)), max(1, int(w))
    multiple_of = max(1, int(multiple_of))

    if mode == "match_reference":
        if ref_h and ref_w:
            return _snap(ref_h, multiple_of), _snap(ref_w, multiple_of)
        # No reference connected: a silent 1:1 passthrough is the least
        # surprising failure, and the node logs it once per run.
        return _snap(h, multiple_of), _snap(w, multiple_of)

    if mode == "long_edge":
        long_edge = max(h, w)
        s = float(target) / long_edge
    elif mode == "short_edge":
        s = float(target) / min(h, w)
    elif mode == "width":
        s = float(target) / w
    elif mode == "height":
        s = float(target) / h
    elif mode == "megapixels":
        s = math.sqrt(max(float(megapixels), 1e-6) * 1e6 / (w * h))
    else:                                   # "factor"
        s = float(scale)

    s = max(s, 1e-6)
    return (_snap(_round_half_up(h * s), multiple_of),
            _snap(_round_half_up(w * s), multiple_of))


# ---------------------------------------------------------------------------
# Resampling — separable, float-safe, phase-explicit
# ---------------------------------------------------------------------------

def _area_weights(x):
    """A HALF-OPEN box: -0.5 <= x < 0.5.

    ``bat_advanced_blend._kernel`` closes the box on both sides (``x.abs() <
    0.5``), which leaves a hole: when an output centre lands exactly on an input
    pixel boundary — which happens for every third sample of a 2:3 resize, so
    480->720 hits it constantly — *both* neighbouring taps evaluate to zero. The
    matrix builder then falls back to nearest for that row, and the fallback's
    rounding is not shift-invariant, so the same output pixel came out
    differently depending on which region it was rendered in. Closing one side
    removes the case entirely rather than papering over it.

    (That is a live, if minor, latent bug in Advanced Blend's `area`
    resize_filter when it is enlarging a plate. Left alone here rather than
    changed under a node this one does not own.)
    """
    return ((x >= -0.5) & (x < 0.5)).to(x.dtype)


def _map_matrix(src_lo, src_hi, slab_lo, n_slab, n_out, filt,
                device, dtype, _cache={}):
    """(n_out, n_slab) weights resampling a source interval to `n_out` samples.

    Coordinates are in source pixels of the FULL frame; the slab handed to the
    matmul starts at `slab_lo`. Splitting those two apart is what makes a region
    render exact: the output sample positions are derived from the interval, so
    destination pixels ``[dx0, dx1)`` of a full-frame resize can be requested on
    their own and land on precisely the grid the full resize would have used.

    Memoised: the frame loop is chunked, and rebuilding the matrix per chunk
    costs more than the resample.
    """
    key = (float(src_lo), float(src_hi), int(slab_lo), int(n_slab), int(n_out),
           filt, str(device), str(dtype))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    if len(_cache) > 64:
        _cache.clear()

    span = max(float(src_hi) - float(src_lo), 1e-9)
    scale = n_out / span                       # output samples per source pixel
    # Minification widens the kernel so it averages the pixels it steps over.
    # Skipping this correction is exactly what makes a naive bilinear downscale
    # alias, and aliasing in a tool for judging detail loss would be a lie in
    # the direction of "you lost more than you did".
    widen = 1.0 / scale if scale < 1.0 else 1.0
    fn, _support = _kernel(filt) if filt != "area" else (_area_weights, 0.5)

    centers = float(src_lo) + (torch.arange(n_out, device=device,
                                            dtype=torch.float32) + 0.5) / scale
    j = (torch.arange(n_slab, device=device, dtype=torch.float32)
         + float(slab_lo) + 0.5)
    d = (j[None, :] - centers[:, None]) / widen

    w = fn(d)
    s = w.sum(dim=1, keepdim=True)
    dead = s.abs() < 1e-12
    if bool(dead.any()):
        # Every tap zero — only reachable when the slab does not cover an
        # output centre. Fall back to nearest for those rows rather than
        # emitting NaN into a picture.
        rows = torch.nonzero(dead.squeeze(1), as_tuple=False).squeeze(1)
        # floor, not round: round-half-to-even is not shift-invariant, so a
        # dead row resolved differently depending on where the slab started.
        nearest = (centers - float(slab_lo)).floor().long().clamp(0, n_slab - 1)
        w[rows] = 0.0
        w[rows, nearest[rows]] = 1.0
        s = w.sum(dim=1, keepdim=True)

    out = (w / s).to(dtype)
    _cache[key] = out
    return out


def _nearest_index(src_lo, src_hi, slab_lo, n_slab, n_out, device):
    """Column indices for a nearest-neighbour resample of the same interval."""
    span = max(float(src_hi) - float(src_lo), 1e-9)
    centers = float(src_lo) + (torch.arange(n_out, device=device,
                                            dtype=torch.float32) + 0.5) * (span / n_out)
    return ((centers - float(slab_lo)).long()
            .clamp(0, max(0, n_slab - 1)))


def resample_region(img, out_h, out_w, filt,
                    src_x=0.0, src_y=0.0, src_w=None, src_h=None,
                    off_x=0, off_y=0):
    """Resample the source interval ``(src_x, src_y, src_w, src_h)`` of `img` to
    ``out_h`` x ``out_w``.

    `img` is (N, H, W, C) float. ``off_x`` / ``off_y`` say where `img` sits in
    the coordinate system the interval is expressed in, so a slab cut out of a
    bigger frame can be passed straight in. Nothing is clamped or quantised.
    """
    n, h, w, c = img.shape
    if src_w is None:
        src_w = w
    if src_h is None:
        src_h = h
    out_h, out_w = max(1, int(out_h)), max(1, int(out_w))

    out = img
    if filt == "nearest":
        xs = _nearest_index(src_x, src_x + src_w, off_x, w, out_w, img.device)
        ys = _nearest_index(src_y, src_y + src_h, off_y, h, out_h, img.device)
        return out[:, ys][:, :, xs]

    mx = _map_matrix(src_x, src_x + src_w, off_x, w, out_w,
                     filt, img.device, img.dtype)
    # (N,H,W,C) -> (N,H,C,W) @ (W,out_w) -> (N,H,C,out_w) -> back
    out = (out.movedim(2, -1) @ mx.transpose(0, 1)).movedim(-1, 2)
    my = _map_matrix(src_y, src_y + src_h, off_y, h, out_h,
                     filt, img.device, img.dtype)
    out = (out.movedim(1, -1) @ my.transpose(0, 1)).movedim(-1, 1)
    return out


def resize_batch(images, out_h, out_w, filt, chunk=CHUNK_FRAMES):
    """Resize an (N,H,W,C) batch, chunked over frames. Unclamped."""
    n, h, w, _c = images.shape
    if (h, w) == (int(out_h), int(out_w)):
        return images
    parts = []
    for i in range(0, n, chunk):
        blk = images[i:i + chunk].to(torch.float32)
        parts.append(resample_region(blk, out_h, out_w, filt))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Preview frame cache
# ---------------------------------------------------------------------------
#
# Keyed by a per-execution uuid rather than by the node id, for the two reasons
# Bat_Framehold's docstring spells out: a node id is only unique within one
# graph (so a reopened workflow can be served another shot's frame), and the id
# this node is handed at execution time is the flattened-prompt id, which does
# not match `node.id` in the browser once the node sits in a subgraph. The node
# id is still recorded, so a re-run drops that node's previous entry instead of
# stacking a second one.
#
# The entry also keeps a WEAK reference to the whole input batch. While the
# upstream node's output is still in ComfyUI's own execution cache — the normal
# case — the viewer can scrub to any frame at full resolution with no re-run and
# no extra memory. When it has been freed (`--cache-none`, a graph edit, a
# different workflow) the reference dies, the viewer falls back to the anchor
# frame and says so, which is a better failure than pinning a gigabyte.

_cache = OrderedDict()          # token -> entry
_cache_lock = threading.Lock()


def _entry_bytes(entry):
    t = entry.get("frame")
    return 0 if t is None else t.numel() * t.element_size()


def _cache_put(token, node_id, frame, batch, meta):
    """Cache one full-resolution preview frame, CPU-side.

    CPU on purpose: holding a 4K frame in VRAM to service a preview takes that
    memory away from the render the artist is actually waiting on. The region
    slice is pushed to the GPU per request instead, which costs a PCIe copy that
    is invisible next to the round trip.
    """
    entry = {
        "node_id": str(node_id),
        "frame": frame.detach().to("cpu", torch.float32).contiguous(),
        "batch": _weak(batch),
        "meta": meta,
    }
    with _cache_lock:
        for old_token, old in list(_cache.items()):
            if old["node_id"] == entry["node_id"]:
                _cache.pop(old_token, None)
        _cache[token] = entry
        total = sum(_entry_bytes(e) for e in _cache.values())
        while len(_cache) > 1 and total > CACHE_MAX_BYTES:
            _, dropped = _cache.popitem(last=False)
            total -= _entry_bytes(dropped)
            logger.debug("preview cache: evicted an entry to stay under %d MB",
                         CACHE_MAX_BYTES >> 20)


def _weak(t):
    try:
        return weakref.ref(t)
    except TypeError:            # not weak-referenceable on this torch build
        return None


def _cache_get(token):
    if not token:
        return None
    with _cache_lock:
        entry = _cache.get(token)
        if entry is not None:
            _cache.move_to_end(token)
        return entry


def _frame_for(entry, index):
    """The requested frame as a (1,H,W,3) CPU float tensor, plus whether it is
    really the frame that was asked for.

    Returns (frame, exact). `exact` is False when the batch has been freed and
    the anchor frame is standing in — the viewer badges that rather than
    pretending, because "this is frame 40" is exactly the claim the artist is
    relying on when they pick a frame to judge on.
    """
    anchor = int(entry["meta"].get("frame", 0))
    index = int(index)
    if index == anchor:
        return entry["frame"], True

    ref = entry.get("batch")
    batch = ref() if ref is not None else None
    if batch is None:
        return entry["frame"], False
    try:
        n = int(batch.shape[0])
        i = max(0, min(index, n - 1))
        f = batch[i:i + 1].detach().to("cpu", torch.float32)
        if f.shape[-1] > 3:
            f = f[..., :3]
        return f.contiguous(), i == index
    except Exception as exc:
        logger.debug("live batch frame %d unavailable (%s); using the anchor",
                     index, exc)
        return entry["frame"], False


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class BatRescale:
    """Change resolution, with a preview that holds its size on screen."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (MODES, {
                    "default": "factor",
                    "tooltip":
                        "What the number below means.\n"
                        "\n"
                        "factor — multiply both axes by `scale`\n"
                        "long_edge / short_edge — put that edge on `target` px\n"
                        "width / height — put that axis on `target` px\n"
                        "megapixels — land on `megapixels` total\n"
                        "match_reference — take the exact size of the "
                        "`reference` input (the only mode that may change the "
                        "aspect ratio)\n"
                        "\n"
                        "Every other mode preserves the aspect ratio; "
                        "`multiple_of` can then shift it by a fraction of a "
                        "percent.",
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.005,
                    "tooltip": "Used by `factor` mode. 0.5 is half resolution "
                               "on each axis (a quarter of the pixels).",
                }),
                "target": ("INT", {
                    "default": 1024, "min": 1, "max": 16384, "step": 1,
                    "tooltip": "Pixels, for the long_edge / short_edge / width "
                               "/ height modes.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 256.0, "step": 0.01,
                    "tooltip": "Target pixel count, for `megapixels` mode.",
                }),
                "filter": (FILTERS, {
                    "default": "lanczos",
                    "tooltip":
                        "Resampling kernel. lanczos is the sharpest and the "
                        "right default for a downscale you intend to upscale "
                        "again; area is a true box average (no ringing, "
                        "slightly softer); nearest is for when you want the "
                        "pixels themselves and no filtering at all.\n"
                        "\n"
                        "All of them widen the kernel when minifying, so none "
                        "of them alias.",
                }),
                "multiple_of": ("INT", {
                    "default": 1, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Snap both output axes to a multiple of this — "
                               "8, 16 or 64 for models that insist. 1 is off.",
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clamp the result to [0,1]. Off by default so a "
                               "scene-linear plate keeps its values above "
                               "white — a lanczos resample can also undershoot "
                               "below black around a hard edge, and clamping "
                               "is how you decide whether that survives.",
                }),
                # Appended last on purpose: ComfyUI restores widgets_values
                # positionally, so inserting a widget anywhere else would shift
                # every saved workflow's values by one.
                "preview_frame": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "step": 1,
                    "tooltip": "Which frame of the batch the on-node preview "
                               "shows. Display only — the rescale applies to "
                               "the whole batch either way. Also steppable "
                               "with the arrow keys on the viewer.",
                }),
            },
            "optional": {
                "reference": ("IMAGE", {
                    "tooltip": "Only read by `match_reference` mode: the output "
                               "lands on exactly this image's size.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("image", "width", "height", "scale")
    FUNCTION = "run"
    CATEGORY = "BAT/image"
    DESCRIPTION = (
        "Change resolution, with a live preview that stays the same size on "
        "screen so you see what the change does to the detail rather than to "
        "the thumbnail. Built for finding how far a plate can come down before "
        "an upscale starts to suffer. Unclamped float resamplers (lanczos / "
        "bicubic / bilinear / area / nearest), aspect preserved, optional "
        "multiple-of snapping."
    )

    def run(self, image, mode, scale, target, megapixels, filter,
            multiple_of, clamp_output, preview_frame=0, reference=None,
            unique_id=None):
        if image.ndim != 4:
            raise ValueError(f"expected an (N,H,W,C) batch, got {tuple(image.shape)}")
        n, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])

        ref_h = ref_w = None
        if reference is not None and reference.ndim == 4:
            ref_h, ref_w = int(reference.shape[1]), int(reference.shape[2])
        if mode == "match_reference" and ref_h is None:
            logger.warning("Bat_Rescale: mode is match_reference but nothing is "
                           "connected to `reference`; passing the size through.")

        filt = filter if filter in FILTERS else "lanczos"
        out_h, out_w = plan_size(h, w, mode, scale, target, megapixels,
                                 multiple_of, ref_h, ref_w)

        out = resize_batch(image, out_h, out_w, filt)
        if clamp_output:
            out = out.clamp(0.0, 1.0)

        idx = max(0, min(int(preview_frame), n - 1))
        ui = {
            "frames": [n],
            "src_w": [w], "src_h": [h],
            "out_w": [out_w], "out_h": [out_h],
            "preview_frame": [idx],
            "scale": [float(out_w) / w],
        }
        if unique_id is not None:
            try:
                ui.update(self._cache_preview(unique_id, image, idx, n, w, h))
            except Exception as exc:
                # A preview is never worth failing a render over.
                logger.warning("Bat_Rescale: could not cache the preview frame "
                               "(%s); the viewer will ask for a re-run.", exc)

        return {"ui": ui, "result": (out, int(out_w), int(out_h),
                                     float(out_w) / w)}

    # ------------------------------------------------------------------
    def _cache_preview(self, unique_id, image, idx, n, w, h):
        """Park the anchor frame for the full-resolution preview service.

        `.clone()` rather than a view: a view of the input batch keeps the whole
        batch alive, which is the one thing the weak reference in the entry is
        there to avoid.
        """
        frame = image[idx:idx + 1].detach()
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        frame = frame.clone()

        token = uuid.uuid4().hex[:16]
        _cache_put(token, unique_id, frame, image,
                   {"frames": int(n), "frame": int(idx),
                    "w": int(w), "h": int(h)})

        # 8-bit whole-frame thumbnail. Two jobs: it is the fit-view draft the
        # browser scales while the truth is in flight, and it is the only thing
        # small enough to keep in localStorage so a reopened workflow shows
        # something before the first Run.
        return {"token": [token], "thumb": [_b64_jpeg(frame[0])]}


def _b64_jpeg(frame, max_dim: int = 768, quality: int = 88) -> str:
    """Base64 JPEG of one (H,W,3) float frame, long edge capped."""
    import base64
    arr = (frame.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Full-resolution preview service
# ---------------------------------------------------------------------------

def _pad_interval(lo, hi, n_out, filt, bound_lo, bound_hi):
    """Whole-pixel range covering ``[lo, hi)`` plus one resampling kernel's reach.

    `n_out` is how many samples that interval is being resampled to, which is
    what sets the kernel's width: minification widens it by the scale ratio, so
    the context a region needs grows as it is scaled down. Clamped to
    ``[bound_lo, bound_hi)`` and never empty.
    """
    span = max(float(hi) - float(lo), 1e-9)
    scale = max(1, int(n_out)) / span
    widen = 1.0 / scale if scale < 1.0 else 1.0
    _fn, support = _kernel(filt)
    pad = int(math.ceil(support * widen)) + 2
    a = max(int(bound_lo), int(math.floor(lo)) - pad)
    b = min(int(bound_hi), int(math.ceil(hi)) + pad)
    return a, max(a + 1, b)


def render_pair(entry, p, roi, out_w, out_h, frame_index, magnify="smooth"):
    """Render the unscaled and rescaled versions of one region, stacked.

    `roi` is (x, y, w, h) in SOURCE pixels. The region is snapped outward to the
    destination grid first, so both halves cover exactly the same source area
    and a wipe between them cannot show a subpixel step.

    Returns (stacked_uint8, info) where `stacked` is (2*out_h, out_w, 3) —
    unscaled on top, rescaled below — and `info` carries the real numbers for
    the readout.
    """
    frame, exact = _frame_for(entry, frame_index)
    fh, fw = int(frame.shape[1]), int(frame.shape[2])

    out_full_h, out_full_w = plan_size(
        fh, fw, p["mode"], p["scale"], p["target"], p["megapixels"],
        p["multiple_of"], p.get("ref_h"), p.get("ref_w"))
    sx, sy = out_full_w / fw, out_full_h / fh

    x, y, rw, rh = (float(v) for v in roi)
    x = max(0.0, min(x, fw - 1.0))
    y = max(0.0, min(y, fh - 1.0))
    rw = max(1.0, min(rw, fw - x))
    rh = max(1.0, min(rh, fh - y))

    # The visible region, snapped OUT to whole SOURCE pixels. Source-integer
    # rather than destination-integer for one reason: at 1:1 view the unscaled
    # half is then a phase-exact identity resample, and a half-pixel shift on
    # the very picture the artist is comparing against would understate the
    # loss they are trying to see.
    sx0, sy0 = int(math.floor(x)), int(math.floor(y))
    sx1, sy1 = int(math.ceil(x + rw)), int(math.ceil(y + rh))
    sx1, sy1 = min(fw, max(sx0 + 1, sx1)), min(fh, max(sy0 + 1, sy1))

    # That same region expressed in destination pixels — fractional, and left
    # fractional. The magnify step below maps this exact interval to the screen
    # box, which is what keeps the two halves covering identical source area:
    # source and destination coordinates are related by a single linear map, so
    # "the same area" survives the trip through the destination grid.
    dsx0, dsx1 = sx0 * sx, sx1 * sx
    dsy0, dsy1 = sy0 * sy, sy1 * sy

    out_w = max(1, min(int(out_w), MAX_OUT_DIM))
    out_h = max(1, min(int(out_h), MAX_OUT_DIM))

    filt = p["filter"]
    magnify_filt = "nearest" if magnify == "nearest" else _up_filter(
        dsx1 - dsx0, dsy1 - dsy0, out_w, out_h)

    # Whole destination pixels to compute: the interval, plus the context the
    # MAGNIFY kernel reaches for outside it.
    dx0, dx1 = _pad_interval(dsx0, dsx1, out_w, magnify_filt, 0, out_full_w)
    dy0, dy1 = _pad_interval(dsy0, dsy1, out_h, magnify_filt, 0, out_full_h)
    dw, dh = dx1 - dx0, dy1 - dy0

    # ...and the source slab those destination pixels need, plus the context the
    # RESCALE kernel reaches for. Both margins are generous by two taps: too
    # small a slab is right in the middle of a region and wrong in a band around
    # its edge, a seam that only shows up once the artist pans.
    slab_x0, slab_x1 = _pad_interval(dx0 / sx, dx1 / sx, dw, filt, 0, fw)
    slab_y0, slab_y1 = _pad_interval(dy0 / sy, dy1 / sy, dh, filt, 0, fh)

    device = torch.device("cuda") if torch.cuda.is_available() else frame.device

    def _work(dev):
        slab = frame[:, slab_y0:slab_y1, slab_x0:slab_x1].to(dev, non_blocking=True)
        # The rescale, on the real destination grid: bit-identical to cropping
        # [dy0:dy1, dx0:dx1] out of a full-frame resize.
        scaled = resample_region(slab, dh, dw, filt,
                                 src_x=dx0 / sx, src_y=dy0 / sy,
                                 src_w=dw / sx, src_h=dh / sy,
                                 off_x=slab_x0, off_y=slab_y0)
        # Back up to the screen box. This is the display decision, and it is why
        # the picture does not change size when the resolution does.
        shown = resample_region(scaled, out_h, out_w, magnify_filt,
                                src_x=dsx0, src_y=dsy0,
                                src_w=dsx1 - dsx0, src_h=dsy1 - dsy0,
                                off_x=dx0, off_y=dy0)
        # The unscaled half: the same source area, straight to the screen box,
        # with no trip through the destination grid.
        ref = resample_region(slab, out_h, out_w,
                              _up_filter(sx1 - sx0, sy1 - sy0, out_w, out_h),
                              src_x=sx0, src_y=sy0,
                              src_w=sx1 - sx0, src_h=sy1 - sy0,
                              off_x=slab_x0, off_y=slab_y0)
        return ref, shown

    try:
        ref, shown = _work(device)
    except torch.cuda.OutOfMemoryError:
        # A preview must never be why a render fails.
        logger.warning("Bat_Rescale: preview did not fit in VRAM; "
                       "falling back to the host for this request")
        torch.cuda.empty_cache()
        ref, shown = _work(torch.device("cpu"))

    # What the round trip cost, over the visible region. Measured against the
    # reference half rather than against the source frame so it answers the
    # question the artist is actually looking at: same subject, same size, how
    # much of it survived. Clamped first, because a difference in values nobody
    # can see is not the difference being judged.
    a = ref[0].clamp(0, 1)
    b = shown[0].clamp(0, 1)
    mse = float(torch.mean((a - b) ** 2).item())
    rms = math.sqrt(mse)
    psnr = float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)

    stacked = torch.cat([a, b], dim=0)
    u8 = (stacked.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)

    info = {
        "out": [out_w, out_h],
        "res": [int(out_full_w), int(out_full_h)],
        "src": [fw, fh],
        "scale": out_full_w / fw,
        # The area actually shown, in source pixels — the client draws its
        # region box and its zoom readout from this rather than from what it
        # asked for, so the two can never disagree.
        "src_rect": [sx0, sy0, sx1 - sx0, sy1 - sy0],
        "dest_rect": [int(dsx0), int(dsy0),
                      max(1, int(round(dsx1 - dsx0))),
                      max(1, int(round(dsy1 - dsy0)))],
        "rms": rms,
        "psnr": None if psnr == float("inf") else psnr,
        "frame": int(frame_index),
        "frame_exact": bool(exact),
        "magnify": "nearest" if magnify == "nearest" else "smooth",
    }
    return u8, info


def _up_filter(in_w, in_h, out_w, out_h):
    """area when we are minifying for transport, lanczos when enlarging.

    Two different jobs wearing one name: at fit zoom the region is bigger than
    the canvas and wants a true average (point sampling a 4K frame to 900px
    keeps one pixel in 18 and shimmers), while past 1:1 the client has asked for
    more pixels than the region holds and lanczos is the honest enlargement.
    """
    return "area" if (out_w < in_w or out_h < in_h) else "lanczos"


def params_from_request(body):
    """Coerce and clamp the client's JSON. This is an HTTP endpoint, and a NaN
    scale reaching plan_size() is an unbounded allocation."""
    mode = str(body.get("mode", "factor"))
    if mode not in MODES:
        mode = "factor"
    filt = str(body.get("filter", "lanczos"))
    if filt not in FILTERS:
        filt = "lanczos"
    return {
        "mode": mode,
        "filter": filt,
        "scale": _num(body.get("scale", 1.0), 1.0, 0.01, 8.0),
        "target": int(_num(body.get("target", 1024), 1024, 1, 16384)),
        "megapixels": _num(body.get("megapixels", 1.0), 1.0, 0.01, 256.0),
        "multiple_of": int(_num(body.get("multiple_of", 1), 1, 1, 256)),
        "ref_h": None if body.get("ref_h") in (None, 0) else int(_num(body.get("ref_h"), 0, 1, 16384)),
        "ref_w": None if body.get("ref_w") in (None, 0) else int(_num(body.get("ref_w"), 0, 1, 16384)),
    }


try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.get("/bat/rescale/info")
    async def _bat_rescale_info(request):
        """Is the frame behind this token still cached?

        Called on workflow load, so a reopened graph gets its viewer back
        without a re-run — the browser remembers only the token.
        """
        entry = _cache_get(request.rel_url.query.get("token", ""))
        if entry is None:
            return web.json_response({"ok": False})
        ref = entry.get("batch")
        return web.json_response({
            "ok": True,
            "frames": int(entry["meta"].get("frames", 1)),
            "frame": int(entry["meta"].get("frame", 0)),
            "src_w": int(entry["frame"].shape[2]),
            "src_h": int(entry["frame"].shape[1]),
            # Whether the viewer can scrub frames without a re-run.
            "live_batch": bool(ref is not None and ref() is not None),
        })

    @server.PromptServer.instance.routes.post("/bat/rescale/render")
    async def _bat_rescale_render(request):
        """Render one region of the cached frame, unscaled and rescaled.

        POST JSON: {token, frame, roi:[x,y,w,h], out_w, out_h, magnify, ...params}
        Returns a PNG twice the requested height — unscaled on top, rescaled
        below — or 409 when there is nothing cached, at which point the client
        just keeps showing its draft.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        entry = _cache_get(str(body.get("token", "")))
        if entry is None:
            return web.json_response(
                {"error": "no cached frame for this token; run the node once"},
                status=409)

        try:
            p = params_from_request(body)
            fh, fw = int(entry["frame"].shape[1]), int(entry["frame"].shape[2])
            roi = body.get("roi") or [0, 0, fw, fh]
            out_w = int(_num(body.get("out_w", 512), 512, 1, MAX_OUT_DIM))
            out_h = int(_num(body.get("out_h", 512), 512, 1, MAX_OUT_DIM))
            frame_index = int(_num(body.get("frame", 0), 0, 0, 999999))
            magnify = "nearest" if str(body.get("magnify")) == "nearest" else "smooth"
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": f"bad request: {exc}"}, status=400)

        loop = asyncio.get_running_loop()
        try:
            # Off the event loop: a fit-view render of a 4K frame is a real
            # torch resample and would otherwise stall every websocket message
            # the frontend is waiting on, execution progress included.
            u8, info = await loop.run_in_executor(
                None, render_pair, entry, p, roi, out_w, out_h,
                frame_index, magnify)
        except Exception as exc:
            logger.warning("Bat_Rescale: preview render failed: %s", exc,
                           exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

        buf = io.BytesIO()
        # PNG, not JPEG: the subject is per-pixel sharpness and a DCT codec puts
        # its own texture exactly where we are looking.
        Image.fromarray(u8, "RGB").save(buf, format="PNG", compress_level=1)
        return web.Response(
            body=buf.getvalue(), content_type="image/png",
            headers={"Cache-Control": "no-store",
                     "X-Bat-Rescale": json.dumps(info)})

except ImportError as _exc:      # pragma: no cover - import-time only
    # No ComfyUI server: unit tests, or the pack imported standalone. The node
    # still works; the viewer just never gets its full-resolution layer.
    logger.debug("preview endpoints not registered (%s)", _exc)
except Exception as _exc:        # pragma: no cover - import-time only
    logger.warning("Bat_Rescale: could not register the preview endpoints; the "
                   "on-node viewer will stay on its draft layer: %s", _exc)
