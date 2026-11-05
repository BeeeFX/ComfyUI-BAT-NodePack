"""
SAM 3.1 runtime — model cache, live prompt decoding, text detection, tracking.
=============================================================================

BAT's own code on top of ComfyUI core's native SAM 3 / 3.1 implementation
(``comfy/ldm/sam3``). Nothing is vendored and nothing is pip-installed: the
checkpoint loads through ``comfy.sd`` like any other model, so core's model
management owns its VRAM.

What this adds over core's own SAM3 nodes, which only track from frame 0,
forwards, from a mask or a text prompt:

  * points / box prompts on ANY frame, tracked both ways from it
    (two ``forward_video`` passes: frames[k:], and frames[:k+1] reversed);
  * a cached per-frame encode so a click costs one decoder pass (~6 ms on an
    A5000) instead of a backbone pass (~120 ms) — that is the live preview;
  * text detections the artist can keep or drop one by one, still tracked
    together, with optional re-detection of newcomers.

Two private core entry points are used, both on purpose and both guarded:
``tracker._forward_sam_heads`` (decoder on cached features) and
``comfy_extras.nodes_sam3._extract_text_prompts`` / ``_refine_mask``. If a core
update renames them, the live path falls back to the public
``forward_segment`` (slower, same result) and text mode raises a clear error.

The SAME functions build the initial mask for the live preview and for the
real run (``initial_mask_from_prompts`` / ``detect_instances``), so what the
artist sees while clicking is exactly what gets tracked — provided the plate
they clicked on was the real frame (see ``FrameSource``).
"""

import hashlib
import logging
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

SAM_SIZE = 1008                 # core resizes every frame to 1008x1008 (squash, no crop)
DEFAULT_MODEL = "sam3.1_multiplex_fp16.safetensors"
NO_MODEL_SENTINEL = "(no SAM 3 checkpoint in models/checkpoints)"

# Matches core's SAM3_Detect default: one extra mask-refine pass after the
# prompt pass. Used for BOTH the live preview and the run's seed mask.
DEFAULT_REFINE_ITERATIONS = 2

# Text detections are NOT refined. Core's _refine_mask re-runs the backbone on
# a crop per instance: measured on 11 people in a 1080p frame (A5000), 190 ms
# unrefined vs 2.9 s at 2 iterations, for masks that agree at IoU 0.976-0.999.
# That is the difference between "updates as you type" and "wait". The run
# uses the same setting, so the live preview stays identical to what is seeded.
TEXT_REFINE_ITERATIONS = 0


# ── model discovery + cache ─────────────────────────────────────────────────

def list_models():
    """SAM 3 / 3.1 checkpoints in models/checkpoints, best first.

    Ranked: 3.1 before 3, then fp16 before the rest (same weights, half the
    VRAM), then by name. Matching on the file name is enough because core
    itself identifies the architecture from the state dict at load time.
    """
    names = [n for n in folder_paths.get_filename_list("checkpoints")
             if "sam3" in n.lower().replace("_", "").replace("-", "").replace(" ", "")]

    def rank(n):
        low = n.lower()
        return ("sam3.1" not in low and "sam31" not in low, "fp16" not in low, low)
    return sorted(names, key=rank)


def model_choices():
    return list_models() or [NO_MODEL_SENTINEL]


_MODELS = {}
_MODEL_LOCK = threading.RLock()


class Sam31Model:
    """A loaded checkpoint: the ModelPatcher, its text encoder, and helpers."""

    def __init__(self, name, patcher, clip):
        self.name = name
        self.patcher = patcher
        self.clip = clip
        self._text_cache = OrderedDict()

    @property
    def net(self):
        return self.patcher.model.diffusion_model

    @property
    def multiplex(self):
        return bool(getattr(self.net.detector.backbone["vision_backbone"], "multiplex", False))

    def prepare(self):
        """Put the weights on the GPU (core may have offloaded them) and
        return (net, device, dtype)."""
        mm.load_model_gpu(self.patcher)
        return self.net, mm.get_torch_device(), self.patcher.model.get_dtype()

    def text_prompts(self, text, device, dtype):
        """Encode a comma-separated prompt -> [(embedding, attention_mask), ...].

        Cached per text: the encoder is small but it runs on every keystroke of
        the live text preview.
        """
        text = (text or "").strip()
        if not text:
            return []
        if self.clip is None:
            raise RuntimeError(f"{self.name} has no text encoder, so text mode can't run.")
        try:
            from comfy_extras.nodes_sam3 import _extract_text_prompts
        except ImportError as exc:      # core renamed its helper
            raise RuntimeError("This ComfyUI build has no SAM3 text-prompt helper "
                               "(comfy_extras.nodes_sam3._extract_text_prompts).") from exc

        cond = self._text_cache.get(text)
        if cond is None:
            tokens = self.clip.tokenize(text)
            cond = self.clip.encode_from_tokens_scheduled(tokens)
            self._text_cache[text] = cond
            while len(self._text_cache) > 32:
                self._text_cache.popitem(last=False)
        # The ":N" max-detections suffix core parses is ignored on purpose:
        # the node keeps every instance above the threshold and lets the
        # artist choose, capped by max_objects.
        return [(emb, mask) for emb, mask, _ in _extract_text_prompts(cond, device, dtype)]


def get_model(name=None):
    """Load (or reuse) a SAM 3 / 3.1 checkpoint from models/checkpoints."""
    if not name or name == NO_MODEL_SENTINEL:
        available = list_models()
        if not available:
            raise FileNotFoundError(
                "No SAM 3 checkpoint found. Put sam3.1_multiplex_fp16.safetensors "
                "(https://huggingface.co/Comfy-Org/sam3.1) in models/checkpoints.")
        name = available[0]
    path = folder_paths.get_full_path_or_raise("checkpoints", name)

    with _MODEL_LOCK:
        cached = _MODELS.get(path)
        if cached is not None:
            return cached
        patcher, clip, _vae, _cv = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=False, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        net = patcher.model.diffusion_model
        if not hasattr(net, "forward_video") or not hasattr(net, "forward_segment"):
            raise ValueError(f"{name} is not a SAM 3 checkpoint (core loaded it as "
                             f"{type(net).__name__}).")
        model = Sam31Model(name, patcher, clip)
        _MODELS[path] = model
        logger.info("[Bat SAM3.1] loaded %s (%s)", name,
                    "multiplex" if model.multiplex else "SAM 3, no multiplex")
        return model


# ── frames ──────────────────────────────────────────────────────────────────

def to_sam_input(frames_hwc, device, dtype):
    """[B,H,W,C] float 0-1 -> [B,3,1008,1008] on device, the way core does it."""
    x = comfy.utils.common_upscale(frames_hwc[..., :3].movedim(-1, 1), SAM_SIZE, SAM_SIZE,
                                   "bilinear", crop="disabled")
    return x.to(device=device, dtype=dtype)


def logits_to_mask(logits, height, width):
    """[1,1,h,w] logits -> [H,W] bool at frame size."""
    up = F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    return up[0, 0] > 0


# ── live preview: encode once, decode per click ─────────────────────────────

class EncodedFrame:
    """Backbone output for one frame, ready for repeated decoder passes."""

    def __init__(self, net, frame_hwc, device, dtype):
        self.height, self.width = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        self.frame_hwc = frame_hwc.detach().float().cpu()   # kept for text-mask refinement
        self.image = to_sam_input(frame_hwc.unsqueeze(0), device, dtype)
        self.device, self.dtype = device, dtype
        bb = net.detector.backbone["vision_backbone"]
        self._fast = None
        try:
            if bb.multiplex:
                _, _, tf, _tp = bb(self.image, tracker_mode="interactive")
            else:
                _, _, tf, _tp = bb(self.image, need_tracker=True)
                scalp = getattr(net.detector, "scalp", 0)
                if scalp > 0:
                    tf = tf[:-scalp]
            high_res = list(tf[:-1])
            feat = tf[-1]
            b, c, h, w = feat.shape
            no_mem = getattr(net.tracker, "interactivity_no_mem_embed", None)
            if no_mem is None:
                no_mem = getattr(net.tracker, "no_mem_embed", None)
            if no_mem is not None:
                flat = feat.flatten(2).permute(0, 2, 1)
                feat = (flat + no_mem.to(flat)).view(b, h, w, c).permute(0, 3, 1, 2)
            if hasattr(net.tracker, "_forward_sam_heads"):
                self._fast = (feat, high_res)
        except Exception as exc:     # core internals moved: use the public path
            logger.warning("[Bat SAM3.1] cached-encode path unavailable (%s); "
                           "live preview will run the full model per click", exc)
            self._fast = None

    def decode(self, net, point_inputs=None, box_inputs=None, mask_inputs=None):
        """One SAM decoder pass. Same maths as core's forward_segment."""
        if self._fast is None:
            return net.forward_segment(self.image, point_inputs=point_inputs,
                                       box_inputs=box_inputs, mask_inputs=mask_inputs)
        feat, high_res = self._fast
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        _, high_res_masks, _, _ = net.tracker._forward_sam_heads(
            backbone_features=feat, point_inputs=point_inputs, mask_inputs=mask_inputs,
            box_inputs=box_inputs, high_res_features=high_res,
            multimask_output=(0 < num_pts <= 1 and box_inputs is None))
        return high_res_masks


def initial_mask_from_prompts(net, encoded, points, box, refine_iterations=DEFAULT_REFINE_ITERATIONS):
    """Points/box (normalised 0-1, frame space) -> [H,W] bool, or None.

    ``points`` is a list of {"x","y","label"} with label 1 = object, 0 = not.
    ``box`` is [x0,y0,x1,y1] or None. Coordinates are normalised so a plate
    proxy and the real frame give the same prompt.
    """
    pts = [p for p in (points or []) if p.get("label") in (0, 1)]
    if not pts and not box:
        return None
    dev, dt = encoded.device, encoded.dtype
    point_inputs = None
    if pts:
        point_inputs = {
            "point_coords": torch.tensor([[[float(p["x"]) * SAM_SIZE, float(p["y"]) * SAM_SIZE] for p in pts]],
                                         dtype=dt, device=dev),
            "point_labels": torch.tensor([[int(p["label"]) for p in pts]], dtype=torch.int32, device=dev),
        }
    box_inputs = None
    if box:
        x0, y0, x1, y1 = (float(v) * SAM_SIZE for v in box)
        box_inputs = torch.tensor([[[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]]],
                                  dtype=dt, device=dev)

    logits = encoded.decode(net, point_inputs=point_inputs, box_inputs=box_inputs)
    for _ in range(max(0, int(refine_iterations) - 1)):
        logits = encoded.decode(net, mask_inputs=logits)
    return logits_to_mask(logits, encoded.height, encoded.width)


def detect_instances(model, net, encoded, text, threshold, max_objects=0,
                     refine_iterations=TEXT_REFINE_ITERATIONS):
    """Run the text detector on one encoded frame.

    Returns a list of {"box": [x0,y0,x1,y1] normalised, "score": float,
    "mask": [H,W] bool}, best score first, NMS-free (the detector already
    de-duplicates), capped at ``max_objects`` (0 = no cap).
    """
    try:
        from comfy_extras.nodes_sam3 import _refine_mask
    except ImportError:
        _refine_mask = None

    prompts = model.text_prompts(text, encoded.device, encoded.dtype)
    H, W = encoded.height, encoded.width
    found = []
    for emb, mask in prompts:
        res = net(encoded.image, text_embeddings=emb, text_mask=mask, threshold=threshold,
                  orig_size=(H, W))
        probs = res["scores"][0].sigmoid()
        keep = probs > threshold
        boxes, scores, masks = res["boxes"][0][keep], probs[keep], res["masks"][0][keep]
        for b, s, m in zip(boxes, scores, masks):
            if _refine_mask is not None:
                full = _refine_mask(net, encoded.frame_hwc.to(encoded.device), m, b, H, W,
                                    encoded.device, encoded.dtype, refine_iterations)[0] > 0
            else:
                full = F.interpolate(m[None, None].float(), size=(H, W), mode="bilinear",
                                     align_corners=False)[0, 0] > 0
            x0, y0, x1, y1 = (float(v) for v in b.tolist())
            found.append({"box": [x0 / W, y0 / H, x1 / W, y1 / H], "score": float(s),
                          "mask": full})
    found.sort(key=lambda d: -d["score"])
    if max_objects and max_objects > 0:
        found = found[:max_objects]
    return found


def match_instances(found, saved):
    """Carry the artist's keep/drop choices onto a fresh detection.

    ``saved`` is the editor's list of {"box": [...], "keep": bool}. Detection
    is deterministic for the same frame + prompt + threshold, but a proxy
    plate or a nudged threshold shifts boxes slightly, so match by box IoU
    rather than by index. Unmatched detections default to kept.
    """
    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    keep = []
    for inst in found:
        best, best_iou = None, 0.0
        for s in saved or []:
            v = iou(inst["box"], s.get("box", [0, 0, 0, 0]))
            if v > best_iou:
                best, best_iou = s, v
        keep.append(bool(best.get("keep", True)) if (best is not None and best_iou >= 0.5) else True)
    return keep


# ── tracking ────────────────────────────────────────────────────────────────

def _track_pass(net, frames_hwc, init_masks, keep_idx, device, dtype, text_prompts, threshold,
                max_objects, detect_interval, pbar):
    """One forward_video call -> [n,H,W] float union of the wanted objects.

    Objects 0..K-1 are the seeds (``keep_idx`` picks which count); anything
    the detector added after them always counts. The union is taken on the
    bit-packed masks before unpacking, the way core's SAM3_TrackToMask does —
    unpacking every object at frame size first would be K full-res batches.
    """
    from comfy.ldm.sam3.tracker import unpack_masks
    n, H, W = frames_hwc.shape[0], frames_hwc.shape[1], frames_hwc.shape[2]
    K = int(init_masks.shape[0])
    res = net.forward_video(
        images=frames_hwc[..., :3].movedim(-1, 1),
        initial_masks=(init_masks.unsqueeze(1).to(device=device, dtype=dtype) if K else None),
        pbar=pbar, text_prompts=text_prompts or None, new_det_thresh=threshold,
        max_objects=max_objects, detect_interval=detect_interval,
        target_device=device, target_dtype=dtype)
    packed = res.get("packed_masks")
    out = torch.zeros(n, H, W)
    if packed is None:
        return out
    sel = [i for i in keep_idx if i < packed.shape[1]] + list(range(K, packed.shape[1]))
    if not sel:
        return out
    union = packed[:, sel[0]].clone()
    for i in sel[1:]:
        union |= packed[:, i]
    # Chunked: one [n,1,H,W] float at 4K is several GB.
    for s in range(0, n, 16):
        m = unpack_masks(union[s:s + 16]).unsqueeze(1).float()
        out[s:s + 16] = (F.interpolate(m, size=(H, W), mode="bilinear",
                                       align_corners=False)[:, 0] > 0.5).float().cpu()
    return out


def track(model, frames, init_masks, output_objects, annotation_idx=0,
          direction="bidirectional", max_frames=-1, text=None, redetect=False,
          threshold=0.5, max_objects=16, detect_interval=1):
    """Track ``init_masks`` ([K,H,W] bool, on the annotation frame) through ``frames``.

    ``output_objects`` lists which of the K seed objects end up in the output
    mask (text mode tracks dropped instances too, so re-detection doesn't
    re-add them, and leaves them out here). Objects the detector adds later
    are always included. Returns [N,H,W] float.
    """
    N, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if not 0 <= annotation_idx < N:
        raise ValueError(f"frame_index_select {annotation_idx} is out of range — this batch has "
                         f"{N} frame(s) (0-{N - 1}).")
    out = torch.zeros(N, H, W)
    K = 0 if init_masks is None else int(init_masks.shape[0])
    # Nothing seeded is only worth a run when re-detection can find something.
    if K == 0 and not (redetect and text):
        return out
    if init_masks is None:
        init_masks = torch.zeros(0, H, W, dtype=torch.bool)

    net, device, dtype = model.prepare()
    text_prompts = model.text_prompts(text, device, dtype) if (redetect and text) else None
    if text_prompts and not model.multiplex:
        logger.warning("[Bat SAM3.1] %s is SAM 3, not 3.1: re-detection during tracking needs "
                       "the multiplex model, so it is off for this run.", model.name)
        text_prompts = None
    if K == 0 and not text_prompts:
        return out
    keep_idx = sorted(set(int(i) for i in output_objects if 0 <= int(i) < K))

    span = N if max_frames is None or max_frames < 0 else int(max_frames) + 1
    fwd = (annotation_idx, min(N, annotation_idx + span))
    bwd = (max(0, annotation_idx - span + 1), annotation_idx + 1)
    todo = 0
    if direction in ("bidirectional", "forward"):
        todo += fwd[1] - fwd[0]
    if direction in ("bidirectional", "backward") and annotation_idx > 0:
        todo += bwd[1] - bwd[0]
    pbar = comfy.utils.ProgressBar(max(1, todo))

    with torch.inference_mode():
        if direction in ("bidirectional", "forward"):
            out[fwd[0]:fwd[1]] = _track_pass(
                net, frames[fwd[0]:fwd[1]], init_masks, keep_idx, device, dtype,
                text_prompts, threshold, max_objects, detect_interval, pbar)
        if direction in ("bidirectional", "backward") and (annotation_idx > 0 or direction == "backward"):
            # Reversed copy of only the backward span — the forward pass above
            # owns the annotation frame when both run.
            rev = torch.flip(frames[bwd[0]:bwd[1]], dims=[0])
            back = torch.flip(_track_pass(
                net, rev, init_masks, keep_idx, device, dtype,
                text_prompts, threshold, max_objects, detect_interval, pbar), dims=[0])
            del rev
            if direction == "bidirectional":
                out[bwd[0]:annotation_idx] = back[:-1]
            else:
                out[bwd[0]:bwd[1]] = back
    return out


# ── frame sources for the live routes ───────────────────────────────────────

class FrameSource:
    """Registry of run tokens -> the frame batch that run saw.

    The batch is held by WEAK reference: it is the upstream node's cached
    output, so while ComfyUI keeps it we can hand the editor exact frames at
    no memory cost, and once ComfyUI drops it we let it go too. The masks
    from the run are small (bit-packed) and held strongly so the scrubber can
    still show them over a plate fetched elsewhere.
    """

    MAX_TOKENS = 8

    def __init__(self):
        self._lock = threading.Lock()
        self._runs = OrderedDict()

    def register(self, frames, masks, meta):
        import weakref
        token = hashlib.sha1(f"{id(frames)}:{meta}:{np.random.rand()}".encode()).hexdigest()[:16]
        try:
            ref = weakref.ref(frames)
        except TypeError:
            ref = lambda f=frames: f      # noqa: E731 — not weakref-able: hold it
        packed = None
        if masks is not None:
            packed = np.packbits(masks.cpu().numpy() > 0.5, axis=-1)
        with self._lock:
            self._runs[token] = {"frames": ref, "masks": packed,
                                 "width": int(frames.shape[2]), "height": int(frames.shape[1]),
                                 "count": int(frames.shape[0]), "meta": meta}
            while len(self._runs) > self.MAX_TOKENS:
                self._runs.popitem(last=False)
        return token

    def get(self, token):
        with self._lock:
            entry = self._runs.get(token)
            if entry is not None:
                self._runs.move_to_end(token)
            return entry

    def frame(self, token, index):
        """Exact frame [H,W,C] float from a run, or None once ComfyUI dropped it."""
        entry = self.get(token)
        if entry is None:
            return None
        frames = entry["frames"]()
        if frames is None or not 0 <= index < frames.shape[0]:
            return None
        return frames[index]

    def mask(self, token, index):
        entry = self.get(token)
        if entry is None or entry["masks"] is None or not 0 <= index < entry["count"]:
            return None
        return np.unpackbits(entry["masks"][index], axis=-1)[..., :entry["width"]].astype(bool)


RUNS = FrameSource()


class EncodeCache:
    """A few encoded frames, keyed by where the pixels came from."""

    MAX = 4

    def __init__(self):
        self._lock = threading.Lock()
        self._items = OrderedDict()

    def get(self, key):
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items.move_to_end(key)
            return item

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.MAX:
                self._items.popitem(last=False)

    def clear(self):
        with self._lock:
            self._items.clear()


ENCODED = EncodeCache()
DETECTIONS = EncodeCache()

# One GPU job at a time from the live routes; the executor has its own thread
# and the routes refuse to run while it is busy (see the segmenter module).
GPU_LOCK = threading.Lock()
