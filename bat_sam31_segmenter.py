"""
Bat SAM 3.1 Segmenter — click or describe an object, see the mask live, track it.
=================================================================================

The SAM 3.1 counterpart of 🦇 SeC Segmenter, built around what SAM 3.1 does
better: a mask on every click in a few milliseconds, text prompts, and
multi-object tracking in one pass. Runs on ComfyUI core's own SAM 3 / 3.1
implementation — see bat_sam31_runtime.py.

One mask out per node (decided 2026-11-04). In click mode that is the one
object you clicked. In text mode it is the concept: every detection you keep,
plus — with ``detect_new_objects`` — anything matching the prompt that shows up
later, merged into one mask.

Measured against SeC on the same clip (A5000, 1080p, 120 frames): same masks
inside a shot (IoU 0.97-0.99), 6x faster load, 2.3 GB vs 8.7 GB VRAM. SeC still
wins across hard edit cuts — it re-finds the object, SAM 3.1 loses it — so SeC
stays in the pack for cut-heavy material.

Editor state travels in ONE hidden widget, ``prompt_state`` (JSON, normalised
0-1 coordinates so a proxy plate and the real frame agree):

    {"points": [{"x": .5, "y": .4, "label": 1}, ...],   # 1 object, 0 not
     "box": [x0, y0, x1, y1] | null,
     "instances": [{"box": [...], "keep": true}, ...]}  # text-mode choices

Live routes (all refuse with 409 while a job is executing — the executor owns
the GPU then):

    POST /bat/sam31/live    prompts -> mask PNG for the annotation frame
    POST /bat/sam31/detect  text    -> instances (box, score, mask PNG)
    GET  /bat/sam31/frame   a run's plate / mask, by token
    GET  /bat/sam31/info    is a run's batch still alive?

The node's ``ui`` payload carries only a run token, never pixels: frames are
fetched through /bat/sam31/frame on demand, so nothing image-sized lands in
ComfyUI's history.
"""

import asyncio
import base64
import io
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import bat_sam31_runtime as rt
from .bat_sec_runtime import MASK_PREVIEW_DEFAULT, MASK_PREVIEW_MODES, draw_mask_overlay

logger = logging.getLogger(__name__)

MODES = ["click", "text"]
TRACKING_DIRECTIONS = ["bidirectional", "forward", "backward"]

# Single source of truth for the advanced settings, mirrored in
# web/bat_sam31_segmenter.js (ADV_DEFAULTS) for the live routes.
DEFAULTS = {
    "model_file": None,              # None -> best SAM 3 checkpoint on disk
    "tracking_direction": "bidirectional",
    "detection_threshold": 0.5,
    "max_objects": 16,
    "detect_interval": 1,
    "max_frames_to_track": -1,
    "refine_iterations": rt.DEFAULT_REFINE_ITERATIONS,
}


def resolve(advanced):
    settings = dict(DEFAULTS)
    if advanced:
        settings.update({k: v for k, v in advanced.items() if k in DEFAULTS})
    return settings


def parse_state(raw):
    """prompt_state JSON -> dict with points / box / instances, tolerant of junk."""
    try:
        state = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        logger.warning("[Bat SAM3.1] prompt_state is not valid JSON — ignoring it")
        state = {}
    if not isinstance(state, dict):
        state = {}

    def unit(v):
        return min(1.0, max(0.0, float(v)))

    points = []
    for p in state.get("points") or []:
        try:
            points.append({"x": unit(p["x"]), "y": unit(p["y"]), "label": 1 if int(p.get("label", 1)) else 0})
        except (KeyError, TypeError, ValueError):
            continue
    box = state.get("box")
    try:
        box = [unit(v) for v in box] if box and len(box) == 4 else None
        if box and (abs(box[2] - box[0]) < 1e-4 or abs(box[3] - box[1]) < 1e-4):
            box = None
    except (TypeError, ValueError):
        box = None
    instances = [i for i in (state.get("instances") or []) if isinstance(i, dict) and "box" in i]
    return {"points": points, "box": box, "instances": instances}


def _mask_seed(input_mask, index, height, width):
    """Pick the annotation frame's mask out of an optional MASK input."""
    if input_mask is None:
        return None
    m = input_mask if input_mask.ndim == 3 else input_mask.unsqueeze(0)
    m = m[min(index, m.shape[0] - 1)].float()
    if m.shape != (height, width):
        m = F.interpolate(m[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    seed = m > 0.5
    return seed if seed.any() else None


class BatSam31Segmenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The clip to segment, as a frame batch."}),
                "mode": (MODES, {
                    "default": "click",
                    "tooltip": "click = place points / a box on the object, the mask updates as you "
                               "click. text = describe it ('person', 'red car'); every match shows on "
                               "the frame and you click a match to keep or drop it.",
                }),
                "text_prompt": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Text mode only. Comma-separate several things to find them all "
                               "('person, dog').",
                }),
                "frame_index_select": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "The frame you annotate. Tracking spreads out from it in both "
                               "directions (see the advanced node).",
                }),
                "detect_new_objects": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Text mode only. Keep looking for the prompt while tracking, so "
                               "things that enter later join the mask. Matches you dropped stay "
                               "dropped.",
                }),
                "mask_preview": (MASK_PREVIEW_MODES, {
                    "default": MASK_PREVIEW_DEFAULT,
                    "tooltip": "How the 'preview' output draws the mask over the clip. Display "
                               "only — never affects 'masks'.",
                }),
                # Editor state — hidden by the JS, the canvas is the interface.
                "prompt_state": ("STRING", {"default": "", "multiline": False, "advanced": True}),
            },
            "optional": {
                "input_mask": ("MASK", {
                    "tooltip": "Optional mask to seed the object instead of clicking, taken on "
                               "the annotation frame. Click mode only; clicks override it.",
                }),
                "advanced": ("SAM31_ADVANCED", {
                    "tooltip": "Optional 🦇 SAM 3.1 Advanced Params node. Unconnected = defaults.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "preview")
    FUNCTION = "segment"
    CATEGORY = "BAT/Segmentation"
    TITLE = "🦇 SAM 3.1 Segmenter"
    DESCRIPTION = """
## SAM 3.1 — click or describe, see it live, track it

**Click mode**
**Click** — the object · **Right-click** — not the object
**Drag** — a box · **Right-click its edge** — remove the box
**Click a point** — remove it

**Text mode**
Type what to find; every match shows on the frame. **Click a match** to keep
or drop it. `detect_new_objects` also picks up matches that appear later.

The mask on the canvas updates as you work. **▶ Track** runs the whole clip
straight away and fills the scrubber; queueing the workflow afterwards reuses
that result.

One mask out per node. Needs `sam3.1_multiplex_fp16.safetensors` in
`models/checkpoints` (https://huggingface.co/Comfy-Org/sam3.1).
"""

    def segment(self, frames, mode, text_prompt, frame_index_select, detect_new_objects,
                mask_preview=MASK_PREVIEW_DEFAULT, prompt_state="", input_mask=None, advanced=None):
        if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError("frames must be a non-empty [batch, height, width, channels] batch.")
        settings = resolve(advanced)
        state = parse_state(prompt_state)
        N, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        k = int(frame_index_select)
        if not 0 <= k < N:
            raise ValueError(f"frame_index_select {k} is out of range — this batch has {N} "
                             f"frame(s) (0-{N - 1}).")

        text = (text_prompt or "").strip()
        has_click = bool(state["points"] or state["box"] or input_mask is not None)
        if (mode == "text" and not text) or (mode != "text" and not has_click):
            # Nothing to track yet. Not an error: this is how the editor gets
            # its first picture when it can't preview the upstream node on its
            # own — run once, the clip arrives through the token, click on it.
            note = ("Type what to find in text_prompt, then ▶ Track" if mode == "text"
                    else "Clip loaded — click the object, then ▶ Track")
            print(f"[Bat SAM3.1] nothing to track yet — returning an empty mask ({note})")
            masks = torch.zeros(N, H, W)
            token = rt.RUNS.register(frames, masks, {"k": k, "mode": mode, "empty": True})
            ui = {"token": [token], "w": [W], "h": [H], "frame_count": [N], "annotation": [k],
                  "note": [note]}
            return {"ui": ui, "result": (masks, draw_mask_overlay(frames, masks, mask_preview))}

        model = rt.get_model(settings["model_file"])
        net, device, dtype = model.prepare()
        redetect = False
        with torch.inference_mode():
            encoded = rt.EncodedFrame(net, frames[k], device, dtype)
            if mode == "text":
                found = rt.detect_instances(model, net, encoded, text, settings["detection_threshold"],
                                            settings["max_objects"])
                keep = rt.match_instances(found, state["instances"])
                redetect = bool(detect_new_objects)
                if redetect:
                    # Track the dropped ones too: re-detection would otherwise
                    # re-add them as "new" on the very next frame.
                    seeds = [f["mask"] for f in found]
                    wanted = [i for i, kp in enumerate(keep) if kp]
                else:
                    seeds = [f["mask"] for f, kp in zip(found, keep) if kp]
                    wanted = list(range(len(seeds)))
                print(f"[Bat SAM3.1] '{text}': {len(found)} match(es) on frame {k}, "
                      f"{sum(keep)} kept{', re-detecting' if redetect else ''}")
            else:
                seed = rt.initial_mask_from_prompts(net, encoded, state["points"], state["box"],
                                                    settings["refine_iterations"])
                if seed is None:
                    seed = _mask_seed(input_mask, k, H, W)
                seeds, wanted = ([seed], [0]) if seed is not None else ([], [])
            del encoded

        init = torch.stack([s.to(device) for s in seeds]) if seeds else torch.zeros(0, H, W, dtype=torch.bool)
        masks = rt.track(model, frames, init, wanted, annotation_idx=k,
                         direction=settings["tracking_direction"],
                         max_frames=settings["max_frames_to_track"],
                         text=text if mode == "text" else None, redetect=redetect,
                         threshold=settings["detection_threshold"],
                         max_objects=settings["max_objects"],
                         detect_interval=settings["detect_interval"])

        token = rt.RUNS.register(frames, masks, {"k": k, "mode": mode})
        ui = {"token": [token], "w": [W], "h": [H], "frame_count": [N], "annotation": [k]}
        return {"ui": ui, "result": (masks, draw_mask_overlay(frames, masks, mask_preview))}


class BatSam31AdvancedParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (rt.model_choices(), {
                    "tooltip": "SAM 3 / 3.1 checkpoint from models/checkpoints. 3.1 multiplex "
                               "tracks many objects in one pass; plain SAM 3 can't re-detect "
                               "while tracking.",
                }),
                "tracking_direction": (TRACKING_DIRECTIONS, {
                    "default": DEFAULTS["tracking_direction"],
                    "tooltip": "bidirectional tracks both ways from the annotation frame. "
                               "forward/backward cover one side only — the other side stays empty.",
                }),
                "detection_threshold": ("FLOAT", {
                    "default": DEFAULTS["detection_threshold"], "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Text mode: how sure the detector must be. Lower finds more (and "
                               "more wrong things).",
                }),
                "max_objects": ("INT", {
                    "default": DEFAULTS["max_objects"], "min": 1, "max": 64,
                    "tooltip": "Text mode: cap on matches tracked at once, kept and dropped "
                               "together. 3.1 tracks them in one pass, so this costs little.",
                }),
                "detect_interval": ("INT", {
                    "default": DEFAULTS["detect_interval"], "min": 1, "max": 100,
                    "tooltip": "With detect_new_objects: look for newcomers every N frames. "
                               "Higher is faster.",
                }),
                "max_frames_to_track": ("INT", {
                    "default": DEFAULTS["max_frames_to_track"], "min": -1, "max": 999999,
                    "tooltip": "-1 = the whole batch. Otherwise this many frames BEYOND the "
                               "annotation frame, per direction; the rest stay empty.",
                }),
                "refine_iterations": ("INT", {
                    "default": DEFAULTS["refine_iterations"], "min": 0, "max": 5,
                    "tooltip": "Click mode: extra passes that tighten the first mask. 2 matches "
                               "ComfyUI's own SAM3 Detect. Applies to the live preview too.",
                }),
            }
        }

    RETURN_TYPES = ("SAM31_ADVANCED",)
    RETURN_NAMES = ("advanced",)
    FUNCTION = "build"
    CATEGORY = "BAT/Segmentation"
    DESCRIPTION = ("Optional settings for the 🦇 SAM 3.1 Segmenter. Without it the segmenter "
                   "uses these same defaults.")

    def build(self, model_file, tracking_direction, detection_threshold, max_objects,
              detect_interval, max_frames_to_track, refine_iterations):
        return ({
            "model_file": None if model_file == rt.NO_MODEL_SENTINEL else model_file,
            "tracking_direction": tracking_direction,
            "detection_threshold": float(detection_threshold),
            "max_objects": int(max_objects),
            "detect_interval": int(detect_interval),
            "max_frames_to_track": int(max_frames_to_track),
            "refine_iterations": int(refine_iterations),
        },)


class BatSam31TrackSink:
    """Internal: the output node the editor's ▶ Track button queues behind the
    segmenter.

    A prompt needs an output node to run, and making the segmenter itself an
    OUTPUT_NODE would make every branch it sits in unskippable. So the button
    sends a trimmed prompt — the segmenter, what feeds it, and this sink — as
    a normal queued job. ComfyUI's cache then hands the result to the next
    full queue run for free. Never placed on a canvas by hand.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ()
    FUNCTION = "sink"
    OUTPUT_NODE = True
    CATEGORY = "BAT/Segmentation"
    DEPRECATED = True          # keeps it out of node search
    DESCRIPTION = "Internal to 🦇 SAM 3.1 Segmenter's ▶ Track button. Not for direct use."

    def sink(self, masks):
        return {}


# ── live routes ─────────────────────────────────────────────────────────────

LIVE_MAX_DIM = 1024


def _busy():
    """True while ComfyUI is executing a prompt — the GPU is the executor's then."""
    try:
        import server
        return bool(server.PromptServer.instance.prompt_queue.currently_running)
    except Exception:
        return False


def _decode_image(b64):
    raw = base64.b64decode(b64.split(",", 1)[-1])
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)
    return arr, raw


def _source_frame(source):
    """Resolve a request's frame: a run's exact frame by token, else an uploaded plate.

    Returns (frame [H,W,C] float, cache key) or (None, reason).
    """
    source = source or {}
    token = str(source.get("token") or "")
    if token:
        idx = int(source.get("frame", 0))
        frame = rt.RUNS.frame(token, idx)
        if frame is not None:
            return frame, f"run:{token}:{idx}"
    if source.get("image"):
        frame, raw = _decode_image(source["image"])
        import hashlib
        return frame, "img:" + hashlib.sha1(raw).hexdigest()
    return None, "no frame: run the node once, or connect a clip the editor can preview"


def _encode(model, frame, key):
    net, device, dtype = model.prepare()
    cache_key = (model.name, key)
    enc = rt.ENCODED.get(cache_key)
    if enc is None or enc.device != device:
        enc = rt.EncodedFrame(net, frame, device, dtype)
        rt.ENCODED.put(cache_key, enc)
    return net, enc


def _mask_png(mask, height, width):
    """[H,W] bool -> base64 PNG (L), downscaled to the live preview size."""
    img = Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255, "L")
    scale = min(1.0, LIVE_MAX_DIM / max(height, width))
    if scale < 1.0:
        img = img.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def live_mask(body):
    frame, key = _source_frame(body.get("source"))
    if frame is None:
        return {"ok": False, "error": key}, 409
    state = parse_state(json.dumps({"points": body.get("points"), "box": body.get("box")}))
    t0 = time.perf_counter()
    with rt.GPU_LOCK, torch.inference_mode():
        model = rt.get_model(body.get("model") or None)
        net, enc = _encode(model, frame, key)
        t1 = time.perf_counter()
        mask = rt.initial_mask_from_prompts(net, enc, state["points"], state["box"],
                                            int(body.get("refine_iterations", rt.DEFAULT_REFINE_ITERATIONS)))
        if enc.device.type == "cuda":
            torch.cuda.synchronize(enc.device)
    t2 = time.perf_counter()
    if mask is None:
        return {"ok": True, "empty": True, "ms": round((t2 - t0) * 1000, 1)}, 200
    return {"ok": True, "mask": _mask_png(mask, enc.height, enc.width),
            "area": float(mask.float().mean()),
            "encode_ms": round((t1 - t0) * 1000, 1), "ms": round((t2 - t1) * 1000, 1)}, 200


def live_detect(body):
    frame, key = _source_frame(body.get("source"))
    if frame is None:
        return {"ok": False, "error": key}, 409
    text = str(body.get("text") or "").strip()
    if not text:
        return {"ok": True, "instances": []}, 200
    threshold = float(body.get("threshold", DEFAULTS["detection_threshold"]))
    max_objects = int(body.get("max_objects", DEFAULTS["max_objects"]))
    t0 = time.perf_counter()
    with rt.GPU_LOCK, torch.inference_mode():
        model = rt.get_model(body.get("model") or None)
        det_key = (model.name, key, text, round(threshold, 3), max_objects)
        found = rt.DETECTIONS.get(det_key)
        if found is None:
            net, enc = _encode(model, frame, key)
            found = rt.detect_instances(model, net, enc, text, threshold, max_objects)
            found = [{"box": f["box"], "score": f["score"],
                      "mask": _mask_png(f["mask"], enc.height, enc.width)} for f in found]
            rt.DETECTIONS.put(det_key, found)
    return {"ok": True, "instances": found, "ms": round((time.perf_counter() - t0) * 1000, 1)}, 200


def run_frame(token, index, kind, max_dim):
    """A run's plate (JPEG) or mask (PNG) at most max_dim on the long edge."""
    entry = rt.RUNS.get(token)
    if entry is None:
        return None, None
    if kind == "mask":
        m = rt.RUNS.mask(token, index)
        if m is None:
            return None, None
        img = Image.fromarray(m.astype(np.uint8) * 255, "L")
        fmt, ctype = "PNG", "image/png"
    else:
        f = rt.RUNS.frame(token, index)
        if f is None:
            return None, None
        arr = (f[..., :3].float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        fmt, ctype = "JPEG", "image/jpeg"
    scale = min(1.0, max_dim / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {"compress_level": 1}))
    return buf.getvalue(), ctype


try:
    import server
    from aiohttp import web

    def _json(payload, status=200):
        return web.json_response(payload, status=status, headers={"Cache-Control": "no-store"})

    async def _run_live(request, fn):
        if _busy():
            return _json({"ok": False, "busy": True, "error": "a job is running — live preview "
                          "resumes when it finishes"}, 409)
        try:
            body = await request.json()
        except Exception:
            return _json({"ok": False, "error": "bad json"}, 400)
        try:
            # Off the event loop: an encode is ~120 ms of GPU work and would
            # stall every websocket message meanwhile.
            payload, status = await asyncio.get_running_loop().run_in_executor(None, fn, body)
        except FileNotFoundError as exc:
            return _json({"ok": False, "error": str(exc)}, 404)
        except Exception as exc:
            logger.warning("[Bat SAM3.1] live request failed: %s", exc, exc_info=True)
            return _json({"ok": False, "error": str(exc)}, 500)
        return _json(payload, status)

    @server.PromptServer.instance.routes.post("/bat/sam31/live")
    async def _bat_sam31_live(request):
        return await _run_live(request, live_mask)

    @server.PromptServer.instance.routes.post("/bat/sam31/detect")
    async def _bat_sam31_detect(request):
        return await _run_live(request, live_detect)

    @server.PromptServer.instance.routes.get("/bat/sam31/info")
    async def _bat_sam31_info(request):
        entry = rt.RUNS.get(request.rel_url.query.get("token", ""))
        if entry is None:
            return _json({"ok": False})
        return _json({"ok": True, "count": entry["count"], "w": entry["width"], "h": entry["height"],
                      "live_frames": entry["frames"]() is not None, "has_masks": entry["masks"] is not None,
                      "annotation": entry["meta"].get("k", 0)})

    @server.PromptServer.instance.routes.get("/bat/sam31/frame")
    async def _bat_sam31_frame(request):
        q = request.rel_url.query
        try:
            index = int(q.get("i", 0))
            max_dim = max(64, min(4096, int(q.get("max", LIVE_MAX_DIM))))
        except ValueError:
            return _json({"ok": False, "error": "bad index"}, 400)
        data, ctype = await asyncio.to_thread(run_frame, q.get("token", ""), index,
                                              q.get("kind", "plate"), max_dim)
        if data is None:
            return _json({"ok": False, "error": "gone — run the node again"}, 410)
        return web.Response(body=data, content_type=ctype, headers={"Cache-Control": "no-store"})

except ImportError as _exc:      # pragma: no cover — no server (tests, standalone import)
    logger.debug("SAM 3.1 live routes not registered (%s)", _exc)
except Exception as _exc:        # pragma: no cover — server module without a running instance
    logger.warning("[Bat SAM3.1] live routes not registered; the editor will only update "
                   "after a run: %s", _exc)
