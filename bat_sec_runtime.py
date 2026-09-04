"""
SeC runtime — model discovery, loading, caching and the segmentation core.
=========================================================================

BAT's own code. Sits on top of the vendored third-party stack in ``bat_sec/``
(see ``bat_sec/NOTICE.md``) and is consumed by ``bat_sec_segmenter.py``.

Ported from ``Comfyui-SecNodes/nodes.py`` (Apache 2.0) with four deliberate
behavioural changes — see the block comments at each site:

  1. Model lifetime is a module-level cache instead of the upstream
     unload-in-place / resurrect-attributes dance (``_MODEL_CACHE``).
  2. bbox + points are sent to SAM2 in ONE call so the box survives
     (``_apply_prompts``). Upstream's two-call version silently dropped it.
  3. RGBA / grayscale / 4-channel frame batches are coerced to RGB instead of
     raising inside ``Image.fromarray`` (``frames_to_pil``).
  4. Frame bytes are rounded rather than truncated (``frames_to_pil``).
"""

import gc
import os
import shutil
import tempfile
import threading

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

import folder_paths

# Vendored SeC stack. Imported lazily inside the loader — importing
# modeling_sec at module scope drags transformers + the whole SAM2 tree into
# every ComfyUI startup, which costs seconds even when no SeC node is used.

MODEL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bat_sec", "model_config")

# Single-file checkpoints, highest priority first. The order sets which one a
# fresh node defaults to when several are present.
_SINGLE_FILE_MODELS = [
    ("SeC-4B-fp16.safetensors", "fp16"),
    ("SeC-4B-bf16.safetensors", "bf16"),
    ("SeC-4B-fp32.safetensors", "fp32"),
    ("SeC-4B-fp8.safetensors", "fp8"),
]

NO_MODEL_SENTINEL = "(no SeC model found — put SeC-4B-fp16.safetensors in models/sams)"

_DTYPE_BY_PRECISION = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    # FP8 checkpoints are loaded and run as FP16. Upstream removed FP8
    # quantization outright: torchao's int8_weight_only produced NaNs in the
    # language model during scene-change detection, which silently destroyed
    # semantic tracking. Quantizing the vision tower alone saved ~300MB (6%)
    # and wasn't worth it.
    "fp8": torch.float16,
}


# ── model discovery ─────────────────────────────────────────────────────────


def list_sec_models():
    """Every SeC checkpoint visible through ComfyUI's ``sams`` folder type.

    Returns dicts of ``{name, path, is_single_file, config_path, precision}``.
    ``config_path`` differs by layout: single-file checkpoints carry weights
    only, so config + tokenizer come from the vendored ``bat_sec/model_config``;
    a sharded HF directory carries its own.
    """
    found = []
    try:
        sams_dirs = folder_paths.get_folder_paths("sams")
    except KeyError:
        return found

    for sams_dir in sams_dirs:
        for filename, precision in _SINGLE_FILE_MODELS:
            path = os.path.join(sams_dir, filename)
            if os.path.isfile(path):
                found.append({
                    "name": filename,
                    "path": path,
                    "is_single_file": True,
                    "config_path": MODEL_CONFIG_DIR,
                    "precision": precision,
                })

        shard_dir = os.path.join(sams_dir, "SeC-4B")
        if os.path.isdir(shard_dir):
            has_config = os.path.isfile(os.path.join(shard_dir, "config.json"))
            has_weights = any(
                os.path.isfile(os.path.join(shard_dir, f))
                for f in ("model.safetensors", "model.safetensors.index.json",
                          "pytorch_model.bin", "pytorch_model.bin.index.json")
            )
            has_tokenizer = os.path.isfile(os.path.join(shard_dir, "tokenizer_config.json"))
            if has_config and has_weights and has_tokenizer:
                found.append({
                    "name": "SeC-4B (sharded)",
                    "path": shard_dir,
                    "is_single_file": False,
                    "config_path": shard_dir,
                    "precision": "fp16",
                })

    return found


def model_choices():
    """Dropdown entries for the model_file widget. Never empty — an empty
    combo box renders as a dead widget with no way to tell the user why."""
    names = [m["name"] for m in list_sec_models()]
    return names or [NO_MODEL_SENTINEL]


def device_choices():
    choices = ["auto", "cpu"]
    if torch.cuda.is_available():
        choices += [f"gpu{i}" for i in range(torch.cuda.device_count())]
    return choices


# ── auto-download ───────────────────────────────────────────────────────────
#
# The node is meant to work with nothing wired to it, which means a missing
# checkpoint has to fix itself. Upstream SecNodes raised with README
# instructions instead; that's a dead end for someone who just dropped the node
# on a canvas.
#
# fp16 is the download target: same quality as bf16, half the size of fp32, and
# the precision upstream recommends. FP8 is deliberately never fetched — it
# loads as fp16 anyway and its quantized path produced NaNs.

HF_REPO_ID = "VeryAladeen/Sec-4B"
HF_DEFAULT_FILE = "SeC-4B-fp16.safetensors"
HF_DEFAULT_SIZE_GB = 7.35

_DOWNLOAD_LOCK = threading.Lock()


def _sams_target_dir():
    """First writable ``sams`` folder, creating it if needed."""
    try:
        candidates = list(folder_paths.get_folder_paths("sams"))
    except KeyError:
        candidates = []
    candidates.append(os.path.join(folder_paths.models_dir, "sams"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                return path
        except OSError:
            continue
    return None


def download_default_model(filename=HF_DEFAULT_FILE):
    """Fetch a single-file SeC checkpoint into models/sams. Returns its path.

    Serialized on a module lock so two queued prompts can't race each other
    into the same 7GB download.
    """
    with _DOWNLOAD_LOCK:
        # Another thread may have finished while we waited for the lock.
        for spec in list_sec_models():
            if spec["name"] == filename:
                return spec["path"]

        target_dir = _sams_target_dir()
        if target_dir is None:
            raise RuntimeError(
                "No writable models/sams folder to download the SeC model into. "
                "Create one, or add a writable 'sams' path to extra_model_paths.yaml."
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is needed to auto-download the SeC model "
                "(pip install huggingface_hub), or download it manually — see below.\n"
                + _manual_instructions(target_dir)
            ) from e

        print(f"\n{'=' * 72}")
        print(f"[Bat SeC] No SeC checkpoint found — downloading one now.")
        print(f"[Bat SeC]   {HF_REPO_ID} / {filename}  (~{HF_DEFAULT_SIZE_GB:.2f} GB)")
        print(f"[Bat SeC]   into {target_dir}")
        print(f"[Bat SeC] One-off; it is resumable and cached. This will take a while.")
        print(f"{'=' * 72}\n")

        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=target_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"Automatic download of the SeC model failed: {e}\n\n"
                + _manual_instructions(target_dir)
            ) from e

        print(f"[Bat SeC] downloaded -> {path}")
        return path


def _manual_instructions(target_dir=None):
    target_dir = target_dir or os.path.join(folder_paths.models_dir, "sams")
    return (
        "Download it manually instead:\n"
        f"  huggingface-cli download {HF_REPO_ID} {HF_DEFAULT_FILE} --local-dir {target_dir}\n"
        f"or grab it from https://huggingface.co/{HF_REPO_ID} and drop it in:\n"
        f"  {target_dir}\n"
        "Then reload the ComfyUI frontend."
    )


# ── device ──────────────────────────────────────────────────────────────────


def resolve_device(device):
    """``auto``/``cpu``/``gpuN`` -> a torch device string."""
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device.startswith("gpu"):
        suffix = device[3:]
        if not suffix.isdigit():
            raise ValueError(f"Invalid GPU device '{device}'. Expected gpu0, gpu1, ...")
        index = int(suffix)
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available but device '{device}' was selected.")
        count = torch.cuda.device_count()
        if index >= count:
            raise ValueError(
                f"GPU {index} not available — this machine has {count} GPU(s) (gpu0-gpu{count - 1})."
            )
        return f"cuda:{index}"
    raise ValueError(f"Unrecognised device '{device}'.")


# ── model cache ─────────────────────────────────────────────────────────────
#
# CHANGE vs upstream (1/4): upstream kept the model alive across runs by
# gutting it in place (`delattr` every submodule, set `_sec_unloaded = True`)
# and later resurrecting it with
#
#     for attr in dir(fresh_model): setattr(model, attr, getattr(fresh_model, attr))
#
# That copies *bound methods* off `fresh_model`, so the subsequent
# `del fresh_model` frees nothing — every unload/reload cycle chained another
# live 4B husk onto the previous one. It only existed because the loader was a
# separate node whose output object ComfyUI cached and handed back.
#
# Merging the loader into the node removes the constraint: "reload" is just a
# cache miss. Load fresh, keep it keyed by everything that affects
# construction, and on unload drop the reference and let Python free it.
_MODEL_CACHE = {}
_CACHE_LOCK = threading.RLock()


def _cache_key(spec, device, use_flash_attn, allow_mask_overlap):
    return (spec["path"], spec["precision"], device, bool(use_flash_attn), bool(allow_mask_overlap))


def unload_all(reason=""):
    """Drop every cached model and give the memory back."""
    with _CACHE_LOCK:
        if _MODEL_CACHE:
            print(f"[Bat SeC] unloading {len(_MODEL_CACHE)} cached model(s){f' ({reason})' if reason else ''}")
        _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _register_dtype_hooks(model, torch_dtype):
    """Cast float inputs to each submodule's own parameter dtype.

    The SeC graph mixes dtypes (the SAM2 tower and the LLM don't agree), so
    without this a mismatched matmul aborts inference. Integer tensors are
    passed through untouched — they're token ids and indices, and casting them
    to fp16 corrupts the embedding lookup. Embedding modules are skipped for
    the same reason.
    """
    def hook(module, args, kwargs):
        try:
            module_dtype = next((p.dtype for p in module.parameters()), None)
            if module_dtype is None or isinstance(module, torch.nn.Embedding):
                return args, kwargs

            def coerce(v):
                if not isinstance(v, torch.Tensor):
                    return v
                if v.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    return v
                return v.to(dtype=module_dtype) if v.dtype != module_dtype else v

            return tuple(coerce(a) for a in args), {k: coerce(v) for k, v in kwargs.items()}
        except Exception:
            # A hook that raises kills the whole forward pass. Passing the
            # args through unchanged is always safe — worst case the dtype
            # mismatch surfaces where it would have without the hook.
            return args, kwargs

    for module in model.modules():
        if any(True for _ in module.parameters(recurse=False)):
            module.register_forward_pre_hook(hook, with_kwargs=True)


def _preflight_dependencies():
    """Name every missing package at once instead of one ImportError at a time.

    The SeC stack is imported lazily so a machine without these still loads the
    pack and every other BAT node. Only this node fails — so it has to say why
    in a way that's actionable without reading a traceback.
    """
    import importlib.util

    required = {
        "transformers": "transformers", "timm": "timm", "peft": "peft",
        "hydra": "hydra-core", "omegaconf": "omegaconf", "einops": "einops",
        "cv2": "opencv-python-headless", "iopath": "iopath",
    }
    missing = sorted({pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None})
    if missing:
        here = os.path.dirname(os.path.abspath(__file__))
        raise RuntimeError(
            "The SeC segmenter needs packages this environment doesn't have: "
            + ", ".join(missing)
            + f"\n\nInstall them with:\n  pip install -r {os.path.join(here, 'requirements.txt')}"
        )


def _build_model(spec, device, torch_dtype, use_flash_attn, allow_mask_overlap):
    from transformers import AutoTokenizer

    from .bat_sec.inference.configuration_sec import SeCConfig
    from .bat_sec.inference.modeling_sec import SeCModel

    config = SeCConfig.from_pretrained(spec["config_path"])
    # Inverted on purpose: the knob users see is "allow overlap", the knob
    # SAM2 takes is "non_overlap_masks".
    config.hydra_overrides_extra = [f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"]

    if device.startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()

    if spec["is_single_file"]:
        state_dict = load_file(spec["path"])
        if spec["precision"] == "fp8":
            state_dict = {
                k: (v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v)
                for k, v in state_dict.items()
            }

        try:
            # Meta-device construction skips initialising 4B parameters we're
            # about to overwrite — worth ~30s per load.
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device

            with init_empty_weights():
                model = SeCModel(config, use_flash_attn=use_flash_attn)
            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, device="cpu", value=param)
        except (ImportError, RuntimeError):
            model = SeCModel(config, use_flash_attn=use_flash_attn)
            model.load_state_dict(state_dict, strict=True)

        del state_dict
        model = model.eval().to(device=device, dtype=torch_dtype)
    else:
        load_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "use_flash_attn": use_flash_attn,
            "low_cpu_mem_usage": True,
        }
        if device.startswith("cuda"):
            load_kwargs["device_map"] = {"": device}
        model = SeCModel.from_pretrained(spec["path"], **load_kwargs).eval()

    tokenizer = AutoTokenizer.from_pretrained(spec["config_path"], trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

    if device.startswith("cuda") and torch_dtype != torch.float32:
        _register_dtype_hooks(model, torch_dtype)

    return model


def get_model(model_file, device="auto", use_flash_attn=True, allow_mask_overlap=True,
              auto_download=True):
    """Load ``model_file`` (or return the cached instance).

    Downloads the default checkpoint first if nothing is installed and
    ``auto_download`` is set, so the node works on a fresh machine with nothing
    wired to it.

    Returns ``(model, cache_key)``. Pass the key to :func:`release` to unload.
    """
    available = list_sec_models()

    if not available:
        if not auto_download:
            raise RuntimeError(
                "No SeC model found and auto-download is off.\n\n" + _manual_instructions()
            )
        download_default_model()
        available = list_sec_models()
        if not available:
            # Downloaded but still not discoverable — almost always the file
            # landing outside a registered 'sams' path.
            raise RuntimeError(
                "Downloaded the SeC model but it is still not visible in any "
                "registered 'sams' folder.\n\n" + _manual_instructions()
            )

    spec = next((m for m in available if m["name"] == model_file), None)

    if spec is None and auto_download and model_file in dict(_SINGLE_FILE_MODELS):
        # A workflow explicitly asked for a precision that isn't on disk but is
        # downloadable — honour the request rather than quietly running another.
        # fp8 is excluded on purpose: it loads as fp16, so there is nothing to
        # gain from a second 7GB file.
        if model_file != "SeC-4B-fp8.safetensors":
            download_default_model(model_file)
            available = list_sec_models()
            spec = next((m for m in available if m["name"] == model_file), None)

    if spec is None:
        # Stale workflows name a checkpoint that has since been removed or
        # renamed. Falling back beats a hard failure, but say so loudly —
        # silently running a different precision than the workflow asked for
        # would be worse.
        spec = available[0]
        if model_file and model_file != NO_MODEL_SENTINEL:
            print(f"[Bat SeC] '{model_file}' not found, falling back to '{spec['name']}'")

    resolved_device = resolve_device(device)
    precision = spec["precision"]
    torch_dtype = _DTYPE_BY_PRECISION.get(precision, torch.float16)

    if resolved_device == "cpu" and torch_dtype != torch.float32:
        print(f"[Bat SeC] CPU inference needs float32 — converting from {precision.upper()} on load")
        torch_dtype = torch.float32

    if torch_dtype == torch.float32 and use_flash_attn:
        # Flash Attention 2 has no fp32 kernel.
        use_flash_attn = False

    key = _cache_key(spec, resolved_device, use_flash_attn, allow_mask_overlap)

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model, key

        # Only one SeC model fits in VRAM at a time in practice, and holding a
        # stale one just to lose the next load to OOM is a bad trade.
        if _MODEL_CACHE:
            unload_all("switching model/device")

        # Raised bare, before the try below rewraps exceptions into
        # "Failed to load SeC model" — a missing-dependency list reads better
        # without that prefix.
        _preflight_dependencies()

        label = spec["name"] if spec["is_single_file"] else "SeC-4B (sharded)"
        print(f"[Bat SeC] loading {label} [{precision.upper()}] on {resolved_device}"
              f"{'' if use_flash_attn else ' (flash-attn off)'}")

        try:
            model = _build_model(spec, resolved_device, torch_dtype, use_flash_attn, allow_mask_overlap)
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Failed to load SeC model '{label}': {e}") from e

        _MODEL_CACHE[key] = model
        print(f"[Bat SeC] ready on {resolved_device}")
        return model, key


def release(key):
    """Unload one cached model."""
    with _CACHE_LOCK:
        if _MODEL_CACHE.pop(key, None) is None:
            return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── prompt parsing ──────────────────────────────────────────────────────────


def parse_points(raw, frame_shape=None):
    """Parse ``[{"x": .., "y": ..}, ...]`` into an (N, 2) float32 array.

    Returns ``(points, problems)``. Out-of-bounds and malformed entries are
    dropped and described in ``problems`` rather than aborting the run — one
    stray click shouldn't cost a multi-minute segmentation. ``points`` is None
    when nothing usable survived.
    """
    import json

    if not raw or not str(raw).strip():
        return None, []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Points are not valid JSON: {e}") from e

    if not isinstance(entries, list):
        raise ValueError(f"Points must be a JSON array, got {type(entries).__name__}")

    points, problems = [], []
    height = width = None
    if frame_shape is not None:
        height, width = int(frame_shape[1]), int(frame_shape[2])

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "x" not in entry or "y" not in entry:
            problems.append(f"point {i} is not a {{'x':..,'y':..}} object")
            continue
        try:
            x, y = float(entry["x"]), float(entry["y"])
        except (TypeError, ValueError):
            problems.append(f"point {i} has non-numeric coordinates")
            continue
        if x < 0 or y < 0:
            problems.append(f"point {i} ({x:.0f}, {y:.0f}) is negative")
            continue
        if width is not None and (x >= width or y >= height):
            problems.append(f"point {i} ({x:.0f}, {y:.0f}) is outside the {width}x{height} frame")
            continue
        points.append([x, y])

    if not points:
        return None, problems
    return np.array(points, dtype=np.float32), problems


def parse_bbox(bbox):
    """Parse a bbox into ``[x1, y1, x2, y2]`` floats, or None.

    Accepts what the BAT/KJ points editors and the usual BBOX producers emit:
    a ``{startX, startY, endX, endY}`` dict (optionally wrapped in a list), a
    4-tuple in xyxy or xywh form, or either of those as a JSON string.

    The JSON-string case is the one that actually fires in Bat_SecSegmenter:
    the editor's ``bboxes`` widget is a STRING holding
    ``[{"startX":..,"startY":..,"endX":..,"endY":..}]``, not a BBOX tuple. Miss
    it and every ctrl+drag box is silently ignored.
    """
    if bbox is None:
        return None

    if isinstance(bbox, str):
        import json
        text = bbox.strip()
        if not text:
            return None
        try:
            bbox = json.loads(text)
        except json.JSONDecodeError:
            return None

    if isinstance(bbox, (list, tuple)) and len(bbox) > 0 and isinstance(bbox[0], dict):
        bbox = bbox[0]

    if isinstance(bbox, dict):
        if not all(k in bbox and bbox[k] is not None for k in ("startX", "startY", "endX", "endY")):
            return None
        x1, y1 = float(bbox["startX"]), float(bbox["startY"])
        x2, y2 = float(bbox["endX"]), float(bbox["endY"])
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        a, b, c, d = (float(v) for v in bbox)
        # xywh vs xyxy is ambiguous in the general case. Treat it as xywh only
        # when reading it as xyxy would give a degenerate box, which is the
        # convention the upstream node used.
        if c <= a or d <= b:
            x1, y1, x2, y2 = a, b, a + c, b + d
        else:
            x1, y1, x2, y2 = a, b, c, d
    else:
        return None

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def clip_to_frame(points, width, height):
    """Drop points that fall outside the frame. Returns ``(points, problems)``.

    Runs AFTER scaling into frame space — see the note in :func:`segment`.
    """
    if points is None or len(points) == 0:
        return None, []

    inside = [(x, y) for x, y in points if 0 <= x < width and 0 <= y < height]
    problems = [
        f"point ({x:.0f}, {y:.0f}) is outside the {width}x{height} frame"
        for x, y in points if not (0 <= x < width and 0 <= y < height)
    ]
    if not inside:
        return None, problems
    return np.array(inside, dtype=np.float32), problems


def scale_prompts(points, bbox, from_size, to_size):
    """Rescale editor-space coordinates into frame space.

    The points editor's coord space is its width/height widgets, which track
    whatever image was loaded as its background. If that background came from a
    different resolution than the frames actually being segmented (a proxy, a
    resized plate), the clicks land in the wrong place — so rescale rather than
    silently segmenting the wrong pixels.
    """
    src_w, src_h = from_size
    dst_w, dst_h = to_size
    if not src_w or not src_h or (src_w == dst_w and src_h == dst_h):
        return points, bbox

    sx, sy = dst_w / src_w, dst_h / src_h
    if points is not None:
        points = points * np.array([sx, sy], dtype=np.float32)
    if bbox is not None:
        bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    return points, bbox


# ── mask preview overlay ────────────────────────────────────────────────────

_PREVIEW_COLOURS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "grey": (0.5, 0.5, 0.5),
    "black": (0.0, 0.0, 0.0),
}

_SEMI_SUFFIX = "(50%)"

# Grey is deliberately solid-only. At 50% over a mid-grey plate it lands within
# ~0.05 of the original value — technically correct, visually useless.
_SEMI_COLOURS = ["red", "green", "blue", "black"]

# Solid first, then the semitransparent variants. Solid tells you exactly which
# pixels are in the mask; semitransparent lets you see the image through the
# tint, which is what you want for judging edges — hence the default.
MASK_PREVIEW_MODES = (
    list(_PREVIEW_COLOURS)
    + [f"{name} {_SEMI_SUFFIX}" for name in _SEMI_COLOURS]
)
MASK_PREVIEW_DEFAULT = f"red {_SEMI_SUFFIX}"


def parse_preview_mode(mode):
    """``"red (50%)"`` -> ``((1.0, 0.0, 0.0), 0.5)``.

    Never returns None: the overlay always draws. (An earlier "off" option that
    skipped the work is gone; a workflow still carrying it falls through to the
    unknown-mode path below and gets the default.)
    """
    name, alpha = str(mode).strip(), 1.0
    if name.endswith(_SEMI_SUFFIX):
        name = name[: -len(_SEMI_SUFFIX)].strip()
        alpha = 0.5

    rgb = _PREVIEW_COLOURS.get(name)
    if rgb is None:
        # A workflow saved against a future/renamed option list. Draw something
        # rather than silently returning the plate unmarked, which would read as
        # "SeC found nothing".
        print(f"[Bat SeC] unknown mask_preview '{mode}' — falling back to {MASK_PREVIEW_DEFAULT}")
        return parse_preview_mode(MASK_PREVIEW_DEFAULT)
    return rgb, alpha


def draw_mask_overlay(frames, masks, mode=MASK_PREVIEW_DEFAULT):
    """Composite ``masks`` over ``frames`` as a ComfyUI IMAGE batch.

    Always allocates: the result is a second float32 batch the size of the input,
    which on a long 4K clip is gigabytes, and ComfyUI gives no way to skip the
    work when the output socket is unconnected.
    """
    rgb, alpha = parse_preview_mode(mode)

    plate = frames.detach().float()
    if plate.shape[-1] < 3:
        plate = plate[..., :1].repeat(1, 1, 1, 3)
    elif plate.shape[-1] > 3:
        plate = plate[..., :3]  # IMAGE outputs are RGB; drop any alpha

    # [N,H,W] -> [N,H,W,1] so it broadcasts across colour. Masks are binary here,
    # but keeping this a per-pixel blend means a soft mask would feather correctly
    # instead of hard-thresholding.
    coverage = masks.detach().float().clamp(0, 1).unsqueeze(-1).to(plate.device)
    weight = coverage * alpha

    colour = torch.tensor(rgb, dtype=plate.dtype, device=plate.device).view(1, 1, 1, 3)
    return (plate * (1.0 - weight) + colour * weight).clamp(0.0, 1.0)


# ── frames ──────────────────────────────────────────────────────────────────


def frames_to_pil(frames):
    """ComfyUI IMAGE batch -> list of RGB PIL images.

    CHANGE vs upstream (3/4 and 4/4): upstream did
    ``Image.fromarray((t*255).clamp(0,255).byte().numpy(), mode='RGB')``, which
    (a) raised for any batch that wasn't exactly 3-channel — RGBA and
    single-channel batches are both common upstream of a segmenter — and
    (b) truncated instead of rounding, biasing every pixel down by up to 1 LSB.
    """
    out = []
    for i in range(frames.shape[0]):
        arr = frames[i].detach().cpu().float()
        if arr.ndim == 2:
            arr = arr.unsqueeze(-1)
        channels = arr.shape[-1]
        if channels < 3:
            arr = arr[..., :1].repeat(1, 1, 3)  # grey (or grey+alpha) -> RGB
        elif channels > 3:
            arr = arr[..., :3]  # drop alpha
        arr = (arr.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def write_frames(pil_images):
    """Write frames to a private temp dir as JPEGs for SAM2's file-based loader.

    SAM2 sorts by ``int(basename)``, hence the zero-padded numeric names. The
    directory is per-call so concurrent jobs can't clobber each other.
    """
    temp_dir = tempfile.mkdtemp(prefix="bat_sec_frames_")
    for i, img in enumerate(pil_images):
        img.save(os.path.join(temp_dir, f"{i:05d}.jpg"), "JPEG", quality=95)
    return temp_dir


# ── segmentation ────────────────────────────────────────────────────────────


def _apply_prompts(model, state, frame_idx, object_id, points, labels, bbox, mask):
    """Seed the tracker on the annotation frame. Returns the init mask.

    CHANGE vs upstream (2/4) — the box no longer gets thrown away.

    Upstream's "handle bbox + points combination properly" branch made two
    calls: box first, then points. But ``add_new_points_or_box`` defaults to
    ``clear_old_points=True``, which resets the stored prompt to just the new
    points — and SAM2 represents a box AS two points with labels 2/3, so the
    second call deleted the box. The bbox therefore had no effect whenever
    points were also supplied.

    SAM2 is built for the combined form: pass both and it concatenates the
    box's two label-2/3 points ahead of the clicks itself. So: one call.
    (Upstream's own bidirectional re-seed did it correctly, which meant the
    forward and backward passes were seeded from *different* prompts.)
    """
    init_mask = None

    if mask is not None:
        _, _, logits = model.grounding_encoder.add_new_mask(
            inference_state=state, frame_idx=frame_idx, obj_id=object_id, mask=mask,
        )
        init_mask = mask

    if points is not None or bbox is not None:
        _, _, logits = model.grounding_encoder.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels if points is not None else None,
            box=bbox,
        )
        init_mask = (logits[0] > 0.0).cpu().numpy()

    return init_mask


def _collect(masks_tensor, seen, model, state, start_idx, max_frames, reverse,
             init_mask, mllm_memory_size, overwrite):
    """Run one propagation pass, writing straight into ``masks_tensor``.

    Writing into a pre-allocated tensor rather than accumulating a dict of
    per-frame masks is what keeps this from spiking ~600-800MB on long clips.
    """
    for frame_idx, obj_ids, logits in model.propagate_in_video(
        state,
        start_frame_idx=start_idx,
        max_frame_num_to_track=max_frames,
        reverse=reverse,
        init_mask=init_mask,
        mllm_memory_size=mllm_memory_size,
    ):
        if not overwrite and frame_idx in seen:
            continue
        if len(obj_ids) == 0:
            continue
        # One tracked object per node instance; use object_id + a second node
        # for a second object. logits[0] is [1, H, W]; peel leading axes rather
        # than squeeze(0), which is a no-op on a 1-row frame.
        mask = (logits[0] > 0.0).cpu().float()
        while mask.dim() > 2:
            mask = mask[0]
        masks_tensor[frame_idx] = mask
        seen.add(frame_idx)


def segment(
    model,
    frames,
    positive_points="",
    negative_points="",
    bbox=None,
    input_mask=None,
    editor_size=None,
    tracking_direction="bidirectional",
    annotation_frame_idx=0,
    object_id=1,
    max_frames_to_track=-1,
    mllm_memory_size=12,
    offload_video_to_cpu=True,
):
    """Track one object through ``frames`` from the prompts on the annotation frame.

    Returns a ``[N, H, W]`` float MASK tensor.
    """
    if frames is None or frames.numel() == 0:
        raise ValueError("No frames provided.")
    if frames.ndim != 4:
        raise ValueError(f"frames must be [batch, height, width, channels], got {tuple(frames.shape)}")

    num_frames, height, width = frames.shape[0], frames.shape[1], frames.shape[2]

    if not 0 <= annotation_frame_idx < num_frames:
        raise ValueError(
            f"annotation_frame_idx {annotation_frame_idx} is out of range — "
            f"this batch has {num_frames} frame(s) (0-{num_frames - 1})."
        )

    # Parse WITHOUT frame bounds, scale into frame space, and only then reject
    # out-of-frame points. Validating first would judge editor-space coordinates
    # against frame dimensions: with an editor coord space larger than the
    # frames (a 4K plate driving an HD segment) every point past the frame width
    # would be thrown away despite being perfectly valid on the canvas.
    pos_points, pos_problems = parse_points(positive_points)
    neg_points, neg_problems = parse_points(negative_points)
    bbox_coords = parse_bbox(bbox)

    if editor_size:
        pos_points, bbox_coords = scale_prompts(pos_points, bbox_coords, editor_size, (width, height))
        neg_points, _ = scale_prompts(neg_points, None, editor_size, (width, height))

    pos_points, dropped = clip_to_frame(pos_points, width, height)
    pos_problems += dropped
    neg_points, dropped = clip_to_frame(neg_points, width, height)
    neg_problems += dropped
    if bbox_coords is not None:
        bbox_coords = [
            min(max(bbox_coords[0], 0.0), width - 1), min(max(bbox_coords[1], 0.0), height - 1),
            min(max(bbox_coords[2], 0.0), width - 1), min(max(bbox_coords[3], 0.0), height - 1),
        ]
        if bbox_coords[2] - bbox_coords[0] < 1 or bbox_coords[3] - bbox_coords[1] < 1:
            print("[Bat SeC] bbox is empty after clipping to the frame — ignoring it")
            bbox_coords = None

    # SAM2 label convention: 1 = include, 0 = exclude.
    points = labels = None
    if pos_points is not None and neg_points is not None:
        points = np.concatenate([pos_points, neg_points], axis=0)
        labels = np.concatenate([
            np.ones(len(pos_points), dtype=np.int32),
            np.zeros(len(neg_points), dtype=np.int32),
        ])
    elif pos_points is not None:
        points, labels = pos_points, np.ones(len(pos_points), dtype=np.int32)
    elif neg_points is not None:
        points, labels = neg_points, np.zeros(len(neg_points), dtype=np.int32)

    mask_prompt = None
    if input_mask is not None:
        if input_mask.dim() == 2:
            mask_2d = input_mask
        elif input_mask.dim() == 3:
            if input_mask.shape[0] > 1:
                # The prompt seeds ONE frame, so a whole mask sequence can't be
                # honoured. Say which one is being used rather than quietly
                # taking the first.
                print(f"[Bat SeC] input_mask has {input_mask.shape[0]} masks — using the first "
                      f"as the prompt on frame {annotation_frame_idx}; the rest are ignored")
            mask_2d = input_mask[0]
        else:
            raise ValueError(f"input_mask must be [H,W] or [B,H,W], got {tuple(input_mask.shape)}")

        mask_h, mask_w = int(mask_2d.shape[0]), int(mask_2d.shape[1])
        if (mask_h, mask_w) != (height, width):
            # SAM2 rescales whatever it's handed to the model's square input, so
            # a same-aspect mask at another resolution is fine — resize it and
            # carry on. A DIFFERENT aspect ratio would be stretched and seed the
            # wrong region, which is the footgun worth refusing.
            frame_ar, mask_ar = width / height, mask_w / mask_h
            if abs(mask_ar - frame_ar) > 0.01 * frame_ar:
                raise ValueError(
                    f"input_mask is {mask_w}x{mask_h} ({mask_ar:.3f}:1) but the frames are "
                    f"{width}x{height} ({frame_ar:.3f}:1). Aspect ratios must match, or the "
                    "mask would be stretched onto the wrong part of the image."
                )
            print(f"[Bat SeC] resizing input_mask {mask_w}x{mask_h} -> {width}x{height}")
            mask_2d = torch.nn.functional.interpolate(
                mask_2d[None, None].float(), size=(height, width), mode="nearest",
            )[0, 0]

        mask_prompt = (mask_2d.cpu().numpy() > 0.5).astype(np.bool_)
        if not mask_prompt.any():
            raise ValueError(
                "input_mask is empty (nothing above 0.5). Feed a mask that actually covers "
                "the object, or leave it unconnected and click the object instead."
            )

    if points is None and bbox_coords is None and mask_prompt is None:
        detail = f" Rejected: {'; '.join(pos_problems + neg_problems)}." if (pos_problems or neg_problems) else ""
        raise ValueError(
            "No usable prompt. Shift+click a positive point in the editor "
            "(or ctrl+drag a box, or feed input_mask)." + detail
        )

    if pos_problems or neg_problems:
        print(f"[Bat SeC] ignored {len(pos_problems + neg_problems)} bad point(s): "
              f"{'; '.join(pos_problems + neg_problems)}")

    # A mask prompt defines the object's extent, so positive clicks outside it
    # are contradictory — drop them rather than let them fight the mask.
    if mask_prompt is not None and points is not None:
        keep = [
            i for i, (x, y) in enumerate(points)
            if 0 <= int(y) < mask_prompt.shape[0]
            and 0 <= int(x) < mask_prompt.shape[1]
            and (labels[i] == 0 or mask_prompt[int(y), int(x)])
        ]
        if len(keep) != len(points):
            print(f"[Bat SeC] dropped {len(points) - len(keep)} positive point(s) outside input_mask")
        points = points[keep] if keep else None
        labels = labels[keep] if keep else None

    video_dir = None
    try:
        pil_images = frames_to_pil(frames)
        video_dir = write_frames(pil_images)

        try:
            offload_state_to_cpu = str(model.device) == "cpu"
        except AttributeError:
            offload_state_to_cpu = False

        state = model.grounding_encoder.init_state(
            video_path=video_dir,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        model.grounding_encoder.reset_state(state)

        init_mask = _apply_prompts(
            model, state, annotation_frame_idx, object_id, points, labels, bbox_coords, mask_prompt,
        )
        if init_mask is None:
            raise RuntimeError("SeC produced no initial mask from the supplied prompt.")

        # propagate_in_video treats max_frame_num_to_track as "frames BEYOND
        # the start frame" (it iterates start..start+max inclusive), so -1 has
        # to become the full length, not length-1.
        span = num_frames if max_frames_to_track < 0 else max_frames_to_track

        masks_tensor = torch.zeros(num_frames, height, width, dtype=torch.float32)
        seen = set()

        if tracking_direction == "bidirectional":
            _collect(masks_tensor, seen, model, state, annotation_frame_idx, span,
                     False, init_mask, mllm_memory_size, overwrite=True)

            # A backward pass needs a clean state: the forward pass left every
            # frame in frames_already_tracked, which would make the re-seed a
            # correction rather than a fresh conditioning frame.
            model.grounding_encoder.reset_state(state)
            _apply_prompts(
                model, state, annotation_frame_idx, object_id, points, labels, bbox_coords, mask_prompt,
            )
            _collect(masks_tensor, seen, model, state, annotation_frame_idx, span,
                     True, init_mask, mllm_memory_size, overwrite=False)
        else:
            _collect(masks_tensor, seen, model, state, annotation_frame_idx, span,
                     tracking_direction == "backward", init_mask, mllm_memory_size, overwrite=True)

        if len(seen) < num_frames:
            missing = num_frames - len(seen)
            print(f"[Bat SeC] {missing} of {num_frames} frame(s) not covered by "
                  f"'{tracking_direction}' tracking from frame {annotation_frame_idx} — left empty")

        return masks_tensor

    finally:
        if video_dir and os.path.isdir(video_dir):
            try:
                shutil.rmtree(video_dir)
            except OSError as e:
                print(f"[Bat SeC] could not remove temp frames {video_dir}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
