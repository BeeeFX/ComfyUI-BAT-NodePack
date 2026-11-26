"""
The networks behind 🦇 Lines' model filters, and how each one is fed.

The architecture files next to this one are vendored (see NOTICE.md); this file
is BAT code. Each `_run_*` takes the padded uint8 RGB frame that
``bat_lines.prepare`` produced and returns a uint8 (H, W) map of white lines on
black at that same padded size, with controlnet_aux's pre- and post-processing
— except TEED, which is fed the way its author feeds it (BGR minus the BIPED
mean) because its weights come from the author, not from controlnet_aux.

Models are kept on the CPU between uses (at most two, LRU) and moved to
ComfyUI's device for a run through a ModelPatcher, so ComfyUI can evict them
when a diffusion model needs the VRAM. Without ComfyUI (the tests) they run
on the CPU.
"""

import functools
import logging
import threading
from collections import OrderedDict

import cv2
import numpy as np
import torch

from .. import bat_lines_models as weights

logger = logging.getLogger("[Bat_Lines]")

_MAX_WARM = 2
_models = OrderedDict()       # weights key -> patcher (or bare module)
_models_lock = threading.Lock()


def weight_key(filter_name, p):
    if filter_name == "lineart realistic":
        return "lineart_coarse" if p.get("lineart_coarse") else "lineart"
    return {"teed": "teed", "hed": "hed", "lineart anime": "anime",
            "lineart manga": "manga", "mlsd": "mlsd"}[filter_name]


def ensure_weights(filter_name, p):
    return weights.ensure(weight_key(filter_name, p), bool(p.get("auto_download", True)))


def _build(key):
    if key == "teed":
        from .ted import TED
        return TED()
    if key in ("lineart", "lineart_coarse"):
        from .informative_drawings import Generator
        return Generator(3, 1, 3)
    if key == "anime":
        from torch import nn
        from .anime2sketch import UnetGenerator
        norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        return UnetGenerator(3, 1, 8, 64, norm_layer=norm, use_dropout=False)
    if key == "manga":
        from .manga_line import res_skip
        return res_skip()
    if key == "mlsd":
        from .mlsd_large import MobileV2_MLSD_Large
        return MobileV2_MLSD_Large()
    if key == "hed":
        from .hed import ControlNetHED_Apache2
        return ControlNetHED_Apache2()
    raise KeyError(key)


def _load_state_dict(path):
    try:
        import comfy.utils
        sd = comfy.utils.load_torch_file(path, safe_load=True)
    except ImportError:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    # netG / erika were saved from DataParallel.
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()}


def _patcher(net):
    try:
        import comfy.model_management as mm
        import comfy.model_patcher
        return comfy.model_patcher.ModelPatcher(
            net, load_device=mm.get_torch_device(), offload_device=torch.device("cpu"))
    except Exception as exc:     # no ComfyUI, or an API change: run unmanaged
        logger.debug("running %s without a ModelPatcher (%s)", type(net).__name__, exc)
        return net


def _get(key, p):
    with _models_lock:
        entry = _models.get(key)
        if entry is not None:
            _models.move_to_end(key)
            return entry
    path = weights.ensure(key, bool(p.get("auto_download", True)))
    net = _build(key)
    net.load_state_dict(_load_state_dict(path), strict=True)
    net.float().eval()
    entry = _patcher(net)
    with _models_lock:
        _models[key] = entry
        while len(_models) > _MAX_WARM:
            _models.popitem(last=False)
    return entry


def _on_device(entry):
    """The module, loaded on the device it should run on."""
    if isinstance(entry, torch.nn.Module):
        return entry, next(entry.parameters()).device
    import comfy.model_management as mm
    mm.load_models_gpu([entry])
    return entry.model, entry.load_device


def safe_step(x, step=2):
    # controlnet_aux util.safe_step
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def _mean_sigmoid(edges, h, w):
    """controlnet_aux's fusion for HED / TEED: every side output at full size,
    averaged, then a sigmoid."""
    maps = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
    maps = [cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in maps]
    return 1 / (1 + np.exp(-np.mean(np.stack(maps, axis=2), axis=2).astype(np.float64)))


# BIPED's mean, BGR order (TEED dataset.py, BIPED_mean[:3]).
_TEED_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _run_teed(net, dev, img, p):
    h, w = img.shape[:2]
    x = img[:, :, ::-1].astype(np.float32) - _TEED_MEAN_BGR
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(dev)
    edge = _mean_sigmoid(net(x), h, w)
    steps = int(p["teed_safe_steps"])
    if steps != 0:
        edge = safe_step(edge, steps)
    return (edge * 255.0).clip(0, 255).astype(np.uint8)


def _run_hed(net, dev, img, p):
    h, w = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(dev)
    edge = _mean_sigmoid(net(x), h, w)
    if p["hed_safe"]:
        edge = safe_step(edge)
    return (edge * 255.0).clip(0, 255).astype(np.uint8)


def _run_lineart(net, dev, img, p):
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(dev) / 255.0
    line = net(x)[0][0].cpu().numpy()
    return 255 - (line * 255.0).clip(0, 255).astype(np.uint8)


def _run_anime(net, dev, img, p):
    h, w = img.shape[:2]
    hn, wn = 256 * int(np.ceil(float(h) / 256.0)), 256 * int(np.ceil(float(w) / 256.0))
    big = cv2.resize(img, (wn, hn), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(np.ascontiguousarray(big.transpose(2, 0, 1))).float()[None].to(dev)
    line = net(x / 127.5 - 1.0)[0, 0] * 127.5 + 127.5
    line = line.cpu().numpy().clip(0, 255).astype(np.uint8)
    return 255 - cv2.resize(line, (w, h), interpolation=cv2.INTER_AREA)


def _run_manga(net, dev, img, p):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = torch.from_numpy(gray).float()[None, None].to(dev)
    line = net(x).cpu().numpy()[0, 0]
    line[line > 255] = 255
    line[line < 0] = 0
    return 255 - line.astype(np.uint8)


def _run_mlsd(net, dev, img, p):
    from .mlsd_utils import pred_lines
    raster = np.zeros(img.shape[:2], dtype=np.uint8)
    try:
        segs = pred_lines(img, net, [img.shape[0], img.shape[1]],
                          float(p["mlsd_score"]), float(p["mlsd_distance"]))
    except IndexError:
        # pred_lines indexes an empty array when nothing passes the thresholds;
        # controlnet_aux swallows that as "no lines", and so do we.
        segs = np.zeros((0, 4), dtype=np.float32)
    for x0, y0, x1, y1 in segs:
        cv2.line(raster, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
    return {"segments": np.asarray(segs, dtype=np.float64).reshape(-1, 4), "raster": raster}


_RUNNERS = {"teed": _run_teed, "hed": _run_hed, "lineart realistic": _run_lineart,
            "lineart anime": _run_anime, "lineart manga": _run_manga, "mlsd": _run_mlsd}


def run(filter_name, img, p):
    entry = _get(weight_key(filter_name, p), p)
    net, dev = _on_device(entry)
    with torch.inference_mode():
        return _RUNNERS[filter_name](net, dev, img, p)
