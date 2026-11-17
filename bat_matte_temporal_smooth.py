"""
Bat_MatteTemporalSmooth — take the frame-to-frame flicker out of a matte.
========================================================================

Per-frame keyers and matting models (CorridorKey2 off a real screen,
MEMatte, SAM) decide every frame's edge on its own, so an edge that should
sit still shimmers by a pixel or two. This averages each pixel of the alpha
(and optionally the foreground colour) with the same pixel in its +-radius
neighbours — but only where the PLATE did not change there, so anything
that actually moves is left alone instead of smeared.

    weight(t, s) = exp(-M_ts / motion_tolerance) * falloff^(|s-t| - 1)
    M_ts         = 5x5 max-filter of  max_c |G_s - G_t|

G is the plate blurred 5x5 (so grain doesn't read as motion). The max-filter
spreads the strongest change nearby onto each pixel: the blur alone dilutes
the evidence of motion exactly at a moving edge, and an edge pixel was then
still averaged with the frames where the object had not arrived yet — a
visible trail, caught by tests/verify_matte_temporal_smooth.py. The gate
compares all three channels, never luma: on the shot this was built for
(skin against an orange wall) skin and wall have the same luminance, and a
luma gate could not see the hand move and dragged its edge.

Measured on that shot (83 frames, 1920x1318, CorridorKey 2 output):
one-frame "blink" pixels 326 -> 27 per frame, premultiplied edge jitter
roughly halved, semi-transparent edge pixels kept (14.8% -> 15.9%). A faint
trail remains on the fastest edges; lower motion_tolerance trades smoothing
for less of it.

Nothing is clamped: an HDR foreground passes through with its range intact,
and alpha stays in whatever range it came in.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GATE_BLUR = 5          # px box blur on the plate before comparing frames
MOTION_SPREAD = 5      # px max-filter on the frame difference (see the docstring)
PLATE_SPACES = ["display (sRGB / Rec.709)", "linear"]


def _as_mask_batch(m):
    if m.dim() == 2:
        m = m.unsqueeze(0)
    elif m.dim() == 4:
        m = m[..., 0] if m.shape[-1] == 1 else m[:, 0]
    return m.float()


def _gate_frame(images, t, size, linear, device):
    """One plate frame -> [3,H,W] blurred, display-referred, at the matte's size."""
    x = images[t, ..., :3].to(device=device, dtype=torch.float32).permute(2, 0, 1)
    if linear:
        # Compare in a display-like encoding: a linear threshold would treat a
        # 0.01 change in the shadows the same as one in the highlights.
        x = x.clamp(min=0.0).pow(1.0 / 2.2)
    x = x.unsqueeze(0)
    if x.shape[-2:] != size:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    p = GATE_BLUR // 2
    x = F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), GATE_BLUR, stride=1)
    return x[0]


def smooth_matte(images, alpha, foreground=None, radius=2, motion_tolerance=0.04,
                 falloff=0.6, linear_plate=False, device=None, on_frame=None):
    """Motion-gated temporal smoothing. Returns (alpha [T,H,W], foreground or None).

    images [T,H,W,C] drives the gate; alpha [T,H,W] (or [H,W]) and optional
    foreground [T,H,W,3] are smoothed. Works frame by frame with a rolling
    2*radius+1 window on ``device``, so memory does not scale with clip length.
    """
    alpha = _as_mask_batch(alpha)
    T = int(alpha.shape[0])
    if images.shape[0] != T:
        raise ValueError(f"the plate has {images.shape[0]} frames but the alpha has {T}; "
                         f"they must be the same clip.")
    if foreground is not None and foreground.shape[0] != T:
        raise ValueError(f"the foreground has {foreground.shape[0]} frames but the alpha has {T}.")
    if foreground is not None and foreground.shape[1:3] != alpha.shape[1:3]:
        raise ValueError(f"foreground is {tuple(foreground.shape[1:3])} but alpha is "
                         f"{tuple(alpha.shape[1:3])}; they must be the same size.")
    device = torch.device(device) if device is not None else torch.device("cpu")
    size = tuple(alpha.shape[1:3])
    tol = max(float(motion_tolerance), 1e-6)
    radius = max(0, int(radius))

    out_a = torch.empty_like(alpha, device="cpu")
    out_f = torch.empty_like(foreground, dtype=torch.float32, device="cpu") if foreground is not None else None
    gates, alphas, fgs = {}, {}, {}

    def fetch(s):
        if s not in gates:
            gates[s] = _gate_frame(images, s, size, linear_plate, device)
            alphas[s] = alpha[s].to(device)
            if foreground is not None:
                fgs[s] = foreground[s].to(device=device, dtype=torch.float32)

    for t in range(T):
        for s in [k for k in gates if k < t - radius]:
            del gates[s], alphas[s]
            fgs.pop(s, None)
        lo, hi = max(0, t - radius), min(T - 1, t + radius)
        for s in range(lo, hi + 1):
            fetch(s)
        acc_a = alphas[t].clone()
        acc_f = fgs[t].clone() if foreground is not None else None
        wsum = torch.ones(size, device=device)
        for s in range(lo, hi + 1):
            if s == t:
                continue
            diff = (gates[s] - gates[t]).abs().amax(0, keepdim=True)[None]
            diff = F.max_pool2d(diff, MOTION_SPREAD, stride=1, padding=MOTION_SPREAD // 2)[0, 0]
            w = torch.exp(-diff / tol) * (falloff ** (abs(s - t) - 1))
            acc_a += w * alphas[s]
            if acc_f is not None:
                acc_f += w[..., None] * fgs[s]
            wsum += w
        out_a[t] = (acc_a / wsum).cpu()
        if acc_f is not None:
            out_f[t] = (acc_f / wsum[..., None]).cpu()
        if on_frame is not None:
            on_frame(t)
    return out_a, out_f


class BatMatteTemporalSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The plate the matte was pulled from. Only used to tell what is "
                               "moving — never altered or output (unless no foreground is wired).",
                }),
                "alpha": ("MASK", {"tooltip": "The flickering matte, e.g. CorridorKey 2's alpha."}),
                "radius": ("INT", {
                    "default": 2, "min": 1, "max": 6,
                    "tooltip": "Frames each side to average with. 2 is a good start; more is "
                               "steadier and trails more on motion.",
                }),
                "motion_tolerance": ("FLOAT", {
                    "default": 0.04, "min": 0.005, "max": 0.5, "step": 0.005,
                    "tooltip": "How much the plate may change at a pixel before that pixel stops "
                               "being averaged. Lower = less trailing on moving edges, less "
                               "smoothing. In display (0-1) units.",
                }),
                "plate_colourspace": (PLATE_SPACES, {
                    "default": PLATE_SPACES[0],
                    "tooltip": "linear = the plate is scene-linear (EXR, ACEScg); it is encoded "
                               "before comparing so motion_tolerance means the same thing.",
                }),
            },
            "optional": {
                "foreground": ("IMAGE", {
                    "tooltip": "Optional foreground colour (CorridorKey 2's foreground). Smoothed "
                               "with the same weights so colour and alpha stay in step.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "foreground")
    FUNCTION = "smooth"
    CATEGORY = "BAT/mask"
    DESCRIPTION = (
        "Removes frame-to-frame flicker from a matte by averaging each pixel with its "
        "neighbouring frames wherever the plate is not moving. Wire the plate, the alpha and "
        "(optionally) the foreground. Without a foreground, the 'foreground' output is the "
        "plate, unchanged.")

    def smooth(self, images, alpha, radius, motion_tolerance, plate_colourspace, foreground=None):
        import comfy.model_management as mm
        import comfy.utils

        pbar = comfy.utils.ProgressBar(int(_as_mask_batch(alpha).shape[0]))

        def on_frame(_t):
            mm.throw_exception_if_processing_interrupted()
            pbar.update(1)

        out_a, out_f = smooth_matte(images, alpha, foreground, radius=radius,
                                    motion_tolerance=motion_tolerance,
                                    linear_plate=(plate_colourspace == "linear"),
                                    device=mm.get_torch_device(), on_frame=on_frame)
        return (out_a, out_f if out_f is not None else images)
