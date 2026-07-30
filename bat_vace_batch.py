import torch
import torch.nn.functional as F


class _AnyType(str):
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class _FlexibleOptional(dict):
    """Optional input dict that accepts any unknown keys, returning a permissive
    type. Lets the frontend (JS) add image_N / mask_N / index_N inputs that
    ComfyUI's executor will route to **kwargs on the Python side.
    """

    def __init__(self, base=None):
        super().__init__()
        if base:
            for k, v in base.items():
                self[k] = v

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        return (_ANY,)


def _resize_image(t, H, W):
    # t: [N, H, W, C]
    if t.shape[1] == H and t.shape[2] == W:
        return t
    tt = t.permute(0, 3, 1, 2)
    tt = F.interpolate(tt, size=(H, W), mode="bilinear", align_corners=False)
    return tt.permute(0, 2, 3, 1).contiguous()


def _resize_mask(t, H, W):
    # t: [N, H, W] or [H, W]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.shape[-2] == H and t.shape[-1] == W:
        return t
    tt = t.unsqueeze(1)
    tt = F.interpolate(tt, size=(H, W), mode="bilinear", align_corners=False)
    return tt.squeeze(1).contiguous()


class VaceBatchTool:
    @classmethod
    def INPUT_TYPES(cls):
        base_optional = {
            "plate_image": ("IMAGE",),
            "plate_mask": ("MASK",),
            "start_image": ("IMAGE",),
            "start_mask": ("MASK",),
            "end_image": ("IMAGE",),
            "end_mask": ("MASK",),
        }
        return {
            "required": {
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 100000}),
                "premultiply": ("BOOLEAN", {"default": True}),
                "fill_color": ("INT", {"default": 127, "min": 0, "max": 255}),
            },
            "optional": _FlexibleOptional(base_optional),
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "build"
    CATEGORY = "BAT/vace"
    DESCRIPTION = "Batch builder for VACE (Video Authoring Compositor) workflows. Composes per-frame image / mask / control inputs into a single stacked batch with optional fill colour and premultiplication."

    def build(self, num_frames, premultiply, fill_color, **kwargs):
        plate_image = kwargs.get("plate_image")
        plate_mask = kwargs.get("plate_mask")
        start_image = kwargs.get("start_image")
        start_mask = kwargs.get("start_mask")
        end_image = kwargs.get("end_image")
        end_mask = kwargs.get("end_mask")

        # Collect dynamic keyframes (image_N, mask_N, index_N)
        keyframes = []
        i = 1
        while True:
            img_key = f"image_{i}"
            mask_key = f"mask_{i}"
            idx_key = f"index_{i}"
            if img_key not in kwargs and mask_key not in kwargs and idx_key not in kwargs:
                break
            kf_img = kwargs.get(img_key)
            kf_mask = kwargs.get(mask_key)
            kf_idx_raw = kwargs.get(idx_key, 0)
            if kf_img is not None or kf_mask is not None:
                try:
                    kf_idx = int(kf_idx_raw)
                except (TypeError, ValueError):
                    kf_idx = 0
                keyframes.append((kf_idx, kf_img, kf_mask))
            i += 1
            if i > 4096:
                break

        # Start/end frames as keyframes (applied last so they win on collision)
        if start_image is not None or start_mask is not None:
            keyframes.append((0, start_image, start_mask))
        if end_image is not None or end_mask is not None:
            keyframes.append((max(0, num_frames - 1), end_image, end_mask))

        # Determine target H, W from the first available image/mask
        H = W = None
        for cand in (plate_image, start_image, end_image, *(kf[1] for kf in keyframes)):
            if cand is not None:
                H, W = int(cand.shape[1]), int(cand.shape[2])
                break
        if H is None:
            for cand in (plate_mask, start_mask, end_mask, *(kf[2] for kf in keyframes)):
                if cand is not None:
                    if cand.ndim == 2:
                        H, W = int(cand.shape[0]), int(cand.shape[1])
                    else:
                        H, W = int(cand.shape[-2]), int(cand.shape[-1])
                    break
        if H is None:
            H, W = 512, 512

        fc = float(fill_color) / 255.0

        images = torch.full((num_frames, H, W, 3), fc, dtype=torch.float32)
        masks = torch.ones((num_frames, H, W), dtype=torch.float32)

        # Plate: fill base sequence; hold last frame if plate is shorter
        if plate_image is not None:
            p = _resize_image(plate_image, H, W)[..., :3].to(images.dtype).cpu()
            n = min(p.shape[0], num_frames)
            images[:n] = p[:n]
            if p.shape[0] < num_frames:
                images[p.shape[0]:] = p[-1:]
        if plate_mask is not None:
            pm = _resize_mask(plate_mask, H, W).to(masks.dtype).cpu()
            n = min(pm.shape[0], num_frames)
            masks[:n] = pm[:n]
            if pm.shape[0] < num_frames:
                masks[pm.shape[0]:] = pm[-1:]

        # Apply keyframes (index out of range is silently skipped)
        for (idx, kf_img, kf_mask) in keyframes:
            if idx < 0 or idx >= num_frames:
                continue
            if kf_img is not None:
                k = _resize_image(kf_img, H, W)[..., :3].to(images.dtype).cpu()
                images[idx] = k[0]
            if kf_mask is not None:
                km = _resize_mask(kf_mask, H, W).to(masks.dtype).cpu()
                masks[idx] = km[0]
            elif kf_img is not None:
                # Image given without a mask → preserve frame (black mask).
                #
                # This `else` used to hang off `if kf_mask is not None` alone, so
                # a keyframe that supplied ONLY a mask (no image) also fell in
                # here and had the mask it just set overwritten with zeros.
                # Now it only fires for the image-without-mask case the comment
                # describes.
                masks[idx] = 0.0

        if premultiply:
            m = masks.unsqueeze(-1).clamp(0.0, 1.0)
            images = images * (1.0 - m) + fc * m

        return (images, masks)
