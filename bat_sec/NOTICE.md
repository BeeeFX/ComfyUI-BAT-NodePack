# bat_sec — third-party attribution

Everything under this directory is **vendored third-party code**, not BAT pack
code. It is carried here so `Bat_SecSegmenter` runs without the upstream
`Comfyui-SecNodes` pack installed. Licensed under Apache 2.0 — see `LICENSE`.

| Component | Origin | Licence |
|---|---|---|
| `inference/` (SeC model, `modeling_sec.py`, `configuration_sec.py`, InternVL/InternLM2/Phi-3 code) | [OpenIXCLab/SeC-4B](https://huggingface.co/OpenIXCLab/SeC-4B) | Apache 2.0 |
| `inference/sam2/`, `configs/` | [SAM 2](https://github.com/facebookresearch/sam2) — Meta Platforms, Inc. | Apache 2.0 |
| `model_config/` | SeC-4B config + tokenizer files | Apache 2.0 |
| Packaging, lazy-hydra fix, local `_target_` registry | [9nate-drake/Comfyui-SecNodes](https://github.com/9nate-drake/Comfyui-SecNodes) v1.2.2 | Apache 2.0 |

## Re-vendoring

The tree is a **byte-identical rsync** of the upstream pack. To update:

```bash
SRC=/path/to/Comfyui-SecNodes
DST=ComfyUI-BAT-NodePack/bat_sec
rsync -a --exclude='__pycache__' $SRC/inference/    $DST/inference/
rsync -a --exclude='__pycache__' $SRC/configs/      $DST/configs/
rsync -a                         $SRC/model_config/ $DST/model_config/
cp $SRC/LICENSE $DST/LICENSE
```

Do **not** hand-edit the vendored files — in particular do not "correct" the
`inference.*` dotted strings in `configs/*.yaml`. They are lookup keys matched
against `LOCAL_CLASS_REGISTRY` in `inference/sam2_video_predictor.py`, not
import paths; the two sides only have to agree with each other. See
`__init__.py` for the full explanation of why the tree relocates unmodified.

BAT's own code lives outside this directory:

- `../bat_sec_runtime.py` — model discovery / load / cache / segmentation core
- `../bat_sec_segmenter.py` — the merged ComfyUI node
- `../bat_sec_advanced.py` — the advanced-parameters node

## Models

Weights are **not** vendored. `Bat_SecSegmenter` finds them through ComfyUI's
`sams` folder type, same as upstream:

```
ComfyUI/models/sams/SeC-4B-fp16.safetensors     (single file, recommended)
ComfyUI/models/sams/SeC-4B-bf16.safetensors
ComfyUI/models/sams/SeC-4B-fp32.safetensors
ComfyUI/models/sams/SeC-4B/                     (sharded HF layout)
```

Single-file checkpoints: <https://huggingface.co/VeryAladeen/Sec-4B>.
FP8 checkpoints load as FP16 — upstream removed FP8 quantization because it
produced NaNs in the language model during scene-change detection.
