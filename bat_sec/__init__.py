"""
bat_sec — vendored SeC (Segment Concept) inference stack.
=========================================================

Third-party code, carried here so the BAT pack's ``Bat_SecSegmenter`` node
works without ``Comfyui-SecNodes`` installed alongside it. See NOTICE.md for
attribution and LICENSE for terms (Apache 2.0).

Layout — mirrors the upstream pack, because the vendored tree is
byte-identical and MUST stay that way to keep working:

    inference/      SeC + SAM2 model code. All intra-tree imports are
                    relative, so the tree relocates cleanly.
    configs/        SAM2 hydra YAMLs. ``inference/sam2_video_predictor.py``
                    loads these as ``__file__/../configs``, which resolves to
                    THIS directory's sibling — do not move one without the
                    other.
    model_config/   SeC-4B config + tokenizer. Fed to
                    ``SeCConfig.from_pretrained`` / ``AutoTokenizer`` for
                    single-file (.safetensors) checkpoints, which carry
                    weights only.

Why no edits were needed when relocating:

  * Every intra-tree import is relative (``from .sam2.modeling...``).
  * The ``_target_: inference.sam2.modeling...`` strings in configs/*.yaml
    are *opaque registry keys*, not import paths — ``build_sam2_video_predictor``
    resolves them through its own LOCAL_CLASS_REGISTRY dict and raises on an
    unknown key rather than importing anything.
  * ``init_sam2_hydra()`` calls ``initialize_config_module("inference.sam2.configs")``,
    but hydra resolves search paths lazily and nothing in this stack ever
    composes a hydra config (configs are read with a plain ``OmegaConf.load``
    on an absolute path), so the stale module name never gets imported.

Consequence: re-vendoring is a straight rsync from upstream. Resist the urge
to "fix" the ``inference.*`` strings — rewriting them breaks the registry
lookup, because the keys and the YAML values have to match each other, not
the filesystem.
"""
