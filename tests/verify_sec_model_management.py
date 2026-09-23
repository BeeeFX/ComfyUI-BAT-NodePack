"""Prove the SeC model lives under ComfyUI's model management, and that the
SeC / Points Editor previews stay out of the prompt history.

1. **Model management** (needs ComfyUI importable — the pack under
   ComfyUI/custom_nodes, or ComfyUI on PYTHONPATH; run in ``--cpu`` mode):
   the cache entry is a ModelPatcher around a holder, ``load_models_gpu``
   registers and moves it, ComfyUI's "Unload models" and ``free_memory`` evict
   it while the cache keeps it for a rebuild-free reload, ``release`` /
   ``unload_all`` free it, and a failed (OOM) load leaves nothing cached.
   The real SeC stack is not built: a small nn.Module with a read-only
   ``device`` property — like transformers' PreTrainedModel — stands in.
2. **Segmenter node**: passes a VRAM estimate, holds no model reference when it
   releases, and returns its strip as a sidecar token that resolves.
3. **Points Editor**: the post-run plate is downscaled to the editor's 1024 px
   but carries the true size (bg_w / bg_h), so the coord space is unchanged.

Without ComfyUI, 1 prints a skip line and 2-3 still run.

    python tests/verify_sec_model_management.py
"""

import base64
import gc
import importlib.util
import io
import os
import sys
import tempfile
import types
import weakref

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


# ─── environment ─────────────────────────────────────────────────────────────

def import_comfy():
    """comfy.model_management in --cpu mode, or None. It initialises a torch
    device at import, so ask for the CPU one — a GPU is not what's under test."""
    comfy_root = os.path.abspath(os.path.join(PACK, "..", ".."))
    if os.path.isfile(os.path.join(comfy_root, "execution.py")) and comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    try:
        argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import comfy.model_management as mm
            import comfy.model_patcher  # noqa: F401
            import folder_paths
        finally:
            sys.argv = argv
        # Keep the sidecars this test writes out of the real temp folder.
        folder_paths.set_temp_directory(tempfile.mkdtemp(prefix="bat_secmm_"))
        return mm
    except Exception as exc:
        print(f"SKIP  model-management checks: ComfyUI not importable ({type(exc).__name__}: {exc})")
        return None


def stub_folder_paths():
    """Just enough of folder_paths for the pack modules to import."""
    fp = types.ModuleType("folder_paths")
    root = tempfile.mkdtemp(prefix="bat_secmm_")
    fp.models_dir = os.path.join(root, "models")
    fp.supported_pt_extensions = {".safetensors", ".pt", ".ckpt"}
    fp.folder_names_and_paths = {}

    def add_model_folder_path(name, path, is_default=False):
        paths = fp.folder_names_and_paths.setdefault(name, ([], set()))[0]
        if path not in paths:
            paths.append(path)

    fp.add_model_folder_path = add_model_folder_path
    fp.get_folder_paths = lambda name: list(fp.folder_names_and_paths[name][0])
    fp.get_temp_directory = lambda: os.path.join(root, "temp")
    sys.modules["folder_paths"] = fp


def load_pack():
    """The pack's SeC modules under a stand-in package (they import relatively)."""
    pkg = types.ModuleType("batpack_secmm")
    pkg.__path__ = [PACK]
    sys.modules["batpack_secmm"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(f"batpack_secmm.{name}", os.path.join(PACK, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    return (load("bat_sec_runtime"), load("bat_sec_segmenter"), load("bat_points_editor"),
            sys.modules["batpack_secmm.bat_ui_ref"])


# ─── 1. model management ─────────────────────────────────────────────────────

def test_model_management(rt, mm, torch):
    print("\n1. SeC is a ComfyUI-managed model")

    class FakeSec(torch.nn.Module):
        """Stands in for SeCModel: a read-only ``device``, like PreTrainedModel."""
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(64, 64)

        @property
        def device(self):
            return next(self.parameters()).device

    built = []

    def fake_build(spec, device, dtype, use_flash_attn, allow_mask_overlap):
        m = FakeSec().to(dtype)
        built.append(weakref.ref(m))
        return m

    moves = []
    saved = (rt._build_model, rt._preflight_dependencies, rt.list_sec_models, rt._SeCHolder.to)
    orig_to = rt._SeCHolder.to

    def spy_to(self, *a, **k):
        moves.append(a[0] if a else k)
        return orig_to(self, *a, **k)

    rt._build_model = fake_build
    rt._preflight_dependencies = lambda: None
    rt.list_sec_models = lambda: [{"name": "SeC-4B-fp16.safetensors", "path": "/fake", "is_single_file": True,
                                   "config_path": "/fake", "precision": "fp16"}]
    rt._SeCHolder.to = spy_to

    def ours_loaded():
        return [lm.model.model for lm in mm.current_loaded_models
                if lm.model is not None and isinstance(lm.model.model, rt._SeCHolder)]

    name = "SeC-4B-fp16.safetensors"
    try:
        model, key = rt.get_model(name, device="cpu", memory_required=rt.inference_memory_estimate(10))
        entry = rt._MODEL_CACHE[key]
        check("the cache entry is a ComfyUI ModelPatcher", type(entry).__name__ == "ModelPatcher")
        check("it wraps the holder; get_model returns the model inside",
              isinstance(entry.model, rt._SeCHolder) and entry.model.sec is model)
        check("load_models_gpu registered it", entry.model in ours_loaded())
        check("the patcher wrote holder.device, not SeC's read-only one", str(entry.model.device) == "cpu")
        check("ComfyUI moved the weights on load", len(moves) >= 1)
        n = len(moves)
        del model

        mm.unload_all_models()
        check("'Unload models' takes it off ComfyUI's loaded list", not ours_loaded())
        check("...moving it to the offload device", len(moves) > n)
        check("the cache keeps it (auto_unload_model off)", key in rt._MODEL_CACHE)

        model, key2 = rt.get_model(name, device="cpu")
        check("the next run reuses it without a rebuild and reloads it",
              len(built) == 1 and key2 == key and entry.model in ours_loaded())
        del model, entry

        mm.free_memory(1e30, mm.get_torch_device())
        check("free_memory (another model needs room) evicts it", not ours_loaded())

        rt.release(key)
        gc.collect()
        check("release drops the cache entry", key not in rt._MODEL_CACHE)
        check("release frees the model", built[0]() is None)
        check("nothing of ours is left on ComfyUI's list", not ours_loaded())

        real_load = mm.load_models_gpu

        def oom(*a, **k):
            raise torch.cuda.OutOfMemoryError("simulated OOM")

        mm.load_models_gpu = oom
        try:
            rt.get_model(name, device="cpu")
            check("a failed load raises", False)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            check("a failed (OOM) load leaves nothing cached", not rt._MODEL_CACHE)
        finally:
            mm.load_models_gpu = real_load

        rt.get_model(name, device="cpu")
        rt.unload_all("test")
        gc.collect()
        check("unload_all empties the cache and ComfyUI's list", not rt._MODEL_CACHE and not ours_loaded())
        check("every model built along the way is freed", all(r() is None for r in built))
    finally:
        rt._build_model, rt._preflight_dependencies, rt.list_sec_models, rt._SeCHolder.to = saved
        rt.unload_all()


# ─── 2. segmenter node ───────────────────────────────────────────────────────

def test_segmenter(rt, sg, ui_ref, torch):
    print("\n2. SeC Segmenter: VRAM estimate, reference hygiene, sidecar strip")
    check("the estimate grows with frames only when they live on the GPU",
          rt.inference_memory_estimate(100, True) == rt.inference_memory_estimate(1, True)
          and rt.inference_memory_estimate(100, False) > rt.inference_memory_estimate(1, False))

    seen = {}

    def fake_get_model(model_file, **kw):
        seen["mem"] = kw.get("memory_required")
        m = torch.nn.Linear(2, 2)
        seen["ref"] = weakref.ref(m)
        return m, "key"

    def fake_release(key):
        gc.collect()
        seen["alive_at_release"] = seen["ref"]() is not None

    saved = (sg.rt.get_model, sg.rt.release, sg.rt.segment)
    sg.rt.get_model, sg.rt.release = fake_get_model, fake_release
    sg.rt.segment = lambda model, frames, **kw: torch.zeros(frames.shape[0], frames.shape[1], frames.shape[2])
    try:
        out = sg.BatSecSegmenter().segment(torch.rand(5, 16, 24, 3), "", "[]", "[]", "", "[{}]",
                                           24, 16, 0, True)
    finally:
        sg.rt.get_model, sg.rt.release, sg.rt.segment = saved

    check("the node passes the VRAM estimate", seen["mem"] == rt.inference_memory_estimate(5, True))
    check("the node holds no model reference when it releases", seen["alive_at_release"] is False)
    check("the strip goes out as a sidecar token", list(out["ui"].keys()) == ["bat_ui"])
    ui = ui_ref.load_ui(out["ui"])
    check("the token resolves to the strip",
          len(ui.get("frames", [])) == 5 and ui.get("w") == [24] and ui.get("frame_count") == [5])


# ─── 3. Points Editor plate ──────────────────────────────────────────────────

def test_points_editor(pe, ui_ref, torch):
    print("\n3. Points Editor: downscaled plate, true coord space, sidecar")
    from PIL import Image

    def plate(w, h):
        r = pe.BatPointsEditor().pointdata("", "", w, h, '[{"x":1,"y":1}]', "", False, "[{}]", "xyxy",
                                           torch.rand(1, h, w, 3))
        check(f"{w}x{h}: the plate goes out as a sidecar token", list(r["ui"].keys()) == ["bat_ui"])
        ui = ui_ref.load_ui(r["ui"])
        img = Image.open(io.BytesIO(base64.b64decode(ui["bg_image"][0])))
        return img.size, ui.get("bg_w"), ui.get("bg_h")

    size, bw, bh = plate(3000, 1500)
    check("a large plate is downscaled to the editor's 1024 px", size == (1024, 512), f"got {size}")
    check("...and carries its true size for the coord space", bw == [3000] and bh == [1500])
    size, bw, bh = plate(64, 32)
    check("a small plate is sent as is", size == (64, 32) and bw == [64] and bh == [32])


def main():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        print(f"SKIP  all checks: {exc}")
        return
    mm = import_comfy()
    if mm is None:
        stub_folder_paths()
    rt, sg, pe, ui_ref = load_pack()

    if mm is not None:
        test_model_management(rt, mm, torch)
    test_segmenter(rt, sg, ui_ref, torch)
    test_points_editor(pe, ui_ref, torch)

    print()
    if _failures:
        print(f"FAILURES: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
