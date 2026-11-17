#!/usr/bin/env python
"""
Checks for bat_prompt_hooks.py — the on_prompt handler that keeps output
pruning honest under ComfyUI's output cache.

Four claims, in the order a regression would hurt:

1. **Wiring a pruned output changes the cache key.** The bug this exists for:
   run HDR Tonal Composite with only image_out wired, wire linear_out, queue
   again — without the stamp the node is a cache hit and the new consumer gets
   the (1,1,1,3) stub from the first run.

2. **A second consumer on an already-built output changes nothing.** The key
   covers the prunable *groups*, not every link, so hanging another Preview off
   `image` must not re-run the node and everything downstream of it.

3. **The node never sees the stamp.** get_input_data only passes declared
   inputs, so the key must reach the cache signature and stop there — a node
   with **kwargs would otherwise receive a surprise argument.

4. **Nothing else is touched**, and a malformed prompt is passed through as is.

Claims 1 and 3 are checked against the real ComfyUI cache-key and input code
when it is importable (the pack under ComfyUI/custom_nodes, or ComfyUI on
PYTHONPATH); otherwise only the handler's own logic is checked.

    python tests/verify_prompt_hooks.py
"""

import asyncio
import copy
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


def load_hooks():
    spec = importlib.util.spec_from_file_location(
        "bat_prompt_hooks", os.path.join(PACK, "bat_prompt_hooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt_hdr(*consumed_slots, extra_preview_on=None):
    """HDR composite `1` fed by a loader `2`; one saver per consumed slot."""
    p = {
        "2": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "1": {"class_type": "Bat_HDRTonalComposite",
              "inputs": {"plate": ["2", 0], "hdr_ai": ["2", 0], "strength": 1.0}},
    }
    for i, slot in enumerate(consumed_slots):
        p[str(10 + i)] = {"class_type": "SaveImage", "inputs": {"images": ["1", slot]}}
    if extra_preview_on is not None:
        p["99"] = {"class_type": "PreviewImage", "inputs": {"images": ["1", extra_preview_on]}}
    return p


def stamp(hooks, prompt):
    body = {"prompt": copy.deepcopy(prompt), "client_id": "x"}
    return hooks.tag_output_consumers(body)["prompt"]


# ─── handler logic ───────────────────────────────────────────────────────────

def test_handler(hooks):
    print("\nhandler")
    key = hooks.INPUT_KEY

    a = stamp(hooks, prompt_hdr(0))
    b = stamp(hooks, prompt_hdr(0, 1))
    c = stamp(hooks, prompt_hdr(0, extra_preview_on=0))
    check("image_out only -> '0'", a["1"]["inputs"][key] == "0", a["1"]["inputs"].get(key))
    check("wiring linear_out changes the stamp",
          a["1"]["inputs"][key] != b["1"]["inputs"][key], b["1"]["inputs"].get(key))
    check("a second consumer of image_out does not",
          a["1"]["inputs"][key] == c["1"]["inputs"][key], c["1"]["inputs"].get(key))

    # Loader: the four layer outputs are built all-or-nothing, so they are one
    # group — wiring a second layer output must not change the key.
    loader = {
        "1": {"class_type": "Bat_Loader", "inputs": {"path": "x.exr"}},
        "5": {"class_type": "Bat_ExrLayer", "inputs": {"layers": ["1", 3]}},
    }
    l1 = stamp(hooks, loader)
    loader["6"] = {"class_type": "PreviewAny", "inputs": {"source": ["1", 4]}}
    l2 = stamp(hooks, loader)
    loader["7"] = {"class_type": "SaveAudio", "inputs": {"audio": ["1", 2]}}
    l3 = stamp(hooks, loader)
    check("Loader: layers group wanted", l1["1"]["inputs"][key] == "1", l1["1"]["inputs"].get(key))
    check("Loader: second layer output is the same group",
          l1["1"]["inputs"][key] == l2["1"]["inputs"][key])
    check("Loader: wiring audio changes the stamp", l3["1"]["inputs"][key] == "0,1",
          l3["1"]["inputs"].get(key))

    # Subgraph execution ids are "outer:inner" strings; links carry the same.
    sub = {
        "4:1": {"class_type": "Bat_AdvancedBlend", "inputs": {}},
        "4:2": {"class_type": "PreviewImage", "inputs": {"images": ["4:1", 2]}},
    }
    s = stamp(hooks, sub)
    check("subgraph ids are matched", s["4:1"]["inputs"][key] == "2", s["4:1"]["inputs"].get(key))

    untouched = stamp(hooks, prompt_hdr(0))
    check("other nodes are not stamped",
          all(key not in n["inputs"] for nid, n in untouched.items() if nid != "1"))

    for bad in ({"prompt": None}, {"prompt": ["nope"]}, {}, None,
                {"prompt": {"1": {"class_type": "Bat_Loader"}}}):
        try:
            out = hooks.tag_output_consumers(copy.deepcopy(bad))
            check(f"malformed {str(bad)[:40]!r} passes through", out == bad, repr(out))
        except Exception as exc:  # the handler must never raise into PromptServer
            check(f"malformed {str(bad)[:40]!r} passes through", False, repr(exc))


# ─── against the real ComfyUI ────────────────────────────────────────────────

def import_comfy():
    """ComfyUI's execution modules, or None. They initialise a torch device at
    import, so ask for the CPU one — a GPU is not what's under test here."""
    comfy_root = os.path.abspath(os.path.join(PACK, "..", ".."))
    if os.path.isfile(os.path.join(comfy_root, "execution.py")) and comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    try:
        argv, sys.argv = sys.argv, [sys.argv[0], "--cpu"]
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import execution
            import nodes
            from comfy_execution.caching import CacheKeySetInputSignature
            from comfy_execution.graph import DynamicPrompt
        finally:
            sys.argv = argv
        return execution, nodes, CacheKeySetInputSignature, DynamicPrompt
    except Exception as exc:
        print(f"  skip  real-ComfyUI checks ({type(exc).__name__}: {exc})")
        return None


class _FakeComposite:
    """Stands in for BatHDRTonalComposite: same class_type, a couple of inputs."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plate": ("IMAGE",), "hdr_ai": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0})}}
    RETURN_TYPES = ("IMAGE",) * 16
    FUNCTION = "composite"


class _FakeLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"


class _NoChange:
    async def get(self, node_id):
        return False


async def signature(CacheKeySetInputSignature, DynamicPrompt, prompt, node_id):
    keys = CacheKeySetInputSignature(DynamicPrompt(prompt), [node_id], _NoChange())
    await keys.add_keys([node_id])
    return keys.get_data_key(node_id)


def test_real_comfy(hooks):
    print("\nreal ComfyUI cache key / inputs")
    got = import_comfy()
    if got is None:
        return
    execution, nodes, CacheKeySetInputSignature, DynamicPrompt = got
    nodes.NODE_CLASS_MAPPINGS["Bat_HDRTonalComposite"] = _FakeComposite
    nodes.NODE_CLASS_MAPPINGS["LoadImage"] = _FakeLoad

    def sig(prompt):
        return asyncio.run(signature(CacheKeySetInputSignature, DynamicPrompt, prompt, "1"))

    raw_a, raw_b = prompt_hdr(0), prompt_hdr(0, 1)
    check("baseline: without the stamp, wiring linear_out is a cache HIT (the bug)",
          sig(raw_a) == sig(raw_b))
    a, b = stamp(hooks, raw_a), stamp(hooks, raw_b)
    check("with the stamp, wiring linear_out is a cache MISS", sig(a) != sig(b))
    c = stamp(hooks, prompt_hdr(0, extra_preview_on=0))
    check("a second consumer of image_out is still a HIT", sig(a) == sig(c))

    inputs = copy.deepcopy(b["1"]["inputs"])
    data, _missing, _v3 = execution.get_input_data(inputs, _FakeComposite, "1")
    check("the node never receives the stamp", hooks.INPUT_KEY not in data, sorted(data))


def main():
    hooks = load_hooks()
    test_handler(hooks)
    test_real_comfy(hooks)
    print()
    if _failures:
        print(f"FAILURES: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
