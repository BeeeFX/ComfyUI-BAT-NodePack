"""
BAT — make output pruning safe under ComfyUI's output cache.

The problem
-----------
Four nodes skip work for outputs nothing reads: Loader (audio, EXR layers),
Video Loader (audio), Advanced Blend (detail / difference) and HDR Tonal
Composite (each image_out / linear_out pair). They find out what is wired by
scanning the hidden PROMPT for links from their own id, and hand an unwired
output a stub — an empty dict, None, or a (1,1,1,3) tensor.

That is only correct while the node actually re-executes. ComfyUI's cache
signature is the node's class, IS_CHANGED and its own *inputs* plus its
ancestors' (comfy_execution/caching.py, get_immediate_node_signature). What
reads the node's outputs is not part of it, and IS_CHANGED cannot help: it is
evaluated with PROMPT = {} (execution.py, IsChangedCache.get → get_input_data
with no dynprompt). So: run with only `image` wired, then wire `linear_out` to
an EXR saver and queue again — the composite is a cache hit and the saver gets
the 1×1 stub from the previous run. Nothing errors.

The fix
-------
Put the wiring into the node's inputs. PromptServer runs every
`on_prompt_handler` on the /prompt body before validation (server.py,
post_prompt → trigger_on_prompt), so this hook writes a `_bat_consumers` key
into each pruning node's inputs listing which *prunable* outputs are read. The
cache signature covers every key of `inputs`, declared or not, while
validation and get_input_data only look at declared inputs — so the key
changes the cache identity and nothing else. No widget, no workflow-format
change, and it works for API submits and both frontends alike.

Only the prunable groups go into the key, so wiring a second consumer onto an
output that was already being produced does not re-run anything. Wiring or
unwiring a prunable output does re-run the node and whatever is downstream of
it — that is the price of the node having skipped the work in the first place.
"""

import logging

logger = logging.getLogger("BAT.prompt_hooks")

INPUT_KEY = "_bat_consumers"

# class_type → groups of output slots that the node computes together. A group
# counts as wanted when any slot in it is read. The slot numbers are the nodes'
# RETURN_NAMES positions; they must track the pruning in each node:
#   bat_loader.py          _AUDIO_SLOT, _LAYER_OUTPUT_SLOTS (all-or-nothing)
#   bat_video_loader.py    _AUDIO_SLOT
#   bat_advanced_blend.py  want[k] for k in range(3)
#   bat_hdr_tonal_composite.py  want_display / want_linear per version
_PRUNED_GROUPS = {
    "Bat_Loader":            ((2,), (3, 4, 5, 6)),
    "Bat_VideoLoader":       ((2,),),
    "Bat_AdvancedBlend":     ((0,), (1,), (2,)),
    "Bat_HDRTonalComposite": tuple((s,) for s in range(16)),
}


def tag_output_consumers(json_data):
    """on_prompt handler: stamp each pruning node with the groups it must build.

    Never raises into the caller (PromptServer would log a traceback per
    prompt) and never rejects a prompt — anything it cannot read is left alone,
    which is exactly today's behaviour.
    """
    try:
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        if not isinstance(prompt, dict):
            return json_data

        targets = {str(nid): _PRUNED_GROUPS[node["class_type"]]
                   for nid, node in prompt.items()
                   if isinstance(node, dict)
                   and node.get("class_type") in _PRUNED_GROUPS}
        if not targets:
            return json_data

        # Same reading of a link as the nodes' own _consumed_slots:
        # [source_id, int slot].
        used = {nid: set() for nid in targets}
        for node in prompt.values():
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                continue
            for value in inputs.values():
                if (isinstance(value, (list, tuple)) and len(value) == 2
                        and isinstance(value[1], int)):
                    slots = used.get(str(value[0]))
                    if slots is not None:
                        slots.add(value[1])

        for nid, groups in targets.items():
            inputs = prompt[nid].get("inputs")
            if not isinstance(inputs, dict):
                continue
            wanted = [i for i, group in enumerate(groups)
                      if any(s in used[nid] for s in group)]
            inputs[INPUT_KEY] = ",".join(map(str, wanted))
    except Exception as exc:
        logger.warning("[BAT] could not tag output consumers (%s); "
                       "output pruning may reuse a stale cached result", exc)
    return json_data


_installed = False


def install():
    """Register the handler once. A missing server (tests, headless import)
    just leaves the nodes as they were."""
    global _installed
    if _installed:
        return
    try:
        import server
        server.PromptServer.instance.add_on_prompt_handler(tag_output_consumers)
        _installed = True
    except Exception as exc:
        logger.warning("[BAT] prompt hook not installed (%s)", exc)
