"""Prove the SAM 3.1 editor's pure JS agrees with the Python it mirrors.

  keepFlagsFor (web)  == bat_sam31_runtime.match_instances  — the keep/drop
      choices shown on the canvas must be the ones the run applies.
  trimPrompt          — the ▶ Track prompt holds exactly the node, its
      ancestors and the sink; nothing downstream, nothing unrelated.

No node binary on this box: the functions are sliced out of
web/bat_sam31_segmenter.js and run under quickjs. CPU only.

    env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_sam31_web.py
"""
import json
import os
import random
import sys
import types

import quickjs

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
JS = os.path.join(PACK, "web", "bat_sam31_segmenter.js")
FAILURES = []


def check(label, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'} {label}" + ("" if condition or not detail else f"\n         {detail}"))
    if not condition:
        FAILURES.append(label)


src = open(JS, encoding="utf-8").read()


def grab(name):
    start = src.index(f"function {name}(")
    depth, i = 0, src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1


ctx = quickjs.Context()
ctx.eval('const SINK_TYPE = "Bat_Sam31TrackSink";\n'
         + "\n".join(grab(n) for n in ("boxIoU", "keepFlagsFor", "trimPrompt")))

# match_instances without importing the runtime (it needs ComfyUI): exec just that function.
rt_src = open(os.path.join(PACK, "bat_sam31_runtime.py"), encoding="utf-8").read()
a = rt_src.index("def match_instances(")
b = rt_src.index("\n# ──", a)
ns = {}
exec(rt_src[a:b], ns)
match_instances = ns["match_instances"]

print("keepFlagsFor == match_instances (2000 random cases)")
rng = random.Random(7)


def rbox():
    x0, y0 = rng.random() * .8, rng.random() * .8
    return [x0, y0, x0 + rng.random() * .2 + .01, y0 + rng.random() * .2 + .01]


mismatch = None
for case in range(2000):
    found = [{"box": rbox()} for _ in range(rng.randint(0, 6))]
    saved = []
    for f in found:
        if rng.random() < .6:   # a jittered copy of a found box, so IoU straddles 0.5
            j = [v + rng.uniform(-.04, .04) for v in f["box"]]
            saved.append({"box": j, "keep": rng.random() < .5})
    saved += [{"box": rbox(), "keep": rng.random() < .5} for _ in range(rng.randint(0, 2))]
    rng.shuffle(saved)
    py = match_instances(found, saved)
    js = ctx.eval(f"JSON.stringify(keepFlagsFor({json.dumps(found)}, {json.dumps(saved)}))")
    if py != json.loads(js):
        mismatch = (found, saved, py, js)
        break
check("identical on every case", mismatch is None, str(mismatch)[:300])
check("missing 'keep' counts as kept (both)",
      match_instances([{"box": [0, 0, .5, .5]}], [{"box": [0, 0, .5, .5]}]) == [True]
      and json.loads(ctx.eval('JSON.stringify(keepFlagsFor([{box:[0,0,.5,.5]}],[{box:[0,0,.5,.5]}]))')) == [True])

print("trimPrompt")
prompt = {
    "1": {"class_type": "Bat_VideoLoader", "inputs": {"path": "/a.mov", "start_frame": 0}},
    "2": {"class_type": "Bat_Sam31AdvancedParams", "inputs": {"max_objects": 16}},
    "3": {"class_type": "Bat_Sam31Segmenter", "inputs": {"frames": ["1", 0], "advanced": ["2", 0],
                                                         "prompt_state": "[1, 2]", "frame_index_select": 4}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 1]}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
    "6:7": {"class_type": "Bat_Sam31Segmenter", "inputs": {"frames": ["5", 0], "prompt_state": ""}},
}
out = json.loads(ctx.eval(f"JSON.stringify(trimPrompt({json.dumps(prompt)}, ['3']))"))
check("keeps node + ancestors + sink only", sorted(out) == ["1", "2", "3", "bat_sam31_track_0"], sorted(out))
check("sink wired to masks output", out["bat_sam31_track_0"] == {"class_type": "Bat_Sam31TrackSink",
                                                                "inputs": {"masks": ["3", 0]}})
check("a 2-item widget value with a string tail is not a link", "1" in out and out["3"]["inputs"]["prompt_state"] == "[1, 2]")
out = json.loads(ctx.eval(f"JSON.stringify(trimPrompt({json.dumps(prompt)}, ['6:7']))"))
check("subgraph path id", sorted(out) == ["5", "6:7", "bat_sam31_track_0"], sorted(out))

print()
print("All checks passed." if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
