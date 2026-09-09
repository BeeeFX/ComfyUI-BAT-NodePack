"""
End-to-end verification of the BAT Profiler against a live ComfyUI.

Unlike verify_profiler.py (which exercises the pieces in isolation),
this drives a real server through the real execution path, so it is the
only thing that proves the hooks actually fire during a prompt.

    cd <comfy root>
    ./env/bin/python main.py --port 8791 --listen 127.0.0.1 &
    ./env/bin/python custom_nodes/ComfyUI-BAT-NodePack/tests/verify_profiler_e2e.py

Uses core CPU-only nodes so it needs no checkpoints. Note that VRAM
therefore reads zero here by design — the CUDA probes are covered in
verify_profiler.py instead.
"""
import json, time, urllib.request

BASE = "http://127.0.0.1:8791"
CID = "e2e-test-client"
W = H = 1024
N = 24
fails = []

def _server_up():
    """Is the ComfyUI this test needs actually running?

    Without this the whole script died on `URLError: Connection refused` and
    counted as a failure, which meant the pack's suite could never be green on
    a box with no server up — and a suite that is always red is a suite nobody
    reads. This is an opt-in integration test: no server means SKIP, not FAIL.
    """
    try:
        urllib.request.urlopen(BASE + "/system_stats", timeout=3).read()
        return True
    except Exception:
        return False


if not _server_up():
    print(f"SKIP  BAT Profiler end-to-end — nothing answering on {BASE}.\n"
          f"      Start one first, from the ComfyUI root:\n"
          f"        ./env/bin/python main.py --port 8791 --listen 127.0.0.1 &")
    raise SystemExit(0)


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("" if cond else f"  {detail}"))
    if not cond: fails.append(name)

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# A graph that needs no checkpoints: 8 frames of 768x768 is ~56 MB of
# float32 payload, big enough that byte accounting is unambiguous.
def make_prompt(color):
    """`color` varies so each run is a genuine cache miss unless we
    deliberately repeat it — otherwise the second run reuses the first's
    outputs and we end up measuring the cache, not the nodes."""
    # Sized to run for well over a second: the sampler ticks at 5 Hz,
    # so a sub-200ms graph legitimately yields no chart data and would
    # make this a test of nothing.
    return {
        "1": {"class_type": "EmptyImage",
              "inputs": {"width": W, "height": H, "batch_size": N, "color": color}},
        "2": {"class_type": "ImageInvert", "inputs": {"image": ["1", 0]}},
        "4": {"class_type": "ImageInvert", "inputs": {"image": ["2", 0]}},
        "5": {"class_type": "ImageInvert", "inputs": {"image": ["4", 0]}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    }

print("\n[routes]")
st = get("/bat/profiler/state")
check("state endpoint responds", "config" in st and "capabilities" in st)
check("off by default", st["config"]["enabled"] is False, str(st["config"]))
check("cuda detected", st["capabilities"]["cuda"] is True, str(st["capabilities"]))

print("\n[unarmed run is not profiled]")
r = post("/prompt", {"prompt": make_prompt(1), "client_id": "some-other-client"})
pid_unarmed = r["prompt_id"]
for _ in range(120):
    if get(f"/history/{pid_unarmed}"): break
    time.sleep(0.5)
try:
    get(f"/bat/profiler/run/{pid_unarmed}")
    check("unarmed run produced no profile", False, "a profile was recorded")
except urllib.error.HTTPError as e:
    check("unarmed run produced no profile", e.code == 404, f"HTTP {e.code}")

print("\n[armed run is profiled]")
st = post("/bat/profiler/arm", {"client_id": CID, "enabled": True})
check("arm reports enabled", st["config"]["enabled"] is True, str(st["config"]))

post("/bat/profiler/claim", {"prompt_id": "pre", "workflow": "wf/e2e.json"})
r = post("/prompt", {"prompt": make_prompt(2), "client_id": CID})
pid = r["prompt_id"]
post("/bat/profiler/claim", {"prompt_id": pid, "workflow": "wf/e2e.json"})

for _ in range(120):
    if get(f"/history/{pid}"): break
    time.sleep(0.5)
time.sleep(1.0)

run = get(f"/bat/profiler/run/{pid}")
nodes = {n["class_type"]: n for n in run["nodes"]}
print("    recorded:", json.dumps(
    {k: {"t": round(v["duration"], 4), "out": v["bytes_out"], "desc": v["desc_out"],
         "ram": v["ram_delta"], "vpeak": v["vram_peak"]} for k, v in nodes.items()}, indent=6))

check("run recorded", run["status"] == "ok", run["status"])
check("all five nodes profiled", len(run["nodes"]) == 5, str(list(nodes)))
check("workflow attributed", run["workflow"] == "wf/e2e.json", str(run["workflow"]))
check("timings non-zero", all(n["duration"] > 0 for n in run["nodes"]))
check("execution order recorded", run["nodes"][0]["class_type"] == "EmptyImage",
      run["nodes"][0]["class_type"])

# 8 x 768 x 768 x 3 x 4 bytes = 56,623,104
expect = N * W * H * 3 * 4
check("EmptyImage payload measured exactly",
      nodes["EmptyImage"]["bytes_out"] == expect,
      f"got {nodes['EmptyImage']['bytes_out']} expected {expect}")
check("payload described",
      f"{N}x{H}x{W}x3" in (nodes["EmptyImage"]["desc_out"] or ""),
      str(nodes["EmptyImage"]["desc_out"]))
check("ImageInvert sees its input",
      nodes["ImageInvert"]["bytes_in"] == expect,
      f"got {nodes['ImageInvert']['bytes_in']}")
check("RAM delta plausible for a 56MB alloc",
      nodes["EmptyImage"]["ram_delta"] > 1_000_000,
      f"got {nodes['EmptyImage']['ram_delta']}")
samples = run.get("samples", [])
check("samples captured for the chart", len(samples) > 1, f"got {len(samples)}")
check("samples carry RAM and a ceiling",
      all(s.get("ram", 0) > 0 and s.get("sys_total", 0) > 0 for s in samples),
      str(samples[:1]))
check("samples span the run",
      len(samples) > 1 and (samples[-1]["t"] - samples[0]["t"]) > 0.1,
      f"span {samples[-1]['t'] - samples[0]['t'] if len(samples)>1 else 0:.3f}s")
print(f"    {len(samples)} samples over "
      f"{samples[-1]['t'] - samples[0]['t']:.2f}s, peak RAM "
      f"{max(s['ram'] for s in samples)/2**30:.2f} GB")
check("baseline captured", run["baseline"].get("sys_total", 0) > 0, str(run["baseline"]))

print("\n[cache hits are marked, not dropped]")
r2 = post("/prompt", {"prompt": make_prompt(2), "client_id": CID})  # same -> all cached
pid2 = r2["prompt_id"]
for _ in range(120):
    if get(f"/history/{pid2}"): break
    time.sleep(0.5)
time.sleep(1.0)
run2 = get(f"/bat/profiler/run/{pid2}")
cached = [n for n in run2["nodes"] if n["cached"]]
check("re-run marks every node cached", len(cached) == 5,
      f"cached={len(cached)} of {len(run2['nodes'])}: "
      + str([(n["class_type"], n["cached"], n["skipped"]) for n in run2["nodes"]]))
check("cached nodes still listed", len(run2["nodes"]) == 5,
      f"got {len(run2['nodes'])} rows")

print("\n[disarm stops collection]")
post("/bat/profiler/arm", {"client_id": CID, "enabled": False})
r3 = post("/prompt", {"prompt": make_prompt(3), "client_id": CID})
pid3 = r3["prompt_id"]
for _ in range(120):
    if get(f"/history/{pid3}"): break
    time.sleep(0.5)
try:
    get(f"/bat/profiler/run/{pid3}")
    check("disarmed run not profiled", False, "still recording after disarm")
except urllib.error.HTTPError as e:
    check("disarmed run not profiled", e.code == 404, f"HTTP {e.code}")

print("\n" + ("FAILURES: " + ", ".join(fails) if fails else "End-to-end: all checks passed."))
raise SystemExit(1 if fails else 0)
