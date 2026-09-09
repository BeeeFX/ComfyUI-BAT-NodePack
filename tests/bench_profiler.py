"""
Measure the profiler's overhead: the same graph, armed vs unarmed.

Needs a live ComfyUI on port 8791 (see verify_profiler_e2e.py).

Read the result as a *per-node constant*, not as a percentage. This
benchmark deliberately uses 29 trivial nodes so the fixed cost is
visible, which makes the percentage close to a worst case. On a real
graph, where a node takes tenths of a second to minutes, the same
constant disappears into the noise.
"""
import json, statistics, time, urllib.request

BASE, CID = "http://127.0.0.1:8791", "bench-client"

def post(p, b):
    req = urllib.request.Request(BASE + p, data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

def get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=60).read())

def prompt(c):
    # 30 nodes, so per-node probe cost is visible rather than lost in noise.
    g = {"1": {"class_type": "EmptyImage",
               "inputs": {"width": 512, "height": 512, "batch_size": 8, "color": c}}}
    prev = "1"
    for i in range(2, 30):
        g[str(i)] = {"class_type": "ImageInvert", "inputs": {"image": [prev, 0]}}
        prev = str(i)
    g["99"] = {"class_type": "PreviewImage", "inputs": {"images": [prev, 0]}}
    return g

def run(color, client):
    t = time.perf_counter()
    pid = post("/prompt", {"prompt": prompt(color), "client_id": client})["prompt_id"]
    while not get(f"/history/{pid}"):
        time.sleep(0.02)
    return time.perf_counter() - t

color = 1000
def series(client, n=7):
    global color
    out = []
    for _ in range(n):
        color += 1
        out.append(run(color, client))
    return sorted(out)[1:-1]     # drop best and worst

post("/bat/profiler/arm", {"client_id": CID, "enabled": False})
warm = series("warmup", 3)

off = series("unarmed-client")
post("/bat/profiler/arm", {"client_id": CID, "enabled": True})
on = series(CID)
post("/bat/profiler/arm", {"client_id": CID, "enabled": False})
off2 = series("unarmed-client")

# Attribute the cost: re-measure with GPU sync off, then with the
# payload walker's budget cut to nothing.
post("/bat/profiler/config", {"sync_cuda": False})
post("/bat/profiler/arm", {"client_id": CID, "enabled": True})
nosync = series(CID)
post("/bat/profiler/config", {"sync_cuda": True})
post("/bat/profiler/arm", {"client_id": CID, "enabled": False})

base = statistics.mean(off + off2)
armed = statistics.mean(on)
print(f"  29-node graph, 5 timed runs each (best/worst dropped)")
print(f"  unarmed : {base*1000:7.1f} ms  (n={len(off+off2)})")
print(f"  armed   : {armed*1000:7.1f} ms  (n={len(on)})")
ns = statistics.mean(nosync)
print(f"  armed, no GPU sync : {ns*1000:7.1f} ms")
print(f"  delta   : {(armed-base)*1000:+7.1f} ms total"
      f"  =  {(armed-base)*1000/29:+.2f} ms per node"
      f"  ({(armed/base-1)*100:+.1f}%)")
print(f"  of which GPU sync  : {(armed-ns)*1000:+7.1f} ms"
      f"  =  {(armed-ns)*1000/29:+.2f} ms per node")
print(f"  probes + payload   : {(ns-base)*1000:+7.1f} ms"
      f"  =  {(ns-base)*1000/29:+.2f} ms per node")
