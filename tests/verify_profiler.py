"""
Verify the BAT Profiler.

Run:  ./env/bin/python ComfyUI/custom_nodes/ComfyUI-BAT-NodePack/tests/verify_profiler.py

Covers the three things that are easy to get quietly wrong:

* the **arming gate** — an unarmed client must take the original code
  path, byte for byte, or we have slowed down everyone on the box;
* the **payload walker** — it must count real tensor bytes, stay inside
  its visit budget, and retain no references to what it walked (a
  profiler that pins output tensors causes the OOM it diagnoses);
* the **record accumulator** — lazy and async nodes re-enter execute(),
  and the record has to sum time while taking a max of peaks.

Also parses both frontend files and exercises the persistence store
under quickjs, since there is no node binary on this box.
"""

import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.abspath(os.path.join(PACK, "..", ".."))

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


def load_profiler():
    """Load bat_profiler.py standalone.

    Its only package-relative import is the REST module, inside
    install() and already guarded, so a flat load is faithful to what
    ComfyUI does apart from the routes.
    """
    sys.path.insert(0, COMFY)
    spec = importlib.util.spec_from_file_location(
        "bat_profiler_test", os.path.join(PACK, "bat_profiler.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────
print("\n[1] payload walker")
prof = load_profiler()

import torch  # noqa: E402


def t_payload():
    a = torch.zeros((4, 64, 64, 3), dtype=torch.float32)   # 4*64*64*3*4 = 196608
    n, desc = prof.payload_bytes(a)
    check("tensor bytes", n == 196608, f"got {n}")
    check("tensor described", desc and "4x64x64x3" in desc and "float32" in desc, f"got {desc}")

    # ComfyUI hands node inputs as {name: [value, ...]} — nested lists
    # and dicts both have to be walked or IMAGE inputs read as zero.
    nested = {"image": [a], "latent": [{"samples": torch.zeros((1, 4, 8, 8))}]}
    n2, _ = prof.payload_bytes(nested)
    check("nested dict/list walked", n2 == 196608 + 1 * 4 * 8 * 8 * 4, f"got {n2}")

    # Non-tensor scalars are not "transfer" and must not inflate counts.
    n3, _ = prof.payload_bytes({"seed": [12345], "text": ["a prompt string"]})
    check("scalars ignored", n3 == 0, f"got {n3}")

    # Budget: a pathological list must not turn profiling into the
    # bottleneck. 10k one-element tensors, budget is 4000 visits.
    big = [torch.zeros((1,), dtype=torch.uint8) for _ in range(10000)]
    n4, _ = prof.payload_bytes(big)
    check("visit budget enforced", 0 < n4 < 10000, f"got {n4}")

    # Retention: the walker must not keep the objects alive.
    import weakref, gc
    victim = torch.zeros((16, 16))
    ref = weakref.ref(victim)
    prof.payload_bytes([victim])
    del victim
    gc.collect()
    check("retains no references", ref() is None,
          "walker is pinning tensors — this would cause OOMs")


t_payload()

# ─────────────────────────────────────────────────────────────────────
print("\n[2] arming gate")


def t_cuda():
    """The e2e run uses CPU-only core nodes, so VRAM reads zero there and
    proves nothing. Exercise the probes directly instead."""
    if prof._device() is None:
        print("  SKIP  no CUDA device")
        return
    base = prof.cuda_stats(include_device=True)
    check("allocator stats readable", "alloc" in base and "reserved" in base, str(base))
    check("device total readable", base.get("dev_total", 0) > 0, str(base))

    prof.cuda_reset_peak()
    before = prof.cuda_peak()
    big = torch.zeros((256, 1024, 1024), dtype=torch.float16, device="cuda")  # 512 MB
    after_alloc = prof.cuda_stats()
    peak = prof.cuda_peak()
    check("allocation is seen", after_alloc["alloc"] - base["alloc"] >= 512 * 2**20,
          f"delta {after_alloc['alloc'] - base['alloc']}")
    check("peak tracks the allocation", peak - before >= 512 * 2**20,
          f"peak delta {peak - before}")

    del big
    torch.cuda.empty_cache()
    freed = prof.cuda_stats()
    check("release is seen", freed["alloc"] < after_alloc["alloc"],
          f"{freed['alloc']} vs {after_alloc['alloc']}")

    # Peak reset is what makes per-node peaks per-node rather than
    # cumulative; if this silently no-ops every row shows the same number.
    prof.cuda_reset_peak()
    check("peak reset works", prof.cuda_peak() <= freed["alloc"] + 2**20,
          f"peak {prof.cuda_peak()} alloc {freed['alloc']}")

    # And it must honour the config switch.
    prof.CONFIG.reset_peak = False
    spike = torch.zeros((128, 1024, 1024), dtype=torch.float16, device="cuda")
    hi = prof.cuda_peak()
    del spike
    torch.cuda.empty_cache()
    prof.cuda_reset_peak()
    check("reset_peak=False leaves the counter alone", prof.cuda_peak() == hi,
          f"{prof.cuda_peak()} vs {hi}")
    prof.CONFIG.reset_peak = True


def t_arming():
    check("off by default", prof.CONFIG.default_enabled is False)
    check("unknown client not armed", prof.is_armed("nobody") is False)
    check("no client falls back to default", prof.is_armed(None) is False)

    prof.arm("client-A", True)
    check("armed client recognised", prof.is_armed("client-A") is True)
    check("other client still unarmed", prof.is_armed("client-B") is False,
          "one artist arming must not arm the box")

    prof.arm("client-A", False)
    check("disarm works", prof.is_armed("client-A") is False)

    # Registry must stay bounded — a browser that never comes back
    # should not leak an entry forever.
    for i in range(prof._ARMED_LIMIT + 40):
        prof.arm(f"c{i}", True)
    check("registry bounded", len(prof._armed) <= prof._ARMED_LIMIT,
          f"grew to {len(prof._armed)}")
    prof._armed.clear()


t_arming()

# ─────────────────────────────────────────────────────────────────────
print("\n[2b] CUDA probes")


t_cuda()

# ─────────────────────────────────────────────────────────────────────
print("\n[3] record accumulation across re-entries")


def t_records():
    run = prof.RunRecord("p1")
    rec = run.get("7", "KSampler", "7")
    # First entry.
    rec.entries = 1
    rec.duration = 2.0
    rec.ram_start, rec.ram_end = 1000, 3000
    rec.vram_peak = 500
    # Second entry, as a lazy/async node produces.
    rec.entries += 1
    rec.duration += 3.0
    rec.ram_end = 4000
    rec.vram_peak = max(rec.vram_peak, 200)

    d = rec.to_dict()
    check("time sums across entries", d["duration"] == 5.0, f"got {d['duration']}")
    check("delta spans first start to last end", d["ram_delta"] == 3000, f"got {d['ram_delta']}")
    check("peak takes the max", d["vram_peak"] == 500, f"got {d['vram_peak']}")
    check("re-entry count kept", d["entries"] == 2)

    s = run.summary()
    check("summary counts nodes", s["node_count"] == 1)
    cached = run.get("8", "LoadImage", "8")
    cached.cached = True
    s = run.summary()
    check("cached excluded from executed", s["executed_count"] == 1 and s["cached_count"] == 1,
          f"got {s['executed_count']}/{s['cached_count']}")


t_records()

# ─────────────────────────────────────────────────────────────────────
print("\n[4] hooks install against the real execution module")


def t_install():
    try:
        import execution
    except Exception as e:
        check("import execution", False, f"{e!r}")
        return

    orig_execute = execution.execute
    orig_god = execution.get_output_data
    orig_async = execution.PromptExecutor.execute_async

    ok = prof.install()
    check("install() succeeds", ok is True)
    check("execute wrapped", execution.execute is not orig_execute)
    check("get_output_data wrapped", execution.get_output_data is not orig_god)
    check("execute_async wrapped", execution.PromptExecutor.execute_async is not orig_async)

    # The signatures we depend on must still be the ones we read.
    import inspect
    sig = list(inspect.signature(orig_execute).parameters)
    check("execute arg 3 is current_item", sig[3] == "current_item", f"got {sig[:8]}")
    check("execute arg 6 is prompt_id", sig[6] == "prompt_id", f"got {sig[:8]}")
    gsig = list(inspect.signature(orig_god).parameters)
    check("get_output_data args", gsig[:4] == ["prompt_id", "unique_id", "obj", "input_data_all"],
          f"got {gsig[:4]}")

    # An unarmed prompt must reach the original function untouched.
    import asyncio
    called = {"n": 0}

    async def fake_orig(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        called["n"] += 1
        return "passthrough"

    gate = prof._wrap_execute_async(fake_orig)
    res = asyncio.run(gate(types.SimpleNamespace(success=True), {}, "pX",
                           {"client_id": "not-armed"}, []))
    check("unarmed prompt passes through", res == "passthrough" and called["n"] == 1)
    check("unarmed prompt starts no run", prof._current is None,
          "an unarmed client is being profiled")

    prof.arm("armed-client", True)
    seen = {}

    async def fake_orig2(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        seen["current"] = prof._current
        return "ok"

    gate2 = prof._wrap_execute_async(fake_orig2)
    asyncio.run(gate2(types.SimpleNamespace(success=True), {}, "pY",
                      {"client_id": "armed-client"}, []))
    check("armed prompt opens a run", seen.get("current") is not None)
    check("run closed at end", prof._current is None)
    check("run archived", any(r.prompt_id == "pY" for r in prof._history))
    prof.arm("armed-client", False)


t_install()

# ─────────────────────────────────────────────────────────────────────
print("\n[5] frontend")


def t_frontend():
    try:
        import quickjs
    except Exception as e:
        check("quickjs available", False, f"{e!r}")
        return

    web = os.path.join(PACK, "web")

    def parse_only(path):
        """Parse without executing: wrap the body in a function that is
        never called. ESM syntax is illegal there, so strip it first."""
        src = open(os.path.join(web, path)).read()
        lines = []
        for ln in src.splitlines():
            if ln.startswith("import "):
                continue          # ESM import — illegal inside a function
            if ln.startswith("export "):
                ln = ln[len("export "):]   # keep the declaration itself
            lines.append(ln)
        body = "\n".join(lines)
        ctx = quickjs.Context()
        try:
            ctx.eval("(function(){" + body + "\n})")
            return None
        except Exception as e:
            return str(e)

    for f in ("bat_profiler_store.js", "bat_profiler.js"):
        err = parse_only(f)
        check(f"{f} parses", err is None, err or "")

    # Exercise the store for real against a stub localStorage.
    src = open(os.path.join(web, "bat_profiler_store.js")).read()
    src = src.replace("export class", "class").replace("export function", "function")
    src = src.replace("export const", "const")
    ctx = quickjs.Context()
    ctx.eval("""
      var _store = {};
      var localStorage = {
        getItem: function(k){ return (k in _store) ? _store[k] : null; },
        setItem: function(k,v){ _store[k] = String(v); },
        removeItem: function(k){ delete _store[k]; }
      };
      var setTimeout = function(fn){ return 0; };
      var clearTimeout = function(){};
    """)
    ctx.eval(src)

    ctx.eval("""
      var s = new ProfilerStore(localStorage);
      s.beginRun("wf/alpha.json", {prompt_id:"r1", started: 100});
      s.upsertNode("r1", {node_id:"3", class_type:"KSampler", duration:5}, "Sampler");
      s.beginRun("wf/beta.json",  {prompt_id:"r2", started: 200});
      s.upsertNode("r2", {node_id:"9", class_type:"VAEDecode", duration:1}, "Decode");
    """)
    a = ctx.eval('JSON.stringify(s.listRuns("wf/alpha.json").map(r=>r.prompt_id))')
    b = ctx.eval('JSON.stringify(s.listRuns("wf/beta.json").map(r=>r.prompt_id))')
    check("per-workflow isolation", a == '["r1"]' and b == '["r2"]', f"alpha={a} beta={b}")

    n = ctx.eval('Object.keys(s.listRuns("wf/alpha.json")[0].nodes).length')
    check("node filed into its own run", n == 1, f"got {n}")

    # Retention: the oldest run drops once the cap is hit.
    ctx.eval("""
      for (var i=0;i<10;i++) s.beginRun("wf/gamma.json", {prompt_id:"g"+i, started:i});
    """)
    g = ctx.eval('s.listRuns("wf/gamma.json").length')
    check("run retention capped", g == ctx.eval("MAX_RUNS_PER_WORKFLOW"), f"got {g}")
    newest = ctx.eval('s.listRuns("wf/gamma.json")[0].prompt_id')
    check("newest run kept first", newest == "g9", f"got {newest}")

    # Sample decimation must halve resolution, not truncate the head:
    # a long run has to keep its whole shape.
    ctx.eval("""
      s.beginRun("wf/delta.json", {prompt_id:"d1", started:0});
      var many = []; for (var i=0;i<2000;i++) many.push({t:i, ram:i});
      s.pushSamples("d1", many);
    """)
    cnt = ctx.eval('s.findRun("d1").run.samples.length')
    first = ctx.eval('s.findRun("d1").run.samples[0].t')
    last = ctx.eval('s.findRun("d1").run.samples[s.findRun("d1").run.samples.length-1].t')
    check("samples decimated not truncated",
          cnt <= ctx.eval("MAX_SAMPLES_PER_RUN") and first == 0 and last >= 1990,
          f"n={cnt} first={first} last={last}")

    # Persistence: a flush must survive a fresh store over the same
    # storage — this is the crash-survival path.
    ctx.eval("s.flush(); var s2 = new ProfilerStore(localStorage);")
    revived = ctx.eval('JSON.stringify(s2.listRuns("wf/alpha.json").map(r=>r.prompt_id))')
    check("survives a reload", revived == '["r1"]', f"got {revived}")

    # A workflow with no runs must not see anyone else's.
    empty = ctx.eval('s2.listRuns("wf/never-run.json").length')
    check("unknown workflow is empty", empty == 0, f"got {empty}")


t_frontend()

# ─────────────────────────────────────────────────────────────────────
print("\n[5b] queuePrompt passthrough")


def t_queue_passthrough():
    """The profiler wraps `api.queuePrompt`, which every queue on the box
    goes through — so the wrapper must be transparent.

    It was not. Declared `(number, prompt)`, it dropped the third
    argument, which is where the frontend puts `partialExecutionTargets`
    (the per-node play button) and `previewMethod`. The backend then saw
    a plain full queue and cooked the WHOLE graph — API nodes included —
    on every single-node run, for anyone who merely had the pack in
    custom_nodes. Reported from the wild 2026-09-15.

    So this runs the shipped IIFE verbatim against a recording stub and
    asserts every argument survives. Arity is the thing under test: a
    fixed-arity wrapper must fail here, and so must the next core
    signature that grows a parameter."""
    try:
        import quickjs
    except Exception as e:
        check("quickjs available", False, f"{e!r}")
        return

    src = open(os.path.join(PACK, "web", "bat_profiler.js")).read()

    # Pull the wrapper out of the shipped file rather than restating it,
    # so the test cannot drift away from what actually ships.
    start = src.index("(function patchQueuePrompt() {")
    end = src.index("})();", start) + len("})();")
    iife = src[start:end]

    ctx = quickjs.Context()
    ctx.eval("""
      var seen = null;
      var api = {
        queuePrompt: function () {
          seen = Array.prototype.slice.call(arguments);
          return Promise.resolve({prompt_id: "p1"});
        },
        fetchApi: function () { return Promise.resolve({}); }
      };
      var pendingClaims = new Map();
      function currentKey() { return "wf/x.json"; }
    """)
    ctx.eval(iife)
    # quickjs has no event loop, but the stub's promise is already
    # resolved, so the await resumes on the first job-queue drain.
    ctx.eval("""
      api.queuePrompt(0, {output:{}, workflow:{}},
                      {partialExecutionTargets:["7"], previewMethod:"latent2rgb"});
    """)

    n = ctx.eval("seen === null ? -1 : seen.length")
    check("all three arguments forwarded", n == 3, f"got {n}")
    targets = ctx.eval("seen === null ? 'null' : JSON.stringify(seen[2])")
    check("partialExecutionTargets survives the wrapper",
          targets == '{"partialExecutionTargets":["7"],"previewMethod":"latent2rgb"}',
          f"got {targets}")
    check("wrapper marks itself, so it installs once",
          ctx.eval("api.queuePrompt.__batProfiler === true"))


t_queue_passthrough()

# ─────────────────────────────────────────────────────────────────────
print("\n[6] crash report")


def t_report():
    """The report exists for one scenario: the process was OOM-killed, so
    nothing raised, nothing was logged, and all that survives is what the
    browser had already written down. Build that scenario and check the
    report reaches the right verdict."""
    try:
        import quickjs
    except Exception as e:
        check("quickjs available", False, f"{e!r}")
        return

    web = os.path.join(PACK, "web")
    src = open(os.path.join(web, "bat_profiler_report.js")).read()
    src = src.replace("export function", "function").replace("export const", "const")
    ctx = quickjs.Context()
    ctx.eval(src)

    GB = 1024 ** 3
    # 24 GB card, 128 GB box. A sampler climbs and the box dies.
    samples = []
    for i in range(60):
        s = {
            "t": 1000.0 + i * 0.2,
            "ram": int(8 * GB + i * 0.9 * GB),
            "sys": int(20 * GB + i * 1.7 * GB),
            "sys_total": 128 * GB,
            "vram": int(1 * GB + i * 0.38 * GB),
            "res": int(1.2 * GB + i * 0.40 * GB),
            "node": "12" if i < 20 else "42",
        }
        if i % 5 == 0:
            s["dev_total"] = 24 * GB
        samples.append(s)

    run = {
        "prompt_id": "abc", "workflow": "wf/wan.json",
        "started": 1000.0, "ended": None, "status": "lost",
        "lastSeen": 1012.0,
        "baseline": {"sys": 20 * GB, "sys_total": 128 * GB,
                     "vram": 1 * GB, "dev_total": 24 * GB},
        "samples": samples,
        "titles": {"12": "Load Plates", "42": "WanVideo Sampler"},
        # 42 started and never completed — it is the suspect.
        "inflight": {"node_id": "42", "title": "WanVideo Sampler",
                     "class_type": "WanVideoSampler", "t": 1004.0},
        "nodes": {
            "12": {"node_id": "12", "class_type": "VHS_LoadVideo", "duration": 3.9,
                   "vram_peak": 2 * GB, "vram_delta": int(1.5 * GB),
                   "ram_peak": 14 * GB, "ram_delta": int(6 * GB),
                   "bytes_out": int(9.4 * GB), "desc_out": "81x1024x1024x3 float32",
                   "io_read": int(4.2 * GB), "io_write": 0,
                   "cached": False, "skipped": False, "entries": 1},
            "13": {"node_id": "13", "class_type": "ImageResize", "duration": 1.2,
                   "vram_peak": int(0.4 * GB), "vram_delta": int(0.3 * GB),
                   "ram_peak": 15 * GB, "ram_delta": int(2.1 * GB),
                   "bytes_out": int(2.1 * GB), "desc_out": "81x512x512x3 float32",
                   "cached": False, "skipped": False, "entries": 1},
            "14": {"node_id": "14", "class_type": "CLIPTextEncode", "duration": 0.3,
                   "vram_peak": int(0.2 * GB), "vram_delta": 0,
                   "ram_peak": 15 * GB, "ram_delta": 0, "bytes_out": 1024,
                   "cached": True, "skipped": False, "entries": 0},
        },
    }
    system = {
        "system": {"ram_total": 128 * GB, "comfyui_version": "0.33.0",
                   "pytorch_version": "2.9.1+cu128",
                   "argv": ["main.py", "--port", "8188", "--reserve-vram", "2"]},
        "devices": [{"name": "NVIDIA RTX A5000", "vram_total": 24 * GB}],
    }

    ctx.eval("var RUN = " + json.dumps(run) + ";")
    ctx.eval("var SYS = " + json.dumps(system) + ";")
    call = ('buildReport({run: RUN, runCount: 4, workflowName: "wan_comp_v12.json",'
            ' system: SYS, config: {sync_cuda: true, reset_peak: true}})')
    report = ctx.eval(call)

    print()
    for line in report.split("\n"):
        print("    | " + line)
    print()

    check("names the workflow", "wan_comp_v12.json" in report)
    check("reaches the OOM verdict",
          "DID NOT FINISH" in report and "OOM kill" in report)
    check("names the node it died in",
          "DIED IN" in report and "WanVideo Sampler" in report,
          "the suspect is the node that never completed, not the last that did")
    check("the suspect leads the report",
          report.index("WanVideo Sampler") < report.index("VHS_LoadVideo"))
    check("reports ceilings, not bare usage",
          "24.0 GB total" in report and "128.0 GB total" in report)
    check("flags the tight ceiling", "HEADROOM" in report)
    check("includes the trajectory", "TRAJECTORY" in report)
    check("separates peak from held",
          "TOP BY VRAM PEAK" in report and "STILL HELD AFTER RETURNING" in report,
          "different bugs, opposite fixes")
    check("includes launch flags with their values",
          "--reserve-vram 2" in report,
          "the value is the point; the bare flag name tells you nothing")
    check("includes the GPU", "RTX A5000" in report)
    check("counts cached nodes", "1 cached" in report)
    check("is small enough to paste", len(report) < 8000, f"{len(report)} chars")

    # A clean run must not cry wolf.
    clean = dict(run)
    clean["status"] = "ok"
    clean["ended"] = 1012.0
    clean["inflight"] = None
    ctx.eval("var CLEAN = " + json.dumps(clean) + ";")
    r2 = ctx.eval('buildReport({run: CLEAN, runCount: 1, workflowName: "x",'
                  ' system: SYS, config: {}})')
    check("clean run reports no death",
          "DID NOT FINISH" not in r2 and "DIED IN" not in r2
          and "Completed normally" in r2)

    # And an empty one must not throw.
    ctx.eval("var EMPTY = {prompt_id:'z', started:1, status:'ok',"
             " nodes:{}, samples:[], titles:{}};")
    r3 = ctx.eval('buildReport({run: EMPTY, runCount: 1, workflowName: "y",'
                  ' system: null, config: {}})')
    check("survives an empty run", isinstance(r3, str) and len(r3) > 0)


t_report()

# ─────────────────────────────────────────────────────────────────────
print("\n[7] chart grid maths")


def t_grid():
    """A grid is only an improvement if its lines land on numbers a human
    reads without thinking. Slice the real functions out of the panel
    (sliced, not copied — a copy would drift) and check the values."""
    try:
        import quickjs
    except Exception as e:
        check("quickjs available", False, f"{e!r}")
        return

    src = open(os.path.join(PACK, "web", "bat_profiler.js")).read()

    def slice_fn(name):
        marker = f"function {name}("
        i = src.index(marker)
        depth, j = 0, src.index("{", i)
        start = j
        while True:
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        return src[i:j + 1]

    ctx = quickjs.Context()
    for fn in ("fmtBytes", "fmtTick", "niceTicks", "niceTimeStep", "shortTime"):
        ctx.eval(slice_fn(fn))

    GB = 1024 ** 3
    ctx.eval(f"var GB = {GB};")

    # The machine in the screenshot: 125 GB of RAM.
    ticks = json.loads(ctx.eval("JSON.stringify(niceTicks(125*GB))"))
    labels = json.loads(ctx.eval(
        "JSON.stringify(niceTicks(125*GB).map(function(v){return fmtTick(v)}))"))
    check("125 GB axis gets 4 gridlines", len(ticks) == 4, f"got {labels}")
    check("125 GB labels are round", labels == ["25 GB", "50 GB", "75 GB", "100 GB"],
          f"got {labels}")

    # The card in the screenshot: 24 GB.
    labels = json.loads(ctx.eval(
        "JSON.stringify(niceTicks(24*GB).map(function(v){return fmtTick(v)}))"))
    check("24 GB labels are round", labels == ["5 GB", "10 GB", "15 GB", "20 GB"],
          f"got {labels}")

    labels = json.loads(ctx.eval(
        "JSON.stringify(niceTicks(8*GB).map(function(v){return fmtTick(v)}))"))
    check("8 GB labels are round", labels == ["2 GB", "4 GB", "6 GB"], f"got {labels}")

    # A fitted axis on a small trace must not produce byte-level noise.
    labels = json.loads(ctx.eval(
        "JSON.stringify(niceTicks(288*1024*1024).map(function(v){return fmtTick(v)}))"))
    check("MB-scale axis stays readable",
          all(("MB" in l) for l in labels) and len(labels) >= 2, f"got {labels}")

    # Invariants that must hold for any ceiling.
    bad = []
    for gb in (1, 2, 6, 11, 12, 16, 24, 32, 40, 48, 64, 80, 96, 125, 128, 192, 256, 512):
        t = json.loads(ctx.eval(f"JSON.stringify(niceTicks({gb}*GB))"))
        if not t:
            bad.append(f"{gb}GB:none")
            continue
        if any(v >= gb * GB for v in t):
            bad.append(f"{gb}GB:over-top")
        gaps = {round(t[i + 1] - t[i]) for i in range(len(t) - 1)}
        if len(gaps) > 1:
            bad.append(f"{gb}GB:uneven")
        if not (2 <= len(t) <= 6):
            bad.append(f"{gb}GB:count={len(t)}")
    check("gridlines are even, inside the axis, 2-6 of them across every"
          " plausible ceiling", not bad, str(bad))

    noisy = []
    for gb in (1, 2, 6, 8, 12, 16, 24, 32, 48, 64, 96, 125, 128, 256):
        labs = json.loads(ctx.eval(
            f"JSON.stringify(niceTicks({gb}*GB).map(function(v){{return fmtTick(v)}}))"))
        noisy += [l for l in labs if l.endswith(".0 GB") or l.endswith(".0 MB")]
    check("no gridline label carries a pointless .0", not noisy, str(noisy[:6]))

    check("zero axis yields no grid",
          json.loads(ctx.eval("JSON.stringify(niceTicks(0))")) == [])

    # Time axis.
    for span, want in ((10, 2), (60, 15), (300, 60), (3600, 900)):
        got = ctx.eval(f"niceTimeStep({span})")
        check(f"time step for {span}s span",
              span / got <= 5 and 2 <= span / got <= 5 and got == want,
              f"got {got}, want {want}")

    check("short time under a minute", ctx.eval("shortTime(45)") == "45s")
    check("short time on the minute", ctx.eval("shortTime(60)") == "1m")
    check("short time with seconds", ctx.eval("shortTime(90)") == "1m30")
    check("short time over an hour", ctx.eval("shortTime(3660)") == "1h01")


t_grid()

# ─────────────────────────────────────────────────────────────────────
print("\n" + ("FAILURES: " + ", ".join(failures) if failures else "All checks passed."))
sys.exit(1 if failures else 0)
