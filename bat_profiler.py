"""
BAT Profiler — per-node RAM / VRAM / time / data-transfer instrumentation
=========================================================================

What this is
------------
A sidebar-driven execution profiler for the *whole* graph (not just BAT
nodes). It answers the two questions you actually ask when a workflow
misbehaves: **which node ate the memory** and **which node ate the
clock**. Every executed node gets a record — wall time, RSS delta and
peak, VRAM delta and peak, bytes in / bytes out, and disk I/O — which is
streamed to the frontend live and kept there per-workflow.

Why it hooks where it hooks
---------------------------
Two module-level functions in ``execution.py`` are wrapped. Both are
called as *globals* from inside ``PromptExecutor``, so rebinding the
module attribute is enough — no source patching, no subclassing.

``execution.execute(server, dynprompt, caches, current_item, ...)``
    The per-node boundary. Wrapping it brackets one node's execution
    exactly, and hands us ``unique_id`` plus the cached/executed
    distinction (a cache hit returns early, before any work happens, so
    those show up as ~0ms "skipped" rows rather than vanishing).

``execution.get_output_data(prompt_id, unique_id, obj, input_data_all, ...)``
    The single normal-path producer of a node's outputs. Measuring
    payload bytes *here* rather than reading them back out of
    ``caches.outputs`` means we never touch the output cache — no LRU
    recency bump, no interaction with the RAM_PRESSURE cache provider.
    It also hands us ``input_data_all``, so "bytes in" is free.

We deliberately do **not** use a ``ProgressHandler``. That is the
official extension point and it does give clean start/finish callbacks
(see ComfyUI-ETC_Farm/progress_reporter.py, which uses exactly that),
but it never sees the node's payloads, which is half of what this panel
is for.

Measurement notes / honest caveats
----------------------------------
* **CUDA is asynchronous.** Without a sync at the node boundary, GPU
  work launched by node A lands on whichever later node happens to
  block, and the timings become fiction. ``sync_cuda`` (default on)
  inserts ``torch.cuda.synchronize()`` at each boundary so attribution
  is correct. It costs a pipeline drain per node — sub-millisecond in
  practice, since ComfyUI barely overlaps work across node boundaries
  anyway — and it can be switched off from the panel.
* **Peak VRAM** comes from ``torch.cuda.max_memory_allocated()``, which
  requires resetting the allocator's peak counter at each node
  boundary. That counter is process-global; nothing in ComfyUI core
  reads it (verified — core sizes its decisions off ``mem_get_info``
  and free VRAM), so resetting it is safe here. The background sampler
  independently tracks a windowed peak that needs no reset, so the
  panel still shows a peak if ``reset_peak`` is disabled.
* **Peak RSS** cannot be read per-window from the kernel at all
  (``VmHWM`` is high-water-since-process-start and cannot be reset), so
  it comes from the sampler: it is a 5 Hz *observation*, and a spike
  that opens and closes inside 200 ms can be missed. Deltas are exact;
  peaks are sampled. The panel labels them accordingly.
* **Disk I/O** records both ``*_bytes`` (block layer) and ``*_chars``
  (syscall level). On NFS — which is where this studio's plates live —
  block-layer counters under-report badly, so the panel shows chars.
* **Async / lazy nodes** re-enter ``execute`` (it returns PENDING and is
  called again). Records accumulate rather than overwrite: time sums,
  peaks take a max, deltas span first-entry to last-exit. For a truly
  async node the wall time inside ``execute`` is not the work's
  duration, and the row is flagged so the panel can say so.

Nothing in here may ever break execution. Every probe is wrapped; a
profiler failure disables the profiler, it does not fail the prompt.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("BAT.Profiler")

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a hard dep of ComfyUI in practice
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ─────────────────────────────────────────────────────────────────────
# Configuration
#
# Deliberately in-memory only. The profiler is on from boot so that an
# unattended OOM is always captured — you should never have to
# reproduce a crash with profiling switched on. The frontend persists
# the user's preference in localStorage and pushes it back on load, so
# there is no settings file to write into the pack directory.
# ─────────────────────────────────────────────────────────────────────
class Config:
    # Off for everybody until somebody asks for it. This is one shared
    # process — most artists on this box are not debugging, and they
    # should not pay for the per-node GPU sync so that one person can.
    default_enabled: bool = False
    sync_cuda: bool = True
    reset_peak: bool = True
    sample_hz_active: float = 5.0
    sample_hz_idle: float = 1.0
    max_runs: int = 10           # server-side ring buffer of finished runs
    max_samples: int = 6000      # ~20 min at 5 Hz before decimation


CONFIG = Config()


# ─────────────────────────────────────────────────────────────────────
# Arming — per client, not per server
#
# A single global switch would mean one artist flipping the profiler on
# changes execution behaviour for everyone else sharing the process,
# which is unacceptable on Stable and the Farm. Instead a browser arms
# *itself*: the frontend registers its ``client_id``, and only prompts
# submitted by an armed client are instrumented. Everyone else's runs
# take the untouched path — one boolean check per node and nothing more.
#
# Arming happens when the toggle is flipped, not when a prompt is
# queued, so there is no race against a prompt that begins executing
# the instant it is submitted.
#
# The registry is in-memory: a server restart forgets it, and the
# frontend re-arms from its stored preference on load.
# ─────────────────────────────────────────────────────────────────────
_ARMED_LIMIT = 64
_armed: Dict[str, float] = {}


def arm(client_id: Optional[str], enabled: bool) -> None:
    if not client_id:
        return
    client_id = str(client_id)
    with _lock:
        if enabled:
            _armed[client_id] = time.time()
            if len(_armed) > _ARMED_LIMIT:
                for k, _ in sorted(_armed.items(), key=lambda kv: kv[1])[:len(_armed) - _ARMED_LIMIT]:
                    _armed.pop(k, None)
        else:
            _armed.pop(client_id, None)
    if enabled:
        ensure_sampler()
    logger.info("[BAT Profiler] client %s %s", client_id, "armed" if enabled else "disarmed")


def is_armed(client_id: Optional[str]) -> bool:
    """Should a prompt from this client be profiled?

    An unrecognised client (an API submission, a farm dispatch) falls
    back to the server default, which is off.
    """
    if not client_id:
        return CONFIG.default_enabled
    with _lock:
        if str(client_id) in _armed:
            return True
    return CONFIG.default_enabled

# Frontends only receive the 5 Hz sample stream while a panel is open.
# Node and run records are always broadcast regardless — they are cheap
# and they are what has to survive a crash.
_SUBSCRIBE_TTL = 20.0
_last_subscribe = 0.0


def note_subscriber() -> None:
    """An open panel wants the live sample stream."""
    global _last_subscribe
    _last_subscribe = time.time()
    ensure_sampler()


def _has_subscriber() -> bool:
    return (time.time() - _last_subscribe) < _SUBSCRIBE_TTL


# ─────────────────────────────────────────────────────────────────────
# Metric probes
# ─────────────────────────────────────────────────────────────────────
_proc = None
if psutil is not None:
    try:
        _proc = psutil.Process()
    except Exception as e:  # pragma: no cover
        logger.warning("[BAT Profiler] psutil.Process() unavailable: %r", e)

_io_supported = True


def rss() -> int:
    """Resident set size of the ComfyUI process, in bytes."""
    if _proc is None:
        return 0
    try:
        return int(_proc.memory_info().rss)
    except Exception:
        return 0


def sys_ram() -> Tuple[int, int]:
    """(used, total) system RAM in bytes — the number that gets you OOM-killed."""
    if psutil is None:
        return (0, 0)
    try:
        vm = psutil.virtual_memory()
        return (int(vm.total - vm.available), int(vm.total))
    except Exception:
        return (0, 0)


def io_counters() -> Dict[str, int]:
    """Cumulative process I/O. ``*_chars`` is syscall-level and is the
    only meaningful one on NFS; ``*_bytes`` is block-layer."""
    global _io_supported
    if _proc is None or not _io_supported:
        return {}
    try:
        c = _proc.io_counters()
    except Exception:
        _io_supported = False
        return {}
    out = {}
    for name in ("read_bytes", "write_bytes", "read_chars", "write_chars"):
        v = getattr(c, name, None)
        if v is not None:
            out[name] = int(v)
    return out


_torch_device = None


def _device():
    """The device ComfyUI is actually running on, asked of ComfyUI itself."""
    global _torch_device
    if _torch_device is not None:
        return _torch_device
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        if getattr(dev, "type", None) != "cuda":
            return None
    except Exception:
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
        except Exception:
            return None
    _torch_device = dev
    return dev


def cuda_stats(include_device: bool = False) -> Dict[str, int]:
    """Allocator stats, and optionally the whole-device figure.

    ``alloc``/``reserved`` are this process's torch allocator — precise,
    and what you can actually act on. ``dev_used`` is the whole card
    including other processes and non-torch allocations; it is the
    number that matches nvidia-smi, and it costs a driver call, so it is
    sampled less often.
    """
    dev = _device()
    if dev is None:
        return {}
    out: Dict[str, int] = {}
    try:
        out["alloc"] = int(torch.cuda.memory_allocated(dev))
        out["reserved"] = int(torch.cuda.memory_reserved(dev))
    except Exception:
        return {}
    if include_device:
        try:
            free, total = torch.cuda.mem_get_info(dev)
            out["dev_used"] = int(total - free)
            out["dev_total"] = int(total)
        except Exception:
            pass
    return out


def cuda_peak() -> int:
    dev = _device()
    if dev is None:
        return 0
    try:
        return int(torch.cuda.max_memory_allocated(dev))
    except Exception:
        return 0


def cuda_reset_peak() -> None:
    if not CONFIG.reset_peak:
        return
    dev = _device()
    if dev is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(dev)
    except Exception:
        pass


def cuda_sync() -> None:
    if not CONFIG.sync_cuda:
        return
    dev = _device()
    if dev is None:
        return
    try:
        torch.cuda.synchronize(dev)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Payload measurement
#
# Walks a node's inputs/outputs and sums the bytes actually carried.
# Two absolute rules:
#   1. Never retain a reference to anything walked. A profiler that
#      holds onto output tensors would pin every intermediate in the
#      graph and cause the exact OOM it is meant to diagnose. Only ints
#      and short strings leave this function.
#   2. Bounded cost. A node returning a 5000-element list must not turn
#      profiling into the bottleneck, hence the visit budget.
# ─────────────────────────────────────────────────────────────────────
_VISIT_BUDGET = 4000
_MAX_DEPTH = 6


class _Walk:
    __slots__ = ("visits", "bytes", "desc")

    def __init__(self):
        self.visits = 0
        self.bytes = 0
        self.desc: Optional[str] = None


def _fmt_tensor(t) -> str:
    try:
        shape = "x".join(str(int(s)) for s in tuple(t.shape))
        dtype = str(t.dtype).replace("torch.", "")
        dev = getattr(t, "device", None)
        loc = f"@{dev.type}" if dev is not None and dev.type != "cpu" else ""
        return f"{shape} {dtype}{loc}"
    except Exception:
        return "tensor"


def _walk(obj: Any, w: _Walk, depth: int = 0) -> None:
    if w.visits >= _VISIT_BUDGET or depth > _MAX_DEPTH:
        return
    w.visits += 1

    if torch is not None and isinstance(obj, torch.Tensor):
        try:
            n = int(obj.element_size()) * int(obj.nelement())
        except Exception:
            n = 0
        w.bytes += n
        # Describe the single largest tensor seen — that is the one the
        # user is looking for when a row says "3.2 GB out".
        if w.desc is None or n > getattr(_walk, "_last", 0):
            _walk._last = n
            w.desc = _fmt_tensor(obj)
        return

    if np is not None and isinstance(obj, np.ndarray):
        try:
            w.bytes += int(obj.nbytes)
        except Exception:
            pass
        return

    if isinstance(obj, (bytes, bytearray, memoryview)):
        try:
            w.bytes += len(obj)
        except Exception:
            pass
        return

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if w.visits >= _VISIT_BUDGET:
                return
            _walk(item, w, depth + 1)
        return

    if isinstance(obj, dict):
        for item in obj.values():
            if w.visits >= _VISIT_BUDGET:
                return
            _walk(item, w, depth + 1)
        return

    # Model wrappers, conditioning objects, latent dicts already covered
    # above. Anything else (ints, strings, custom objects) is not a
    # meaningful "transfer" and is skipped rather than deep-reflected.


def payload_bytes(obj: Any) -> Tuple[int, Optional[str]]:
    """Return (bytes, description-of-largest-tensor). Retains nothing."""
    w = _Walk()
    _walk._last = 0
    try:
        _walk(obj, w)
    except Exception:
        return (0, None)
    return (w.bytes, w.desc)


# ─────────────────────────────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────────────────────────────
class NodeRecord:
    """One node within one run. Accumulates across PENDING re-entries."""

    __slots__ = (
        "node_id", "class_type", "display_node", "cached", "skipped", "executed",
        "error", "async_node",
        "entries", "duration", "started",
        "ram_start", "ram_end", "ram_peak",
        "vram_start", "vram_end", "vram_peak",
        "dev_peak", "bytes_in", "bytes_out", "desc_in", "desc_out",
        "io_read", "io_write",
    )

    def __init__(self, node_id: str, class_type: str, display_node: str):
        self.node_id = node_id
        self.class_type = class_type
        self.display_node = display_node
        self.cached = False      # served from the output cache
        self.skipped = False     # never reached — branch not needed this run
        self.executed = False    # get_output_data actually fired for it
        self.error = False
        self.async_node = False
        self.entries = 0
        self.duration = 0.0
        self.started = 0.0
        self.ram_start = 0
        self.ram_end = 0
        self.ram_peak = 0
        self.vram_start = 0
        self.vram_end = 0
        self.vram_peak = 0
        self.dev_peak = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.desc_in = None
        self.desc_out = None
        self.io_read = 0
        self.io_write = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_type": self.class_type,
            "display_node": self.display_node,
            "cached": self.cached,
            "skipped": self.skipped,
            "error": self.error,
            "async_node": self.async_node,
            "entries": self.entries,
            "duration": round(self.duration, 6),
            "started": round(self.started, 3),
            "ram_delta": self.ram_end - self.ram_start,
            "ram_peak": self.ram_peak,
            "vram_delta": self.vram_end - self.vram_start,
            "vram_peak": self.vram_peak,
            "dev_peak": self.dev_peak,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "desc_in": self.desc_in,
            "desc_out": self.desc_out,
            "io_read": self.io_read,
            "io_write": self.io_write,
        }


class RunRecord:
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        self.workflow: Optional[str] = None
        self.started = time.time()
        self.ended: Optional[float] = None
        self.status = "running"
        self.nodes: Dict[str, NodeRecord] = {}
        self.order: List[str] = []
        self.samples: List[Dict[str, Any]] = []
        self.baseline: Dict[str, Any] = {}
        self.error: Optional[Dict[str, Any]] = None

    def get(self, node_id: str, class_type: str, display_node: str) -> NodeRecord:
        rec = self.nodes.get(node_id)
        if rec is None:
            rec = NodeRecord(node_id, class_type, display_node)
            self.nodes[node_id] = rec
            self.order.append(node_id)
        return rec

    def summary(self) -> Dict[str, Any]:
        executed = [n for n in self.nodes.values() if not n.cached]
        return {
            "prompt_id": self.prompt_id,
            "workflow": self.workflow,
            "started": self.started,
            "ended": self.ended,
            "status": self.status,
            "node_count": len(self.nodes),
            "executed_count": len(executed),
            "cached_count": len(self.nodes) - len(executed),
            "duration": (self.ended - self.started) if self.ended else (time.time() - self.started),
            "cpu_time": sum(n.duration for n in executed),
            "peak_ram": max((n.ram_peak for n in self.nodes.values()), default=0),
            "peak_vram": max((n.vram_peak for n in self.nodes.values()), default=0),
            "peak_dev": max((n.dev_peak for n in self.nodes.values()), default=0),
            "bytes_out": sum(n.bytes_out for n in self.nodes.values()),
            "error": self.error,
        }

    def to_dict(self, with_samples: bool = True) -> Dict[str, Any]:
        d = self.summary()
        d["baseline"] = self.baseline
        d["nodes"] = [self.nodes[nid].to_dict() for nid in self.order]
        if with_samples:
            d["samples"] = self.samples
        return d


# ─────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────
_lock = threading.RLock()
_current: Optional[RunRecord] = None
_history: deque = deque(maxlen=CONFIG.max_runs)
# prompt_id -> workflow key, claimed by the frontend at queue time so a
# run is attributed to the tab that submitted it even if the user
# switches tabs while it executes.
_claims: Dict[str, str] = {}
# unique_id -> (bytes_in, bytes_out, desc_in, desc_out), handed from the
# get_output_data hook to the execute hook. Sizes only, never objects.
_payloads: Dict[str, Tuple[int, int, Optional[str], Optional[str]]] = {}


def claim(prompt_id: str, workflow: Optional[str]) -> None:
    with _lock:
        if workflow:
            _claims[prompt_id] = workflow
            if _current is not None and _current.prompt_id == prompt_id:
                _current.workflow = workflow
        # Bounded — a browser that queues and never returns must not grow this.
        if len(_claims) > 64:
            for k in list(_claims)[:-32]:
                _claims.pop(k, None)


def get_run(prompt_id: str) -> Optional[RunRecord]:
    with _lock:
        if _current is not None and _current.prompt_id == prompt_id:
            return _current
        for r in _history:
            if r.prompt_id == prompt_id:
                return r
    return None


def state_payload(workflow: Optional[str] = None,
                  client_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        runs = list(_history)
        if _current is not None:
            runs.append(_current)
        if workflow:
            runs = [r for r in runs if r.workflow == workflow]
        return {
            "config": {
                # "enabled" is this client's own arming state, not a
                # server-wide switch — see arm() for why.
                "enabled": is_armed(client_id) if client_id else CONFIG.default_enabled,
                "sync_cuda": CONFIG.sync_cuda,
                "reset_peak": CONFIG.reset_peak,
                "sample_hz": CONFIG.sample_hz_active,
            },
            "capabilities": {
                "psutil": psutil is not None,
                "cuda": _device() is not None,
                "io": _io_supported,
            },
            "current": _current.prompt_id if _current else None,
            "runs": [r.summary() for r in runs],
        }


# ─────────────────────────────────────────────────────────────────────
# Broadcast
# ─────────────────────────────────────────────────────────────────────
def _send(event: str, data: Dict[str, Any]) -> None:
    try:
        from server import PromptServer
        inst = PromptServer.instance
        if inst is None:
            return
        # Broadcast (sid=None): the panel is a monitor, and a second
        # browser window watching the same box is a legitimate use.
        inst.send_sync(event, data, None)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Background sampler
#
# Gives the panel a real curve — without it a single ten-minute sampler
# node is one flat step and you cannot see memory climbing inside it.
# It also supplies the windowed RAM peak, which the kernel will not give
# us any other way.
# ─────────────────────────────────────────────────────────────────────
class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(name="BAT-Profiler-Sampler", daemon=True)
        self._stop = threading.Event()
        # Idling between runs must not delay the first sample of the
        # next one: _begin_run wakes the thread rather than letting it
        # sleep out its idle interval. Without this a short run finishes
        # before the sampler ever notices it started, and the chart is
        # empty for exactly the runs that are quick enough to re-run.
        self._wake = threading.Event()
        self._wlock = threading.Lock()
        self._window: Optional[Dict[str, int]] = None
        self._pending: List[Dict[str, Any]] = []
        self._active_node: Optional[str] = None
        self._dev_tick = 0

    # -- per-node peak window ----------------------------------------
    def open_window(self, node_id: Optional[str], ram: int, vram: int) -> None:
        with self._wlock:
            self._active_node = node_id
            self._window = {"ram": ram, "vram": vram, "dev": 0}

    def close_window(self, ram: int, vram: int) -> Dict[str, int]:
        with self._wlock:
            w = self._window or {"ram": 0, "vram": 0, "dev": 0}
            out = {
                "ram": max(w.get("ram", 0), ram),
                "vram": max(w.get("vram", 0), vram),
                "dev": w.get("dev", 0),
            }
            self._window = None
            self._active_node = None
            return out

    def _fold(self, ram: int, vram: int, dev: int) -> None:
        with self._wlock:
            w = self._window
            if w is None:
                return
            if ram > w["ram"]:
                w["ram"] = ram
            if vram > w["vram"]:
                w["vram"] = vram
            if dev > w["dev"]:
                w["dev"] = dev

    # -- loop ---------------------------------------------------------
    def run(self) -> None:
        last_flush = 0.0
        while not self._stop.is_set():
            try:
                with _lock:
                    run = _current
                active = run is not None
                hz = CONFIG.sample_hz_active if active else CONFIG.sample_hz_idle
                interval = 1.0 / max(hz, 0.2)

                # Only sample while a run is actually going. Between
                # runs the thread costs one None check a second, and
                # puts nothing on the websocket at all.
                if run is not None:
                    # The whole-device figure needs a driver call, so it
                    # is taken once a second rather than every tick.
                    self._dev_tick += 1
                    want_dev = (self._dev_tick % max(int(hz), 1)) == 0
                    now = time.time()
                    r = rss()
                    cu = cuda_stats(include_device=want_dev)
                    used, total = sys_ram()
                    self._fold(r, cu.get("alloc", 0), cu.get("dev_used", 0))

                    sample = {
                        "t": round(now, 3),
                        "ram": r,
                        "sys": used,
                        "sys_total": total,
                        "vram": cu.get("alloc", 0),
                        "res": cu.get("reserved", 0),
                        "node": self._active_node,
                    }
                    if want_dev and "dev_used" in cu:
                        sample["dev"] = cu["dev_used"]
                        sample["dev_total"] = cu.get("dev_total", 0)

                    if run is not None:
                        with _lock:
                            if _current is run:
                                run.samples.append(sample)
                                if len(run.samples) > CONFIG.max_samples:
                                    # Decimate oldest half rather than
                                    # truncating — a long run keeps its
                                    # whole shape, at lower resolution.
                                    half = len(run.samples) // 2
                                    run.samples[:half] = run.samples[:half:2]

                    # Batched at 2 Hz rather than one frame per sample:
                    # same data, a fifth of the websocket chatter.
                    if _has_subscriber():
                        self._pending.append(sample)
                        if (now - last_flush) >= 0.5 and self._pending:
                            _send("bat.profiler.samples",
                                  {"prompt_id": run.prompt_id, "samples": self._pending})
                            self._pending = []
                            last_flush = now
                    elif self._pending:
                        self._pending = []
            except Exception:
                pass
            self._wake.clear()
            self._wake.wait(interval)

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


_sampler: Optional[Sampler] = None
_sampler_lock = threading.Lock()


def ensure_sampler() -> None:
    """Start the sampler on first use.

    Deliberately not started at install: a box where nobody ever opens
    the panel should not have a profiler thread on it at all.
    """
    global _sampler
    with _sampler_lock:
        if _sampler is None:
            _sampler = Sampler()
            _sampler.start()
            logger.info("[BAT Profiler] sampler started")


# ─────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────
_installed = False
_orig_execute = None
_orig_get_output_data = None
_orig_execute_async = None


def _begin_run(prompt_id: str) -> RunRecord:
    global _current
    with _lock:
        if _current is not None and _current.prompt_id == prompt_id:
            return _current
        if _current is not None:
            _end_run_locked("interrupted")
        run = RunRecord(prompt_id)
        run.workflow = _claims.get(prompt_id)
        cu = cuda_stats(include_device=True)
        used, total = sys_ram()
        run.baseline = {
            "ram": rss(),
            "sys": used,
            "sys_total": total,
            "vram": cu.get("alloc", 0),
            "res": cu.get("reserved", 0),
            "dev": cu.get("dev_used", 0),
            "dev_total": cu.get("dev_total", 0),
        }
        _current = run
        _payloads.clear()
    # After the run is published, never before: the sampler reads
    # _current on its very first tick.
    ensure_sampler()
    if _sampler is not None:
        _sampler.wake()
    _send("bat.profiler.run", {"phase": "start", "run": run.summary(), "baseline": run.baseline})
    return run


def _end_run_locked(status: str) -> Optional[RunRecord]:
    global _current
    run = _current
    if run is None:
        return None
    run.ended = time.time()
    run.status = status
    _history.append(run)
    _current = None
    _payloads.clear()
    return run


def _note_unvisited(run: RunRecord, prompt: Dict[str, Any], caches) -> None:
    """Account for prompt nodes that ``execute()`` never saw.

    ComfyUI does not merely short-circuit a cached node — ExecutionList
    never adds it (comfy_execution/graph.py, add_strong_link: a cached
    upstream node is not walked at all), so the profiler's per-node hook
    is never called for it. Without this pass, re-running an unchanged
    graph shows one row for the output node and nothing else, which
    reads as if the profiler had broken.

    A node with a live cache entry is reported as cached; one without is
    reported as skipped — a branch this run did not need, such as an
    untaken lazy input.
    """
    if not isinstance(prompt, dict):
        return
    for node_id, node in prompt.items():
        nid = str(node_id)
        if nid in run.nodes:
            continue
        try:
            class_type = node.get("class_type", "?")
        except Exception:
            class_type = "?"
        rec = run.get(nid, class_type, nid)
        rec.entries = 0
        try:
            rec.cached = caches.outputs.get_local(nid) is not None
        except Exception:
            rec.cached = False
        rec.skipped = not rec.cached


def _end_run(status: str, prompt: Optional[Dict[str, Any]] = None, caches=None) -> None:
    with _lock:
        run = _current
        if run is not None and prompt is not None and caches is not None:
            try:
                _note_unvisited(run, prompt, caches)
            except Exception:
                pass
        run = _end_run_locked(status)
    if run is not None:
        # The end message carries the rows that were never streamed
        # live, because execute() never ran for them.
        _send("bat.profiler.run",
              {"phase": "end", "run": run.summary(),
               "nodes": [run.nodes[n].to_dict() for n in run.order
                         if run.nodes[n].entries == 0]})


async def _wrapped_get_output_data(prompt_id, unique_id, obj, input_data_all, *a, **kw):
    """Measure the payload crossing this node. Sizes only — nothing retained."""
    # `_current` is only set for a prompt from an armed client, so this
    # is the whole cost of the profiler for everybody else: one identity
    # check against None, twice per node.
    profiling = _current is not None
    b_in = d_in = None
    if profiling:
        try:
            b_in, d_in = payload_bytes(input_data_all)
        except Exception:
            b_in, d_in = None, None
    result = await _orig_get_output_data(prompt_id, unique_id, obj, input_data_all, *a, **kw)
    if profiling:
        try:
            output_data = result[0] if isinstance(result, tuple) and result else None
            b_out, d_out = payload_bytes(output_data)
            with _lock:
                _payloads[str(unique_id)] = (b_in or 0, b_out, d_in, d_out)
        except Exception:
            pass
    return result


async def _wrapped_execute(*args, **kwargs):
    """Bracket one node's execution."""
    run = _current
    if run is None:
        # Not an armed client's prompt — straight through, untouched.
        return await _orig_execute(*args, **kwargs)

    # Positional-with-fallback so an upstream signature change degrades
    # to "no profiling" instead of breaking the prompt.
    try:
        dynprompt = args[1] if len(args) > 1 else kwargs.get("dynprompt")
        unique_id = str(args[3] if len(args) > 3 else kwargs.get("current_item"))
        prompt_id = str(args[6] if len(args) > 6 else kwargs.get("prompt_id"))
        node = dynprompt.get_node(unique_id)
        class_type = node.get("class_type", "?")
        display_node = str(dynprompt.get_display_node_id(unique_id))
    except Exception:
        return await _orig_execute(*args, **kwargs)

    if run.prompt_id != prompt_id:
        # A prompt we are not profiling interleaved with one we are.
        return await _orig_execute(*args, **kwargs)

    sampler = _sampler
    try:
        cuda_sync()
        cuda_reset_peak()
        cu0 = cuda_stats()
        r0 = rss()
        io0 = io_counters()
        if sampler is not None:
            sampler.open_window(unique_id, r0, cu0.get("alloc", 0))
        t0 = time.perf_counter()
    except Exception:
        return await _orig_execute(*args, **kwargs)

    failed = False
    try:
        result = await _orig_execute(*args, **kwargs)
        return result
    except BaseException:
        failed = True
        raise
    finally:
        try:
            cuda_sync()
            dt = time.perf_counter() - t0
            cu1 = cuda_stats()
            r1 = rss()
            io1 = io_counters()
            win = sampler.close_window(r1, cu1.get("alloc", 0)) if sampler is not None else {}

            with _lock:
                rec = run.get(unique_id, class_type, display_node)
                if rec.entries == 0:
                    rec.started = t0
                    rec.ram_start = r0
                    rec.vram_start = cu0.get("alloc", 0)
                rec.entries += 1
                rec.duration += dt
                rec.ram_end = r1
                rec.vram_end = cu1.get("alloc", 0)
                rec.ram_peak = max(rec.ram_peak, win.get("ram", r1))
                # Allocator peak is exact when reset_peak is on; the
                # sampled window is the fallback and never wrong-low by
                # more than one tick.
                rec.vram_peak = max(rec.vram_peak, cuda_peak(), win.get("vram", 0))
                rec.dev_peak = max(rec.dev_peak, win.get("dev", 0))
                for k, dst in (("read_chars", "io_read"), ("write_chars", "io_write")):
                    if k in io0 and k in io1:
                        setattr(rec, dst, getattr(rec, dst) + max(0, io1[k] - io0[k]))
                pay = _payloads.pop(unique_id, None)
                if pay is not None:
                    rec.executed = True
                    rec.bytes_in, rec.bytes_out = pay[0], pay[1]
                    rec.desc_in, rec.desc_out = pay[2], pay[3]
                # A cache hit returns from execute() before
                # get_output_data is ever reached, so "did the payload
                # hook fire" is a fact rather than the timing guess this
                # used to be — a fast node that legitimately returns no
                # tensors was being mislabelled as cached.
                rec.cached = not rec.executed
                if failed:
                    rec.error = True
                snapshot = rec.to_dict()

            _send("bat.profiler.node", {"prompt_id": prompt_id, "node": snapshot})
        except Exception:
            pass


def _wrap_execute_async(orig):
    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        # The single arming gate for a whole prompt. `client_id` is the
        # browser that submitted it, so an artist who never opened the
        # panel runs on the original code path from here down.
        if not is_armed(extra_data.get("client_id") if isinstance(extra_data, dict) else None):
            return await orig(self, prompt, prompt_id, extra_data, execute_outputs)
        _begin_run(str(prompt_id))
        status = "ok"
        try:
            return await orig(self, prompt, prompt_id, extra_data, execute_outputs)
        except BaseException:
            status = "error"
            raise
        finally:
            try:
                if not getattr(self, "success", True):
                    status = "error"
            except Exception:
                pass
            _end_run(status, prompt, getattr(self, "caches", None))
    return execute_async


def install() -> bool:
    """Wrap the execution hooks and start the sampler. Idempotent."""
    global _installed, _orig_execute, _orig_get_output_data, _orig_execute_async, _sampler
    if _installed:
        return True
    if psutil is None:
        logger.warning("[BAT Profiler] psutil unavailable — profiler disabled.")
        return False
    try:
        import execution
    except Exception as e:
        logger.warning("[BAT Profiler] cannot import execution: %r", e)
        return False

    try:
        _orig_execute = execution.execute
        _orig_get_output_data = execution.get_output_data
        _orig_execute_async = execution.PromptExecutor.execute_async
        execution.execute = _wrapped_execute
        execution.get_output_data = _wrapped_get_output_data
        execution.PromptExecutor.execute_async = _wrap_execute_async(_orig_execute_async)
    except Exception as e:
        logger.warning("[BAT Profiler] could not install hooks: %r", e)
        return False

    try:
        from . import bat_profiler_api  # noqa: F401  (registers routes on import)
    except Exception as e:
        logger.warning("[BAT Profiler] REST routes unavailable: %r", e)

    _installed = True
    logger.info("[BAT Profiler] hooks installed, idle (cuda=%s). "
                "Nothing is profiled until a client arms itself.",
                _device() is not None)
    return True
