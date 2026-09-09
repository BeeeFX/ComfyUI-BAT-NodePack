/**
 * BAT Profiler — sidebar panel
 * ============================
 *
 * Answers "which node ate the box?" for the graph in the *current tab*.
 * Streams per-node time / RAM / VRAM / payload / disk-I/O from the
 * backend (see bat_profiler.py), keeps a live chart of memory against
 * the machine's actual ceilings, ranks nodes by whichever metric you
 * are hunting, and jumps the canvas to the offender when you click it.
 *
 * Three design points worth knowing before you edit this:
 *
 * • **Per-workflow isolation is claimed at queue time, not run time.**
 *   `api.queuePrompt` is wrapped so the prompt_id is bound to whichever
 *   workflow submitted it. Binding at execution time instead would
 *   mis-file every run where you queue from one tab and switch to
 *   another while it cooks — which is the normal way to use ComfyUI.
 *
 * • **Records are persisted as they arrive**, not at run end. The
 *   backend's history dies with the process, so the localStorage copy
 *   is the only thing that survives the OOM kill you are trying to
 *   diagnose. After a crash: reload, open this panel, and the last
 *   node to complete is still there.
 *
 * • **All CSS is `bat-prof-` prefixed.** The packs in this suite share
 *   one global stylesheet namespace and generic class names silently
 *   restyle each other.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ProfilerStore, workflowKey } from "./bat_profiler_store.js";
import { buildReport } from "./bat_profiler_report.js";

const TAB_ID = "bat.profiler";
const PREF_KEY = "bat.profiler.prefs";

// ─────────────────────────────────────────────────────────────────────
// Formatting
// ─────────────────────────────────────────────────────────────────────
function fmtBytes(n, signed) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = signed && n > 0 ? "+" : n < 0 ? "−" : "";
  let v = Math.abs(n);
  if (v < 1024) return `${sign}${v} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let u = -1;
  do { v /= 1024; u++; } while (v >= 1024 && u < units.length - 1);
  return `${sign}${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[u]}`;
}

function fmtDuration(s) {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  if (s < 0.001) return "<1 ms";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  if (m < 60) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

function fmtClock(epoch) {
  try {
    return new Date(epoch * 1000).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch (e) {
    return "";
  }
}

// ─────────────────────────────────────────────────────────────────────
// Metrics — what you can rank by, and how each is worded.
//
// The distinction between *peak* and *delta* is the whole diagnostic
// value here, so the descriptions spell it out: peak is the transient
// high-water mark that decides whether you OOM, delta is what the node
// still holds after it returns, which is what makes a long graph creep
// upward until it dies twenty nodes later.
// ─────────────────────────────────────────────────────────────────────
const METRICS = {
  duration:   { label: "Time",        get: (n) => n.duration,   fmt: (v) => fmtDuration(v), hint: "Wall time inside the node." },
  vram_peak:  { label: "VRAM peak",   get: (n) => n.vram_peak,  fmt: (v) => fmtBytes(v),    hint: "High-water VRAM while the node ran — the number that decides whether you OOM." },
  vram_delta: { label: "VRAM held",   get: (n) => n.vram_delta, fmt: (v) => fmtBytes(v, true), hint: "VRAM still allocated after the node returned. Positive values accumulate down the graph." },
  ram_peak:   { label: "RAM peak",    get: (n) => n.ram_peak,   fmt: (v) => fmtBytes(v),    hint: "High-water process RSS, sampled at 5 Hz." },
  ram_delta:  { label: "RAM held",    get: (n) => n.ram_delta,  fmt: (v) => fmtBytes(v, true), hint: "Process RSS still held after the node returned." },
  bytes_out:  { label: "Data out",    get: (n) => n.bytes_out,  fmt: (v) => fmtBytes(v),    hint: "Bytes of tensor payload the node handed down its links." },
  bytes_in:   { label: "Data in",     get: (n) => n.bytes_in,   fmt: (v) => fmtBytes(v),    hint: "Bytes of tensor payload fed into the node." },
  io:         { label: "Disk I/O",    get: (n) => (n.io_read || 0) + (n.io_write || 0), fmt: (v) => fmtBytes(v), hint: "Syscall-level read + write. Block-layer counters under-report on NFS, so this is the chars figure." },
  order:      { label: "Run order",   get: (n) => n.started,    fmt: (v, n) => fmtDuration(n.duration), hint: "Execution order." },
};

// ─────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────
const store = new ProfilerStore();

const state = {
  root: null,
  wfKey: null,
  wfName: null,
  selectedRun: null,   // prompt_id, or null to follow the live run
  liveRun: null,
  backendCurrent: null,   // what the server says is executing right now
  sort: "duration",
  desc: true,
  hideCached: true,
  capabilities: {},
  // `config.enabled` is *this browser's* arming state, not a server
  // switch — see the /arm endpoint. `wantArmed` is what the stored
  // preference asked for, re-applied on load so a reload after a crash
  // comes back still recording.
  config: { enabled: false, sync_cuda: true, reset_peak: true },
  wantArmed: false,
  attached: false,   // has the sidebar actually mounted our root yet?
  sized: false,      // have we redrawn once real dimensions existed?
  poll: null,
  heartbeat: null,
  chartDirty: false,
};

function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem(PREF_KEY) || "{}");
    if (p.sort in METRICS) state.sort = p.sort;
    if (typeof p.desc === "boolean") state.desc = p.desc;
    if (typeof p.hideCached === "boolean") state.hideCached = p.hideCached;
    if (typeof p.armed === "boolean") state.wantArmed = p.armed;
  } catch (e) { /* first run */ }
}

function savePrefs() {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify({
      sort: state.sort, desc: state.desc, hideCached: state.hideCached,
      armed: !!state.config.enabled,
    }));
  } catch (e) { /* quota — prefs are not worth pruning for */ }
}

// ─────────────────────────────────────────────────────────────────────
// Workflow identity
//
// A local copy rather than an import of ETC_Core's etc-paths.js: this
// pack ships to its own repository and must stand alone.
// ─────────────────────────────────────────────────────────────────────
function activeWorkflow() {
  return app.extensionManager?.workflow?.activeWorkflow
      || app.workflowManager?.activeWorkflow
      || null;
}

function currentKey() {
  return workflowKey(activeWorkflow());
}

function currentName() {
  const wf = activeWorkflow();
  return wf?.filename || wf?.name || wf?.path?.split(/[\\/]/).pop() || "Unsaved workflow";
}

// ─────────────────────────────────────────────────────────────────────
// Run selection
// ─────────────────────────────────────────────────────────────────────
function runsForTab() {
  return reconcileRuns(store.listRuns(state.wfKey || "__unattributed__"));
}

/**
 * Relabel runs that never reported an ending.
 *
 * A run stuck at "running" that neither this page nor the backend
 * considers live did not finish — the process executing it is gone.
 * That is precisely the OOM-kill signature: SIGKILL raises nothing and
 * sends no error event, the websocket simply stops. Calling it "lost"
 * rather than leaving it "running" is what lets the report state a
 * verdict, instead of showing a run that appears to still be going
 * three days later.
 */
function reconcileRuns(runs) {
  for (const r of runs) {
    if (r.status !== "running" || r.ended) continue;
    if (r.prompt_id === state.liveRun) continue;
    if (r.prompt_id === state.backendCurrent) continue;
    r.status = "lost";
    store.markDirty(r.workflow || state.wfKey);
  }
  return runs;
}

function activeRun() {
  const runs = runsForTab();
  if (state.selectedRun) {
    const found = runs.find((r) => r.prompt_id === state.selectedRun);
    if (found) return found;
  }
  return runs[0] || null;
}

function nodeRows(run) {
  if (!run) return [];
  let rows = run.order.map((id) => run.nodes[id]).filter(Boolean);
  if (state.hideCached) rows = rows.filter((n) => !n.cached && !n.skipped);
  const m = METRICS[state.sort] || METRICS.duration;
  rows.sort((a, b) => {
    const d = (m.get(a) || 0) - (m.get(b) || 0);
    return state.desc ? -d : d;
  });
  return rows;
}

// ─────────────────────────────────────────────────────────────────────
// Jump to node
//
// Subgraphs make this non-trivial: from frontend 1.49 the contents of a
// subgraph are invisible to `getNodeById` on the root graph, so a node
// executing at id "12:4" cannot be selected directly. The backend sends
// `display_node` (the outer container) alongside, and we fall back to
// it — the panel marks such rows so it is clear you were taken to the
// subgraph rather than the node itself.
// ─────────────────────────────────────────────────────────────────────
function resolveNode(rec) {
  const graph = app.graph;
  if (!graph) return null;
  const tries = [rec.node_id, rec.display_node, String(rec.node_id).split(":")[0]];
  for (const t of tries) {
    if (t === undefined || t === null) continue;
    const n = graph.getNodeById?.(Number(t)) || graph.getNodeById?.(t);
    if (n) return { node: n, exact: String(t) === String(rec.node_id) };
  }
  return null;
}

function jumpToNode(rec) {
  const hit = resolveNode(rec);
  if (!hit) {
    toast(`Node ${rec.node_id} is no longer in this graph.`);
    return;
  }
  const { node } = hit;
  const canvas = app.canvas;
  try {
    canvas.selectNodes?.([node]);
    if (canvas.centerOnNode) canvas.centerOnNode(node);
    else if (canvas.ds) {
      canvas.ds.offset[0] = -node.pos[0] + canvas.canvas.width / (2 * canvas.ds.scale);
      canvas.ds.offset[1] = -node.pos[1] + canvas.canvas.height / (2 * canvas.ds.scale);
    }
    app.graph.setDirtyCanvas(true, true);
  } catch (e) {
    console.warn("[BAT Profiler] jump failed:", e);
  }
}

function toast(msg) {
  const el = state.root?.querySelector(".bat-prof-toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("bat-prof-toast-on");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("bat-prof-toast-on"), 2600);
}

// ─────────────────────────────────────────────────────────────────────
// Backend plumbing
// ─────────────────────────────────────────────────────────────────────
function clientId() {
  return api.clientId || api.initialClientId || null;
}

let systemStats = null;
async function fetchSystem() {
  if (systemStats) return systemStats;
  try {
    const res = await api.fetchApi("/system_stats");
    if (res.ok) systemStats = await res.json();
  } catch (e) { /* report degrades to "unknown machine" */ }
  return systemStats;
}

async function fetchState() {
  try {
    const cid = clientId();
    const q = cid ? `?client_id=${encodeURIComponent(cid)}` : "";
    const res = await api.fetchApi(`/bat/profiler/state${q}`);
    if (!res.ok) return;
    const data = await res.json();
    state.capabilities = data.capabilities || {};
    state.config = data.config || state.config;
    state.backendCurrent = data.current || null;
  } catch (e) { /* backend older than the panel — degrade quietly */ }
}

/**
 * Arm or disarm profiling for this browser.
 *
 * Nothing anywhere is instrumented until this is called: the backend
 * only profiles prompts submitted by an armed client_id, so an artist
 * who never opens this panel runs entirely untouched code. That is the
 * whole reason enablement lives here rather than in a server flag.
 */
async function setArmed(enabled) {
  const cid = clientId();
  state.config.enabled = enabled;
  state.wantArmed = enabled;
  savePrefs();
  render();
  if (!cid) {
    toast("No client id yet — try again in a moment.");
    return;
  }
  try {
    const res = await api.fetchApi("/bat/profiler/arm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: cid, enabled }),
    });
    if (res.ok) {
      const data = await res.json();
      state.capabilities = data.capabilities || state.capabilities;
      state.config = data.config || state.config;
      render();
    }
  } catch (e) {
    toast("Could not reach the profiler backend.");
  }
}

/** Re-apply the stored preference after a reload, so a browser that was
 *  recording before a crash comes back recording. */
async function restoreArming() {
  if (state.wantArmed && !state.config.enabled) await setArmed(true);
}

async function pushConfig(patch) {
  Object.assign(state.config, patch);
  try {
    await api.fetchApi("/bat/profiler/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
  } catch (e) { /* ignore */ }
  render();
}

function heartbeat() {
  // Tells the backend a panel is open; the 5 Hz sample stream is only
  // broadcast while somebody is listening. Node and run records arrive
  // regardless, so a run profiled with the panel shut is still fully
  // recorded — it just has no chart.
  api.fetchApi("/bat/profiler/subscribe", { method: "POST" }).catch(() => {});
}

// Claim runs for the submitting tab. Wrapped once, at module load.
(function patchQueuePrompt() {
  if (!api.queuePrompt || api.queuePrompt.__batProfiler) return;
  const orig = api.queuePrompt.bind(api);
  const wrapped = async function (number, prompt) {
    const wf = currentKey();
    const res = await orig(number, prompt);
    try {
      const promptId = res?.prompt_id;
      if (promptId) {
        api.fetchApi("/bat/profiler/claim", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt_id: promptId, workflow: wf }),
        }).catch(() => {});
        pendingClaims.set(String(promptId), wf);
      }
    } catch (e) { /* never block a queue on the profiler */ }
    return res;
  };
  wrapped.__batProfiler = true;
  api.queuePrompt = wrapped;
})();

// prompt_id -> workflow key, so the frontend files a run correctly even
// before the backend echoes the claim back.
const pendingClaims = new Map();

function keyForRun(promptId) {
  const claimed = pendingClaims.get(String(promptId));
  if (claimed !== undefined) return claimed;
  return currentKey();
}

// ─────────────────────────────────────────────────────────────────────
// Websocket stream
// ─────────────────────────────────────────────────────────────────────
api.addEventListener("bat.profiler.run", (ev) => {
  const d = ev.detail || {};
  const run = d.run || {};
  if (!run.prompt_id) return;
  if (d.phase === "start") {
    const wf = run.workflow || keyForRun(run.prompt_id);
    store.beginRun(wf, { ...run, baseline: d.baseline });
    state.liveRun = run.prompt_id;
  } else {
    // Cached and skipped nodes never reach execute(), so they cannot be
    // streamed live — they arrive here, at the end, as one batch.
    for (const rec of d.nodes || []) {
      let title = null;
      try {
        title = app.graph?.getNodeById?.(Number(rec.display_node))?.title || null;
      } catch (e) { /* ignore */ }
      store.upsertNode(run.prompt_id, rec, title);
    }
    store.endRun(run.prompt_id, run);
    if (state.liveRun === run.prompt_id) state.liveRun = null;
    pendingClaims.delete(String(run.prompt_id));
  }
  scheduleRender();
});

api.addEventListener("bat.profiler.node", (ev) => {
  const d = ev.detail || {};
  if (!d.node || !d.prompt_id) return;
  // Capture the title now: the record has to stay readable after the
  // node is deleted, renamed, or the workflow is closed entirely.
  let title = null;
  try {
    const n = app.graph?.getNodeById?.(Number(d.node.display_node));
    title = n?.title || null;
  } catch (e) { /* ignore */ }
  if (!store.findRun(d.prompt_id)) {
    store.beginRun(keyForRun(d.prompt_id), { prompt_id: d.prompt_id, started: Date.now() / 1000 });
  }
  store.upsertNode(d.prompt_id, d.node, title);
  scheduleRender();
});

/**
 * Track the node ComfyUI is *currently* executing, using core's own
 * `executing` event rather than anything of ours.
 *
 * This is the single most important field in a crash report and it
 * cannot come from the profiler's own per-node records: those are
 * written when a node *finishes*, and the node that kills the process
 * never finishes. Core emits `executing` when a node starts, so the
 * last one of these with no matching completion record is the suspect.
 */
api.addEventListener("executing", (ev) => {
  const d = ev.detail;
  const nodeId = (d && typeof d === "object") ? d.node : d;
  const promptId = (d && typeof d === "object") ? d.prompt_id : null;
  const hit = store.findRun(promptId || state.liveRun);
  if (!hit) return;
  if (nodeId === null || nodeId === undefined) {
    hit.run.inflight = null;
  } else {
    let title = null, type = null;
    try {
      const n = app.graph?.getNodeById?.(Number(d.display_node ?? nodeId));
      title = n?.title || null;
      type = n?.type || null;
    } catch (e) { /* ignore */ }
    hit.run.inflight = {
      node_id: String(nodeId), title, class_type: type, t: Date.now() / 1000,
    };
  }
  // A heartbeat of "we were still alive at this moment", so the report
  // can say how long the doomed node ran before the lights went out.
  hit.run.lastSeen = Date.now() / 1000;
  store.markDirty(hit.wfKey);
});

api.addEventListener("bat.profiler.samples", (ev) => {
  const d = ev.detail || {};
  if (!d.samples?.length) return;
  // Attribute to the run the backend stamped on the message rather
  // than to whatever we last saw start.
  const promptId = d.prompt_id || state.liveRun;
  if (!promptId) return;
  const hit = store.pushSamples(promptId, d.samples);
  if (hit) {
    const last = d.samples[d.samples.length - 1];
    if (last?.t) hit.lastSeen = last.t;
  }
  state.chartDirty = true;
  scheduleChart();
});

// Persist immediately when the page is going away — a deliberate tab
// close should not lose the run that is on screen.
window.addEventListener("pagehide", () => store.flush());
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") store.flush();
});

// ─────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────
/**
 * Is the panel still ours to draw into?
 *
 * The subtlety that cost an afternoon: `registerSidebarTab`'s `render`
 * callback hands over the container element BEFORE it is attached to
 * the document, so `root.isConnected` is false on the very first call.
 * Guarding on it directly meant the first render bailed, the watch
 * interval immediately tore itself down, and the panel stayed blank
 * forever. (ETC_Documentation's equivalent guard is a bare `if (!root)`
 * — which is why that panel works.)
 *
 * So: treat "not connected" as death only once we have actually seen it
 * connected. Before that it just means "not mounted yet".
 */
function panelAlive() {
  const r = state.root;
  if (!r) return false;
  if (r.isConnected) { state.attached = true; return true; }
  return !state.attached;
}

let renderQueued = false;
function scheduleRender() {
  if (renderQueued || !panelAlive()) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    if (panelAlive()) render();
  });
}

let chartQueued = false;
function scheduleChart() {
  if (chartQueued || !panelAlive()) return;
  chartQueued = true;
  requestAnimationFrame(() => {
    chartQueued = false;
    if (panelAlive()) drawCharts();
  });
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function render() {
  const root = state.root;
  if (!root) return;
  // Wrapped so a fault in one builder shows up as an error and a
  // message in the panel, rather than as an empty black rectangle.
  try {
    root.innerHTML = "";
    root.className = "bat-prof-root";

    root.appendChild(buildHeader());
    root.appendChild(buildPower());
    root.appendChild(buildCharts());
    root.appendChild(buildSummary());
    root.appendChild(buildControls());
    root.appendChild(buildList());
    root.appendChild(buildFooter());
    root.appendChild(el("div", "bat-prof-toast"));

    drawCharts();
  } catch (e) {
    console.error("[BAT Profiler] render failed:", e);
    root.innerHTML = "";
    root.className = "bat-prof-root";
    const err = el("div", "bat-prof-empty",
      "The profiler panel failed to draw — see the browser console.");
    root.appendChild(err);
  }
}

function buildHeader() {
  const h = el("div", "bat-prof-header");

  const top = el("div", "bat-prof-header-top");
  const name = el("div", "bat-prof-wf", state.wfName || "—");
  name.title = `Profiles are kept per workflow. This panel shows only runs from “${state.wfName}”.`;
  top.appendChild(name);

  const live = el("span", "bat-prof-live");
  if (state.liveRun) {
    live.classList.add("bat-prof-live-on");
    live.textContent = "running";
  } else if (!state.config.enabled) {
    live.classList.add("bat-prof-live-off");
    live.textContent = "paused";
  } else {
    live.textContent = "idle";
  }
  top.appendChild(live);
  h.appendChild(top);

  const runs = runsForTab();
  const picker = el("select", "bat-prof-select");
  if (!runs.length) {
    const o = el("option", null, "No runs recorded yet");
    o.value = "";
    picker.appendChild(o);
    picker.disabled = true;
  } else {
    runs.forEach((r, i) => {
      const o = el("option");
      o.value = r.prompt_id;
      const dur = r.ended ? fmtDuration(r.ended - r.started)
                : r.status === "lost" ? fmtDuration((r.lastSeen || r.started) - r.started)
                : "running";
      const flag = r.status === "error" ? " ⚠"
                 : r.status === "lost" ? " ☠ did not finish"
                 : r.status === "interrupted" ? " ⨯" : "";
      o.textContent = `${i === 0 ? "Latest" : fmtClock(r.started)} · ${dur}${flag}`;
      picker.appendChild(o);
    });
    picker.value = activeRun()?.prompt_id || "";
    picker.onchange = () => { state.selectedRun = picker.value; render(); };
  }
  h.appendChild(picker);
  return h;
}

// ── power toggle ────────────────────────────────────────────────────
/**
 * The single control that decides whether anything is recorded at all,
 * so it gets the width and the colour rather than hiding among the
 * footer buttons. Switching it off leaves already-stored profiles
 * alone — it stops collection, it does not erase history.
 */
function buildPower() {
  const on = !!state.config.enabled;
  const row = el("div", `bat-prof-power${on ? " bat-prof-power-on" : ""}`);
  row.setAttribute("role", "switch");
  row.setAttribute("aria-checked", String(on));
  row.tabIndex = 0;

  const sw = el("div", "bat-prof-switch");
  sw.appendChild(el("div", "bat-prof-knob"));
  row.appendChild(sw);

  const text = el("div", "bat-prof-power-text");
  text.appendChild(el("div", "bat-prof-power-title",
    on ? "Profiling this browser" : "Profiling off"));
  text.appendChild(el("div", "bat-prof-power-sub", on
    ? `${state.config.sample_hz || 5} Hz${state.config.sync_cuda ? " · GPU-synced" : ""} · others on this server unaffected`
    : "nothing is measured — switch on to start recording"));
  row.appendChild(text);

  row.title = "Arms profiling for THIS browser only. Nothing is measured until you "
    + "switch it on, and other people using this same ComfyUI are unaffected either "
    + "way. Left on, it stays on across reloads, so an unattended crash is still "
    + "captured. Cost while armed: a sub-millisecond probe per node plus one 5 Hz "
    + "sampler thread.";

  const flip = () => setArmed(!state.config.enabled);
  row.onclick = flip;
  row.onkeydown = (e) => {
    if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); }
  };
  return row;
}

// ── charts ──────────────────────────────────────────────────────────
function buildCharts() {
  const wrap = el("div", "bat-prof-charts");
  for (const spec of [
    { id: "ram", label: "System RAM" },
    { id: "vram", label: "VRAM" },
  ]) {
    if (spec.id === "vram" && state.capabilities.cuda === false) continue;
    const box = el("div", "bat-prof-chart");
    const head = el("div", "bat-prof-chart-head");
    head.appendChild(el("span", "bat-prof-chart-label", spec.label));
    head.appendChild(el("span", `bat-prof-chart-val bat-prof-chart-val-${spec.id}`, ""));
    box.appendChild(head);
    const cv = el("canvas", `bat-prof-canvas bat-prof-canvas-${spec.id}`);
    cv.height = 56;
    box.appendChild(cv);
    wrap.appendChild(box);
  }
  return wrap;
}

/**
 * Both charts plot against the machine's real ceiling (total RAM /
 * total VRAM) rather than auto-scaling to the data. Auto-scaling makes
 * every run look equally alarming; a fixed ceiling tells you at a
 * glance how much headroom you actually had.
 */
function drawCharts() {
  const run = activeRun();
  const samples = run?.samples || [];
  const root = state.root;
  if (!root) return;

  const specs = [
    {
      id: "ram",
      series: [
        { key: "sys", color: "#5b8dd6", fill: true, label: "system" },
        { key: "ram", color: "#9ecbff", fill: false, label: "ComfyUI" },
      ],
      totalKey: "sys_total",
      baseTotal: run?.baseline?.sys_total,
    },
    {
      id: "vram",
      series: [
        { key: "vram", color: "#d68a3c", fill: true, label: "allocated" },
        { key: "res", color: "#f0c48a", fill: false, label: "reserved" },
      ],
      totalKey: "dev_total",
      baseTotal: run?.baseline?.dev_total,
    },
  ];

  for (const spec of specs) {
    const cv = root.querySelector(`.bat-prof-canvas-${spec.id}`);
    if (!cv) continue;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 260;
    const h = 56;
    if (cv.width !== Math.round(w * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const css = getComputedStyle(root);
    ctx.strokeStyle = css.getPropertyValue("--border-color") || "#444";
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();
    ctx.globalAlpha = 1;

    let total = spec.baseTotal || 0;
    for (const s of samples) if (s[spec.totalKey]) total = Math.max(total, s[spec.totalKey]);
    let peak = 0;
    for (const s of samples) for (const ser of spec.series) peak = Math.max(peak, s[ser.key] || 0);
    // If we never learned the ceiling, fall back to headroom over the
    // observed peak so the curve is still readable.
    const max = total > 0 ? total : peak * 1.25 || 1;

    const label = root.querySelector(`.bat-prof-chart-val-${spec.id}`);
    if (label) {
      label.textContent = samples.length
        ? `${fmtBytes(peak)} peak${total ? ` / ${fmtBytes(total)}` : ""}`
        : "no samples";
    }

    if (samples.length < 2) {
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = css.getPropertyValue("--descrip-text") || "#888";
      ctx.font = "11px sans-serif";
      ctx.fillText("waiting for a run…", 6, h / 2 + 4);
      ctx.globalAlpha = 1;
      continue;
    }

    const t0 = samples[0].t;
    const t1 = samples[samples.length - 1].t;
    const span = Math.max(t1 - t0, 0.001);
    const x = (s) => ((s.t - t0) / span) * w;
    const y = (v) => h - Math.min(1, (v || 0) / max) * (h - 3) - 1;

    for (const ser of spec.series) {
      ctx.beginPath();
      samples.forEach((s, i) => {
        const px = x(s), py = y(s[ser.key]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      if (ser.fill) {
        ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
        ctx.fillStyle = ser.color; ctx.globalAlpha = 0.22; ctx.fill();
        ctx.globalAlpha = 1;
        ctx.beginPath();
        samples.forEach((s, i) => {
          const px = x(s), py = y(s[ser.key]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
      }
      ctx.strokeStyle = ser.color;
      ctx.lineWidth = ser.fill ? 1.5 : 1;
      ctx.stroke();
    }
  }
}

// ── summary ─────────────────────────────────────────────────────────
function buildSummary() {
  const run = activeRun();
  const wrap = el("div", "bat-prof-summary");
  if (!run) {
    wrap.appendChild(el("div", "bat-prof-empty",
      "Queue this workflow and the profile appears here."));
    return wrap;
  }
  const nodes = Object.values(run.nodes);
  const executed = nodes.filter((n) => !n.cached && !n.skipped);
  const stats = [
    ["Wall", fmtDuration(run.ended ? run.ended - run.started : Date.now() / 1000 - run.started)],
    ["In nodes", fmtDuration(executed.reduce((a, n) => a + (n.duration || 0), 0))],
    ["Nodes", `${executed.length}${nodes.length - executed.length ? ` +${nodes.length - executed.length} skipped` : ""}`],
    ["Peak VRAM", fmtBytes(Math.max(0, ...nodes.map((n) => n.vram_peak || 0)))],
  ];
  for (const [k, v] of stats) {
    const s = el("div", "bat-prof-stat");
    s.appendChild(el("div", "bat-prof-stat-k", k));
    s.appendChild(el("div", "bat-prof-stat-v", v));
    wrap.appendChild(s);
  }
  if (run.status === "error") {
    wrap.appendChild(el("div", "bat-prof-warn",
      "This run ended in an error — the last row is where it stopped."));
  } else if (run.status === "lost") {
    // The interesting case: no error was ever reported because the
    // process did not live long enough to report one.
    const w = el("div", "bat-prof-warn bat-prof-warn-hard");
    const who = run.inflight && !run.nodes?.[run.inflight.node_id]
      ? `${run.inflight.title || run.inflight.class_type || `node ${run.inflight.node_id}`}`
      : null;
    w.textContent = who
      ? `Did not finish — the backend stopped responding while running “${who}”. `
        + `No error was raised, which is what an OOM kill looks like.`
      : "Did not finish — the backend stopped responding and raised no error, "
        + "which is what an OOM kill looks like.";
    wrap.appendChild(w);
    const hint = el("div", "bat-prof-warn",
      "Hit “Copy report” below and paste it to whoever is helping you debug it.");
    wrap.appendChild(hint);
  }
  return wrap;
}

// ── controls ────────────────────────────────────────────────────────
function buildControls() {
  const bar = el("div", "bat-prof-controls");

  const sel = el("select", "bat-prof-select bat-prof-sort");
  for (const [key, m] of Object.entries(METRICS)) {
    const o = el("option", null, `Rank by ${m.label}`);
    o.value = key;
    sel.appendChild(o);
  }
  sel.value = state.sort;
  sel.title = METRICS[state.sort].hint;
  sel.onchange = () => { state.sort = sel.value; savePrefs(); render(); };
  bar.appendChild(sel);

  const dir = el("button", "bat-prof-btn bat-prof-icon", state.desc ? "↓" : "↑");
  dir.title = state.desc ? "Worst first" : "Best first";
  dir.onclick = () => { state.desc = !state.desc; savePrefs(); render(); };
  bar.appendChild(dir);

  const cached = el("button", `bat-prof-btn${state.hideCached ? "" : " bat-prof-btn-on"}`,
    state.hideCached ? "cached hidden" : "cached shown");
  cached.title = "Cached and not-run nodes do no work and are excluded by default. "
    + "Show them to confirm what the run actually skipped — useful when a graph "
    + "finishes suspiciously fast.";
  cached.onclick = () => { state.hideCached = !state.hideCached; savePrefs(); render(); };
  bar.appendChild(cached);

  return bar;
}

// ── node list ───────────────────────────────────────────────────────
function buildList() {
  const list = el("div", "bat-prof-list");
  const run = activeRun();
  const rows = nodeRows(run);
  if (!rows.length) {
    if (run) list.appendChild(el("div", "bat-prof-empty", "No node records in this run."));
    return list;
  }

  const m = METRICS[state.sort] || METRICS.duration;
  const max = Math.max(...rows.map((n) => Math.abs(m.get(n) || 0)), 1);

  for (const rec of rows) {
    const row = el("div", "bat-prof-row");
    if (rec.error) row.classList.add("bat-prof-row-error");
    if (rec.cached || rec.skipped) row.classList.add("bat-prof-row-cached");

    const title = run.titles?.[rec.node_id] || rec.class_type;
    const line1 = el("div", "bat-prof-row-line");
    const nameEl = el("span", "bat-prof-name", title);
    nameEl.title = `${rec.class_type} · node ${rec.node_id}`;
    line1.appendChild(nameEl);
    line1.appendChild(el("span", "bat-prof-value", m.fmt(m.get(rec), rec)));
    row.appendChild(line1);

    const bar = el("div", "bat-prof-bar");
    const fill = el("div", "bat-prof-bar-fill");
    fill.style.width = `${Math.min(100, (Math.abs(m.get(rec) || 0) / max) * 100)}%`;
    if ((m.get(rec) || 0) < 0) fill.classList.add("bat-prof-bar-neg");
    bar.appendChild(fill);
    row.appendChild(bar);

    const detail = [];
    if (state.sort !== "duration") detail.push(fmtDuration(rec.duration));
    if (state.sort !== "vram_peak" && rec.vram_peak) detail.push(`VRAM ${fmtBytes(rec.vram_peak)}`);
    if (rec.ram_delta) detail.push(`RAM ${fmtBytes(rec.ram_delta, true)}`);
    if (rec.bytes_out) detail.push(`out ${fmtBytes(rec.bytes_out)}`);
    if (rec.io_read || rec.io_write) detail.push(`disk ${fmtBytes((rec.io_read || 0) + (rec.io_write || 0))}`);
    if (rec.cached) detail.push("cached");
    else if (rec.skipped) detail.push("not run");
    if (rec.async_node || rec.entries > 1) detail.push(`${rec.entries}× entered`);
    const sub = el("div", "bat-prof-sub", detail.join(" · "));
    if (rec.desc_out) sub.title = `output ${rec.desc_out}`;
    row.appendChild(sub);

    row.onclick = () => jumpToNode(rec);
    row.title = "Click to select and centre this node on the canvas";
    list.appendChild(row);
  }
  return list;
}

// ── footer ──────────────────────────────────────────────────────────
function buildFooter() {
  const f = el("div", "bat-prof-footer");

  const sync = el("button", `bat-prof-btn${state.config.sync_cuda ? " bat-prof-btn-on" : ""}`, "GPU sync");
  sync.title = "CUDA runs asynchronously, so without a sync at each node boundary "
    + "GPU work is billed to whichever later node happens to block. Leave this on "
    + "unless you are chasing sub-millisecond CPU timings.";
  sync.onclick = () => pushConfig({ sync_cuda: !state.config.sync_cuda });
  f.appendChild(sync);

  const rep = el("button", "bat-prof-btn bat-prof-btn-primary", "Copy report");
  rep.title = "Copy a plain-text diagnostic of this run — machine ceilings, the "
    + "memory trajectory, which node peaked, which nodes kept memory, and (after a "
    + "crash) which node was running when the process died. Paste it straight into "
    + "a chat message.";
  rep.onclick = () => copyReport();
  f.appendChild(rep);

  const exp = el("button", "bat-prof-btn", "Export");
  exp.title = "Download this run as JSON";
  exp.onclick = () => exportRun();
  f.appendChild(exp);

  const clear = el("button", "bat-prof-btn", "Clear");
  clear.title = "Discard the stored profiles for this workflow only";
  clear.onclick = () => {
    store.clear(state.wfKey);
    state.selectedRun = null;
    render();
  };
  f.appendChild(clear);

  return f;
}

/**
 * Copy text to the clipboard.
 *
 * `navigator.clipboard` only exists in a secure context, and ComfyUI is
 * routinely reached over plain http:// on a LAN address — where it is
 * simply undefined. So the deprecated execCommand path below is not
 * legacy cruft, it is the path that actually runs most of the time.
 */
async function copyText(text) {
  try {
    if (window.isSecureContext && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (e) { /* fall through to the textarea */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, text.length);
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (e) {
    return false;
  }
}

/** Last resort: show the report so it can be selected by hand. */
function showReportFallback(text) {
  const overlay = el("div", "bat-prof-overlay");
  const box = el("div", "bat-prof-overlay-box");
  box.appendChild(el("div", "bat-prof-overlay-title",
    "Copy failed — select all and copy manually"));
  const ta = document.createElement("textarea");
  ta.className = "bat-prof-overlay-text";
  ta.value = text;
  ta.readOnly = true;
  box.appendChild(ta);
  const close = el("button", "bat-prof-btn", "Close");
  close.onclick = () => overlay.remove();
  box.appendChild(close);
  overlay.appendChild(box);
  state.root.appendChild(overlay);
  ta.focus();
  ta.select();
}

async function copyReport() {
  const run = activeRun();
  if (!run) { toast("No run recorded for this workflow yet."); return; }
  await fetchSystem();
  let text;
  try {
    text = buildReport({
      run,
      runCount: runsForTab().length,
      workflowName: state.wfName,
      system: systemStats,
      config: state.config,
    });
  } catch (e) {
    console.error("[BAT Profiler] report build failed:", e);
    toast("Could not build the report — see console.");
    return;
  }
  const ok = await copyText(text);
  if (ok) {
    toast(`Report copied (${(text.length / 1024).toFixed(1)} KB) — paste it into chat.`);
  } else {
    showReportFallback(text);
  }
}

function exportRun() {
  const run = activeRun();
  if (!run) return;
  try {
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `bat-profile-${(state.wfName || "workflow").replace(/\W+/g, "_")}-${Math.round(run.started)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (e) {
    toast("Export failed — see console.");
    console.warn("[BAT Profiler] export failed:", e);
  }
}

// ─────────────────────────────────────────────────────────────────────
// Tab-change watch
//
// There is no dependable "active workflow changed" event across
// frontend versions, so we poll — but only while the panel is on
// screen, and the loop terminates the moment its root is detached
// rather than rescheduling itself forever.
// ─────────────────────────────────────────────────────────────────────
function startWatch(root) {
  stopWatch();
  state.poll = setInterval(() => {
    if (!panelAlive()) { stopWatch(); return; }
    // First tick after the sidebar actually mounts us: redraw, because
    // the charts were sized against a detached element with no width.
    if (root.isConnected && !state.sized) {
      state.sized = true;
      render();
      return;
    }
    const key = currentKey();
    if (key !== state.wfKey) {
      state.wfKey = key;
      state.wfName = currentName();
      state.selectedRun = null;
      render();
    }
  }, 750);
  state.heartbeat = setInterval(() => {
    if (!panelAlive()) { stopWatch(); return; }
    heartbeat();
  }, 8000);
  heartbeat();
}

function stopWatch() {
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
  if (state.heartbeat) { clearInterval(state.heartbeat); state.heartbeat = null; }
}

// ─────────────────────────────────────────────────────────────────────
// Styles — every class prefixed `bat-prof-`. The packs in this suite
// share one global stylesheet namespace, and a generic class name here
// silently restyles somebody else's panel.
// ─────────────────────────────────────────────────────────────────────
const CSS = `
.bat-prof-root { display:flex; flex-direction:column; gap:8px; padding:8px;
  height:100%; box-sizing:border-box; overflow:hidden; position:relative;
  font-size:12px; color:var(--fg-color,#ddd); }
.bat-prof-header { display:flex; flex-direction:column; gap:6px; }
.bat-prof-header-top { display:flex; align-items:center; gap:8px; }
.bat-prof-wf { flex:1; font-weight:600; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.bat-prof-live { font-size:10px; text-transform:uppercase; letter-spacing:.06em;
  padding:2px 6px; border-radius:9px; background:var(--comfy-input-bg,#222);
  color:var(--descrip-text,#888); flex:none; }
.bat-prof-live-on { background:#2a4d2a; color:#8fd98f; }
.bat-prof-live-off { background:#4d3a2a; color:#e0b080; }
.bat-prof-select { width:100%; background:var(--comfy-input-bg,#222);
  color:var(--input-text,#ddd); border:1px solid var(--border-color,#444);
  border-radius:4px; padding:3px 5px; font-size:11px; }
.bat-prof-power { display:flex; align-items:center; gap:9px; cursor:pointer;
  padding:7px 9px; border-radius:6px; user-select:none;
  background:var(--comfy-input-bg,#222); border:1px solid var(--border-color,#444);
  transition:background .15s, border-color .15s; }
.bat-prof-power:hover { border-color:#888; }
.bat-prof-power:focus-visible { outline:2px solid #6aa9ff; outline-offset:1px; }
.bat-prof-power-on { background:#1e3b26; border-color:#3f7a52; }
.bat-prof-switch { position:relative; flex:none; width:34px; height:18px;
  border-radius:9px; background:#555; transition:background .15s; }
.bat-prof-power-on .bat-prof-switch { background:#4fae6f; }
.bat-prof-knob { position:absolute; top:2px; left:2px; width:14px; height:14px;
  border-radius:50%; background:#eee; transition:transform .15s; }
.bat-prof-power-on .bat-prof-knob { transform:translateX(16px); }
.bat-prof-power-text { flex:1; min-width:0; }
.bat-prof-power-title { font-weight:600; font-size:12px; }
.bat-prof-power-on .bat-prof-power-title { color:#9fe0b0; }
.bat-prof-power-sub { font-size:10px; color:var(--descrip-text,#888);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bat-prof-charts { display:flex; flex-direction:column; gap:6px; }
.bat-prof-chart-head { display:flex; justify-content:space-between;
  font-size:10px; color:var(--descrip-text,#888); margin-bottom:1px; }
.bat-prof-canvas { width:100%; height:56px; display:block; }
.bat-prof-summary { display:flex; flex-wrap:wrap; gap:6px; }
.bat-prof-stat { flex:1 1 60px; background:var(--comfy-input-bg,#222);
  border-radius:4px; padding:4px 6px; }
.bat-prof-stat-k { font-size:9px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--descrip-text,#888); }
.bat-prof-stat-v { font-size:12px; font-weight:600; }
.bat-prof-warn { flex:1 1 100%; font-size:11px; color:#e0a060; }
.bat-prof-controls { display:flex; gap:4px; align-items:center; }
.bat-prof-sort { flex:1; }
.bat-prof-btn { background:var(--comfy-input-bg,#222); color:var(--input-text,#ddd);
  border:1px solid var(--border-color,#444); border-radius:4px; padding:3px 7px;
  font-size:11px; cursor:pointer; white-space:nowrap; }
.bat-prof-btn:hover { border-color:#888; }
.bat-prof-btn-on { background:#2f4a6d; border-color:#4a7ab0; color:#cfe3ff; }
.bat-prof-icon { min-width:26px; }
.bat-prof-list { flex:1; overflow-y:auto; overflow-x:hidden;
  display:flex; flex-direction:column; gap:3px; margin:0 -2px; padding:0 2px; }
.bat-prof-row { background:var(--comfy-input-bg,#222); border-radius:4px;
  padding:5px 7px; cursor:pointer; border-left:2px solid transparent; }
.bat-prof-row:hover { border-left-color:#6aa9ff; background:var(--comfy-menu-bg,#2a2a2a); }
.bat-prof-row-error { border-left-color:#c05050; }
.bat-prof-row-cached { opacity:.55; }
.bat-prof-row-line { display:flex; justify-content:space-between; gap:6px;
  align-items:baseline; }
.bat-prof-name { flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bat-prof-value { font-variant-numeric:tabular-nums; font-weight:600; flex:none; }
.bat-prof-bar { height:3px; background:var(--border-color,#3a3a3a); border-radius:2px;
  margin:4px 0 3px; overflow:hidden; }
.bat-prof-bar-fill { height:100%; background:#6aa9ff; border-radius:2px; }
.bat-prof-bar-neg { background:#5fbf7f; }
.bat-prof-sub { font-size:10px; color:var(--descrip-text,#888);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bat-prof-btn-primary { background:#2f4a6d; border-color:#4a7ab0; color:#cfe3ff;
  font-weight:600; }
.bat-prof-btn-primary:hover { background:#3a5a83; border-color:#6f9fd8; }
.bat-prof-warn-hard { color:#f0a0a0; font-weight:600; }
.bat-prof-overlay { position:absolute; inset:0; background:rgba(0,0,0,.75);
  display:flex; align-items:center; justify-content:center; padding:10px; z-index:20; }
.bat-prof-overlay-box { display:flex; flex-direction:column; gap:6px;
  width:100%; height:100%; }
.bat-prof-overlay-title { font-size:11px; color:#f0c48a; }
.bat-prof-overlay-text { flex:1; width:100%; resize:none; font-family:monospace;
  font-size:10px; background:#111; color:#ddd; border:1px solid #444; border-radius:4px;
  padding:6px; }
.bat-prof-footer { display:flex; gap:4px; flex-wrap:wrap; }
.bat-prof-empty { padding:14px 8px; text-align:center; font-size:11px;
  color:var(--descrip-text,#888); }
.bat-prof-toast { position:absolute; bottom:12px; left:12px; right:12px;
  background:#333; border:1px solid #555; border-radius:4px; padding:6px 8px;
  font-size:11px; opacity:0; pointer-events:none; transition:opacity .2s; }
.bat-prof-toast-on { opacity:1; }
`;

function injectCSS() {
  if (document.getElementById("bat-prof-css")) return;
  const s = document.createElement("style");
  s.id = "bat-prof-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

// ─────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────
app.registerExtension({
  name: "BAT.Profiler",
  async setup() {
    injectCSS();
    loadPrefs();
    await fetchState();
    await restoreArming();

    if (!app.extensionManager?.registerSidebarTab) {
      console.warn("[BAT Profiler] sidebar tab API unavailable");
      return;
    }
    try {
      app.extensionManager.registerSidebarTab({
        id: TAB_ID,
        icon: "pi pi-chart-bar",
        title: "BAT Profiler",
        tooltip: "Per-node time / RAM / VRAM / data profiling for this workflow",
        type: "custom",
        render: (root) => {
          state.root = root;
          // Fresh mount: the element may not be in the document yet.
          state.attached = !!root.isConnected;
          state.sized = false;
          state.wfKey = currentKey();
          state.wfName = currentName();
          startWatch(root);
          render();
        },
        destroy: () => { stopWatch(); store.flush(); },
      });
      console.log("[BAT Profiler] sidebar tab registered");
    } catch (e) {
      console.error("[BAT Profiler] sidebar registration failed:", e);
    }
  },
});
