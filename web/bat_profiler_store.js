/**
 * BAT Profiler — per-workflow persistent store
 * ============================================
 *
 * Two jobs, both of which are the reason the panel is useful rather
 * than merely pretty:
 *
 * 1. **Isolation.** A profile belongs to the workflow it was run from.
 *    Open five tabs and each sees only its own history — the analysis
 *    of a WAN render must never show up in an unrelated comp graph.
 *    Runs are bucketed by a stable workflow key (see `workflowKey`).
 *
 * 2. **Survival.** The backend keeps its run history in RAM, so it
 *    loses everything if the server dies — which is exactly the crash
 *    you are trying to diagnose. This store is the copy that outlives
 *    it: node records are written to localStorage as they stream in,
 *    so after an OOM kill you reload the page and the last thing the
 *    graph did is still sitting there, with the node that did it at
 *    the top of the list.
 *
 * Deliberately free of any ComfyUI import so it can be exercised
 * head-less (see tests/test_profiler_store.py, which runs this file
 * under quickjs against a stub storage).
 */

export const NS = "bat.profiler.v1";
const INDEX_KEY = `${NS}.index`;

// Retention. Sized so a heavy user with a dozen workflows stays well
// inside the ~5 MB localStorage budget: a 200-node run is ~90 KB of
// node records, and samples are decimated to a fixed ceiling.
export const MAX_RUNS_PER_WORKFLOW = 6;
export const MAX_SAMPLES_PER_RUN = 900;
export const MAX_WORKFLOWS = 24;

/** FNV-1a — short, stable, collision-tolerable key for a workflow path. */
export function hashKey(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/**
 * Resolve the active workflow to a stable identity.
 *
 * `path` is preferred because it survives a rename of the tab and a
 * reload of the browser. Unsaved workflows have no path, so we fall
 * back through the workflow store's own key/id before finally using
 * the display name — at which point two unsaved tabs both called
 * "Unsaved Workflow" would share a bucket, which is the least-bad
 * outcome available and self-corrects the moment either is saved.
 */
export function workflowKey(wf) {
  if (!wf) return null;
  const id = wf.path || wf.key || wf.id || (wf.name ? `unsaved:${wf.name}` : null);
  return id ? String(id) : null;
}

export class ProfilerStore {
  constructor(storage) {
    this.storage = storage || (typeof localStorage !== "undefined" ? localStorage : null);
    this.cache = new Map();      // wfKey -> bucket
    this.dirty = new Set();      // wfKeys awaiting flush
    this.flushTimer = null;
    this.flushDelay = 1000;
  }

  // ── raw storage ────────────────────────────────────────────────
  _read(key) {
    if (!this.storage) return null;
    try {
      const raw = this.storage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  _write(key, value) {
    if (!this.storage) return false;
    const payload = JSON.stringify(value);
    try {
      this.storage.setItem(key, payload);
      return true;
    } catch (e) {
      // Almost certainly QuotaExceededError. Drop the least recently
      // touched workflows and try once more; if it still fails we give
      // up silently rather than throwing inside a websocket handler.
      this.evict(0.5);
      try {
        this.storage.setItem(key, payload);
        return true;
      } catch (e2) {
        return false;
      }
    }
  }

  _bucketKey(wfKey) {
    return `${NS}.wf.${hashKey(wfKey)}`;
  }

  // ── index (LRU across workflows) ───────────────────────────────
  index() {
    return this._read(INDEX_KEY) || { workflows: {} };
  }

  touch(wfKey) {
    const idx = this.index();
    idx.workflows[wfKey] = { seen: Date.now(), bucket: this._bucketKey(wfKey) };
    const entries = Object.entries(idx.workflows).sort((a, b) => b[1].seen - a[1].seen);
    if (entries.length > MAX_WORKFLOWS) {
      for (const [k, v] of entries.slice(MAX_WORKFLOWS)) {
        try { this.storage && this.storage.removeItem(v.bucket); } catch (e) { /* ignore */ }
        delete idx.workflows[k];
        this.cache.delete(k);
      }
    }
    this._write(INDEX_KEY, idx);
  }

  /** Drop the oldest `fraction` of workflow buckets to reclaim quota. */
  evict(fraction) {
    const idx = this.index();
    const entries = Object.entries(idx.workflows).sort((a, b) => a[1].seen - b[1].seen);
    const drop = Math.max(1, Math.floor(entries.length * (fraction || 0.5)));
    for (const [k, v] of entries.slice(0, drop)) {
      try { this.storage && this.storage.removeItem(v.bucket); } catch (e) { /* ignore */ }
      delete idx.workflows[k];
      this.cache.delete(k);
    }
    try { this.storage && this.storage.setItem(INDEX_KEY, JSON.stringify(idx)); } catch (e) { /* ignore */ }
  }

  // ── buckets ────────────────────────────────────────────────────
  bucket(wfKey) {
    if (!wfKey) wfKey = "__unattributed__";
    if (this.cache.has(wfKey)) return this.cache.get(wfKey);
    const b = this._read(this._bucketKey(wfKey)) || { workflow: wfKey, runs: [] };
    this.cache.set(wfKey, b);
    return b;
  }

  markDirty(wfKey) {
    this.dirty.add(wfKey || "__unattributed__");
    if (this.flushTimer) return;
    // Debounced: a hard crash costs at most one second of records, and
    // the record that matters — the last node to complete before the
    // process died — has virtually always already landed.
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      this.flush();
    }, this.flushDelay);
  }

  flush() {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    for (const wfKey of this.dirty) {
      const b = this.cache.get(wfKey);
      if (b) this._write(this._bucketKey(wfKey), b);
    }
    this.dirty.clear();
  }

  // ── run lifecycle ──────────────────────────────────────────────
  beginRun(wfKey, summary) {
    const b = this.bucket(wfKey);
    let run = b.runs.find((r) => r.prompt_id === summary.prompt_id);
    if (!run) {
      run = {
        prompt_id: summary.prompt_id,
        workflow: wfKey,
        started: summary.started || Date.now() / 1000,
        ended: null,
        status: "running",
        baseline: summary.baseline || {},
        nodes: {},
        order: [],
        samples: [],
        titles: {},
      };
      b.runs.unshift(run);
      if (b.runs.length > MAX_RUNS_PER_WORKFLOW) b.runs.length = MAX_RUNS_PER_WORKFLOW;
    }
    this.touch(wfKey || "__unattributed__");
    this.markDirty(wfKey);
    return run;
  }

  findRun(promptId) {
    for (const [wfKey, b] of this.cache) {
      const run = b.runs.find((r) => r.prompt_id === promptId);
      if (run) return { wfKey, run };
    }
    return null;
  }

  upsertNode(promptId, record, title) {
    const hit = this.findRun(promptId);
    if (!hit) return null;
    const { wfKey, run } = hit;
    if (!(record.node_id in run.nodes)) run.order.push(record.node_id);
    run.nodes[record.node_id] = record;
    if (title) run.titles[record.node_id] = title;
    this.markDirty(wfKey);
    return run;
  }

  pushSamples(promptId, samples) {
    const hit = this.findRun(promptId);
    if (!hit) return null;
    const { wfKey, run } = hit;
    for (const s of samples) run.samples.push(s);
    // Halve resolution over the whole series rather than dropping the
    // head, so a long run keeps its full shape as it ages. Loops until
    // actually under budget: one pass only halves, so a burst larger
    // than twice the cap would otherwise leave the buffer over budget
    // for good. The newest sample is always kept — it is the one the
    // live chart is drawing.
    while (run.samples.length > MAX_SAMPLES_PER_RUN) {
      const last = run.samples[run.samples.length - 1];
      const kept = [];
      for (let i = 0; i < run.samples.length; i += 2) kept.push(run.samples[i]);
      if (kept[kept.length - 1] !== last) kept.push(last);
      run.samples = kept;
    }
    this.markDirty(wfKey);
    return run;
  }

  endRun(promptId, summary) {
    const hit = this.findRun(promptId);
    if (!hit) return null;
    const { wfKey, run } = hit;
    run.ended = summary.ended || Date.now() / 1000;
    run.status = summary.status || "ok";
    if (summary.error) run.error = summary.error;
    this.markDirty(wfKey);
    this.flush();   // run boundaries are cheap and worth persisting at once
    return run;
  }

  listRuns(wfKey) {
    return this.bucket(wfKey).runs;
  }

  clear(wfKey) {
    const key = wfKey || "__unattributed__";
    this.cache.set(key, { workflow: key, runs: [] });
    this.markDirty(key);
    this.flush();
  }
}
