/**
 * BAT — shared node-lifecycle teardown for the on-node editors.
 *
 * Why this exists
 * ---------------
 * The canvas editors each spin up long-lived resources when their node is
 * created — playback `setInterval`s, `ResizeObserver`s, `IntersectionObserver`s,
 * `requestAnimationFrame` watcher loops, and window-level pointer listeners —
 * but none of them (outside the points-editor fork) had an `onRemoved` hook.
 * Deleting a node therefore left all of that running against a detached DOM,
 * forever:
 *
 *   • Playback intervals kept firing setFrame() → render() on a canvas that is
 *     no longer in the document.
 *   • The rAF watchers in bat_crop / bat_ref_aligner were the worst case: their
 *     `if (!root.isConnected) { requestAnimationFrame(watch); return; }` guard
 *     RE-SCHEDULED itself when detached, so every Crop / RefAligner node ever
 *     placed contributed a permanent per-frame callback that kept the whole
 *     editor closure alive (decoded background image included).
 *   • Retained `state.previewFrames` arrays hold up to a few hundred decoded
 *     Image objects per node — hundreds of MB on a busy shot graph.
 *
 * On a long studio session (nodes added/removed repeatedly) this accumulates
 * without bound and is a large part of why the editors get sluggish.
 *
 * Usage
 * -----
 *   import { registerCleanup, batTrack, isNodeAlive } from "./bat_lifecycle.js";
 *
 *   const track = batTrack(node);              // per-node resource tracker
 *   track.interval(setInterval(tick, 40));     // cleared on removal
 *   track.observer(new ResizeObserver(render), root);   // observed + disconnected
 *   track.listener(window, "mousemove", onMove);        // removed on removal
 *   track.rafLoop(function watch() { ... });   // stops when the node dies
 *   track.dispose(() => { state.previewFrames = null; });  // arbitrary cleanup
 *
 * Every editor should also gate its own render()/tick paths on
 * `isNodeAlive(node)` when it can be reached asynchronously.
 */

const KEY = "_batLifecycle";

/**
 * Identity of the currently-open workflow, for scoping per-node localStorage.
 *
 * The editors' preview caches were keyed on `node.id` alone, which is only
 * unique WITHIN a graph — so opening a different workflow whose node 14 happens
 * to be a Roto restored the other shot's background plate, at the wrong
 * imgW/imgH, and shapes drew against a bogus reference. Including the workflow
 * identity makes the key unique across graphs.
 *
 * `app` is imported lazily (inside the call) so this module stays usable in
 * contexts where the ComfyUI app module isn't loaded.
 */
export function batWorkflowKey(app) {
    try {
        const g = app?.graph;
        const id = g?.extra?.workflow_id || g?.extra?.workflowId
                || g?.extra?.ds?.workflow_id || "";
        if (id) return String(id);
    } catch (_) { /* fall through */ }
    try { return String(window.location?.pathname || "_"); } catch (_) { return "_"; }
}

/** Build a workflow-scoped, node-scoped localStorage key. */
export function batNodeCacheKey(app, prefix, node) {
    return `${prefix}_${batWorkflowKey(app)}_${node?.id ?? "_"}`;
}

function bag(node) {
    if (!node[KEY]) {
        node[KEY] = {
            dead: false,
            intervals: [],
            timeouts: [],
            observers: [],
            listeners: [],
            disposers: [],
        };
    }
    return node[KEY];
}

/** True until the node's onRemoved has fired. Async callbacks should check it. */
export function isNodeAlive(node) {
    return !!node && !(node[KEY] && node[KEY].dead);
}

/**
 * Chain a callback onto an existing node method without clobbering whatever is
 * already there (other extensions hook these too).
 */
function chain(target, name, fn) {
    const prev = target[name];
    target[name] = function (...args) {
        const r = prev ? prev.apply(this, args) : undefined;
        try { fn.call(this, ...args); } catch (e) {
            console.error(`[BAT.lifecycle] ${name} handler failed:`, e);
        }
        return r;
    };
}

/**
 * Install the teardown hook on a node instance and return a tracker.
 * Safe to call more than once per node — the hook is only installed once.
 */
export function batTrack(node) {
    const b = bag(node);
    if (!b._hooked) {
        b._hooked = true;
        chain(node, "onRemoved", () => runCleanup(node));
    }
    return {
        interval(id) { b.intervals.push(id); return id; },
        timeout(id) { b.timeouts.push(id); return id; },
        /** Track an observer; pass `target` to observe it in the same call. */
        observer(obs, target, options) {
            if (target && obs && typeof obs.observe === "function") {
                obs.observe(target, options);
            }
            b.observers.push(obs);
            return obs;
        },
        listener(target, type, handler, options) {
            target.addEventListener(type, handler, options);
            b.listeners.push([target, type, handler, options]);
            return handler;
        },
        /**
         * Run a rAF loop that STOPS when the node is removed. `fn` is called
         * once per frame; it must not re-schedule itself.
         */
        rafLoop(fn) {
            const step = () => {
                if (b.dead) return;              // node gone: stop rescheduling
                try { fn(); } catch (e) {
                    console.error("[BAT.lifecycle] rAF loop failed, stopping:", e);
                    return;
                }
                requestAnimationFrame(step);
            };
            requestAnimationFrame(step);
        },
        dispose(fn) { b.disposers.push(fn); return fn; },
        get dead() { return b.dead; },
    };
}

/** Tear everything down. Idempotent. */
export function runCleanup(node) {
    const b = node && node[KEY];
    if (!b || b.dead) return;
    b.dead = true;
    for (const id of b.intervals) { try { clearInterval(id); } catch (_) {} }
    for (const id of b.timeouts) { try { clearTimeout(id); } catch (_) {} }
    for (const o of b.observers) { try { o.disconnect(); } catch (_) {} }
    for (const [t, ty, h, o] of b.listeners) {
        try { t.removeEventListener(ty, h, o); } catch (_) {}
    }
    for (const fn of b.disposers) {
        try { fn(); } catch (e) { console.error("[BAT.lifecycle] disposer failed:", e); }
    }
    b.intervals.length = 0;
    b.timeouts.length = 0;
    b.observers.length = 0;
    b.listeners.length = 0;
    b.disposers.length = 0;
}

/**
 * Convenience for the common "register cleanup for a node type" case: install
 * the onRemoved hook on the prototype so every instance is covered even if the
 * editor build path is skipped/guarded.
 */
export function registerCleanup(nodeType) {
    chain(nodeType.prototype, "onRemoved", function () { runCleanup(this); });
}

/* ══════════════════════════════════════════════════════════════════════
 * Surviving a graph reload (Ctrl+Z / Ctrl+Y)
 * ══════════════════════════════════════════════════════════════════════
 *
 * Why this exists
 * ---------------
 * Undo in ComfyUI is not a graph diff. `ChangeTracker.updateState()` does:
 *
 *     await app.loadGraphData(prevState, false, false, this.workflow, ...)
 *
 * which runs `LGraph.clear()` — firing `onRemoved` on EVERY node, i.e. the
 * teardown above — and then rebuilds every node from the workflow JSON. So one
 * Ctrl+Z anywhere on the canvas destroys and recreates every BAT editor on the
 * graph, and anything an editor holds that is not in the serialised widget
 * values is gone.
 *
 * The preview pixels are exactly that. They arrive ONCE, as base64 in the
 * node's `onExecuted` message, and nothing in core replays them: the frontend
 * keeps the payload in `app.nodeOutputs` but only re-reads it for its own
 * image widgets. Every canvas editor in this pack therefore went black (or
 * froze on the single low-res localStorage thumbnail) after an unrelated undo,
 * and only a re-run brought it back.
 *
 * How it works
 * ------------
 * Remember the last `onExecuted` payload per node, and replay it into the
 * rebuilt node once the graph has finished configuring. Replay goes through
 * `node.onExecuted(message)` — the exact path a real run takes — so every
 * editor's existing ingest code is reused verbatim and no editor needs to know
 * that a reload happened. The handlers in this pack are all pure "ingest this
 * payload" and all guarded on their `_batXxxIngest` hook existing, so replay is
 * idempotent and is a no-op if the editor didn't build.
 *
 * Scoping
 * -------
 * The key folds in `node.graph.id` — the graph's own UUID, which
 * `LGraph.asSerialisable()` writes out and `_configureBase()` restores. That
 * makes it stable across an undo of the same workflow, but distinct for every
 * other workflow AND for each subgraph. Opening a different shot can therefore
 * never replay this shot's frames into a node that happens to share its id
 * (the bug `batNodeCacheKey` above exists to avoid, in the same spirit).
 *
 * One residual side effect
 * ------------------------
 * Roto / AnimatedCrop / AnimatedGrade stamp the plate's imgW/imgH/frameCount
 * into their `state` JSON widget from inside the ingest, so a replay writes
 * them too. In the case that matters — undoing something that happened AFTER
 * the run — the widget already holds those numbers and the write is
 * byte-identical, so the ChangeTracker sees no diff. Undoing back to BEFORE
 * that node's first run does stamp them onto a state that lacked them, which
 * marks the workflow modified and can cost one extra undo step. The values are
 * correct metadata about the plate on screen (they are what makes shapes draw
 * at the right relative scale), so this is left as-is deliberately rather than
 * papered over with a suppress-persist flag whose lifetime cannot be defined:
 * the ingests settle asynchronously, so there is no honest moment to clear it.
 *
 * Memory
 * ------
 * We retain a reference to the same payload object core's nodeOutputStore is
 * already holding — one Map entry per executed BAT node, not a copy of the
 * frames. Entries for a graph are dropped when that graph is next configured
 * from scratch, and the whole map is bounded by MAX_STASH.
 */

/** Hard ceiling on remembered payloads, oldest evicted first. */
const MAX_STASH = 64;

/** `${graphId}:${nodeId}` -> the last onExecuted message for that node. */
const LAST_EXEC = new Map();

function execKey(node) {
    // node.graph is set by LGraph.add() before configure()/onExecuted, and is
    // the subgraph (not the root) for a nested node — which is what we want.
    const gid = node?.graph?.id ?? "_";
    return `${gid}:${node?.id ?? "_"}`;
}

/** Remember `message` as the latest execution result for `node`. */
export function batStashExecuted(node, message) {
    if (!node || !message) return;
    const key = execKey(node);
    // Re-insert so Map iteration order tracks recency for the eviction below.
    LAST_EXEC.delete(key);
    LAST_EXEC.set(key, message);
    while (LAST_EXEC.size > MAX_STASH) {
        LAST_EXEC.delete(LAST_EXEC.keys().next().value);
    }
}

/**
 * True once a replay has been scheduled for this node — i.e. its preview is
 * about to be repainted from the last real run.
 *
 * Editors that also restore a low-res thumbnail from localStorage must check
 * this and skip that path, otherwise the thumbnail's async `Image.onload` can
 * land after the replayed full-res strip and clobber it. The flag is set
 * synchronously inside `onAfterGraphConfigured`, which runs in the same task as
 * `configure()` — so it is always set before any image decode can complete.
 */
export function batPreviewWillReplay(node) {
    return !!(node && node._batExecReplay);
}

/** Replay the remembered payload into `node`. Returns true if there was one. */
export function batReplayExecuted(node) {
    if (!node) return false;
    const msg = LAST_EXEC.get(execKey(node));
    if (!msg) return false;
    try {
        node.onExecuted?.(msg);
    } catch (e) {
        console.error("[BAT.lifecycle] preview replay failed:", e);
        return false;
    }
    return true;
}

/**
 * Install stash + replay on a node type. Call once, at the top of the pack's
 * `beforeRegisterNodeDef` for any node whose preview is fed by `onExecuted`.
 *
 * Nothing happens on a fresh page load or for a node that has not run in this
 * session — the stash is empty and both hooks are no-ops.
 */
export function batReplayLastExecution(nodeType) {
    const proto = nodeType?.prototype;
    if (!proto || proto._batReplayHooked) return;
    proto._batReplayHooked = true;

    chain(proto, "onExecuted", function (message) {
        batStashExecuted(this, message);
    });

    // Fired by core's triggerCallbackOnAllNodes() after configure() finishes,
    // for nodes inside subgraphs too. Deferring by a task puts the replay after
    // the editors' own `setTimeout(0)` widget-state restore, which is queued
    // from the earlier onConfigure — so the replayed frames are the last word.
    chain(proto, "onAfterGraphConfigured", function () {
        const node = this;
        if (!LAST_EXEC.has(execKey(node))) return;
        node._batExecReplay = true;              // see batPreviewWillReplay()
        setTimeout(() => {
            if (isNodeAlive(node)) batReplayExecuted(node);
        }, 0);
    });
}
