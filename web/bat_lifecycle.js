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
