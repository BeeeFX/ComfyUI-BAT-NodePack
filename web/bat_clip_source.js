/**
 * Where a node's IMAGE batch comes from — shared by the segmenter editors.
 *
 * Moved verbatim out of bat_sec_segmenter.js (2026-11-04) so the SAM 3.1
 * Segmenter can preview any frame of the incoming clip the same way SeC does,
 * without a second copy of the loader trim maths. Two things depend on these
 * staying exact:
 *
 *   - sourceFrameIndex / sourceBatchLength must be inverses of what each
 *     loader actually emits. tests/verify_sec_frame_bounds.py replays them
 *     against the loaders' real selection semantics under quickjs.
 *   - Nothing here may touch `app` or the DOM above the "per-node preview
 *     state" marker: that test evaluates the slice as a plain script.
 */

import { api } from "../../scripts/api.js";

// Node types we know how to pull an exact frame out of, mapped to the widget
// holding the file path. Each of these also trims/steps its output, so the
// batch index has to be mapped back to a file index — see sourceFrameIndex()
// and sourceBatchLength(), which must stay in step with each other.
//
// VoltLoader's `path` can hold a VRI rather than a filesystem path. We don't
// try to tell them apart: the frame-picker route simply fails on a VRI and we
// fall through to the next strategy, which is the right outcome either way.
export const PATH_SOURCES = {
    Bat_VideoLoader: "path",
    Bat_FramePicker: "path",
    VoltLoader: "path",
    // 🦇 Loader (2026-09-15) — same frames[skip::nth][:cap] window as VoltLoader
    // (bat_loader._select). Missing here until 2026-11-04, so neither segmenter
    // could preview a clip loaded through it before the first run.
    Bat_Loader: "path",
};

export const MAX_UPSTREAM_HOPS = 6;

// ── upstream resolution ─────────────────────────────────────────────────────

/** Walk back from `node`'s IMAGE inputs looking for a known file-backed source. */
export function findPathSource(node, hops = 0) {
    if (!node?.graph || hops > MAX_UPSTREAM_HOPS) return null;

    const pathWidgetName = PATH_SOURCES[node.type];
    if (pathWidgetName) {
        const w = node.widgets?.find((w) => w.name === pathWidgetName);
        if (w?.value) return { node, path: String(w.value) };
    }

    for (const input of node.inputs ?? []) {
        if (input.link == null) continue;
        // Only follow image-ish links. Following a MASK or a MODEL upstream
        // would wander into an unrelated branch of the graph.
        if (input.type && input.type !== "IMAGE" && input.type !== "*") continue;
        const link = node.graph.links?.get(input.link);
        if (!link) continue;
        const origin = node.graph.getNodeById(link.origin_id);
        if (!origin) continue;
        const found = findPathSource(origin, hops + 1);
        if (found) return found;
    }
    return null;
}

export function widgetNum(node, name, fallback) {
    const v = node?.widgets?.find((w) => w.name === name)?.value;
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

/**
 * Map an index in the *batch* to an index in the *source file*.
 *
 * Batch index and file index are not the same thing once a loader trims or
 * decimates, so each known source needs its own mapping:
 *
 *   Bat_VideoLoader — emits [start_frame ..] every select_every_nth frame.
 *   VoltLoader      — emits frames[skip_first_frames::select_every_nth]
 *                     truncated to frame_load_cap.
 *   Bat_FramePicker — emits exactly one frame, the one at frame_index. The
 *                     batch index is therefore always 0 and carries no
 *                     information; using it would show file frame 0.
 */
export function sourceFrameIndex(sourceNode, batchIdx, fileFrames = 0) {
    const num = (name, fallback) => widgetNum(sourceNode, name, fallback);
    const n = Math.max(0, Number(fileFrames) || 0);

    if (sourceNode.type === "Bat_VideoLoader") {
        // load() clamps start INTO the clip before striding, so a start past
        // the end still yields the last frame. Without the clamp an
        // out-of-clip start_frame asked the frame route for a frame that
        // doesn't exist, and the preview silently fell through to a worse
        // strategy. Only possible when the caller knows the clip length.
        let start = Math.max(0, num("start_frame", 0));
        if (n) start = Math.min(start, n - 1);
        const step = Math.max(1, num("select_every_nth", 1));
        return start + batchIdx * step;
    }
    if (sourceNode.type === "VoltLoader" || sourceNode.type === "Bat_Loader") {
        // frames[skip::step][:cap] — see VoltLoader._load_standard/_load_exr
        // and bat_loader._select.
        // No clamp needed: a skip past the end makes the slice empty, so
        // sourceBatchLength reports 0 and no batch index is reachable.
        const skip = Math.max(0, num("skip_first_frames", 0));
        const step = Math.max(1, num("select_every_nth", 1));
        return skip + batchIdx * step;
    }
    if (sourceNode.type === "Bat_FramePicker") {
        return Math.max(0, num("frame_index", 0));
    }
    return batchIdx;
}

/**
 * How many frames the batch will hold, given the source file's frame count.
 *
 * The inverse of sourceFrameIndex: it answers "what is the largest batch index
 * that exists", which is what bounds `frame_index_select`. Returns 0 when the
 * trim selects nothing.
 */
export function sourceBatchLength(sourceNode, fileFrames) {
    const num = (name, fallback) => widgetNum(sourceNode, name, fallback);
    const n = Math.max(0, Number(fileFrames) || 0);
    if (!n) return 0;

    if (sourceNode.type === "Bat_FramePicker") return 1;

    if (sourceNode.type === "Bat_VideoLoader") {
        // Mirrors Bat_VideoLoader.load(): start and end are clamped into the
        // clip, end < 0 means "to the last frame", then it strides.
        const start = Math.max(0, Math.min(num("start_frame", 0), n - 1));
        let end = num("end_frame", -1);
        if (end < 0 || end >= n) end = n - 1;
        if (end < start) end = start;
        const step = Math.max(1, num("select_every_nth", 1));
        return Math.floor((end - start) / step) + 1;
    }

    if (sourceNode.type === "VoltLoader" || sourceNode.type === "Bat_Loader") {
        const skip = Math.max(0, num("skip_first_frames", 0));
        const step = Math.max(1, num("select_every_nth", 1));
        const cap = Math.max(0, num("frame_load_cap", 0));   // 0 = no cap
        if (skip >= n) return 0;
        const strided = Math.floor((n - 1 - skip) / step) + 1;
        return cap > 0 ? Math.min(cap, strided) : strided;
    }

    return n;
}

/**
 * Frame count of a source file, cached per path.
 *
 * Shared by the plate preview (which needs it to map a batch index onto the
 * right file frame) and the widget bounds (which need it to know how long the
 * batch will be). Both used to be able to disagree; one lookup means they
 * can't. 0 means "unknown" — a VRI, an unreadable path, or no server.
 */
const _fileFrameCache = new Map();
const FILE_FRAME_CACHE_MAX = 64;

export async function fileFrameCount(path) {
    if (_fileFrameCache.has(path)) return _fileFrameCache.get(path);
    let frames = 0;
    try {
        const res = await fetch(api.apiURL(
            `/bat/frame-picker/info?path=${encodeURIComponent(path)}`));
        if (res.ok) {
            const info = await res.json();
            if (info?.ok) frames = Number(info.frames) || 0;
        }
    } catch (_) { /* offline or unreadable — treat as unknown */ }
    // Cache the miss too: a VRI will never resolve and re-asking on every
    // arrow-key scrub would hammer the route.
    if (_fileFrameCache.size >= FILE_FRAME_CACHE_MAX) {
        _fileFrameCache.delete(_fileFrameCache.keys().next().value);
    }
    _fileFrameCache.set(path, frames);
    return frames;
}

// ── per-node preview state (end of the pure section) ────────────────────────

export function loadImage(src, { crossOrigin = false } = {}) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        if (crossOrigin) img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}
