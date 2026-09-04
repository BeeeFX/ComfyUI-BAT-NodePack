/**
 * Bat SeC Segmenter — front-end.
 *
 * Puts the BAT Points Editor canvas on the Bat_SecSegmenter node and keeps its
 * background plate in sync with `frame_index_select`.
 *
 * The canvas itself is the *same* class the standalone Points Editor uses
 * (imported, not forked) — only the background sourcing differs. Reaching the
 * annotation frame without running a multi-minute segmentation takes three
 * strategies, tried best-first:
 *
 *   1. Post-run strip — after any execution the node ships back a strided JPEG
 *      strip of the whole clip (`ui.frames`). Scrubbing frame_index_select
 *      then re-previews instantly from memory, no re-run. This is the only
 *      source that is guaranteed to be the exact pixels SeC will see.
 *   2. Upstream file — if the chain feeding `frames` resolves to a node holding
 *      a file path, decode frame N server-side via the existing
 *      /bat/frame-picker/frame route (video, stills, EXR, image sequences).
 *      Live, exact, no execution. Approximate only if something between that
 *      node and this one retimes or trims.
 *   3. Upstream node preview — the stock resolveSourcePreview path (LoadImage's
 *      image widget, VHS's videopreview element). Gives frame 0 / the current
 *      video position rather than frame N.
 *
 * Plus the editor's own paste / drag / "Load Image" for setting a plate by hand,
 * which always wins until the next successful auto-update.
 *
 * A plate sourced this way lives only in the editor (and a per-node
 * localStorage copy of the annotation frame) — unlike a pasted one it never
 * becomes `properties.imgData`. Two places have to know that: editor_base's
 * "Reset canvas" carries the live image across instead of reloading imgData,
 * and clone() hands the copy the strip + cached plate so duplicating a node
 * doesn't blank its canvas.
 *
 * The same upstream resolution also bounds `frame_index_select` — see
 * updateFrameBounds. It indexes the frame batch, and the runtime RAISES on an
 * out-of-range value, so the widget's max is narrowed to the batch length as
 * soon as that can be worked out.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { chainCallback, resolveSourcePreview, captureVideoFrame } from "./bat_points_editor/utility.js";
import { BaseEditorCanvas } from "./bat_points_editor/editor_base.js";
import { BatPointsEditor } from "./bat_points_editor/point_editor_canvas.js";
import { batNodeCacheKey, batReplayLastExecution, batPreviewWillReplay } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_SecSegmenter";

// Widgets that exist only to carry editor state to Python. All hidden — the
// canvas is the interface. Names must match bat_sec_segmenter.py's INPUT_TYPES.
const PLUMBING_WIDGETS = [
    "points_store", "coordinates", "neg_coordinates",
    "bbox_store", "bboxes", "width", "height",
];

// Node types we know how to pull an exact frame out of, mapped to the widget
// holding the file path. Each of these also trims/steps its output, so the
// batch index has to be mapped back to a file index — see sourceFrameIndex()
// and sourceBatchLength(), which must stay in step with each other.
//
// VoltLoader's `path` can hold a VRI rather than a filesystem path. We don't
// try to tell them apart: the frame-picker route simply fails on a VRI and we
// fall through to the next strategy, which is the right outcome either way.
const PATH_SOURCES = {
    Bat_VideoLoader: "path",
    Bat_FramePicker: "path",
    VoltLoader: "path",
};

const MAX_UPSTREAM_HOPS = 6;

// ── upstream resolution ─────────────────────────────────────────────────────

/** Walk back from `node`'s IMAGE inputs looking for a known file-backed source. */
function findPathSource(node, hops = 0) {
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

function widgetNum(node, name, fallback) {
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
function sourceFrameIndex(sourceNode, batchIdx, fileFrames = 0) {
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
    if (sourceNode.type === "VoltLoader") {
        // frames[skip::step][:cap] — see VoltLoader._load_standard/_load_exr.
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
function sourceBatchLength(sourceNode, fileFrames) {
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

    if (sourceNode.type === "VoltLoader") {
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

async function fileFrameCount(path) {
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

// ── per-node preview state ──────────────────────────────────────────────────

function stripCacheKey(node) {
    // Workflow-scoped: keying on node.id alone collides across graphs, which
    // would restore another shot's plate at another shot's dimensions.
    return batNodeCacheKey(app, "bat_sec_plate", node);
}

function saveCachedPlate(node, data) {
    try { localStorage.setItem(stripCacheKey(node), JSON.stringify(data)); }
    catch (_) { /* quota or storage disabled — the plate is a convenience, not state */ }
}

function loadCachedPlate(node) {
    try {
        const raw = localStorage.getItem(stripCacheKey(node));
        return raw ? JSON.parse(raw) : null;
    } catch (_) { return null; }
}

function loadImage(src, { crossOrigin = false } = {}) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        if (crossOrigin) img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

// ── the editor subclass ─────────────────────────────────────────────────────

class BatSecPointsEditor extends BatPointsEditor {
    // Room for the three visible widgets under the canvas (frame_index_select,
    // auto_unload_model, mask_preview) — not the Points Editor's nine.
    get editorHeightOffset() { return 180; }
}

// ── background updating ─────────────────────────────────────────────────────

/**
 * Install `node.batSecSetPlate(img, realW, realH)`.
 *
 * Goes through handleImageLoad with an explicit coord override so the editor's
 * coord space is the clip's TRUE resolution even when the preview pixels are a
 * downscaled proxy. Clicks then land in real frame space and Python's
 * scale_prompts becomes a no-op. Without the override the coord space would
 * silently become the proxy's size.
 */
function installPlateSetter(node) {
    node.batSecSetPlate = (img, realW, realH) => {
        const editor = node.editor;
        if (!editor) return;
        const coord = (realW && realH) ? { width: realW, height: realH } : null;
        editor.handleImageLoad(img, img, coord);
    };
}

/** Strategy 1 — the strip returned by the last execution. */
async function plateFromStrip(node, batchIdx) {
    const strip = node._batSecStrip;
    if (!strip?.frames?.length) return false;

    const stride = Math.max(1, strip.stride || 1);
    // The strip is every Nth frame, so land on the nearest one we actually have
    // rather than failing. Clamp so an out-of-range index still shows something.
    const slot = Math.min(strip.frames.length - 1, Math.max(0, Math.round(batchIdx / stride)));
    const img = strip.frames[slot];
    if (!img) return false;

    node.batSecSetPlate(img, strip.w, strip.h);
    return true;
}

/** Strategy 2 — decode frame N out of an upstream file, server-side.
 *
 * Uses the frame-picker route rather than the video-loader one: it is the same
 * shape but also handles stills, EXR and image sequences.
 */
async function plateFromUpstreamFile(node, batchIdx) {
    const source = findPathSource(node);
    if (!source) return false;

    // Pass the clip length so the mapping clamps exactly the way the loader
    // does; 0 when we can't find out, which just means no clamp.
    const fileFrame = sourceFrameIndex(
        source.node, batchIdx, await fileFrameCount(source.path));
    const encPath = encodeURIComponent(source.path);

    // Ask for the file's true dimensions so the editor's coord space is native
    // even though the JPEG we fetch is capped at 1024 on the long edge. Only
    // video files answer this; for anything else we fall back to the JPEG's own
    // size and let Python's scale_prompts map the clicks back up.
    let realW = 0, realH = 0;
    try {
        const res = await fetch(api.apiURL(`/bat/video-info?path=${encPath}`));
        if (res.ok) {
            const info = await res.json();
            if (info?.ok) { realW = Number(info.width) || 0; realH = Number(info.height) || 0; }
        }
    } catch (_) { /* optimisation only */ }

    try {
        const img = await loadImage(
            api.apiURL(`/bat/frame-picker/frame?path=${encPath}&frame=${fileFrame}&max_w=1024`)
        );
        node.batSecSetPlate(img, realW, realH);
        return true;
    } catch (_) {
        return false;
    }
}

/** Strategy 3 — whatever preview the immediate upstream node already has. */
async function plateFromNodePreview(node) {
    const slots = (node.inputs ?? [])
        .map((inp, i) => (inp.name === "frames" ? i : -1))
        .filter((i) => i >= 0);

    for (const slot of slots) {
        const source = resolveSourcePreview(node, slot);
        if (!source) continue;

        if (source.isVideo && source.videoEl) {
            await new Promise((resolve) => {
                captureVideoFrame(source.videoEl, (canvas) => {
                    node.batSecSetPlate(canvas, canvas.width, canvas.height);
                    resolve();
                });
                // captureVideoFrame waits on loadeddata and may never fire.
                setTimeout(resolve, 2000);
            });
            return true;
        }
        if (source.url) {
            try {
                const img = await loadImage(source.url, { crossOrigin: true });
                node.batSecSetPlate(img, img.width, img.height);
                return true;
            } catch (_) { /* try the next slot */ }
        }
    }
    return false;
}

/**
 * Refresh the plate for the current frame_index_select.
 *
 * Serialised per node via a generation counter: frame_index_select fires a
 * callback per arrow-key press, and without this the slowest fetch would win
 * and leave the canvas showing a frame the user has already scrubbed past.
 */
async function refreshPlate(node, { allowNodePreview = true } = {}) {
    if (!node.editor) return;

    // Re-derive the widget bounds here as well as on connect/run, so retrimming
    // the upstream loader takes effect without a re-run. Cheap: it short-
    // circuits on an unchanged trim signature, and the file's frame count is
    // cached per path. Deliberately not awaited — scrubbing must stay
    // responsive. It cannot recurse: the clamp fires this function again, but
    // by then the signature matches and updateFrameBounds returns immediately.
    updateFrameBounds(node);

    const gen = (node._batSecPlateGen = (node._batSecPlateGen || 0) + 1);
    const idxWidget = node.widgets?.find((w) => w.name === "frame_index_select");
    const batchIdx = Math.max(0, Number(idxWidget?.value) || 0);

    const stale = () => gen !== node._batSecPlateGen || !node.editor;

    if (await plateFromStrip(node, batchIdx)) return;
    if (stale()) return;
    if (await plateFromUpstreamFile(node, batchIdx)) return;
    if (stale()) return;
    if (allowNodePreview && await plateFromNodePreview(node)) return;
}

/** Paint a cached plate onto the editor. Returns false if there was nothing
 *  usable, so callers can fall through to a live refresh. */
async function applyCachedPlate(node, cached) {
    if (!cached?.frame) return false;
    try {
        const img = await loadImage(`data:image/jpeg;base64,${cached.frame}`);
        if (!node.editor) return false;
        node.batSecSetPlate(img, cached.w, cached.h);
        return true;
    } catch (_) {
        return false;
    }
}

// ── frame_index_select bounds ───────────────────────────────────────────────

/**
 * Constrain `frame_index_select` to the batch that will actually arrive.
 *
 * The widget ships with max=999999 because Python can't know the batch length
 * until execution — but `bat_sec_runtime.segment` RAISES on an out-of-range
 * annotation index, so typing a plausible-looking frame number and hitting Run
 * costs you a failed job several seconds in. Once we can work out the length
 * we clamp the widget so the mistake isn't reachable.
 *
 * Two sources, best first:
 *   1. The post-run strip's `frame_count` — exact, it IS what the node got.
 *   2. The upstream file's frame count via /bat/frame-picker/info, mapped
 *      through the loader's trim by sourceBatchLength().
 *
 * When neither answers (an unknown loader, a VRI, a generated batch) the
 * bounds are left wide open — a max that's too small would block a perfectly
 * valid index, which is worse than no max at all.
 */
async function updateFrameBounds(node) {
    const widget = node.widgets?.find((w) => w.name === "frame_index_select");
    if (!widget) return;

    let count = Number(node._batSecStrip?.frameCount) || 0;

    if (!count) {
        const source = findPathSource(node);
        if (!source) return;
        // Cache per (path + trim settings): this runs on every connection
        // change and every execution, and the answer only moves when one of
        // those inputs does.
        const sig = [
            source.path,
            source.node.type,
            widgetNum(source.node, "start_frame", ""),
            widgetNum(source.node, "end_frame", ""),
            widgetNum(source.node, "skip_first_frames", ""),
            widgetNum(source.node, "frame_load_cap", ""),
            widgetNum(source.node, "select_every_nth", ""),
        ].join("|");
        if (node._batSecBoundsSig === sig) return;

        const fileFrames = await fileFrameCount(source.path);
        if (!fileFrames) return;   // unknown — better no max than a wrong one

        node._batSecBoundsSig = sig;
        count = sourceBatchLength(source.node, fileFrames);
    }

    if (!count || count < 1) return;

    const max = count - 1;
    widget.options = widget.options || {};
    widget.options.min = 0;
    widget.options.max = max;
    // Snap a now-illegal value into range rather than leaving it to fail at
    // execute time. Only ever narrows, so it can't silently move a valid pick.
    const current = Number(widget.value) || 0;
    if (current > max || current < 0) {
        widget.value = Math.min(Math.max(0, current), max);
        console.info(`[Bat SeC] frame_index_select ${current} is outside the ` +
                     `${count}-frame batch (0-${max}); clamped to ${widget.value}.`);
        widget.callback?.(widget.value);
    }
    node.graph?.setDirtyCanvas(true, true);
}

// ── extension ───────────────────────────────────────────────────────────────

app.registerExtension({
    name: "BATNodePack.SecSegmenter",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        // A graph reload (Ctrl+Z is one) destroys and rebuilds every node, so
        // replay the last run's preview payload into the new instance.
        batReplayLastExecution(nodeType);

        // Duplicating a node used to blank its canvas. The plate lives in two
        // places that a copy can't reach on its own — the in-memory strip from
        // the last run, and a localStorage entry keyed on the node's id (which
        // the copy doesn't have yet: LiteGraph assigns it in graph.add, after
        // clone() returns). So stash both on the copy here and let onAdded
        // apply them once it has an id.
        //
        // chainCallback can't be used: clone() has a return value.
        const origClone = nodeType.prototype.clone;
        nodeType.prototype.clone = function () {
            const copy = origClone ? origClone.apply(this, arguments) : null;
            if (copy) {
                if (this._batSecStrip) copy._batSecStrip = this._batSecStrip;
                const plate = loadCachedPlate(this);
                if (plate) copy._batSecPendingPlate = plate;
            }
            return copy;
        };

        chainCallback(nodeType.prototype, "onAdded", function () {
            const plate = this._batSecPendingPlate;
            if (!plate) return;
            delete this._batSecPendingPlate;
            // Re-key the cache onto this copy's own id so it survives a
            // reload the same way the original's does. This is the part that
            // actually matters: the editor usually doesn't exist yet at
            // onAdded (setupNode creates it on a 0ms timeout), so the paint
            // below is just a fast path and the copy's own startup pass picks
            // the plate up from the cache we've just written.
            saveCachedPlate(this, plate);
            applyCachedPlate(this, plate);
        });

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            const node = this;

            BaseEditorCanvas.setupNode(this, nodeData, {
                editorClass: BatSecPointsEditor,
                // These three must match what BatPointsEditor's constructor
                // passes to initEditorPreamble/initEditor — it hardcodes them.
                editorKey: "pointsEditor",
                heightKey: "pointsEditorHeight",
                className: "bat-points-editor",
                menuItems: {
                    "Refresh from clip": { action: () => refreshPlate(node, { allowNodePreview: true }) },
                    "Load Image": { action: (ed) => ed.openImageFilePicker() },
                    "Clear Image": { action: (ed) => ed.clearBackgroundImage() },
                },
                hiddenWidgets: PLUMBING_WIDGETS,
                initialSize: [600, 640],
                extraProperties: [
                    ["points", node.constructor.type, "string"],
                    ["neg_points", node.constructor.type, "string"],
                ],
            });

            installPlateSetter(node);

            // Scrubbing the annotation frame re-previews it.
            const idxWidget = node.widgets?.find((w) => w.name === "frame_index_select");
            if (idxWidget) {
                const orig = idxWidget.callback;
                idxWidget.callback = function (...args) {
                    const r = orig?.apply(this, args);
                    refreshPlate(node);
                    return r;
                };
            }

            // Connecting or swapping the clip re-previews too. Debounced: a
            // reroute drag fires onConnectionsChange several times.
            chainCallback(node, "onConnectionsChange", function (type) {
                if (type != null && type !== 1) return;
                clearTimeout(node._batSecConnDebounce);
                node._batSecConnDebounce = setTimeout(() => {
                    // Bounds first: a shorter clip may make the current
                    // frame_index_select illegal, and refreshPlate should
                    // preview the clamped frame, not the stale one.
                    updateFrameBounds(node).finally(() => refreshPlate(node));
                }, 150);
            });

            // "Reset canvas" clears the picks and, for a node whose plate came
            // from the clip rather than from a paste, has no imgData to reload.
            // editor_base carries the live image across when it can; this is
            // the fallback for when it can't (no plate was showing yet).
            node.onEditorReset = () => refreshPlate(node);

            // Show the last plate this machine saw for this node, so reopening
            // a workflow isn't a grey canvas until the first Run.
            setTimeout(async () => {
                if (!node.editor || node.properties?.imgData) return;
                // A strip replayed from this session's last run beats the
                // cached plate — this fires after it, so bail rather than
                // overwrite the live plate with the stale one.
                if (batPreviewWillReplay(node)) { updateFrameBounds(node); return; }
                const cached = loadCachedPlate(node);
                if (!(await applyCachedPlate(node, cached))) refreshPlate(node);
                updateFrameBounds(node);
            }, 120);
        });

        // Ingest the post-run strip.
        chainCallback(nodeType.prototype, "onExecuted", function (message) {
            if (!message) return;
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const frames = message.frames || [];
            const w = Number(one(message.w)) || 0;
            const h = Number(one(message.h)) || 0;
            if (!frames.length || !w || !h) return;

            const node = this;
            Promise.all(frames.map((b64) => loadImage(`data:image/jpeg;base64,${b64}`)))
                .then((images) => {
                    node._batSecStrip = {
                        frames: images,
                        w, h,
                        stride: Math.max(1, Number(one(message.stride)) || 1),
                        frameCount: Number(one(message.frame_count)) || images.length,
                    };

                    // The strip's frame_count is the exact batch length, so
                    // this is the most reliable moment to bound the widget.
                    updateFrameBounds(node);

                    const idx = Math.max(0, Number(
                        node.widgets?.find((wg) => wg.name === "frame_index_select")?.value) || 0);
                    refreshPlate(node);

                    // Cache only the annotation frame, not the whole strip —
                    // 240 base64 JPEGs would blow the localStorage quota.
                    const stride = node._batSecStrip.stride;
                    const slot = Math.min(frames.length - 1, Math.max(0, Math.round(idx / stride)));
                    saveCachedPlate(node, { frame: frames[slot], w, h });
                })
                .catch((e) => console.error("[Bat SeC] could not decode preview strip:", e));
        });
    },
});
