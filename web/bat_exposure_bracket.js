/**
 * Bat_ExposureBracket / Bat_ExposureMerge — dynamic slots that work under both
 * ComfyUI node renderers.
 *
 * Why plain litegraph calls are enough
 * ------------------------------------
 * Nodes 2.0 does not read `node.outputs` directly. `useGraphNodeManager`
 * replaces the property with a getter returning a `shallowReactive` array:
 *
 *     const reactiveOutputs = shallowReactive(node.outputs ?? [])
 *     Object.defineProperty(node, 'outputs', { get: () => reactiveOutputs, ... })
 *
 * and `NodeSlots.vue` renders `v-for="(output, index) in nodeData.outputs"`
 * over exactly that array. `shallowReactive` tracks array mutation, so
 * litegraph's own `addOutput` / `removeOutput` — which push and splice
 * `this.outputs` — drive the Vue renderer with no special handling. Nodes 1.0
 * is litegraph natively. One code path, both renderers.
 *
 * The one thing `shallowReactive` does NOT track is a property change on an
 * element already in the array: `outputs[i].label = 'x'` mutates the raw
 * object and Vue never hears about it. So relabelling assigns a fresh object
 * to the index instead, which IS tracked, spreading the old slot so the
 * `links` array and everything litegraph cares about survives.
 *
 * Slot order is not cosmetic
 * --------------------------
 * The pipe is output 0. ComfyUI maps outputs to RETURN_TYPES by INDEX, and
 * `removeOutput` decrements `origin_slot` on every link after the removed
 * slot — so with the pipe last, dropping the stop count would quietly repoint
 * a live BAT_BRACKET link at an IMAGE. With the pipe first, trimming only ever
 * pops the tail and every surviving index still means what it did.
 *
 * Inputs are keyed by NAME in the prompt, not index, so the merge node's
 * dynamic inputs carry no such constraint and can be trimmed freely.
 *
 * The stop strip
 * --------------
 * The splitter also carries a live preview: one thumbnail per output slot,
 * showing what that stop will actually look like, plus a big view of whichever
 * one is selected.
 *
 * It answers a question the node otherwise cannot. `count`, `spacing` and
 * `direction` describe a bracket in the abstract, but whether the bracket is
 * worth rendering depends entirely on the plate: exposing down a plate already
 * clipped flat at 1.0 recovers nothing, it just makes a darker flat region, and
 * eight LTX generations is an expensive way to find that out. The Python ships a
 * scene-LINEAR tile with its above-white values intact and this file re-derives
 * every stop from it — so a plate with real range shows structure appearing in
 * the highlights as the stops go down, and a clipped one visibly doesn't.
 *
 * The clip overlay makes it unambiguous: red where a stop has hit white, blue
 * where it has hit black, with the percentages next to each thumbnail. The
 * conclusion is usually "spacing is too wide" or "there is nothing up there".
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { hdrSupported, decodeHdrTile, imageDataToSource } from "./bat_hdr_preview.js";
import { batTrack, registerCleanup, batNodeCacheKey, isNodeAlive } from "./bat_lifecycle.js";
import { exposeToSdr, encodeFromLinear, toLinear } from "./bat_transfer.js";

const BRACKET_TYPE = "Bat_ExposureBracket";
const MERGE_TYPE = "Bat_ExposureMerge";
const PIPE_TYPE = "BAT_BRACKET";
const MAX_STOPS = 8;          // must match MAX_STOPS in bat_exposure_bracket.py

// ── mirror of parse_stops / build_stops in the .py ───────────────────────
function parseStops(text) {
    if (!text || !text.trim()) return null;
    const out = [];
    for (const tok of text.replace(/,/g, " ").split(/\s+/)) {
        if (!tok) continue;
        const v = parseFloat(tok);
        if (!isFinite(v)) return null;      // same refusal-to-guess as Python
        out.push(v);
    }
    return out.length ? out.slice(0, MAX_STOPS) : null;
}

function buildStops(count, spacing, direction, custom) {
    const parsed = parseStops(custom);
    if (parsed) return parsed;
    const n = Math.max(1, Math.min(Math.round(count) || 1, MAX_STOPS));
    if (direction === "up") return Array.from({ length: n }, (_, i) => i * spacing);
    if (direction === "symmetric") {
        const start = -(n - 1) / 2;
        return Array.from({ length: n }, (_, i) => (start + i) * spacing);
    }
    return Array.from({ length: n }, (_, i) => -i * spacing);
}

const evLabel = (ev) =>
    `${ev > 0 ? "+" : ""}${(+ev).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")} EV`;

/**
 * Set a slot's display label without breaking Nodes 2.0 reactivity.
 * Assigning a fresh object to the index is tracked by shallowReactive;
 * mutating the existing object is not.
 */
function setSlotLabel(node, arrName, index, label) {
    const arr = node[arrName];
    const slot = arr?.[index];
    if (!slot || slot.label === label) return;
    try {
        arr[index] = { ...slot, label };
    } catch (_) {
        slot.label = label;      // frozen array: at least Nodes 1.0 will show it
    }
}

// ── the splitter ─────────────────────────────────────────────────────────
function syncBracketOutputs(node) {
    if (!node.outputs) return;
    const W = (n) => node.widgets?.find((w) => w.name === n);
    const num = (n, d) => { const v = +(W(n)?.value); return isFinite(v) ? v : d; };
    const str = (n, d) => { const w = W(n); return w == null ? d : String(w.value); };

    const stops = buildStops(num("count", 3), num("spacing", 1.5),
                             str("direction", "down"), str("custom_stops", ""));
    const want = 1 + stops.length;            // slot 0 is the pipe

    // Grow first, then shrink — and only ever from the tail, so a link on a
    // surviving slot keeps both its connection and its index.
    while (node.outputs.length < want) {
        const i = node.outputs.length;        // 1-based ev index == i
        node.addOutput(`ev_${i}`, "IMAGE");
    }
    while (node.outputs.length > want) {
        node.removeOutput(node.outputs.length - 1);
    }

    setSlotLabel(node, "outputs", 0, "bracket");
    stops.forEach((ev, i) => setSlotLabel(node, "outputs", i + 1, evLabel(ev)));

    // Keep the count widget honest when custom_stops is driving the list —
    // otherwise the widget says 3 while five slots are showing.
    const cw = W("count");
    if (cw && parseStops(str("custom_stops", "")) && cw.value !== stops.length) {
        cw.value = stops.length;
    }
    node.setDirtyCanvas?.(true, true);
    // Push the new stop list to any merge node downstream of the pipe.
    try { notifyMergeNodes(node); } catch (e) {
        console.warn("[Bat_ExposureBracket] could not notify merge nodes:", e);
    }
}

/** litegraph has moved links between a plain object and a Map; accept both. */
function getLink(graph, id) {
    if (id == null || !graph) return null;
    return graph.links?.[id] ?? graph._links?.get?.(id) ?? null;
}

/**
 * Walk back from a merge node's `bracket` input to the Bat_ExposureBracket
 * feeding it, and return its resolved stop list.
 *
 * Follows pass-through nodes (Reroute and friends) for a few hops, because a
 * pipe routed round the back of a graph is normal and the merge should still
 * know how many exposures it is being sent. Returns null if the trail goes
 * cold, and the caller falls back to growing from what is wired.
 */
function resolveBracketStops(node) {
    const graph = node.graph ?? app.graph;
    if (!graph) return null;
    let slot = node.inputs?.find((s) => s.name === "bracket");
    for (let hop = 0; hop < 8; hop++) {
        const link = getLink(graph, slot?.link);
        if (!link) return null;
        const src = graph.getNodeById?.(link.origin_id);
        if (!src) return null;
        if (src.type === BRACKET_TYPE || src.comfyClass === BRACKET_TYPE) {
            const W = (n) => src.widgets?.find((w) => w.name === n);
            const num = (n, d) => { const v = +(W(n)?.value); return isFinite(v) ? v : d; };
            const str = (n, d) => { const w = W(n); return w == null ? d : String(w.value); };
            return buildStops(num("count", 3), num("spacing", 1.5),
                              str("direction", "down"), str("custom_stops", ""));
        }
        // A reroute-ish node: one input carrying the same thing straight through.
        if (src.inputs?.length === 1) { slot = src.inputs[0]; continue; }
        return null;
    }
    return null;
}

/** Re-sync every Bat_ExposureMerge downstream of this bracket's pipe output. */
function notifyMergeNodes(node) {
    const graph = node.graph ?? app.graph;
    const links = node.outputs?.[0]?.links;
    if (!graph || !links) return;
    const seen = new Set();
    const visit = (nodeId, depth) => {
        if (depth > 8 || seen.has(nodeId)) return;
        seen.add(nodeId);
        const n = graph.getNodeById?.(nodeId);
        if (!n) return;
        if (n.type === MERGE_TYPE || n.comfyClass === MERGE_TYPE) { syncMergeInputs(n); return; }
        for (const out of n.outputs ?? [])
            for (const lid of out.links ?? []) {
                const l = getLink(graph, lid);
                if (l) visit(l.target_id, depth + 1);
            }
    };
    for (const lid of links) {
        const l = getLink(graph, lid);
        if (l) visit(l.target_id, 0);
    }
}

// ── the merge ────────────────────────────────────────────────────────────
/**
 * Match the hdr_* inputs to the bracket that is actually feeding this node.
 *
 * The pipe's *contents* only exist at execution time, but the bracket NODE is
 * right there in the graph, so its stop list can be read directly from its
 * widgets. That gives exactly as many inputs as there are exposures, each
 * labelled with the EV it expects — change the bracket's count and the merge
 * follows.
 *
 * With no bracket connected (or the trail lost through something unexpected)
 * it degrades to growing one spare slot past whatever is wired, so the node is
 * still usable stand-alone.
 */
function syncMergeInputs(node) {
    if (!node.inputs) return;
    const idxOf = (name) => node.inputs.findIndex((s) => s.name === name);
    let lastConnected = 0;
    for (let i = 1; i <= MAX_STOPS; i++) {
        const s = node.inputs[idxOf(`hdr_${i}`)];
        if (s && s.link != null) lastConnected = i;
    }
    const stops = resolveBracketStops(node);
    // Never shrink past a live link: the artist may drop the bracket's count
    // while a later pass is still wired, and silently cutting their connection
    // is worse than briefly showing one slot too many.
    const want = stops
        ? Math.min(MAX_STOPS, Math.max(stops.length, lastConnected))
        : Math.min(MAX_STOPS, lastConnected + 1);

    let have = 0;
    for (let i = 1; i <= MAX_STOPS; i++) if (idxOf(`hdr_${i}`) !== -1) have = i;

    for (let i = have + 1; i <= want; i++) node.addInput(`hdr_${i}`, "IMAGE");
    for (let i = have; i > want; i--) {
        const at = idxOf(`hdr_${i}`);
        // Never remove a connected slot — the user may have wired a later one
        // and then unwired an earlier one; trimming through it would silently
        // drop their link.
        if (at !== -1 && node.inputs[at].link == null) node.removeInput(at);
    }
    for (let i = 1; i <= MAX_STOPS; i++) {
        const at = idxOf(`hdr_${i}`);
        if (at === -1) continue;
        const ev = stops?.[i - 1];
        // Label with the EV the bracket will send, so a mis-ordered wiring is
        // visible on the node rather than only in the result.
        setSlotLabel(node, "inputs", at,
                     ev == null ? `hdr ${i}` : `hdr ${i}  ${evLabel(ev)}`);
    }
    node.setDirtyCanvas?.(true, true);
}

// ═════════════════════════════════════════════════════════════════════════
// Shared preview plumbing
// ═════════════════════════════════════════════════════════════════════════

const LUMA_R = 0.2126, LUMA_G = 0.7152, LUMA_B = 0.0722;   // mirrors _LUMA
const W_FLOOR = 1e-4;                                      // mirrors _W_FLOOR
const K_MIN = 1 / 64, K_MAX = 64;                          // mirrors _K_MIN/_K_MAX
const MIN_OVERLAP = 0.002;                                 // mirrors _MIN_OVERLAP
const T_EPS = 1e-6;                                        // mirrors _EPS

/** Mirrors `_well_exposed`: Mertens-style, peaks at mid-grey. */
function wellExposed(luma, sigma) {
    const d = luma - 0.5;
    return Math.exp(-(d * d) / (2 * sigma * sigma));
}

/**
 * Decode one plate tile into a scene-linear Float32 buffer (RGB, stride 3).
 *
 * `mode` is applied here rather than by the sender so a `plate_gamma_mode`
 * flip repaints without a re-run — see the note in the Python's
 * `_preview_payload`. Pass mode `null` for a tile that is already linear
 * (which is what the merge node ships).
 */
function tileToLinear(src, mode) {
    if (!src) return null;
    if (!mode) return src.data;
    const n = src.data.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = toLinear(src.data[i], mode);
    return out;
}

/** Decode a base64 JPEG into the SourceBuffer shape the tiles use. */
async function jpegToSource(b64) {
    const img = new Image();
    await new Promise((res, rej) => {
        img.onload = res; img.onerror = rej;
        img.src = `data:image/jpeg;base64,${b64}`;
    });
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(img, 0, 0);
    return imageDataToSource(cx.getImageData(0, 0, c.width, c.height));
}

/** A tile if the browser can inflate one, else the JPEG, else null. */
async function decodeEither(tile, jpeg, tag) {
    if (tile && hdrSupported()) {
        try { return { src: await decodeHdrTile(tile), hdr: true }; }
        catch (e) { console.warn(`[${tag}] tile decode failed:`, e); }
    }
    if (jpeg) {
        try { return { src: await jpegToSource(jpeg), hdr: false }; }
        catch (e) { console.warn(`[${tag}] jpeg decode failed:`, e); }
    }
    return null;
}

/**
 * The viewer-exposure strip shared by both previews.
 *
 * Both nodes deal in unbounded linear data, and a monitor has nothing above
 * white, so a viewer exposure is not a luxury here — at 0 EV the canvas shows
 * the [0,1] slice and nothing else, and "did the highlights merge sensibly" is
 * a question about what is above it. Display only; never reaches the render.
 */
function buildExposureStrip(storageKey, onChange, extra) {
    const el = document.createElement("div");
    el.style.cssText = `display:flex; align-items:center; gap:5px; padding:4px 6px;
        background:#141414; border-top:1px solid #2a2a2a; font:11px monospace;
        color:#9aa; flex:0 0 auto; user-select:none; flex-wrap:wrap;`;

    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}") || {}; } catch (_) {}
    const state = { exposure: Number.isFinite(saved.exposure) ? saved.exposure : 0 };

    if (extra) el.appendChild(extra);

    const label = document.createElement("span");
    label.textContent = "view exp";
    label.style.cssText = "opacity:0.7; margin-left:2px;";
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "-10"; slider.max = "10"; slider.step = "0.25";
    slider.value = String(state.exposure);
    slider.style.cssText = "flex:1 1 60px; min-width:54px; accent-color:#6af;";
    slider.title = "Viewer exposure in stops — display only, never baked into the render.\n"
        + "Expose DOWN to see what sits above white; UP to check the shadows.";
    const readout = document.createElement("span");
    readout.style.cssText = "min-width:44px; text-align:right; color:#cde;";
    const reset = document.createElement("button");
    reset.textContent = "0"; reset.title = "Reset viewer exposure";
    reset.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
        border-radius:3px; padding:2px 6px; font:11px monospace; cursor:pointer;`;

    el.append(label, slider, readout, reset);

    function refresh() {
        const sign = state.exposure > 0 ? "+" : "";
        readout.textContent = `${sign}${state.exposure.toFixed(2)}`;
        readout.style.color = state.exposure === 0 ? "#788" : "#cde";
    }
    function commit() {
        try { localStorage.setItem(storageKey, JSON.stringify({ exposure: state.exposure })); }
        catch (_) {}
        refresh(); onChange?.();
    }
    slider.addEventListener("input", () => { state.exposure = parseFloat(slider.value) || 0; commit(); });
    reset.addEventListener("click", () => { state.exposure = 0; slider.value = "0"; commit(); });
    for (const e of [slider, reset]) e.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    refresh();

    return {
        el,
        // 0 EV must be exactly 1.0, not Math.pow(2, 0) rounding noise.
        get gain() { return state.exposure === 0 ? 1 : Math.pow(2, state.exposure); },
        get exposure() { return state.exposure; },
    };
}

/** A small toggle button in the house style. */
function toggleButton(text, title, onClick) {
    const b = document.createElement("button");
    b.textContent = text; b.title = title;
    b.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
        border-radius:3px; padding:2px 7px; font:11px monospace; cursor:pointer;`;
    b.addEventListener("click", onClick);
    b.addEventListener("pointerdown", (e) => e.stopPropagation());
    b.setActive = (on) => {
        b.style.background = on ? "#2b4a63" : "#222";
        b.style.color = on ? "#cfe8ff" : "#9aa";
        b.style.borderColor = on ? "#4a7fa8" : "#333";
    };
    b.setActive(false);
    return b;
}

// ═════════════════════════════════════════════════════════════════════════
// Bat_ExposureBracket — the stop strip
// ═════════════════════════════════════════════════════════════════════════

/**
 * Render one exposure of the linear plate into an RGBA buffer, and count how
 * much of it hit each clip point.
 *
 * The clip counts are the strip's actual payload. A thumbnail tells you a stop
 * looks darker; the percentages tell you whether that stop has anything in the
 * highlights for LTX to work with, which is the decision the artist is making.
 * Counted on the pre-encode linear value against the exposure, so they mean
 * "clipped by this exposure" rather than "happens to land on 255 after an
 * 8-bit round".
 */
function renderStop(lin, w, h, ev, mode, viewGain, showClip, dst) {
    const scale = Math.pow(2, ev);
    const n = w * h;
    let clipHi = 0, clipLo = 0;
    for (let i = 0, p = 0, q = 0; i < n; i++, p += 3, q += 4) {
        const r = lin[p] * scale, g = lin[p + 1] * scale, b = lin[p + 2] * scale;
        // Any channel at or past the clip point counts the pixel — a blown red
        // is a blown pixel even if green still has room.
        const hi = r >= 1 || g >= 1 || b >= 1;
        const lo = r <= 0 && g <= 0 && b <= 0;
        if (hi) clipHi++;
        if (lo) clipLo++;

        if (showClip && hi) { dst[q] = 255; dst[q + 1] = 40; dst[q + 2] = 40; dst[q + 3] = 255; continue; }
        if (showClip && lo) { dst[q] = 40; dst[q + 1] = 90; dst[q + 2] = 255; dst[q + 3] = 255; continue; }

        for (let c = 0; c < 3; c++) {
            let v = exposeToSdr(lin[p + c], scale, mode);
            if (viewGain !== 1) v *= viewGain;
            dst[q + c] = v > 1 ? 255 : (v < 0 ? 0 : Math.round(v * 255));
        }
        dst[q + 3] = 255;
    }
    return { clipHi: clipHi / n, clipLo: clipLo / n };
}

function buildBracketPreview(node) {
    const track = batTrack(node);

    const root = document.createElement("div");
    root.style.cssText = `position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; overflow:hidden;`;

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative; flex:1 1 auto; min-height:0; display:flex; background:#000;";
    const big = document.createElement("canvas");
    big.style.cssText = "width:100%; height:100%; object-fit:contain; display:block;";
    stage.appendChild(big);

    const hint = document.createElement("div");
    hint.style.cssText = `position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;`;
    hint.textContent = "Run once to populate the preview.";
    stage.appendChild(hint);

    const badge = document.createElement("div");
    badge.style.cssText = `position:absolute; right:6px; top:4px; font:10px monospace;
        color:#cde; background:rgba(0,0,0,0.62); padding:2px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; text-align:right; line-height:1.45;`;
    badge.style.display = "none";
    stage.appendChild(badge);

    // Shown only when the high-precision tile is unavailable, because in that
    // case every stop below 0 EV is derived from clamped data and the strip is
    // structurally unable to answer the question it exists for.
    const warn = document.createElement("div");
    warn.style.cssText = `position:absolute; left:6px; top:4px; font:10px monospace;
        color:#ffcf8a; background:rgba(60,30,0,0.75); padding:2px 6px; border-radius:3px;
        pointer-events:none; display:none; max-width:70%;`;
    warn.textContent = "8-bit fallback — clipped source, recovery not shown";
    stage.appendChild(warn);

    root.appendChild(stage);

    // The strip of per-stop thumbnails. Wraps rather than scrolls: eight
    // thumbnails at a readable size fit two rows on any node wide enough to be
    // worth putting this editor on, and a horizontal scrollbar inside a node
    // body is a worse thing to hunt for than a second row.
    const strip = document.createElement("div");
    strip.style.cssText = `display:flex; flex-wrap:wrap; gap:4px; padding:5px 6px;
        background:#0e0e0e; border-top:1px solid #2a2a2a; flex:0 0 auto;
        align-content:flex-start; overflow-y:auto; max-height:46%;`;
    root.appendChild(strip);

    const state = {
        src: null,          // SourceBuffer of the plate as received
        lin: null,          // ...decoded to scene-linear with the current mode
        linMode: null,      // which mode `lin` was decoded with
        hdr: false,         // came from the 16-bit tile rather than the JPEG
        meta: null,
        selected: 0,        // which stop the big view shows
        needsRun: false,    // preview_frame moved; only Python can act on it
        showClip: false,
        cells: [],          // [{wrap, canvas, ctx, label, ev}]
    };
    node._batBracketState = state;

    const W = (n) => node.widgets?.find((w) => w.name === n);
    const num = (n, d) => { const v = +(W(n)?.value); return isFinite(v) ? v : d; };
    const str = (n, d) => { const w = W(n); return w == null ? d : String(w.value); };

    const clipBtn = toggleButton("clip", "Flag clipped pixels: red where a stop has hit "
        + "white, blue where it has hit black.\nThe fastest way to see whether a stop is "
        + "doing anything on this plate.", () => {
            state.showClip = !state.showClip;
            clipBtn.setActive(state.showClip);
            schedule();
        });

    const bar = buildExposureStrip(
        batNodeCacheKey(app, "bat_bracket_view", node), () => schedule(), clipBtn);
    root.appendChild(bar.el);

    /** Decode to linear on demand, caching against the current gamma mode. */
    function linearFor(mode) {
        if (!state.src) return null;
        if (state.lin && state.linMode === mode) return state.lin;
        state.lin = tileToLinear(state.src, mode);
        state.linMode = mode;
        return state.lin;
    }

    function ensureCells(count) {
        while (state.cells.length < count) {
            const i = state.cells.length;
            const wrap = document.createElement("div");
            wrap.style.cssText = `position:relative; flex:0 0 auto; cursor:pointer;
                border:1px solid #2a2a2a; border-radius:3px; overflow:hidden;
                background:#000; line-height:0;`;
            const canvas = document.createElement("canvas");
            canvas.style.cssText = "display:block; width:104px; height:auto;";
            const label = document.createElement("div");
            label.style.cssText = `position:absolute; left:0; right:0; bottom:0;
                font:9px monospace; color:#cde; background:rgba(0,0,0,0.66);
                padding:1px 3px; white-space:pre; line-height:1.3; text-align:center;`;
            wrap.append(canvas, label);
            wrap.addEventListener("pointerdown", (e) => {
                e.stopPropagation();
                state.selected = i;
                schedule();
            });
            strip.appendChild(wrap);
            state.cells.push({
                wrap, canvas, label,
                ctx: canvas.getContext("2d", { willReadFrequently: true }),
            });
        }
        // Hide rather than destroy — the artist walks `count` up and down while
        // dialling a bracket in, and rebuilding the DOM each time costs a
        // visible flicker for nothing.
        state.cells.forEach((c, i) => {
            c.wrap.style.display = i < count ? "block" : "none";
        });
    }

    function render() {
        const mode = str("plate_gamma_mode", state.meta?.gamma_mode || "srgb");
        const lin = linearFor(mode);
        if (!lin || !state.src) return;
        const w = state.src.width, h = state.src.height;

        const stops = buildStops(num("count", 3), num("spacing", 1.5),
                                 str("direction", "down"), str("custom_stops", ""));
        ensureCells(stops.length);
        if (state.selected >= stops.length) state.selected = stops.length - 1;

        const gain = bar.gain;
        const clips = [];
        for (let i = 0; i < stops.length; i++) {
            const cell = state.cells[i];
            if (cell.canvas.width !== w || cell.canvas.height !== h) {
                cell.canvas.width = w; cell.canvas.height = h;
            }
            const img = cell.ctx.createImageData(w, h);
            const c = renderStop(lin, w, h, stops[i], mode, gain, state.showClip, img.data);
            cell.ctx.putImageData(img, 0, 0);
            clips.push(c);
            const pct = (v) => (v <= 0 ? "0" : (v < 0.001 ? "<0.1" : (v * 100).toFixed(1)));
            cell.label.textContent = `${evLabel(stops[i])}\n▲${pct(c.clipHi)}% ▼${pct(c.clipLo)}%`;
            const on = i === state.selected;
            cell.wrap.style.borderColor = on ? "#4a7fa8" : "#2a2a2a";
            cell.wrap.style.boxShadow = on ? "inset 0 0 0 1px #4a7fa8" : "none";
        }

        // Big view of the selected stop.
        const sel = Math.max(0, Math.min(state.selected, stops.length - 1));
        if (big.width !== w || big.height !== h) { big.width = w; big.height = h; }
        const bctx = big.getContext("2d", { willReadFrequently: true });
        const bimg = bctx.createImageData(w, h);
        renderStop(lin, w, h, stops[sel], mode, gain, state.showClip, bimg.data);
        bctx.putImageData(bimg, 0, 0);

        const c = clips[sel];
        badge.textContent =
            `${evLabel(stops[sel])}   stop ${sel + 1}/${stops.length}\n`
            + `clipped  ▲ ${(c.clipHi * 100).toFixed(2)}%   ▼ ${(c.clipLo * 100).toFixed(2)}%`
            + (state.meta?.frames > 1 ? `\nframe ${state.meta.frame}/${state.meta.frames - 1}` : "")
            + (state.needsRun ? "\n⟳ Run to apply preview_frame" : "");
        badge.style.display = "block";
        warn.style.display = state.hdr ? "none" : "block";
        hint.style.display = "none";
    }

    let raf = 0;
    function schedule() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            if (!isNodeAlive(node)) return;
            try { render(); }
            catch (e) { console.error("[Bat_ExposureBracket] preview failed:", e); }
        });
    }
    node._batBracketRepaint = schedule;
    node._batBracketNeedsRun = () => { state.needsRun = true; schedule(); };

    // ── ingest ───────────────────────────────────────────────────────────
    const cacheKey = batNodeCacheKey(app, "bat_bracket_preview", node);

    node._batBracketIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        const jpeg = one(msg.plate_jpeg);
        const got = await decodeEither(one(msg.plate_tile), jpeg, "Bat_ExposureBracket");
        if (!got || !isNodeAlive(node)) return;

        state.src = got.src;
        state.hdr = got.hdr;
        state.needsRun = false;
        state.lin = null; state.linMode = null;
        state.meta = {
            gamma_mode: one(msg.gamma_mode) || "srgb",
            frames: Number(one(msg.frames)) || 1,
            frame: Number(one(msg.preview_frame)) || 0,
            w: Number(one(msg.w)) || 0,
            h: Number(one(msg.h)) || 0,
        };

        // JPEG only — the 16-bit tile is a few hundred KB against a ~5MB
        // origin-wide localStorage budget shared with every other BAT node's
        // cache. Reopening shows the fallback until the next run.
        if (jpeg) {
            try {
                localStorage.setItem(cacheKey, JSON.stringify({ jpeg, meta: state.meta }));
            } catch (_) {}
        }
        schedule();
    };

    node._batBracketRestore = () => {
        let c = null;
        try { c = JSON.parse(localStorage.getItem(cacheKey) || "null"); } catch (_) {}
        if (!c?.jpeg) return;
        node._batBracketIngest({
            plate_jpeg: [c.jpeg],
            gamma_mode: [c.meta?.gamma_mode || "srgb"],
            frames: [c.meta?.frames || 1], preview_frame: [c.meta?.frame || 0],
            w: [c.meta?.w || 0], h: [c.meta?.h || 0],
        });
    };

    track.dispose(() => {
        state.src = null; state.lin = null; state.cells = [];
    });

    return root;
}

// ═════════════════════════════════════════════════════════════════════════
// Bat_ExposureMerge — the merge preview
// ═════════════════════════════════════════════════════════════════════════

/**
 * Per-pass hues for the weight view, walked around the wheel so adjacent
 * stops are never adjacent colours — with eight passes an ordered hue ramp
 * makes neighbours indistinguishable, which is exactly the comparison the
 * view exists for.
 */
const PASS_HUES = [
    [0.20, 0.55, 1.00],  // blue
    [1.00, 0.62, 0.14],  // orange
    [0.36, 0.85, 0.36],  // green
    [0.95, 0.34, 0.72],  // magenta
    [0.98, 0.86, 0.24],  // yellow
    [0.30, 0.86, 0.90],  // cyan
    [0.72, 0.48, 0.98],  // violet
    [0.92, 0.40, 0.32],  // red
];

/**
 * Mirror of `_alignment` — one scalar per pass, bringing them onto the
 * reference's scale.
 *
 * The `auto` branch measures over the tile, not the frame, so it is an
 * estimate; the caller labels it as one. Everything else, including the
 * fallbacks to the nominal ratio and the K_MIN/K_MAX sanity range, matches the
 * Python exactly — those fallbacks are what fire when a bracket barely
 * overlaps, and a preview that quietly used a different rule there would
 * mislead precisely when the artist most needs the truth.
 */
function computeAlignment(passes, plateLin, w, h, mode, align, reference, sigma) {
    const n = passes.length;
    if (align === "off") return { scales: new Array(n).fill(1), measured: false };

    const nominal = passes.map((p) => Math.pow(2, -p.ev));
    const refIdx = reference === "first"
        ? 0
        : passes.reduce((best, p, i) => (Math.abs(p.ev) < Math.abs(passes[best].ev) ? i : best), 0);

    if (align === "nominal") {
        return { scales: nominal.map((k) => k / nominal[refIdx]), measured: false, refIdx };
    }
    if (n === 1) return { scales: [1], measured: false, refIdx };

    const total = w * h;
    const wRef = passWeights(plateLin, w, h, passes[refIdx].ev, mode, sigma);
    const scales = [];
    let anyMeasured = false;
    for (let i = 0; i < n; i++) {
        if (i === refIdx) { scales.push(1); continue; }
        const wI = passWeights(plateLin, w, h, passes[i].ev, mode, sigma);
        let ovSum = 0, den = 0, num = 0;
        const refData = passes[refIdx].lin, iData = passes[i].lin;
        for (let px = 0, p = 0; px < total; px++, p += 3) {
            const ov = wRef[px] * wI[px];
            if (ov === 0) continue;
            ovSum += ov;
            den += ov * (iData[p] * LUMA_R + iData[p + 1] * LUMA_G + iData[p + 2] * LUMA_B);
            num += ov * (refData[p] * LUMA_R + refData[p + 1] * LUMA_G + refData[p + 2] * LUMA_B);
        }
        const frac = ovSum / Math.max(total, 1);
        let k;
        if (frac < MIN_OVERLAP || den <= T_EPS || num <= 0) {
            k = nominal[i] / nominal[refIdx];
        } else {
            k = num / den;
            if (!isFinite(k) || k < K_MIN || k > K_MAX) k = nominal[i] / nominal[refIdx];
            else anyMeasured = true;
        }
        scales.push(k);
    }
    return { scales, measured: anyMeasured, refIdx };
}

/** Mirror of `_well_exposed(_luminance(_expose_to_sdr(...)))` for one pass. */
function passWeights(plateLin, w, h, ev, mode, sigma) {
    const scale = Math.pow(2, ev);
    const n = w * h;
    const out = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 3) {
        const r = exposeToSdr(plateLin[p], scale, mode);
        const g = exposeToSdr(plateLin[p + 1], scale, mode);
        const b = exposeToSdr(plateLin[p + 2], scale, mode);
        out[i] = wellExposed(r * LUMA_R + g * LUMA_G + b * LUMA_B, sigma);
    }
    return out;
}

/**
 * Mirror of `_weighted_merge`: sum(w_i * k_i * pass_i) / sum(w_i).
 * Returns the merged linear buffer plus the per-pass weights, which the
 * weight view and the probe both read.
 */
function mergeTiles(passes, plateLin, w, h, mode, scales, sigma) {
    const n = w * h;
    const out = new Float32Array(n * 3);
    const wsum = new Float32Array(n);
    const weights = passes.map((p) => passWeights(plateLin, w, h, p.ev, mode, sigma));

    for (let i = 0; i < passes.length; i++) {
        const wt = weights[i], k = scales[i], data = passes[i].lin;
        for (let px = 0, p = 0; px < n; px++, p += 3) {
            const a = wt[px] * k;
            out[p] += data[p] * a;
            out[p + 1] += data[p + 1] * a;
            out[p + 2] += data[p + 2] * a;
            wsum[px] += wt[px];
        }
    }
    for (let px = 0, p = 0; px < n; px++, p += 3) {
        const d = wsum[px] < W_FLOOR ? W_FLOOR : wsum[px];
        out[p] = Math.max(out[p] / d, 0);
        out[p + 1] = Math.max(out[p + 1] / d, 0);
        out[p + 2] = Math.max(out[p + 2] / d, 0);
    }
    return { out, weights, wsum };
}

function buildMergePreview(node) {
    const track = batTrack(node);

    const root = document.createElement("div");
    root.style.cssText = `position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; overflow:hidden;`;

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative; flex:1 1 auto; min-height:0; display:flex; background:#000;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; object-fit:contain; display:block;";
    stage.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;`;
    hint.textContent = "Run once to populate the preview.";
    stage.appendChild(hint);

    const badge = document.createElement("div");
    badge.style.cssText = `position:absolute; right:6px; top:4px; font:10px monospace;
        color:#cde; background:rgba(0,0,0,0.62); padding:3px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; text-align:right; line-height:1.45;
        display:none;`;
    stage.appendChild(badge);

    const probe = document.createElement("div");
    probe.style.cssText = `position:absolute; left:6px; top:4px; font:10px monospace;
        color:#cde; background:rgba(0,0,0,0.62); padding:3px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; display:none; line-height:1.45;`;
    stage.appendChild(probe);

    root.appendChild(stage);

    // Legend for the weight view: which colour is which stop. Only shown in
    // that view, because a colour key against a normal image is just clutter.
    const legend = document.createElement("div");
    legend.style.cssText = `display:none; flex-wrap:wrap; gap:6px; padding:4px 6px;
        background:#0e0e0e; border-top:1px solid #2a2a2a; font:10px monospace;
        color:#9aa; flex:0 0 auto;`;
    root.appendChild(legend);

    const viewRow = document.createElement("div");
    viewRow.style.cssText = `display:flex; flex-wrap:wrap; gap:4px; padding:4px 6px;
        background:#111; border-top:1px solid #2a2a2a; flex:0 0 auto; align-items:center;`;
    root.appendChild(viewRow);

    const state = {
        plateLin: null,     // Float32 RGB, scene-linear (already linear on the wire)
        passes: [],         // [{ev, lin}]
        w: 0, h: 0,
        meta: null,
        shipped: null,      // {scales, align, reference, sigma} from the last run
        view: "merged",     // "merged" | "weights" | "pass:<i>"
        merged: null, weights: null, wsum: null, scales: null, measured: false,
    };
    node._batMergeState = state;

    const W = (n) => node.widgets?.find((w) => w.name === n);
    const num = (n, d) => { const v = +(W(n)?.value); return isFinite(v) ? v : d; };
    const str = (n, d) => { const w = W(n); return w == null ? d : String(w.value); };

    const bar = buildExposureStrip(
        batNodeCacheKey(app, "bat_merge_view", node), () => schedule(), null);
    root.appendChild(bar.el);

    let viewButtons = [];
    function buildViewButtons() {
        viewRow.textContent = "";
        legend.textContent = "";
        viewButtons = [];
        const add = (id, text, title) => {
            const b = toggleButton(text, title, () => { state.view = id; schedule(); });
            viewRow.appendChild(b);
            viewButtons.push([id, b]);
        };
        add("merged", "Merged", "The merged HDR. Use the view exposure to look above white — "
            + "at 0 EV you are only seeing the [0,1] slice of an unbounded result.");
        add("weights", "Weights", "False colour: each pass gets a colour and every pixel is "
            + "tinted by the mix of passes contributing to it.\nA region in one flat colour "
            + "comes from one pass alone — increase well_exposed_sigma if you wanted "
            + "averaging there. Brightness is the summed weight; near-black means every "
            + "pass was badly exposed and the merge is running on the floor.");
        state.passes.forEach((p, i) => {
            add(`pass:${i}`, `${i + 1}`,
                `Solo the ${evLabel(p.ev)} pass, scaled by its alignment factor.\n`
                + "If one pass looks wildly brighter or darker than the others here, "
                + "alignment is the problem, not the weighting.");
            const chip = document.createElement("span");
            const hue = PASS_HUES[i % PASS_HUES.length];
            const css = `rgb(${hue.map((c) => Math.round(c * 255)).join(",")})`;
            chip.style.cssText = "display:inline-flex; align-items:center; gap:3px;";
            const sw = document.createElement("span");
            sw.style.cssText = `width:9px; height:9px; border-radius:2px; background:${css};
                display:inline-block;`;
            const tx = document.createElement("span");
            tx.textContent = evLabel(p.ev);
            chip.append(sw, tx);
            legend.appendChild(chip);
        });
    }

    function refreshButtons() {
        for (const [id, b] of viewButtons) b.setActive(state.view === id);
        legend.style.display = state.view === "weights" ? "flex" : "none";
    }

    function render() {
        if (!state.plateLin || !state.passes.length) return;
        const w = state.w, h = state.h, n = w * h;
        const mode = state.meta?.gamma_mode || "srgb";
        const sigma = num("well_exposed_sigma", 0.2);
        const align = str("align", "auto");
        const reference = str("reference", "auto");

        // Trust the render's own alignment scales while the widgets still hold
        // the values they were computed from. They were measured over the whole
        // frame; anything computed here is measured over a tile 1/300th the
        // size, and k is exactly the number a mis-levelled pass turns on.
        const sh = state.shipped;
        const shippedValid = !!sh
            && sh.scales.length === state.passes.length
            && sh.align === align && sh.reference === reference
            && Math.abs(sh.sigma - sigma) < 1e-9;

        let scales, estimated;
        if (shippedValid) {
            scales = sh.scales;
            estimated = false;
        } else {
            const a = computeAlignment(state.passes, state.plateLin, w, h, mode,
                                      align, reference, sigma);
            scales = a.scales;
            estimated = align === "auto" && state.passes.length > 1;
        }

        const { out, weights, wsum } = mergeTiles(
            state.passes, state.plateLin, w, h, mode, scales, sigma);
        state.merged = out; state.weights = weights; state.wsum = wsum;
        state.scales = scales; state.measured = !estimated;

        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const img = ctx.createImageData(w, h);
        const dst = img.data;
        const gain = bar.gain;

        if (state.view === "weights") {
            for (let px = 0, q = 0; px < n; px++, q += 4) {
                let r = 0, g = 0, b = 0;
                const tot = wsum[px] < W_FLOOR ? W_FLOOR : wsum[px];
                for (let i = 0; i < state.passes.length; i++) {
                    const f = weights[i][px] / tot;
                    const hue = PASS_HUES[i % PASS_HUES.length];
                    r += hue[0] * f; g += hue[1] * f; b += hue[2] * f;
                }
                // Modulate by the SUMMED weight, so "nothing was well exposed
                // here" reads as dark rather than as a confident colour. The
                // view exposure applies, which is how you check a dark region
                // is genuinely starved rather than just dim.
                const v = Math.min(wsum[px], 1) * gain;
                dst[q] = Math.min(255, Math.round(r * v * 255));
                dst[q + 1] = Math.min(255, Math.round(g * v * 255));
                dst[q + 2] = Math.min(255, Math.round(b * v * 255));
                dst[q + 3] = 255;
            }
        } else {
            const solo = state.view.startsWith("pass:") ? parseInt(state.view.slice(5), 10) : -1;
            const src = solo >= 0 && state.passes[solo] ? state.passes[solo].lin : out;
            const k = solo >= 0 ? (scales[solo] ?? 1) : 1;
            for (let px = 0, p = 0, q = 0; px < n; px++, p += 3, q += 4) {
                for (let c = 0; c < 3; c++) {
                    // Display transform: the data is scene-linear and unbounded,
                    // so it goes through the plate's own encoding to be looked
                    // at. Same curve the bracket used on the way in.
                    const lin = src[p + c] * k * gain;
                    const v = encodeFromLinear(lin, mode);
                    dst[q + c] = v > 1 ? 255 : (v < 0 ? 0 : Math.round(v * 255));
                }
                dst[q + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);

        const kTxt = scales.map((k, i) =>
            `${evLabel(state.passes[i].ev)}  x${k.toFixed(3)}`).join("\n");
        badge.textContent =
            `${state.passes.length} pass${state.passes.length === 1 ? "" : "es"}`
            + `   sigma ${sigma.toFixed(2)}\n${kTxt}`
            + (estimated ? "\n(k estimated from the tile)" : "");
        badge.style.display = "block";
        hint.style.display = "none";
        refreshButtons();
    }

    let raf = 0;
    function schedule() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            if (!isNodeAlive(node)) return;
            try { render(); }
            catch (e) { console.error("[Bat_ExposureMerge] preview failed:", e); }
        });
    }
    node._batMergeRepaint = schedule;

    // Value probe — reads the merged linear value and each pass's weight, which
    // together are the whole diagnosis for "why is this region wrong".
    canvas.addEventListener("pointermove", (e) => {
        if (!state.merged || !state.w) return;
        const r = canvas.getBoundingClientRect();
        const sc = Math.min(r.width / state.w, r.height / state.h);
        const cw = state.w * sc, ch = state.h * sc;
        const x = Math.floor((e.clientX - r.left - (r.width - cw) / 2) / sc);
        const y = Math.floor((e.clientY - r.top - (r.height - ch) / 2) / sc);
        if (x < 0 || y < 0 || x >= state.w || y >= state.h) { probe.style.display = "none"; return; }
        const px = y * state.w + x, p = px * 3;
        const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));
        const lines = [`merged ${f(state.merged[p])} ${f(state.merged[p + 1])} ${f(state.merged[p + 2])}`];
        const tot = Math.max(state.wsum[px], W_FLOOR);
        state.passes.forEach((pa, i) => {
            const wt = state.weights[i][px];
            lines.push(`${evLabel(pa.ev).padStart(8)}  w ${wt.toFixed(3)}`
                + `  ${(100 * wt / tot).toFixed(0).padStart(3)}%`);
        });
        probe.textContent = lines.join("\n");
        probe.style.display = "block";
    });
    canvas.addEventListener("pointerleave", () => { probe.style.display = "none"; });
    canvas.addEventListener("pointerdown", (e) => e.stopPropagation());

    // ── ingest ───────────────────────────────────────────────────────────
    node._batMergeIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        const plateTile = one(msg.plate_lin_tile);
        const passTiles = one(msg.pass_tiles);
        if (!plateTile || !passTiles || !hdrSupported()) {
            // No JPEG fallback here, and deliberately: the merge operates on
            // unbounded linear passes, and an 8-bit clamped stand-in would
            // produce a picture that looks like a merge but answers none of
            // the questions this preview is for.
            hint.textContent = hdrSupported()
                ? "Preview tiles unavailable — see the log."
                : "This browser has no DecompressionStream, so the preview tiles can't be inflated.";
            hint.style.display = "block";
            return;
        }

        let plate, passes;
        try {
            plate = await decodeHdrTile(plateTile);
            const evs = one(msg.pass_evs) || [];
            passes = [];
            for (let i = 0; i < passTiles.length; i++) {
                const t = await decodeHdrTile(passTiles[i]);
                passes.push({ ev: Number(evs[i]) || 0, lin: t.data, width: t.width, height: t.height });
            }
        } catch (e) {
            console.warn("[Bat_ExposureMerge] tile decode failed:", e);
            return;
        }
        if (!isNodeAlive(node)) return;

        // Every tile comes off the same sampler at the same long edge, so a
        // disagreement means something upstream changed shape mid-payload;
        // bail rather than index off the end of the shortest buffer.
        const bad = passes.find((p) => p.width !== plate.width || p.height !== plate.height);
        if (bad) {
            console.warn("[Bat_ExposureMerge] preview tiles disagree on size; skipping.");
            return;
        }

        state.plateLin = plate.data;
        state.w = plate.width; state.h = plate.height;
        state.passes = passes;
        state.meta = {
            gamma_mode: one(msg.gamma_mode) || "srgb",
            frames: Number(one(msg.frames)) || 1,
        };
        const scales = one(msg.scales);
        state.shipped = Array.isArray(scales) ? {
            scales: scales.map(Number),
            align: one(msg.k_align) || "auto",
            reference: one(msg.k_reference) || "auto",
            sigma: Number(one(msg.k_sigma)),
        } : null;

        if (state.view.startsWith("pass:")
            && parseInt(state.view.slice(5), 10) >= passes.length) {
            state.view = "merged";
        }
        buildViewButtons();
        schedule();
    };

    track.dispose(() => {
        state.plateLin = null; state.passes = []; state.merged = null;
        state.weights = null; state.wsum = null;
    });

    buildViewButtons();
    refreshButtons();
    return root;
}

app.registerExtension({
    name: "Bat_ExposureBracket",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === BRACKET_TYPE) {
            registerCleanup(nodeType);

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated?.apply(this, arguments);
                const self = this;   // widget callbacks are invoked with the
                                     // WIDGET as `this`, not the node.
                // plate_gamma_mode repaints but does not re-slot: it changes how
                // the plate decodes, not how many stops there are. The other
                // four do both.
                for (const name of ["count", "spacing", "direction", "custom_stops",
                                    "plate_gamma_mode"]) {
                    const w = this.widgets?.find((x) => x.name === name);
                    if (!w) continue;
                    const orig = w.callback;
                    w.callback = function (v) {
                        try { orig?.call(this, v); } catch (_) {}
                        if (name !== "plate_gamma_mode") syncBracketOutputs(self);
                        self._batBracketRepaint?.();
                    };
                }
                // preview_frame is the one widget this file cannot act on: which
                // frame is shipped is Python's decision, made at execution time.
                // Watched anyway so the strip can SAY a Run is needed — silently
                // doing nothing is what makes a working control feel broken, and
                // it is exactly how the same widget on Bat_AdvancedBlend came to
                // be reported as not working.
                {
                    const w = this.widgets?.find((x) => x.name === "preview_frame");
                    if (w) {
                        const orig = w.callback;
                        w.callback = function (v) {
                            try { orig?.call(this, v); } catch (_) {}
                            self._batBracketNeedsRun?.();
                        };
                    }
                }

                const el = buildBracketPreview(this);
                // Dual-mode sizing — Nodes 2.0 derives node height from
                // computeLayoutSize, so a bare `this.size` is ignored there.
                addBatDOMWidget(this, "bat_bracket_preview", "bat_bracket_preview", el, {
                    minWidth: 380, height: 440, growable: true,
                });
                clampNodeSize(this, 380, 440);

                // Defer: the widgets exist but the node is not in the graph yet,
                // and addOutput before that point has nowhere to register links.
                setTimeout(() => syncBracketOutputs(this), 0);
                // Same reason the other editors defer their cache restore —
                // node.id is only final once LiteGraph finishes construction,
                // and it is part of the cache key.
                setTimeout(() => this._batBracketRestore?.(), 0);
                return r;
            };

            // A saved workflow restores its own slot list; re-derive once
            // configure() has run so the two cannot disagree.
            const onConfigured = nodeType.prototype.onAfterGraphConfigured;
            nodeType.prototype.onAfterGraphConfigured = function () {
                const r = onConfigured?.apply(this, arguments);
                syncBracketOutputs(this);
                this._batBracketRepaint?.();
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted?.apply(this, arguments);
                if (message && this._batBracketIngest) this._batBracketIngest(message);
                return r;
            };
        }

        if (nodeData.name === MERGE_TYPE) {
            registerCleanup(nodeType);

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onCreated?.apply(this, arguments);
                const self = this;
                // These three are the whole merge: what each pass is scaled by
                // and how widely the weighting spreads. All three repaint from
                // the tiles with no re-run.
                for (const name of ["align", "well_exposed_sigma", "reference"]) {
                    const w = this.widgets?.find((x) => x.name === name);
                    if (!w) continue;
                    const orig = w.callback;
                    w.callback = function (v) {
                        try { orig?.call(this, v); } catch (_) {}
                        self._batMergeRepaint?.();
                    };
                }

                const el = buildMergePreview(this);
                addBatDOMWidget(this, "bat_merge_preview", "bat_merge_preview", el, {
                    minWidth: 380, height: 420, growable: true,
                });
                clampNodeSize(this, 380, 420);

                setTimeout(() => syncMergeInputs(this), 0);
                return r;
            };
            const onConn = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const r = onConn?.apply(this, arguments);
                // After litegraph has finished updating the link, not during.
                setTimeout(() => syncMergeInputs(this), 0);
                return r;
            };
            const onConfigured = nodeType.prototype.onAfterGraphConfigured;
            nodeType.prototype.onAfterGraphConfigured = function () {
                const r = onConfigured?.apply(this, arguments);
                syncMergeInputs(this);
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted?.apply(this, arguments);
                if (message && this._batMergeIngest) this._batMergeIngest(message);
                return r;
            };
        }
    },
});
