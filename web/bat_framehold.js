/**
 * BAT — Bat_Framehold front-end: an arrow-key scrubber for picking hold frames.
 *
 * The node takes a 0-based index spec ("8", "0-10", "0, 5, 8-12") and gates an
 * IMAGE batch down to those frames. Writing that spec used to be blind: type an
 * index, run the graph, look at what came out, adjust. This turns the node body
 * into a viewer for the batch it was last handed.
 *
 * It opens COLLAPSED — a single 26px strip carrying a chevron and the live
 * output summary, so a Framehold parked in a graph costs almost no canvas and
 * still tells you what it is emitting. Click the strip (or press Enter on it)
 * to drop the viewer down. The state lives in node.properties, so it travels
 * with the workflow. Collapsed, the node makes no frame requests at all.
 *
 * Expanded, the viewer is —
 *
 *   Enter      (collapsed) open the viewer
 *   ← / →      step one frame        (Shift: ten)
 *   Home/End   first / last frame
 *   [ / ]      previous / next SELECTED frame
 *   Space      play the batch at the preview fps
 *   Enter      set the spec to the frame you are on
 *   A          add this frame to the spec
 *   R          extend the last token of the spec to here (shift-click range)
 *
 * — with a `9 / 50` position counter and an output counter that says how many
 * frames the spec currently resolves to and how long that runs at the preview
 * fps. A tick strip under the image maps the whole batch and marks every frame
 * the spec selects, and an input/output toggle flips the playhead between
 * walking the source batch and walking the frames the node will emit — the
 * only way to see a hold or a reorder as it will actually play.
 *
 * Where the frames come from
 * --------------------------
 * The backend parks a thumbnail strip of the INPUT batch in RAM on each
 * execution (see bat_framehold.py) and serves one JPEG per request, keyed by a
 * token it mints per execution. So the strip is not in the workflow, not in
 * localStorage, and not in the execution payload — only the token is, which is
 * what lets a reopened workflow get its scrubber back without a re-run.
 *
 * Note the routes take the token and nothing else. Keying them on the node id
 * would have been the obvious thing and is quietly broken inside a subgraph:
 * the id the backend is handed is the flattened prompt id, not the `node.id`
 * this code can see.
 *
 * Two things carry state instead of the strip:
 *   • the token, in workflow-scoped localStorage (batNodeCacheKey);
 *   • the preview fps, in one studio-wide localStorage key — it is a viewing
 *     preference, not a property of the shot, and deliberately NOT a node
 *     widget: it must not end up in the prompt or dirty a workflow.
 *
 * Nodes 1.0 / 2.0
 * ---------------
 * Everything here is one DOM widget added through bat_node_layout.js, which
 * carries the sizing contract both renderers need (Nodes 2.0 derives node
 * height from computeLayoutSize; Nodes 1.0 from node.size). No litegraph
 * drawing, no `node.onResize` — the layout helper's ResizeObserver fires under
 * both. Widget reads and writes go through `widget.value` / `widget.callback`,
 * which the Vue renderer proxies into its value store, and a low-frequency
 * poll catches value changes that arrive by neither route (undo, a workflow
 * load, another extension writing the widget).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    addBatDOMWidget, clampNodeSize, disposeBatLayout, refreshBatLayout,
} from "./bat_node_layout.js";
import { batTrack, isNodeAlive, batNodeCacheKey } from "./bat_lifecycle.js";

const NODE_TYPE    = "Bat_Framehold";
const INFO_ROUTE   = "/bat/framehold/info";
const FRAME_ROUTE  = "/bat/framehold/frame";

const FPS_CHOICES  = [8, 12, 16, 23.976, 24, 25, 30, 48, 50, 60];
const FPS_KEY      = "bat_framehold_fps";
const TOKEN_PREFIX = "bat_framehold_tok";

// How many frames either side of the playhead to warm up. Six covers a fast
// key-repeat and a play start without queueing dozens of requests on a scrub.
const PRELOAD_AHEAD = 6;
// Cap on retained decoded Images. Beyond this the browser's own HTTP cache
// (the routes send max-age, and a token'd URL is immutable) is fast enough.
const PRELOAD_MAX   = 96;

// Collapsed the node is just its summary strip; expanded it is the strip plus
// the viewer. HEADER_H must match the header's own CSS height for the collapsed
// node to come out flush — it is the number both renderers are told.
const HEADER_H   = 26;
const BODY_H     = 320;
const EXPANDED_H = HEADER_H + BODY_H;

// Collapse state lives in node.properties, not localStorage: it is a property
// of this graph's layout, like the node's position, and should travel with the
// workflow rather than following the artist onto an unrelated shot.
const PROP_EXPANDED = "bat_framehold_expanded";

// A spec long enough to hang the UI is a typo, not a request. The backend has
// no such cap because it is bounded by the batch length anyway — this only
// guards the live preview's expansion of a not-yet-run spec.
const MAX_EXPAND    = 200000;

/* ─── frame-spec grammar ──────────────────────────────────────────────────────
 *
 * A mirror of _parse_frames() in bat_framehold.py — same tokens, same clamping,
 * same descending-range behaviour — so the selection this draws is the
 * selection the node will render. tools/test_framehold_parity.py runs both
 * against the same cases; if you change one, change the other.
 *
 * The one deliberate difference: `n <= 0` (nothing run yet) skips clamping
 * instead of clamping everything to -1, so the counters still read sensibly
 * before the first execution.
 */
export function parseFrameSpec(spec, n) {
    const out = [];
    if (!spec || !String(spec).trim()) return { list: out, error: null };

    const clamp = n > 0;
    const resolve = (v) => {
        if (v < 0) v = n + v;
        if (!clamp) return v;
        if (v < 0) return 0;
        if (v > n - 1) return n - 1;
        return v;
    };
    const rangeRe  = /^(-?\d+)\s*-\s*(-?\d+)$/;
    const singleRe = /^-?\d+$/;

    for (const raw of String(spec).split(",")) {
        const tok = raw.trim();
        if (!tok) continue;
        const m = rangeRe.exec(tok);
        if (m) {
            const lo = resolve(parseInt(m[1], 10));
            const hi = resolve(parseInt(m[2], 10));
            const step = hi >= lo ? 1 : -1;
            for (let v = lo; ; v += step) {
                out.push(v);
                if (v === hi || out.length > MAX_EXPAND) break;
            }
        } else if (singleRe.test(tok)) {
            out.push(resolve(parseInt(tok, 10)));
        } else {
            // Says what IS accepted, not just what failed — this string is the
            // node's only inline explanation of the grammar (the full one is
            // the `frames` widget's tooltip, from INPUT_TYPES).
            return {
                list: out,
                error: `cannot parse "${tok}" — expected an index (5, -1) ` +
                       `or a range (0-10, -5--1)`,
            };
        }
        if (out.length > MAX_EXPAND) {
            return { list: out, error: "spec expands to too many frames" };
        }
    }
    return { list: out, error: null };
}

/** First index a spec token addresses — the anchor for an R (range) extend. */
function tokenStart(tok, n) {
    const m = /^(-?\d+)\s*-\s*(-?\d+)$/.exec(tok.trim());
    const raw = m ? parseInt(m[1], 10) : parseInt(tok.trim(), 10);
    if (!Number.isFinite(raw)) return null;
    if (raw >= 0) return raw;
    return n > 0 ? Math.max(0, n + raw) : raw;
}

function fmtDuration(frames, fps) {
    if (!fps || !Number.isFinite(fps)) return "—";
    const s = frames / fps;
    if (s < 60) return `${s.toFixed(2)}s`;
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
}

function loadFps() {
    const v = parseFloat(localStorage.getItem(FPS_KEY));
    return FPS_CHOICES.includes(v) ? v : 25;   // UK studio default
}

/* ─── DOM construction ───────────────────────────────────────────────────── */

const BTN_CSS = "background:#242424;color:#ddd;border:1px solid #3a3a3a;" +
    "border-radius:3px;font:11px/1 monospace;padding:4px 7px;cursor:pointer;" +
    "min-width:24px;";

function mkBtn(label, title, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.style.cssText = BTN_CSS;
    b.onmouseenter = () => { b.style.background = "#2f2f2f"; };
    b.onmouseleave = () => { b.style.background = "#242424"; };
    b.onclick = (e) => { e.preventDefault(); onClick(e); };
    return b;
}

function buildEditor(node) {
    const root = document.createElement("div");
    root.tabIndex = 0;      // so the keyboard shortcuts have somewhere to land
    root.style.cssText =
        "display:flex;flex-direction:column;height:100%;width:100%;" +
        "box-sizing:border-box;background:#0c0c0c;border:1px solid #2a2a2a;" +
        "border-radius:4px;overflow:hidden;outline:none;";

    // ── header (the whole node when collapsed) ──────────────────────────
    // Deliberately not a separate litegraph button widget: this way the one
    // control is also the live readout, there is no extra widget row, and
    // nothing new lands in widgets_values to upset the positional deal that
    // maps saved values onto `frames`.
    const header = document.createElement("div");
    header.title = "Click to show / hide the frame scrubber";
    header.style.cssText =
        `display:flex;align-items:center;gap:6px;height:${HEADER_H}px;` +
        "box-sizing:border-box;padding:0 8px;background:#161616;cursor:pointer;" +
        "font:11px monospace;color:#8a8a8a;flex:none;user-select:none;";

    header.onmouseenter = () => { header.style.background = "#1d1d1d"; };
    header.onmouseleave = () => { header.style.background = "#161616"; };

    const chev = document.createElement("span");
    chev.textContent = "▸";
    chev.style.cssText = "color:#9cf;width:10px;flex:none;";

    const chevLabel = document.createElement("span");
    chevLabel.textContent = "preview";
    chevLabel.style.cssText = "color:#666;flex:none;";

    const status = document.createElement("div");
    status.style.cssText =
        "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
    status.textContent = "out — · —";
    header.append(chev, chevLabel, status);

    // ── body (everything the collapse hides) ────────────────────────────
    const body = document.createElement("div");
    body.style.cssText =
        "display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden;";

    // ── viewer ──────────────────────────────────────────────────────────
    const stage = document.createElement("div");
    stage.style.cssText =
        "position:relative;flex:1;min-height:60px;background:#000;overflow:hidden;";

    const img = document.createElement("img");
    img.draggable = false;
    img.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;" +
        "display:block;image-rendering:auto;";
    // A 404 here means the strip is gone — the server restarted, or the LRU
    // evicted it for a bigger batch elsewhere in the graph. Fall back to the
    // "run once" hint rather than leaving a broken-image glyph on the node.
    img.addEventListener("error", () => {
        const st = node._batFH;
        if (!st || !st.token) return;
        st.token = "";
        st.pre.clear();
        render(node);
    });
    stage.appendChild(img);

    const badge = document.createElement("div");
    badge.style.cssText =
        "position:absolute;left:6px;top:5px;font:10px monospace;color:#9aa;" +
        "background:rgba(0,0,0,0.55);padding:1px 5px;border-radius:2px;" +
        "pointer-events:none;";
    stage.appendChild(badge);

    // Whether the frame on screen is one the spec actually selects. The whole
    // point of the node is that relationship, so it gets a permanent readout
    // rather than something you infer from the tick strip.
    const inSel = document.createElement("div");
    inSel.style.cssText =
        "position:absolute;right:6px;top:5px;font:10px monospace;" +
        "padding:1px 5px;border-radius:2px;pointer-events:none;";
    stage.appendChild(inSel);

    const hint = document.createElement("div");
    hint.style.cssText =
        "position:absolute;inset:0;display:flex;align-items:center;" +
        "justify-content:center;text-align:center;font:11px monospace;" +
        "color:#777;padding:12px;pointer-events:none;";
    hint.textContent = "Run the graph once to load the preview strip.";
    stage.appendChild(hint);

    // ── scrub strip ─────────────────────────────────────────────────────
    const scrub = document.createElement("canvas");
    scrub.style.cssText =
        "display:block;width:100%;height:20px;cursor:pointer;background:#141414;" +
        "border-top:1px solid #222;border-bottom:1px solid #222;";

    // ── transport ───────────────────────────────────────────────────────
    const bar = document.createElement("div");
    bar.style.cssText =
        "display:flex;flex-wrap:wrap;align-items:center;gap:4px;padding:5px 6px;" +
        "background:#161616;font:11px monospace;color:#bbb;";

    const counter = document.createElement("span");
    counter.style.cssText =
        "min-width:64px;text-align:center;color:#eee;font:12px monospace;" +
        "background:#101010;border:1px solid #333;border-radius:3px;padding:3px 6px;";
    counter.textContent = "— / —";

    const idxInput = document.createElement("input");
    idxInput.type = "number";
    idxInput.title = "Frame index (0-based, as written in the spec)";
    idxInput.style.cssText =
        "width:58px;background:#101010;color:#9cf;border:1px solid #333;" +
        "border-radius:3px;font:11px monospace;padding:3px 4px;";
    // LiteGraph listens for keys on the document; without this, typing an
    // index would also trigger canvas shortcuts (and our own root handler).
    idxInput.addEventListener("keydown", (e) => e.stopPropagation());

    const fpsChip = mkBtn(`${loadFps()}`, "Preview fps (click to cycle) — view only, never sent to the prompt", () => {});
    fpsChip.style.color = "#9aa";

    const modeChip = mkBtn("input", "Scrub the input batch / the frames this node outputs", () => {});
    modeChip.style.minWidth = "46px";

    // Step glyphs are chevrons, not triangles: a "▶" next to the play button
    // reads as a second play button.
    const playBtn  = mkBtn("▶", "Play / pause (Space)", () => {});
    const first    = mkBtn("⏮", "First frame (Home)", () => {});
    const back10   = mkBtn("«", "Back ten frames (Shift+←)", () => {});
    const back1    = mkBtn("‹", "Back one frame (←)", () => {});
    const fwd1     = mkBtn("›", "Forward one frame (→)", () => {});
    const fwd10    = mkBtn("»", "Forward ten frames (Shift+→)", () => {});
    const last     = mkBtn("⏭", "Last frame (End)", () => {});
    const setBtn   = mkBtn("Set", "Set the spec to this frame (Enter)", () => {});
    const addBtn   = mkBtn("+Add", "Add this frame to the spec (A)", () => {});
    const rangeBtn = mkBtn("↔Range", "Extend the last spec token to this frame (R)", () => {});

    setBtn.style.color = "#9cf";
    addBtn.style.color = "#9cf";
    rangeBtn.style.color = "#9cf";

    const spacer = document.createElement("span");
    spacer.style.cssText = "flex:1;";

    bar.append(first, back10, back1, playBtn, fwd1, fwd10, last,
               counter, idxInput, spacer, modeChip, fpsChip,
               setBtn, addBtn, rangeBtn);

    body.append(stage, scrub, bar);
    root.append(header, body);

    return {
        root, header, chev, body, stage, img, badge, inSel, hint, scrub,
        counter, idxInput, fpsChip, modeChip, playBtn, first, back10, back1,
        fwd1, fwd10, last, setBtn, addBtn, rangeBtn, status,
    };
}

/* ─── state + rendering ──────────────────────────────────────────────────── */

function specWidget(node) {
    return node.widgets?.find((w) => w.name === "frames") || null;
}

function frameURL(st, i) {
    return api.apiURL(`${FRAME_ROUTE}?token=${encodeURIComponent(st.token)}&i=${i}`);
}

/** Warm the browser cache around the playhead so stepping never flashes. */
function preload(st, centre) {
    if (!st.token || !st.frames) return;
    for (let d = 1; d <= PRELOAD_AHEAD; d++) {
        for (const i of [centre + d, centre - d]) {
            if (i < 0 || i >= st.frames || st.pre.has(i)) continue;
            const im = new Image();
            im.src = frameURL(st, i);
            st.pre.set(i, im);
        }
    }
    // Map iteration order is insertion order, so the oldest warmed frames go
    // first — which is also the furthest from wherever the playhead has got to.
    while (st.pre.size > PRELOAD_MAX) {
        st.pre.delete(st.pre.keys().next().value);
    }
}

/**
 * Resolve the current spec against the batch length.
 *
 * Memoised on (spec, batch length) because render() runs this and render() also
 * runs once per frame during playback — re-expanding "0-500" twenty-five times
 * a second to get the same answer is free to avoid.
 *
 * `passthrough` mirrors a real backend behaviour that is easy to miss: an empty
 * spec does NOT output nothing, it logs and passes the whole batch through. The
 * counters have to say so, or the node reads as "out 0 frames" while it is
 * actually emitting fifty.
 */
function refreshSelection(node) {
    const st = node._batFH;
    const w = specWidget(node);
    const spec = w ? String(w.value ?? "") : "";
    if (st.parsedFor === `${st.frames}\u0000${spec}`) return;

    const { list, error } = parseFrameSpec(spec, st.frames);
    st.spec = spec;
    st.sel = list;
    st.selSet = new Set(list);
    st.specError = error;
    st.passthrough = !error && list.length === 0;
    st.parsedFor = `${st.frames}\u0000${spec}`;
}

/** How many frames the node will actually emit for the current spec. */
function outputCount(node) {
    const st = node._batFH;
    return st.passthrough ? st.frames : st.sel.length;
}

/** The source frame index the playhead is on, in either mode. */
function currentSource(node) {
    const st = node._batFH;
    if (st.mode === "output" && !st.passthrough) {
        if (!st.sel.length) return 0;
        const p = Math.max(0, Math.min(st.pos, st.sel.length - 1));
        return st.sel[p];
    }
    // Passthrough output == the input batch, so the two timelines coincide.
    return Math.max(0, Math.min(st.pos, Math.max(0, st.frames - 1)));
}

/** How many positions the playhead can occupy in the current mode. */
function trackLength(node) {
    const st = node._batFH;
    return st.mode === "output" ? outputCount(node) : st.frames;
}

function paintScrub(node) {
    const st = node._batFH;
    const c = st.ui.scrub;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round((c.clientWidth || 1) * dpr));
    const ch = Math.max(1, Math.round((c.clientHeight || 20) * dpr));
    if (c.width !== cw) c.width = cw;
    if (c.height !== ch) c.height = ch;

    const ctx = c.getContext("2d");
    ctx.fillStyle = "#141414";
    ctx.fillRect(0, 0, cw, ch);

    const n = st.frames;
    if (!n) return;

    // Selection ticks, always drawn against the INPUT timeline even in output
    // mode — the strip is a map of the source batch, and moving it under the
    // artist when they flip modes would make it useless as one.
    const barW = Math.max(1 * dpr, cw / n);
    ctx.fillStyle = "#3a6ea5";
    for (const i of st.selSet) {
        if (i < 0 || i >= n) continue;
        ctx.fillRect((i / n) * cw, 0, barW, ch);
    }

    const src = currentSource(node);
    ctx.fillStyle = st.selSet.has(src) ? "#7ec0ee" : "#e8e8e8";
    ctx.fillRect(Math.max(0, (src / n) * cw - dpr), 0, Math.max(2 * dpr, barW), ch);
}

function render(node) {
    const st = node._batFH;
    const ui = st.ui;
    refreshSelection(node);

    const n = st.frames;
    const len = trackLength(node);
    if (len > 0) st.pos = Math.max(0, Math.min(st.pos, len - 1));
    const src = currentSource(node);

    // Collapsed, the viewer is not on screen — so it does not fetch. A node
    // parked in a graph should cost nothing until you open it.
    if (n && st.token && !st.collapsed) {
        ui.hint.style.display = "none";
        const url = frameURL(st, src);
        if (ui.img.getAttribute("src") !== url) ui.img.src = url;
        preload(st, src);
    } else {
        ui.img.removeAttribute("src");
        ui.hint.style.display = "flex";
    }

    // "9 / 50" — human position over the track the playhead is walking.
    ui.counter.textContent = len ? `${st.pos + 1} / ${len}` : "— / —";
    if (document.activeElement !== ui.idxInput) ui.idxInput.value = n ? String(src) : "";
    ui.idxInput.max = n ? String(n - 1) : "0";

    ui.badge.textContent = n
        ? `${st.srcW}×${st.srcH} · in ${n}f · ${fmtDuration(n, st.fps)}` +
          (st.mode === "output" ? ` · src ${src}` : "")
        : "no preview";

    const selected = st.passthrough || st.selSet.has(src);
    ui.inSel.textContent = n
        ? (st.passthrough ? "all frames" : (selected ? "selected" : "not selected"))
        : "";
    ui.inSel.style.color = selected ? "#111" : "#bbb";
    ui.inSel.style.background = selected ? "rgba(126,192,238,0.9)" : "rgba(0,0,0,0.55)";

    const outN = outputCount(node);
    const holds = st.sel.length - new Set(st.sel).size;
    if (st.specError) {
        ui.status.textContent = `spec error — ${st.specError}`;
        ui.status.style.color = "#e08080";
    } else if (!n) {
        // Collapsed with no strip yet, the header is the only thing on the
        // node — "out 0 frames" would read as a result rather than as "this
        // has not run".
        ui.status.style.color = "#666";
        ui.status.textContent = "run once to load the scrubber";
    } else {
        ui.status.style.color = "#8a8a8a";
        ui.status.textContent =
            `out ${outN} frame${outN === 1 ? "" : "s"} · ${fmtDuration(outN, st.fps)} ` +
            `@ ${st.fps}fps` + (holds > 0 ? `  (${holds} held/repeated)` : "") +
            (st.passthrough ? "  (empty spec — whole batch passes through)" : "") +
            (n ? `   ·   in ${n}f · ${fmtDuration(n, st.fps)}` : "");
    }

    ui.modeChip.textContent = st.mode;
    ui.modeChip.style.color = st.mode === "output" ? "#9cf" : "#9aa";
    ui.playBtn.textContent = st.timer ? "❚❚" : "▶";

    if (!st.collapsed) paintScrub(node);
}

/* ─── actions ────────────────────────────────────────────────────────────── */

function seek(node, pos) {
    const st = node._batFH;
    const len = trackLength(node);
    if (!len) return;
    st.pos = Math.max(0, Math.min(Math.round(pos), len - 1));
    render(node);
}

function step(node, delta) {
    seek(node, node._batFH.pos + delta);
}

/** Jump to the previous / next frame the spec selects (input mode only). */
function stepSelected(node, dir) {
    const st = node._batFH;
    if (st.mode === "output") return step(node, dir);
    const sorted = [...st.selSet].sort((a, b) => a - b);
    if (!sorted.length) return;
    const cur = currentSource(node);
    const next = dir > 0
        ? sorted.find((i) => i > cur)
        : [...sorted].reverse().find((i) => i < cur);
    if (next !== undefined) seek(node, next);
}

function writeSpec(node, value) {
    const w = specWidget(node);
    if (!w) return;
    w.value = value;
    node._batFH.parsedFor = null;
    // Both renderers: the callback is what the Vue widget and litegraph agree
    // on for "the artist changed this", and it is what any other extension
    // hooked to the widget is listening for.
    try { w.callback?.call(w, value); } catch (_) {}
    node.setDirtyCanvas?.(true, true);
    render(node);
}

function actionSet(node)  { writeSpec(node, String(currentSource(node))); }

function actionAdd(node) {
    const st = node._batFH;
    const cur = String(currentSource(node));
    const base = (st.spec || "").trim().replace(/,\s*$/, "");
    writeSpec(node, base ? `${base}, ${cur}` : cur);
}

function actionRange(node) {
    const st = node._batFH;
    const cur = currentSource(node);
    const toks = (st.spec || "").split(",").map((t) => t.trim()).filter(Boolean);
    if (!toks.length) return actionSet(node);
    // Replace the last token with anchor→here, so R behaves like a shift-click:
    // "3" → "3-8", and "3-7" → "3-8" rather than stacking a second range.
    const anchor = tokenStart(toks[toks.length - 1], st.frames);
    if (anchor === null) return actionSet(node);
    toks[toks.length - 1] = anchor === cur ? String(cur) : `${anchor}-${cur}`;
    writeSpec(node, toks.join(", "));
}

function stopPlay(node) {
    const st = node._batFH;
    if (st.timer) { clearInterval(st.timer); st.timer = null; }
}

function togglePlay(node) {
    const st = node._batFH;
    if (st.timer) { stopPlay(node); render(node); return; }
    const len = trackLength(node);
    if (!len || !st.token) return;
    st.timer = setInterval(() => {
        // The node can be deleted mid-playback; the interval is also registered
        // with batTrack, but checking here stops a final tick painting into a
        // detached DOM in the window between the two.
        if (!isNodeAlive(node)) { stopPlay(node); return; }
        const l = trackLength(node);
        if (!l) { stopPlay(node); render(node); return; }
        st.pos = (st.pos + 1) % l;
        render(node);
    }, Math.max(16, 1000 / (st.fps || 25)));
    render(node);
}

/* ─── collapse ───────────────────────────────────────────────────────────── */

/**
 * Show or hide the viewer, keeping the summary strip.
 *
 * The height the widget advertises is a function of this flag (see install()),
 * so both renderers pick the new size up from refreshBatLayout(): Nodes 2.0
 * re-derives it from computeLayoutSize, Nodes 1.0 gets an explicit resize.
 */
function setCollapsed(node, collapsed, opts = {}) {
    const st = node._batFH;
    if (!st) return;
    st.collapsed = !!collapsed;
    if (st.collapsed) stopPlay(node);
    st.ui.body.style.display = st.collapsed ? "none" : "flex";
    st.ui.chev.textContent = st.collapsed ? "▸" : "▾";
    if (!opts.silent) {
        try { (node.properties ||= {})[PROP_EXPANDED] = !st.collapsed; } catch (_) {}
    }
    refreshBatLayout(node, st.widget, { shrink: st.collapsed });
    render(node);
    // Opening onto a stale playhead should still show a frame: render() only
    // sets the image when expanded, so the first expand is what triggers the
    // fetch. Focus follows so the arrow keys work without a further click.
    if (!st.collapsed) { try { st.ui.root.focus({ preventScroll: true }); } catch (_) {} }
}


/* ─── strip ingest ───────────────────────────────────────────────────────── */

function applyStrip(node, info) {
    const st = node._batFH;
    const hadStrip = !!st.token;
    const changed = st.token !== info.token;
    st.token = info.token;
    st.frames = info.frames | 0;
    st.srcW = info.src_w | 0;
    st.srcH = info.src_h | 0;
    // Every execution mints a new token, so the warmed URLs are all stale.
    if (changed) { st.pre.clear(); st.parsedFor = null; }
    // Land on the frame the spec already points at — but ONLY when the
    // scrubber is being populated for the first time (a first run, or a
    // workflow reopen). Doing it on every execution would yank the playhead
    // back to the spec each time the graph re-runs, which is precisely when
    // the artist is stepping around looking for the next hold.
    if (changed && !hadStrip) {
        refreshSelection(node);
        st.pos = st.mode === "output" ? 0
            : Math.max(0, Math.min(st.sel[0] ?? 0, Math.max(0, st.frames - 1)));
    }
    try {
        localStorage.setItem(batNodeCacheKey(app, TOKEN_PREFIX, node), info.token);
    } catch (_) { /* private mode / quota — the preview just won't survive a reload */ }
    render(node);
}

/**
 * Ask the backend whether the strip behind a remembered token is still there.
 *
 * Called on load, so a reopened workflow gets its scrubber back without a
 * re-run. A miss (server restarted, or the LRU evicted it) is not an error —
 * the node just stays on its "run once" hint.
 */
async function probeStrip(node, token) {
    const st = node._batFH;
    if (!token) return;
    const seq = ++st.probeSeq;
    try {
        const r = await fetch(api.apiURL(
            `${INFO_ROUTE}?token=${encodeURIComponent(token)}`));
        const info = await r.json();
        if (seq !== st.probeSeq || !isNodeAlive(node)) return;
        if (info.ok) applyStrip(node, info);
    } catch (_) { /* server restarted or route missing — stay on the hint */ }
}

/** Re-attach to the strip this node was last given, if the backend still has it. */
function restoreStrip(node) {
    if (!isNodeAlive(node) || !node._batFH) return;
    // `silent` so merely reopening a workflow doesn't write the property back
    // and mark it dirty. A node that has never been expanded has no property
    // and stays closed — the deliberate default, so dropping a Framehold into
    // a graph costs one strip of canvas.
    setCollapsed(node, node.properties?.[PROP_EXPANDED] !== true, { silent: true });
    let tok = null;
    try { tok = localStorage.getItem(batNodeCacheKey(app, TOKEN_PREFIX, node)); }
    catch (_) { /* private mode */ }
    if (tok) probeStrip(node, tok);
    else render(node);
}


/* ─── wiring ─────────────────────────────────────────────────────────────── */

function install(node) {
    const ui = buildEditor(node);
    const track = batTrack(node);

    node._batFH = {
        ui,
        token: "", frames: 0, srcW: 0, srcH: 0,
        pos: 0, mode: "input", fps: loadFps(),
        spec: "", sel: [], selSet: new Set(), specError: null,
        passthrough: false, parsedFor: null,
        collapsed: true, widget: null,
        pre: new Map(), timer: null, probeSeq: 0, lastSpec: null,
    };
    const st = node._batFH;

    // ── transport wiring ────────────────────────────────────────────────
    ui.first.onclick   = () => seek(node, 0);
    ui.back10.onclick  = () => step(node, -10);
    ui.back1.onclick   = () => step(node, -1);
    ui.fwd1.onclick    = () => step(node, +1);
    ui.fwd10.onclick   = () => step(node, +10);
    ui.last.onclick    = () => seek(node, trackLength(node) - 1);
    ui.playBtn.onclick = () => togglePlay(node);
    ui.setBtn.onclick   = () => actionSet(node);
    ui.addBtn.onclick   = () => actionAdd(node);
    ui.rangeBtn.onclick = () => actionRange(node);

    ui.fpsChip.onclick = () => {
        const i = FPS_CHOICES.indexOf(st.fps);
        st.fps = FPS_CHOICES[(i + 1) % FPS_CHOICES.length];
        try { localStorage.setItem(FPS_KEY, String(st.fps)); } catch (_) {}
        if (st.timer) { stopPlay(node); togglePlay(node); }   // re-time playback
        ui.fpsChip.textContent = String(st.fps);
        render(node);
    };

    ui.modeChip.onclick = () => {
        // Keep the frame under the playhead when flipping modes where that is
        // possible, so the toggle reads as a change of timeline rather than a
        // jump cut.
        const src = currentSource(node);
        st.mode = st.mode === "input" ? "output" : "input";
        if (st.mode === "output") {
            const at = st.sel.indexOf(src);
            st.pos = at >= 0 ? at : 0;
        } else {
            st.pos = src;
        }
        render(node);
    };

    ui.idxInput.onchange = () => {
        const v = parseInt(ui.idxInput.value, 10);
        if (!Number.isFinite(v) || !st.frames) return render(node);
        const i = Math.max(0, Math.min(v, st.frames - 1));
        if (st.mode === "output") {
            const at = st.sel.indexOf(i);
            if (at >= 0) st.pos = at; else { st.mode = "input"; st.pos = i; }
        } else {
            st.pos = i;
        }
        render(node);
    };

    // ── scrub strip ─────────────────────────────────────────────────────
    const scrubTo = (clientX) => {
        const r = ui.scrub.getBoundingClientRect();
        if (!r.width || !st.frames) return;
        const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
        const srcIdx = Math.min(st.frames - 1, Math.floor(frac * st.frames));
        if (st.mode === "output") {
            // The strip is the INPUT timeline in both modes, so a click has to
            // be mapped back onto the output list — nearest selected position
            // at or before the click, which is what "scrub to here" means when
            // the output only samples parts of the source.
            let best = 0;
            for (let i = 0; i < st.sel.length; i++) {
                if (st.sel[i] <= srcIdx) best = i;
            }
            st.pos = best;
        } else {
            st.pos = srcIdx;
        }
        render(node);
    };
    ui.scrub.addEventListener("pointerdown", (e) => {
        stopPlay(node);
        ui.scrub.setPointerCapture?.(e.pointerId);
        st.scrubbing = true;
        scrubTo(e.clientX);
        e.preventDefault();
    });
    ui.scrub.addEventListener("pointermove", (e) => {
        if (st.scrubbing) scrubTo(e.clientX);
    });
    const endScrub = () => { st.scrubbing = false; };
    ui.scrub.addEventListener("pointerup", endScrub);
    ui.scrub.addEventListener("pointercancel", endScrub);

    ui.header.addEventListener("click", () => setCollapsed(node, !st.collapsed));

    // ── keyboard ────────────────────────────────────────────────────────
    ui.root.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        // Collapsed, there is nothing on screen for these to move. Enter opens
        // the panel — the one shortcut that still makes sense from the strip.
        if (st.collapsed) {
            if (e.key !== "Enter") return;
            setCollapsed(node, false);
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        // A focused button already answers Space and Enter itself. Acting on
        // them here too fires the shortcut twice — and "twice" for a toggle is
        // a no-op, which reads as the key having done nothing. (bat_video_combine
        // has this same wart: click Play, then press Space, and nothing happens.)
        if (e.target.tagName === "BUTTON" && (e.key === " " || e.key === "Enter")) return;
        let handled = true;
        switch (e.key) {
            case "ArrowLeft":  step(node, e.shiftKey ? -10 : -1); break;
            case "ArrowRight": step(node, e.shiftKey ? +10 : +1); break;
            case "Home":       seek(node, 0); break;
            case "End":        seek(node, trackLength(node) - 1); break;
            case "[":          stepSelected(node, -1); break;
            case "]":          stepSelected(node, +1); break;
            case " ":          togglePlay(node); break;
            case "Enter":      actionSet(node); break;
            case "a": case "A": actionAdd(node); break;
            case "r": case "R": actionRange(node); break;
            default: handled = false;
        }
        if (handled) {
            // Without stopPropagation LiteGraph's document-level handler also
            // acts on the event — ←/→ would step a frame here AND move the
            // canvas selection to the next node. Same pattern as
            // bat_video_combine.js / bat_video_loader.js.
            e.preventDefault();
            e.stopPropagation();
        }
    });
    // Focus on any click inside, so the shortcuts work without tabbing in —
    // but never steal focus from a button or the index box we just clicked.
    ui.root.addEventListener("mousedown", () => {
        setTimeout(() => {
            const a = document.activeElement;
            if (!ui.root.contains(a) || a === document.body) {
                ui.root.focus({ preventScroll: true });
            }
        }, 0);
    });

    // ── the DOM widget ──────────────────────────────────────────────────
    // Both heights are FUNCTIONS of the collapse flag, which is the whole
    // mechanism: every layout pass in either renderer asks again, so toggling
    // the flag and calling refreshBatLayout() is all a collapse takes. Pinning
    // max === min while collapsed stops the node being dragged taller than its
    // own summary strip.
    st.widget = addBatDOMWidget(node, "bat_framehold_preview", "bat_framehold_preview",
        ui.root, {
            minWidth: 320,
            height: () => (st.collapsed ? HEADER_H : EXPANDED_H),
            growable: true,
            maxHeight: () => (st.collapsed ? HEADER_H : null),
            onResize: () => { if (isNodeAlive(node) && !st.collapsed) paintScrub(node); },
        });
    // Only floors the WIDTH: a height floor would fight the collapse, and the
    // widget's own height contract covers the expanded case in both renderers.
    clampNodeSize(node, 340, 0);

    // ── spec tracking ───────────────────────────────────────────────────
    // The widget callback covers a normal edit in both renderers; the poll
    // covers everything that writes the value without going through it —
    // undo/redo, a workflow load, another extension. A string compare every
    // 250 ms is nothing next to being wrong about what is selected.
    const w = specWidget(node);
    if (w) {
        const orig = w.callback;
        w.callback = function (v) {
            try { orig?.apply(this, arguments); } catch (_) {}
            if (!isNodeAlive(node)) return;
            st.parsedFor = null;
            render(node);
        };
    }
    track.interval(setInterval(() => {
        if (!isNodeAlive(node)) return;
        const cur = specWidget(node)?.value;
        if (cur !== st.lastSpec) { st.lastSpec = cur; st.parsedFor = null; render(node); }
    }, 250));

    setCollapsed(node, true, { silent: true });

    track.dispose(() => {
        stopPlay(node);
        st.pre.clear();
        st.ui.img.removeAttribute("src");
        // bat_node_layout exports this but nothing in the pack calls it, so
        // every editor's ResizeObserver currently outlives its node. Not fixing
        // that here beyond this node's own observer.
        disposeBatLayout(node);
    });

    render(node);
}

app.registerExtension({
    name: "BAT.Framehold",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            install(this);
            // node.id is only final once LiteGraph has finished construction,
            // and the strip is keyed on it — hence the deferred re-read rather
            // than capturing it in install().
            // The remembered token is stored per (workflow, node.id), and
            // node.id is only final once LiteGraph has finished construction —
            // hence the defer rather than reading it inline.
            setTimeout(() => restoreStrip(this), 0);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => restoreStrip(this), 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batFH) return r;
            const one = (v) => (Array.isArray(v) ? v[0] : v);
            const token = one(message.token);
            if (!token) return r;
            applyStrip(this, {
                token,
                frames: one(message.frames),
                src_w: one(message.src_w),
                src_h: one(message.src_h),
            });
            return r;
        };
    },
});
