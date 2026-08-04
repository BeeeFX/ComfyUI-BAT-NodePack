/**
 * BAT Video Combine — canvas-side player UI.
 *
 * Sibling Python: bat_video_combine.py.
 *
 * The player adds on top of a bare `<video>` element:
 *   - Play / Pause + frame-step buttons (single + 10-frame jumps).
 *   - Speed dropdown (0.25× / 0.5× / 1× / 2×).
 *   - Mute + volume slider (visible only when the file has audio).
 *   - Timeline scrubber with two draggable loop-in/out pins for review.
 *   - Hover-scrub thumbnails (debounced, in-memory cached).
 *   - "Save current frame as PNG" → /bat/video/save_frame endpoint.
 *   - Fullscreen.
 *   - Keyboard shortcuts (focus-scoped): Space, ←/→, ,/., Home/End, F, M, Esc.
 *
 * ProRes / FFV1 / h265 outputs come back with browser_playable=false in
 * the onExecuted payload; for those the <video> src points at the
 * /bat/video/preview endpoint (server-side transcode) instead of /view.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { batTrack, batNodeCacheKey } from "./bat_lifecycle.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";

const NODE_TYPE = "Bat_VideoCombine";

// ─── Dynamic per-format codec widgets ────────────────────────────────────────
//
// The node exposes one "format" COMBO (h264 / prores / webm / …). Each format
// declares its own knobs (crf, preset, profile, …) in bat_video_formats/*.json;
// we fetch those specs from /bat/video/formats and, whenever the chosen format
// changes, swap the widgets shown below "format" to match. The widget *names*
// equal the JSON keys, so ComfyUI's graphToPrompt serialises them straight into
// the node's inputs, where the Python combine() picks them up via
// **format_widget_kwargs — no extra plumbing needed.
//
// `derived` widgets (e.g. ProRes pix_fmt) aren't shown; they're recomputed from
// a visible widget (profile) so the user can't pick an invalid combo. They're
// still serialised (the backend re-derives authoritatively regardless).

let _formatSpecsPromise = null;
function getFormatSpecs() {
    // Fetched once per session; the specs are static for a server run.
    if (!_formatSpecsPromise) {
        _formatSpecsPromise = api.fetchApi("/bat/video/formats")
            .then((r) => r.json())
            .catch((err) => {
                console.error("[Bat_VideoCombine] could not load format specs:", err);
                return {};
            });
    }
    return _formatSpecsPromise;
}

// Marker so we only ever remove widgets WE added (never the node's own).
const CODEC_WIDGET_FLAG = "_batCodecWidget";

function removeCodecWidgets(node) {
    if (!node.widgets) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const w = node.widgets[i];
        if (!w[CODEC_WIDGET_FLAG]) continue;
        // Prefer the node's own removeWidget (clears _widgetSlotsDirty / store
        // registration in modern litegraph); fall back to splice on older builds.
        if (typeof node.removeWidget === "function") {
            try { node.removeWidget(w); }
            catch (e) { node.widgets.splice(i, 1); }
        } else {
            node.widgets.splice(i, 1);
        }
        if (typeof w.onRemove === "function") { try { w.onRemove(); } catch (e) {} }
    }
}

// Build (but don't insert) a litegraph widget object from one JSON spec.
function makeCodecWidget(node, spec, savedValue) {
    const type = String(spec.type || "STRING").toUpperCase();
    const hasSaved = savedValue !== undefined && savedValue !== null;
    const w = {
        name: spec.name,
        [CODEC_WIDGET_FLAG]: true,
        // `label` is what litegraph draws; `name` is what serialises.
        label: spec.label || spec.name,
        options: {},
    };

    if (type === "COMBO" || Array.isArray(spec.options)) {
        w.type = "combo";
        w.options = { values: spec.options || [] };
        w.value = hasSaved ? savedValue : (spec.default ?? (spec.options || [])[0]);
    } else if (type === "INT" || type === "FLOAT") {
        w.type = "number";
        w.options = {
            min: spec.min, max: spec.max,
            // INT steps by whole numbers; litegraph's internal step is /10.
            step: type === "INT" ? 10 : 1,
            precision: type === "INT" ? 0 : 2,
        };
        w.value = hasSaved ? savedValue : (spec.default ?? spec.min ?? 0);
        w.callback = function (v) {
            let val = Number(v);
            if (type === "INT") val = Math.round(val);
            if (spec.min != null) val = Math.max(spec.min, val);
            if (spec.max != null) val = Math.min(spec.max, val);
            this.value = val;
        };
    } else if (type === "BOOLEAN") {
        w.type = "toggle";
        w.value = hasSaved ? savedValue : !!spec.default;
    } else {
        w.type = "text";
        w.value = hasSaved ? savedValue : (spec.default ?? "");
    }

    // Hidden (derived) widgets still serialise — the prompt serialiser keys off
    // widget.name and only skips options.serialize===false, not w.hidden — but
    // must not draw or reserve a row. Current litegraph excludes widgets with
    // `w.hidden` from layout (`this.widgets.filter(w => !w.hidden)`); the
    // computeSize:[0,-4] is a fallback for older builds that size by height.
    if (spec.hidden) {
        // `type = "hidden"` is what Nodes 2.0 keys its collapse rule on: the
        // Vue DOMWidget's computeLayoutSize returns all-zero sizes only when
        // `this.type === "hidden"`. Setting just `w.hidden = true` left the type
        // as "number"/"combo"/"text", so derived widgets the artist must NOT
        // touch (e.g. ProRes pix_fmt) would render as visible, editable rows.
        // The other two are the Nodes 1.0 legs: litegraph filters on `w.hidden`,
        // and computeSize:[0,-4] covers older builds that size by height.
        w.type = "hidden";
        w.hidden = true;
        w.computeSize = () => [0, -4];
    }

    w.node = node;
    return w;
}

// Apply a format's `derived` rules: set each hidden target widget's value from
// its source widget via the JSON map. Called on build and on source change.
function applyDerived(node, derived) {
    if (!derived) return;
    for (const [target, rule] of Object.entries(derived)) {
        const src = node.widgets?.find((w) => w.name === rule.from);
        const dst = node.widgets?.find((w) => w.name === target);
        if (!dst) continue;
        const key = src != null ? String(src.value) : undefined;
        const mapped = (rule.map && key in rule.map) ? rule.map[key]
                     : (rule.default ?? dst.value);
        dst.value = mapped;
    }
}

// Rebuild the codec widgets for the node's currently-selected format.
// `savedValues` (name→value) lets us restore values after a workflow load.
async function rebuildCodecWidgets(node, formatLabel, savedValues) {
    const specs = await getFormatSpecs();
    const def = specs[formatLabel];
    removeCodecWidgets(node);
    if (!def) {
        console.warn("[Bat_VideoCombine] no widget spec for format", formatLabel,
                     "— known:", Object.keys(specs));
        return;
    }
    console.info("[Bat_VideoCombine] building", (def.widgets || []).length,
                 "codec widgets for", formatLabel,
                 "→", (def.widgets || []).map((w) => w.name));

    // Add via addCustomWidget (NOT a raw node.widgets.push): litegraph wraps the
    // plain spec into a concrete widget class (toConcreteWidget), marks the slot
    // layout dirty and binds the node id — a bare push skips all of that and the
    // widget silently never renders.
    //
    // addCustomWidget *appends* to the tail of node.widgets. That tail is past
    // the DOM player widget (added in onNodeCreated), and a DOM widget occupies
    // every row from its own `y` down to the node's bottom edge. Anything left
    // after it therefore lands inside the player's area and is painted over by
    // the player's <div> — present in node.widgets, positioned, but invisible.
    // (This is what made the ProRes `profile` dropdown vanish: it sat at y≈336
    // behind a player spanning y=190→bottom of a 360px-tall node.) So after each
    // append we move the widget to just *before* the player, giving it a normal
    // row above the player. Codec widgets still sit after every static widget, so
    // positional widgets_values restore on load never shifts onto a static one.
    // (The DOM player is serialize:false, so the serialiser skips it anyway.)
    const playerIndex = () => node.widgets.findIndex((w) => w.name === "bat_video_player");
    const derived = def.derived || {};
    for (const spec of def.widgets || []) {
        const saved = savedValues ? savedValues[spec.name] : undefined;
        const plain = makeCodecWidget(node, spec, saved);
        // addCustomWidget returns the concrete instance; fall back to the plain
        // object on any older build that lacks the method.
        const w = (typeof node.addCustomWidget === "function")
            ? node.addCustomWidget(plain)
            : (node.widgets.push(plain), plain);
        // Carry the marker onto the concrete instance (toClass copies own
        // props, but be explicit so removeCodecWidgets always finds it).
        w[CODEC_WIDGET_FLAG] = true;
        // Same three-way hide as makeCodecWidget: type for Nodes 2.0, `hidden`
        // for current litegraph, computeSize for older builds. (Serialisation is
        // unaffected — the frontend keys that off serializeValue/options.serialize,
        // never the widget type — so derived values still reach the backend.)
        if (spec.hidden) {
            w.type = "hidden";
            w.hidden = true;
            w.computeSize = () => [0, -4];
        }
        // Relocate from the tail to just before the DOM player so it draws in a
        // real row instead of behind the player. No-op when there's no player
        // widget (the index is then -1 and the widget is already at the tail).
        const pi = playerIndex();
        if (pi !== -1 && pi < node.widgets.length - 1) {
            const idx = node.widgets.indexOf(w);
            if (idx !== -1) {
                node.widgets.splice(idx, 1);
                node.widgets.splice(playerIndex(), 0, w);
            }
        }
        // When a source of a derived rule changes, recompute the target.
        const drivesDerived = Object.values(derived).some((r) => r.from === spec.name);
        if (drivesDerived) {
            const prev = w.callback;
            w.callback = function (v) {
                if (typeof prev === "function") prev.call(this, v);
                applyDerived(node, derived);
            };
        }
    }
    applyDerived(node, derived);

    // addCustomWidget doesn't resize the node (unlike addWidget, which calls
    // expandToFitContent), so force a re-layout to grow the node enough to draw
    // the newly inserted rows.
    relayoutNode(node);
}

// Minimum on-canvas size for the node (matches the floor set in onNodeCreated
// so the DOM player keeps usable room).
const MIN_NODE_W = 420;
const MIN_NODE_H = 360;

// Force a node to recompute its size so freshly added widgets actually draw.
// computeSize() sums every *visible* widget row (the player included, now that
// codec widgets sit ahead of it), so we take its height as authoritative rather
// than clamping up from the current height — clamping is what previously pinned
// the node to the player's size and left no room for the codec rows. We keep a
// hard floor (MIN_NODE_*) so removing all codec widgets can't shrink the node
// below a usable player, and grow-only on width to preserve a manual resize.
function relayoutNode(node) {
    try {
        if (typeof node.computeSize === "function") {
            const computed = node.computeSize();
            const w = Math.max(node.size?.[0] || 0, computed[0], MIN_NODE_W);
            const h = Math.max(computed[1] || 0, MIN_NODE_H);
            if (typeof node.setSize === "function") node.setSize([w, h]);
            else node.size = [w, h];
        } else if (typeof node.expandToFitContent === "function") {
            node.expandToFitContent();
        }
    } catch (e) {
        console.warn("[Bat_VideoCombine] relayout failed:", e);
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
}

// Hook the node's "format" COMBO so changing it rebuilds the codec widgets, and
// do the initial build. The widget is created by ComfyUI's own node setup, which
// across frontend versions may finish just after our onNodeCreated runs — so we
// retry on animation frames until it exists (bounded), then wire it exactly once.
function wireFormatWidget(node, attempt) {
    attempt = attempt || 0;
    const formatWidget = node.widgets?.find((w) => w.name === "format");
    if (!formatWidget) {
        if (attempt < 20) {
            const raf = (typeof requestAnimationFrame === "function")
                ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
            raf(() => wireFormatWidget(node, attempt + 1));
        } else {
            console.warn("[Bat_VideoCombine] 'format' widget never appeared; codec widgets disabled.");
        }
        return;
    }
    if (formatWidget[CODEC_WIDGET_FLAG + "_wired"]) return;  // hook once
    formatWidget[CODEC_WIDGET_FLAG + "_wired"] = true;
    console.info("[Bat_VideoCombine] wired format widget, value =", formatWidget.value,
                 "| addCustomWidget?", typeof node.addCustomWidget);

    const prevCb = formatWidget.callback;
    formatWidget.callback = function (value, ...rest) {
        if (typeof prevCb === "function") prevCb.call(this, value, ...rest);
        rebuildCodecWidgets(node, value);
    };
    rebuildCodecWidgets(node, formatWidget.value);
}

function fmtTime(secs) {
    if (!isFinite(secs) || secs < 0) secs = 0;
    const cs = Math.floor((secs - Math.floor(secs)) * 100);
    const s  = Math.floor(secs) % 60;
    const m  = Math.floor(secs / 60);
    return `${m}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

// ── last-preview persistence ────────────────────────────────────────────
// `state.preview` used to be populated ONLY from onExecuted, and the DOM player
// widget is serialize:false — so switching ComfyUI tabs (which tears the widget
// down and re-runs onConfigure) left the node with no <video src> and the player
// went black until the graph was re-run. The encoded file is still sitting in
// the output dir, so all we actually lost was the small JSON reference to it.
//
// We stash that reference in localStorage (per-machine, like the roto/anim-crop
// bg-thumb caches) rather than in a serialised widget: Bat_VideoCombine restores
// its codec widgets POSITIONALLY from widgets_values (see onConfigure), so adding
// another serialised widget would shift that offset and break every saved
// workflow.
//
// The key includes the workflow identity, not just node.id. Keying on node id
// alone (as the roto/anim-crop caches do) collides across workflows — node 14 in
// another graph would restore this graph's video.
function _vcPreviewCacheKey(node) {
    // Shared helper so all four editors scope their caches identically.
    return batNodeCacheKey(app, "bat_vc_preview", node);
}

function _vcSavePreview(node, preview) {
    try {
        if (!preview) localStorage.removeItem(_vcPreviewCacheKey(node));
        else localStorage.setItem(_vcPreviewCacheKey(node), JSON.stringify(preview));
    } catch (_) { /* quota exceeded / storage disabled — non-fatal */ }
}

function _vcLoadPreview(node) {
    try {
        const raw = localStorage.getItem(_vcPreviewCacheKey(node));
        if (!raw) return null;
        const p = JSON.parse(raw);
        // Minimum viable payload — anything less can't build a URL.
        return (p && typeof p.filename === "string" && p.filename) ? p : null;
    } catch (_) { return null; }
}

function buildPreviewUrl(preview) {
    // Non-browser-playable formats go through the on-demand transcoder.
    const route = preview.browser_playable === false ? "/bat/video/preview" : "/view";
    const params = new URLSearchParams({
        filename: preview.filename,
        type:     preview.type || "output",
        subfolder: preview.subfolder || "",
    });
    return api.apiURL(`${route}?${params.toString()}`);
}

function buildPlayer(node) {
    const root = document.createElement("div");
    root.className = "bat-vc-root";
    root.tabIndex = 0;   // focusable so keyboard shortcuts can target it.
    root.style.cssText = `
        position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px;
        overflow:hidden; outline:none; min-height:240px;
        font:11px sans-serif; color:#cde;
    `;

    // ── video element ─────────────────────────────────────────────────
    const videoEl = document.createElement("video");
    videoEl.muted = true;
    videoEl.loop = true;
    videoEl.playsInline = true;
    videoEl.style.cssText = "flex:1; min-height:120px; width:100%; background:#000; display:block;";
    root.appendChild(videoEl);

    // ── hidden decoder used to grab hover-thumbnails ──────────────────
    // Decoupled from the visible player so the user can scrub the thumb
    // without disturbing playback. Same src as the main video; seek to
    // a hover time, grab the frame via canvas.drawImage, surface as a
    // data URL on the hoverThumb <img>. This replaces the previous
    // server-side /bat/video/frame fetch — zero network, sub-50 ms when
    // the encode uses all-I-frames (which it now does for H264/H265/VP9;
    // see the matching `-g 1` change in bat_video_formats/).
    const thumbVideo = document.createElement("video");
    thumbVideo.muted = true;
    thumbVideo.playsInline = true;
    thumbVideo.preload = "auto";
    thumbVideo.style.cssText = "display:none;";
    root.appendChild(thumbVideo);
    const thumbCanvas = document.createElement("canvas");

    // ── timeline ──────────────────────────────────────────────────────
    // height is set dynamically via _syncResponsiveSizes (below) — scales
    // mildly with the node width so a wide review node gets a chunkier
    // scrub bar that's easier to click.
    const timeline = document.createElement("div");
    timeline.className = "bat-vc-timeline";
    timeline.style.cssText = `
        position:relative; background:#1a1d22;
        border-top:1px solid #222; border-bottom:1px solid #222;
        cursor:pointer; user-select:none;
    `;
    const progress = document.createElement("div");
    progress.style.cssText = "position:absolute; left:0; top:0; bottom:0; width:0; background:rgba(76,158,255,0.45);";
    timeline.appendChild(progress);

    const playhead = document.createElement("div");
    playhead.style.cssText = "position:absolute; top:-2px; bottom:-2px; width:2px; background:#7ab8ff; pointer-events:none;";
    timeline.appendChild(playhead);

    const inPin = document.createElement("div");
    inPin.title = "Loop start — drag to move, right-click to clear";
    inPin.style.cssText = "position:absolute; top:-3px; bottom:-3px; width:6px; background:#f4b860; border-radius:1px; cursor:ew-resize; display:none;";
    timeline.appendChild(inPin);

    const outPin = document.createElement("div");
    outPin.title = "Loop end — drag to move, right-click to clear";
    outPin.style.cssText = "position:absolute; top:-3px; bottom:-3px; width:6px; background:#f4b860; border-radius:1px; cursor:ew-resize; display:none;";
    timeline.appendChild(outPin);

    const hoverThumb = document.createElement("img");
    // Width is set dynamically (see _syncResponsiveSizes below) — it
    // scales with the node width so the thumb stays usefully sized on
    // both a 420 px starter node and a stretched 1200 px review node.
    hoverThumb.style.cssText = `
        position:absolute; bottom:18px; left:0; height:auto;
        border:1px solid #4c9eff; border-radius:3px; background:#000;
        box-shadow:0 4px 12px rgba(0,0,0,0.6); pointer-events:none;
        display:none; transform:translateX(-50%);
    `;
    timeline.appendChild(hoverThumb);

    root.appendChild(timeline);

    // ── controls row ──────────────────────────────────────────────────
    const controls = document.createElement("div");
    controls.style.cssText = `
        display:flex; align-items:center; gap:6px; padding:5px 8px;
        background:#15181d; border-top:1px solid #222;
    `;
    const btn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.title = title;
        b.style.cssText = `
            background:none; border:1px solid #2a2f37; color:#cdd;
            padding:2px 6px; font-size:11px; border-radius:3px; cursor:pointer;
            min-width:22px;
            white-space:nowrap; line-height:1; display:inline-flex;
            align-items:center; justify-content:center;
        `;
        b.onmouseover = () => { b.style.background = "rgba(76,158,255,0.15)"; };
        b.onmouseout  = () => { b.style.background = "none"; };
        return b;
    };

    const playBtn  = btn("▶", "Play / Pause (Space)");
    const stepBack10 = btn("⏪", "Back 10 frames (,)");
    // Mirror the step-1 icons (|◀ and ▶|) so both seek-by-frame buttons
    // visually carry the same "bar against the play direction" affordance.
    const stepBack   = btn("|◀", "Back 1 frame (←)");
    const stepFwd    = btn("▶|", "Forward 1 frame (→)");
    const stepFwd10  = btn("⏩", "Forward 10 frames (.)");

    const speedSel = document.createElement("select");
    speedSel.title = "Playback speed";
    speedSel.style.cssText = "background:#1a1d22; color:#cdd; border:1px solid #2a2f37; border-radius:3px; font-size:11px; padding:1px 2px;";
    for (const v of [0.25, 0.5, 1, 2]) {
        const o = document.createElement("option");
        o.value = String(v);
        o.textContent = `${v}×`;
        if (v === 1) o.selected = true;
        speedSel.appendChild(o);
    }

    const muteBtn = btn("🔊", "Mute (M)");
    muteBtn.style.display = "none";
    const volSlider = document.createElement("input");
    volSlider.type = "range";
    volSlider.min = "0"; volSlider.max = "100"; volSlider.value = "100";
    volSlider.title = "Volume";
    volSlider.style.cssText = "width:60px; display:none;";

    const timeLabel = document.createElement("span");
    timeLabel.style.cssText = "flex:1; text-align:center; font-family:monospace; color:#9ab;";
    timeLabel.textContent = "0:00.00 · frame 0 / 0";

    const saveFrameBtn = btn("📷", "Save current frame as PNG");
    const fullscreenBtn = btn("⛶", "Fullscreen (F)");

    // Play sits CENTERED between the seek pair — the conventional video-
    // player layout. Reading left-to-right: ⏪ |◀ ▶ ▶| ⏩ · speed · audio
    // · time · 📷 ⛶
    controls.append(
        stepBack10, stepBack, playBtn, stepFwd, stepFwd10,
        speedSel, muteBtn, volSlider, timeLabel, saveFrameBtn, fullscreenBtn,
    );
    root.appendChild(controls);

    // ── status row (filename + transcode hint) ───────────────────────
    const statusRow = document.createElement("div");
    statusRow.style.cssText = "padding:4px 8px; font-size:10px; color:#789; background:#0d0f12; border-top:1px solid #222; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;";
    statusRow.textContent = "Run the workflow to populate the preview.";
    root.appendChild(statusRow);

    // ── state ─────────────────────────────────────────────────────────
    const state = {
        preview: null,     // last onExecuted payload
        fps: 24,           // from /bat/video/meta or preview.frame_rate
        frameCount: 0,
        hasAudio: false,
        loopIn: null,      // seconds, or null
        loopOut: null,
        // Hover-thumb coalescing — only the latest requested time gets
        // resolved. See _drawThumbAtPendingTime / thumbVideo "seeked".
        thumbPendingTime: null,
        thumbSeeking: false,
        // Frame-stepping uses the EXACT presentation time of the displayed
        // frame (via rVFC's metadata.mediaTime) as the anchor for stepBack
        // / stepFwd. pendingTarget holds the ABSOLUTE target frame index
        // the user has clicked toward but the browser hasn't yet drawn;
        // it stays set until displayedMediaTime confirms we landed on it.
        // Using an absolute pending target (vs. a delta counter that
        // resets on every rVFC tick) avoids two races:
        //  - rVFC firing for the still-composited OLD frame mid-seek,
        //    which would reset the delta and make the next click do
        //    nothing.
        //  - rVFC firing for an INTERMEDIATE frame the browser drew
        //    while servicing a multi-frame seek, which would cause the
        //    next click to anchor too far forward (+2 in one click).
        displayedMediaTime: 0,
        pendingTarget: null,
    };
    node._batVCState = state;

    // ── responsive sizing ────────────────────────────────────────────
    // Recompute the timeline height + hover-thumb width whenever the
    // node resizes. Width-driven because the timeline always spans the
    // node body via flex — height/width track that span proportionally,
    // clamped so we don't get extremes at either end.
    function _syncResponsiveSizes() {
        const w = root.getBoundingClientRect().width || 420;
        const tlH = Math.round(Math.max(12, Math.min(22, 10 + w / 75)));
        timeline.style.height = tlH + 'px';
        // Hover-thumb gets ~28% of the timeline width, clamped to a
        // readable range. Stored on the element so the hover handler
        // only writes back to the DOM when the value actually changes.
        const tw = Math.round(Math.max(80, Math.min(220, w * 0.28)));
        if (hoverThumb._lastWidth !== tw) {
            hoverThumb.style.width = tw + 'px';
            hoverThumb._lastWidth = tw;
        }
    }
    _syncResponsiveSizes();
    // ResizeObserver runs on the next animation frame, batches well, and
    // doesn't require us to hook into LiteGraph's onResize. Kept on the
    // node so we can disconnect if the node ever exposes a teardown hook.
    try {
        // The teardown hook this comment anticipated now exists — batTrack
        // disconnects the observer (and pauses/releases the <video>) when the
        // node is removed.
        const track = batTrack(node);
        track.observer(new ResizeObserver(_syncResponsiveSizes), root);
        node._batVCResizeObserver = null;   // superseded by the tracker
        track.dispose(() => {
            // Release the media element so a deleted node stops buffering and
            // drops its decoded frames.
            try {
                videoEl.pause();
                videoEl.removeAttribute("src");
                videoEl.load();
                thumbVideo.pause();
                thumbVideo.removeAttribute("src");
                thumbVideo.load();
            } catch (_) {}
            state.preview = null;
        });
    } catch (_) { /* very old browsers — sizes stay at the initial values */ }

    // ── derived helpers ──────────────────────────────────────────────
    // Source-of-truth for "what frame is on screen right now?" Prefer the
    // exact presentation-time PTS of the just-composited frame (from
    // rVFC's metadata.mediaTime, tracked into state.displayedMediaTime).
    // Falling back to videoEl.currentTime is hazardous: after a
    // _stepByFrames seek currentTime holds (frame + 0.5)/fps — the
    // mid-frame target we asked for — which Math.round pushes to
    // (frame + 1). Using floor (since frame N occupies [N/fps,
    // (N+1)/fps)) keeps the readout honest even on the brief window
    // before the first rVFC tick fires.
    const _displayedTime = () => {
        if (typeof state.displayedMediaTime === "number" && state.displayedMediaTime > 0) {
            return state.displayedMediaTime;
        }
        return videoEl.currentTime || 0;
    };
    const currentFrame = () => {
        if (!state.fps) return 0;
        // +1e-6 nudges a PTS that sits exactly on a frame boundary (e.g.
        // 5/24 = 0.20833…) past floating-point noise to land on the
        // expected frame index.
        const f = Math.floor(_displayedTime() * state.fps + 1e-6);
        return Math.max(0, Math.min((state.frameCount || 1) - 1, f));
    };
    const snapToFrame = (frame) => {
        if (!state.fps) return;
        const f = Math.max(0, Math.min(state.frameCount - 1, frame));
        videoEl.currentTime = (f + 0.5) / state.fps;   // mid-frame for accuracy
    };
    const updateTimeLabel = () => {
        const f = currentFrame();
        const total = state.frameCount || "?";
        // Time readout also rides _displayedTime so the centiseconds
        // match the frame the user is looking at — otherwise mid-frame
        // currentTime values produced 0.04s of "off" on the readout.
        timeLabel.textContent = `${fmtTime(_displayedTime())} · frame ${f + 1} / ${total}`;
    };

    // ── controls wiring ──────────────────────────────────────────────
    playBtn.onclick = () => {
        if (videoEl.paused) videoEl.play().catch(() => {});
        else videoEl.pause();
    };
    // Play vs pause icons must read at the same visual weight. The
    // Unicode `⏸` (U+23F8 "Double Vertical Bar") is from a different
    // block than `▶` and most system fonts render it noticeably
    // smaller / lighter / greyer (especially macOS). Two HEAVY VERTICAL
    // BARs (U+275A) sit in the Dingbats block alongside other heavy
    // glyphs and match the `▶` triangle's weight when stacked.
    videoEl.addEventListener("play",  () => { playBtn.textContent = "❚❚"; });
    videoEl.addEventListener("pause", () => {
        playBtn.textContent = "▶";
        // User-initiated pause (or our own pause from _stepByFrames):
        // re-anchor on the current time so the FIRST backward step uses
        // fresh data on Firefox, where rVFC doesn't track playback.
        // Don't clobber pendingTarget here — _stepByFrames might have
        // just set it on the same turn of the event loop.
        if (state.pendingTarget == null) {
            state.displayedMediaTime = videoEl.currentTime;
        }
    });

    // Convert a PTS in seconds into the displayed-frame index. floor()
    // because frame N occupies [N/fps, (N+1)/fps); +1e-6 nudges past
    // floating-point noise on exact frame boundaries.
    function _frameForTime(t) {
        if (!state.fps) return 0;
        return Math.max(0, Math.min(
            (state.frameCount || 1) - 1,
            Math.floor((t || 0) * state.fps + 1e-6),
        ));
    }

    // Explicit absolute seek to a frame, with all step-state bookkeeping
    // done UP FRONT — before the browser even starts the seek. Setting
    // displayedMediaTime and the label eagerly means:
    //   • the on-screen frame counter updates the moment the user clicks,
    //     not after the seek completes a few ms later;
    //   • the NEXT click anchors on the just-requested frame, so two
    //     rapid clicks always result in two frames of motion;
    //   • an rVFC tick that fires for the still-composited OLD frame
    //     between click and seek-completion can't drag displayedMediaTime
    //     backwards (the rVFC handler ignores reports that don't match
    //     pendingTarget — see below).
    // If the seek physically lands on a DIFFERENT frame (broken codec on
    // legacy non-I-frame video, etc.), the seeked handler reconciles by
    // overwriting displayedMediaTime with the actual landed frame.
    function _seekToFrame(newFrame) {
        if (!state.fps) return;
        const fps = state.fps;
        const maxFrame = Math.max(0, (state.frameCount || 1) - 1);
        newFrame = Math.max(0, Math.min(maxFrame, newFrame));
        state.pendingTarget = newFrame;
        state.displayedMediaTime = newFrame / fps;
        const target = (newFrame + 0.5) / fps;
        _paintPlayhead(target);
        updateTimeLabel();
        videoEl.currentTime = target;
    }

    // Step by N frames. The anchor is the LAST REQUESTED target frame
    // if a seek is still in flight (pendingTarget) — otherwise the
    // last-displayed frame. This is what makes rapid clicks count even
    // when rVFC is firing mid-seek for stale frames.
    function _stepByFrames(delta) {
        if (!state.fps) return;
        videoEl.pause();
        const anchorFrame = (state.pendingTarget != null)
            ? state.pendingTarget
            : _frameForTime(state.displayedMediaTime);
        _seekToFrame(anchorFrame + delta);
    }
    stepBack.onclick    = () => _stepByFrames(-1);
    stepFwd.onclick     = () => _stepByFrames(+1);
    stepBack10.onclick  = () => _stepByFrames(-10);
    stepFwd10.onclick   = () => _stepByFrames(+10);

    speedSel.onchange = () => { videoEl.playbackRate = parseFloat(speedSel.value) || 1; };

    muteBtn.onclick = () => {
        videoEl.muted = !videoEl.muted;
        muteBtn.textContent = videoEl.muted ? "🔇" : "🔊";
    };
    volSlider.oninput = () => {
        videoEl.volume = parseInt(volSlider.value, 10) / 100;
        if (videoEl.volume > 0 && videoEl.muted) {
            videoEl.muted = false;
            muteBtn.textContent = "🔊";
        }
    };

    fullscreenBtn.onclick = () => {
        if (document.fullscreenElement) document.exitFullscreen();
        else videoEl.requestFullscreen?.();
    };

    // Save the CURRENTLY-DISPLAYED frame as a PNG straight into the
    // browser's Downloads folder. Client-side: draw the visible <video>
    // into a same-size canvas, toBlob → object URL → synthetic anchor
    // click with the `download` attribute. No server round-trip.
    //
    // (The previous handler POSTed to /bat/video/save_frame which wrote
    // a file into ComfyUI's output dir — useful as a server-side artefact
    // but artists asked for the browser download instead. The server
    // endpoint is left intact for any other caller; this UI just stops
    // using it.)
    saveFrameBtn.onclick = () => {
        if (!videoEl.videoWidth || !videoEl.videoHeight) {
            statusRow.textContent = "Save frame: video not ready yet.";
            return;
        }
        saveFrameBtn.disabled = true;
        const canvas = document.createElement("canvas");
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        try {
            canvas.getContext("2d").drawImage(videoEl, 0, 0);
        } catch (e) {
            statusRow.textContent = `Save frame failed: ${e.message}`;
            saveFrameBtn.disabled = false;
            return;
        }
        canvas.toBlob((blob) => {
            if (!blob) {
                statusRow.textContent = "Save frame failed: empty blob.";
                saveFrameBtn.disabled = false;
                return;
            }
            const baseName = ((state.preview && state.preview.filename) || "frame")
                .replace(/\.[^./\\]+$/, "");
            const frameNum = String(currentFrame() + 1).padStart(5, "0");
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${baseName}_f${frameNum}.png`;
            // Anchor must be in the DOM for the synthetic click to dispatch
            // a download in Firefox. Chromium tolerates a detached anchor
            // but the in-DOM path works everywhere.
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            // Browsers start the download synchronously here, but the blob
            // URL has to stay valid until the file is actually written —
            // revoking too quickly corrupts the download in Chromium.
            setTimeout(() => URL.revokeObjectURL(url), 2000);
            statusRow.textContent = `Downloaded ${a.download} to your Downloads folder.`;
            saveFrameBtn.disabled = false;
        }, "image/png");
    };

    // ── timeline interaction ─────────────────────────────────────────
    const timelinePosToTime = (clientX) => {
        const r = timeline.getBoundingClientRect();
        const x = Math.max(0, Math.min(r.width, clientX - r.left));
        return videoEl.duration ? (x / r.width) * videoEl.duration : 0;
    };
    const timeToPctStr = (t) => {
        if (!videoEl.duration) return "0%";
        return `${(t / videoEl.duration) * 100}%`;
    };

    let dragging = null;   // null | "scrub" | "in" | "out"
    timeline.addEventListener("pointerdown", (e) => {
        if (e.button === 2) return;
        if (e.target === inPin)       dragging = "in";
        else if (e.target === outPin) dragging = "out";
        else {
            dragging = "scrub";
            // A deliberate scrub invalidates whatever step click was
            // mid-seek — otherwise the next stepBack/Fwd would anchor
            // on the pre-scrub target and either skip or fail to move.
            state.pendingTarget = null;
        }
        timeline.setPointerCapture(e.pointerId);
        handleTimelineMove(e);
    });
    timeline.addEventListener("pointermove", (e) => {
        if (dragging) handleTimelineMove(e);
        else          handleTimelineHover(e);
    });
    timeline.addEventListener("pointerup", (e) => {
        const wasScrub = (dragging === "scrub");
        dragging = null;
        try { timeline.releasePointerCapture(e.pointerId); } catch (_) {}
        if (wasScrub) {
            // After a fastSeek-driven drag the video may have landed on
            // the wrong frame (nearest keyframe rather than the exact
            // target). Snap precisely now that the user has settled.
            const f = (state.scrubPendingFrame != null)
                ? state.scrubPendingFrame
                : Math.round(timelinePosToTime(e.clientX) * state.fps);
            state.scrubPendingFrame = null;
            _seekToFrame(f);
        }
    });
    timeline.addEventListener("pointercancel", () => { dragging = null; });
    timeline.addEventListener("pointerleave", () => {
        hoverThumb.style.display = "none";
    });
    // Right-click on a pin clears it.
    timeline.addEventListener("contextmenu", (e) => {
        if (e.target === inPin)  { state.loopIn  = null; inPin.style.display  = "none"; e.preventDefault(); }
        if (e.target === outPin) { state.loopOut = null; outPin.style.display = "none"; e.preventDefault(); }
    });

    function handleTimelineMove(e) {
        const t = timelinePosToTime(e.clientX);
        if (dragging === "scrub") {
            // The playhead repaints + the time label update immediately
            // off the requested time, so the cursor glides with the
            // mouse even while the video decoder is still chewing on
            // the previous seek.
            _paintPlayhead(t);
            const f = Math.max(0, Math.min((state.frameCount || 1) - 1,
                                            Math.round(t * state.fps)));
            state.displayedMediaTime = f / state.fps;
            updateTimeLabel();
            // Coalesce seeks — only ONE seek in flight at a time.
            // pointermove can fire ~120 Hz on a high-DPI trackpad, but
            // each currentTime= write triggers a decoder seek that may
            // take 10-30ms even for all-I-frame encodes; firing one
            // per move event piles up backlog and the playhead lags
            // the cursor. Instead we stash the latest requested frame
            // and only kick off the next seek when the previous one
            // resolves (see the `seeked` handler).
            state.scrubPendingFrame = f;
            if (!state.scrubSeeking) _drainScrub();
        } else if (dragging === "in") {
            state.loopIn = t;
            inPin.style.left = timeToPctStr(t);
        } else if (dragging === "out") {
            state.loopOut = t;
            outPin.style.left = timeToPctStr(t);
        }
    }

    function _drainScrub() {
        if (state.scrubPendingFrame == null) return;
        const f = state.scrubPendingFrame;
        state.scrubPendingFrame = null;
        state.scrubSeeking = true;
        // Optimistically anchor stepping math on the requested frame —
        // a follow-up frame-step click should pick up where the scrub
        // visually landed, not on the previous video position.
        state.pendingTarget = f;
        try {
            // fastSeek hints to the browser to seek to the nearest
            // keyframe — for all-I-frame encodes that's exact, for
            // long-GOP encodes it's an approximation that resolves
            // far quicker than currentTime=. We snap to exact frame on
            // pointerup (see below) for the final position.
            if (typeof videoEl.fastSeek === "function") {
                videoEl.fastSeek((f + 0.5) / state.fps);
            } else {
                videoEl.currentTime = (f + 0.5) / state.fps;
            }
        } catch (_) {
            state.scrubSeeking = false;
            state.scrubPendingFrame = f;
        }
    }

    // Single source of truth for the playhead + progress geometry.
    // Called from rVFC during playback, from the drag handler for an
    // immediate-response scrub, and from timeupdate as a Firefox fallback.
    function _paintPlayhead(t) {
        const pct = timeToPctStr(t);
        progress.style.width = pct;
        playhead.style.left = pct;
    }

    function handleTimelineHover(e) {
        if (!state.preview || !state.fps || !state.frameCount) return;
        if (!thumbVideo.duration) return;   // hidden decoder still loading
        const t = timelinePosToTime(e.clientX);
        const r = timeline.getBoundingClientRect();
        hoverThumb.style.left = `${e.clientX - r.left}px`;
        // Show the container immediately — the image populates from the
        // previous successful draw while the new seek completes, so the
        // user never sees a black flash.
        hoverThumb.style.display = "block";

        // Coalesce: stash the latest requested time. If a seek is already
        // in flight, _drawThumbAfterSeek picks up the pending value when
        // it finishes. This means rapid mouse-moves don't queue up N seeks
        // — only the most recent target ever gets resolved.
        state.thumbPendingTime = (t + 0.5 / (state.fps || 24));   // mid-frame, like snapToFrame
        if (!state.thumbSeeking) _drawThumbAtPendingTime();
    }

    function _drawThumbAtPendingTime() {
        const t = state.thumbPendingTime;
        if (t == null) return;
        state.thumbPendingTime = null;
        state.thumbSeeking = true;
        try {
            thumbVideo.currentTime = t;
        } catch (_) {
            state.thumbSeeking = false;
        }
    }

    thumbVideo.addEventListener("seeked", () => {
        // Drop the rendered frame into a canvas sized to the visible
        // hoverThumb. Aspect ratio is taken from the source — falls back
        // to 16:9 if videoWidth/Height haven't populated yet.
        try {
            const w = hoverThumb._lastWidth || 160;
            const aspect = (thumbVideo.videoWidth && thumbVideo.videoHeight)
                ? (thumbVideo.videoHeight / thumbVideo.videoWidth)
                : 0.5625;
            const h = Math.max(1, Math.round(w * aspect));
            if (thumbCanvas.width !== w)  thumbCanvas.width  = w;
            if (thumbCanvas.height !== h) thumbCanvas.height = h;
            const ctx = thumbCanvas.getContext("2d");
            ctx.drawImage(thumbVideo, 0, 0, w, h);
            // JPEG keeps the data URL short — PNG would balloon for 200 px
            // thumbnails. Quality 0.78 is the sweet spot for screen-sized
            // preview tiles; visible artefacts only appear below ~0.6.
            hoverThumb.src = thumbCanvas.toDataURL("image/jpeg", 0.78);
        } catch (_) { /* tainted canvas / decoder failure — keep old thumb */ }
        state.thumbSeeking = false;
        // Chase the latest pending position if the user kept moving.
        if (state.thumbPendingTime != null) _drawThumbAtPendingTime();
    });

    // ── keyboard shortcuts ────────────────────────────────────────────
    root.addEventListener("keydown", (e) => {
        // Don't steal keystrokes that belong to a text input inside us.
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
        let handled = true;
        switch (e.key) {
            case " ":   playBtn.click(); break;
            case "ArrowLeft":  stepBack.click(); break;
            case "ArrowRight": stepFwd.click(); break;
            case ",":   stepBack10.click(); break;
            case ".":   stepFwd10.click(); break;
            case "Home": _seekToFrame(0); break;
            case "End":  _seekToFrame((state.frameCount || 1) - 1); break;
            case "f": case "F": fullscreenBtn.click(); break;
            case "m": case "M": if (state.hasAudio) muteBtn.click(); break;
            case "Escape":
                state.loopIn = state.loopOut = null;
                inPin.style.display = outPin.style.display = "none";
                break;
            default: handled = false;
        }
        if (handled) {
            // stopPropagation prevents LiteGraph's document-level keydown
            // listener from also acting on this event — without it, ←/→
            // would step a frame here AND simultaneously navigate to the
            // previous/next node on the canvas. Matches the pattern in
            // sibling bat_video_loader.js.
            e.preventDefault();
            e.stopPropagation();
        }
    });

    // Auto-focus the player root on any click inside it so the keyboard
    // handler above starts receiving events without the artist needing
    // to Tab in. Native button focus (e.g. on Play) stays where it lands
    // — we only steal focus when the click goes to a non-focusable child
    // (video element, timeline, empty player background).
    root.addEventListener("mousedown", () => {
        setTimeout(() => {
            const a = document.activeElement;
            if (!root.contains(a) || a === document.body) {
                root.focus({ preventScroll: true });
            }
        }, 0);
    });

    // ── per-frame loop enforcement + playhead repaint ────────────────
    // Loop enforcement (in/out pins) lives in timeupdate so it fires
    // even if the rVFC callback is throttled away during background tabs.
    videoEl.addEventListener("timeupdate", () => {
        if (state.loopIn != null && state.loopOut != null
                && state.loopOut > state.loopIn
                && videoEl.currentTime >= state.loopOut) {
            // Loop wraparound is a non-step seek — abandon any pending
            // step target so the next click anchors on reality, not on
            // wherever the pre-loop click was heading.
            state.pendingTarget = null;
            videoEl.currentTime = state.loopIn;
        }
        // Firefox fallback for the playhead update — when rVFC isn't
        // available we depend on timeupdate's ~4 Hz cadence.
        if (!videoEl.requestVideoFrameCallback) {
            _paintPlayhead(videoEl.currentTime);
            updateTimeLabel();
        }
    });
    videoEl.addEventListener("seeked", () => {
        // After any seek (drag-scrub, frame step, in/out enforcement),
        // make sure the playhead reflects where the video actually
        // landed. _paintPlayhead is idempotent, so calling it during
        // drag-scrub on top of the drag handler's call is harmless.
        _paintPlayhead(videoEl.currentTime);
        // If a scrub coalesced another target while this seek was in
        // flight, kick off the next one. Done BEFORE the pendingTarget
        // reconcile below so the in-flight chain stays alive.
        state.scrubSeeking = false;
        if (state.scrubPendingFrame != null) _drainScrub();
        const actualFrame = _frameForTime(videoEl.currentTime);
        if (state.pendingTarget != null) {
            // Reconcile the eager update from _seekToFrame against
            // what the browser actually landed on. Matching pendingTarget
            // confirms the optimistic state — keep displayedMediaTime at
            // the exact-PTS value we set it to (more precise than the
            // (N+0.5)/fps mid-frame target that currentTime holds).
            // Mismatch means the seek didn't make it (legacy non-I-frame
            // encode, codec quirk) — sync to the truth so the label
            // doesn't lie.
            if (actualFrame !== state.pendingTarget) {
                state.displayedMediaTime = actualFrame / state.fps;
            }
            state.pendingTarget = null;
        } else {
            // Non-step seek (drag-scrub release, loop wraparound, etc.):
            // just sync to where we landed.
            state.displayedMediaTime = actualFrame / state.fps;
        }
        updateTimeLabel();
    });
    videoEl.addEventListener("loadedmetadata", () => {
        // Reset the stepping anchor when a fresh source loads.
        state.displayedMediaTime = videoEl.currentTime || 0;
        state.pendingTarget = null;
        updateTimeLabel();
    });

    // requestVideoFrameCallback gives us a paint-aligned tick for
    // EVERY displayed video frame — typically 24/30/60 Hz depending
    // on the source — vs timeupdate's ~4 Hz. The result is a playhead
    // that glides instead of stepping.
    //
    // We also use this as the truth source for frame stepping:
    // metadata.mediaTime is the EXACT presentation timestamp of the
    // frame the browser just composited. Anchoring stepBack/stepFwd
    // on this value (rather than re-rounding currentTime) makes
    // backward stepping reliable on intra-only encodes.
    if (typeof videoEl.requestVideoFrameCallback === "function") {
        const onVideoFrame = (_now, metadata) => {
            if (metadata && typeof metadata.mediaTime === "number") {
                if (state.pendingTarget == null) {
                    // No step in flight — rVFC is the truth source for
                    // playback / drag-scrub. Accept the update.
                    state.displayedMediaTime = metadata.mediaTime;
                } else if (_frameForTime(metadata.mediaTime) === state.pendingTarget) {
                    // The seek has confirmed our target. Lock in the
                    // exact PTS and release the pending flag.
                    state.displayedMediaTime = metadata.mediaTime;
                    state.pendingTarget = null;
                }
                // The OTHER case — pendingTarget set but reported frame
                // doesn't match — is an rVFC tick for the still-composited
                // OLD frame, or an intermediate frame composited mid-seek.
                // Drop it on the floor; the eager update from _seekToFrame
                // is what we want to keep showing until the real seek
                // catches up.
            }
            _paintPlayhead(videoEl.currentTime);
            updateTimeLabel();
            // The callback fires once per displayed frame; re-arm it on
            // EVERY tick (even while paused) so a manual seek or a frame
            // step still gets its repaint. The browser only schedules the
            // next tick once the next frame is composited, so there's no
            // CPU cost when paused.
            videoEl.requestVideoFrameCallback(onVideoFrame);
        };
        videoEl.requestVideoFrameCallback(onVideoFrame);
    }

    // ── exposed API for onExecuted ───────────────────────────────────
    // `restoring` is true when we're re-applying a cached preview after a tab
    // switch / workflow reload rather than reacting to a fresh encode. In that
    // case we don't autoplay (the artist didn't just ask for this) and we don't
    // re-write the cache entry we just read.
    async function loadPreview(preview, restoring = false) {
        state.preview = preview;
        if (!restoring) _vcSavePreview(node, preview);
        // Seed metadata from the encode payload — fast path so the
        // counter shows the right total before /bat/video/meta resolves.
        state.fps = preview.frame_rate || 24;
        state.frameCount = preview.frame_count || 0;
        state.hasAudio = false;
        state.thumbPendingTime = null;
        state.thumbSeeking = false;
        muteBtn.style.display = volSlider.style.display = "none";
        // Drop any thumb from the previous file so the first hover on
        // the new file doesn't flash an unrelated frame.
        hoverThumb.removeAttribute("src");

        const url = buildPreviewUrl(preview);
        videoEl.src = url;
        videoEl.load();
        // Autoplay only for a fresh encode. On a restore we load the first frame
        // and sit paused, so returning to a tab doesn't start N videos playing.
        if (!restoring) videoEl.play().catch(() => {});
        // Hidden decoder follows the visible video — same URL, same byte
        // range cache, so the browser usually services its seeks from
        // already-buffered data.
        thumbVideo.src = url;
        thumbVideo.load();

        const playableNote = preview.browser_playable === false
            ? "(transcoded preview)"  : "";
        statusRow.textContent = `${preview.filename}  ·  ${preview.format || ""}  ${playableNote}`;

        // Now probe the actual file — ffprobe is the source of truth for
        // fps + frame count (the encode payload guesses based on pingpong
        // arithmetic, but doesn't see what the codec actually wrote).
        try {
            const params = new URLSearchParams({
                filename: preview.filename,
                type:     preview.type || "output",
                subfolder: preview.subfolder || "",
            });
            const resp = await api.fetchApi(`/bat/video/meta?${params.toString()}`);
            if (resp.ok) {
                const meta = await resp.json();
                if (meta.fps)         state.fps         = meta.fps;
                if (meta.frame_count) state.frameCount  = meta.frame_count;
                state.hasAudio = !!meta.has_audio;
                if (state.hasAudio) {
                    muteBtn.style.display = "inline-block";
                    volSlider.style.display = "inline-block";
                }
                updateTimeLabel();
            }
        } catch (_) { /* meta is best-effort */ }
    }

    node._batVCLoadPreview = loadPreview;

    // Re-apply the last preview after a tab switch / workflow reload. Verified
    // against the server first: the cached reference can outlive the file (temp
    // dir cleared, output pruned, encode overwritten), and pointing <video> at a
    // 404 shows the same black player we're trying to fix. A HEAD-ish GET on the
    // meta route is cheap and tells us whether the file is still there.
    node._batVCRestorePreview = async function restorePreview() {
        if (state.preview) return;              // a fresh encode already landed
        const cached = _vcLoadPreview(node);
        if (!cached) return;
        try {
            const params = new URLSearchParams({
                filename: cached.filename,
                type:     cached.type || "output",
                subfolder: cached.subfolder || "",
            });
            const resp = await api.fetchApi(`/bat/video/meta?${params.toString()}`);
            if (!resp.ok) { _vcSavePreview(node, null); return; }
        } catch (_) {
            // Server unreachable — leave the cache alone and skip the restore;
            // a later run (or reload) can try again.
            return;
        }
        if (state.preview) return;              // raced with an encode; it wins
        loadPreview(cached, true);
    };

    // Kick a restore on build. onConfigure also calls this for the
    // workflow-load path, where the cached entry may not be readable until the
    // graph's identity is known.
    setTimeout(() => { node._batVCRestorePreview?.(); }, 0);

    // ResizeObserver isn't necessary — flex layout + native <video> handle
    // the scaling. The DPR pixelation we fixed elsewhere doesn't apply here
    // because we don't render to a canvas at all.

    return root;
}

app.registerExtension({
    name: "Bat_VideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            // The player and the codec widgets are independent — guard the
            // player build so an addDOMWidget signature change in some frontend
            // version can't abort onNodeCreated before the codec widgets wire up.
            try {
                const el = buildPlayer(this);
                // Dual-mode sizing (see bat_node_layout.js). Under Nodes 2.0 the
                // node height comes from computeLayoutSize, so the this.size
                // assignment below is a no-op there and the player would
                // otherwise disagree with the node box.
                addBatDOMWidget(this, "bat_video_player", "bat_video_player", el, {
                    minWidth: MIN_NODE_W, height: MIN_NODE_H, growable: true,
                });
                // Wider than the default to give the controls + scrubber room.
                clampNodeSize(this, MIN_NODE_W, MIN_NODE_H);
            } catch (e) {
                console.error("[Bat_VideoCombine] player widget setup failed:", e);
            }

            // Wire the format COMBO so picking a codec swaps in its knobs, and
            // build the knobs for the current selection right away. onConfigure
            // (below) rebuilds again with saved values when loading a workflow.
            //
            // The "format" widget is created by ComfyUI's own node setup. Across
            // frontend versions that can land slightly after this onNodeCreated
            // hook, so we don't assume it's present yet: wireFormatWidget retries
            // on the next frame until it appears, then hooks it once.
            wireFormatWidget(this);
            return r;
        };

        // On workflow load, ComfyUI restores the static widget values
        // positionally (including "format") and THEN calls onConfigure. At
        // that point the codec widgets don't exist yet, so their saved values
        // sit unused at the tail of info.widgets_values. We rebuild the codec
        // widgets for the restored format and re-apply those tail values in
        // order (their save order equals the format's JSON widget order).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const formatWidget = this.widgets?.find((w) => w.name === "format");
            if (!formatWidget) return r;

            // Static (serialised) widgets currently present — codec widgets
            // aren't built yet, so this count is exactly the static prefix
            // length in the saved widgets_values array.
            const savedVals = Array.isArray(info?.widgets_values) ? info.widgets_values : null;
            const staticCount = this.widgets.filter(
                (w) => w.name && w.options?.serialize !== false
            ).length;

            getFormatSpecs().then((specs) => {
                const def = specs[formatWidget.value];
                let savedByName;
                if (savedVals && def) {
                    // Tail values after the static prefix map to this format's
                    // widgets in JSON order (the same order they serialised in).
                    const tail = savedVals.slice(staticCount);
                    savedByName = {};
                    (def.widgets || []).forEach((spec, i) => {
                        if (i < tail.length) savedByName[spec.name] = tail[i];
                    });
                }
                rebuildCodecWidgets(this, formatWidget.value, savedByName);
            });
            // Re-apply the last preview for this workflow+node. Deferred a tick
            // so the graph's identity (used in the cache key) is settled.
            setTimeout(() => { this._batVCRestorePreview?.(); }, 0);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && Array.isArray(message.gifs) && message.gifs[0]
                    && typeof this._batVCLoadPreview === "function") {
                this._batVCLoadPreview(message.gifs[0]);
            }
            return r;
        };
    },
});
