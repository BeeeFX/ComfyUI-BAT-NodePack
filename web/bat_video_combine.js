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
 *   - View-transform dropdown, EXR sequences only (see below).
 *
 * ProRes / FFV1 / h265 outputs come back with browser_playable=false in
 * the onExecuted payload; for those the <video> src points at the
 * /bat/video/preview endpoint (server-side transcode) instead of /view.
 *
 * Image sequences (EXR / PNG) come back with is_sequence=true and a printf
 * pattern (`%05d.exr`) as their filename. They go down the same transcode
 * route, but a pattern carries no frame rate of its own, so every
 * /bat/video/* call for a sequence must pass `fps` — otherwise ffmpeg's
 * image2 demuxer assumes 25 and the preview plays at the wrong speed. EXR
 * additionally takes `trc`: the node writes the IMAGE tensor into EXR
 * verbatim, so the 1:1 default matches what an h264 export of the same
 * frames looks like, while a genuinely scene-linear render needs sRGB or
 * Rec.709 applied or it reads near-black.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { batTrack, batNodeCacheKey, batReplayLastExecution } from "./bat_lifecycle.js";
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
// one or more visible widgets (profile, bit_depth, alpha) so the user can't pick
// an invalid combo — a codec's pix_fmt list mixes two independent axes (chroma /
// alpha layout and bit depth), and only the axes are worth an artist's attention.
// They're still serialised (the backend re-derives authoritatively regardless).

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
        const values = spec.options || [];
        w.options = { values };
        // A saved value that the format no longer offers must not stick: an
        // option list can change shape between versions (EXR's `compression`
        // listed four names ffmpeg rejects outright), and litegraph would
        // happily keep drawing the stale one until the encode failed.
        const savedIsValid = hasSaved && (!values.length || values.includes(savedValue));
        w.value = savedIsValid ? savedValue : (spec.default ?? values[0]);
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
        // inferSourcesFromDerived hands back map-key STRINGS ("true"/"false"),
        // and a bare `savedValue` would make the string "false" a truthy toggle.
        w.value = hasSaved ? (savedValue === true || savedValue === "true")
                           : !!spec.default;
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

// ── `derived` rule helpers — mirrors of the same-named ones in
//    bat_video_combine.py. Keep the two in step: the backend re-derives
//    authoritatively at encode time, so a divergence shows up as a node whose
//    UI disagrees with the file it just wrote.
const DERIVED_SEP = "|";

// A rule's `from` is a bare name for single-source rules ("profile") and an
// array for composite ones (["layout", "bit_depth"]), whose map keys join the
// source values with `sep`: "yuv444|12".
function derivedSources(rule) {
    const from = rule?.from;
    if (from == null) return [];
    return Array.isArray(from) ? from.slice() : [from];
}

// JSON spells booleans "true"/"false"; the alpha toggles are BOOLEAN widgets,
// so their map keys have to be spelled the same way on both sides.
function mapKey(value) {
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
}

// Apply a format's `derived` rules: set each hidden target widget's value from
// its source widget(s) via the JSON map. Called on build and on source change.
function applyDerived(node, derived) {
    if (!derived) return;
    for (const [target, rule] of Object.entries(derived)) {
        const dst = node.widgets?.find((w) => w.name === target);
        if (!dst) continue;
        const sources = derivedSources(rule);
        const key = sources
            .map((n) => mapKey(node.widgets?.find((w) => w.name === n)?.value))
            .join(rule.sep || DERIVED_SEP);
        const mapped = (rule.map && key in rule.map) ? rule.map[key]
                     : (rule.default ?? dst.value);
        dst.value = mapped;
    }
}

// Back-fill a derived rule's *sources* from a restored *target*. Mirrors
// _infer_sources() in bat_video_combine.py — see that docstring for the why.
//
// Short version: `bit_depth` (and the alpha/layout widgets beside it) were
// appended to the format JSONs after those formats shipped. A workflow saved
// before then has only the old visible `pix_fmt` in its widgets_values tail;
// the new widgets restore as undefined, and re-deriving pix_fmt from their
// defaults would quietly change a saved deliverable's depth (a 10-bit h265
// coming back 8-bit). So run the map backwards to recover what produced it.
function inferSourcesFromDerived(def, savedByName) {
    if (!savedByName || !def || !def.derived) return;
    for (const [target, rule] of Object.entries(def.derived)) {
        const saved = savedByName[target];
        if (saved === undefined || saved === null) continue;
        const sources = derivedSources(rule);
        const missing = sources.filter(
            (n) => savedByName[n] === undefined || savedByName[n] === null);
        if (!sources.length || !missing.length) continue;
        const want = mapKey(saved);
        const hit = Object.entries(rule.map || {}).find(([, v]) => mapKey(v) === want);
        if (!hit) continue;
        const parts = String(hit[0]).split(rule.sep || DERIVED_SEP);
        if (parts.length !== sources.length) continue;
        sources.forEach((n, i) => {
            if (missing.includes(n)) savedByName[n] = parts[i];
        });
    }
}

// ─── Legacy static-widget layout migration ───────────────────────────────────
//
// ComfyUI restores widgets_values POSITIONALLY — the Nth saved value goes to
// the Nth serialisable widget and the names are never consulted. So dropping a
// static widget silently shifts every value after it onto the wrong widget in
// every workflow saved before the change. `loop_count` and `pingpong` went on
// 2026-08-06 (both features were retired), which is exactly two such holes:
//
//   saved by Stable: [24.0, 0, "shot_a", "video/prores-mov", false, true, …codec]
//   restored into:   [frame_rate,  filename_prefix,  format,      save_output,  …codec]
//                     24.0 ✓       0 ✗               "shot_a" ✗   "video/…" ✗
//
// filename_prefix comes back as "0", format falls back to its default (the
// saved prefix is not a valid combo option), save_output is a truthy string,
// and the codec tail is read two slots early so crf/preset/bit_depth land on
// each other. That is the "widgets have the value of the widget above them"
// report from artists opening Stable-era workflows on Beta.
//
// The fix is to work out which historical layout wrote the saved array and
// rewrite it into the current one — matched by NAME, not position — before
// LiteGraph deals it out (see the configure() patch in beforeRegisterNodeDef).
// Codec widgets need no equivalent: every format JSON has only ever had knobs
// *appended*, so their tail still lines up (and the entries added later are
// recovered by inferSourcesFromDerived above).
//
// Layouts are newest-first. Each entry is the exact ordered list of
// serialisable static widgets that version of the node had. Whenever a static
// widget is added, removed or reordered, push the new order on the front and
// LEAVE the old entries — otherwise this bug comes straight back.
const STATIC_LAYOUTS = [
    // Current.
    ["frame_rate", "filename_prefix", "format", "save_output"],
    // Pre-2026-08-06. Still what the Stable and Farm releases ship, so this is
    // the layout nearly every old workflow on disk was written by.
    ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"],
];

// `typeof` each static value as it serialises. This is what tells the layouts
// apart, and it is unambiguous by construction: they differ at index 1
// (`loop_count` is a number, `filename_prefix` is a string), so at most one can
// type-check against any given array. Deliberately not length-based — the array
// also carries a per-format codec tail whose length varies with the format.
const STATIC_TYPES = {
    frame_rate:      "number",
    loop_count:      "number",
    filename_prefix: "string",
    format:          "string",
    pingpong:        "boolean",
    save_output:     "boolean",
};

// Identify which STATIC_LAYOUTS entry wrote a saved widgets_values array.
// Returns null when nothing fits, so the caller can leave the positional
// restore untouched rather than scramble an array it doesn't understand.
//
// `formatLabels` (the format COMBO's current options) is a tie-breaker only,
// never a requirement: a workflow may legitimately name a format we have since
// dropped (`video/av1-webm` did go), and that must not veto a clean type match.
function matchStaticLayout(savedVals, formatLabels) {
    const fits = STATIC_LAYOUTS.filter(
        (layout) => layout.length <= savedVals.length
            && layout.every((name, i) => typeof savedVals[i] === STATIC_TYPES[name]));
    if (!fits.length) return null;
    return fits.find((l) => formatLabels?.includes(savedVals[l.indexOf("format")]))
        || fits[0];
}

// What the Python INPUT_TYPES declares for each static widget: its options (a
// COMBO's choices) and its default. Read off nodeData at registration rather
// than hardcoded here, so the JS can't quietly disagree with the node the day a
// default or a format list moves.
function declaredStatics(nodeData) {
    const out = {};
    for (const [name, spec] of Object.entries(nodeData?.input?.required || {})) {
        if (!Array.isArray(spec)) continue;
        const [type, opts] = spec;
        out[name] = {
            options: Array.isArray(type) ? type : null,
            default: (opts && typeof opts === "object") ? opts.default : undefined,
        };
    }
    return out;
}

// Rewrite a saved widgets_values array from whatever layout wrote it into the
// current one, IN PLACE, so LiteGraph's positional deal lands on the right
// widgets in the first place. Silently does nothing when the array's layout
// isn't recognised — loading a workflow as-saved is a much better failure than
// shuffling an array we don't understand.
function migrateWidgetsValues(info, declared) {
    const saved = info?.widgets_values;
    // Some frontend versions serialise an object keyed by widget name. That
    // form is immune to reordering by construction, so there is nothing to do.
    if (!Array.isArray(saved)) return;

    const current = STATIC_LAYOUTS[0];
    const layout = matchStaticLayout(saved, declared.format?.options);
    if (!layout) {
        console.warn("[Bat_VideoCombine] widgets_values matches no known static layout",
                     "— loading it as saved:", saved);
        return;
    }

    const byName = {};
    layout.forEach((name, i) => { byName[name] = saved[i]; });

    const prefix = current.map((name) => {
        const spec = declared[name] || {};
        let v = byName[name];
        // A static widget added since this workflow was saved has no value in
        // it, so its declared default is the only honest answer.
        if (v === undefined) return spec.default;
        // A COMBO value that is no longer on offer must not survive: formats do
        // get retired (`video/av1-webm` went), and litegraph draws the stale
        // string quite happily right up until the encode fails.
        if (spec.options?.length && !spec.options.includes(v)) {
            const fallback = spec.options.includes(spec.default)
                ? spec.default : spec.options[0];
            console.warn("[Bat_VideoCombine] saved", name, "=", v,
                         "is no longer offered — falling back to", fallback);
            v = fallback;
        }
        return v;
    });

    // The codec tail rides along untouched: every format JSON has only ever had
    // widgets APPENDED, so the saved entries still line up with the current
    // JSON order and onConfigure recovers the ones added since.
    info.widgets_values = prefix.concat(saved.slice(layout.length));

    if (layout !== current) {
        console.info("[Bat_VideoCombine] migrated a workflow saved with the",
                     layout.join("/"), "widget layout");
    }
}

// Rank a bit_depth option so we can pick the deepest one a format offers.
// Options are strings: "8" / "10" / "12" / "16", and EXR's "16f" / "32f".
// Ranking on the leading digits sorts both families correctly (within one
// format's list, which is the only place they are ever compared).
function depthRank(option) {
    const m = String(option).match(/\d+/);
    return m ? Number(m[0]) : 0;
}

function highestDepth(spec) {
    const opts = (spec && spec.options) || [];
    if (!opts.length) return undefined;
    return opts.reduce((best, o) => (depthRank(o) > depthRank(best) ? o : best), opts[0]);
}

// Rebuild the codec widgets for the node's currently-selected format.
// `savedValues` (name→value) lets us restore values after a workflow load.
// `opts.preferMaxBitDepth` picks the deepest bit_depth the new format offers
// instead of its JSON default — used when the artist switches format by hand,
// NOT on workflow load (where the saved value must win).
async function rebuildCodecWidgets(node, formatLabel, savedValues, opts) {
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
    const preferMax = !!(opts && opts.preferMaxBitDepth);
    for (const spec of def.widgets || []) {
        let saved = savedValues ? savedValues[spec.name] : undefined;
        // Hand-switching format: land on the deepest depth this codec can
        // write rather than its JSON default, so going 8-bit -> ProRes doesn't
        // quietly stay at the shallow end. Never on workflow load — a saved
        // value is the artist's choice and must survive.
        if (preferMax && spec.name === "bit_depth" && saved === undefined) {
            saved = highestDepth(spec);
        }
        const plain = makeCodecWidget(node, spec, saved);
        const intended = plain.value;
        // addCustomWidget returns the concrete instance; fall back to the plain
        // object on any older build that lacks the method.
        const w = (typeof node.addCustomWidget === "function")
            ? node.addCustomWidget(plain)
            : (node.widgets.push(plain), plain);
        // Force the value through the setter, which writes into the frontend's
        // widget-value store.
        //
        // That store is keyed by (graph, node, widget NAME) and survives the
        // widget itself: LGraphNode.removeWidget splices the widget off the
        // node but never unregisters its state. registerWidget then bails out
        // early — `let n = getWidget(id); if (n && n.type === state.type)
        // return n` — so a rebuilt widget with the same name AND type adopts
        // the *previous* format's value and throws away the one we just
        // computed. That is why switching 16-bit FFV1 -> h264 left bit_depth
        // reading "16" until you reopened the dropdown (and why crf/preset
        // carried across format switches too). Assigning here is the only
        // public-API way to make the intended value stick.
        if (w.value !== intended) w.value = intended;
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
        const drivesDerived = Object.values(derived)
            .some((r) => derivedSources(r).includes(spec.name));
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
        // Hand-picked format change: take the new codec's deepest bit depth.
        rebuildCodecWidgets(node, value, undefined, { preferMaxBitDepth: true });
    };
    // Initial build for a freshly-dropped node keeps the format's curated JSON
    // default (h264 at 8-bit, the sane review-copy setting) — "deepest" only
    // applies once the artist deliberately switches to another codec.
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

// A sequence preview is one of these: the payload says so, or (for an entry
// restored from an older localStorage write, which has no is_sequence flag)
// the filename still carries its printf token.
function _vcIsSequence(preview) {
    return !!preview && (preview.is_sequence === true || /%%?\d*d/.test(preview.filename || ""));
}

function _vcIsExr(preview) {
    if (!preview) return false;
    if (typeof preview.is_exr === "boolean") return preview.is_exr;
    return /\.exr$/i.test(preview.filename || "");
}

// Query params every /bat/video/* route accepts for this preview. `fps` and
// `trc` only mean anything for a sequence (see the module header) but are
// harmless elsewhere, so they're sent unconditionally rather than branched on.
function buildPreviewParams(preview, viewTrc) {
    const params = new URLSearchParams({
        filename: preview.filename,
        type:     preview.type || "output",
        subfolder: preview.subfolder || "",
    });
    if (preview.frame_rate) params.set("fps", String(preview.frame_rate));
    if (viewTrc) params.set("trc", viewTrc);
    return params;
}

function buildPreviewUrl(preview, viewTrc) {
    // Non-browser-playable formats — ProRes, FFV1, h265, image sequences —
    // go through the on-demand transcoder.
    const route = preview.browser_playable === false || _vcIsSequence(preview)
        ? "/bat/video/preview" : "/view";
    return api.apiURL(`${route}?${buildPreviewParams(preview, viewTrc).toString()}`);
}

// ── EXR view transform ────────────────────────────────────────────────────
// Kept OUT of the node's widgets on purpose: the codec widgets serialise
// positionally into widgets_values (see the /bat/video/formats docstring), so
// adding one would shift every workflow already saved with an EXR format.
// This is a preview-only display setting, so it lives on the player and
// persists per node in localStorage instead.
const VC_VIEWS = [
    ["",       "1:1",     "Read EXR values straight (matches an h264 export of the same frames)"],
    ["srgb",   "sRGB",    "Apply the sRGB curve — for scene-linear renders"],
    ["rec709", "Rec.709", "Apply the Rec.709 curve — for scene-linear renders"],
];

function _vcViewCacheKey(node) {
    return batNodeCacheKey(app, "bat_vc_view", node);
}

function _vcLoadView(node) {
    try {
        const v = localStorage.getItem(_vcViewCacheKey(node)) || "";
        return VC_VIEWS.some(([id]) => id === v) ? v : "";
    } catch (_) { return ""; }
}

function _vcSaveView(node, view) {
    try { localStorage.setItem(_vcViewCacheKey(node), view || ""); } catch (_) {}
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

    // EXR-only; hidden for every other format (see VC_VIEWS).
    const viewSel = document.createElement("select");
    viewSel.title = "EXR view transform (preview only — does not change the saved frames)";
    viewSel.style.cssText = "background:#1a1d22; color:#cdd; border:1px solid #2a2f37; border-radius:3px; font-size:11px; padding:1px 2px; display:none;";
    for (const [id, label, title] of VC_VIEWS) {
        const o = document.createElement("option");
        o.value = id; o.textContent = label; o.title = title;
        viewSel.appendChild(o);
    }

    const saveFrameBtn = btn("📷", "Save current frame as PNG");
    const fullscreenBtn = btn("⛶", "Fullscreen (F)");

    // Play sits CENTERED between the seek pair — the conventional video-
    // player layout. Reading left-to-right: ⏪ |◀ ▶ ▶| ⏩ · speed · audio
    // · time · 📷 ⛶
    controls.append(
        stepBack10, stepBack, playBtn, stepFwd, stepFwd10,
        speedSel, viewSel, muteBtn, volSlider, timeLabel, saveFrameBtn, fullscreenBtn,
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
        viewTrc: _vcLoadView(node),   // "" | "srgb" | "rec709", EXR only
        isSequence: false,
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
    // Every seek goes through this. The player is `loop = true`, so landing on
    // (or past) `duration` fires `ended` and the element immediately wraps to 0
    // — which is what made the playhead flick between the last frame and the
    // first while the artist was still holding the handle at the end of the
    // timeline: the wrap repainted at frame 1, the drag pushed it back to the
    // end, repeat. The mid-frame target for the last frame, (frameCount - 0.5) /
    // fps, sits past `duration` whenever the container's duration is a hair
    // shorter than frameCount / fps (rounded fps, 29.97 vs 30, a short final
    // frame), so this is reachable on ordinary footage. Stop a millisecond
    // short: still well inside the final frame, never on the wrap point.
    const safeSeekTime = (t) => {
        const d = videoEl.duration;
        const capped = (Number.isFinite(d) && d > 0) ? Math.min(t, d - 1e-3) : t;
        return Math.max(0, capped);
    };
    const snapToFrame = (frame) => {
        if (!state.fps) return;
        const f = Math.max(0, Math.min(state.frameCount - 1, frame));
        videoEl.currentTime = safeSeekTime((f + 0.5) / state.fps);   // mid-frame for accuracy
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
        const target = safeSeekTime((newFrame + 0.5) / fps);
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

    // Changing the view re-transcodes server-side (a different cache entry),
    // so we reload the src and land back on the frame the artist was looking
    // at rather than snapping to zero.
    viewSel.onchange = () => {
        state.viewTrc = viewSel.value;
        _vcSaveView(node, state.viewTrc);
        if (!state.preview) return;
        const wasPaused = videoEl.paused;
        const at = _displayedTime();
        const url = buildPreviewUrl(state.preview, state.viewTrc);
        const restore = () => {
            videoEl.removeEventListener("loadeddata", restore);
            try { videoEl.currentTime = at; } catch (_) {}
            if (!wasPaused) videoEl.play().catch(() => {});
        };
        videoEl.addEventListener("loadeddata", restore);
        videoEl.src = url;
        videoEl.load();
        thumbVideo.src = url;
        thumbVideo.load();
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
    // Was the player running when this scrub started? Playback is suspended for
    // the duration of the drag and restored on release.
    let resumeAfterScrub = false;
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
            // Take `loop` off for the length of the drag, and put playback on
            // hold with it.
            //
            // Dragging to (or past — the position clamps) the right-hand end
            // asks for a seek at the very end of the media. On a `loop = true`
            // element that puts it into its ended state, and the browser wraps
            // the position to 0 — this does NOT require playback to be running,
            // which is why pausing alone didn't fix it. The wrap fires the
            // `seeked` handler, which repaints the playhead at 0 while the mouse
            // is still holding the right-hand end; the next pointermove paints
            // it back at 100%, and it flickers between the two ends.
            //
            // Looping at the end of PLAYBACK is the point of the player and is
            // deliberately preserved — `loop` is restored on release.
            state.scrubPrevLoop = videoEl.loop;
            videoEl.loop = false;
            resumeAfterScrub = !videoEl.paused;
            if (resumeAfterScrub) videoEl.pause();
        }
        timeline.setPointerCapture(e.pointerId);
        handleTimelineMove(e);
    });
    timeline.addEventListener("pointermove", (e) => {
        if (dragging) handleTimelineMove(e);
        else          handleTimelineHover(e);
    });

    // Restore whatever the scrub borrowed: `loop` first, then playback, so the
    // element never resumes with looping still disabled.
    function _endScrub() {
        if (state.scrubPrevLoop != null) {
            videoEl.loop = state.scrubPrevLoop;
            state.scrubPrevLoop = null;
        }
        if (resumeAfterScrub) {
            resumeAfterScrub = false;
            videoEl.play().catch(() => {});
        }
    }

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
        _endScrub();
    });
    timeline.addEventListener("pointercancel", () => {
        dragging = null;
        _endScrub();
    });
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
            const target = safeSeekTime((f + 0.5) / state.fps);
            if (typeof videoEl.fastSeek === "function") {
                videoEl.fastSeek(target);
            } else {
                videoEl.currentTime = target;
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
        if (!videoEl.requestVideoFrameCallback && dragging !== "scrub") {
            _paintPlayhead(videoEl.currentTime);
            updateTimeLabel();
        }
    });
    videoEl.addEventListener("seeked", () => {
        // After any seek (drag-scrub, frame step, in/out enforcement),
        // make sure the playhead reflects where the video actually
        // landed — EXCEPT under an active scrub, where the mouse is the
        // authority and the decoder is chasing it. Repainting from
        // videoEl.currentTime there fights the drag: any seek that resolves
        // somewhere other than the requested position (an approximate
        // fastSeek landing, a clamp at the end of the media) yanks the
        // playhead away from the cursor until the next pointermove puts it
        // back. The drag handler already paints every move, and pointerup
        // does a final exact snap.
        if (dragging !== "scrub") _paintPlayhead(videoEl.currentTime);
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
            // Under an active scrub the mouse owns the playhead — see the
            // note on the `seeked` handler. This tick fires once per
            // composited frame, so it's the loudest of the async repaints
            // and the one that visibly drags the handle off the cursor.
            if (dragging !== "scrub") {
                _paintPlayhead(videoEl.currentTime);
                updateTimeLabel();
            }
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
        state.isSequence = _vcIsSequence(preview);
        state.thumbPendingTime = null;
        state.thumbSeeking = false;
        muteBtn.style.display = volSlider.style.display = "none";
        // Only EXR carries a linear/display ambiguity worth a control.
        viewSel.style.display = _vcIsExr(preview) ? "inline-block" : "none";
        viewSel.value = state.viewTrc;
        // Drop any thumb from the previous file so the first hover on
        // the new file doesn't flash an unrelated frame.
        hoverThumb.removeAttribute("src");

        const url = buildPreviewUrl(preview, state.viewTrc);
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

        const playableNote = (preview.browser_playable === false || state.isSequence)
            ? "(transcoded preview)"  : "";
        // A sequence's filename is a printf pattern, which reads as nothing on
        // its own — show the directory it lives in as well.
        const label = state.isSequence
            ? `${preview.subfolder || ""}/${preview.filename}`
            : preview.filename;
        statusRow.textContent = `${label}  ·  ${preview.format || ""}  ${playableNote}`;

        // Now probe the actual file — ffprobe is the source of truth for
        // fps + frame count (the encode payload reports the input batch size,
        // but doesn't see what the codec actually wrote).
        try {
            const params = buildPreviewParams(preview, state.viewTrc);
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
            const params = buildPreviewParams(cached, state.viewTrc);
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

        // A graph reload (Ctrl+Z is one) destroys and rebuilds every node, so
        // replay the last run's preview payload into the new instance.
        batReplayLastExecution(nodeType);

        // Rewrite pre-2026-08-06 saved data before LiteGraph applies it. This
        // sits on configure() rather than onConfigure() because by the time
        // that callback fires the values have already been dealt out to the
        // wrong widgets — and every load path (opening a workflow, dropping a
        // rendered MOV on the canvas, pasting, undo, a subgraph definition)
        // comes through here. Same shape as the Volt Loader's VRI migration,
        // see ComfyUI-Volt_Loader/web/volt_vri.js.
        //
        // `configure` is inherited, so this reads it off the prototype chain.
        // Guard the migration: dropping the real configure() would leave every
        // loaded node blank, a far worse failure than not migrating.
        const declared = declaredStatics(nodeData);
        const origConfigure = nodeType.prototype.configure;
        if (!nodeType.prototype._batVCLegacyConfigure) {
            nodeType.prototype.configure = function (info) {
                try {
                    migrateWidgetsValues(info, declared);
                } catch (e) {
                    console.warn("[Bat_VideoCombine] legacy migration failed;",
                                 "loading as saved:", e);
                }
                return origConfigure?.apply(this, arguments);
            };
            nodeType.prototype._batVCLegacyConfigure = true;
        }

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
            // length in the saved widgets_values array. (Workflows saved by an
            // older layout were already rewritten into the current one by the
            // configure() patch below, so this holds for those too.)
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
                    // Widgets appended to a format AFTER this workflow was
                    // saved have no entry in the tail. Where one of them drives
                    // a derived value the tail *does* carry (bit_depth ->
                    // pix_fmt), recover it by inverting the map rather than
                    // letting its default silently re-derive the pixel format.
                    inferSourcesFromDerived(def, savedByName);
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
