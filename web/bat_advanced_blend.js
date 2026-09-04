/**
 * Bat_AdvancedBlend — live canvas preview of the blend and its frequency
 * separation.
 *
 * Workflow:
 *   1. Run the node once → Python pushes two pixel-aligned tiles (the
 *      conformed plates A and B) through the `{"ui": ...}` channel.
 *   2. As the artist drags low_mix / high_mix / split_radius / soften_* etc.,
 *      this file re-runs bat_advanced_blend.py's algorithm on those tiles and
 *      repaints.
 *   3. The real full-resolution result is computed by Python on the NEXT run.
 *      The preview is a faithful approximation at tile resolution, not the
 *      bit-exact output.
 *
 * Why the tiles are pre-conformed
 * -------------------------------
 * The two plates are different resolutions — that is the whole premise of the
 * node — and reconciling them is the render's first step. Doing that here as
 * well would mean shipping both at native size and reimplementing a Lanczos
 * resampler in JS, and the preview would still disagree with the render
 * wherever the two implementations rounded differently. Python conforms the
 * preview frame through exactly the same path the render uses and ships the
 * result, so the only thing this file has to mirror is the blend itself.
 *
 * Radii are the one thing that does need converting: every widget value is in
 * working-resolution pixels, and the tile is smaller, so `radiusScale()` maps
 * them down. A blur therefore reads the same here as it will at full res.
 *
 * Two layers, and why
 * -------------------
 * The canvas shows one of two things depending on how recently you touched
 * something.
 *
 * The DRAFT layer is this file re-running the blend on a tile in the browser.
 * It is instant, which is what a slider drag needs, and it is an approximation:
 * the radii are scaled into tile space, so a blur is right in proportion but not
 * in detail.
 *
 * The FULL layer is a request to `/bat/advanced_blend/render`, which runs the
 * REAL `_core` on the real plates at their real resolution and returns a PNG of
 * exactly the region on screen. At fit that is the whole frame rendered properly
 * and downscaled for transport; zoomed in it is native pixels. It is not a
 * mirror of the render — it *is* the render — so it is strictly more correct
 * than the draft, including at fit zoom, where the draft's scaled radii are its
 * one real compromise.
 *
 * Why not just make the draft full-resolution? Payload and compute, both hard
 * walls. The tiles are lossless 16-bit because fine grain is precisely what this
 * node manipulates and precisely what a lossy codec throws away, and lossless
 * means the size is entropy-bound: two full-res 4K tiles would be ~110MB of
 * base64 per execution, and the JS blend over 8.3M pixels ~1.8 billion
 * operations. Neither is fixable by tuning. Moving the work to the server
 * sidesteps both, because the client then only needs the finished picture at
 * screen size.
 *
 * Three further things make what you see genuinely sharp rather than merely
 * large:
 *
 *   • The blend composites into an OFFSCREEN buffer and the visible canvas blits
 *     it under a zoom/pan transform. Previously the canvas was sized to the tile
 *     and stretched by `object-fit: contain`, so the browser smoothly upscaled a
 *     small bitmap to node size — which was most of why it read as soft,
 *     independent of any tile size.
 *   • Smoothing is off at or above 1:1, so you see real pixels rather than a
 *     bilinear guess at them.
 *   • The canvas backing store is sized in DEVICE pixels while every transform
 *     computation stays in CSS pixels — the latter because that is the contract
 *     bat_zoom_control.js reads back for zoom-toward-cursor.
 *
 * There is deliberately no viewer exposure. Range is not thrown away — the tiles
 * carry values above white and the blend runs on them unclamped — but the budget
 * here goes on resolution, because the judgement this preview exists for is
 * per-pixel texture.
 *
 * Keeping a drag legible at full resolution
 * ----------------------------------------
 * Two earlier attempts got this wrong. The first dropped to a fixed
 * half-resolution mip for the whole of any drag; the second chose a mip
 * adaptively from measured repaint time. Both traded away pixels, and pixels are
 * the one thing this preview cannot trade: it exists to show what a change does
 * to high-frequency detail, and downscaling destroys exactly that.
 *
 * The adaptive version also penalised the common case for no reason. The
 * Gaussians are memoised on their radius, so only `split_radius`, `soften_a` and
 * `soften_b` recompute them — every other slider is a cheap per-pixel pass over
 * cached blurs and was being downscaled for nothing.
 *
 * So the draft now blends the FULL tile, always, and the correct degradation for
 * a detail tool is applied instead: drop frames, not pixels. That only works off
 * the main thread — a 768x432 blend at a wide radius is hundreds of milliseconds
 * of uninterruptible loop, which would freeze the graph and the slider being
 * dragged — so it runs in `bat_blend_worker.js`, with requests coalesced to the
 * newest. A slow blend therefore costs latency, never responsiveness.
 *
 * The server layer helps here too, and used to refuse to: pure debouncing meant
 * it only ever fired once the artist let go, precisely when they had stopped
 * needing it. It is now throttled as well as debounced, paced by its own
 * measured round trip, so when it can keep up it refreshes mid-drag. Zoomed in,
 * where a 1:1 crop costs ~10ms, the real render effectively follows your hand.
 *
 * Performance
 * -----------
 * A 768px tile is ~330k pixels and a repaint needs up to four Gaussians (two
 * pre-blurs, two band splits). Besides the ladder above, the blurs are memoised
 * on their radius — so dragging `high_mix` re-runs none of them and
 * `split_radius` re-runs two — and a 120ms idle timer always promotes the draft
 * to the full tile once the hand stops, whatever the drag could afford.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { hdrSupported, decodeHdrTile, imageDataToSource } from "./bat_hdr_preview.js";
import { batTrack, registerCleanup, batNodeCacheKey, isNodeAlive } from "./bat_lifecycle.js";
import { attachZoomControl } from "./bat_zoom_control.js";
import { blendTile, makeBlurCache, paintView, neutralHigh } from "./bat_blend_core.js";

const NODE_TYPE = "Bat_AdvancedBlend";

const VIEWS = [
    ["result", "Result", "The blend."],
    ["a",      "A",      "Plate A alone, conformed to the working resolution.\nsoften_a is NOT applied here — this is the input as the node received it."],
    ["b",      "B",      "Plate B alone, conformed to the working resolution."],
    ["detail", "Detail", "The high-frequency band actually being applied, on 0.5 grey.\nOnly meaningful with frequency_separation on. Use the amplification button next to these buttons to make it readable."],
    ["diff",   "Diff",   "Signed A - B on 0.5 grey: where, and how far, the upscale departed from the original.\nAmplify it with the x-button to see the small stuff."],
];

// ── the Advanced section ─────────────────────────────────────────────────
/**
 * Widgets folded into the collapsible "Advanced" section, in node order.
 * Everything NOT listed stays always-visible, which is the node's simple face:
 * `blend_mode` and `mix`, i.e. an ordinary compositing blend.
 *
 * `resize_mode` / `resize_filter` are in here despite mattering to the headline
 * use case, because their defaults already handle it — B is resampled up to A's
 * size with Lanczos, which is what you want when A is the upscale.
 */
const ADVANCED_WIDGETS = [
    "resize_mode", "resize_filter",
    "frequency_separation", "split_radius", "detail_mode",
    "low_mix", "high_mix", "detail_gain", "detail_limit",
    "soften_a", "soften_b",
    "clamp_output", "preview_frame",
];

/**
 * Fold the advanced widgets into the frontend's own collapsible section.
 *
 * Both renderers support this natively in frontend 1.49.x, but they read
 * different properties, so both have to be set:
 *
 *   • Nodes 1.0 / litegraph — `LGraphNode.isWidgetVisible()` reads
 *     `widget.hidden` and `widget.advanced` off the widget object itself.
 *   • Nodes 2.0 / Vue — `isWidgetVisible(options, showAdvanced, linked)` in
 *     `useProcessedWidgets.ts` reads `options.hidden` / `options.advanced`.
 *
 * Both then gate on `node.showAdvanced`, which litegraph SERIALISES, so the
 * expanded/collapsed state survives a save and reload. And both treat a widget
 * that has been converted to a connected input as visible regardless (`linked`
 * in 2.0), so wiring something up never hides it from you.
 *
 * This function only sets metadata. The switch itself is a plain DOM button
 * inside the preview widget — see buildAdvancedHeader() — and that is not a
 * stylistic choice.
 *
 * WHY THERE IS NO TOGGLE *WIDGET* HERE
 * ------------------------------------
 * There was, briefly: a `serialize = false` button spliced in under `mix`.
 * Litegraph's `configure()` does honour that flag in both directions — it skips
 * such widgets when building `widgets_values` and when walking it back — so the
 * reasoning looked sound and save/reload worked.
 *
 * Copy/paste corrupted the node. The button consumed the slot belonging to
 * `resize_mode`, and every value behind it shifted up one: `frequency_separation`
 * received a number and read `true`, `split_radius` received a string and read
 * `NaN`, `detail_mode` received a number and went blank. So at least one of the
 * several code paths that consume `widgets_values` — litegraph's `configure`, the
 * Vue `widgetValueStore`, the clipboard deserialiser, `migrateWidgetsValues`,
 * `fallbackWidgetsValuesNames` — does not honour `serialize === false`.
 *
 * The lesson is the general one, not the specific one: a non-serialising widget
 * in the MIDDLE of the list is only safe if every one of those paths agrees to
 * skip it, and that is not a bet worth taking for an affordance. A DOM widget is
 * immune by construction — it is `serialize: false` *and* last, so even a path
 * that ignores the flag can only consume a trailing slot with nothing behind it
 * to shift. Hence the header lives in the preview.
 */
function markAdvancedWidgets(node) {
    if (!node.widgets) return;
    for (const name of ADVANCED_WIDGETS) {
        const w = node.widgets.find((x) => x.name === name);
        if (!w) continue;
        w.advanced = true;                        // Nodes 1.0
        w.options = w.options || {};
        w.options.advanced = true;                // Nodes 2.0
    }
}

/**
 * Snapshot the pristine widget values, and repair invalid ones later.
 *
 * `onNodeCreated` runs before `configure()`, so whatever the widgets hold at
 * that moment IS the node definition's defaults — no hardcoded table needed,
 * and no way for it to drift from the Python.
 *
 * The repair exists because the bug described above already shipped: nodes
 * pasted while it was live carry shifted values, and they are saved that way.
 * Without this, opening such a workflow leaves NaN radii and blank dropdowns
 * that the artist has to spot and fix by hand — and a NaN radius reaches the
 * blur, which is not merely cosmetic.
 *
 * It only ever touches values that cannot possibly be legitimate: a non-finite
 * number, or a combo value that is not one of its own options. A wrong-but-valid
 * value is left alone, because silently "correcting" those would be a worse bug
 * than the one being repaired.
 */
function snapshotWidgetDefaults(node) {
    const snap = {};
    for (const w of node.widgets || []) {
        if (w?.name != null) snap[w.name] = w.value;
    }
    node._batAdvBlendDefaults = snap;
}

export function repairWidgetValues(widgets, defaults) {
    const fixed = [];
    for (const w of widgets || []) {
        if (!w || w.name == null || !(w.name in defaults)) continue;
        const def = defaults[w.name];
        const v = w.value;
        let bad = false;

        const options = w.options?.values ?? w.values;
        if (Array.isArray(options) && options.length) {
            // A combo whose value is not one of its options. This is how the
            // shift showed up as blank dropdowns.
            bad = !options.includes(v);
        } else if (typeof def === "number") {
            bad = typeof v !== "number" || !Number.isFinite(v);
        } else if (typeof def === "boolean") {
            // Booleans survive a shift looking plausible ("true" for a number),
            // so only a genuinely non-boolean is repaired.
            bad = typeof v !== "boolean";
        }

        if (bad) {
            w.value = def;
            fixed.push(w.name);
        }
    }
    return fixed;
}

function repairNode(node) {
    const defaults = node._batAdvBlendDefaults;
    if (!defaults) return;
    const fixed = repairWidgetValues(node.widgets, defaults);
    if (fixed.length) {
        console.warn(`[Bat_AdvancedBlend] node ${node.id}: reset ${fixed.length} `
            + `impossible widget value(s) to their defaults: ${fixed.join(", ")}. `
            + "An earlier build corrupted these on copy/paste; check them against "
            + "what you intended.");
        node.setDirtyCanvas?.(true, true);
    }
}

// ── the preview ──────────────────────────────────────────────────────────

/** Amplification steps for the Detail / Diff views. See the note in render(). */
const AMP_STEPS = [1, 4, 16, 64];

/**
 * Draft resolution while a slider is moving.
 *
 * FULL is the default and, for this node, close to a correctness requirement
 * rather than a preference. The preview exists to show what a change does to
 * high-frequency detail; downscaling destroys precisely that, so a smooth
 * quarter-resolution preview of a sharpening change shows nothing worth seeing.
 *
 * An earlier version chose a mip adaptively from measured repaint time. It was
 * the wrong instinct twice over. First, the correct degradation for a detail
 * tool is to drop FRAMES, not PIXELS — a full-resolution picture a moment late
 * is useful; a fast blurry one is not. Second, it penalised the common case for
 * no reason: the Gaussians are memoised on their radius, so only `split_radius`,
 * `soften_a` and `soften_b` recompute them. Every other slider — `mix`,
 * `low_mix`, `high_mix`, `detail_gain`, `detail_limit`, the modes — is a cheap
 * per-pixel pass over cached blurs and was being downscaled for nothing.
 *
 * Frames are now dropped in a worker instead (bat_blend_worker.js), so a slow
 * blend costs latency rather than a frozen slider. The lower settings remain for
 * anyone who would rather have smoothness on a weak machine, but nothing selects
 * them automatically.
 */
const DRAFT_LEVELS = [
    ["full", "Full", 0,
     "Never show a lower-quality picture while you are adjusting a widget.\n"
     + "The whole tile is blended on every repaint, and the full-resolution "
     + "render already on screen is HELD while its replacement computes, instead "
     + "of dropping back to the draft — so the canvas never flickers between two "
     + "sharpnesses mid-drag. It goes a moment stale rather than a moment soft, "
     + "and the badge says so.\n"
     + "Panning and zooming still fall back to the draft: those move the region, "
     + "so the old render genuinely no longer fits.\n"
     + "A wide split_radius or soften value costs latency, not smoothness — the "
     + "blend runs in a worker, so the slider never stops responding."],
    ["half", "Half", 1,
     "Blend at half tile resolution while dragging. Smoother, and it hides "
     + "exactly the detail this node is for. Only worth it on a slow machine."],
    ["quarter", "Quarter", 2,
     "Quarter tile resolution. Framing only — do not judge sharpening on this."],
];

/** Mip levels kept for the reduced settings above. */
const MIP_LEVELS = 3;

/**
 * Identity of a full-resolution render request.
 *
 * `runId` is in here and must stay in here. Everything else describes what to
 * draw; `runId` describes WHICH FRAME it is drawn from, and the server renders
 * from whatever the last execution cached. Without it, changing `preview_frame`
 * and re-running produced a key identical to the one already on screen — so the
 * request was skipped as redundant and the previous frame's render stayed
 * pinned on top of the correct new draft, for ever. That looked exactly like
 * `preview_frame` not working, and nothing about the symptom pointed here.
 *
 * The same applies to any future input that changes what Python ships rather
 * than how it is combined: it belongs in `runId`'s bucket, not in the params.
 */
export function fullRequestKey({ params, view, amp, region, runId }) {
    return JSON.stringify([runId, params, view, amp,
                           region.world, region.outW, region.outH]);
}

/**
 * Should the full-resolution render already on screen be held while its
 * replacement computes, rather than dropped back to the draft?
 *
 * Three terms, and the middle one is the whole point of the Full setting:
 *
 *   • `viewChanged` — a pan or zoom moved the REGION, so the render no longer
 *     covers what is on screen and would be drawn over the wrong part of the
 *     picture. It must go. This is the exception that makes the rest safe, and
 *     the easiest thing to lose in a later edit.
 *   • `quality === "full"` — Full means "never show me a lower-quality picture
 *     while I adjust a widget". Half and Quarter mean the opposite, so they keep
 *     the simpler drop-to-draft behaviour.
 *   • `hasFull` — nothing to hold on the first pass.
 *
 * Exported for the test rather than for any caller.
 */
export function shouldHoldFullRender({ quality, hasFull, viewChanged }) {
    return !viewChanged && quality === "full" && !!hasFull;
}

function buildPreview(node) {
    const track = batTrack(node);

    const root = document.createElement("div");
    root.style.cssText = `position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; overflow:hidden;`;

    // The Advanced switch. A DOM button rather than a litegraph widget — see
    // the long note on markAdvancedWidgets() for why that distinction matters.
    // It sits at the top of the preview, which is directly under the widget
    // block, so it reads as the divider between the parameters and the picture.
    const advBar = document.createElement("div");
    advBar.style.cssText = `display:flex; align-items:center; gap:6px; padding:4px 8px;
        background:#151a1f; border-bottom:1px solid #2a2a2a; font:11px monospace;
        color:#9aa; flex:0 0 auto; cursor:pointer; user-select:none;`;
    const advCaret = document.createElement("span");
    advCaret.style.cssText = "color:#6f8fa8; width:9px; display:inline-block;";
    const advLabel = document.createElement("span");
    const advCount = document.createElement("span");
    advCount.style.cssText = "margin-left:auto; color:#68757f;";
    advBar.append(advCaret, advLabel, advCount);
    advBar.title = "Show or hide the frequency-separation settings.\n"
        + "Collapsed, this node is an ordinary blend: pick a mode and a mix.\n"
        + "Expanded, you get the band split that lets you take an upscale's "
        + "resolution without its sharpening.";

    function refreshAdvBar() {
        const on = !!node.showAdvanced;
        advCaret.textContent = on ? "▾" : "▸";
        advLabel.textContent = "Advanced";
        const n = ADVANCED_WIDGETS.filter(
            (name) => node.widgets?.some((w) => w.name === name)).length;
        advCount.textContent = on ? "hide" : `${n} settings`;
        advBar.style.background = on ? "#1b2530" : "#151a1f";
    }
    advBar.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        if (typeof node.toggleAdvanced === "function") {
            // Preferred: also bumps the graph version, re-fits the node to its
            // new content, and marks the canvas dirty.
            node.toggleAdvanced();
        } else {
            node.showAdvanced = !node.showAdvanced;
        }
        refreshAdvBar();
        node.setDirtyCanvas?.(true, true);
    });
    node._batRefreshAdvBar = refreshAdvBar;
    root.appendChild(advBar);

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative; flex:1 1 auto; min-height:0; display:flex; background:#000;";
    // The VISIBLE canvas is sized to its CSS box; the draft tile and the
    // server's full-resolution PNG are both drawn into it through the same
    // zoom/pan transform. See the "Two layers" note at the top.
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; display:block;";
    stage.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;`;
    hint.textContent = "Run once to populate the preview.";
    stage.appendChild(hint);

    const badge = document.createElement("div");
    badge.style.cssText = `position:absolute; right:6px; top:4px; font:10px monospace;
        color:#9aa; background:rgba(0,0,0,0.62); padding:2px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; text-align:right; line-height:1.45;
        display:none;`;
    stage.appendChild(badge);

    const probe = document.createElement("div");
    probe.style.cssText = `position:absolute; left:6px; top:4px; font:10px monospace;
        color:#cde; background:rgba(0,0,0,0.62); padding:3px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; display:none; line-height:1.45;`;
    stage.appendChild(probe);

    root.appendChild(stage);

    const ctx = canvas.getContext("2d");
    // Only used by the main-thread fallback below; the worker keeps its own.
    const blurs = makeBlurCache();

    /**
     * The blend worker. Optional in every direction — a browser that refuses to
     * construct it, or a deployment where the module fails to load, falls back
     * to blending on the main thread. That is slower to the point of stuttering
     * at wide radii, but it is correct, and a preview must never be the reason a
     * node stops working.
     */
    let worker = null, workerReady = false;
    let jobId = 0, jobInFlight = 0, jobPending = null;
    try {
        worker = new Worker(new URL("./bat_blend_worker.js", import.meta.url),
                            { type: "module" });
        worker.onerror = (e) => {
            console.warn("[Bat_AdvancedBlend] blend worker failed; falling back "
                         + "to the main thread:", e.message || e);
            try { worker.terminate(); } catch (_) {}
            worker = null; workerReady = false;
        };
        worker.onmessage = (e) => onWorkerMessage(e.data);
        track.dispose(() => { try { worker?.terminate(); } catch (_) {} });
    } catch (e) {
        console.warn("[Bat_AdvancedBlend] no blend worker available; blending on "
                     + "the main thread:", e);
        worker = null;
    }

    // Offscreen buffer the blend is composited into, at TILE resolution. Kept
    // separate from the visible canvas so compute resolution and display size
    // are independent — which is what makes zoom/pan and the interactive
    // half-resolution tier below possible at all.
    const off = document.createElement("canvas");
    const offCtx = off.getContext("2d", { willReadFrequently: true });

    const viewKey = batNodeCacheKey(app, "bat_advblend_view", node);
    const saved = (() => {
        try { return JSON.parse(localStorage.getItem(viewKey) || "{}") || {}; }
        catch (_) { return {}; }
    })();

    const state = {
        a: null, b: null,          // full-resolution SourceBuffers
        // Mip ladder, only used by the reduced Half/Quarter settings. Level 0
        // is the tile itself and is the default for everything.
        aMips: null, bMips: null,
        quality: DRAFT_LEVELS.some((q) => q[0] === saved.quality) ? saved.quality : "full",
        draftLevel: 0,
        // Last blend duration, for the badge. Reported, not acted on: the
        // resolution is the artist's choice now, not a measurement's.
        blendMs: 0,
        mask: null, maskImg: null,
        meta: null,
        view: VIEWS.some(v => v[0] === saved.view) ? saved.view : "result",
        amp: AMP_STEPS.includes(saved.amp) ? saved.amp : 4,
        holding: false,
        // Display transform, owned by bat_zoom_control.js. 1 = fit.
        dispZoom: Number.isFinite(saved.dispZoom) ? saved.dispZoom : 1,
        panX: 0, panY: 0,
        dispScale: 1, offX: 0, offY: 0,
        // Last computed frame, and which tier it came from.
        out: null, high: null, tw: 0, th: 0, tier: "lo",
        // The full-resolution layer: a PNG rendered by the REAL Python on the
        // real plates, fetched once the artist stops adjusting. `key` records
        // the exact params/view/region it was rendered for, so it is only ever
        // drawn while it still corresponds to what is on screen.
        full: null, fullKey: null, fullRect: null, fullPending: false,
        // Showing a full-resolution render of settings that have since moved on,
        // while its replacement is computed. Only ever true on Full.
        fullStale: false,
        // Bumped on every execution. Part of the full-render request key, so a
        // new payload can never be mistaken for the one already on screen —
        // see fullRequestKey().
        runId: 0,
        // A widget changed that only Python can act on, so the preview on screen
        // is out of date until the next Run.
        needsRun: false,
        // Measured round trip for a server render, used to pace the requests.
        serverMs: 0,
    };
    node._batAdvBlendState = state;

    const W = (n) => node.widgets?.find(w => w.name === n);
    const num = (n, d) => { const w = W(n); const v = w ? +w.value : NaN; return Number.isFinite(v) ? v : d; };
    const str = (n, d) => { const w = W(n); return w ? String(w.value) : d; };
    const bool = (n, d) => { const w = W(n); return w ? !!w.value : d; };

    function saveView() {
        try {
            localStorage.setItem(viewKey, JSON.stringify({
                view: state.view, amp: state.amp, dispZoom: state.dispZoom,
            }));
        } catch (_) { /* quota / disabled storage — still works this session */ }
    }

    /**
     * Widget radii are in working-resolution pixels; the tile is smaller, and
     * the interactive tier is smaller again. Both scalings fold in here so
     * everything below blendTile() is in one coordinate system.
     */
    function radiusScale(src) {
        const full = state.meta?.w || 0;
        const tile = src?.width || 0;
        if (!full || !tile) return 1;
        const s = tile / full;
        return (s > 0 && Number.isFinite(s)) ? s : 1;
    }

    /**
     * Widget values as the blend consumes them.
     *
     * Radii convert from working-resolution pixels to tile pixels here, so
     * everything downstream is in one coordinate system — and `src` decides
     * which, because the worker blends the full tile while the reduced settings
     * blend a mip.
     */
    function currentParams(src) {
        if (!src) return null;
        const rs = radiusScale(src);
        return {
            blend_mode: str("blend_mode", "over"),
            mix: num("mix", 1),
            frequency_separation: bool("frequency_separation", false),
            split_radius: num("split_radius", 4) * rs,
            detail_mode: str("detail_mode", "subtract"),
            low_mix: num("low_mix", 1),
            high_mix: num("high_mix", 1),
            detail_gain: num("detail_gain", 1),
            detail_limit: num("detail_limit", 0),
            soften_a: num("soften_a", 0) * rs,
            soften_b: num("soften_b", 0) * rs,
            // Default matches the Python: OFF, so HDR / scene-linear input keeps
            // its values above white. Only the 8-bit paint clips, and that is a
            // display limit.
            clamp_output: bool("clamp_output", false),
        };
    }

    function maskFor(w, h) {
        if (!state.maskImg) return null;
        const key = `${w}x${h}`;
        if (state.mask && state.mask.key === key) return state.mask.data;
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        const mctx = c.getContext("2d", { willReadFrequently: true });
        mctx.drawImage(state.maskImg, 0, 0, w, h);
        const px = mctx.getImageData(0, 0, w, h).data;
        const out = new Float32Array(w * h);
        for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = px[p] / 255;
        state.mask = { key, data: out };
        return out;
    }

    /** Box-halve a SourceBuffer. Area-average, not point sample: the whole
     *  point of this tier is a faithful-looking stand-in during a drag, and a
     *  point-sampled half would alias the texture the artist is judging. */
    function halve(src) {
        if (!src) return null;
        const w = Math.max(1, src.width >> 1), h = Math.max(1, src.height >> 1);
        const out = new Float32Array(w * h * 3);
        const sw = src.width, d = src.data;
        for (let y = 0; y < h; y++) {
            const y0 = (y * 2) * sw, y1 = Math.min(y * 2 + 1, src.height - 1) * sw;
            for (let x = 0; x < w; x++) {
                const x0 = x * 2, x1 = Math.min(x * 2 + 1, sw - 1);
                const p00 = (y0 + x0) * 3, p01 = (y0 + x1) * 3;
                const p10 = (y1 + x0) * 3, p11 = (y1 + x1) * 3;
                const q = (y * w + x) * 3;
                for (let c = 0; c < 3; c++) {
                    out[q + c] = (d[p00 + c] + d[p01 + c] + d[p10 + c] + d[p11 + c]) * 0.25;
                }
            }
        }
        return { data: out, width: w, height: h, hdr: src.hdr, lo: src.lo, hi: src.hi };
    }

    // ── display transform ────────────────────────────────────────────────
    /**
     * Everything here is in CSS pixels, NOT backing-store pixels, because
     * that is the contract bat_zoom_control.js expects: it measures the
     * pointer as `(clientX - rect.left) * (canvas.clientWidth / rect.width)`
     * and anchors zoom-toward-cursor by reading back state.dispScale/offX/offY.
     * Feeding it device pixels would make it zoom toward the wrong point on
     * any HiDPI display. Device-pixel resolution is bought separately, by the
     * `ctx.setTransform(dpr, ...)` in present().
     */
    function recomputeDisplay() {
        const cw = canvas.clientWidth || 1, ch = canvas.clientHeight || 1;
        const iw = state.tw || 1, ih = state.th || 1;
        // bat_zoom_control.js reads state.imgW/imgH directly to anchor
        // zoom-toward-cursor. They are part of its contract, not optional: with
        // them unset it anchors against a width of 0 and the wheel drifts the
        // image instead of zooming into the point under the pointer.
        state.imgW = iw; state.imgH = ih;
        const fit = Math.min(cw / iw, ch / ih);
        state.dispScale = fit * (state.dispZoom || 1);
        state.offX = (cw - iw * state.dispScale) / 2 + (state.panX || 0);
        state.offY = (ch - ih * state.dispScale) / 2 + (state.panY || 0);
    }

    /** Blit the offscreen buffer to the visible canvas under the transform. */
    function present() {
        // Backing store at device resolution, drawing in CSS pixels. Without
        // this the canvas is a CSS-pixel bitmap stretched by the browser, which
        // throws away half the detail on a HiDPI screen before the artist ever
        // sees it — the tile can be as sharp as you like and it would still
        // look soft.
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const cw = Math.max(1, stage.clientWidth), ch = Math.max(1, stage.clientHeight);
        const bw = Math.round(cw * dpr), bh = Math.round(ch * dpr);
        if (canvas.width !== bw || canvas.height !== bh) {
            canvas.width = bw; canvas.height = bh;
        }
        recomputeDisplay();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, cw, ch);
        if (!state.tw) return;
        // Smoothing on when minifying, off at or above 1:1. Above 1:1 the
        // artist is inspecting individual pixels, and a bilinear upscale would
        // hide exactly the per-pixel sharpening this node adjusts.
        ctx.imageSmoothingEnabled = state.dispScale < 1;
        ctx.imageSmoothingQuality = "high";

        // The draft always goes down first, even when the full layer is
        // available: the full render covers only the region that was visible
        // when it was requested, and drawing the draft underneath means a
        // partially-stale edge shows the approximation rather than black.
        ctx.drawImage(off, state.offX, state.offY,
                      state.tw * state.dispScale, state.th * state.dispScale);

        if (state.full && state.fullRect) {
            const r = state.fullRect;
            // The full render was produced for an exact tile-space rectangle;
            // place it back on that rectangle under the current transform. It
            // is drawn 1:1 with the screen by construction (the request asked
            // for the on-screen pixel size), so smoothing is off.
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(state.full,
                          state.offX + r.x * state.dispScale,
                          state.offY + r.y * state.dispScale,
                          r.w * state.dispScale, r.h * state.dispScale);
        }
    }

    // ── the algorithm — mirrors BatAdvancedBlend._core ───────────────────
    /** The mip level the artist has selected. No measurement, no surprises. */
    function draftLevel() {
        const row = DRAFT_LEVELS.find((q) => q[0] === state.quality);
        return row ? row[2] : 0;
    }

    function compute(level) {
        const A = state.aMips?.[level] || state.a;
        const B = state.bMips?.[level] || state.b;
        if (!A || !B) return false;
        const w = A.width, h = A.height, n = w * h;
        if (B.width !== w || B.height !== h) {
            hint.textContent = "Preview tiles disagree on size — run the node again.";
            hint.style.display = "block";
            return false;
        }

        const p = currentParams(A);
        if (!p) return false;
        const neutral = neutralHigh(p.detail_mode);

        const mask = maskFor(w, h);
        const t0 = performance.now();
        const { out: blended, high } = blendTile(A.data, B.data, w, h, mask, p, blurs);
        state.blendMs = performance.now() - t0;

        const view = state.holding ? "b" : state.view;
        if (off.width !== w || off.height !== h) { off.width = w; off.height = h; }
        const img = offCtx.createImageData(w, h);
        paintView({
            view, a: A.data, b: B.data, out: blended, high,
            neutral, amp: state.amp, w, h, rgba: img.data,
        });
        offCtx.putImageData(img, 0, 0);

        state.out = blended; state.high = high;
        state.tw = w; state.th = h; state.draftLevel = level;
        hint.style.display = "none";
        return true;
    }

    function paintBadge() {
        const m = state.meta;
        const lines = [];
        if (m?.w) lines.push(`${m.w}x${m.h}${m.frames > 1 ? `  f${m.frame}/${m.frames - 1}` : ""}`);
        // Report the scale against the WORKING resolution, not the tile: "1:1"
        // has to mean one plate pixel per screen pixel or it is a lie.
        const workScale = (state.meta?.w && state.tw)
            ? state.dispScale * (state.tw / state.meta.w) : state.dispScale;
        const draftPx = state.tw && state.th ? `${state.tw}x${state.th}` : "";
        const cost = state.blendMs >= 1 ? `  ${Math.round(state.blendMs)}ms` : "";
        const layer = state.full
            ? (state.fullStale ? "full res — updating…" : "full res")
            : (state.fullPending ? `draft ${draftPx}${cost} — rendering…`
               : `draft ${draftPx}${cost}`);
        lines.push(`${workScale.toFixed(2)}:1   ${layer}`);
        if (state.needsRun) {
            lines.push("⟳ Run to apply preview_frame / resize");
        }
        badge.textContent = lines.join("\n");
        badge.style.display = lines.length ? "block" : "none";
    }

    // ── scheduling: draft tier on interaction, full tier once it settles ──
    //
    // A 768px tile is ~440k pixels and a repaint needs up to four Gaussians; at
    // a wide split_radius that is far too slow to keep up with a slider. So a
    // drag repaints the half-resolution mip (a quarter of the work) and a timer
    // promotes it to the full tile once the artist stops moving. The result
    // simply sharpens a moment after you let go.
    /**
     * Adopt a finished worker render.
     *
     * Anything but the newest job is discarded: during a drag the editor keeps
     * asking, and an answer to a superseded question would flicker the canvas
     * backwards. The float buffers only arrive with a settled frame, so the
     * probe reads the settled values — which is right, since nobody hovers for a
     * readout while dragging.
     */
    function onWorkerMessage(msg) {
        if (!msg || !isNodeAlive(node)) return;
        if (msg.type === "error") {
            console.warn("[Bat_AdvancedBlend] worker blend failed:", msg.message);
            jobInFlight = 0; flushPending();
            return;
        }
        if (msg.type !== "rendered") return;
        jobInFlight = 0;
        if (msg.id === jobId) {
            state.blendMs = msg.ms;
            state.tw = msg.w; state.th = msg.h;
            if (off.width !== msg.w || off.height !== msg.h) {
                off.width = msg.w; off.height = msg.h;
            }
            offCtx.putImageData(new ImageData(msg.rgba, msg.w, msg.h), 0, 0);
            if (msg.out) { state.out = msg.out; state.high = msg.high; }
            hint.style.display = "none";
            present(); paintBadge();
        }
        flushPending();
    }

    function flushPending() {
        if (!jobPending || jobInFlight) return;
        const job = jobPending;
        jobPending = null;
        postJob(job);
    }

    /** Send a render to the worker, coalescing to the newest request. */
    function postJob(job) {
        if (!worker || !workerReady) return false;
        if (jobInFlight) { jobPending = job; return true; }
        jobInFlight = ++jobId;
        job.id = jobInFlight;
        worker.postMessage(job);
        return true;
    }

    const SETTLE_MS = 120;
    let raf = 0, settle = 0, wantLevel = 0;

    function draw(level, settled) {
        if (!isNodeAlive(node)) return;
        // The worker only holds the full tile, so the reduced settings and the
        // no-worker case both go through the synchronous path.
        if (level === 0 && worker && workerReady) {
            const p = currentParams(state.a);
            if (p && postJob({
                type: "render", p,
                view: state.holding ? "b" : state.view,
                amp: state.amp,
                // Float buffers are 8MB a frame; only ask for them when the
                // picture has settled and the probe might actually read them.
                wantValues: !!settled,
            })) {
                paintBadge();
                return;
            }
        }
        try {
            if (compute(level)) { present(); paintBadge(); }
        } catch (e) {
            console.error("[Bat_AdvancedBlend] preview failed:", e);
        }
    }

    // ── the full-resolution layer ─────────────────────────────────────────
    //
    // Everything above this point runs in the browser on a downscaled tile. The
    // request below asks the server to run the REAL blend, on the real plates,
    // at their real resolution, over exactly the region currently on screen —
    // so at fit it is the whole frame rendered properly and downscaled for
    // transport, and zoomed in it is native pixels. It is not a mirror of the
    // render; it is the render.
    //
    // Fired on settle rather than continuously, which is the whole trade: a
    // drag stays live on the draft, and the truth arrives a moment later.
    // Baseline debounce. The real delay adapts to how long the server actually
    // takes (see invalidateFull), because that ranges from ~10ms for a 1:1 crop
    // to a few hundred for a whole 4K frame, and a fixed wait is either
    // needlessly slow at one end or a request flood at the other.
    const FULL_DELAY = 140;
    let fullTimer = 0, fullAbort = null, lastFullAt = 0;

    /** The params the server needs, in the same shape blendTile() consumes. */
    function paramsSnapshot() {
        return {
            blend_mode: str("blend_mode", "over"),
            mix: num("mix", 1),
            frequency_separation: bool("frequency_separation", false),
            // NOT scaled here: these are working-resolution pixels, which is
            // exactly what the server wants. Scaling them is a draft-only
            // concession and the reason the draft is an approximation at all.
            split_radius: num("split_radius", 4),
            detail_mode: str("detail_mode", "subtract"),
            low_mix: num("low_mix", 1),
            high_mix: num("high_mix", 1),
            detail_gain: num("detail_gain", 1),
            detail_limit: num("detail_limit", 0),
            soften_a: num("soften_a", 0),
            soften_b: num("soften_b", 0),
            clamp_output: bool("clamp_output", false),
        };
    }

    /**
     * The on-screen region, as a tile-space rect (for drawing back) plus a
     * working-resolution rect (for the server) and the device-pixel size to
     * render it at.
     */
    function visibleRegion() {
        const cw = Math.max(1, canvas.clientWidth), ch = Math.max(1, canvas.clientHeight);
        const sc = state.dispScale || 1;
        // Invert the display transform to find which part of the tile is on
        // screen, then pad a pixel so a half-covered edge is not left showing
        // the draft.
        let x0 = Math.floor((0 - state.offX) / sc) - 1;
        let y0 = Math.floor((0 - state.offY) / sc) - 1;
        let x1 = Math.ceil((cw - state.offX) / sc) + 1;
        let y1 = Math.ceil((ch - state.offY) / sc) + 1;
        x0 = Math.max(0, x0); y0 = Math.max(0, y0);
        x1 = Math.min(state.tw, x1); y1 = Math.min(state.th, y1);
        if (x1 <= x0 || y1 <= y0) return null;

        const tileRect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
        const k = (state.meta?.w && state.tw) ? state.meta.w / state.tw : 1;
        const world = {
            x: Math.floor(tileRect.x * k), y: Math.floor(tileRect.y * k),
            w: Math.max(1, Math.round(tileRect.w * k)),
            h: Math.max(1, Math.round(tileRect.h * k)),
        };
        // Render at the size it will actually occupy in device pixels, capped
        // so a silly node size can't ask for a gigapixel PNG. Never more than
        // the region really has, either — asking the server to enlarge is just
        // a slower way of doing what the browser would do anyway.
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const outW = Math.max(1, Math.min(Math.round(tileRect.w * sc * dpr), world.w, 4096));
        const outH = Math.max(1, Math.min(Math.round(tileRect.h * sc * dpr), world.h, 4096));
        return { tileRect, world, outW, outH };
    }

    /** Cancel any pending or in-flight request, keeping the picture on screen. */
    function abortFull() {
        state.fullPending = false;
        clearTimeout(fullTimer);
        try { fullAbort?.abort(); } catch (_) {}
        fullAbort = null;
    }

    /** Cancel, and throw the picture away too. */
    function dropFull() {
        abortFull();
        state.full = null; state.fullKey = null; state.fullRect = null;
        state.fullStale = false;
    }

    async function requestFull() {
        if (!isNodeAlive(node) || !state.tw || state.holding) return;
        if (node.id == null) return;
        const region = visibleRegion();
        if (!region) return;

        const key = fullRequestKey({
            params: paramsSnapshot(), view: state.view, amp: state.amp,
            region, runId: state.runId,
        });
        if (state.fullKey === key && state.full) return;

        const ctl = new AbortController();
        fullAbort = ctl;
        state.fullPending = true;
        paintBadge();
        const started = performance.now();
        try {
            const res = await fetch("/bat/advanced_blend/render", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: ctl.signal,
                body: JSON.stringify({
                    node_id: String(node.id),
                    roi: [region.world.x, region.world.y, region.world.w, region.world.h],
                    out_w: region.outW, out_h: region.outH,
                    view: state.view, amp: state.amp,
                    ...paramsSnapshot(),
                }),
            });
            if (!res.ok) {
                // 409 just means this node has not run since the server came up,
                // so there is nothing cached to render from. Keep the draft and
                // say nothing — it is not a failure the artist caused.
                if (res.status !== 409) {
                    console.warn("[Bat_AdvancedBlend] full-resolution render failed:",
                                 res.status, await res.text().catch(() => ""));
                }
                // If we were holding a stale render waiting for this, let it go —
                // better the current draft than a sharp picture of settings the
                // artist has since moved on from, held indefinitely.
                if (state.fullStale) dropFull();
                state.fullPending = false;
                present(); paintBadge();
                return;
            }
            const blob = await res.blob();
            const img = (typeof createImageBitmap === "function")
                ? await createImageBitmap(blob)
                : await new Promise((ok, no) => {
                    const i = new Image();
                    i.onload = () => ok(i); i.onerror = no;
                    i.src = URL.createObjectURL(blob);
                });
            if (!isNodeAlive(node) || ctl.signal.aborted) return;
            // Only adopt it if nothing has moved since the request went out.
            const stillKey = fullRequestKey({
                params: paramsSnapshot(), view: state.view, amp: state.amp,
                region, runId: state.runId,
            });
            if (stillKey !== key) { state.fullPending = false; return; }
            state.full = img;
            state.fullKey = key;
            state.fullRect = region.tileRect;
            state.fullPending = false;
            // Round trip, not just server compute: encode, transfer and decode
            // all count, because what the throttle needs to know is how often a
            // new one can realistically land.
            state.fullStale = false;
            state.serverMs = performance.now() - started;
            lastFullAt = performance.now();
            present(); paintBadge();
        } catch (e) {
            if (e?.name !== "AbortError") {
                console.warn("[Bat_AdvancedBlend] full-resolution render error:", e);
                if (state.fullStale) dropFull();
                present();
            }
            state.fullPending = false;
            paintBadge();
        }
    }

    /**
     * Anything that changes what should be on screen invalidates the full layer
     * and re-arms the request.
     *
     * Pure debouncing was the wrong shape here: while a slider is moving it
     * keeps resetting, so the true full-resolution picture never appeared until
     * the artist let go — exactly when they no longer needed it to. So there is
     * a throttle alongside it: if the server has been keeping up and nothing is
     * in flight, fire straight away mid-drag. Zoomed in, where a 1:1 crop costs
     * ~10ms, that means the real render effectively follows your hand.
     */
    function invalidateFull(interactive, viewChanged) {
        // The distinction that stops the canvas flickering between two
        // sharpnesses while a slider moves.
        //
        // A pan or zoom moves the REGION, so the render on screen no longer
        // covers what is on screen — it has to go, and the draft (which covers
        // the whole tile) takes over until a new one arrives.
        //
        // A widget change only alters CONTENT. The existing render is then
        // sharp-but-a-moment-stale rather than wrong-in-the-wrong-place, and on
        // Full that is the better of the two things to be looking at: dropping
        // to the draft trades the resolution this node exists to show for a
        // freshness the replacement supplies ~150ms later anyway. So hold it,
        // mark it stale, and swap when the new one lands.
        //
        // Half and Quarter keep the old behaviour, because choosing them is
        // choosing responsiveness over fidelity.
        const holdStale = shouldHoldFullRender({
            quality: state.quality, hasFull: !!state.full, viewChanged,
        });
        if (holdStale) {
            state.fullStale = true;
            abortFull();
        } else {
            dropFull();
        }
        const rtt = state.serverMs || 0;
        // Never hammer harder than the server can answer; never dawdle when it
        // is fast.
        const delay = Math.max(FULL_DELAY, Math.min(rtt, 400));
        const throttle = Math.max(2 * rtt, 180);
        if (interactive && rtt && rtt < 250
            && performance.now() - lastFullAt >= throttle) {
            requestFull();
            return;
        }
        fullTimer = track.timeout(setTimeout(requestFull, delay));
    }

    function schedule(interactive) {
        invalidateFull(interactive);
        if (interactive) {
            wantLevel = draftLevel();
            clearTimeout(settle);
            // Always promote to the full tile once the hand stops, whatever the
            // drag was able to afford.
            settle = track.timeout(setTimeout(() => draw(0, true), SETTLE_MS));
        } else {
            wantLevel = 0;
        }
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            draw(wantLevel);
            wantLevel = 0;
        });
    }
    node._batAdvBlendRepaint = () => schedule(false);

    // ── control bar ──────────────────────────────────────────────────────
    const bar = document.createElement("div");
    bar.style.cssText = `display:flex; align-items:center; gap:4px; padding:4px 6px;
        background:#141414; border-top:1px solid #2a2a2a; font:11px monospace;
        color:#9aa; flex:0 0 auto; user-select:none; flex-wrap:wrap;`;

    const mkBtn = (text, title, onClick) => {
        const b = document.createElement("button");
        b.textContent = text; b.title = title;
        b.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
            border-radius:3px; padding:2px 7px; font:11px monospace; cursor:pointer;`;
        b.addEventListener("click", onClick);
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };

    const viewBtns = VIEWS.map(([id, label, tip]) => {
        const b = mkBtn(label, tip, () => {
            state.view = id; saveView(); refreshBar(); schedule(false);
        });
        bar.appendChild(b);
        return [id, b];
    });

    // Draft-resolution control. A plain <select> rather than a widget: this is a
    // viewer preference, and putting it in the widget list would serialise it
    // into every workflow (and, historically, corrupt them — see
    // markAdvancedWidgets).
    const qualSel = document.createElement("select");
    qualSel.title = "Resolution the preview repaints at WHILE you drag a slider.\n"
        + "The settled picture is always the full tile, and the true "
        + "full-resolution render follows a moment later regardless.";
    qualSel.style.cssText = `background:#222; color:#cde; border:1px solid #333;
        border-radius:3px; padding:1px 3px; font:11px monospace; cursor:pointer;`;
    for (const [id, label, , tip] of DRAFT_LEVELS) {
        const o = document.createElement("option");
        o.value = id; o.textContent = label; o.title = tip;
        qualSel.appendChild(o);
    }
    qualSel.value = state.quality;
    qualSel.addEventListener("change", () => {
        state.quality = qualSel.value;
        saveView();
        // Redraw at the newly-allowed level immediately, so the choice is
        // visible without having to touch a slider to find out.
        draw(draftLevel());
        paintBadge();
    });
    qualSel.addEventListener("pointerdown", (e) => e.stopPropagation());
    const qualLabel = document.createElement("span");
    qualLabel.textContent = "drag";
    qualLabel.style.cssText = "opacity:0.7;";
    qualLabel.title = qualSel.title;
    bar.append(qualLabel, qualSel);

    const ampBtn = mkBtn("x4",
        "Amplification for the Detail and Diff views, pivoted around mid grey.\n"
        + "Those views carry differences of a few code values, so they need a gain to "
        + "read at all. Inspection only — it never touches the Result view or the render.",
        () => {
            const i = AMP_STEPS.indexOf(state.amp);
            state.amp = AMP_STEPS[(i + 1) % AMP_STEPS.length];
            saveView(); refreshBar(); schedule(false);
        });
    bar.appendChild(ampBtn);

    // ── help ─────────────────────────────────────────────────────────────
    // The node has a fair number of interactions that are invisible until
    // someone tells you about them — hold-to-compare above all, which is the
    // one an artist would use most and the one least likely to be discovered.
    const HELP_ROWS = [
        ["Hold left-drag on image", "Temporarily show plate B — the fastest A/B"],
        ["Middle-mouse drag", "Pan"],
        ["Scroll wheel", "Zoom toward the cursor (0.2x – 4x)"],
        ["Double-click image", "Reset the view"],
        ["⤾  /  click the %", "Reset the view"],
        ["Hover the image", "Read out out / A / B / detail values at that pixel"],
        ["Result A B Detail Diff", "Which layer to look at"],
        ["x1 x4 x16 x64", "Amplify Detail / Diff around mid grey (inspection only)"],
        ["drag: Full/Half/…", "Full (default) never drops quality while you adjust a "
                              + "widget — it holds the full-resolution render and goes "
                              + "briefly stale instead of briefly soft. Half/Quarter "
                              + "trade that for responsiveness"],
        ["▸ Advanced", "Show or hide the frequency-separation settings"],
        ["Badge, top right", "Scale against the plate's real resolution, the draft's "
                             + "current pixel size, and whether you are looking at the "
                             + "draft or the full-resolution render"],
    ];

    const help = document.createElement("div");
    help.style.cssText = `position:absolute; right:6px; bottom:6px; z-index:7;
        display:none; max-width:min(420px, 94%); background:rgba(8,10,12,0.95);
        border:1px solid #35404a; border-radius:4px; padding:7px 9px;
        font:10px monospace; color:#cdd; line-height:1.5; pointer-events:none;
        box-shadow:0 4px 14px rgba(0,0,0,0.6);`;
    {
        const title = document.createElement("div");
        title.textContent = "Preview controls";
        title.style.cssText = "color:#8fb7d9; margin-bottom:4px; letter-spacing:0.4px;";
        help.appendChild(title);
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid; grid-template-columns:auto 1fr; gap:2px 10px;";
        for (const [k, v] of HELP_ROWS) {
            const kk = document.createElement("span");
            kk.textContent = k;
            kk.style.cssText = "color:#e6c07b; white-space:nowrap;";
            const vv = document.createElement("span");
            vv.textContent = v;
            vv.style.cssText = "color:#b9c4cc;";
            grid.append(kk, vv);
        }
        help.appendChild(grid);
        const foot = document.createElement("div");
        foot.textContent = "Drags show a fast draft; the full-resolution render "
            + "arrives a moment after you stop.";
        foot.style.cssText = "color:#7b8894; margin-top:5px; white-space:normal;";
        help.appendChild(foot);
    }
    stage.appendChild(help);

    const helpBtn = mkBtn("?", "Show the preview's controls and shortcuts", () => {});
    // Hover, not click — asking someone to click a help button to find out how
    // to use a help button is its own small joke. Focus works too, for keyboard.
    const showHelp = () => { help.style.display = "block"; };
    const hideHelp = () => { help.style.display = "none"; };
    helpBtn.addEventListener("pointerenter", showHelp);
    helpBtn.addEventListener("pointerleave", hideHelp);
    helpBtn.addEventListener("focus", showHelp);
    helpBtn.addEventListener("blur", hideHelp);
    helpBtn.style.marginLeft = "auto";
    bar.appendChild(helpBtn);

    function refreshBar() {
        const active = state.holding ? "b" : state.view;
        for (const [id, b] of viewBtns) {
            const on = active === id;
            b.style.background = on ? "#2b4a63" : "#222";
            b.style.color = on ? "#cfe8ff" : "#9aa";
            b.style.borderColor = on ? "#4a7fa8" : "#333";
        }
        const inspecting = state.view === "detail" || state.view === "diff";
        ampBtn.textContent = `x${state.amp}`;
        ampBtn.style.display = inspecting ? "" : "none";
        ampBtn.style.background = state.amp === 1 ? "#222" : "#2b4a63";
        ampBtn.style.color = state.amp === 1 ? "#9aa" : "#cfe8ff";
        ampBtn.style.borderColor = state.amp === 1 ? "#333" : "#4a7fa8";
    }
    refreshBar();
    root.appendChild(bar);

    // Zoom / pan. This is what actually delivers "look at it properly": at 1:1
    // and above the tile's own pixels are drawn without smoothing, so real
    // sharpening is visible rather than a bilinear approximation of it.
    let zoomCtl = null;
    try {
        zoomCtl = attachZoomControl({
            wrap: stage, canvas, state,
            onChange: () => {
                saveView();
                // A pan or zoom changes which region the full layer should
                // cover, so it stops being valid immediately — drop back to the
                // draft and re-request once the artist settles. This is the one
                // case that must NOT hold the stale render: it would be drawn
                // over the wrong part of the picture.
                invalidateFull(false, true);
                present(); paintBadge();
            },
        });
    } catch (e) {
        console.warn("[Bat_AdvancedBlend] zoom control unavailable:", e);
    }

    // Hold anywhere on the image to flip back to plate B — the fastest possible
    // A/B, and the one an artist reaches for constantly when judging how much
    // of an upscale to let through. Left button only, so the zoom control's
    // middle-mouse pan is untouched.
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation();
        state.holding = true;
        refreshBar(); schedule(false);
    });
    const release = () => {
        if (!state.holding) return;
        state.holding = false;
        refreshBar(); schedule(false);
    };
    track.listener(window, "pointerup", release);
    canvas.addEventListener("pointerleave", () => { probe.style.display = "none"; release(); });

    canvas.addEventListener("pointermove", (e) => {
        if (!state.out || !state.tw) return;
        const r = canvas.getBoundingClientRect();
        // Invert the display transform, in the same CSS-pixel space
        // bat_zoom_control.js uses (its canvasPos()), so the readout stays
        // correct at any zoom, pan or device pixel ratio.
        const px = (e.clientX - r.left) * (canvas.clientWidth / Math.max(r.width, 1));
        const py = (e.clientY - r.top) * (canvas.clientHeight / Math.max(r.height, 1));
        const x = Math.floor((px - state.offX) / state.dispScale);
        const y = Math.floor((py - state.offY) / state.dispScale);
        if (x < 0 || y < 0 || x >= state.tw || y >= state.th) { probe.style.display = "none"; return; }
        const p = (y * state.tw + x) * 3;
        const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));
        const A = state.aMips?.[state.draftLevel] || state.a;
        const B = state.bMips?.[state.draftLevel] || state.b;
        probe.textContent =
            `out    ${f(state.out[p])} ${f(state.out[p + 1])} ${f(state.out[p + 2])}\n` +
            `A      ${f(A.data[p])} ${f(A.data[p + 1])} ${f(A.data[p + 2])}\n` +
            `B      ${f(B.data[p])} ${f(B.data[p + 1])} ${f(B.data[p + 2])}\n` +
            `detail ${f(state.high[p])} ${f(state.high[p + 1])} ${f(state.high[p + 2])}`;
        probe.style.display = "block";
    });

    // ── widget watching ──────────────────────────────────────────────────
    node._batAdvBlendWatch = () => {
        for (const name of ["blend_mode", "mix", "frequency_separation",
                            "split_radius", "detail_mode", "low_mix", "high_mix",
                            "detail_gain", "detail_limit", "soften_a", "soften_b",
                            "clamp_output"]) {
            const wd = W(name);
            if (!wd) continue;
            const orig = wd.callback;
            // Numeric widgets fire this continuously through a drag, which is
            // exactly when the draft tier earns its keep.
            wd.callback = function (v) {
                try { orig?.call(this, v); } catch (_) {}
                schedule(true);
            };
        }
        // resize_mode / resize_filter / preview_frame cannot be recomputed here:
        // all three change what Python SHIPS, not how the tiles are combined.
        // They are still watched, though, so the node can SAY so — silently
        // doing nothing is what makes a working control feel broken.
        for (const name of ["resize_mode", "resize_filter", "preview_frame"]) {
            const wd = W(name);
            if (!wd) continue;
            const orig = wd.callback;
            wd.callback = function (v) {
                try { orig?.call(this, v); } catch (_) {}
                state.needsRun = true;
                paintBadge();
            };
        }
    };

    // ── ingest ───────────────────────────────────────────────────────────
    const cacheKey = batNodeCacheKey(app, "bat_advblend_preview", node);

    async function decodePlate(tile, jpeg, which) {
        if (tile && hdrSupported()) {
            try { return await decodeHdrTile(tile); }
            catch (e) { console.warn(`[Bat_AdvancedBlend] ${which} tile:`, e); }
        }
        if (!jpeg) return null;
        // Fallback only. The JPEG is clamped and 8-bit, and its own compression
        // noise lands in the high band — good enough to frame a look, not to
        // judge a detail transfer against.
        const img = new Image();
        await new Promise((res, rej) => {
            img.onload = res; img.onerror = rej;
            img.src = `data:image/jpeg;base64,${jpeg}`;
        });
        const back = document.createElement("canvas");
        back.width = img.naturalWidth; back.height = img.naturalHeight;
        const bctx = back.getContext("2d", { willReadFrequently: true });
        bctx.drawImage(img, 0, 0);
        return imageDataToSource(bctx.getImageData(0, 0, back.width, back.height));
    }

    node._batAdvBlendIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        state.meta = {
            w: Number(one(msg.w)) || 0,
            h: Number(one(msg.h)) || 0,
            frames: Number(one(msg.frames)) || 1,
            frame: Number(one(msg.preview_frame)) || 0,
        };

        const jpegA = one(msg.jpeg_a);
        const jpegB = one(msg.jpeg_b);
        state.a = await decodePlate(one(msg.tile_a), jpegA, "A");
        state.b = await decodePlate(one(msg.tile_b), jpegB, "B");
        if (!isNodeAlive(node)) return;
        // Build the ladder once per run rather than per repaint: halving a
        // 768x432 float buffer is not free, and a drag would otherwise pay for
        // it on every frame.
        const ladder = (src) => {
            const out = [src];
            for (let l = 1; l < MIP_LEVELS; l++) out.push(halve(out[l - 1]));
            return out;
        };
        state.aMips = state.a ? ladder(state.a) : null;
        state.bMips = state.b ? ladder(state.b) : null;

        // Hand the full tile to the worker. Copies, not transfers: the main
        // thread still needs these for the reduced settings, the A/B views and
        // the mask rasteriser. One 8MB copy per execution, against a repaint
        // that no longer blocks the UI, is a trade worth making every time.
        if (worker && state.a && state.b) {
            try {
                const mk = maskFor(state.a.width, state.a.height);
                worker.postMessage({
                    type: "tiles",
                    a: state.a.data.slice(), b: state.b.data.slice(),
                    mask: mk ? mk.slice() : null,
                    w: state.a.width, h: state.a.height,
                });
                workerReady = true;
                jobPending = null; jobInFlight = 0;
            } catch (e) {
                console.warn("[Bat_AdvancedBlend] could not seed the blend worker:", e);
                workerReady = false;
            }
        }

        const maskB64 = one(msg.mask_png);
        state.mask = null;
        if (maskB64) {
            const mimg = new Image();
            await new Promise((res, rej) => {
                mimg.onload = res; mimg.onerror = rej;
                mimg.src = `data:image/png;base64,${maskB64}`;
            });
            state.maskImg = mimg;
        } else {
            state.maskImg = null;
        }

        blurs.clear();
        // A new execution means new content, possibly a different frame of the
        // batch entirely. The render on screen is not a moment out of date, it
        // is of something else — so it goes, rather than being held the way a
        // widget change holds it.
        state.runId++;
        state.needsRun = false;
        dropFull();

        // Cache the JPEGs only. The two 16-bit tiles are megabytes and
        // localStorage is a ~5MB origin-wide budget shared with every other BAT
        // node's thumbnail cache, so stashing them there would evict the caches
        // that actually need to survive a reload.
        if (jpegA && jpegB) {
            try {
                localStorage.setItem(cacheKey, JSON.stringify({
                    jpeg_a: jpegA, jpeg_b: jpegB, meta: state.meta,
                }));
            } catch (_) { /* quota — the preview repopulates on the next run */ }
        }

        schedule(false);
    };

    node._batAdvBlendRestore = () => {
        let c = null;
        try { c = JSON.parse(localStorage.getItem(cacheKey) || "null"); } catch (_) {}
        if (!c?.jpeg_a || !c?.jpeg_b) return;
        node._batAdvBlendIngest({
            jpeg_a: [c.jpeg_a], jpeg_b: [c.jpeg_b],
            w: [c.meta?.w || 0], h: [c.meta?.h || 0],
            frames: [c.meta?.frames || 1], preview_frame: [c.meta?.frame || 0],
        });
    };

    // Re-present (not recompute) whenever the box changes: the blend is
    // resolution-bound to the tile, only the blit depends on the node size.
    try {
        const ro = new ResizeObserver(() => { present(); paintBadge(); });
        ro.observe(stage);
        track.observer(ro, stage);
    } catch (_) { /* no ResizeObserver: the rAF repaints still cover it */ }

    track.dispose(() => {
        clearTimeout(settle);
        dropFull();
        // attachZoomControl puts one listener on `document` (the X11
        // middle-click paste guard), so it does need tearing down — everything
        // else it binds dies with the widget's DOM.
        try { zoomCtl?.destroy?.(); } catch (_) {}
        state.a = state.b = state.aMips = state.bMips = null;
        state.out = state.high = null;
        state.mask = null; state.maskImg = null;
        blurs.clear();
    });

    return root;
}

app.registerExtension({
    name: "Bat_AdvancedBlend",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        registerCleanup(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = buildPreview(this);
            // Dual-mode sizing: Nodes 2.0 derives node height from
            // computeLayoutSize, so a bare addDOMWidget plus `this.size` leaves
            // the node and the widget disagreeing. See bat_node_layout.js.
            addBatDOMWidget(this, "bat_advblend_preview", "bat_advblend_preview", el, {
                minWidth: 380, height: 460, growable: true,
            });
            this._batAdvBlendWatch?.();
            markAdvancedWidgets(this);
            // Before configure() runs, so these really are the definition's
            // defaults — see snapshotWidgetDefaults().
            snapshotWidgetDefaults(this);
            this._batRefreshAdvBar?.();
            clampNodeSize(this, 380, 460);
            // Defer until node.id is finalised — LiteGraph assigns it as part
            // of construction, so the cache key isn't stable until after.
            setTimeout(() => this._batAdvBlendRestore?.(), 0);
            return r;
        };

        // `showAdvanced` is serialised by litegraph, so a workflow reopens with
        // the section as the artist left it — the header has to catch up. This
        // is also where impossible widget values get repaired, because it runs
        // after every path that could have written them.
        const onConfigured = nodeType.prototype.onAfterGraphConfigured;
        nodeType.prototype.onAfterGraphConfigured = function () {
            const res = onConfigured ? onConfigured.apply(this, arguments) : undefined;
            repairNode(this);
            this._batRefreshAdvBar?.();
            this._batAdvBlendRepaint?.();
            return res;
        };

        // A pasted node is configured without onAfterGraphConfigured firing, so
        // the repair has to hook the configure path too. Deferred, because the
        // frontend writes widget values through more than one mechanism and not
        // all of them have finished by the time configure() returns.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => {
                if (!isNodeAlive(this)) return;
                repairNode(this);
                this._batRefreshAdvBar?.();
                this._batAdvBlendRepaint?.();
            }, 0);
            return res;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && this._batAdvBlendIngest) this._batAdvBlendIngest(message);
            return r;
        };
    },
});
