/**
 * Bat_HDRTonalComposite — live canvas preview.
 *
 * Re-runs the whole composite in JS on a downscaled tile so the tonal
 * thresholds can be dialled in at slider speed instead of one graph run per
 * guess. The pixel path below mirrors `_composite_chunk` in the sibling .py;
 * if you change one, change the other.
 *
 * Why this node needs a *high-precision* preview more than the grade nodes do
 * -------------------------------------------------------------------------
 * Both inputs carry the information the artist is tuning against, and both
 * would be destroyed by the usual 8-bit JPEG thumbnail. The HDR input runs to
 * tens of units above white — JPEG it and every one of those values becomes
 * 255, so the preview would show a flat white sky and no amount of exposing
 * down would recover it. And the plate's crushed blacks are exactly the few
 * codes above zero that a JPEG's chroma subsampling throws away first.
 *
 * So both sides ship as zlib'd 16-bit range-normalised tiles (see
 * bat_hdr_preview.py) and every calculation here runs in Float32, quantising
 * to 8-bit only at the final putImageData.
 *
 * The viewer exposure
 * -------------------
 * Applied to the merged SCENE-LINEAR result, before the display transform —
 * which is the only position where it can do its job. Exposing down 3 stops
 * pulls a value of 8.0 into displayable range so you can see the structure the
 * HDR recovered above white; exposing up opens the blacks. Put the same gain
 * *after* the tonemap and both directions just wash out, because the tonemap
 * has already thrown the out-of-range data away.
 *
 * It is a viewer control, not a node parameter: it never touches the render,
 * and it is deliberately not a litegraph widget (state lives in localStorage
 * per node) so it cannot shift the positional widget offsets of saved
 * workflows. The node's own `preview_exposure` widget is the baked one.
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { hdrSupported, decodeHdrTile, imageDataToSource } from "./bat_hdr_preview.js";
import { batReplayLastExecution } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_HDRTonalComposite";
const MAX_HDR_VERSIONS = 8;   // must match MAX_HDR_VERSIONS in the .py
const ADD_BTN = "+ HDR version";
const RESET_BTN = "Reset to defaults";
const RESET_ARM = "Sure? \u2014 click again";
// How long the armed state lasts before it disarms itself. Long enough to be a
// deliberate second click, short enough that a stray click minutes later can
// never land on a primed button.
const RESET_ARM_MS = 4000;
const RM_BTN = "\u2212 HDR version";

/**
 * Make the slots match `node.properties.hdr_versions`.
 *
 * Idempotent and driven from one number, so create / configure / the buttons
 * all call the same thing and cannot disagree. Both the optional hdr_ai_*
 * inputs and the paired outputs are declared in Python, so the frontend spawns
 * ALL of them on creation — this trims back to the count rather than building
 * up from nothing.
 *
 * Outputs are only ever added or removed at the TAIL, in pairs. ComfyUI maps
 * outputs to RETURN_TYPES by index, and litegraph's removeOutput decrements
 * origin_slot on every link after the one removed — so trimming from the
 * middle would repoint a live linear_out link at an image_out.
 */
function syncVersions(node) {
    const want = Math.max(1, Math.min(node.properties?.hdr_versions | 0 || 1,
                                      MAX_HDR_VERSIONS));
    for (let i = MAX_HDR_VERSIONS; i >= 2; i--) {
        const at = node.findInputSlot ? node.findInputSlot(`hdr_ai_${i}`) : -1;
        if (i <= want && at === -1) node.addInput(`hdr_ai_${i}`, "IMAGE");
        else if (i > want && at !== -1) node.removeInput(at);
    }
    const wantOut = want * 2;
    while ((node.outputs?.length || 0) < wantOut) {
        const v = ((node.outputs.length / 2) | 0) + 1;
        node.addOutput(v === 1 ? "image_out" : `image_out_${v}`, "IMAGE");
        node.addOutput(v === 1 ? "linear_out" : `linear_out_${v}`, "IMAGE");
    }
    while ((node.outputs?.length || 0) > wantOut) {
        node.removeOutput(node.outputs.length - 1);
    }
    node.setDirtyCanvas?.(true, true);
}

/**
 * Widget name -> default, read from the node definition the server sent.
 *
 * Taken from `nodeData` rather than hardcoded, so the button cannot drift out
 * of step with the Python: change a default in INPUT_TYPES and this follows on
 * the next reload. Combos without an explicit default fall back to their first
 * option, which is what the frontend itself does when it builds the widget.
 */
function collectDefaults(nodeData) {
    const out = {};
    for (const group of ["required", "optional"]) {
        const spec = nodeData?.input?.[group];
        if (!spec) continue;
        for (const [name, entry] of Object.entries(spec)) {
            const [type, opts] = Array.isArray(entry) ? entry : [entry, undefined];
            if (opts && Object.prototype.hasOwnProperty.call(opts, "default")) {
                out[name] = opts.default;
            } else if (Array.isArray(type) && type.length) {
                out[name] = type[0];
            }
        }
    }
    return out;
}

/**
 * Restore every widget that has a known default.
 *
 * Goes through each widget's own callback rather than assigning `.value`
 * directly — that is what repaints the live canvas and what any other
 * extension hooked onto the widget expects. Slot COUNT is deliberately left
 * alone: how many hdr_ai inputs are wired is graph structure, not a value, and
 * silently dropping someone's connections is not what "reset the widgets"
 * means.
 */
function resetToDefaults(node, defaults) {
    let n = 0;
    for (const w of node.widgets || []) {
        if (!(w.name in defaults)) continue;      // buttons, the DOM canvas
        const v = defaults[w.name];
        if (w.value === v) continue;
        w.value = v;
        try { w.callback?.call(w, v); } catch (e) {
            console.warn(`[Bat_HDRTonalComposite] reset of ${w.name} threw:`, e);
        }
        n++;
    }
    node.setDirtyCanvas?.(true, true);
    return n;
}

function addResetButton(node, defaults) {
    let armed = 0, timer = null;
    const disarm = (btn) => {
        armed = 0;
        if (timer) { clearTimeout(timer); timer = null; }
        btn.name = RESET_BTN;
        node.setDirtyCanvas?.(true, true);
    };
    const btn = node.addWidget("button", RESET_BTN, null, () => {
        if (!armed) {
            // First click only arms it. Resetting two dozen tuned widgets is
            // not something to do on a mis-click.
            armed = 1;
            btn.name = RESET_ARM;
            node.setDirtyCanvas?.(true, true);
            timer = setTimeout(() => disarm(btn), RESET_ARM_MS);
            return;
        }
        const n = resetToDefaults(node, defaults);
        disarm(btn);
        console.log(`[Bat_HDRTonalComposite] reset ${n} widget(s) to defaults`);
    });
    btn.serialize = false;
    return btn;
}

function addVersionButtons(node) {
    const add = node.addWidget("button", ADD_BTN, null, () => {
        node.properties.hdr_versions =
            Math.min((node.properties.hdr_versions | 0 || 1) + 1, MAX_HDR_VERSIONS);
        syncVersions(node);
    });
    const rm = node.addWidget("button", RM_BTN, null, () => {
        node.properties.hdr_versions =
            Math.max((node.properties.hdr_versions | 0 || 1) - 1, 1);
        syncVersions(node);
    });
    // Buttons are UI, not parameters. Serialising them would shift every saved
    // workflow's positional widgets_values offsets.
    add.serialize = false; rm.serialize = false;
}

// Mirrors of the Python constants. Kept as literals with the same names so a
// change on either side is greppable across both files.
const EPS = 1e-6;
const BLACK_FLOOR = 1e-3;
const LUMA_R = 0.2126, LUMA_G = 0.7152, LUMA_B = 0.0722;

/**
 * Viewer display transform — how the canvas encodes for YOUR monitor.
 *
 * Display only. It never touches image_out, linear_out, or the tonal key, so
 * moving it cannot change a render or shift a threshold you have dialled in.
 *
 * "auto" follows what image_out is doing, which keeps the canvas honest for a
 * display-referred plate: decode and encode cancel and the Plate view is
 * byte-exact with the input. The exception is a scene-linear plate, where
 * there is no plate curve to match and image_out falls back to sRGB — there
 * auto picks Rec.709 instead, because anyone feeding scene-linear into a comp
 * node is far likelier to be on a Rec.709 grading monitor than a generic sRGB
 * desktop. That is the one case where the canvas and image_out differ, and
 * the bar says so when it happens.
 */
const VIEW_TRANSFORMS = [
    ["auto", "auto"], ["ocio", "OCIO"], ["srgb", "sRGB"], ["rec709", "Rec.709"],
    ["gamma_2_2", "2.2"], ["gamma_2_4", "2.4"], ["raw", "raw"],
];

/**
 * Inflate the 3D view LUT the backend baked out of the OCIO config.
 *
 * This is the only view option that is a real *transform* rather than a
 * curve. An ACES Output Transform is RRT + ODT — a filmic tone curve with a
 * path-to-white and gamut work — and no gamma setting is its inverse, which
 * is why swapping sRGB for Rec.709 does nothing to close the gap against a
 * viewer that applies one.
 */
async function decodeViewLut(payload) {
    const bin = atob(payload.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const stream = new Blob([bytes]).stream()
        .pipeThrough(new DecompressionStream("deflate"));
    const raw = await new Response(stream).arrayBuffer();
    const n = payload.size | 0;
    const want = n * n * n * 3;
    const u16 = new Uint16Array(raw, 0, raw.byteLength >> 1);
    if (u16.length < want) throw new Error(`view LUT short: ${u16.length} < ${want}`);
    const data = new Float32Array(want);
    for (let i = 0; i < want; i++) data[i] = u16[i] / 65535;
    return { data, size: n, minEv: +payload.minEv, maxEv: +payload.maxEv,
             name: payload.name, config: payload.config };
}

/**
 * Trilinear lookup, log2 shaper. Mirrors the axis the backend baked:
 * `t = (log2(max(x, 2^minEv)) - minEv) / (maxEv - minEv)`, r-major indexing.
 *
 * Trilinear rather than tetrahedral on purpose — measured against OCIO
 * evaluated directly, trilinear on a 33^3 grid is 1.27/255 worst case over a
 * real frame, and tetrahedral's extra accuracy only shows on synthetic
 * extreme-saturation samples that do not occur in a photograph. Not worth the
 * extra branchy code in a per-pixel loop.
 */
function lutApply(lut, rgb, out) {
    const n = lut.size, span = lut.maxEv - lut.minEv, floor = Math.pow(2, lut.minEv);
    const c = [0, 0, 0], i0 = [0, 0, 0], f = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
        const v = rgb[k] > floor ? rgb[k] : floor;
        let t = (Math.log2(v) - lut.minEv) / span;
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        const x = t * (n - 1);
        const i = Math.min(Math.floor(x), n - 2);
        c[k] = x; i0[k] = n === 1 ? 0 : i; f[k] = x - i0[k];
    }
    const d = lut.data;
    out[0] = out[1] = out[2] = 0;
    for (let dr = 0; dr < 2; dr++) {
        const wr = dr ? f[0] : 1 - f[0]; if (wr === 0) continue;
        const ir = Math.min(i0[0] + dr, n - 1);
        for (let dg = 0; dg < 2; dg++) {
            const wg = dg ? f[1] : 1 - f[1]; if (wg === 0) continue;
            const ig = Math.min(i0[1] + dg, n - 1);
            for (let db = 0; db < 2; db++) {
                const wb = db ? f[2] : 1 - f[2]; if (wb === 0) continue;
                const ib = Math.min(i0[2] + db, n - 1);
                const w = wr * wg * wb, o = ((ir * n + ig) * n + ib) * 3;
                out[0] += w * d[o]; out[1] += w * d[o + 1]; out[2] += w * d[o + 2];
            }
        }
    }
}

const VIEWS = [
    ["result", "Result", "The composite."],
    ["plate",  "Plate",  "The untouched plate, through the same display transform."],
    ["hdr",    "HDR",    "The exposure-matched LTX HDR input on its own. Expose down to see what is up there."],
    ["weight", "Weight", "The tonal key. White = HDR fully in, black = plate untouched.\nRed tint = shadow ramp, blue tint = highlight ramp."],
];

// ── transfer functions (mirrors of the Python) ───────────────────────────
function srgbToLinear(x) {
    return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function linearToSrgb(x) {
    return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}
// Mirrors _rec709_to_linear in the .py — the BT.709 OETF inverse, which is a
// different curve from sRGB (65% apart in the shadows), not a synonym for it.
function rec709ToLinear(x) {
    return x < 0.081 ? x / 4.5 : Math.pow((x + 0.099) / 1.099, 1 / 0.45);
}
function linearToRec709(x) {
    return x < 0.018 ? 4.5 * x : 1.099 * Math.pow(x, 0.45) - 0.099;
}
// Mirrors _encode_from_linear: the key is read in the PLATE's encoding, so a
// threshold means the same number the artist sees on a pixel probe.
function encodeFromLinear(x, mode) {
    if (x < 0) x = 0;
    if (mode === "rec709") return linearToRec709(x);
    if (mode === "gamma_2_2") return Math.pow(x, 1 / 2.2);
    if (mode === "gamma_2_4") return Math.pow(x, 1 / 2.4);
    return linearToSrgb(x);
}
function toLinear(x, mode) {
    if (x < 0) x = 0;
    if (mode === "linear") return x;
    if (mode === "rec709") return rec709ToLinear(x);
    if (mode === "gamma_2_2") return Math.pow(x, 2.2);
    if (mode === "gamma_2_4") return Math.pow(x, 2.4);
    return srgbToLinear(x);
}
function smoothstep(e0, e1, x) {
    let span = e1 - e0;
    if (Math.abs(span) < 1e-8) span = span >= 0 ? 1e-8 : -1e-8;
    let t = (x - e0) / span;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return t * t * (3 - 2 * t);
}
// Mirrors _SHOULDER_TARGET / _knee_from_headroom / _soft_rolloff in the .py.
const SHOULDER_TARGET = 0.95;
function kneeFromHeadroom(stops) {
    if (!(stops > 0)) return 1.0;
    const W = Math.pow(2, stops), T = SHOULDER_TARGET;
    const k = T - Math.sqrt(Math.max(T * T - T + W * (1 - T), 0));
    return Math.min(Math.max(k, 0.02), 0.999);
}
function softRolloff(x, knee) {
    if (knee >= 1.0) return x > 1 ? 1 : x;
    if (knee < 0) knee = 0;
    if (x <= knee) return x;
    const s = 1 - knee, u = (x - knee) / s;
    return knee + s * u / (1 + u);
}

/** Separable Gaussian on a Float32 scalar map, reflect-padded. Mirrors
 *  _gaussian_blur_hw. Radius is in TILE pixels — the caller scales it down
 *  from the widget's full-resolution value. */
function blurMap(src, w, h, radius) {
    if (radius <= 0) return src;
    const sigma = Math.max(radius / 2, 0.5);
    const build = (r) => {
        const k = new Float32Array(2 * r + 1);
        let s = 0;
        for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k[i + r] = v; s += v; }
        for (let i = 0; i < k.length; i++) k[i] /= s;
        return k;
    };
    const refl = (i, n) => (i < 0 ? -i : (i >= n ? 2 * n - 2 - i : i));
    let cur = src;
    const rx = Math.min(radius, Math.max(w - 1, 0));
    if (rx > 0) {
        const k = build(rx), tmp = new Float32Array(w * h);
        for (let y = 0; y < h; y++) {
            const row = y * w;
            for (let x = 0; x < w; x++) {
                let a = 0;
                for (let i = -rx; i <= rx; i++) a += cur[row + refl(x + i, w)] * k[i + rx];
                tmp[row + x] = a;
            }
        }
        cur = tmp;
    }
    const ry = Math.min(radius, Math.max(h - 1, 0));
    if (ry > 0) {
        const k = build(ry), tmp = new Float32Array(w * h);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let a = 0;
                for (let i = -ry; i <= ry; i++) a += cur[refl(y + i, h) * w + x] * k[i + ry];
                tmp[y * w + x] = a;
            }
        }
        cur = tmp;
    }
    return cur;
}

// ── preview cache (JPEG only — see the note in bat_grade.js) ─────────────
const cacheKey = (node) => `bat_hdrcomp_preview_${node?.id ?? "_"}`;
function saveCache(node, d) { try { localStorage.setItem(cacheKey(node), JSON.stringify(d)); } catch (_) {} }
function loadCache(node) {
    try { const r = localStorage.getItem(cacheKey(node)); return r ? JSON.parse(r) : null; }
    catch (_) { return null; }
}
const viewKey = (node) => `bat_hdrcomp_view_${node?.id ?? "_"}`;

function buildPreview(node) {
    const root = document.createElement("div");
    root.style.cssText = `position:relative; display:flex; flex-direction:column;
        background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; overflow:hidden;`;

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative; flex:1 1 auto; min-height:0; display:flex; background:#000;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%; height:100%; object-fit:contain; display:block; image-rendering:auto;";
    stage.appendChild(canvas);

    const hint = document.createElement("div");
    hint.style.cssText = `position:absolute; left:6px; bottom:4px; font:11px monospace;
        color:#9aa; pointer-events:none; text-shadow:0 1px 2px #000;`;
    hint.textContent = "Run once to populate the preview.";
    stage.appendChild(hint);

    // Pixel probe. The whole node is an argument about values in the extremes,
    // so being able to read one is worth the few lines.
    const probe = document.createElement("div");
    probe.style.cssText = `position:absolute; right:6px; top:4px; font:10px monospace;
        color:#cde; background:rgba(0,0,0,0.62); padding:3px 6px; border-radius:3px;
        pointer-events:none; white-space:pre; display:none; line-height:1.45;`;
    stage.appendChild(probe);

    root.appendChild(stage);

    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    const saved = (() => { try { return JSON.parse(localStorage.getItem(viewKey(node)) || "{}") || {}; } catch (_) { return {}; } })();
    const state = {
        plate: null,       // SourceBuffer, display-referred plate
        hdr: null,         // SourceBuffer, scene-linear HDR
        meta: null,        // {k_batch, k_valid, k_mid_low, k_mid_high, k_gamma_mode, w, h, frames}
        view: VIEWS.some(v => v[0] === saved.view) ? saved.view : "result",
        exposure: Number.isFinite(saved.exposure) ? saved.exposure : 0,
        viewTransform: VIEW_TRANSFORMS.some(v => v[0] === saved.viewTransform)
            ? saved.viewTransform : "auto",
        renderMode: null, resolvedView: null,
        viewLut: null,          // decoded 3D LUT, or null until a run ships one
        holding: false,    // mouse held on the canvas -> temporarily show the plate
        // Last computed frame, kept so the probe can report real numbers
        // rather than re-deriving them from the 8-bit canvas.
        lin: null, wmap: null, wsh: null, whi: null, tw: 0, th: 0,
    };
    node._batHdrCompState = state;

    const W = (n) => node.widgets?.find(w => w.name === n);
    const num = (n, d) => { const w = W(n); const v = w ? +w.value : NaN; return Number.isFinite(v) ? v : d; };
    const str = (n, d) => { const w = W(n); return w ? String(w.value) : d; };
    const bool = (n, d) => { const w = W(n); return w ? !!w.value : d; };

    // ── the composite, mirroring _composite_chunk ────────────────────────
    function applyComposite() {
        const P = state.plate, Hs = state.hdr;
        if (!P) return;
        const w = P.width, h = P.height, n = w * h;
        const haveHdr = !!Hs && Hs.width === w && Hs.height === h;

        const gammaMode = str("plate_gamma_mode", "srgb");
        const shStart = num("shadow_start", 0.12), shFull = num("shadow_full", 0.03);
        const shStr = num("shadow_strength", 1), hiStr = num("highlight_strength", 1);
        const hiStart = num("highlight_start", 0.75), hiFull = num("highlight_full", 0.98);
        const shTone = num("shadow_tone_transfer", 1), hiTone = num("highlight_tone_transfer", 0);
        const detTr = num("detail_transfer", 1);
        const keepChroma = num("preserve_plate_chroma", 0.85);
        const rgbMix = num("hdr_rgb_mix", 0.25);
        const maxDetailGain = num("max_detail_gain", 4);
        const hdrCeiling = num("hdr_ceiling", 0);
        const blurR = Math.round(num("blur_radius", 0));
        const detR = Math.round(num("detail_radius", 16));
        const tonemap = str("preview_tonemap", "soft_rolloff");
        const outModeRaw = str("output_gamma_mode", "match_plate");
        const renderMode = outModeRaw === "match_plate" ? gammaMode : outModeRaw;
        // What image_out will encode with, vs what this canvas encodes with.
        let vt = state.viewTransform || "auto";
        if (vt === "ocio" && !state.viewLut) vt = "auto";   // not shipped yet
        // auto prefers the OCIO LUT whenever one has been shipped: setting
        // preview_ocio_view IS the artist saying "show me my view transform",
        // and making them also change a second control here just meant the
        // widget looked broken. Pick a specific mode to override.
        const outMode = vt !== "auto" ? vt
            : state.viewLut ? "ocio"
            : (renderMode === "linear" ? "rec709" : renderMode);
        state.renderMode = renderMode;
        state.resolvedView = outMode;
        const knee = kneeFromHeadroom(num("highlight_headroom", 2.0));
        const bakedExp = Math.pow(2, num("preview_exposure", 0));
        const viewExp = Math.pow(2, state.exposure);

        // Radii are authored against the full-resolution frame; the tile is a
        // fraction of it, so a radius of 16 would blur the whole preview.
        const scale = state.meta?.w ? (w / state.meta.w) : 1;
        const detRT = Math.max(0, Math.round(detR * scale));
        const blurRT = Math.max(0, Math.round(blurR * scale));

        // 1 ── decode the plate, derive luminances and the tonal key.
        const plateLin = new Float32Array(n * 3);
        const ySrc = new Float32Array(n);
        const yKey = new Float32Array(n);
        for (let i = 0, p = 0; i < n; i++, p += 3) {
            const r = toLinear(P.data[p], gammaMode);
            const g = toLinear(P.data[p + 1], gammaMode);
            const b = toLinear(P.data[p + 2], gammaMode);
            plateLin[p] = r; plateLin[p + 1] = g; plateLin[p + 2] = b;
            const y = r * LUMA_R + g * LUMA_G + b * LUMA_B;
            ySrc[i] = y;
            yKey[i] = encodeFromLinear(y, gammaMode);   // key in the plate's encoding
        }

        // 2 ── the HDR side, with the midtone exposure match.
        const hdrM = new Float32Array(n * 3);
        const yHdr = new Float32Array(n);
        let k = 1.0;
        if (haveHdr && bool("auto_match_mids", true)) k = matchRatio(yKey, ySrc, Hs, n);
        if (haveHdr) {
            for (let i = 0, p = 0; i < n; i++, p += 3) {
                const r = Math.max(0, Hs.data[p] * k), g = Math.max(0, Hs.data[p + 1] * k),
                      b = Math.max(0, Hs.data[p + 2] * k);
                hdrM[p] = r; hdrM[p + 1] = g; hdrM[p + 2] = b;
                yHdr[i] = r * LUMA_R + g * LUMA_G + b * LUMA_B;
            }
        }

        // 3 ── the two tonal ramps.
        let wSh = new Float32Array(n), wHi = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const y = yKey[i];
            let a = (1 - smoothstep(shFull, shStart, y)) * shStr;
            let b = smoothstep(hiStart, hiFull, y) * hiStr;
            wSh[i] = a < 0 ? 0 : (a > 1 ? 1 : a);
            wHi[i] = b < 0 ? 0 : (b > 1 ? 1 : b);
        }
        if (blurRT > 0) { wSh = blurMap(wSh, w, h, blurRT); wHi = blurMap(wHi, w, h, blurRT); }

        const wmap = new Float32Array(n);
        for (let i = 0; i < n; i++) wmap[i] = Math.max(wSh[i], wHi[i]);

        // 4 ── level, then local contrast, then the safety cap.
        // The level stage is a lerp, so it is bounded by the inputs and needs
        // no ceiling — capping it here would truncate the HDR's scene range.
        const yMix = new Float32Array(n);
        const yLevel = new Float32Array(n);
        const aTone = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            let a = Math.max(wSh[i] * shTone, wHi[i] * hiTone);
            a = a < 0 ? 0 : (a > 1 ? 1 : a);
            aTone[i] = a;
            yLevel[i] = haveHdr ? ySrc[i] + (yHdr[i] - ySrc[i]) * a : ySrc[i];
            yMix[i] = yLevel[i];
        }
        if (haveHdr && detTr > 0 && detRT > 0) {
            const bs = blurMap(ySrc, w, h, detRT), bh = blurMap(yHdr, w, h, detRT);
            for (let i = 0; i < n; i++) {
                const dSrc = Math.log(Math.max(ySrc[i], 0) + BLACK_FLOOR) - Math.log(Math.max(bs[i], 0) + BLACK_FLOOR);
                const dHdr = Math.log(Math.max(yHdr[i], 0) + BLACK_FLOOR) - Math.log(Math.max(bh[i], 0) + BLACK_FLOOR);
                const extra = Math.max(wmap[i] * detTr - aTone[i], 0);
                const lim = Math.log(Math.max(maxDetailGain, 1 + 1e-6));
                let boost = extra * (dHdr - dSrc);
                if (boost > lim) boost = lim; else if (boost < -lim) boost = -lim;
                yMix[i] = yLevel[i] * Math.exp(boost);
            }
        }
        for (let i = 0; i < n; i++) {
            let v = yMix[i];
            if (hdrCeiling > 0 && v > hdrCeiling) v = hdrCeiling;
            yMix[i] = v < 0 ? 0 : v;
        }

        // 5 ── reach the target luminance without moving the colour.
        const lin = new Float32Array(n * 3);
        for (let i = 0, p = 0; i < n; i++, p += 3) {
            const ys = ySrc[i], yh = yHdr[i];
            const conf = ys / (ys + BLACK_FLOOR);
            const c = conf * keepChroma;
            const invS = ys > EPS ? 1 / ys : 0, invH = yh > EPS ? 1 / yh : 0;
            const d = yMix[i] - ys;
            for (let ch = 0; ch < 3; ch++) {
                const dp = ys > EPS ? plateLin[p + ch] * invS : 1;
                const dh = (haveHdr && yh > EPS) ? hdrM[p + ch] * invH : 1;
                let v = plateLin[p + ch] + d * (dp * c + dh * (1 - c));
                lin[p + ch] = v < 0 ? 0 : v;
            }
            // 6 ── the HDR's colour at our luminance (never its brightness).
            if (haveHdr && rgbMix > 0 && yh > EPS) {
                let m = wmap[i] * rgbMix; m = m < 0 ? 0 : (m > 1 ? 1 : m);
                const s = yMix[i] * invH;
                for (let ch = 0; ch < 3; ch++) lin[p + ch] = lin[p + ch] * (1 - m) + hdrM[p + ch] * s * m;
            }
        }

        state.lin = lin; state.wmap = wmap; state.wsh = wSh; state.whi = wHi;
        state.tw = w; state.th = h;

        // 7 ── display. Viewer exposure goes in BEFORE the tonemap, which is
        // the only place it can reveal what sits above white.
        const showPlate = state.holding || state.view === "plate";
        const out = ctx.createImageData(w, h);
        const dst = out.data;
        const display = (v) => {
            let x = v * bakedExp * viewExp;
            if (tonemap === "reinhard") x = x / (1 + x);
            else if (tonemap === "soft_rolloff") x = softRolloff(x, knee);
            x = x < 0 ? 0 : (x > 1 ? 1 : x);
            const s = outMode === "raw" ? x : encodeFromLinear(x, outMode);
            return s > 1 ? 255 : (s < 0 ? 0 : (s * 255 + 0.5) | 0);
        };

        // The LUT is a 3-channel transform, so it cannot go through the
        // per-channel `display()` above — it replaces the tonemap AND the
        // encode in one lookup.
        const useLut = outMode === "ocio" && state.viewLut;
        const lutIn = [0, 0, 0], lutOut = [0, 0, 0];
        const gain = bakedExp * viewExp;
        const writeLut = (src, p, dst, q) => {
            lutIn[0] = src[p] * gain; lutIn[1] = src[p + 1] * gain; lutIn[2] = src[p + 2] * gain;
            lutApply(state.viewLut, lutIn, lutOut);
            for (let k = 0; k < 3; k++) {
                const v = lutOut[k];
                dst[q + k] = v > 1 ? 255 : (v < 0 ? 0 : (v * 255 + 0.5) | 0);
            }
        };

        for (let i = 0, p = 0, q = 0; i < n; i++, p += 3, q += 4) {
            if (state.view === "weight" && !state.holding) {
                // Greyscale weight, tinted so the two ramps stay tellable
                // apart at a glance while tuning them.
                const g = (wmap[i] * 255 + 0.5) | 0;
                const sh = wSh[i] > wHi[i];
                dst[q] = sh ? g : (g * 0.55) | 0;
                dst[q + 1] = (g * 0.75) | 0;
                dst[q + 2] = sh ? (g * 0.55) | 0 : g;
            } else {
                const src = showPlate ? plateLin
                          : (state.view === "hdr" && haveHdr) ? hdrM : lin;
                if (useLut) {
                    writeLut(src, p, dst, q);
                } else {
                    dst[q] = display(src[p]);
                    dst[q + 1] = display(src[p + 1]);
                    dst[q + 2] = display(src[p + 2]);
                }
            }
            dst[q + 3] = 255;
        }
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        ctx.putImageData(out, 0, 0);
        hint.style.display = haveHdr ? "none" : "block";
        if (!haveHdr) {
            hint.textContent = state.plate
                ? "hdr_ai tile missing — showing the plate only. Re-run the node."
                : "Run once to populate the preview.";
        }
        bar.refresh();
    }

    /** Midtone exposure-match ratio. Prefers the exact whole-batch value the
     *  backend computed, but only while the widgets it was derived from still
     *  hold the same values — otherwise dragging mid_low would show a stale
     *  match. Falls back to deriving it from this one tile. */
    function matchRatio(yKey, ySrc, Hs, n) {
        const midLow = num("mid_low", 0.10), midHigh = num("mid_high", 0.70);
        const m = state.meta;
        if (m && m.k_valid && str("match_scope", "whole_batch") === "whole_batch"
            && Math.abs(m.k_mid_low - midLow) < 1e-9 && Math.abs(m.k_mid_high - midHigh) < 1e-9
            && m.k_gamma_mode === str("plate_gamma_mode", "srgb")) {
            return m.k_batch;
        }
        const F = 0.05;
        const loHi = Math.min(midLow + F, midHigh), hiLo = Math.max(midHigh - F, midLow);
        let numr = 0, den = 0;
        for (let i = 0, p = 0; i < n; i++, p += 3) {
            const mask = smoothstep(midLow, loHi, yKey[i]) * smoothstep(midHigh, hiLo, yKey[i]);
            if (mask <= 0) continue;
            numr += ySrc[i] * mask;
            den += (Hs.data[p] * LUMA_R + Hs.data[p + 1] * LUMA_G + Hs.data[p + 2] * LUMA_B) * mask;
        }
        if (!(den > EPS) || !(numr > 0)) return 1;
        return Math.min(Math.max(numr / den, 1 / 64), 64);
    }

    // ── viewer strip ─────────────────────────────────────────────────────
    const bar = (() => {
        const el = document.createElement("div");
        el.style.cssText = `display:flex; align-items:center; gap:6px; padding:4px 6px;
            background:#141414; border-top:1px solid #2a2a2a; font:11px monospace;
            color:#9aa; flex:0 0 auto; user-select:none;`;

        const btns = VIEWS.map(([id, label, tip]) => {
            const b = document.createElement("button");
            b.textContent = label; b.title = tip;
            b.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
                border-radius:3px; padding:2px 7px; font:11px monospace; cursor:pointer;`;
            b.addEventListener("click", () => { state.view = id; commit(); });
            b.addEventListener("pointerdown", (e) => e.stopPropagation());
            el.appendChild(b);
            return [id, b];
        });

        const label = document.createElement("span");
        label.textContent = "view exp"; label.style.cssText = "opacity:0.7; margin-left:4px;";
        const slider = document.createElement("input");
        slider.type = "range"; slider.min = "-10"; slider.max = "10"; slider.step = "0.25";
        slider.value = String(state.exposure);
        slider.style.cssText = "flex:1 1 auto; min-width:56px; accent-color:#6af;";
        slider.title = "Viewer exposure in stops — display only, never baked into the render.\n"
            + "Expose DOWN to see the highlight detail the HDR recovered above white.\n"
            + "Expose UP to check the shadows. Applied before the tonemap, so both directions work.";
        const readout = document.createElement("span");
        readout.style.cssText = "min-width:50px; text-align:right; color:#cde;";
        const reset = document.createElement("button");
        reset.textContent = "0"; reset.title = "Reset viewer exposure";
        reset.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
            border-radius:3px; padding:2px 6px; font:11px monospace; cursor:pointer;`;

        const vtSel = document.createElement("select");
        vtSel.style.cssText = `background:#222; color:#9aa; border:1px solid #333;
            border-radius:3px; font:11px monospace; padding:1px 2px; cursor:pointer;`;
        for (const [id, txt] of VIEW_TRANSFORMS) {
            const o = document.createElement("option");
            o.value = id; o.textContent = txt; vtSel.appendChild(o);
        }
        vtSel.value = state.viewTransform;

        el.append(label, slider, readout, reset, vtSel);

        function refresh() {
            for (const [id, b] of btns) {
                const on = state.view === id;
                b.style.background = on ? "#2b4a63" : "#222";
                b.style.color = on ? "#cfe8ff" : "#9aa";
                b.style.borderColor = on ? "#4a7fa8" : "#333";
            }
            const sign = state.exposure > 0 ? "+" : "";
            readout.textContent = `${sign}${state.exposure.toFixed(2)}`;
            readout.style.color = state.exposure === 0 ? "#788" : "#cde";

            // Flag it when the canvas is not encoding the way image_out will,
            // so a mismatch is never silent.
            const differs = state.resolvedView && state.renderMode &&
                            state.resolvedView !== state.renderMode;
            const lutOn = state.resolvedView === "ocio";
            vtSel.style.borderColor = differs ? "#7a6a3a" : "#333";
            vtSel.style.color = differs ? "#d8c48a" : "#9aa";
            vtSel.title =
                "Display transform for THIS canvas only — never baked into "
                + "image_out, linear_out, or the tonal key.\n"
                + `auto follows image_out (currently ${state.renderMode || "?"}), `
                + "except on a scene-linear plate where it uses Rec.709 rather "
                + "than image_out's sRGB fallback.\n"
                + (lutOn
                    ? `Showing the ${state.viewLut?.name || "OCIO"} view transform `
                      + `baked from ${state.viewLut?.config || "your config"} — `
                      + `image_out will be ${state.renderMode}.`
                    : differs
                    ? `Showing ${state.resolvedView} — image_out will be ${state.renderMode}.`
                    : "Matching image_out.");
        }
        function commit() {
            try {
                localStorage.setItem(viewKey(node), JSON.stringify({
                    view: state.view, exposure: state.exposure,
                    viewTransform: state.viewTransform,
                }));
            } catch (_) {}
            refresh(); schedule();
        }
        slider.addEventListener("input", () => { state.exposure = parseFloat(slider.value) || 0; commit(); });
        vtSel.addEventListener("change", () => { state.viewTransform = vtSel.value; commit(); });
        reset.addEventListener("click", () => { state.exposure = 0; slider.value = "0"; commit(); });
        for (const e of [slider, reset, vtSel]) e.addEventListener("pointerdown", (ev) => ev.stopPropagation());
        refresh();
        return { el, refresh };
    })();
    root.appendChild(bar.el);

    // Hold anywhere on the image to flip back to the plate — the fastest
    // possible A/B, and the one an artist reaches for constantly.
    canvas.addEventListener("pointerdown", (e) => {
        e.stopPropagation(); state.holding = true; applyComposite();
    });
    const release = () => { if (state.holding) { state.holding = false; applyComposite(); } };
    window.addEventListener("pointerup", release);
    canvas.addEventListener("pointerleave", () => { probe.style.display = "none"; release(); });

    // Value probe.
    canvas.addEventListener("pointermove", (e) => {
        if (!state.lin || !state.tw) return;
        const r = canvas.getBoundingClientRect();
        // object-fit: contain — find the letterboxed content box.
        const sc = Math.min(r.width / state.tw, r.height / state.th);
        const cw = state.tw * sc, ch = state.th * sc;
        const x = Math.floor((e.clientX - r.left - (r.width - cw) / 2) / sc);
        const y = Math.floor((e.clientY - r.top - (r.height - ch) / 2) / sc);
        if (x < 0 || y < 0 || x >= state.tw || y >= state.th) { probe.style.display = "none"; return; }
        const i = y * state.tw + x, p = i * 3;
        const f = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4));
        probe.textContent =
            `linear  ${f(state.lin[p])} ${f(state.lin[p + 1])} ${f(state.lin[p + 2])}\n` +
            `weight  ${state.wmap[i].toFixed(3)}  (sh ${state.wsh[i].toFixed(2)} / hi ${state.whi[i].toFixed(2)})`;
        probe.style.display = "block";
    });

    let raf = 0;
    function schedule() {
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; try { applyComposite(); } catch (err) { console.error("[Bat_HDRTonalComposite] preview failed:", err); } });
    }

    node._batHdrCompWatch = () => {
        for (const name of ["plate_gamma_mode", "auto_match_mids", "match_scope",
                            "mid_low", "mid_high", "shadow_start", "shadow_full",
                            "shadow_strength", "highlight_start", "highlight_full",
                            "highlight_strength", "shadow_tone_transfer",
                            "highlight_tone_transfer", "detail_transfer", "detail_radius",
                            "preserve_plate_chroma", "hdr_rgb_mix", "max_detail_gain", "hdr_ceiling",
                            "blur_radius", "preview_tonemap", "highlight_headroom", "output_gamma_mode",
                            "preview_exposure"]) {
            const wd = W(name);
            if (!wd) continue;
            const orig = wd.callback;
            wd.callback = function (v) { try { orig?.call(this, v); } catch (_) {} schedule(); };
        }
    };

    node._batHdrCompIngest = async (msg) => {
        const one = (v) => (Array.isArray(v) ? v[0] : v);
        state.meta = {
            k_batch: Number(one(msg.k_batch)) || 1,
            k_valid: !!one(msg.k_valid),
            k_mid_low: Number(one(msg.k_mid_low)),
            k_mid_high: Number(one(msg.k_mid_high)),
            k_gamma_mode: one(msg.k_gamma_mode),
            w: Number(one(msg.w)) || 0,
            h: Number(one(msg.h)) || 0,
            frames: Number(one(msg.frames)) || 1,
        };

        const plateTile = one(msg.plate_tile);
        const hdrT = one(msg.hdr_tile);
        const jpeg = one(msg.plate_jpeg);

        state.plate = null;
        if (plateTile && hdrSupported()) {
            try { state.plate = await decodeHdrTile(plateTile); }
            catch (e) { console.warn("[Bat_HDRTonalComposite] plate tile:", e); }
        }
        if (!state.plate && jpeg) {
            // Fallback only. The JPEG is clamped and 8-bit, so the blacks it
            // shows are not the blacks the node will act on — good enough to
            // frame a shot, not to trust a threshold against.
            const img = new Image();
            await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = `data:image/jpeg;base64,${jpeg}`; });
            const back = document.createElement("canvas");
            back.width = img.naturalWidth; back.height = img.naturalHeight;
            const bctx = back.getContext("2d", { willReadFrequently: true });
            bctx.drawImage(img, 0, 0);
            state.plate = imageDataToSource(bctx.getImageData(0, 0, back.width, back.height));
        }

        const lutPayload = one(msg.view_lut);
        if (lutPayload && hdrSupported()) {
            try { state.viewLut = await decodeViewLut(lutPayload); }
            catch (e) {
                console.warn("[Bat_HDRTonalComposite] view LUT failed to decode:", e);
                state.viewLut = null;
            }
        } else if (lutPayload === undefined) {
            // preview_ocio_view is off this run. Keep whatever we had so the
            // canvas does not lose its look mid-session, but a run that
            // explicitly ships nothing clears it.
        } else {
            state.viewLut = null;
        }

        state.hdr = null;
        if (hdrT && hdrSupported()) {
            try { state.hdr = await decodeHdrTile(hdrT); }
            catch (e) { console.warn("[Bat_HDRTonalComposite] hdr tile:", e); }
        }

        if (jpeg) saveCache(node, { jpeg, meta: state.meta });
        schedule();
    };

    // Restore something to look at on workflow reopen. Only the plate JPEG is
    // cached — the two 16-bit tiles are a few hundred KB each and localStorage
    // is a ~5MB origin-wide budget shared with every other BAT node's cache.
    node._batHdrCompRestore = () => {
        const c = loadCache(node);
        if (!c?.jpeg) return;
        node._batHdrCompIngest({ plate_jpeg: [c.jpeg], ...(c.meta ? {
            w: [c.meta.w], h: [c.meta.h], frames: [c.meta.frames], k_batch: [1], k_valid: [false],
        } : {}) });
    };

    return root;
}

app.registerExtension({
    name: "Bat_HDRTonalComposite",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;
        const DEFAULTS = collectDefaults(nodeData);

        // A graph reload (Ctrl+Z is one) destroys and rebuilds every node, so
        // replay the last run's preview payload into the new instance.
        batReplayLastExecution(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.properties = this.properties || {};
            if (typeof this.properties.hdr_versions !== "number")
                this.properties.hdr_versions = 1;
            addVersionButtons(this);
            addResetButton(this, DEFAULTS);
            setTimeout(() => syncVersions(this), 0);
            const el = buildPreview(this);
            addBatDOMWidget(this, "bat_hdrcomp_preview", "bat_hdrcomp_preview", el, {
                minWidth: 380, height: 460, growable: true,
            });
            this._batHdrCompWatch?.();
            clampNodeSize(this, 380, 460);
            setTimeout(() => this._batHdrCompRestore?.(), 0);
            return r;
        };

        // A saved workflow restores its own slot list AND the property; make
        // the two agree once configure() has run.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this.properties = this.properties || {};
            const saved = info?.properties?.hdr_versions;
            if (typeof saved === "number") {
                this.properties.hdr_versions = saved;
            } else if (typeof this.properties.hdr_versions !== "number") {
                // Pre-dates the feature: infer from the slots it was saved with
                // so an old workflow does not silently lose its extra outputs.
                this.properties.hdr_versions =
                    Math.max(1, Math.floor((this.outputs?.length || 2) / 2));
            }
            syncVersions(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && this._batHdrCompIngest) this._batHdrCompIngest(message);
            return r;
        };
    },
});
