/**
 * Shared high-precision preview plumbing for the BAT grading nodes.
 *
 * The JPEG thumbnail these previews normally use is clamped, 8-bit and lossy,
 * so a 16-bit render and an 8-bit render reach the browser byte-identical —
 * no amount of exposing up or down in the viewer can tell them apart. The
 * Python side (bat_hdr_preview.py) now also ships a small zlib'd 16-bit tile
 * of the same frame; this module inflates it and hands back a float buffer.
 *
 * The rule that makes the whole thing worth doing: grade in float, quantise
 * to 8-bit only at putImageData. Your monitor is 8-bit either way — what
 * matters is that the arithmetic runs on the source's real levels, so a view
 * exposure pushed over a gradient bands on an 8-bit source and stays smooth
 * on a 16-bit one.
 *
 * Both Bat_Grade and Bat_AnimatedGrade use this, and their pixel loops both
 * consume `SourceBuffer` — RGB interleaved Float32, stride 3 — so the 8-bit
 * and high-precision paths share one code path instead of two that drift.
 */

// zlib.compress() on the Python side emits a zlib-wrapped stream, which is
// what "deflate" means in the Compression Streams API ("deflate-raw" is the
// headerless variant). Safari shipped this in 16.4; anything older falls back
// to the JPEG path rather than breaking the preview.
export function hdrSupported() {
    return typeof DecompressionStream === "function";
}

/**
 * Inflate a {data, w, h, lo, hi} tile into a SourceBuffer.
 * @returns {Promise<{data: Float32Array, width: number, height: number, hdr: boolean, lo: number, hi: number}>}
 */
export async function decodeHdrTile(tile) {
    const bin = atob(tile.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    const raw = await new Response(stream).arrayBuffer();

    // The tile is little-endian uint16; Uint16Array follows platform endianness,
    // so read through a DataView on any big-endian host rather than assuming.
    const count = raw.byteLength >> 1;
    let u16;
    if (_littleEndian()) {
        u16 = new Uint16Array(raw, 0, count);
    } else {
        const view = new DataView(raw);
        u16 = new Uint16Array(count);
        for (let i = 0; i < count; i++) u16[i] = view.getUint16(i * 2, true);
    }

    const expected = tile.w * tile.h * 3;
    if (count < expected) {
        throw new Error(`hdr tile short: got ${count} samples, expected ${expected}`);
    }

    // Undo the range normalisation the packer applied. This is where values
    // above white come back — the JPEG path had already clamped them away.
    const lo = Number(tile.lo) || 0;
    const hi = Number(tile.hi);
    const k = ((isFinite(hi) ? hi : lo + 1) - lo) / 65535;
    const out = new Float32Array(expected);
    for (let i = 0; i < expected; i++) out[i] = lo + u16[i] * k;

    return { data: out, width: tile.w, height: tile.h, hdr: true, lo, hi: lo + k * 65535 };
}

let _le = null;
function _littleEndian() {
    if (_le === null) {
        _le = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
    }
    return _le;
}

/**
 * Convert an 8-bit ImageData (RGBA, stride 4) into the same SourceBuffer shape
 * the high-precision path produces, so the pixel loops never branch on which
 * source they were handed.
 */
export function imageDataToSource(imageData) {
    const src = imageData.data;
    const n = imageData.width * imageData.height;
    const out = new Float32Array(n * 3);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        out[i * 3]     = src[p]     / 255;
        out[i * 3 + 1] = src[p + 1] / 255;
        out[i * 3 + 2] = src[p + 2] / 255;
    }
    return { data: out, width: imageData.width, height: imageData.height, hdr: false, lo: 0, hi: 1 };
}

/**
 * Build the viewer strip that sits under a grade preview: an Inspect toggle
 * (swap the 8-bit JPEG source for the high-precision tile) and a view exposure
 * in stops.
 *
 * The exposure is a VIEWER control, not a grade parameter — it is applied
 * after the grade, purely for display, and never reaches the render. That is
 * also why neither control is a litegraph widget: both nodes restore their
 * codec/keyframe values positionally from widgets_values, so adding a
 * serialised widget would shift every saved workflow's offsets. State lives
 * in localStorage per node instead, the same way the preview thumbnails do.
 */
export function buildInspectBar({ node, storageKey, onChange, hasHdr }) {
    const bar = document.createElement("div");
    bar.style.cssText = `
        display:flex; align-items:center; gap:8px; padding:4px 6px;
        background:#141414; border-top:1px solid #2a2a2a;
        font:11px monospace; color:#9aa; flex:0 0 auto; user-select:none;
    `;

    const saved = _loadState(storageKey);
    const state = {
        inspect: !!saved.inspect,
        exposure: Number.isFinite(saved.exposure) ? saved.exposure : 0,
    };

    const toggle = document.createElement("button");
    toggle.textContent = "Inspect";
    toggle.title =
        "Grade the 16-bit source instead of the 8-bit preview JPEG.\n" +
        "Combine with view exposure to check whether a render really carries " +
        "more than 8 bits: an 8-bit source bands, a 16-bit one stays smooth.";
    toggle.style.cssText = `
        background:#222; color:#9aa; border:1px solid #333; border-radius:3px;
        padding:2px 8px; font:11px monospace; cursor:pointer;
    `;

    const label = document.createElement("span");
    label.textContent = "exp";
    label.style.cssText = "opacity:0.7;";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-8"; slider.max = "8"; slider.step = "0.25";
    slider.value = String(state.exposure);
    slider.style.cssText = "flex:1 1 auto; min-width:60px; accent-color:#6af;";
    slider.title = "View exposure in stops — display only, never baked into the render.";

    const readout = document.createElement("span");
    readout.style.cssText = "min-width:52px; text-align:right; color:#cde;";

    const reset = document.createElement("button");
    reset.textContent = "0";
    reset.title = "Reset view exposure to 0 stops";
    reset.style.cssText = `
        background:#222; color:#9aa; border:1px solid #333; border-radius:3px;
        padding:2px 6px; font:11px monospace; cursor:pointer;
    `;

    bar.append(toggle, label, slider, readout, reset);

    function paintChrome() {
        const available = hasHdr();
        toggle.disabled = !available;
        toggle.style.opacity = available ? "1" : "0.4";
        toggle.style.cursor = available ? "pointer" : "default";
        const on = available && state.inspect;
        toggle.style.background = on ? "#2b4a63" : "#222";
        toggle.style.color = on ? "#cfe8ff" : "#9aa";
        toggle.style.borderColor = on ? "#4a7fa8" : "#333";
        if (!available) {
            toggle.title = hdrSupported()
                ? "Run the node once to load a high-precision tile."
                : "This browser has no DecompressionStream, so the high-precision tile can't be inflated.";
        }
        const sign = state.exposure > 0 ? "+" : "";
        readout.textContent = `${sign}${state.exposure.toFixed(2)} EV`;
        readout.style.color = state.exposure === 0 ? "#788" : "#cde";
    }

    function commit() {
        _saveState(storageKey, state);
        paintChrome();
        onChange?.();
    }

    toggle.addEventListener("click", () => {
        if (!hasHdr()) return;
        state.inspect = !state.inspect;
        commit();
    });
    slider.addEventListener("input", () => {
        state.exposure = parseFloat(slider.value) || 0;
        commit();
    });
    reset.addEventListener("click", () => {
        state.exposure = 0;
        slider.value = "0";
        commit();
    });
    // Stop litegraph from reading a drag on the slider as a node drag.
    for (const el of [slider, toggle, reset]) {
        el.addEventListener("pointerdown", (e) => e.stopPropagation());
    }

    paintChrome();

    return {
        el: bar,
        refresh: paintChrome,
        get inspect() { return state.inspect && hasHdr(); },
        // 0 EV must be exactly 1.0, not Math.pow(2, 0) rounding noise.
        get gain() { return state.exposure === 0 ? 1 : Math.pow(2, state.exposure); },
        get exposure() { return state.exposure; },
    };
}

function _loadState(key) {
    try { return JSON.parse(localStorage.getItem(key) || "{}") || {}; }
    catch (_) { return {}; }
}

function _saveState(key, state) {
    try { localStorage.setItem(key, JSON.stringify(state)); }
    catch (_) { /* quota / disabled storage — the control still works this session */ }
}
