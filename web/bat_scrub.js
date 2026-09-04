/**
 * BAT — a draggable value bar, for the on-node panels.
 *
 * Why not `<input type="range">`
 * -----------------------------
 * Three reasons, all of which bite hardest exactly where these controls live —
 * a narrow column inside a node.
 *
 * • **Only the thumb is grabbable.** A native range gives you a ~12px target in
 *   a 40px control. Clicking the track jumps but does not start a drag in every
 *   browser, so the gesture that feels obvious half works.
 * • **The drag dies when the pointer leaves.** A range input has no pointer
 *   capture, so sliding out of a cramped row — which is most attempts — drops
 *   the drag mid-gesture. That is the "doesn't catch the mouse" feeling.
 * • **Resolution is the control's width.** 100 steps through 40 pixels is 2.5%
 *   per pixel and no way to do better.
 *
 * This is the scrubber pattern instead: the WHOLE bar is the target, the pointer
 * is captured for the duration, and shift gives fine relative adjustment so
 * precision is not limited by width. Click to set, drag to adjust, shift-drag to
 * refine, double-click to reset, wheel to step, arrows when focused.
 *
 * Everything is computed from `getBoundingClientRect()` as a NORMALISED
 * position, which makes it immune to the graph zoom. Nodes 1.0 CSS-transforms
 * DOM widgets by the canvas scale, so any control doing its own arithmetic in
 * raw pixels has to divide that out; working in fractions of the measured rect
 * sidesteps it entirely.
 *
 * Shared deliberately: nine files in this pack still use a native range, and
 * they all have this problem. New panels should use this.
 */

/** Snap to the step grid and clamp, without leaving float dust behind. */
export function snapToStep(value, min, max, step) {
    let v = Number(value);
    if (!Number.isFinite(v)) v = min;
    if (step > 0) {
        v = Math.round((v - min) / step) * step + min;
        // Math.round(0.3/0.01)*0.01 is 0.30000000000000004; the decimal count
        // implied by the step is the precision the caller actually meant.
        const dp = Math.max(0, Math.min(10,
            Math.ceil(-Math.log10(step)) + 1));
        v = Number(v.toFixed(dp));
    }
    return Math.max(min, Math.min(max, v));
}

/** Absolute value for a pointer at `clientX` over a bar of `rect`. */
export function valueFromPosition(clientX, rect, min, max, step) {
    const w = Math.max(1, rect.width);
    const t = Math.max(0, Math.min(1, (clientX - rect.left) / w));
    return snapToStep(min + t * (max - min), min, max, step);
}

/**
 * Build the control.
 *
 * @param {object} o
 *   value, min, max, step        the usual
 *   defaultValue                 double-click target (defaults to `value`)
 *   format                       (v) => string shown in the bar
 *   onInput                      (v) => void, continuously during a drag
 *   onCommit                     (v) => void, once the gesture ends
 *   title                        tooltip; the gesture list is appended to it
 *   fineFactor                   shift-drag multiplier (default 0.2)
 * @returns {{el: HTMLElement, set: Function, get: Function}}
 */
export function makeScrubber(o = {}) {
    const min = Number.isFinite(o.min) ? o.min : 0;
    const max = Number.isFinite(o.max) ? o.max : 1;
    const step = Number.isFinite(o.step) && o.step > 0 ? o.step : 0.01;
    const fine = Number.isFinite(o.fineFactor) ? o.fineFactor : 0.2;
    const fmt = o.format || ((v) => v.toFixed(2));
    const def = Number.isFinite(o.defaultValue) ? o.defaultValue : o.value;

    let value = snapToStep(o.value, min, max, step);

    const el = document.createElement("div");
    el.tabIndex = 0;
    el.style.cssText = `position:relative; flex:1 1 60px; min-width:44px; height:15px;
        background:#1b1b1b; border:1px solid #333; border-radius:3px;
        overflow:hidden; cursor:ew-resize; user-select:none; touch-action:none;
        outline:none;`;
    el.title = (o.title ? o.title + "\n\n" : "")
        + "Click or drag anywhere on the bar. Shift-drag to refine, "
        + "double-click to reset, wheel to step, arrows when focused.";

    const fill = document.createElement("div");
    fill.style.cssText = `position:absolute; left:0; top:0; bottom:0;
        background:linear-gradient(90deg,#2b4a63,#3d6c92); pointer-events:none;`;
    const text = document.createElement("div");
    text.style.cssText = `position:absolute; inset:0; display:flex;
        align-items:center; justify-content:flex-end; padding-right:4px;
        font:10px monospace; color:#dde; pointer-events:none;
        text-shadow:0 1px 2px rgba(0,0,0,0.8);`;
    el.append(fill, text);

    function paint() {
        const t = max > min ? (value - min) / (max - min) : 0;
        fill.style.width = `${Math.max(0, Math.min(1, t)) * 100}%`;
        text.textContent = fmt(value);
    }

    function setValue(v, fireInput) {
        const next = snapToStep(v, min, max, step);
        if (next === value) return;
        value = next;
        paint();
        if (fireInput) o.onInput?.(value);
    }

    // ── gesture ──────────────────────────────────────────────────────────
    // `anchor` is re-established whenever the shift key changes state, so
    // switching into and out of fine mode mid-drag does not make the value
    // jump: each phase measures from where that phase started.
    let dragging = false;
    let anchor = null;   // {x, value, fine}

    function reanchor(clientX) {
        anchor = { x: clientX, value, fine: true };
    }

    el.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        // Stop litegraph reading this as a node drag, and the browser as a text
        // selection.
        e.stopPropagation();
        e.preventDefault();
        dragging = true;
        try { el.setPointerCapture(e.pointerId); } catch (_) {}
        if (e.shiftKey) {
            // Started in fine mode: refine from where it already is rather than
            // jumping to the click.
            reanchor(e.clientX);
        } else {
            anchor = null;
            setValue(valueFromPosition(e.clientX, el.getBoundingClientRect(),
                                       min, max, step), true);
        }
    });

    el.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        e.stopPropagation();
        e.preventDefault();
        const rect = el.getBoundingClientRect();
        if (e.shiftKey) {
            if (!anchor || !anchor.fine) reanchor(e.clientX);
            // Normalised by the rect, so the feel does not change with the
            // graph zoom.
            const dt = (e.clientX - anchor.x) / Math.max(1, rect.width);
            setValue(anchor.value + dt * (max - min) * fine, true);
        } else {
            if (anchor && anchor.fine) anchor = null;   // left fine mode
            setValue(valueFromPosition(e.clientX, rect, min, max, step), true);
        }
    });

    const end = (e) => {
        if (!dragging) return;
        dragging = false;
        anchor = null;
        try { el.releasePointerCapture(e.pointerId); } catch (_) {}
        o.onCommit?.(value);
    };
    el.addEventListener("pointerup", end);
    el.addEventListener("pointercancel", end);

    el.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        e.preventDefault();
        setValue(def, true);
        o.onCommit?.(value);
    });

    el.addEventListener("wheel", (e) => {
        // Swallowed so the graph underneath does not zoom instead.
        e.stopPropagation();
        e.preventDefault();
        const mult = e.shiftKey ? 10 : 1;
        setValue(value + (e.deltaY < 0 ? step : -step) * mult, true);
        o.onCommit?.(value);
    }, { passive: false });

    el.addEventListener("keydown", (e) => {
        const mult = e.shiftKey ? 10 : 1;
        let d = 0;
        if (e.key === "ArrowLeft" || e.key === "ArrowDown") d = -step * mult;
        else if (e.key === "ArrowRight" || e.key === "ArrowUp") d = step * mult;
        else if (e.key === "Home") { setValue(min, true); o.onCommit?.(value); return; }
        else if (e.key === "End") { setValue(max, true); o.onCommit?.(value); return; }
        else return;
        e.stopPropagation();
        e.preventDefault();
        setValue(value + d, true);
        o.onCommit?.(value);
    });
    el.addEventListener("focus", () => { el.style.borderColor = "#4a7fa8"; });
    el.addEventListener("blur", () => { el.style.borderColor = "#333"; });

    paint();
    return {
        el,
        get: () => value,
        /** Set without firing onInput — for syncing from outside. */
        set: (v) => setValue(v, false),
    };
}
