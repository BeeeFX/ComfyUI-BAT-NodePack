/**
 * BAT — Bat_VideoLoader front-end.
 *
 * Replaces three declared widgets on the node with two custom ones:
 *   1. `path`        → BAT.PATH (rounded textbox + path-autocomplete popup,
 *                      mirrors Volt Loader behaviour but uses /bat/getpath)
 *   2. `start_frame` ┐
 *   3. `end_frame`   ┘ → BAT.TRIM (single dual-handle range slider that
 *                       drives the two underlying INT widgets, plus a
 *                       first-frame / last-frame preview row above)
 *
 * The underlying INT widgets are kept (and hidden from the canvas) so
 * the values still serialize and reach the back-end at execute time.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Bat_VideoLoader";
const PATH_ROUTE   = "/bat/getpath";
const INFO_ROUTE   = "/bat/video-info";
const FRAME_ROUTE  = "/bat/video-frame";
const STREAM_ROUTE = "/bat/video-stream";

// ─── Path autocomplete popup (port of VOLTPATH searchBox) ───────────────────

function pathStem(p) {
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? [p.slice(0, i + 1), p.slice(i + 1)] : ["", p];
}

function openPathSearch(event, widget) {
    if (widget._prompt) return true;
    widget._prompt = true;

    const dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded";
    dialog.innerHTML =
        '<span class="name">Video Path</span>' +
        '<input autofocus type="text" class="value">' +
        '<button class="rounded">OK</button>' +
        '<div class="helper"></div>';
    dialog.close = () => { dialog.remove(); widget._prompt = false; };
    document.body.append(dialog);
    if (app.canvas.ds.scale > 1) dialog.style.transform = `scale(${app.canvas.ds.scale})`;

    const input = dialog.querySelector(".value");
    const opts = dialog.querySelector(".helper");
    input.value = widget.value || "";

    let timer = null;
    let lastDir = null;
    let options = [];
    const extensions = widget.options.bat_path_extensions;

    function commit(v) {
        widget.value = v;
        widget.callback?.(v);
        dialog.close();
    }

    input.addEventListener("keydown", (e) => {
        if (e.keyCode === 27) dialog.close();
        else if (e.keyCode === 13) commit(input.value);
        else if (e.keyCode === 9) {
            if (opts.firstChild) {
                input.value = lastDir + opts.firstChild.innerText;
                e.preventDefault(); e.stopPropagation();
                refresh();
            }
        } else {
            if (timer) clearTimeout(timer);
            timer = setTimeout(refresh, 10);
            return;
        }
        e.preventDefault(); e.stopPropagation();
    });

    dialog.querySelector("button").onclick = () => commit(input.value);

    const rect = app.canvas.canvas.getBoundingClientRect();
    if (event) {
        dialog.style.left = (event.clientX - 20 - rect.left) + "px";
        dialog.style.top = (event.clientY - 20 - rect.top) + "px";
    }

    async function refresh() {
        timer = null;
        const [dir, rem] = pathStem(input.value);
        if (lastDir !== dir) {
            const params = new URLSearchParams({ path: dir });
            if (extensions) params.set("extensions", extensions);
            try {
                const r = await fetch(api.apiURL(`${PATH_ROUTE}?${params}`));
                options = await r.json();
            } catch { options = []; }
            lastDir = dir;
        }
        opts.innerHTML = "";
        for (const name of options) {
            if (!name.toLowerCase().startsWith(rem.toLowerCase())) continue;
            const el = document.createElement("div");
            el.innerText = name;
            const isDir = name.endsWith("/");
            el.className = "litegraph lite-search-item" + (isDir ? " is-dir" : "");
            el.onclick = () => {
                if (isDir) { input.value = lastDir + name; refresh(); input.focus(); }
                else      { commit(lastDir + name); }
            };
            opts.appendChild(el);
        }
    }

    setTimeout(() => { input.focus(); refresh(); }, 10);
    return true;
}

function drawPathWidget(ctx, node, w, y, H) {
    const m = 15;
    const showText = app.canvas.ds.scale >= 0.5;
    ctx.textAlign = "left";
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(m, y, w - m * 2, H, [H * 0.5]);
    ctx.fill();
    if (showText) {
        ctx.stroke();
        ctx.save();
        ctx.beginPath();
        ctx.rect(m, y, w - m * 2, H);
        ctx.clip();
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
        ctx.fillText(this.name, m * 2 + 5, y + H * 0.7);
        ctx.textAlign = "right";
        ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : "#777";
        let val = String(this.value || "");
        if (val.length > 35) val = "…" + val.slice(-32);
        ctx.fillText(val, w - m * 2 - 5, y + H * 0.7);
        ctx.restore();
    }
}

function makePathWidget(name, defaultValue, options) {
    const w = {
        name,
        type: "BAT.PATH",
        value: defaultValue || "",
        options: options || {},
        draw: drawPathWidget,
        mouse(event, pos, node) {
            if (event.type !== "pointerdown") return false;
            return openPathSearch(event, this);
        },
        computeSize() { return [200, LiteGraph.NODE_WIDGET_HEIGHT]; },
    };
    return w;
}

// ─── Trim-range widget (NLE-style scrubber) ─────────────────────────────────

const PREVIEW_H    = 110;
const TIMELINE_H   = 28;
const CONTROLS_H   = 26;   // single row: step buttons (left) + status (right)
const PAD          = 6;
const TRIM_TOTAL_H = PREVIEW_H + TIMELINE_H + CONTROLS_H + PAD * 2 + 4;

const HANDLE_R     = 8;
const HANDLE_HIT_R = 14;
const STEP_BTN_W   = 22;
const STEP_BTN_H   = CONTROLS_H - 4;

const ACCENT_IN    = "#7ec0ee";
const ACCENT_OUT   = "#f4b860";
const TRACK_BG     = "#222";
const TRACK_FILL   = "#3a7fb6";
const PILL_BG      = "rgba(0,0,0,0.72)";
const PILL_TEXT    = "#fff";

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function formatTimecode(frame, fps) {
    if (!fps || fps <= 0) return `${frame}f`;
    const totalSec = frame / fps;
    const m = Math.floor(totalSec / 60);
    const s = totalSec - m * 60;
    const ss = s.toFixed(3).padStart(6, "0"); // e.g. "07.500"
    return `${m}:${ss}`;
}

function niceTickStep(frameCount) {
    const targetTicks = 10;
    const raw = frameCount / targetTicks;
    const steps = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10000];
    for (const s of steps) if (s >= raw) return s;
    return steps[steps.length - 1];
}

function debounce(fn, ms) {
    let t = null;
    return function (...args) {
        if (t) clearTimeout(t);
        t = setTimeout(() => { t = null; fn.apply(this, args); }, ms);
    };
}

function drawPill(ctx, cx, cy, text, opts = {}) {
    const padX = opts.padX ?? 6;
    const padY = opts.padY ?? 3;
    const font = opts.font ?? "11px monospace";
    const align = opts.align ?? "center"; // "center" | "left" | "right"
    ctx.font = font;
    const w = ctx.measureText(text).width + padX * 2;
    const h = 16 + (padY - 3) * 2;
    let x = cx - w / 2;
    if (align === "left")  x = cx;
    if (align === "right") x = cx - w;
    const y = cy - h / 2;
    ctx.fillStyle = opts.bg ?? PILL_BG;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, h / 2);
    ctx.fill();
    if (opts.borderColor) {
        ctx.strokeStyle = opts.borderColor;
        ctx.lineWidth = 1;
        ctx.stroke();
    }
    ctx.fillStyle = opts.fg ?? PILL_TEXT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
    return { x, y, w, h };
}

function drawStepButton(ctx, x, y, w, h, glyph, hot) {
    ctx.fillStyle = hot ? "#444" : "#2a2a2a";
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ddd";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(glyph, x + w / 2, y + h / 2 + 0.5);
}

function makeTrimWidget(node, startIntWidget, endIntWidget) {
    const w = {
        name: "trim",
        type: "BAT.TRIM",
        value: null,

        // Volatile state. start/end live on the INT widgets — single
        // source of truth that ComfyUI serialises.
        _state: {
            frameCount: 0,
            fps: 0,
            width: 0,
            height: 0,
            startImg: null,
            endImg: null,
            startImgFrame: -1,
            endImgFrame: -1,
            dragMode: null,          // "start" | "end" | "range" | null
            dragAnchor: null,        // { startAt, endAt, pointerFrame }
            lastTouched: "start",    // for keyboard nudging
            hoverFrame: null,        // for hairline + tooltip
            hoverInTimeline: false,
        },
        get _start() { return startIntWidget.value | 0; },
        set _start(v) { startIntWidget.value = v | 0; },
        get _end()   { return endIntWidget.value | 0; },
        set _end(v)  { endIntWidget.value = v | 0; },

        computeSize() { return [360, TRIM_TOTAL_H]; },

        // ── Layout helpers ─────────────────────────────────────────────────
        _layout(width, y) {
            const inner = width - PAD * 2;
            const thumbGap = 8;
            const thumbW = (inner - thumbGap) / 2;
            return {
                width,
                inner,
                left: PAD,
                right: PAD + inner,
                thumbW,
                thumbGap,
                thumbY: y + PAD,
                thumbH: PREVIEW_H,
                timelineY: y + PAD + PREVIEW_H + 4,
                timelineH: TIMELINE_H,
                controlsY: y + PAD + PREVIEW_H + 4 + TIMELINE_H + 4,
                controlsH: CONTROLS_H,
            };
        },

        // Step-button hit rects (depend on layout). Returns { in:[..], out:[..] }
        // where each entry is { x, y, w, h, dir, mag }.
        _stepButtonRects(L) {
            const y = L.controlsY + (L.controlsH - STEP_BTN_H) / 2;
            const w = STEP_BTN_W, h = STEP_BTN_H;
            // IN group, then a label gap, then OUT group. Kept tight so
            // the status text on the right has room.
            const groupGap = 18;
            const inX  = L.left + 20;                       // 20px reserved for "IN" label
            const outX = inX + (w * 4) + groupGap + 26;     // 26px reserved for "OUT" label
            const buttons = [];
            ["start", "end"].forEach((which) => {
                const x0 = which === "start" ? inX : outX;
                [{ glyph: "‹‹", mag: -10 }, { glyph: "‹", mag: -1 },
                 { glyph: "›", mag: 1 }, { glyph: "››", mag: 10 }].forEach((b, i) => {
                    buttons.push({
                        x: x0 + i * w,
                        y, w, h,
                        which,
                        mag: b.mag,
                        glyph: b.glyph,
                    });
                });
            });
            return buttons;
        },

        _frameToX(frame, L) {
            const st = this._state;
            if (st.frameCount < 2) return L.left;
            return L.left + (frame / (st.frameCount - 1)) * L.inner;
        },
        _xToFrame(x, L) {
            const st = this._state;
            if (st.frameCount < 2) return 0;
            const t = clamp((x - L.left) / L.inner, 0, 1);
            return Math.round(t * (st.frameCount - 1));
        },

        // ── Drawing ────────────────────────────────────────────────────────
        draw(ctx, n, width, y, H) {
            this._last_w = width;
            this._last_y = y;
            const st = this._state;
            const L = this._layout(width, y);

            // ── Thumbnails ────────────────────────────────────────────────
            this._drawThumb(ctx, L, "start", L.left,                                st.startImg, this._start);
            this._drawThumb(ctx, L, "end",   L.left + L.thumbW + L.thumbGap,        st.endImg,   this._end);

            // ── Timeline track ────────────────────────────────────────────
            const tlMidY = L.timelineY + L.timelineH / 2;
            const trackY = tlMidY - 4;
            const trackH = 8;

            // Background track
            ctx.fillStyle = TRACK_BG;
            ctx.beginPath();
            ctx.roundRect(L.left, trackY, L.inner, trackH, 3);
            ctx.fill();

            if (st.frameCount > 1) {
                // Tick marks
                const tickStep = niceTickStep(st.frameCount);
                ctx.strokeStyle = "#3a3a3a";
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let f = 0; f <= st.frameCount - 1; f += tickStep) {
                    const tx = this._frameToX(f, L);
                    ctx.moveTo(tx, L.timelineY + 2);
                    ctx.lineTo(tx, L.timelineY + L.timelineH - 2);
                }
                ctx.stroke();

                // Selected band
                const sx = this._frameToX(this._start, L);
                const ex = this._frameToX(this._end, L);
                ctx.fillStyle = TRACK_FILL;
                ctx.fillRect(sx, trackY, Math.max(0, ex - sx), trackH);

                // Hover hairline
                if (st.hoverFrame != null && st.dragMode == null) {
                    const hx = this._frameToX(st.hoverFrame, L);
                    ctx.strokeStyle = "#888";
                    ctx.setLineDash([2, 3]);
                    ctx.beginPath();
                    ctx.moveTo(hx, L.timelineY);
                    ctx.lineTo(hx, L.timelineY + L.timelineH);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    drawPill(
                        ctx, hx, L.timelineY - 10,
                        `f ${st.hoverFrame} · ${formatTimecode(st.hoverFrame, st.fps)}`,
                        { font: "10px monospace" }
                    );
                }

                // Handles
                this._drawHandle(ctx, sx, tlMidY, "start", ACCENT_IN);
                this._drawHandle(ctx, ex, tlMidY, "end",   ACCENT_OUT);

                // Drag tooltip
                if (st.dragMode === "start") {
                    drawPill(
                        ctx, sx, L.timelineY - 14,
                        `IN  f ${this._start} · ${formatTimecode(this._start, st.fps)}`,
                        { font: "11px monospace", borderColor: ACCENT_IN }
                    );
                } else if (st.dragMode === "end") {
                    drawPill(
                        ctx, ex, L.timelineY - 14,
                        `OUT  f ${this._end} · ${formatTimecode(this._end, st.fps)}`,
                        { font: "11px monospace", borderColor: ACCENT_OUT }
                    );
                } else if (st.dragMode === "range") {
                    const cx = (sx + ex) / 2;
                    const dur = this._end - this._start;
                    drawPill(
                        ctx, cx, L.timelineY - 14,
                        `${dur + 1}f · ${formatTimecode(dur + 1, st.fps)}`,
                        { font: "11px monospace", borderColor: "#aaa" }
                    );
                }
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("enter a video path to enable trim", L.left + L.inner / 2, tlMidY);
            }

            // ── Step buttons + group labels (single row) ──────────────────
            const ctlMidY = L.controlsY + L.controlsH / 2;
            const rects = this._stepButtonRects(L);
            for (const b of rects) {
                drawStepButton(ctx, b.x, b.y, b.w, b.h, b.glyph, false);
            }
            ctx.font = "10px sans-serif";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            ctx.fillStyle = ACCENT_IN;
            ctx.fillText("IN", rects[0].x - 18, ctlMidY);
            ctx.fillStyle = ACCENT_OUT;
            ctx.fillText("OUT", rects[4].x - 24, ctlMidY);

            // ── Status text on the same row, right-aligned ────────────────
            const selFrames = (this._end - this._start + 1) | 0;
            const stride = Math.max(1, this._strideValue());
            const outFrames = Math.ceil(selFrames / stride);
            const secs = st.fps > 0 ? (selFrames / st.fps) : 0;
            const parts = [];
            if (stride > 1) {
                parts.push(`${selFrames}f →${outFrames}f`);
                parts.push(`÷${stride}`);
            } else {
                parts.push(`${selFrames}f`);
            }
            if (st.fps > 0) parts.push(`${secs.toFixed(2)}s`);
            if (st.fps > 0) parts.push(`${st.fps.toFixed(2)}fps`);
            if (st.width && st.height) parts.push(`${st.width}×${st.height}`);
            ctx.fillStyle = "#aab";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText(parts.join("  ·  "), L.right, ctlMidY);
        },

        // Resolve the select_every_nth widget's current value (if the node
        // declares one). Falls back to 1 so existing workflows still work.
        _strideValue() {
            const n = findNodeForWidget(this);
            const w = n?.widgets?.find(w => w.name === "select_every_nth");
            return (w?.value | 0) || 1;
        },

        _drawThumb(ctx, L, label, x, img, frameIdx) {
            const st = this._state;
            const y = L.thumbY, tw = L.thumbW, th = L.thumbH;
            ctx.fillStyle = "#0e0e0e";
            ctx.fillRect(x, y, tw, th);
            if (img && img.complete && img.naturalWidth) {
                const r = Math.min(tw / img.naturalWidth, th / img.naturalHeight);
                const dw = img.naturalWidth * r;
                const dh = img.naturalHeight * r;
                ctx.drawImage(img, x + (tw - dw) / 2, y + (th - dh) / 2, dw, dh);
            } else {
                ctx.fillStyle = "#666";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(st.frameCount ? "…" : "no video", x + tw / 2, y + th / 2);
            }
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y + 0.5, tw - 1, th - 1);

            // IN/OUT badge top-left
            const accent = label === "start" ? ACCENT_IN : ACCENT_OUT;
            const badge = label === "start" ? "IN" : "OUT";
            drawPill(ctx, x + 16, y + 10, badge, {
                font: "10px sans-serif",
                bg: accent,
                fg: "#000",
                padX: 5, padY: 2,
            });

            // Frame + timecode pills bottom-left
            const pillY1 = y + th - 22;
            const pillY2 = y + th - 6;
            drawPill(ctx, x + 6, pillY1, `f ${frameIdx}`, {
                font: "11px monospace", align: "left", padX: 6, padY: 3,
            });
            drawPill(ctx, x + 6, pillY2, formatTimecode(frameIdx, st.fps), {
                font: "11px monospace", align: "left", padX: 6, padY: 3,
            });
        },

        _drawHandle(ctx, x, y, which, accent) {
            const st = this._state;
            const active = st.dragMode === which;
            const focused = st.lastTouched === which;
            // Focus ring (last-touched handle)
            if (focused) {
                ctx.fillStyle = accent + "55"; // 33% alpha
                ctx.beginPath();
                ctx.arc(x, y, HANDLE_R + 3, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.fillStyle = active ? accent : "#ddd";
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_R, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            // Center dot
            ctx.fillStyle = accent;
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, Math.PI * 2);
            ctx.fill();
        },

        // ── Mouse / wheel ──────────────────────────────────────────────────
        mouse(event, pos, n) {
            const st = this._state;
            const width = this._last_w || n.size[0];
            const y0 = this._last_y || 0;
            const L = this._layout(width, y0);

            // Translate clientX → widget-local-X (consistent across versions).
            const canvasEl = app.canvas.canvas;
            const rect = canvasEl.getBoundingClientRect();
            const scale = app.canvas.ds.scale;
            const off = app.canvas.ds.offset;
            const widgetOriginGraphX = n.pos[0];
            const widgetOriginGraphY = n.pos[1];
            const clientToWidgetX = (cx) => (cx - rect.left) / scale - off[0] - widgetOriginGraphX;
            const clientToGraphY  = (cy) => (cy - rect.top)  / scale - off[1] - widgetOriginGraphY;

            // Hover (pointermove with no drag) — only updates hover state.
            const isMove = event.type === "pointermove" || event.type === "mousemove";
            const isDown = event.type === "pointerdown" || event.type === "mousedown" || event.type === "down";
            const isWheel = event.type === "wheel";

            if (isMove && st.dragMode == null) {
                const lx = event.clientX != null ? clientToWidgetX(event.clientX) : pos[0];
                const ly = event.clientY != null ? clientToGraphY(event.clientY) : pos[1];
                const inTL = ly >= L.timelineY - 4 && ly <= L.timelineY + L.timelineH + 4;
                const newHover = (inTL && st.frameCount > 1) ? this._xToFrame(lx, L) : null;
                if (newHover !== st.hoverFrame || inTL !== st.hoverInTimeline) {
                    st.hoverFrame = newHover;
                    st.hoverInTimeline = inTL;
                    n.setDirtyCanvas(true, true);
                }
                return false;
            }

            // Mouse wheel — nudge nearest handle if pointer is over the timeline.
            if (isWheel && st.frameCount > 1) {
                const lx = event.clientX != null ? clientToWidgetX(event.clientX) : pos[0];
                const sx = this._frameToX(this._start, L);
                const ex = this._frameToX(this._end, L);
                const nearStart = Math.abs(lx - sx) <= HANDLE_HIT_R;
                const nearEnd   = Math.abs(lx - ex) <= HANDLE_HIT_R;
                if (nearStart || nearEnd) {
                    const which = nearStart && (!nearEnd || Math.abs(lx - sx) < Math.abs(lx - ex)) ? "start" : "end";
                    const sign = event.deltaY < 0 ? +1 : -1;
                    const mag = event.shiftKey ? 10 : 1;
                    nudgeHandle(this, which, sign * mag);
                    if (event.preventDefault) event.preventDefault();
                    return true;
                }
                return false;
            }

            if (!isDown) return false;
            if (st.frameCount < 2) return false;

            const lx = event.clientX != null ? clientToWidgetX(event.clientX) : pos[0];
            const ly = event.clientY != null ? clientToGraphY(event.clientY) : pos[1];

            // 1. Step buttons
            for (const b of this._stepButtonRects(L)) {
                if (lx >= b.x && lx <= b.x + b.w && ly >= b.y && ly <= b.y + b.h) {
                    nudgeHandle(this, b.which, b.mag);
                    return true;
                }
            }

            // 2 / 3 / 4 / 5. Timeline-band interactions (broad y check)
            const tlBand = ly >= L.timelineY - 6 && ly <= L.timelineY + L.timelineH + 6;
            if (!tlBand) return false;

            const sx = this._frameToX(this._start, L);
            const ex = this._frameToX(this._end, L);
            const dStart = Math.abs(lx - sx);
            const dEnd   = Math.abs(lx - ex);

            let mode;
            if (dStart <= HANDLE_HIT_R && dStart <= dEnd) {
                mode = "start";
            } else if (dEnd <= HANDLE_HIT_R) {
                mode = "end";
            } else if (lx > sx && lx < ex) {
                mode = "range";
                st.dragAnchor = {
                    startAt: this._start,
                    endAt: this._end,
                    pointerFrame: this._xToFrame(lx, L),
                };
            } else {
                // Outside selected band — snap nearest handle to click then drag it
                mode = dStart < dEnd ? "start" : "end";
                const f = this._xToFrame(lx, L);
                if (mode === "start") this._start = Math.min(f, this._end);
                else                  this._end   = Math.max(f, this._start);
            }
            st.dragMode = mode;
            st.lastTouched = mode === "range" ? st.lastTouched : mode;
            n.setDirtyCanvas(true, true);

            const self = this;
            const applyAt = (xLocal) => {
                const f = self._xToFrame(xLocal, L);
                if (st.dragMode === "start") {
                    self._start = Math.min(f, self._end);
                } else if (st.dragMode === "end") {
                    self._end = Math.max(f, self._start);
                } else if (st.dragMode === "range" && st.dragAnchor) {
                    const a = st.dragAnchor;
                    const dur = a.endAt - a.startAt;
                    const delta = f - a.pointerFrame;
                    let ns = clamp(a.startAt + delta, 0, st.frameCount - 1 - dur);
                    self._start = ns;
                    self._end = ns + dur;
                }
                n.setDirtyCanvas(true, true);
                syncPreviewRange(n);
                // Live thumbnail refresh while dragging (debounced).
                if (st.dragMode === "start" || st.dragMode === "range") debouncedFetchStart(n);
                if (st.dragMode === "end"   || st.dragMode === "range") debouncedFetchEnd(n);
            };

            const onMove = (e) => applyAt(clientToWidgetX(e.clientX));
            const cleanup = () => {
                canvasEl.removeEventListener("pointermove", onMove, true);
                canvasEl.removeEventListener("pointerup",   onUp,   true);
                canvasEl.removeEventListener("pointercancel", onUp, true);
                window.removeEventListener("pointerup", onUp, true);
                window.removeEventListener("mouseup",   onUp, true);
                window.removeEventListener("blur",      onUp);
                st.dragMode = null;
                st.dragAnchor = null;
                fetchFrame(n, "start");
                fetchFrame(n, "end");
                syncPreviewRange(n);
                n.setDirtyCanvas(true, true);
            };
            const onUp = () => cleanup();
            canvasEl.addEventListener("pointermove",   onMove, true);
            canvasEl.addEventListener("pointerup",     onUp,   true);
            canvasEl.addEventListener("pointercancel", onUp,   true);
            window.addEventListener("pointerup", onUp, true);
            window.addEventListener("mouseup",   onUp, true);
            window.addEventListener("blur",      onUp);
            return true;
        },
    };

    return w;
}

// Debounced thumbnail fetchers (per-node — cheap closures created once
// here, scoped to fetchFrame keyed by node identity is fine because
// fetchFrame already short-circuits when the requested frame matches the
// cached one).
const debouncedFetchStart = debounce((n) => fetchFrame(n, "start"), 180);
const debouncedFetchEnd   = debounce((n) => fetchFrame(n, "end"),   180);

// Commit a frame change driven by something other than mouse drag (step
// buttons, wheel, keyboard). Clamps, writes through, focuses the handle,
// refreshes the canvas, and triggers a debounced thumb refresh.
function commitFrame(widget, which, newFrame) {
    const st = widget._state;
    const f = clamp(newFrame | 0, 0, Math.max(0, st.frameCount - 1));
    if (which === "start") widget._start = Math.min(f, widget._end);
    else                   widget._end   = Math.max(f, widget._start);
    st.lastTouched = which;
}

function nudgeHandle(widget, which, delta) {
    const cur = which === "start" ? widget._start : widget._end;
    commitFrame(widget, which, cur + delta);
    // We don't have the node reference here cheaply; the widget is drawn
    // again on next dirty — find the node by walking app.graph._nodes.
    const node = findNodeForWidget(widget);
    if (node) {
        node.setDirtyCanvas(true, true);
        syncPreviewRange(node);
        if (which === "start") debouncedFetchStart(node);
        else                   debouncedFetchEnd(node);
    }
}

function findNodeForWidget(widget) {
    const nodes = app.graph?._nodes || [];
    for (const n of nodes) {
        if (n.widgets && n.widgets.includes(widget)) return n;
    }
    return null;
}

// Push the trim widget's current state into the preview's playback range.
// Cheap: only seeks when the current time falls outside the new window.
function syncPreviewRange(node) {
    const preview = node?._batPreview;
    if (!preview) return;
    const trim = node.widgets.find(w => w.type === "BAT.TRIM");
    if (!trim) return;
    preview.setRange(trim._start, trim._end, trim._state.fps);
}

// ─── Loading / preview fetching ─────────────────────────────────────────────

async function fetchVideoInfo(node) {
    const pathW = node.widgets.find(w => w.name === "path");
    const trimW = node.widgets.find(w => w.name === "trim");
    if (!pathW || !trimW || !pathW.value) return;
    try {
        const r = await fetch(api.apiURL(`${INFO_ROUTE}?path=${encodeURIComponent(pathW.value)}`));
        const info = await r.json();
        if (!info.ok) { console.warn("[Bat] video-info:", info.error); return; }
        const st = trimW._state;
        st.frameCount = info.frame_count | 0;
        st.fps = info.fps || 0;
        st.width = info.width | 0;
        st.height = info.height | 0;
        // Clamp existing start/end into the new range. If the stored end
        // is -1 (default sentinel: "play to the last frame"), snap it to
        // the last frame so the slider reflects the actual range used at
        // execute time.
        if (trimW._end < 0 || trimW._end >= st.frameCount) trimW._end = st.frameCount - 1;
        if (trimW._start < 0 || trimW._start >= st.frameCount) trimW._start = 0;
        if (trimW._end < trimW._start) trimW._end = trimW._start;
        // Force thumbnail refresh (file may have changed under us).
        st.startImg = null; st.endImg = null;
        st.startImgFrame = -1; st.endImgFrame = -1;
        // Push to the preview widget too.
        if (node._batPreview) {
            node._batPreview.setSource(api.apiURL(`${STREAM_ROUTE}?path=${encodeURIComponent(pathW.value)}`));
            node._batPreview.setRange(trimW._start, trimW._end, st.fps);
        }
        node.setDirtyCanvas(true, true);
        fetchFrame(node, "start");
        fetchFrame(node, "end");
    } catch (e) {
        console.warn("[Bat] video-info fetch failed:", e);
    }
}

function fetchFrame(node, which) {
    const pathW = node.widgets.find(w => w.name === "path");
    const trimW = node.widgets.find(w => w.name === "trim");
    if (!pathW?.value || !trimW) return;
    const st = trimW._state;
    const frame = which === "start" ? trimW._start : trimW._end;
    if (frame < 0) return;
    if (which === "start" && st.startImgFrame === frame) return;
    if (which === "end" && st.endImgFrame === frame) return;

    const url = api.apiURL(
        `${FRAME_ROUTE}?path=${encodeURIComponent(pathW.value)}&frame=${frame}&max_w=320`
    );
    const img = new Image();
    img.onload = () => {
        if (which === "start") { st.startImg = img; st.startImgFrame = frame; }
        else                   { st.endImg = img;   st.endImgFrame = frame; }
        node.setDirtyCanvas(true, true);
    };
    img.onerror = () => {};
    img.src = url;
}

// ─── Preview widget (HTML5 <video> with transport) ──────────────────────────
//
// Returns { element, setSource, setRange, dispose } — attached to the node
// via node.addDOMWidget. The browser handles decoding so playback is smooth
// natively. setRange clamps current playback into [startFrame, endFrame] and
// loops there; the underlying <video> still has the full file loaded so
// seeking within the trim window is instant.

function makePreviewWidget() {
    const css = (el, s) => Object.assign(el.style, s);

    const root = document.createElement("div");
    css(root, {
        display: "flex", flexDirection: "column",
        background: "#0a0a0a", border: "1px solid #2a2a2a",
        borderRadius: "4px", overflow: "hidden",
        fontFamily: "monospace", color: "#ddd",
        width: "100%", height: "100%",
        boxSizing: "border-box",
    });

    const videoWrap = document.createElement("div");
    css(videoWrap, {
        flex: "1 1 auto", minHeight: "0",
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "#000",
    });
    const video = document.createElement("video");
    css(video, {
        maxWidth: "100%", maxHeight: "100%",
        width: "auto", height: "100%",
        outline: "none", display: "block",
    });
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    videoWrap.appendChild(video);

    // ── Scrubber row ───────────────────────────────────────────────────────
    const scrubRow = document.createElement("div");
    css(scrubRow, {
        display: "flex", alignItems: "center", gap: "10px",
        padding: "6px 10px", background: "#141414",
        fontSize: "11px",
    });
    const scrubber = document.createElement("div");
    css(scrubber, {
        flex: "1 1 auto", height: "6px", background: "#2a2a2a",
        borderRadius: "3px", position: "relative", cursor: "pointer",
    });
    const scrubFill = document.createElement("div");
    css(scrubFill, {
        position: "absolute", left: "0", top: "0", bottom: "0",
        background: "#3a7fb6", borderRadius: "3px", width: "0%",
    });
    const scrubKnob = document.createElement("div");
    css(scrubKnob, {
        position: "absolute", top: "50%", left: "0",
        width: "10px", height: "10px", marginLeft: "-5px", marginTop: "-5px",
        background: "#ddd", border: "1px solid #000", borderRadius: "50%",
    });
    scrubber.append(scrubFill, scrubKnob);

    const timeText = document.createElement("span");
    css(timeText, { opacity: "0.85", fontVariantNumeric: "tabular-nums",
                    whiteSpace: "nowrap", minWidth: "120px", textAlign: "right" });
    timeText.textContent = "0:00.000 / 0:00.000";
    scrubRow.append(scrubber, timeText);

    // ── Transport row ──────────────────────────────────────────────────────
    const transportRow = document.createElement("div");
    css(transportRow, {
        display: "flex", alignItems: "center", gap: "4px",
        padding: "4px 10px 8px", background: "#141414",
        justifyContent: "center",
    });
    function btn(glyph, title, opts = {}) {
        const b = document.createElement("button");
        b.textContent = glyph;
        b.title = title;
        css(b, {
            background: "#2a2a2a", color: "#ddd", border: "1px solid #444",
            borderRadius: "3px", padding: "3px 8px", cursor: "pointer",
            fontSize: opts.big ? "14px" : "12px", lineHeight: "1",
            minWidth: opts.big ? "36px" : "26px",
            fontFamily: "inherit",
        });
        b.onmouseover = () => b.style.background = "#3a3a3a";
        b.onmouseout  = () => b.style.background = (b.dataset.on === "true") ? "#3a7fb6" : "#2a2a2a";
        b.onmousedown = (e) => e.stopPropagation();  // don't let ComfyUI start a node-drag
        return b;
    }
    const skipBackBig = btn("‹‹", "Jump -10 frames");
    const skipBack    = btn("‹",  "Previous frame");
    const playBtn     = btn("▶",  "Play / Pause", { big: true });
    const skipFwd     = btn("›",  "Next frame");
    const skipFwdBig  = btn("››", "Jump +10 frames");
    const loopBtn     = btn("⟳",  "Loop");
    loopBtn.dataset.on = "true";
    loopBtn.style.background = "#3a7fb6";
    transportRow.append(skipBackBig, skipBack, playBtn, skipFwd, skipFwdBig, loopBtn);

    root.append(videoWrap, scrubRow, transportRow);

    // ── Behaviour ──────────────────────────────────────────────────────────
    const state = { startFrame: 0, endFrame: 0, fps: 24, loop: true };

    const frameToSec = (f) => f / Math.max(0.001, state.fps);
    const fmt = (t) => {
        if (!isFinite(t) || t < 0) t = 0;
        const m = Math.floor(t / 60);
        const ss = (t - m * 60).toFixed(3).padStart(6, "0");
        return `${m}:${ss}`;
    };

    function paint() {
        const sStart = frameToSec(state.startFrame);
        const sEnd   = frameToSec(state.endFrame + 1);
        const dur    = Math.max(0.001, sEnd - sStart);
        const cur    = clamp(video.currentTime - sStart, 0, dur);
        const pct    = (cur / dur) * 100;
        scrubFill.style.width = pct + "%";
        scrubKnob.style.left  = pct + "%";
        timeText.textContent  = `${fmt(cur)} / ${fmt(dur)}`;
    }

    let raf = null;
    function tick() {
        const sStart = frameToSec(state.startFrame);
        const sEnd   = frameToSec(state.endFrame + 1);
        if (!video.paused) {
            if (video.currentTime >= sEnd) {
                if (state.loop) {
                    video.currentTime = sStart;
                } else {
                    video.pause();
                    video.currentTime = sEnd - 1 / 1000;
                }
            }
        }
        paint();
        if (!video.paused) raf = requestAnimationFrame(tick);
    }
    function startTick() {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(tick);
    }
    function stopTick() {
        if (raf) cancelAnimationFrame(raf);
        raf = null;
        paint();
    }

    video.addEventListener("play",  () => { playBtn.textContent = "⏸"; startTick(); });
    video.addEventListener("pause", () => { playBtn.textContent = "▶"; stopTick(); });
    video.addEventListener("seeked", paint);
    video.addEventListener("loadedmetadata", () => {
        try { video.currentTime = frameToSec(state.startFrame); } catch {}
        paint();
    });

    playBtn.onclick = () => {
        const sStart = frameToSec(state.startFrame);
        const sEnd   = frameToSec(state.endFrame + 1);
        if (video.paused) {
            if (video.currentTime >= sEnd - 0.001 || video.currentTime < sStart) {
                video.currentTime = sStart;
            }
            video.play().catch(() => {});
        } else {
            video.pause();
        }
    };

    function step(deltaFrames) {
        video.pause();
        const sStart = frameToSec(state.startFrame);
        const sEnd   = frameToSec(state.endFrame);
        const next = clamp(video.currentTime + deltaFrames / state.fps, sStart, sEnd);
        video.currentTime = next;
    }
    skipBack.onclick    = () => step(-1);
    skipFwd.onclick     = () => step(+1);
    skipBackBig.onclick = () => step(-10);
    skipFwdBig.onclick  = () => step(+10);

    loopBtn.onclick = () => {
        const next = loopBtn.dataset.on !== "true";
        loopBtn.dataset.on = String(next);
        loopBtn.style.background = next ? "#3a7fb6" : "#2a2a2a";
        state.loop = next;
    };

    // Scrubber drag
    function seekFromClient(clientX) {
        const rect = scrubber.getBoundingClientRect();
        const t = clamp((clientX - rect.left) / rect.width, 0, 1);
        const sStart = frameToSec(state.startFrame);
        const sEnd   = frameToSec(state.endFrame + 1);
        video.currentTime = sStart + t * (sEnd - sStart);
    }
    scrubber.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        const wasPaused = video.paused;
        video.pause();
        try { scrubber.setPointerCapture(e.pointerId); } catch {}
        seekFromClient(e.clientX);
        const onMove = (ev) => seekFromClient(ev.clientX);
        const onUp = () => {
            scrubber.removeEventListener("pointermove", onMove);
            scrubber.removeEventListener("pointerup", onUp);
            scrubber.removeEventListener("pointercancel", onUp);
            if (!wasPaused) video.play().catch(() => {});
        };
        scrubber.addEventListener("pointermove", onMove);
        scrubber.addEventListener("pointerup", onUp);
        scrubber.addEventListener("pointercancel", onUp);
    });

    let currentSrc = null;
    return {
        element: root,
        get videoElement() { return video; },

        setSource(url) {
            if (url === currentSrc) return;
            currentSrc = url;
            video.pause();
            video.src = url;
            video.load();
        },

        setRange(startFrame, endFrame, fps) {
            state.startFrame = startFrame | 0;
            state.endFrame   = endFrame | 0;
            state.fps        = fps && fps > 0 ? fps : 24;
            const sStart = frameToSec(state.startFrame);
            const sEnd   = frameToSec(state.endFrame + 1);
            if (video.readyState >= 1) {
                if (video.currentTime < sStart || video.currentTime >= sEnd) {
                    try { video.currentTime = sStart; } catch {}
                }
            }
            paint();
        },

        dispose() {
            stopTick();
            video.pause();
            video.src = "";
        },
    };
}

// ─── Global keyboard shortcuts ──────────────────────────────────────────────
// When a Bat_VideoLoader node is the (only) selected node, arrow keys and
// a few NLE-style bindings nudge / snap the trim handles. Skipped when the
// active focus target is an input/textarea so the path popup keeps working.

function selectedTrimWidget() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return null;
    const ids = Object.keys(sel);
    if (ids.length !== 1) return null;
    const n = sel[ids[0]];
    if (!n || n.type !== NODE_TYPE || !n.widgets) return null;
    return n.widgets.find(w => w.type === "BAT.TRIM") || null;
}

window.addEventListener("keydown", (e) => {
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || e.target?.isContentEditable) return;
    const tw = selectedTrimWidget();
    if (!tw) return;
    const st = tw._state;
    if (st.frameCount < 2) return;

    const mag = e.shiftKey ? 10 : 1;
    let handled = true;
    switch (e.key) {
        case "ArrowLeft":  nudgeHandle(tw, st.lastTouched, -mag); break;
        case "ArrowRight": nudgeHandle(tw, st.lastTouched, +mag); break;
        case ",":          nudgeHandle(tw, st.lastTouched, -1);   break;
        case ".":          nudgeHandle(tw, st.lastTouched, +1);   break;
        case "Home":       commitFrame(tw, st.lastTouched, 0); finalize(); break;
        case "End":        commitFrame(tw, st.lastTouched, st.frameCount - 1); finalize(); break;
        case "i": case "I":
            if (st.hoverFrame != null) { commitFrame(tw, "start", st.hoverFrame); finalize("start"); }
            break;
        case "o": case "O":
            if (st.hoverFrame != null) { commitFrame(tw, "end", st.hoverFrame); finalize("end"); }
            break;
        default: handled = false;
    }
    if (handled) { e.preventDefault(); e.stopPropagation(); }

    function finalize(only) {
        const node = findNodeForWidget(tw);
        if (!node) return;
        node.setDirtyCanvas(true, true);
        if (!only || only === "start") debouncedFetchStart(node);
        if (!only || only === "end")   debouncedFetchEnd(node);
    }
});

// ─── Extension hook ─────────────────────────────────────────────────────────

app.registerExtension({
    name: "BAT.VideoLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!this.widgets) return r;

            // Find the declared widgets.
            const pathIdx = this.widgets.findIndex(w => w.name === "path");
            const startInt = this.widgets.find(w => w.name === "start_frame");
            const endInt = this.widgets.find(w => w.name === "end_frame");

            // Replace path widget with autocompleting one.
            if (pathIdx >= 0) {
                const orig = this.widgets[pathIdx];
                const config = nodeData.input?.required?.path;
                const opts = (config && config[1]) || {};
                const path = makePathWidget("path", orig.value || "", opts);
                path.callback = () => fetchVideoInfo(this);
                this.widgets[pathIdx] = path;
            }

            // Keep the INT widgets visible so the user can type exact
            // start/end frame numbers. The trim widget reads them via
            // getters on every draw, so two-way sync is automatic.
            // Just attach callbacks that enforce start <= end and trigger
            // a thumbnail refresh when the user edits them directly.
            this._batStartInt = startInt;
            this._batEndInt = endInt;

            const self = this;
            const refreshAfterTyped = (which) => {
                if (startInt && endInt && startInt.value > endInt.value) {
                    if (which === "start") startInt.value = endInt.value;
                    else                   endInt.value   = startInt.value;
                }
                if (startInt && startInt.value < 0) startInt.value = 0;
                if (endInt   && endInt.value   < 0) endInt.value   = 0;
                self.setDirtyCanvas(true, true);
                syncPreviewRange(self);
                debouncedFetchStart(self);
                debouncedFetchEnd(self);
            };
            if (startInt) {
                const orig = startInt.callback;
                startInt.callback = function (v) {
                    if (orig) orig.apply(this, arguments);
                    refreshAfterTyped("start");
                };
            }
            if (endInt) {
                const orig = endInt.callback;
                endInt.callback = function (v) {
                    if (orig) orig.apply(this, arguments);
                    refreshAfterTyped("end");
                };
            }

            // Append the trim widget after the INT widgets so the
            // text inputs appear above the scrubber.
            if (startInt && endInt) {
                const trim = makeTrimWidget(this, startInt, endInt);
                this.widgets.push(trim);
            }

            // Attach the HTML5 video preview underneath. addDOMWidget
            // positions a DOM element over the canvas wherever this node
            // is, and ComfyUI clips/scrolls it for us. serialize:false so
            // it isn't persisted in the workflow JSON.
            const preview = makePreviewWidget();
            this._batPreview = preview;
            try {
                this.addDOMWidget("video_preview", "video_preview", preview.element, {
                    serialize: false,
                    hideOnZoom: false,
                });
            } catch (e) {
                console.warn("[Bat] addDOMWidget failed, falling back:", e);
            }

            // For freshly-added nodes only; on workflow restore, the
            // saved widget values aren't applied until after onConfigure,
            // so the info-fetch there gets the right start/end.
            queueMicrotask(() => fetchVideoInfo(this));

            // Resize node to fit. Bump min width so the IN/OUT step button
            // clusters + status line don't crowd each other, and add room
            // for the preview pane below the trim widget.
            const computed = this.computeSize();
            const previewH = 260;
            this.size = [
                Math.max(420, this.size[0] || 0, computed[0]),
                computed[1] + previewH,
            ];
            return r;
        };

        // Fires after ComfyUI restores widget values from the saved
        // workflow. Re-fetch info so the trim widget reflects the
        // persisted start_frame / end_frame.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            queueMicrotask(() => fetchVideoInfo(this));
            return r;
        };
    },
});
