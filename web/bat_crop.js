/**
 * BAT — Bat_Crop front-end.
 *
 * On-node drag-rectangle crop editor. The input image is delivered as a
 * base64 preview in the node's onExecuted message (run once → editor
 * populates, same pattern as Bat_RefAligner). The four crop_* INT widgets
 * are the source of truth and serialise; the canvas reads / writes them.
 *
 * Drag UX:
 *   – click inside the box → move it (clamped to image bounds)
 *   – drag a corner → resize from that corner (uniform aspect when
 *     aspect_lock is on, ratio parsed from `aspect_ratio`)
 *   – drag an edge midpoint → resize that single axis
 *   – click on empty image outside the box → start a fresh rectangle
 */

import { app } from "../../scripts/app.js";
import { addBatDOMWidget, clampNodeSize } from "./bat_node_layout.js";
import { attachZoomControl } from "./bat_zoom_control.js";
import { batTrack } from "./bat_lifecycle.js";

const NODE_TYPE = "Bat_Crop";

function parseRatio(str) {
    if (!str) return null;
    const s = String(str).trim().toLowerCase();
    if (!s || s === "free") return null;
    if (s.includes(":")) {
        const [a, b] = s.split(":").map(parseFloat);
        if (a > 0 && b > 0) return a / b;
    }
    if (s.includes("/")) {
        const [a, b] = s.split("/").map(parseFloat);
        if (a > 0 && b > 0) return a / b;
    }
    const f = parseFloat(s);
    return isFinite(f) && f > 0 ? f : null;
}

function makeEditor(node) {
    const root = document.createElement("div");
    root.style.cssText = "position:relative;width:100%;height:100%;background:#0a0a0a;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:100%;display:block;touch-action:none;";
    const hint = document.createElement("div");
    hint.style.cssText = "position:absolute;left:8px;bottom:6px;font:11px monospace;color:#9aa;pointer-events:none;text-shadow:0 1px 2px #000;";
    hint.textContent = "Run once to load the image";
    const bar = document.createElement("div");
    bar.style.cssText = "position:absolute;top:6px;right:8px;font:11px monospace;color:#cdd;background:rgba(0,0,0,0.55);padding:3px 8px;border-radius:4px;";
    const info = document.createElement("span");
    info.textContent = "no image yet";
    bar.appendChild(info);
    root.append(canvas, hint, bar);

    const ctx = canvas.getContext("2d");
    const state = {
        img: null, imgW: 0, imgH: 0,
        dispScale: 1, offX: 0, offY: 0,
        dispZoom: 1,               // display-only zoom (1 = fit-to-canvas)
        handles: [], drag: null,
    };

    const W = (n) => node.widgets.find(w => w.name === n);
    const get = (n) => (W(n)?.value) ?? 0;
    const set = (n, v) => {
        const w = W(n);
        if (!w) return;
        w.value = v;
        try { w.callback?.call(w, v); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };

    function recomputeDisplay(cw, ch) {
        if (!state.imgW) { state.dispScale = 1; state.offX = 0; state.offY = 0; return; }
        // Fit-to-canvas, then apply the display-only zoom + pan. Centred at
        // zoom 1 / pan 0; zooming out reveals margin (the out-of-frame area
        // a crop can extend into), pan/zoom-to-cursor shift the centre.
        const fit = Math.min(cw / state.imgW, ch / state.imgH);
        state.dispScale = fit * (state.dispZoom || 1);
        state.offX = (cw - state.imgW * state.dispScale) / 2 + (state.panX || 0);
        state.offY = (ch - state.imgH * state.dispScale) / 2 + (state.panY || 0);
    }
    const c2d = (x, y) => ({ x: state.offX + x * state.dispScale, y: state.offY + y * state.dispScale });
    const d2c = (x, y) => ({ x: (x - state.offX) / state.dispScale, y: (y - state.offY) / state.dispScale });

    // Whether the crop must stay inside the canvas. Defaults to true when
    // the widget is missing (older graphs) so behaviour is unchanged.
    const constrained = () => {
        const w = W("constrain_to_canvas");
        return w ? !!w.value : true;
    };

    function clampRectToImage() {
        if (!state.imgW) return;
        // Off when the artist unlocked "constrain to canvas" — the rect may
        // then extend past the edge and the backend zero-pads the overhang.
        if (!constrained()) return;
        // When the rect is rotated it can legitimately stick out past the
        // image bounds (the backend grid_sample handles that with zero
        // padding); only clamp in the axis-aligned case.
        if (Math.abs(get("crop_angle") || 0) > 1e-3) return;
        let x = get("crop_x") | 0, y = get("crop_y") | 0;
        let w = get("crop_w") | 0, h = get("crop_h") | 0;
        x = Math.max(0, Math.min(x, Math.max(0, state.imgW - 1)));
        y = Math.max(0, Math.min(y, Math.max(0, state.imgH - 1)));
        w = Math.max(1, Math.min(w, state.imgW - x));
        h = Math.max(1, Math.min(h, state.imgH - y));
        if (x !== get("crop_x")) set("crop_x", x);
        if (y !== get("crop_y")) set("crop_y", y);
        if (w !== get("crop_w")) set("crop_w", w);
        if (h !== get("crop_h")) set("crop_h", h);
    }

    function render() {
        // Zoom-stable sizing — see bat_ref_aligner.js render() for the
        // detailed rationale. offsetWidth is LiteGraph-zoom independent;
        // multiplying in devicePixelRatio keeps the bitmap crisp on
        // hi-DPI and after a zoom-out → zoom-in round trip.
        const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
        const cssW = Math.max(1, Math.floor(root.offsetWidth));
        const cssH = Math.max(1, Math.floor(root.offsetHeight));
        const bw = Math.floor(cssW * dpr);
        const bh = Math.floor(cssH * dpr);
        if (canvas.width !== bw || canvas.height !== bh) {
            canvas.width = bw;
            canvas.height = bh;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w0 = cssW, h0 = cssH;
        recomputeDisplay(w0, h0);

        ctx.clearRect(0, 0, w0, h0);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, w0, h0);

        if (state.img) {
            const o = c2d(0, 0);
            ctx.drawImage(state.img, o.x, o.y, state.imgW * state.dispScale, state.imgH * state.dispScale);
        }

        state.handles = [];
        if (state.img && state.imgW) {
            const cx = get("crop_x") | 0, cy = get("crop_y") | 0;
            const cw = get("crop_w") | 0, ch = get("crop_h") | 0;
            const ang = (get("crop_angle") || 0) * Math.PI / 180;
            const cos = Math.cos(ang), sin = Math.sin(ang);
            const rectCx = cx + cw / 2, rectCy = cy + ch / 2;
            const cd = c2d(rectCx, rectCy);
            const dispW = cw * state.dispScale, dispH = ch * state.dispScale;

            // Compute the four corners of the (possibly rotated) crop rect in
            // display pixels. lx/ly are unit offsets along the rect's local
            // axes (±1), so corners are at (lx*w/2, ly*h/2) before rotation.
            const cornerDisp = (lx, ly) => {
                const rx = lx * cw / 2, ry = ly * ch / 2;       // rect-local image px
                const dx = (rx * cos - ry * sin) * state.dispScale;
                const dy = (rx * sin + ry * cos) * state.dispScale;
                return { x: cd.x + dx, y: cd.y + dy };
            };
            const cTL = cornerDisp(-1, -1);
            const cTR = cornerDisp( 1, -1);
            const cBR = cornerDisp( 1,  1);
            const cBL = cornerDisp(-1,  1);

            // Dim outside the rotated crop rect.
            const o = c2d(0, 0);
            const iW = state.imgW * state.dispScale;
            const iH = state.imgH * state.dispScale;
            ctx.save();
            ctx.fillStyle = "rgba(0,0,0,0.55)";
            const dim = new Path2D();
            dim.rect(o.x, o.y, iW, iH);
            dim.moveTo(cTL.x, cTL.y);
            dim.lineTo(cTR.x, cTR.y);
            dim.lineTo(cBR.x, cBR.y);
            dim.lineTo(cBL.x, cBL.y);
            dim.closePath();
            ctx.fill(dim, "evenodd");
            ctx.restore();

            // Crop border (rotated polygon).
            ctx.strokeStyle = "#7ec0ee";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(cTL.x, cTL.y);
            ctx.lineTo(cTR.x, cTR.y);
            ctx.lineTo(cBR.x, cBR.y);
            ctx.lineTo(cBL.x, cBL.y);
            ctx.closePath();
            ctx.stroke();

            // 4 corner + 4 edge-midpoint handles. Stored with their local
            // (lx, ly) so mouse math can drive resize math from the same data.
            const HR = 5;
            const defs = [
                { name: "tl", lx: -1, ly: -1 },
                { name: "tr", lx:  1, ly: -1 },
                { name: "br", lx:  1, ly:  1 },
                { name: "bl", lx: -1, ly:  1 },
                { name: "t",  lx:  0, ly: -1 },
                { name: "r",  lx:  1, ly:  0 },
                { name: "b",  lx:  0, ly:  1 },
                { name: "l",  lx: -1, ly:  0 },
            ];
            ctx.fillStyle = "#fff";
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 1.5;
            for (const d of defs) {
                const p = cornerDisp(d.lx, d.ly);
                ctx.beginPath();
                ctx.rect(p.x - HR, p.y - HR, HR * 2, HR * 2);
                ctx.fill(); ctx.stroke();
                state.handles.push({ name: d.name, x: p.x, y: p.y, lx: d.lx, ly: d.ly });
            }

            // Rotation handle: a small disc above the top edge, 30 display px
            // perpendicular to the (rotated) top edge.
            const topMid = cornerDisp(0, -1);
            const rotX = topMid.x + 30 * sin;        // R(θ) * (0,-1) = (sin, -cos)
            const rotY = topMid.y + 30 * (-cos);
            ctx.strokeStyle = "#f4b860";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(topMid.x, topMid.y);
            ctx.lineTo(rotX, rotY);
            ctx.stroke();
            ctx.fillStyle = "#f4b860";
            ctx.strokeStyle = "#000";
            ctx.beginPath();
            ctx.arc(rotX, rotY, 6, 0, Math.PI * 2);
            ctx.fill(); ctx.stroke();
            state.handles.push({ name: "rot", x: rotX, y: rotY });
        }

        // Info bar: WxH px · angle → snapped output dims · lock badge.
        const cw = get("crop_w") | 0, ch = get("crop_h") | 0;
        const ang = get("crop_angle") || 0;
        const snap = Math.max(1, get("snap_to") | 0);
        const outW = Math.max(snap, Math.round(cw / snap) * snap);
        const outH = Math.max(snap, Math.round(ch / snap) * snap);
        const angTxt = Math.abs(ang) >= 0.05 ? `  · ${ang.toFixed(1)}°` : "";
        info.textContent = state.imgW
            ? `${cw}×${ch}${angTxt}  →  ${outW}×${outH}`
                + (get("aspect_lock") ? "  · 🔒" : "")
                + (constrained() ? "" : "  · ⛶ free")
            : "no image yet";
    }

    function localMouse(e) {
        const r = canvas.getBoundingClientRect();
        // CSS-pixel canvas coords (see bat_ref_aligner.js for rationale).
        // clientWidth is layout-box; r.width includes LiteGraph zoom.
        return {
            x: (e.clientX - r.left) * (canvas.clientWidth  / r.width),
            y: (e.clientY - r.top)  * (canvas.clientHeight / r.height),
        };
    }
    function handleAt(p) {
        const R = 10;
        for (const h of state.handles) {
            if (Math.abs(h.x - p.x) <= R && Math.abs(h.y - p.y) <= R) return h.name;
        }
        return null;
    }
    function insideBox(p) {
        // Hit-test in image (rect-local) coords so it works for rotated rects too.
        if (!state.imgW) return false;
        const cx = get("crop_x") | 0, cy = get("crop_y") | 0;
        const cw = get("crop_w") | 0, ch = get("crop_h") | 0;
        const ang = (get("crop_angle") || 0) * Math.PI / 180;
        const cosA = Math.cos(ang), sinA = Math.sin(ang);
        const ip = d2c(p.x, p.y);
        const dx = ip.x - (cx + cw / 2), dy = ip.y - (cy + ch / 2);
        const u =  cosA * dx + sinA * dy;
        const v = -sinA * dx + cosA * dy;
        return Math.abs(u) <= cw / 2 && Math.abs(v) <= ch / 2;
    }
    function insideImage(p) {
        if (!state.imgW) return false;
        const o = c2d(0, 0);
        return p.x >= o.x && p.x <= o.x + state.imgW * state.dispScale
            && p.y >= o.y && p.y <= o.y + state.imgH * state.dispScale;
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!state.img) return;
        const p = localMouse(e);
        const cP = d2c(p.x, p.y);
        const handleName = handleAt(p);
        let mode = null;
        if (handleName === "rot") mode = "rot";
        else if (handleName) mode = "resize:" + handleName;
        else if (insideBox(p)) mode = "move";
        else if (insideImage(p)) mode = "new";
        if (!mode) return;

        e.preventDefault();
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}

        const ratio = parseRatio(get("aspect_ratio"));
        const lock = !!get("aspect_lock") && ratio !== null;
        const baseW = get("crop_w") | 0, baseH = get("crop_h") | 0;
        const baseX = get("crop_x") | 0, baseY = get("crop_y") | 0;
        const angDeg = get("crop_angle") || 0;
        const angRad = angDeg * Math.PI / 180;
        const cos = Math.cos(angRad), sin = Math.sin(angRad);
        const base = {
            startC: cP,
            x: baseX, y: baseY, w: baseW, h: baseH,
            cx: baseX + baseW / 2, cy: baseY + baseH / 2,
            angDeg, cos, sin, lock, ratio,
        };
        if (mode.startsWith("resize:")) {
            // Anchor = the opposite point of the dragged handle, in image
            // coords; stays fixed under rotation through the drag.
            const hSpec = state.handles.find(h => h.name === mode.slice(7));
            const lx = hSpec ? hSpec.lx : 0;
            const ly = hSpec ? hSpec.ly : 0;
            const ax_loc = -lx * baseW / 2;
            const ay_loc = -ly * baseH / 2;
            base.handleLx = lx;
            base.handleLy = ly;
            base.anchorImg = {
                x: base.cx + cos * ax_loc - sin * ay_loc,
                y: base.cy + sin * ax_loc + cos * ay_loc,
            };
        } else if (mode === "rot") {
            base.startMouseAng = Math.atan2(cP.y - base.cy, cP.x - base.cx);
            base.startAngDeg = angDeg;
        } else if (mode === "new") {
            set("crop_x", Math.round(cP.x));
            set("crop_y", Math.round(cP.y));
            set("crop_w", 1);
            set("crop_h", 1);
            // Drawing a fresh rect — drop any rotation so it's axis-aligned.
            if (Math.abs(angDeg) > 1e-3) set("crop_angle", 0);
            base.angDeg = 0; base.cos = 1; base.sin = 0;
        }
        state.drag = { mode, base };
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!state.drag) return;
        const p = localMouse(e);
        const cP = d2c(p.x, p.y);
        const { mode, base } = state.drag;
        const Wimg = state.imgW, Himg = state.imgH;
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const rotated = Math.abs(base.angDeg) > 1e-3;
        // When "constrain to canvas" is off the axis-aligned rect is allowed
        // to run past the image edge (backend zero-pads the overhang).
        const con = constrained();

        if (mode === "rot") {
            const curAng = Math.atan2(cP.y - base.cy, cP.x - base.cx);
            let deg = base.startAngDeg + (curAng - base.startMouseAng) * 180 / Math.PI;
            deg = ((deg + 180) % 360 + 360) % 360 - 180;
            set("crop_angle", +deg.toFixed(1));
            return;
        }

        let x = base.x, y = base.y, w = base.w, h = base.h;

        if (mode === "move") {
            const dx = cP.x - base.startC.x, dy = cP.y - base.startC.y;
            if (rotated || !con) {
                x = Math.round(base.x + dx);
                y = Math.round(base.y + dy);
            } else {
                x = clamp(Math.round(base.x + dx), 0, Math.max(0, Wimg - base.w));
                y = clamp(Math.round(base.y + dy), 0, Math.max(0, Himg - base.h));
            }
            w = base.w; h = base.h;
        } else if (mode === "new") {
            const anchX = base.startC.x, anchY = base.startC.y;
            const dirX = cP.x >= anchX ? 1 : -1;
            const dirY = cP.y >= anchY ? 1 : -1;
            let nw = Math.abs(cP.x - anchX), nh = Math.abs(cP.y - anchY);
            if (base.lock) {
                const wantH = nw / base.ratio;
                if (wantH > nh) nh = wantH; else nw = nh * base.ratio;
            }
            if (con && base.lock) {
                // Uniform shrink about the anchor (the initial click) so the
                // ratio survives the canvas clamp.
                let s = 1;
                if (dirX > 0) s = Math.min(s, (Wimg - anchX) / Math.max(1e-6, nw));
                else          s = Math.min(s, anchX / Math.max(1e-6, nw));
                if (dirY > 0) s = Math.min(s, (Himg - anchY) / Math.max(1e-6, nh));
                else          s = Math.min(s, anchY / Math.max(1e-6, nh));
                s = Math.max(0, Math.min(1, s));
                nw *= s; nh *= s;
                x = Math.round(dirX > 0 ? anchX : anchX - nw);
                y = Math.round(dirY > 0 ? anchY : anchY - nh);
                w = Math.max(1, Math.round(nw));
                h = Math.max(1, Math.round(nh));
            } else if (con) {
                const nx = Math.min(anchX, cP.x), ny = Math.min(anchY, cP.y);
                x = clamp(Math.round(nx), 0, Math.max(0, Wimg - 1));
                y = clamp(Math.round(ny), 0, Math.max(0, Himg - 1));
                w = clamp(Math.round(nw), 1, Wimg - x);
                h = clamp(Math.round(nh), 1, Himg - y);
            } else {
                x = Math.round(dirX > 0 ? anchX : anchX - nw);
                y = Math.round(dirY > 0 ? anchY : anchY - nh);
                w = Math.max(1, Math.round(nw));
                h = Math.max(1, Math.round(nh));
            }
        } else if (rotated) {
            // Rotation-aware resize: project the mouse displacement from the
            // (fixed) anchor onto the rect's local axes.
            const { handleLx: lx, handleLy: ly, anchorImg, cos, sin } = base;
            const dx = cP.x - anchorImg.x, dy = cP.y - anchorImg.y;
            const u =  cos * dx + sin * dy;
            const v = -sin * dx + cos * dy;
            const isCorner = (lx !== 0 && ly !== 0);
            let new_w = base.w, new_h = base.h;
            if (isCorner) {
                new_w = Math.max(1, Math.abs(u));
                new_h = Math.max(1, Math.abs(v));
                if (base.lock) {
                    const wantH = new_w / base.ratio;
                    if (wantH > new_h) new_h = wantH;
                    else new_w = new_h * base.ratio;
                }
            } else if (lx === 0) {
                new_h = Math.max(1, Math.abs(v));
            } else {
                new_w = Math.max(1, Math.abs(u));
            }
            // new_center = anchor + R(θ) * (lx * new_w/2, ly * new_h/2)
            const offx = lx * new_w / 2;
            const offy = ly * new_h / 2;
            const cx = anchorImg.x + cos * offx - sin * offy;
            const cy = anchorImg.y + sin * offx + cos * offy;
            x = Math.round(cx - new_w / 2);
            y = Math.round(cy - new_h / 2);
            w = Math.max(1, Math.round(new_w));
            h = Math.max(1, Math.round(new_h));
        } else { // resize:<which>, axis-aligned (no rotation)
            const which = mode.slice(7);
            const r0 = base.x + base.w, b0 = base.y + base.h;
            let nx = base.x, ny = base.y, nr = r0, nb = b0;
            if (which.includes("l")) nx = cP.x;
            if (which.includes("r")) nr = cP.x;
            if (which.includes("t")) ny = cP.y;
            if (which.includes("b")) nb = cP.y;
            // Edge midpoint handles only move that one axis.
            if (which === "t" || which === "b") { nx = base.x; nr = r0; }
            if (which === "l" || which === "r") { ny = base.y; nb = b0; }
            if (nr < nx) { const t = nr; nr = nx; nx = t; }
            if (nb < ny) { const t = nb; nb = ny; ny = t; }
            let nw = nr - nx, nh = nb - ny;
            const locked = base.lock && which.length === 2;
            if (locked) {
                const wantH = nw / base.ratio;
                if (wantH > nh) {
                    if (which.includes("t")) ny = nb - wantH; else nb = ny + wantH;
                    nh = wantH;
                } else {
                    const wantW = nh * base.ratio;
                    if (which.includes("l")) nx = nr - wantW; else nr = nx + wantW;
                    nw = wantW;
                }
            }
            if (con && locked) {
                // Aspect-locked AND constrained: shrink the rect UNIFORMLY
                // about the fixed anchor corner so it fits the canvas without
                // breaking the ratio (clamping each edge independently would
                // deform it). The anchor is the corner NOT being dragged.
                const ax = which.includes("l") ? nr : nx;   // fixed x edge
                const ay = which.includes("t") ? nb : ny;   // fixed y edge
                const dx = which.includes("l") ? -nw : nw;  // signed extent from anchor
                const dy = which.includes("t") ? -nh : nh;
                let s = 1;
                if (dx > 0)      s = Math.min(s, (Wimg - ax) / dx);
                else if (dx < 0) s = Math.min(s, (0 - ax) / dx);
                if (dy > 0)      s = Math.min(s, (Himg - ay) / dy);
                else if (dy < 0) s = Math.min(s, (0 - ay) / dy);
                s = Math.max(0, Math.min(1, s));
                const fx = ax + dx * s, fy = ay + dy * s;   // scaled far corner
                x = Math.round(Math.min(ax, fx));
                y = Math.round(Math.min(ay, fy));
                w = Math.max(1, Math.round(Math.abs(fx - ax)));
                h = Math.max(1, Math.round(Math.abs(fy - ay)));
            } else if (con) {
                x = Math.round(Math.max(0, nx));
                y = Math.round(Math.max(0, ny));
                w = Math.max(1, Math.round(Math.min(nr, Wimg) - x));
                h = Math.max(1, Math.round(Math.min(nb, Himg) - y));
            } else {
                x = Math.round(nx);
                y = Math.round(ny);
                w = Math.max(1, Math.round(nr - nx));
                h = Math.max(1, Math.round(nb - ny));
            }
        }

        set("crop_x", x);
        set("crop_y", y);
        set("crop_w", w);
        set("crop_h", h);
    });

    const endDrag = (e) => {
        if (state.drag) {
            state.drag = null;
            try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
            clampRectToImage();
        }
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // Tracked so it's disconnected when the node is deleted (see bat_lifecycle).
    const track = batTrack(node);
    track.observer(new ResizeObserver(() => render()), root);

    // Display-only zoom control (bottom-left). Lets the artist pull back to
    // see area outside the frame when placing an off-canvas crop.
    attachZoomControl({ wrap: root, canvas, state, onChange: render, corner: "bl" });

    node._batCrop = { root, canvas, ctx, state, render };

    // rAF watcher: re-render whenever any controlling widget changes from
    // ANY source (handles, number widgets, typed values, workflow restore).
    let lastSig = "";
    let lastConstrain = constrained();
    // track.rafLoop stops rescheduling once the node is removed. The old
    // hand-rolled loop did `if (!root.isConnected) { requestAnimationFrame(watch);
    // return; }` — which re-armed itself FOREVER after the node was deleted,
    // leaking a per-frame callback (and this whole closure) for the session.
    track.rafLoop(() => {
        if (!root.isConnected) return;   // detached but node still alive: idle
        // Re-clamp the rect the moment the artist turns constrain back ON,
        // so a rect that was dragged off-canvas snaps back inside.
        const con = constrained();
        if (con && !lastConstrain && !state.drag) clampRectToImage();
        lastConstrain = con;
        const sig = `${get("crop_x")}|${get("crop_y")}|${get("crop_w")}|${get("crop_h")}|${get("crop_angle")}|${get("snap_to")}|${get("aspect_lock")}|${get("aspect_ratio")}|${con}`;
        if (sig !== lastSig) { lastSig = sig; render(); }
    });

    function ingest(b64, w, h) {
        const img = new Image();
        img.onload = () => {
            state.img = img;
            state.imgW = w || img.naturalWidth;
            state.imgH = h || img.naturalHeight;
            // Auto-fit the rect to the full image only on a truly fresh node
            // (still at the Python defaults). Reloaded workflows skip this so
            // the user's chosen rect is preserved.
            const atDefaults =
                get("crop_x") === 0 && get("crop_y") === 0 &&
                get("crop_w") === 512 && get("crop_h") === 512;
            if (atDefaults && (state.imgW !== 512 || state.imgH !== 512)) {
                set("crop_x", 0); set("crop_y", 0);
                set("crop_w", state.imgW); set("crop_h", state.imgH);
            }
            clampRectToImage();
            hint.textContent = "";
            render();
        };
        img.src = `data:image/jpeg;base64,${b64}`;
    }
    node._batCropIngest = ingest;

    render();
    return root;
}

app.registerExtension({
    name: "BAT.Crop",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const el = makeEditor(this);
            // Dual-mode sizing. Under Nodes 2.0 this root (height:100%, no
            // min-height) had NO height source and collapsed to ~1px.
            addBatDOMWidget(this, "crop_editor", "crop_editor", el, {
                minWidth: 460, height: 580, growable: true,
            });
            clampNodeSize(this, 460, 580);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (!message || !this._batCropIngest) return r;
            const one = (v) => Array.isArray(v) ? v[0] : v;
            if (message.preview) {
                this._batCropIngest(one(message.preview), one(message.w), one(message.h));
            }
            return r;
        };
    },
});
