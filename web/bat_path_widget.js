/**
 * BAT — shared path-autocomplete widget.
 *
 * A rounded canvas textbox that opens a LiteGraph-style search popup with
 * filesystem autocomplete. 🦇 Video Loader and 🦇 Frame Picker both use it,
 * each pointing it at its own `/getpath` route; it was copy-pasted between the
 * two before this module existed, so a fix to one silently missed the other.
 *
 * Three details are less obvious than they look:
 *
 *  * **The width.** A canvas-drawn widget gets its draw width as
 *    `widget.width || node.size[0]`, and `WidgetLegacy.vue` stamps
 *    `widget.width` with the width of whatever container last drew it — the
 *    parameters panel, or the node's own DOM copy. Once stamped, the pill
 *    freezes at that width: it hangs off the body of a narrow node and stops
 *    short on a wide one, and the frontend's hit-testing (which uses the same
 *    expression) is off by the same amount, so clicks land beside the pill.
 *    `unpinWidgetWidth` makes the property permanently unset, which is the
 *    fallback both expressions want.
 *
 *  * **Fitting the value.** Cutting at a fixed character count throws away
 *    most of a widened node and still overflows a narrow one, so measure
 *    instead — and trim from the FRONT, because the informative end of a path
 *    is the filename. The label steps aside entirely when the value needs the
 *    whole row.
 *
 *  * **Committing.** Assigning `widget.value` fires nothing. Anything
 *    listening the way the rest of the frontend does (`node.onWidgetChanged`,
 *    which the BaseWidget setter raises for normal widgets) never hears a
 *    value picked in the dialog, so a dependent widget can sit stale on a
 *    freshly chosen path. Announce it both ways.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { unpinWidgetWidth } from "./bat_node_layout.js";

export function pathStem(p) {
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? [p.slice(0, i + 1), p.slice(i + 1)] : ["", p];
}

// Trim `text` from the front until it fits `maxWidth` px, prefixing "…".
function fitTail(ctx, text, maxWidth) {
    if (maxWidth <= 0) return "";
    if (ctx.measureText(text).width <= maxWidth) return text;
    // Binary search the longest tail that fits, rather than measuring
    // character by character on every canvas redraw.
    let lo = 0, hi = text.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;              // drop `mid` chars from the front
        if (ctx.measureText("…" + text.slice(mid)).width <= maxWidth) hi = mid;
        else lo = mid + 1;
    }
    return "…" + text.slice(lo);
}

function commitValue(widget, node, value) {
    const oldValue = widget.value;
    widget.value = value;
    widget.callback?.(value);
    node?.onWidgetChanged?.(widget.name, value, oldValue, widget);
    node?.setDirtyCanvas?.(true, true);
}

function openPathSearch(event, widget, node) {
    if (widget._prompt) return true;
    widget._prompt = true;

    const dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded";
    dialog.innerHTML =
        `<span class="name">${widget._batTitle || "Path"}</span>` +
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
    const route = widget._batRoute;

    function commit(v) {
        commitValue(widget, node, v);
        dialog.close();
    }

    input.addEventListener("keydown", (e) => {
        if (e.keyCode === 27) dialog.close();            // ESC
        else if (e.keyCode === 13) commit(input.value);  // ENTER
        else if (e.keyCode === 9) {                      // TAB — complete
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
            if (extensions) {
                params.set("extensions",
                           Array.isArray(extensions) ? extensions.join(",") : extensions);
            }
            try {
                const r = await fetch(api.apiURL(`${route}?${params}`));
                options = r.ok ? await r.json() : [];
                if (!Array.isArray(options)) options = [];
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
            // Sequence patterns (render.####.exr) are synthesised by the route
            // rather than being real files — colour them so it's clear which
            // entries stand for a whole clip.
            if (name.includes("#")) el.style.color = "#8ae234";
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

// Extra height for the optional subtitle line (Nodes 2.0 only — see below).
const SUBTITLE_H = 14;

function drawPathWidget(ctx, node, widgetWidth, y, H) {
    const m = 15;
    // The zoom cut-off is a Nodes 1.0 economy. Under 2.0 this is painted once
    // into the widget's own canvas and NOT repainted on zoom (a CSS transform
    // doesn't fire WidgetLegacy's ResizeObserver), so a node mounted zoomed
    // out kept a blank pill after zooming back in.
    const showText = !!LiteGraph.vueNodesMode || app.canvas.ds.scale >= 0.5;
    // Under 2.0 the node's summary line lives inside the widget (the node's
    // onDrawForeground is never called there), so the pill takes the top of
    // the row and the subtitle the strip below it.
    const subtitle = (this._batSubtitle && LiteGraph.vueNodesMode)
        ? this._batSubtitle() : null;
    if (this._batSubtitle && LiteGraph.vueNodesMode) H -= SUBTITLE_H;
    // See the note at the top: `widget.width` is unpinned, so the width handed
    // in is the node's own on the graph canvas and the container's in the
    // parameters panel — which is what we want in both places.
    const width = widgetWidth || node?.size?.[0] || 200;

    ctx.textAlign = "left";
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(m, y, width - m * 2, H, [H * 0.5]);
    ctx.fill();
    if (!showText) return;

    if (!this.disabled) ctx.stroke();
    ctx.save();
    ctx.beginPath();
    ctx.rect(m, y, width - m * 2, H);
    ctx.clip();

    const textLeft = m * 2 + 5;
    const textRight = width - m * 2 - 5;
    const rowWidth = textRight - textLeft;
    const val = String(this.value || "");
    const labelWidth = ctx.measureText(this.name).width;
    const gap = 12;
    const showLabel = rowWidth - labelWidth - gap >= 120;

    if (showLabel) {
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
        ctx.fillText(this.name, textLeft, y + H * 0.7);
    }
    ctx.textAlign = "right";
    ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : "#777";
    const avail = rowWidth - (showLabel ? labelWidth + gap : 0);
    ctx.fillText(fitTail(ctx, val, avail), textRight, y + H * 0.7);
    ctx.restore();

    const [subText, subColour] = subtitle || [];
    if (subText) {
        ctx.save();
        ctx.font = "10px monospace";
        ctx.fillStyle = subColour || LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
        ctx.textAlign = "right";
        ctx.fillText(fitTail(ctx, subText, width - m * 2), width - m - 5,
                     y + H + SUBTITLE_H - 3);
        ctx.restore();
    }
}

/**
 * Build the widget.
 *
 * @param {object}  spec
 * @param {string}  spec.name    widget name (must match the node's input)
 * @param {string}  spec.value   initial value
 * @param {object}  spec.options the input's options dict (bat_path_extensions)
 * @param {string}  spec.route   the /getpath route to autocomplete against
 * @param {string}  spec.title   heading shown on the search popup
 * @param {Function} [spec.subtitle] () => [text, colour]: a one-line summary
 *                   drawn under the pill under Nodes 2.0 only, where a node's
 *                   own onDrawForeground (the 1.0 place for it) never runs
 */
export function makeBatPathWidget({ name = "path", value = "", options = {},
                                    route = "/bat/getpath", title = "Path",
                                    subtitle = null } = {}) {
    const w = {
        name,
        type: "BAT.PATH",
        value: value || "",
        options: options || {},
        _batRoute: route,
        _batTitle: title,
        _batSubtitle: subtitle,
        draw: drawPathWidget,
        mouse(event, pos, node) {
            // pointerdown only: `mouse` also receives moves and wheel events,
            // and opening the dialog on those makes the field impossible to
            // drag past.
            if (event.type !== "pointerdown") return false;
            return openPathSearch(event, this, node);
        },
        computeSize() {
            const extra = (this._batSubtitle && LiteGraph.vueNodesMode) ? SUBTITLE_H : 0;
            return [200, LiteGraph.NODE_WIDGET_HEIGHT + extra];
        },
    };
    return unpinWidgetWidth(w);
}

/**
 * Put a path widget into `node.widgets[index]` and return the widget that is
 * actually there.
 *
 * Unpinning has to happen AFTER insertion. The frontend adopts a plain-object
 * widget when it lands in node.widgets (widgetMap.ts adoptConcreteWidget), and
 * that descriptor merge keeps BaseWidget's own `width` field in place of the
 * accessor unpinWidgetWidth installed — so an unpin done in makeBatPathWidget
 * alone is silently undone and the pill pins to whatever width last drew it.
 */
export function installBatPathWidget(node, index, widget) {
    node.widgets[index] = widget;
    const live = node.widgets[index] || widget;
    unpinWidgetWidth(live);
    return live;
}
