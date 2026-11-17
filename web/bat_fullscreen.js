/**
 * BAT — fullscreen (maximise-in-window) mode for the on-node editors.
 *
 * Why
 * ---
 * The advanced editors (Roto, Animated Crop, Animated Grade, Layered Images,
 * HDR Tonal Composite, Rescale, Video Combine) are real tools living inside a
 * node, and a node is a small box on a canvas the artist has to zoom to exactly
 * the right scale before the editor is usable. This adds a ⛶ button in the
 * editor's top-right corner: one click and the SAME editor fills the ComfyUI
 * window. An editor that already HAS a fullscreen control of its own passes
 * `button: false` and wires the returned toggle() to it — see Video Combine.
 *
 * The editor is moved, not rebuilt. Every bit of state — the roto document, the
 * decoded frames, the undo stack, the worker caches — is held in closures over
 * the root element's build function, so reparenting the root is the only way to
 * get a big editor that is still the node's editor. A second instance would
 * have to synchronise, and would be wrong the first time someone edited either.
 *
 * The one trap: the frontend takes its element back
 * -------------------------------------------------
 * On frontend 1.49.6 a DOM widget's element is not ours to move. Each widget is
 * rendered by `DomWidget.vue`, which owns a wrapper div and re-appends our root
 * into it on every visibility change:
 *
 *     const mountElementIfVisible = () => {
 *       if (!(widgetState.visible && isDOMWidget(widget) && widgetElement.value)) return
 *       if (widgetElement.value.contains(widget.element)) return
 *       widget.element.classList.add('h-full', 'w-full')
 *       widgetElement.value.appendChild(widget.element)          // ← steals it back
 *     }
 *     watch(() => widgetState.visible, () => mountElementIfVisible())
 *
 * and `widgetState.visible` flips every time the node scrolls in or out of
 * view, so a naive appendChild() into an overlay survives until the artist
 * nudges the canvas. Note the guard clause, though: the re-mount is skipped
 * entirely while the widget is not visible, and visibility comes from
 * `DomWidgets.vue`:
 *
 *     if (!widget.isVisible() || !widgetState.active) { widgetState.visible = false; continue }
 *
 * with `isVisible()` in `domWidget.ts` being
 *
 *     !this.hidden && !this.computedDisabled && this.node.isWidgetVisible(this)
 *
 * So `widget.hidden = true` is the whole defence: the frontend stops managing
 * the element, and hands it straight back when we clear the flag on exit. We
 * deliberately do NOT set `widget.type = "hidden"` — that is the flag
 * addBatDOMWidget's computeLayoutSize keys its zero-height answer on, and
 * zeroing the height would collapse the node under the overlay and make it jump
 * on the way back out.
 *
 * What this does NOT do
 * ---------------------
 * Native `requestFullscreen()`. The Fullscreen API puts the element in the top
 * layer, above everything ComfyUI appends to document.body — its menus, colour
 * pickers, litegraph context menus and toasts would all render behind the
 * editor. Maximising inside the window keeps the whole application coherent and
 * costs only the browser chrome.
 *
 * Usage
 * -----
 *   import { addBatFullscreen } from "./bat_fullscreen.js";
 *
 *   // in buildEditor(), next to the element it describes:
 *   canvasWrap.dataset.batFsMount = "1";
 *
 *   // in onNodeCreated():
 *   const w = addBatDOMWidget(node, "bat_roto_editor", "bat_roto_editor", el, {...});
 *   addBatFullscreen(node, w, el);
 *
 * `el` is the root that gets moved into the overlay. The `data-bat-fs-mount`
 * marker only says where the button hangs — put it on the editor's picture area
 * so the control lands in the same place on every node. Anything the editor
 * already draws in that corner (a readout badge, a help panel) moves left by
 * BUTTON_BOX to make room.
 */

import { refreshBatLayout } from "./bat_node_layout.js";
import { batTrack, isNodeAlive } from "./bat_lifecycle.js";

/**
 * Overlay stacking order.
 *
 * Measured against the frontend's own bands (1.49.6): DOM widgets sit at the
 * node's index in `graph.nodes` (getDomWidgetZIndex — single digits in
 * practice), side panels and the settings dialog occupy 1000–2001, and
 * tooltips, litegraph context menus and toasts live at 8888–99999.
 *
 * 3000 is chosen to sit between those two groups on purpose: above every
 * graph-level surface the overlay is replacing, and BELOW the notification
 * layer, so an execution error raised while the artist is mid-edit is still
 * visible instead of being hidden behind the editor.
 */
const OVERLAY_Z = 3000;

/**
 * Horizontal room the corner button needs, measured from the right edge of the
 * picture area — the pill's own width plus its 6px inset. Editors with a
 * readout in that corner offset it by this much; bump both together if the
 * label ever changes length.
 */
export const BUTTON_BOX = 104;

const STYLE_ID = "bat-fullscreen-style";

/**
 * Class names are prefixed `bat-fs-` rather than anything generic: every pack
 * in the suite shares one CSS namespace, so a bare `.overlay` or `.fs-head`
 * here would restyle somebody else's panel.
 */
const CSS = `
.bat-fs-overlay {
    position: fixed; inset: 0; z-index: ${OVERLAY_Z};
    display: flex; flex-direction: column;
    background: #07070a;
    font: 12px sans-serif; color: #cde;
    outline: none;
}
.bat-fs-head {
    flex: 0 0 auto; display: flex; align-items: center; gap: 10px;
    padding: 4px 8px; background: #121218; border-bottom: 1px solid #2a2f37;
    user-select: none;
}
.bat-fs-title { font-weight: 600; color: #dde; white-space: nowrap; }
.bat-fs-spacer { flex: 1 1 auto; }
.bat-fs-hint { font: 11px monospace; color: #7b8595; white-space: nowrap; }
.bat-fs-close {
    background: #222; color: #cde; border: 1px solid #3a3f47; border-radius: 3px;
    font: 12px/1 monospace; padding: 4px 9px; cursor: pointer;
}
.bat-fs-close:hover { background: #303640; border-color: #4a515c; }
.bat-fs-body {
    flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column;
    padding: 6px;
}
.bat-fs-btn {
    position: absolute; top: 6px; right: 6px; z-index: 40;
    height: 26px; padding: 0 10px 0 8px;
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(14,16,20,0.88); color: #e6edf5;
    border: 1px solid #5a6472; border-radius: 4px;
    font: 600 11px/1 system-ui, -apple-system, "Segoe UI", sans-serif;
    letter-spacing: .02em; white-space: nowrap; cursor: pointer;
    opacity: .92; box-shadow: 0 1px 4px rgba(0,0,0,.55);
    transition: background .12s ease, border-color .12s ease, opacity .12s ease;
}
.bat-fs-btn:hover { opacity: 1; background: #2b333e; border-color: #8a97a8; }
.bat-fs-btn:focus-visible { opacity: 1; outline: 2px solid #7ab8ff; outline-offset: 1px; }
.bat-fs-btn > .bat-fs-glyph { font: 15px/1 monospace; }
`;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
}

/**
 * The single open editor, or null. One at a time: two overlays would stack and
 * the lower one would be unreachable, and the artist has no way to tell which
 * node they are editing.
 */
let active = null;

/** Close whatever is open, if anything. Safe to call at any time. */
export function exitBatFullscreen() {
    if (active) active.exit();
}

/** Is this node's editor the one currently maximised? */
export function isBatFullscreen(node) {
    return !!active && active.node === node;
}

export function addBatFullscreen(node, widget, el, opts = {}) {
    if (!node || !widget || !el) return null;
    // onNodeCreated can run more than once for a node across a clone or a
    // re-register; a second button would stack on top of the first.
    if (el.dataset.batFsWired === "1") return null;
    el.dataset.batFsWired = "1";

    ensureStyles();

    const { title = null, mount = null, button = true,
            onEnter = null, onExit = null } = opts;

    let btn = null, btnGlyph = null, btnLabel = null;

    // `button: false` for an editor that ALREADY has a fullscreen control of
    // its own — Video Combine's player has one in its transport bar, and a
    // second floating pill over the picture would be two controls for one
    // thing. That caller wires its own button to the returned toggle() and
    // relabels it from onEnter / onExit.
    if (button) {
        // Where the ⛶ button hangs. It should be the editor's PICTURE area, so
        // the control lands in the same place on every node the way a viewer's
        // maximise button does — not the root, which for rescale starts with a
        // toolbar and for roto / animated grade is a canvas with a sidebar
        // beside it.
        //
        // Resolved from a `data-bat-fs-mount` marker rather than an argument
        // because every editor builds its DOM in a `buildEditor(node)` that
        // returns only the root: the canvas wrap is a local in there and is
        // simply not in scope at the addBatFullscreen() call site in
        // onNodeCreated. A marker is one line in the builder, next to the
        // element it describes. Falls back to the root, which is correct for an
        // editor whose picture starts at the top.
        const host = mount || el.querySelector("[data-bat-fs-mount]") || el;
        if (getComputedStyle(host).position === "static") {
            // Absolute positioning would otherwise escape to the nearest
            // positioned ancestor and put the button somewhere arbitrary.
            host.style.position = "relative";
        }

        btn = document.createElement("button");
        btn.type = "button";
        btn.className = "bat-fs-btn";
        // Glyph AND word. ⛶ on its own was 22px at 55% opacity over a dark
        // plate, which is invisible in practice — the control nobody finds may
        // as well not exist. The label is what makes it findable; the glyph is
        // what makes it recognisable once you know it's there.
        btnGlyph = document.createElement("span");
        btnGlyph.className = "bat-fs-glyph";
        btnGlyph.textContent = "⛶";
        btnLabel = document.createElement("span");
        btnLabel.textContent = "Fullscreen";
        btn.append(btnGlyph, btnLabel);
        btn.title = "Fullscreen";
        btn.setAttribute("aria-label", "Fullscreen");
        // The editors below read pointer events on their own canvases; none of
        // this button's belong to them, and litegraph must not see them either.
        btn.addEventListener("pointerdown", (e) => e.stopPropagation());
        btn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            toggle();
        });
        host.appendChild(btn);
    }

    // Where the root came from, so it goes back exactly there. Captured at
    // enter() rather than now: the frontend has not mounted the element yet
    // when the editor is built.
    let home = null;       // { parent, next, style }
    let overlay = null;
    let onKey = null;

    const enter = () => {
        if (active) active.exit();          // one at a time
        if (!isNodeAlive(node)) return;

        home = {
            parent: el.parentNode,
            next: el.nextSibling,
            // One string captures every inline property at once — including the
            // --comfy-widget-min-height / -max-height custom properties
            // addBatDOMWidget publishes, and each editor's own cssText.
            style: el.getAttribute("style"),
        };

        overlay = document.createElement("div");
        overlay.className = "bat-fs-overlay";
        overlay.tabIndex = -1;
        // A modal, as far as the frontend is concerned — and that is what makes
        // the keyboard ours. Core's isModalOpen() (utils/modalUtil.ts) counts
        // any rendered [role=dialog][aria-modal=true], and both of its callers
        // then stand down: keybindHandler, which otherwise preventDefault()s
        // every bare Escape (Comfy.Graph.ExitSubgraph) before onKey below can
        // see it, and ChangeTracker, whose Ctrl+Z reloads the whole graph —
        // closing this overlay — on top of the editor's own undo. The cost:
        // core shortcuts (Ctrl+Enter, Ctrl+S) are inert while maximised, the
        // same as under any other dialog.
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");

        const head = document.createElement("div");
        head.className = "bat-fs-head";

        const titleEl = document.createElement("div");
        titleEl.className = "bat-fs-title";
        titleEl.textContent = title || node.title || "BAT";
        head.appendChild(titleEl);
        overlay.setAttribute("aria-label", titleEl.textContent);

        const spacer = document.createElement("div");
        spacer.className = "bat-fs-spacer";
        head.appendChild(spacer);

        const hint = document.createElement("div");
        hint.className = "bat-fs-hint";
        hint.textContent = "Esc to exit";
        head.appendChild(hint);

        const close = document.createElement("button");
        close.type = "button";
        close.className = "bat-fs-close";
        close.textContent = "✕  Exit fullscreen";
        close.addEventListener("click", (e) => { e.preventDefault(); exit(); });
        head.appendChild(close);

        const body = document.createElement("div");
        body.className = "bat-fs-body";

        overlay.appendChild(head);
        overlay.appendChild(body);

        // Stop the frontend re-mounting the root into its own wrapper while we
        // hold it — see the header comment. Must happen BEFORE the move, so no
        // layout pass in between sees a visible widget with no element.
        widget.hidden = true;
        try { node.graph?.incrementVersion?.(); } catch (_) { /* older graphs */ }

        // Fill the overlay. The editors size themselves from their own
        // ResizeObserver on their canvas wrap, so this is all they need:
        // min-height 0 releases the design-height floor each editor sets in its
        // cssText (roto pins 480px), and the two custom properties go with it so
        // nothing re-derives the node height from them while we are out.
        el.dataset.batFullscreen = "1";
        el.style.setProperty("flex", "1 1 auto");
        el.style.setProperty("width", "100%");
        el.style.setProperty("height", "auto");
        el.style.setProperty("min-height", "0");
        el.style.setProperty("max-height", "none");
        el.style.removeProperty("--comfy-widget-min-height");
        el.style.removeProperty("--comfy-widget-max-height");

        body.appendChild(el);
        document.body.appendChild(overlay);

        if (btn) {
            btnGlyph.textContent = "⤡";
            btnLabel.textContent = "Exit";
            btn.title = "Exit fullscreen (Esc)";
            btn.setAttribute("aria-label", "Exit fullscreen");
        }

        // Bubble phase on window, deliberately — it runs AFTER the editor's own
        // keydown handler on the root, so `defaultPrevented` tells us whether
        // the editor already used the key. Roto's Escape deselects points and
        // closes an open shape; that has to keep working, and only a press it
        // ignored should close the overlay.
        onKey = (e) => {
            if (e.key !== "Escape" || e.defaultPrevented) return;
            e.preventDefault();
            e.stopPropagation();
            exit();
        };
        window.addEventListener("keydown", onKey);

        // Focus the editor root when it takes keys (roto and rescale set
        // tabIndex=0), otherwise the overlay, so Escape has somewhere to land.
        try { (el.tabIndex >= 0 ? el : overlay).focus({ preventScroll: true }); }
        catch (_) { /* focus is best-effort */ }

        active = { node, exit };
        try { onEnter?.(); } catch (e) { console.error("[BAT.fullscreen] onEnter failed:", e); }
    };

    const exit = () => {
        if (active?.node !== node) return;
        active = null;

        if (onKey) { window.removeEventListener("keydown", onKey); onKey = null; }

        delete el.dataset.batFullscreen;
        if (home?.style != null) el.setAttribute("style", home.style);
        else el.removeAttribute("style");

        // Put the root back in its wrapper. `next` is re-validated because the
        // wrapper may have been re-rendered underneath us; if the wrapper is
        // gone entirely (a graph reload rebuilt the node) the element is simply
        // detached, and clearing `hidden` lets DomWidget.vue's own
        // mountElementIfVisible re-adopt it on the next visibility flip.
        const parent = home?.parent;
        if (parent?.isConnected) {
            const next = home.next?.parentNode === parent ? home.next : null;
            parent.insertBefore(el, next);
        } else {
            el.remove();
        }
        home = null;

        try { overlay?.remove(); } catch (_) {}
        overlay = null;

        if (btn) {
            btnGlyph.textContent = "⛶";
            btnLabel.textContent = "Fullscreen";
            btn.title = "Fullscreen";
            btn.setAttribute("aria-label", "Fullscreen");
        }

        if (isNodeAlive(node)) {
            widget.hidden = false;
            // Re-publishes the height custom properties we stripped and nudges
            // both renderers to re-derive the node's size.
            refreshBatLayout(node, widget);
        }

        try { onExit?.(); } catch (e) { console.error("[BAT.fullscreen] onExit failed:", e); }
    };

    const toggle = () => { (active?.node === node ? exit : enter)(); };

    // Deleting the node — or any Ctrl+Z, which clears and rebuilds the whole
    // graph — must not leave an overlay holding a dead editor.
    try { batTrack(node).dispose(() => { if (active?.node === node) exit(); }); }
    catch (e) { console.error("[BAT.fullscreen] could not register teardown:", e); }

    return { enter, exit, toggle, button: btn };
}
