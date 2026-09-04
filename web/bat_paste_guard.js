/**
 * BAT — X11 middle-click paste guard, for the whole pack.
 *
 * The bug
 * -------
 * On Linux/X11 the middle mouse button pastes the PRIMARY selection, and the
 * browser delivers that as a `paste` event on the DOCUMENT. ComfyUI listens for
 * exactly that (`usePaste.ts`: `useEventListener(document, 'paste', ...)`) and
 * answers by pasting whatever nodes are on the clipboard. So a middle-click
 * anywhere inside a node's DOM widget drops a copy of your clipboard nodes onto
 * the graph — most visibly when middle-dragging to pan a preview, which is a
 * deliberate gesture, but an accidental middle-click does it too.
 *
 * Why DOM widgets specifically: litegraph's own `processMouseDown` calls
 * `preventDefault()`, which suppresses the native paste over the graph canvas.
 * A pointerdown that lands on a DOM widget never reaches that handler, so every
 * on-node HTML editor is a hole in an otherwise-covered surface.
 *
 * Why it cannot be fixed locally
 * ------------------------------
 * `preventDefault()` on our own `pointerdown` / `auxclick` does not stop it. The
 * paste is dispatched by the selection machinery rather than by our event, and
 * it arrives at the `document` rather than at our element — so an element-level
 * listener never sees it either. The only place to intercept it is the document,
 * in CAPTURE phase, which is what beats ComfyUI's bubble-phase handler.
 *
 * Three things keep this from being a blunt instrument
 * ----------------------------------------------------
 * • **Armed, not permanent.** The swallow only fires within GUARD_MS of a middle
 *   press or release inside a marked widget. Suppressing `paste` unconditionally
 *   would break Ctrl+V across the whole application — far worse than the bug.
 * • **Opt-in per widget.** Only elements marked with `markBatWidget()` arm it.
 *   That is deliberate rather than lazy: in a text field, middle-click paste is
 *   a real X11 feature someone may well want, so `bat_filename_prefix`'s editor
 *   is intentionally NOT marked. Canvases have no such use for it.
 * • **One listener, ever.** Installed once per page, not once per node. A
 *   per-instance document listener is the accumulating leak `bat_lifecycle.js`
 *   exists to prevent, and the editors that predate any teardown would never
 *   have released it.
 *
 * Coverage comes from `bat_node_layout.js`, which marks every element that goes
 * through `addBatDOMWidget()` — so every current and future BAT DOM widget is
 * covered without touching its file. The two editors that call `addDOMWidget`
 * directly (`bat_points_editor`, and `bat_filename_prefix` which is excluded on
 * purpose) are handled at their own call sites.
 */

/** How long a middle-button interaction keeps the guard armed. */
const GUARD_MS = 400;

/** Marker attribute. Not a class: nothing styles it, and the BAT packs share a
 *  CSS namespace, so a class here would be an invitation to collide. */
export const BAT_WIDGET_ATTR = "data-bat-widget";

let _until = 0;
let _installed = false;

function _swallowPaste(e) {
    if (performance.now() > _until) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
}

/** Suppress a document `paste` for the next GUARD_MS. */
export function armBatPasteGuard() {
    _until = performance.now() + GUARD_MS;
}

function _onMiddle(e) {
    if (e.button !== 1) return;
    const t = e.target;
    if (!t || typeof t.closest !== "function") return;
    if (!t.closest(`[${BAT_WIDGET_ATTR}]`)) return;
    armBatPasteGuard();
}

/**
 * Install the guard. Idempotent — call it from anywhere, as often as you like.
 *
 * Both pointerdown AND pointerup arm it, because the paste is delivered on
 * release: a middle-drag that lasts longer than GUARD_MS would otherwise have
 * expired by the time it lets go.
 */
export function installBatPasteGuard() {
    if (_installed) return;
    _installed = true;
    try {
        document.addEventListener("paste", _swallowPaste, true);
        document.addEventListener("pointerdown", _onMiddle, true);
        document.addEventListener("pointerup", _onMiddle, true);
    } catch (e) {
        console.error("[BAT.pasteGuard] could not install:", e);
        _installed = false;
    }
}

/** Mark an element (and its subtree) as one where a middle-click must not paste. */
export function markBatWidget(el) {
    try { el?.setAttribute?.(BAT_WIDGET_ATTR, ""); } catch (_) {}
    installBatPasteGuard();
    return el;
}

/** Test seam: is the guard currently armed? */
export function _batPasteGuardArmed() {
    return performance.now() <= _until;
}
