#!/usr/bin/env python
"""
Prove that maximising an on-node editor gives it back unharmed.

    $ ../../../env/bin/python tests/verify_fullscreen.py

Why this needs a test at all
----------------------------
`bat_fullscreen.js` does the one thing a ComfyUI extension is not supposed to
do: it takes a DOM widget's element out of the slot the frontend put it in.
Everything that can go wrong with that is invisible in a screenshot and shows up
minutes later, in another node, as "my roto editor went blank".

Three failure modes are worth a harness:

1. **The frontend steals the element back.** `DomWidget.vue` re-appends
   `widget.element` into its own wrapper on every visibility change, and
   visibility flips whenever the node scrolls in or out of view. The defence is
   `widget.hidden = true`, which makes `isVisible()` false and short-circuits
   the re-mount. `test_frontend_cannot_steal_element` reproduces the frontend's
   own `mountElementIfVisible` verbatim and calls it while fullscreen.

2. **The editor does not come back the same.** The root has to return to the
   exact slot it left — same parent, same position among siblings — carrying the
   exact inline style string it had, because that string holds each editor's
   own cssText *and* the `--comfy-widget-*` custom properties addBatDOMWidget
   publishes. `test_exit_restores_everything` compares it byte for byte.

3. **A re-measure fights the overlay.** `addBatDOMWidget`'s publishVars() writes
   the node's design height back onto the root, and `refreshBatLayout()` calls
   it with force=true — so any editor that re-measures while maximised (Layered
   Images does, whenever the layer stack changes) would stamp a 520px floor onto
   an element that is supposed to fill the screen. That interlock is a guard
   clause in bat_node_layout.js, tested here against the REAL addBatDOMWidget
   rather than a copy of it.

Plus the two behaviours an artist would notice immediately: Escape must not be
swallowed by the overlay when the editor wanted it (roto's Esc deselects), and
deleting a node — which is also what every Ctrl+Z does, via LGraph.clear() —
must not leave an overlay on screen holding a dead editor.

The DOM stub
------------
Richer than the one in verify_bypass_switch.py, because parentage is the thing
under test: it tracks parentNode / nextSibling / isConnected and implements
insertBefore, and its `style` serialises to and from a cssText string so the
save-and-restore round trip is a real round trip rather than an object identity
check.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import auto_stub_js, strip_modules

import quickjs

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), "web")


def read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as fh:
        return fh.read()


# ── the stubbed browser ──────────────────────────────────────────────────
DOM = r"""
var __idSeq = 0;

function stubStyle() {
    var props = {};
    var s = {
        _props: props,
        setProperty: function (k, v) { props[k] = String(v); },
        removeProperty: function (k) { delete props[k]; },
        getPropertyValue: function (k) { return props[k] || ""; },
        _text: function () {
            var out = [];
            for (var k in props) out.push(k + ":" + props[k]);
            return out.join("; ");
        },
        _load: function (text) {
            for (var k in props) delete props[k];
            var parts = String(text || "").split(";");
            for (var i = 0; i < parts.length; i++) {
                var p = parts[i].trim();
                if (!p) continue;
                var c = p.indexOf(":");
                if (c === -1) continue;
                props[p.slice(0, c).trim()] = p.slice(c + 1).trim();
            }
        },
    };
    // The camelCase fields the code assigns directly, mapped onto the same
    // backing store the cssText round trip reads — otherwise `style.minHeight =
    // x` and `getAttribute("style")` would disagree and the restore test would
    // pass for the wrong reason.
    var alias = { position: "position", minHeight: "min-height",
                  maxHeight: "max-height", width: "width", height: "height",
                  display: "display", cssText: null };
    Object.keys(alias).forEach(function (js) {
        if (js === "cssText") return;
        Object.defineProperty(s, js, {
            get: function () { return props[alias[js]] || ""; },
            set: function (v) { props[alias[js]] = String(v); },
            configurable: true, enumerable: false,
        });
    });
    Object.defineProperty(s, "cssText", {
        get: function () { return s._text(); },
        set: function (v) { s._load(v); },
        configurable: true, enumerable: false,
    });
    return s;
}

function stubEl(tag) {
    var e = {
        tagName: (tag || "div").toUpperCase(),
        _id: ++__idSeq,
        children: [], parentNode: null,
        dataset: {}, attrs: {},
        style: stubStyle(),
        tabIndex: -1, textContent: "", title: "", type: "",
        className: "",
        _listeners: {}, _removed: false,
        classList: { add: function () {}, remove: function () {} },

        get isConnected() {
            var n = this;
            while (n) { if (n === document.body || n === document.head) return true; n = n.parentNode; }
            return false;
        },
        get nextSibling() {
            var p = this.parentNode;
            if (!p) return null;
            var i = p.children.indexOf(this);
            return (i === -1 || i + 1 >= p.children.length) ? null : p.children[i + 1];
        },

        appendChild: function (c) {
            if (c.parentNode) c.parentNode._detach(c);
            c.parentNode = this; this.children.push(c); return c;
        },
        // Variadic append(), which the pack uses as freely as appendChild.
        append: function () {
            for (var i = 0; i < arguments.length; i++) this.appendChild(arguments[i]);
        },
        insertBefore: function (c, ref) {
            if (c.parentNode) c.parentNode._detach(c);
            c.parentNode = this;
            var i = ref ? this.children.indexOf(ref) : -1;
            if (i === -1) this.children.push(c); else this.children.splice(i, 0, c);
            return c;
        },
        _detach: function (c) {
            var i = this.children.indexOf(c);
            if (i !== -1) this.children.splice(i, 1);
            c.parentNode = null;
        },
        remove: function () {
            if (this.parentNode) this.parentNode._detach(this);
            this._removed = true;
        },
        contains: function (other) {
            var n = other;
            while (n) { if (n === this) return true; n = n.parentNode; }
            return false;
        },

        setAttribute: function (k, v) {
            this.attrs[k] = String(v);
            if (k === "style") this.style._load(v);
        },
        getAttribute: function (k) {
            if (k === "style") {
                var t = this.style._text();
                return t === "" ? null : t;
            }
            return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null;
        },
        removeAttribute: function (k) {
            delete this.attrs[k];
            if (k === "style") this.style._load("");
        },

        // Only the one selector bat_fullscreen.js uses.
        querySelector: function (sel) {
            if (sel !== "[data-bat-fs-mount]") throw new Error("unsupported selector " + sel);
            var stack = this.children.slice();
            while (stack.length) {
                var n = stack.shift();
                if (n.dataset && n.dataset.batFsMount) return n;
                stack = stack.concat(n.children || []);
            }
            return null;
        },

        addEventListener: function (t, fn) { (this._listeners[t] = this._listeners[t] || []).push(fn); },
        removeEventListener: function (t, fn) {
            var l = this._listeners[t] || [];
            var i = l.indexOf(fn); if (i !== -1) l.splice(i, 1);
        },
        focus: function () { globalThis.__focused = this; },
        blur: function () {},
        _fire: function (t, ev) {
            var l = (this._listeners[t] || []).slice();
            for (var i = 0; i < l.length; i++) l[i](ev || globalThis.mkEvent());
        },
    };
    return e;
}

globalThis.mkEvent = function (over) {
    var ev = { defaultPrevented: false,
               preventDefault: function () { this.defaultPrevented = true; },
               stopPropagation: function () {},
               stopImmediatePropagation: function () {} };
    for (var k in (over || {})) ev[k] = over[k];
    return ev;
};

var document = {
    _byId: {},
    createElement: function (t) { return stubEl(t); },
    getElementById: function (id) { return document._byId[id] || null; },
    head: stubEl("head"), body: stubEl("body"),
};
// ensureStyles() registers the <style> by id; model that so the "inject once"
// claim is actually exercised.
(function () {
    var realAppend = document.head.appendChild;
    document.head.appendChild = function (c) {
        if (c.id) document._byId[c.id] = c;
        return realAppend.call(this, c);
    };
})();

globalThis.__winKeys = [];
var window = {
    addEventListener: function (t, fn) { if (t === "keydown") globalThis.__winKeys.push(fn); },
    removeEventListener: function (t, fn) {
        var i = globalThis.__winKeys.indexOf(fn);
        if (i !== -1) globalThis.__winKeys.splice(i, 1);
    },
};
/** Deliver a keydown the way the browser would, after the editor has seen it. */
globalThis.pressKey = function (key, alreadyHandled) {
    var ev = globalThis.mkEvent({ key: key });
    if (alreadyHandled) ev.preventDefault();      // the editor consumed it
    var l = globalThis.__winKeys.slice();
    for (var i = 0; i < l.length; i++) l[i](ev);
    return ev;
};

function getComputedStyle(el) {
    return {
        position: el.style.position || "static",
        getPropertyValue: function (k) { return el.style.getPropertyValue(k); },
    };
}

function ResizeObserver(fn) {
    this._fn = fn;
    this.observe = function () {}; this.disconnect = function () {};
}
"""

# ── stubs for the pack's own imports (these WIN over auto_stub) ──────────
PACK_STUBS = r"""
var app = { extensionManager: null, ui: null };

globalThis.__disposers = [];
function batTrack(node) {
    return {
        dispose: function (fn) { globalThis.__disposers.push([node, fn]); return fn; },
        interval: function (i) { return i; }, timeout: function (t) { return t; },
        observer: function (o) { return o; }, listener: function () {},
    };
}
function isNodeAlive(node) { return !!node && !node._dead; }
/** Simulate LGraph.clear() / node deletion firing the teardown. */
globalThis.removeNode = function (node) {
    node._dead = true;
    globalThis.__disposers.forEach(function (d) { if (d[0] === node) d[1](); });
};

function markBatWidget() {}

globalThis.makeNode = function (title) {
    var widgets = [];
    return {
        title: title || "🦇 Test",
        size: [400, 400],
        widgets: widgets,
        graph: { incrementVersion: function () { globalThis.__versions++; } },
        setDirtyCanvas: function () {},
        setSize: function (s) { this.size = s; },
        computeSize: function () { return [400, 400]; },
        isWidgetVisible: function () { return true; },
        addDOMWidget: function (name, type, element, options) {
            var w = { name: name, type: type, element: element, options: options,
                      hidden: false, computedDisabled: false, node: this,
                      isVisible: function () {
                          return !this.hidden && !this.computedDisabled
                                 && this.node.isWidgetVisible(this);
                      } };
            widgets.push(w);
            return w;
        },
    };
};
globalThis.__versions = 0;

/**
 * The frontend's own re-mount, copied from DomWidget.vue (1.49.6). `wrapper`
 * stands in for `widgetElement.value`. Returns true if it moved the element.
 */
globalThis.frontendRemount = function (widget, wrapper) {
    var visible = widget.isVisible();          // DomWidgets.vue's gate
    if (!(visible && widget.element && wrapper)) return false;
    if (wrapper.contains(widget.element)) return false;
    wrapper.appendChild(widget.element);
    return true;
};
"""


def make_ctx():
    """One quickjs context with the DOM, the stubs and the two real modules."""
    layout = read("bat_node_layout.js")
    fullscreen = read("bat_fullscreen.js")

    ctx = quickjs.Context()
    ctx.eval(DOM)
    # auto-stubs FIRST so anything the modules import merely exists; the curated
    # stubs and then the real modules redeclare over the top and win.
    ctx.eval(auto_stub_js(layout, fullscreen))
    ctx.eval(PACK_STUBS)
    ctx.eval(strip_modules(layout))
    ctx.eval(strip_modules(fullscreen))

    # A node whose editor root carries a picture area, wired exactly as the six
    # editors wire theirs.
    ctx.eval(r"""
    globalThis.build = function (title) {
        var node = makeNode(title);
        var root = document.createElement("div");
        root.style.cssText = "position:relative; display:flex; min-height:540px; background:#0a0a0a";
        var picture = document.createElement("div");
        picture.style.cssText = "position:relative; flex:1";
        picture.dataset.batFsMount = "1";
        root.appendChild(picture);

        var widget = addBatDOMWidget(node, "test_editor", "test_editor", root, {
            minWidth: 640, height: 540, growable: true,
        });
        var handle = addBatFullscreen(node, widget, root);

        // The frontend's wrapper, and its first mount.
        var wrapper = document.createElement("div");
        document.body.appendChild(wrapper);
        wrapper.appendChild(root);

        return { node: node, root: root, picture: picture,
                 widget: widget, handle: handle, wrapper: wrapper };
    };

    // Video Combine's shape: the player already owns a fullscreen button in
    // its transport bar, so the helper must add none and hand back a toggle
    // the caller drives.
    globalThis.buildNoButton = function () {
        var node = makeNode("\ud83e\udd87 Video Combine");
        var root = document.createElement("div");
        root.style.cssText = "position:relative; display:flex; min-height:240px";
        var picture = document.createElement("div");
        picture.dataset.batFsMount = "1";
        root.appendChild(picture);
        var widget = addBatDOMWidget(node, "bat_video_player", "bat_video_player", root, {
            minWidth: 420, height: 300, growable: true,
        });
        globalThis.__ownLabel = "Fullscreen";
        var handle = addBatFullscreen(node, widget, root, {
            button: false,
            onEnter: function () { globalThis.__ownLabel = "Exit"; },
            onExit:  function () { globalThis.__ownLabel = "Fullscreen"; },
        });
        var wrapper = document.createElement("div");
        document.body.appendChild(wrapper);
        wrapper.appendChild(root);
        return { node: node, root: root, picture: picture,
                 widget: widget, handle: handle, wrapper: wrapper };
    };
    """)
    return ctx


FAILURES = []


def check(name, ctx, expr, want):
    got = ctx.eval(expr)
    if got != want:
        FAILURES.append(f"{name}: {expr}\n      expected {want!r}, got {got!r}")
        print(f"  FAIL {name}: expected {want!r}, got {got!r}")
    else:
        print(f"  ok   {name}")


def test_button_mounts_on_picture_area():
    print("\nbutton placement")
    ctx = make_ctx()
    ctx.eval("var t = build();")
    check("button is on the picture area, not the root", ctx,
          "t.picture.children.length", 1)
    check("button carries a glyph AND a label", ctx,
          "t.picture.children[0].children.map(function (c) { return c.textContent; }).join('|')",
          "⛶|Fullscreen")
    check("root has no button of its own", ctx,
          "t.root.children.filter(function (c) { return c.className === 'bat-fs-btn'; }).length", 0)
    # A second call must not add a second button.
    ctx.eval("addBatFullscreen(t.node, t.widget, t.root);")
    check("re-wiring adds no second button", ctx, "t.picture.children.length", 1)
    check("stylesheet injected once", ctx,
          "document.head.children.filter(function (c) { return c.id === 'bat-fullscreen-style'; }).length", 1)


def test_no_button_mode():
    """Video Combine supplies its own control, so the helper must add none."""
    print("\nan editor that owns its fullscreen control")
    ctx = make_ctx()
    ctx.eval("var t = buildNoButton();")
    check("no button is added anywhere", ctx,
          "t.picture.children.length + t.root.children.filter("
          "function (c) { return c.className === 'bat-fs-btn'; }).length", 0)
    check("but a toggle is returned", ctx, "typeof t.handle.toggle", "function")
    ctx.eval("t.handle.toggle();")
    check("toggle maximises", ctx, "t.widget.hidden", True)
    check("the caller's own label was updated", ctx, "__ownLabel", "Exit")
    ctx.eval("t.handle.toggle();")
    check("toggle restores", ctx, "t.widget.hidden", False)
    check("and the label went back", ctx, "__ownLabel", "Fullscreen")
    check("the editor is home", ctx, "t.root.parentNode === t.wrapper", True)


def test_video_combine_is_wired():
    print("\nVideo Combine uses the overlay, not the browser's")
    src = read("bat_video_combine.js")
    problems = []
    if 'from "./bat_fullscreen.js"' not in src:
        problems.append("no import")
    if "addBatFullscreen(this, playerWidget, el" not in src:
        problems.append("not called")
    if "button: false" not in src:
        problems.append("would add a second button")
    # The whole point: the native path must be gone.
    if "requestFullscreen?.()" in src or "document.exitFullscreen()" in src:
        problems.append("still calls native fullscreen")
    if "if (state.loopIn == null && state.loopOut == null)" not in src:
        problems.append("Escape is still swallowed unconditionally")
    if problems:
        FAILURES.append(f"bat_video_combine.js: {', '.join(problems)}")
        print(f"  FAIL bat_video_combine.js: {', '.join(problems)}")
    else:
        print("  ok   bat_video_combine.js")


def test_enter_moves_the_editor():
    print("\nentering fullscreen")
    ctx = make_ctx()
    ctx.eval("var t = build(); t.handle.enter();")
    check("overlay is on document.body", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 1)
    check("root left the frontend's wrapper", ctx, "t.wrapper.children.length", 0)
    check("root is inside the overlay body", ctx,
          "(function () { var o = document.body.children.filter(function (c) "
          "{ return c.className === 'bat-fs-overlay'; })[0];"
          " return o.children[1].children[0] === t.root; })()", True)
    check("widget marked hidden", ctx, "t.widget.hidden", True)
    check("widget type NOT zeroed", ctx, "t.widget.type", "test_editor")
    check("root flagged for the layout guard", ctx, "t.root.dataset.batFullscreen", "1")
    check("design-height floor released", ctx, "t.root.style.getPropertyValue('min-height')", "0")
    check("min-height custom property stripped", ctx,
          "t.root.style.getPropertyValue('--comfy-widget-min-height')", "")
    check("max-height custom property stripped", ctx,
          "t.root.style.getPropertyValue('--comfy-widget-max-height')", "")
    check("graph version bumped", ctx, "__versions > 0", True)
    check("editor took focus", ctx, "__focused === t.root || __focused !== null", True)


def test_frontend_cannot_steal_element():
    print("\nthe frontend's re-mount is defeated")
    ctx = make_ctx()
    ctx.eval("var t = build();")
    # Baseline: with the widget visible, the frontend WOULD re-adopt a loose root.
    check("sanity — frontend re-mounts a loose element", ctx,
          "(function () { var spare = document.createElement('div');"
          " document.body.appendChild(spare); spare.appendChild(t.root);"
          " var moved = frontendRemount(t.widget, t.wrapper);"
          " return moved && t.wrapper.contains(t.root); })()", True)
    ctx.eval("t.handle.enter();")
    check("while fullscreen the re-mount is a no-op", ctx,
          "frontendRemount(t.widget, t.wrapper)", False)
    check("root is still in the overlay", ctx, "t.wrapper.children.length", 0)
    # ...even when the node scrolls out of view and back, which is the flip that
    # used to yank it.
    check("survives a visibility flip", ctx,
          "(function () { frontendRemount(t.widget, t.wrapper);"
          " frontendRemount(t.widget, t.wrapper);"
          " return t.wrapper.children.length; })()", 0)


def test_exit_restores_everything():
    print("\nexiting restores the editor exactly")
    ctx = make_ctx()
    ctx.eval(r"""
    var t = build();
    // Put siblings either side, so "same parent" is not enough to pass.
    var before = document.createElement('div');
    var after  = document.createElement('div');
    t.wrapper.insertBefore(before, t.root);
    t.wrapper.appendChild(after);
    globalThis.__styleBefore = t.root.getAttribute('style');
    globalThis.__indexBefore = t.wrapper.children.indexOf(t.root);
    t.handle.enter();
    t.handle.exit();
    """)
    check("root is back in its wrapper", ctx, "t.root.parentNode === t.wrapper", True)
    check("root is back at the same index", ctx,
          "t.wrapper.children.indexOf(t.root) === __indexBefore", True)
    check("inline style restored byte for byte", ctx,
          "t.root.getAttribute('style') === __styleBefore", True)
    check("fullscreen flag cleared", ctx, "t.root.dataset.batFullscreen === undefined", True)
    check("widget visible again", ctx, "t.widget.hidden", False)
    check("overlay gone from the document", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 0)
    check("escape listener removed", ctx, "__winKeys.length", 0)
    check("button label reset", ctx,
          "t.picture.children[0].children.map(function (c) { return c.textContent; }).join('|')",
          "⛶|Fullscreen")
    # And the frontend is back in charge.
    check("frontend re-mount works again", ctx,
          "(function () { t.wrapper._detach(t.root);"
          " return frontendRemount(t.widget, t.wrapper) && t.wrapper.contains(t.root); })()", True)


def test_layout_guard():
    print("\na re-measure cannot fight the overlay")
    ctx = make_ctx()
    ctx.eval("var t = build();")
    check("publishVars set a floor while docked", ctx,
          "t.root.style.getPropertyValue('--comfy-widget-min-height')", "540px")
    ctx.eval("t.handle.enter();")
    # This is what refreshBatLayout() does, and Layered Images calls it whenever
    # the layer stack changes. force=true is the caller that used to win.
    ctx.eval("t.widget._batPublishVars();")
    check("forced re-publish leaves min-height released", ctx,
          "t.root.style.getPropertyValue('min-height')", "0")
    check("forced re-publish does not restore the floor", ctx,
          "t.root.style.getPropertyValue('--comfy-widget-min-height')", "")
    ctx.eval("t.handle.exit();")
    check("floor is republished on the way out", ctx,
          "t.root.style.getPropertyValue('--comfy-widget-min-height')", "540px")


def test_escape():
    print("\nEscape belongs to the editor first")
    ctx = make_ctx()
    ctx.eval("var t = build(); t.handle.enter();")
    ctx.eval("pressKey('Escape', true);")     # the editor consumed it
    check("consumed Escape does not close the overlay", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 1)
    ctx.eval("pressKey('a', false);")
    check("an unrelated key does not close it", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 1)
    ctx.eval("pressKey('Escape', false);")    # the editor ignored it
    check("unconsumed Escape closes the overlay", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 0)
    check("and the editor is home", ctx, "t.root.parentNode === t.wrapper", True)


def test_one_at_a_time():
    print("\none editor maximised at a time")
    ctx = make_ctx()
    ctx.eval("var a = build('A'); var b = build('B'); a.handle.enter(); b.handle.enter();")
    check("exactly one overlay", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 1)
    check("the first editor went home", ctx, "a.root.parentNode === a.wrapper", True)
    check("the first widget is visible again", ctx, "a.widget.hidden", False)
    check("the second editor is maximised", ctx, "b.widget.hidden", True)
    check("isBatFullscreen tracks the right node", ctx,
          "isBatFullscreen(b.node) && !isBatFullscreen(a.node)", True)


def test_teardown():
    print("\ndeleting the node (and every Ctrl+Z) closes the overlay")
    ctx = make_ctx()
    ctx.eval("var t = build(); t.handle.enter(); removeNode(t.node);")
    check("overlay removed", ctx,
          "document.body.children.filter(function (c) { return c.className === 'bat-fs-overlay'; }).length", 0)
    check("escape listener removed", ctx, "__winKeys.length", 0)
    check("no crash restoring into a dead node", ctx, "t.node._dead", True)

    # The harsher case: the wrapper itself is gone, as after a graph reload.
    ctx2 = make_ctx()
    ctx2.eval("var t = build(); t.handle.enter(); t.wrapper.remove(); t.handle.exit();")
    check("root is detached rather than re-homed into a dead wrapper", ctx2,
          "t.root.parentNode === null", True)
    check("widget left visible so the frontend re-adopts it", ctx2, "t.widget.hidden", False)


def test_roto_escape_is_releasable():
    """The roto edit that makes 'Esc deselects, Esc again exits' possible."""
    print("\nroto's Escape releases when it has nothing to do")
    src = read("bat_roto.js")
    marker = "if (!mutated && !hadFocus) { handled = false; break; }"
    ok = marker in src
    print(f"  {'ok  ' if ok else 'FAIL'} roto reports a no-op Escape as unhandled")
    if not ok:
        FAILURES.append("bat_roto.js: a no-op Escape still calls preventDefault, "
                        "so it would trap the artist in fullscreen")


def test_all_six_are_wired():
    print("\nall six editors are wired")
    for name in ("bat_roto.js", "bat_animated_crop.js", "bat_animated_grade.js",
                 "bat_layered_images.js", "bat_hdr_tonal_composite.js",
                 "bat_rescale.js"):
        src = read(name)
        problems = []
        if 'from "./bat_fullscreen.js"' not in src:
            problems.append("no import")
        if "addBatFullscreen(this, editorWidget, el)" not in src:
            problems.append("not called")
        if 'dataset.batFsMount = "1"' not in src:
            problems.append("no picture-area marker")
        if problems:
            FAILURES.append(f"{name}: {', '.join(problems)}")
            print(f"  FAIL {name}: {', '.join(problems)}")
        else:
            print(f"  ok   {name}")


def main():
    print("BAT fullscreen verification")
    test_button_mounts_on_picture_area()
    test_no_button_mode()
    test_video_combine_is_wired()
    test_enter_moves_the_editor()
    test_frontend_cannot_steal_element()
    test_exit_restores_everything()
    test_layout_guard()
    test_escape()
    test_one_at_a_time()
    test_teardown()
    test_roto_escape_is_releasable()
    test_all_six_are_wired()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
