"""Bat_Framehold — drive the on-node scrubber headlessly and check what it does.

The parser has its own parity test; this one covers the layer above it — what
←/→, Enter, A, R, the input/output toggle and the empty-spec passthrough
actually do to the `frames` widget and the counters. There is no node binary on
these boxes and no DOM, so this stubs just enough of both (quickjs + a handful
of fake elements) to run web/bat_framehold.js as written.

    env/bin/python tools/test_framehold_ui.py
"""

import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)

# Enough of a DOM that buildEditor() can construct the editor and render() can
# write into it. Elements are plain bags with the handful of members the module
# touches; nothing here simulates layout, only identity and text.
DOM_STUB = r"""
var __els = [];
function _el(tag) {
    var e = {
        tagName: (tag || "div").toUpperCase(),
        style: { cssText: "", setProperty: function(){}, },
        children: [], textContent: "", value: "", _attrs: {},
        _handlers: {},
        appendChild: function(c){ this.children.push(c); return c; },
        append: function(){ for (var i=0;i<arguments.length;i++) this.children.push(arguments[i]); },
        addEventListener: function(t, f){ this._handlers[t] = f; },
        removeEventListener: function(){},
        setAttribute: function(k,v){ this._attrs[k] = v; },
        getAttribute: function(k){ return k === "src" ? (this.src === undefined ? null : this.src) : (this._attrs[k] === undefined ? null : this._attrs[k]); },
        removeAttribute: function(k){ if (k === "src") this.src = undefined; delete this._attrs[k]; },
        focus: function(){}, contains: function(){ return false; },
        setPointerCapture: function(){},
        getBoundingClientRect: function(){ return { left: 0, top: 0, width: 200, height: 20 }; },
        getContext: function(){ return { fillStyle:"", fillRect:function(){}, clearRect:function(){} }; },
        clientWidth: 200, clientHeight: 20,
    };
    __els.push(e);
    return e;
}
var document = { createElement: _el, activeElement: null };
var window = { devicePixelRatio: 1 };
var __store = {};
var localStorage = {
    getItem: function(k){ return __store[k] === undefined ? null : __store[k]; },
    setItem: function(k,v){ __store[k] = String(v); },
};
var __intervals = [];
function setInterval(f, ms){ __intervals.push(f); return __intervals.length; }
function clearInterval(){}
function setTimeout(f){ f(); return 0; }
function fetch(){ return { then: function(){ return { catch: function(){} }; } }; }
var Image = function(){ this.src = ""; };

var __ext = null;
var app = { registerExtension: function(e){ __ext = e; } };
var api = { apiURL: function(u){ return u; } };
var __widgets = [];
// The layout helper is exercised for real by the renderers, not here — this
// only records that a re-measure was asked for, and with what shrink flag.
var __layout = [];
function addBatDOMWidget(node, name, type, el, opts){
    __widgets.push(el);
    return { element: el, _opts: opts,
             height: function(){ return opts.height(); },
             cap: function(){ return opts.maxHeight(); } };
}
function clampNodeSize(){}
function disposeBatLayout(){}
function refreshBatLayout(node, w, opts){ __layout.push(opts && opts.shrink === true); }
function batTrack(){ return { interval: function(i){ return i; }, dispose: function(){} }; }
function isNodeAlive(){ return true; }
function batNodeCacheKey(a, p, n){ return p + "_" + n.id; }

// A stand-in ComfyUI node: one STRING widget, which is all this editor reads.
function mkNode(spec) {
    return { id: 7, widgets: [{ name: "frames", value: spec, callback: null }],
             properties: {}, setDirtyCanvas: function(){} };
}
function specOf(node){ return node.widgets[0].value; }
"""


def load(spec_default="0", frames=50):
    import quickjs

    src = open(os.path.join(PACK, "web", "bat_framehold.js"), encoding="utf-8").read()
    # Matches multi-line import blocks too, not just one-liners.
    src = re.sub(r"^import\s[\s\S]*?from\s+['\"][^'\"]+['\"];", "", src, flags=re.M)
    src = re.sub(r"^export ", "", src, flags=re.M)

    ctx = quickjs.Context()
    ctx.eval(DOM_STUB)
    ctx.eval(src)
    ctx.eval(f"""
        var node = mkNode({json.dumps(spec_default)});
        install(node);
        applyStrip(node, {{ token: "tok123", frames: {frames}, src_w: 1920, src_h: 1080 }});
        setCollapsed(node, false);
        function state() {{
            var st = node._batFH;
            return JSON.stringify({{
                spec: specOf(node), pos: st.pos, mode: st.mode,
                src: currentSource(node), track: trackLength(node),
                out: outputCount(node), counter: st.ui.counter.textContent,
                status: st.ui.status.textContent, badge: st.ui.badge.textContent,
                inSel: st.ui.inSel.textContent, img: st.ui.img.src || null,
                err: st.specError,
            }});
        }}
        function layout() {{
            var st = node._batFH;
            return JSON.stringify({{
                collapsed: st.collapsed,
                chev: st.ui.chev.textContent,
                bodyDisplay: st.ui.body.style.display,
                height: st.widget.height(), cap: st.widget.cap() ?? null,
                prop: node.properties["bat_framehold_expanded"] ?? null,
                cap2: 0,
                lastShrink: __layout.length ? __layout[__layout.length-1] : null,
            }});
        }}
        function key(k, shift) {{
            node._batFH.ui.root._handlers["keydown"]({{
                key: k, shiftKey: !!shift, target: {{ tagName: "DIV" }},
                preventDefault: function(){{}}, stopPropagation: function(){{}},
            }});
        }}
    """)
    return ctx


def state(ctx):
    return json.loads(ctx.eval("state()"))


def layout(ctx):
    return json.loads(ctx.eval("layout()"))


FAILS = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILS.append(label)
    print(f"{'ok ' if ok else 'FAIL'}  {label}: {got!r}" + ("" if ok else f"  expected {want!r}"))


def main():
    # ── collapse ────────────────────────────────────────────────────────
    # load() opens the panel, so build a fresh one to see the default.
    c = load("8", 50)
    c.eval("var n2 = mkNode('8'); install(n2);")
    c.eval("var _sv = node; node = n2;")
    check("a new node starts collapsed", layout(c)["collapsed"], True)
    check("...showing the closed chevron", layout(c)["chev"], "\u25b8")
    check("...with the body hidden", layout(c)["bodyDisplay"], "none")
    check("...advertising just the header height", layout(c)["height"], 26)
    check("...pinned so it cannot be dragged taller", layout(c)["cap"], 26)
    check("...and no property written, so the workflow is not dirtied",
          layout(c)["prop"], None)

    c.eval("node._batFH.ui.header._handlers['click']()")
    l = layout(c)
    check("clicking the header expands it", l["collapsed"], False)
    check("...to the full height", l["height"], 346)
    check("...uncapping it so it can be dragged taller", l["cap"], None)
    check("...showing the open chevron", l["chev"], "\u25be")
    check("...and recording the choice in the workflow", l["prop"], True)
    check("...asking the layout to grow, not shrink", l["lastShrink"], False)

    c.eval("node._batFH.ui.header._handlers['click']()")
    l = layout(c)
    check("clicking again collapses it", l["collapsed"], True)
    check("...asking the layout to shrink", l["lastShrink"], True)
    check("...and recording that too", l["prop"], False)

    # A collapsed node must not fetch frames or run the keyboard.
    c.eval("applyStrip(node, { token: 'tokC', frames: 50, src_w: 1920, src_h: 1080 })")
    check("collapsed, no frame is fetched", state(c)["img"], None)
    check("collapsed, the header still summarises the output",
          state(c)["status"],
          "out 1 frame · 0.04s @ 25fps   ·   in 50f · 2.00s")
    before = state(c)["src"]
    c.eval("key('ArrowRight')")
    check("collapsed, arrow keys are inert", state(c)["src"], before)
    c.eval("key('Enter')")
    check("collapsed, Enter opens the panel", layout(c)["collapsed"], False)
    check("...and only then does it fetch a frame", state(c)["img"] is not None, True)
    c.eval("node = _sv;")

    # ── stepping and the counters ───────────────────────────────────────
    ctx = load("8", 50)
    s = state(ctx)
    check("opens on the frame the spec holds", s["src"], 8)
    check("counter is 1-based over the batch", s["counter"], "9 / 50")
    check("badge reports source size and input duration",
          s["badge"], "1920×1080 · in 50f · 2.00s")
    check("playhead frame is flagged as selected", s["inSel"], "selected")
    check("image url carries the token, not the node id",
          s["img"], "/bat/framehold/frame?token=tok123&i=8")
    check("status counts the output and its duration",
          s["status"], "out 1 frame · 0.04s @ 25fps   ·   in 50f · 2.00s")

    ctx.eval("key('ArrowRight')")
    check("→ steps one frame", state(ctx)["src"], 9)
    check("stepping does not touch the spec", state(ctx)["spec"], "8")
    check("stepping off the selection is flagged", state(ctx)["inSel"], "not selected")
    ctx.eval("key('ArrowRight', true)")
    check("shift+→ steps ten", state(ctx)["src"], 19)
    ctx.eval("key('ArrowLeft', true)")
    ctx.eval("key('ArrowLeft')")
    check("shift+← and ← step back", state(ctx)["src"], 8)
    ctx.eval("key('End')")
    check("End goes to the last frame", state(ctx)["src"], 49)
    check("counter at the end", state(ctx)["counter"], "50 / 50")
    ctx.eval("key('Home')")
    check("Home goes to the first", state(ctx)["src"], 0)

    # ── setting the hold ────────────────────────────────────────────────
    ctx.eval("seek(node, 12); key('Enter')")
    check("Enter sets the spec to the frame you are on", state(ctx)["spec"], "12")
    ctx.eval("seek(node, 20); key('a')")
    check("A appends to the spec", state(ctx)["spec"], "12, 20")
    check("...and the output count follows", state(ctx)["out"], 2)
    ctx.eval("seek(node, 25); key('r')")
    check("R extends the last token to here", state(ctx)["spec"], "12, 20-25")
    check("...expanding the output to the range", state(ctx)["out"], 7)
    ctx.eval("seek(node, 22); key('r')")
    check("R again re-anchors rather than stacking", state(ctx)["spec"], "12, 20-22")

    # ── [ and ] walk the selection ──────────────────────────────────────
    ctx.eval("seek(node, 0); key(']')")
    check("] jumps to the next selected frame", state(ctx)["src"], 12)
    ctx.eval("key(']')")
    check("] again", state(ctx)["src"], 20)
    ctx.eval("key('[')")
    check("[ jumps back", state(ctx)["src"], 12)

    # ── output mode ─────────────────────────────────────────────────────
    ctx.eval("node._batFH.ui.modeChip.onclick()")
    s = state(ctx)
    check("output mode walks the emitted list", s["mode"], "output")
    check("...whose length is the output count", s["track"], 4)
    check("...keeping the frame under the playhead", s["src"], 12)
    check("...and counting over the output", s["counter"], "1 / 4")
    ctx.eval("key('ArrowRight')")
    check("stepping in output mode follows the spec order", state(ctx)["src"], 20)
    check("badge names the source frame in output mode",
          state(ctx)["badge"].endswith("· src 20"), True)

    # ── a re-run must not yank the playhead back ────────────────────────
    ctx.eval("node._batFH.ui.modeChip.onclick(); seek(node, 31);")
    check("scrubbed away from the selection", state(ctx)["src"], 31)
    ctx.eval("applyStrip(node, { token: 'tok456', frames: 50, src_w: 1920, src_h: 1080 })")
    check("a re-run keeps the playhead where the artist left it",
          state(ctx)["src"], 31)
    check("...and re-points the image at the new strip",
          state(ctx)["img"], "/bat/framehold/frame?token=tok456&i=31")

    # ── holds and reorders ──────────────────────────────────────────────
    ctx2 = load("5, 5, 5, 1", 50)
    s = state(ctx2)
    check("repeats are counted as held frames",
          s["status"], "out 4 frames · 0.16s @ 25fps  (2 held/repeated)   ·   in 50f · 2.00s")

    # ── empty spec is a passthrough, not an empty output ────────────────
    ctx3 = load("", 50)
    s = state(ctx3)
    check("empty spec reports the whole batch", s["out"], 50)
    check("...and says why", "empty spec" in s["status"], True)
    check("...and marks every frame as passing", s["inSel"], "all frames")

    # ── a spec the backend would refuse ─────────────────────────────────
    ctx4 = load("5-", 50)
    s = state(ctx4)
    check("a spec that would crash the render is flagged before you run, "
          "and the message says what IS accepted",
          s["status"],
          'spec error — cannot parse "5-" — expected an index (5, -1) '
          'or a range (0-10, -5--1)')

    # ── negative indices resolve against the real batch length ──────────
    ctx5 = load("-1", 50)
    check("-1 opens on the last frame", state(ctx5)["src"], 49)

    # ── no strip yet ────────────────────────────────────────────────────
    ctx6 = load("0", 50)
    ctx6.eval("node._batFH.token = ''; node._batFH.frames = 0; "
              "node._batFH.parsedFor = null; render(node);")
    s = state(ctx6)
    check("with no strip the counter is blank", s["counter"], "— / —")
    check("...and no image is requested", s["img"], None)
    check("...and the header says to run, not 'out 0 frames'",
          s["status"], "run once to load the scrubber")

    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
