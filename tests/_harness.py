"""Shared bits for the quickjs-driven verification scripts.

Why this exists
---------------
Every harness in here evaluates a `web/*.js` extension with its ComfyUI
imports replaced by hand-written stubs. That works until an editor picks up
another shared helper — and then the harness fails with something that looks
nothing like the cause. On 2026-09-04 a pack-wide change added
`batReplayLastExecution` / `batPreviewWillReplay` to every preview editor, and
four harnesses went red at once:

  • verify_advanced_blend / verify_layered_images — "no onNodeCreated", because
    an undefined identifier threw inside `beforeRegisterNodeDef` before it
    could install its hooks.
  • verify_video_combine_migration — reported the UNMIGRATED widget values,
    because the same throw landed before its `onConfigure` was installed. That
    reads exactly like a live retro-compat bug ("filename_prefix got 0") and
    cost an hour of triage to prove it wasn't one.
  • verify_exposure_bracket — a parse error, because the new import spanned
    several lines and its strip regex was line-anchored.

None of them was a product defect, and a suite that cries wolf is a suite that
gets switched off. So the stubs are now derived from the extension's own import
statements instead of listed by hand: a helper that only has to *exist* gets one
for free, and only the ones whose behaviour matters stay curated.
"""

import re

_JS_KEYWORDS = {"import", "from", "as"}


def imported_names(*sources):
    """Every identifier a set of ES modules imports inside `{...}`."""
    names = set()
    for src in sources:
        for block in re.findall(r"^import\s*{[^}]*}", src, flags=re.M | re.S):
            names.update(n for n in re.findall(r"[A-Za-z_$][\w$]*", block)
                         if n not in _JS_KEYWORDS)
    return names


def auto_stub_js(*sources):
    """No-op JS declarations for everything `sources` imports.

    Emit this BEFORE the harness's curated stubs, never after: duplicate
    function declarations in one script resolve to the last one, so anything
    whose behaviour actually matters (`batTrack` handing back a tracker,
    `app.registerExtension` capturing the extension) has to come second to win.

    A no-op returning `undefined` is also the right answer for the predicate
    case: `batPreviewWillReplay()` falsy is precisely "a fresh page load with
    nothing to replay".
    """
    return "".join(f"function {n}() {{}}\n" for n in sorted(imported_names(*sources)))


def strip_modules(src):
    """Turn an ES module into a plain script quickjs can evaluate.

    `re.S` is load-bearing — an import spanning several lines is invisible to a
    line-anchored pattern, and the survivor reaches quickjs as
    `SyntaxError: expecting '('`.
    """
    src = re.sub(r"^import .*?;\s*$", "", src, flags=re.M | re.S)
    src = re.sub(r"^export ", "", src, flags=re.M)
    return src
