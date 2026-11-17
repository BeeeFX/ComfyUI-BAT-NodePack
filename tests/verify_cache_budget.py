#!/usr/bin/env python
"""
Checks for batCacheSet() in web/bat_lifecycle.js — the localStorage budget the
editors' preview thumbnails are written through.

Why it matters: the thumbnails share the origin's ~5 MB localStorage with
ComfyUI's unsaved-workflow drafts, and the frontend answers a full quota by
evicting the user's oldest drafts. Once the caches became per-workflow they
stopped being bounded by the node-id range, so without a budget they would grow
until ComfyUI started throwing drafts away. Five claims:

1. Writes under the budget are all kept.
2. Past the budget the least-recently-WRITTEN entry goes first, and rewriting
   a key makes it the newest.
3. A write the browser refuses (quota) returns false and never throws.
4. The one-time sweep drops the orphaned path-scoped keys ("…_/_14") and
   Grade's id-only keys, and nothing else — other BAT settings, UUID-scoped
   caches and foreign keys survive.
5. The index survives a garbage value in its own slot.

    pip install quickjs
    python tests/verify_cache_budget.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from _harness import auto_stub_js, strip_modules  # noqa: E402

_failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        _failures.append(label)


# A Storage with a hard character quota, counted the way browsers do (key +
# value), so "the browser refused the write" can be exercised too.
STORAGE = r"""
function makeStorage(quota) {
    const m = new Map();
    const used = () => { let n = 0; for (const [k, v] of m) n += k.length + v.length; return n; };
    return {
        get length() { return m.size; },
        key(i) { return Array.from(m.keys())[i] ?? null; },
        getItem(k) { return m.has(k) ? m.get(k) : null; },
        setItem(k, v) {
            v = String(v);
            const before = m.has(k) ? k.length + m.get(k).length : 0;
            if (used() - before + k.length + v.length > quota) {
                const e = new Error("quota"); e.name = "QuotaExceededError"; throw e;
            }
            m.set(k, v);
        },
        removeItem(k) { m.delete(k); },
        _keys() { return Array.from(m.keys()).sort(); },
    };
}
var localStorage = makeStorage(10000000);
"""


def load(quickjs):
    src = open(os.path.join(PACK, "web", "bat_lifecycle.js"), encoding="utf-8").read()
    ctx = quickjs.Context()
    ctx.eval(auto_stub_js(src))
    ctx.eval(STORAGE)
    ctx.eval(strip_modules(src))
    return ctx


def fresh(quickjs, quota=10000000, seed=None):
    """A new module instance (so the once-per-page sweep runs again)."""
    ctx = load(quickjs)
    ctx.eval(f"localStorage = makeStorage({quota});")
    for k, v in (seed or {}).items():
        ctx.eval(f"localStorage.setItem({k!r}, {v!r});")
    return ctx


def keys(ctx):
    return set(ctx.eval("JSON.stringify(localStorage._keys())")[2:-2].split('","')) - {""}


def main():
    try:
        import quickjs
    except ImportError:
        print("skip  (pip install quickjs)")
        return

    budget = 1_500_000
    chunk = "x" * 400_000   # ~ a large JPEG preview as a data URL

    print("\nbudget")
    ctx = fresh(quickjs)
    for i in range(3):
        ctx.eval(f"batCacheSet('bat_roto_preview_wf_{i}', {chunk!r})")
    check("under budget: all three kept",
          {f"bat_roto_preview_wf_{i}" for i in range(3)} <= keys(ctx))
    ctx.eval(f"batCacheSet('bat_roto_preview_wf_0', {chunk!r})")      # rewrite → newest
    ctx.eval(f"batCacheSet('bat_roto_preview_wf_3', {chunk!r})")      # 4 x 400k > budget
    k = keys(ctx)
    check("past budget: the least-recently-written entry is evicted",
          "bat_roto_preview_wf_1" not in k, sorted(k))
    check("rewriting a key made it the newest (kept)", "bat_roto_preview_wf_0" in k)
    check("the new entry is stored", "bat_roto_preview_wf_3" in k)
    total = ctx.eval("JSON.parse(localStorage.getItem('bat_cache_lru')).reduce((s, e) => s + e.n, 0)")
    check("index total stays within the budget", total <= budget, total)

    print("\nrefused writes")
    ctx = fresh(quickjs, quota=50_000)
    ok = ctx.eval(f"batCacheSet('bat_vc_preview_wf_1', {'y' * 80_000!r})")
    check("a write the browser refuses returns false", ok is False, ok)
    check("...and leaves nothing half-written", "bat_vc_preview_wf_1" not in keys(ctx))

    print("\nlegacy sweep")
    seed = {
        "bat_roto_preview_/_14": "old",            # page-path scoped: orphaned
        "bat_vc_view_/comfy/_3": "old",            # subpath install: orphaned
        "bat_framehold_tok_/_7": "old",
        "bat_grade_preview_14": "old",             # id-only Grade key: orphaned
        "bat_roto_preview_1f0c-uuid_14": "keep",   # current format
        "bat_grade_preview_1f0c-uuid_14": "keep",
        "bat_roto_view": "keep",                   # a plain BAT setting
        "bat-frame-cell": "keep",
        "Comfy.Workflow.Draft": "keep",            # not ours
    }
    ctx = fresh(quickjs, seed=seed)
    ctx.eval("batCacheSet('bat_animcrop_preview_wf_1', 'z')")
    k = keys(ctx)
    for key, fate in seed.items():
        if fate == "old":
            check(f"swept {key}", key not in k)
        else:
            check(f"kept {key}", key in k)

    print("\ncorrupt index")
    ctx = fresh(quickjs, seed={"bat_cache_lru": "{not json"})
    ok = ctx.eval("batCacheSet('bat_sec_plate_wf_1', 'p')")
    check("a garbage index is replaced, not fatal", ok is True, ok)

    print()
    if _failures:
        print(f"FAILURES: {', '.join(_failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
