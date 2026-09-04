"""
Bat_Framehold — select specific frames from an IMAGE batch by index.

The `frames` string is a comma-separated list of tokens; each token is
either a single index or an inclusive range:

    "0"        → frame index 0 (the first frame)
    "0-10"     → frames 0,1,2,…,10 (inclusive on both ends)
    "0, 5"     → frames 0 and 5
    "0-2, 7, 10-12"  → 0,1,2,7,10,11,12

Indices are 0-based. Negative indices count from the end (-1 = last),
matching Python slicing intuition. A descending range like "10-5" is
read as 10,9,…,5 so you can reverse a run. Out-of-range indices are
clamped to the valid batch range (and a warning is logged) rather than
crashing a render mid-graph. Order and duplicates are preserved exactly
as written — "5, 0, 5" yields three frames in that order — so this
doubles as a simple reorder/hold tool.

Selection is a single ``index_select`` on the batch dimension: one
vectorised gather, no per-frame Python copy.

The on-node preview
-------------------
Typing an index blind and re-running to see which frame you got is the
slow way to pick a hold. So on every execution the node parks a small
uint8 thumbnail strip of its INPUT batch in a process-local LRU and the
front-end (``web/bat_framehold.js``) scrubs that strip with the arrow
keys — no re-run per step. The viewer is collapsed by default and makes
no requests until it is opened; the summary line it collapses to is fed
by the `ui` payload below, not by the strip.

Three deliberate choices:

* **Server-side strip, fetched per frame, rather than base64 in the `ui`
  payload.** The sibling editors (Bat_AnimatedGrade / Bat_Roto) inline up
  to 240 *strided* JPEGs in their execution result, which is fine when the
  strip is only being scrubbed. This node's whole job is naming an exact
  index, so a strided strip would be actively wrong — arrow-stepping would
  skip frames the spec can still address. Every frame therefore has to be
  reachable, and shipping every frame of a 600-frame batch through the UI
  websocket on each run is not. The strip stays in RAM here and the browser
  pulls (and caches) only the frames the artist actually looks at.
* **Bounded, and degrading rather than refusing.** Thumbnail size is chosen
  from the batch length so one entry can't park half a gigabyte in RAM; the
  LRU then caps the total across nodes. A long batch gets smaller
  thumbnails, never no preview.
* **Keyed by a per-execution token, not by the node id.** The id is the
  obvious key and it is the wrong one twice over: it is only unique within
  one graph, so after a reload node 12 of another workflow would be served
  somebody else's shot; and the id this node is handed at execution time is
  the *flattened prompt* id, which does not match `node.id` in the browser
  once the node sits inside a subgraph. A uuid handed to the front-end in
  the execution payload has neither problem. The id is still recorded, but
  only so a re-run drops that node's previous strip instead of stacking a
  second one.

The preview is display-only: it is clamped to [0,1] and JPEG'd, while the
node itself passes the batch through untouched — superwhites and all.
"""

import io
import logging
import re
import uuid
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import server

logger = logging.getLogger("[Bat_Framehold]")

# Long edge of a preview thumbnail, before the per-entry memory budget gets a
# say. 480 is readable on a node you have deliberately dragged wide, and small
# enough that a 120-frame batch (the WAN ballpark) fits in ~45 MB.
PREVIEW_MAX_DIM = 480
# Per-entry and total ceilings for the strip cache. The total is what stops a
# graph with six Frameholds in it pinning a gigabyte; the per-entry one is what
# makes a single very long batch degrade its thumbnail size rather than eat the
# whole budget on its own.
_ENTRY_BYTES_MAX = 64 * 1024 * 1024
_CACHE_BYTES_MAX = 192 * 1024 * 1024

# token -> {node_id, thumbs (N,h,w,3 uint8), n, src_w, src_h, bytes}
_PREVIEW: "OrderedDict[str, dict]" = OrderedDict()


def _parse_frames(spec: str, n: int) -> list:
    """Parse the frame-spec string into an explicit ordered index list,
    resolving negatives and clamping to [0, n-1]."""
    out = []
    if not spec or not spec.strip():
        return out

    def _resolve(v: int) -> int:
        # Keep the value as written for the log message — `v` gets reassigned by
        # the negative-index conversion, so logging it after that printed the
        # resolved number and made the warning confusing ("index -3 below
        # range" when the artist typed -13).
        requested = v
        if v < 0:
            v = n + v
        if v < 0:
            logger.warning("Bat_Framehold: index %d below range; clamped to 0.", requested)
            return 0
        if v > n - 1:
            logger.warning("Bat_Framehold: index %d past last frame %d; clamped.",
                           requested, n - 1)
            return n - 1
        return v

    # Explicit grammar instead of hunting for "the first '-' after position 0",
    # which mis-parsed several real inputs: "5-" silently produced an empty
    # bound (skipped with only a log line), and "--1" split into an empty low
    # bound. A typo in a shot submission then rendered the wrong frames with no
    # hard error. Ranges and singles are now matched exactly, and anything that
    # doesn't match raises so the mistake surfaces immediately.
    #
    # NOTE: web/bat_framehold.js mirrors this grammar so the on-node preview can
    # highlight the selection and flag a bad spec before you run. Change one,
    # change the other — tools/test_framehold_parity.py checks they agree.
    range_re = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")
    single_re = re.compile(r"^-?\d+$")

    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        m = range_re.match(tok)
        if m:
            lo, hi = _resolve(int(m.group(1))), _resolve(int(m.group(2)))
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        elif single_re.match(tok):
            out.append(_resolve(int(tok)))
        else:
            raise ValueError(
                f"Bat_Framehold: cannot parse frame spec token {tok!r}. "
                f"Expected an index (e.g. 5, -1) or a range (e.g. 0-10, -5--1)."
            )
    return out


# ─── Preview strip ────────────────────────────────────────────────────────────


def _thumb_dims(h: int, w: int, n: int):
    """Thumbnail height/width for an n-frame batch of h×w frames.

    Starts at PREVIEW_MAX_DIM on the long edge and shrinks until the whole
    strip fits the per-entry budget, so batch length trades against thumbnail
    size instead of against having a preview at all. Floored at 96px: below
    that you can no longer tell two frames apart, which defeats the point.
    """
    longest = max(1, max(h, w))
    dim = min(PREVIEW_MAX_DIM, longest)
    th, tw = h, w
    for _ in range(12):
        s = dim / longest
        th, tw = max(1, round(h * s)), max(1, round(w * s))
        if n * th * tw * 3 <= _ENTRY_BYTES_MAX or dim <= 96:
            break
        dim = max(96, int(dim * 0.7))
    return th, tw


def _thumbnails(images: torch.Tensor, th: int, tw: int, chunk: int = 16) -> np.ndarray:
    """Decimate an (N,H,W,C) float batch to an (N,th,tw,3) uint8 strip.

    Two stages: a plain strided slice down to roughly 1.5–2× the target, then a
    box average the rest of the way. Neither half is arbitrary.

    * The strided slice is what keeps this off the render's critical path. A
      straight `interpolate()` reads every pixel of every frame — 1.8 GB of
      traffic on a 300-frame 1080p batch — and `index_select` on the row axis,
      the obvious "gather only what you need", measured *slower* than the
      whole strided pipeline (1.1 s vs 0.5 s for that batch) because it
      gathers row by row. Basic slicing is the fast path here.
    * The box average is what stops the preview lying. Point-sampling a 1080p
      frame to 336px keeps one pixel in 32, so fine detail aliases into a
      shimmer that changes on every frame — exactly the artefact that makes
      you pick the wrong hold frame while stepping through a batch.

    Chunked so the intermediate float copy stays in the tens of MB whatever the
    batch length.
    """
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    out = np.empty((n, th, tw, 3), dtype=np.uint8)
    # Integer strides, so the prescale lands at or above 1.5× the target and
    # adaptive_avg_pool2d finishes the (non-integer) remainder.
    sy = max(1, h // max(1, int(th * 1.5)))
    sx = max(1, w // max(1, int(tw * 1.5)))

    for i in range(0, n, chunk):
        blk = images[i:i + chunk]
        if int(blk.shape[3]) >= 3:
            blk = blk[:, ::sy, ::sx, :3]
        else:
            # A single-channel batch reaching an IMAGE input is unusual but not
            # impossible (mask-shaped data). Grey it rather than dropping the
            # preview for the whole node.
            blk = blk[:, ::sy, ::sx, :1].expand(-1, -1, -1, 3)
        blk = blk.permute(0, 3, 1, 2).float().contiguous()
        blk = F.adaptive_avg_pool2d(blk, (th, tw))
        blk = blk.permute(0, 2, 3, 1).clamp(0, 1).mul_(255.0).add_(0.5)
        out[i:i + blk.shape[0]] = blk.to(torch.uint8).cpu().numpy()
    return out


def _cache_put(token: str, entry: dict) -> None:
    """Store a strip under `token`, dropping this node's previous one first and
    then evicting oldest-first until the total is back inside the budget."""
    node_id = entry["node_id"]
    for old_token, old in list(_PREVIEW.items()):
        if old["node_id"] == node_id:
            _PREVIEW.pop(old_token, None)
    _PREVIEW[token] = entry
    total = sum(e["bytes"] for e in _PREVIEW.values())
    while total > _CACHE_BYTES_MAX and len(_PREVIEW) > 1:
        _, dropped = _PREVIEW.popitem(last=False)
        total -= dropped["bytes"]


def _cache_get(token: str):
    """Fetch a strip by its token. An unknown token is simply a miss — which is
    what a stale front-end (server restarted, entry evicted) gets, and it reads
    on the node as "run once" rather than as another shot's frames."""
    if not token:
        return None
    entry = _PREVIEW.get(token)
    if entry is None:
        return None
    _PREVIEW.move_to_end(token)
    return entry


def _build_preview(node_id, images) -> dict:
    """Cache a thumbnail strip of `images` and return the `ui` payload for it.

    A preview is never worth failing a render over, so every failure here
    degrades to "no preview" and the node still returns its frames.
    """
    if images.ndim != 4:
        raise ValueError(f"expected an (N,H,W,C) batch, got shape {tuple(images.shape)}")
    n = int(images.shape[0])
    h, w = int(images.shape[1]), int(images.shape[2])
    th, tw = _thumb_dims(h, w, n)
    thumbs = _thumbnails(images, th, tw)
    token = uuid.uuid4().hex[:16]
    _cache_put(token, {
        "node_id": str(node_id),
        "thumbs": thumbs,
        "n": n,
        "src_w": w,
        "src_h": h,
        "bytes": int(thumbs.nbytes),
    })
    return {
        "token": [token],
        "frames": [n],
        "src_w": [w],
        "src_h": [h],
        "thumb_w": [tw],
        "thumb_h": [th],
    }


class BatFramehold:
    """Pick / reorder frames of an IMAGE batch by an index spec string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The batch to select from. Index 0 is this "
                               "batch's first frame, not a shot frame number.",
                }),
                "frames": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "placeholder": "e.g.  0   |   0-10   |   0, 5, 8-12",
                    # The whole grammar, on the field it applies to. Tooltips
                    # render with `white-space: pre-wrap`, so the line breaks
                    # survive; the layout is a "token — meaning" list rather
                    # than aligned columns because the tooltip font is
                    # proportional and columns come out ragged.
                    "tooltip": (
                        "Which frames to keep — 0-based indices, comma-separated. "
                        "Each item is either one index or an inclusive range.\n"
                        "\n"
                        "0  —  the first frame\n"
                        "0-10  —  frames 0 to 10, BOTH ends included (11 frames)\n"
                        "0, 5  —  just those two\n"
                        "0-2, 7, 10-12  —  any mix of the above\n"
                        "-1  —  the last frame; negatives count from the end\n"
                        "-5--1  —  the last five\n"
                        "10-5  —  a descending range: that run, reversed\n"
                        "5, 0, 5  —  order and repeats are kept, so this both "
                        "reorders and holds\n"
                        "\n"
                        "Out-of-range indices clamp to the batch rather than "
                        "failing the render. Left empty, the whole batch passes "
                        "through unchanged.\n"
                        "\n"
                        "Easier: click the preview strip on the node to open the "
                        "scrubber, step with ← / →, and press Enter to hold the "
                        "frame you are looking at."
                    ),
                }),
            },
            # Keys the preview strip to this node instance. Hidden inputs don't
            # appear in the graph and don't disturb existing workflows.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "run"
    CATEGORY = "BAT/Utility"
    DESCRIPTION = (
        "Select frames from an image batch by index. Single: `0`. "
        "Inclusive range: `0-10`. List: `0, 5`. Mix: `0-2, 7, 10-12`. "
        "Negatives count from the end; order and duplicates are preserved. "
        "Click the preview strip on the node to open a scrubber: ←/→ step the "
        "input batch, Enter sets the hold, A adds the frame to the selection."
    )

    def run(self, images, frames, unique_id=None):
        n = images.shape[0]
        idxs = _parse_frames(frames, n)

        ui = {}
        if unique_id is not None:
            try:
                ui = _build_preview(unique_id, images)
            except Exception as exc:  # pragma: no cover - preview is never fatal
                logger.warning("Bat_Framehold: preview strip failed (%s); the "
                               "node still ran, the scrubber just stays empty.", exc)
                ui = {}

        if not idxs:
            logger.warning("Bat_Framehold: no valid frames parsed from %r; "
                           "passing the batch through unchanged.", frames)
            return {"ui": ui, "result": (images, int(n))}
        index = torch.tensor(idxs, dtype=torch.long, device=images.device)
        picked = images.index_select(0, index)
        return {"ui": ui, "result": (picked, int(picked.shape[0]))}


# ─── API routes ──────────────────────────────────────────────────────────────


@server.PromptServer.instance.routes.get("/bat/framehold/info")
async def bat_framehold_info(request):
    """Is the strip behind this token still in the cache?

    Called on workflow load, so a reopened graph gets its scrubber back without
    a re-run — the front-end remembers only the token, in workflow-scoped
    localStorage.
    """
    token = request.rel_url.query.get("token", "")
    entry = _cache_get(token)
    if entry is None:
        return server.web.json_response({"ok": False})
    return server.web.json_response({
        "ok": True,
        "token": token,
        "frames": entry["n"],
        "src_w": entry["src_w"],
        "src_h": entry["src_h"],
        "thumb_w": int(entry["thumbs"].shape[2]),
        "thumb_h": int(entry["thumbs"].shape[1]),
    })


@server.PromptServer.instance.routes.get("/bat/framehold/frame")
async def bat_framehold_frame(request):
    """One frame of the strip as a JPEG.

    Encoded per request rather than at execution time: the strip is already in
    RAM as uint8, one small JPEG costs about a millisecond, and an artist only
    ever looks at a fraction of a long batch. Encoding all of them up front
    would put that cost on every render instead.
    """
    q = request.rel_url.query
    entry = _cache_get(q.get("token", ""))
    if entry is None:
        return server.web.Response(status=404, text="no preview for this token")
    try:
        i = int(q.get("i", "0"))
    except ValueError:
        i = 0
    i = max(0, min(i, entry["n"] - 1))

    buf = io.BytesIO()
    Image.fromarray(entry["thumbs"][i], "RGB").save(buf, format="JPEG", quality=82)
    return server.web.Response(
        body=buf.getvalue(),
        content_type="image/jpeg",
        # The token makes a URL immutable — the same (token, i) can never
        # mean a different frame — so the browser may cache hard. That is what
        # keeps arrow-key scrubbing instant on the second pass and lets the
        # preview play back without hammering the server.
        headers={"Cache-Control": "private, max-age=3600"},
    )
