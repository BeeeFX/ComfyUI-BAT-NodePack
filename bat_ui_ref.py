"""
BAT — keep the editors' preview payloads out of ComfyUI's prompt history.

The problem
-----------
Every on-node editor (Roto, Animated Crop/Grade, Grade, Advanced Blend, Layered
Images, HDR Tonal Composite, SeC, …) ships its preview in the node's ``ui``
dict: base64 JPEG strips, PNG masks, zlib'd float tiles. That dict is not only
sent to the browser once — ComfyUI stores it in the prompt history
(``PromptQueue.task_done`` → ``history[prompt_id]["outputs"]``, up to
``MAXIMUM_HISTORY_SIZE`` = 10 000 prompts) and re-sends it on every fully cached
re-run (``emit_cached_output``). Measured: 7–21 MB per run for a 120-frame
Animated Crop strip, 15–20 MB per SeC run. On a shared box that is gigabytes of
server RAM after a day of iterating, plus a multi-MB ``/history`` response.

The fix
-------
The node writes the payload to ``<temp>/bat_ui/<token>.json`` and returns a
``ui`` of just ``{"bat_ui": [token]}`` — a few dozen bytes in the history. The
browser fetches ``/bat/ui/<token>`` and hands the editor the exact dict it used
to receive (``web/bat_ui_ref.js`` does that for every BAT node, so no editor had
to change). ComfyUI empties its temp directory at startup, and the sidecar
folder is capped by size, so the files never outlive their usefulness: a cached
re-run within the session re-sends the same token and it still resolves.

Keys the *frontend itself* renders (``images``, ``text``, …) must stay inline —
pass them in ``keep``.
"""

import json
import logging
import os
import re
import tempfile
import threading
import uuid

try:
    import folder_paths
except ImportError:          # imported outside ComfyUI (tests)
    folder_paths = None

logger = logging.getLogger("BAT.ui_ref")

UI_KEY = "bat_ui"
_SUBDIR = "bat_ui"
# Sidecars kept per server session before the oldest go. A run is typically
# 0.1–20 MB, so this is hundreds of runs — far more than anyone scrolls back to.
MAX_BYTES = 1 << 30
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_lock = threading.Lock()


def _sidecar_dir():
    base = folder_paths.get_temp_directory() if folder_paths else tempfile.gettempdir()
    path = os.path.join(base, _SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _prune(directory, keep):
    """Oldest-first delete once the folder passes MAX_BYTES (down to 3/4 of it)."""
    entries = []
    total = 0
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        p = os.path.join(directory, name)
        try:
            st = os.stat(p)
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, p))
        total += st.st_size
    if total <= MAX_BYTES:
        return
    entries.sort()
    for _mtime, size, p in entries:
        if total <= MAX_BYTES * 3 // 4:
            break
        if p == keep:
            continue
        try:
            os.remove(p)
            total -= size
        except OSError:
            pass


def stash_ui(payload, keep=()):
    """Replace a heavy ``ui`` dict with a reference to it on disk.

    Returns ``{"bat_ui": [token], **{k: payload[k] for k in keep}}``. If the
    payload cannot be written (read-only temp, not JSON-able) it is returned
    unchanged — the editor then works exactly as before, just without the
    history saving.
    """
    if not isinstance(payload, dict) or not payload:
        return payload
    inline = {k: payload[k] for k in keep if k in payload}
    rest = {k: v for k, v in payload.items() if k not in keep}
    if not rest:
        return payload
    try:
        data = json.dumps(rest, separators=(",", ":")).encode("utf-8")
        token = uuid.uuid4().hex
        with _lock:
            directory = _sidecar_dir()
            final = os.path.join(directory, token + ".json")
            part = final + ".part"
            with open(part, "wb") as f:
                f.write(data)
            os.replace(part, final)
            _prune(directory, keep=final)
    except Exception as exc:
        logger.warning("[BAT] preview payload kept inline (%s)", exc)
        return payload
    inline[UI_KEY] = [token]
    return inline


def load_ui(ui):
    """The full ``ui`` dict a stashed one stands for (tests, server-side use).

    Accepts either a stashed ``ui`` or an inline one, so callers need not know
    which they got. Returns None when the sidecar is gone.
    """
    if not isinstance(ui, dict) or UI_KEY not in ui:
        return ui
    token = (ui.get(UI_KEY) or [None])[0]
    if not isinstance(token, str) or not _TOKEN_RE.match(token):
        return None
    try:
        with open(os.path.join(_sidecar_dir(), token + ".json"), "rb") as f:
            rest = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    out = {k: v for k, v in ui.items() if k != UI_KEY}
    out.update(rest)
    return out


def _register_routes():
    try:
        import server
        from aiohttp import web
    except Exception:
        return

    @server.PromptServer.instance.routes.get("/bat/ui/{token}")
    async def get_ui_payload(request):
        token = request.match_info.get("token", "")
        if not _TOKEN_RE.match(token):
            return web.Response(status=404)
        path = os.path.join(_sidecar_dir(), token + ".json")
        if not os.path.isfile(path):
            return web.Response(status=404)
        # A token names one immutable payload, so the browser may keep it.
        return web.FileResponse(path, headers={
            "Content-Type": "application/json",
            "Cache-Control": "private, max-age=86400, immutable",
        })


try:
    _register_routes()
except Exception as _exc:  # no PromptServer instance (tests / headless import)
    logger.debug("BAT ui-ref route not registered (%s)", _exc)
