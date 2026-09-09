"""
BAT Profiler — HTTP surface
===========================

The websocket stream (``bat.profiler.*``) is the live path; these routes
are the *catch-up* path. They matter in exactly the cases the live
stream cannot cover:

* the panel is opened after a run has already finished,
* the browser is reloaded mid-run,
* ComfyUI is still up but the frontend lost the socket.

Note the asymmetry that makes the whole feature work: the backend keeps
the last few runs in RAM, so it loses everything if the *server* dies —
which is precisely the crash you are trying to diagnose. The frontend's
localStorage copy is the one that survives that, which is why the panel
writes every node record as it arrives rather than at run end.
"""

from __future__ import annotations

import logging

from aiohttp import web

from . import bat_profiler as prof

logger = logging.getLogger("BAT.Profiler")

try:
    from server import PromptServer
    routes = PromptServer.instance.routes
except Exception as e:  # pragma: no cover
    routes = None
    logger.warning("[BAT Profiler] no PromptServer routes: %r", e)


if routes is not None:

    @routes.get("/bat/profiler/state")
    async def bat_profiler_state(request):
        """Config, capabilities, and run summaries — optionally filtered
        to one workflow so a tab never sees another tab's runs."""
        prof.note_subscriber()
        workflow = request.query.get("workflow") or None
        client_id = request.query.get("client_id") or None
        return web.json_response(prof.state_payload(workflow, client_id))

    @routes.get("/bat/profiler/run/{prompt_id}")
    async def bat_profiler_run(request):
        """One run in full, including the sample series for the graphs."""
        run = prof.get_run(request.match_info["prompt_id"])
        if run is None:
            return web.json_response({"error": "unknown run"}, status=404)
        with_samples = request.query.get("samples", "1") != "0"
        return web.json_response(run.to_dict(with_samples=with_samples))

    @routes.post("/bat/profiler/subscribe")
    async def bat_profiler_subscribe(request):
        """Heartbeat from an open panel. The 5 Hz sample stream is only
        broadcast while somebody is listening; node and run records are
        always sent regardless, because those are what has to survive a
        crash whether or not the panel happened to be open."""
        prof.note_subscriber()
        return web.json_response({"ok": True})

    @routes.post("/bat/profiler/claim")
    async def bat_profiler_claim(request):
        """Bind a prompt_id to the workflow that submitted it.

        Called by the frontend at queue time rather than at execution
        time on purpose: you can queue from tab A and immediately switch
        to tab B, and the run still belongs to A.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        prompt_id = body.get("prompt_id")
        workflow = body.get("workflow")
        if not prompt_id:
            return web.json_response({"error": "prompt_id required"}, status=400)
        prof.claim(str(prompt_id), workflow)
        return web.json_response({"ok": True})

    @routes.post("/bat/profiler/arm")
    async def bat_profiler_arm(request):
        """Arm or disarm profiling **for the calling browser only**.

        This is the on/off switch behind the panel's big toggle. It is
        per client_id on purpose: this is one shared ComfyUI process and
        most people on it are not debugging, so one artist enabling the
        profiler must not add a per-node GPU sync to everybody else's
        renders. Only prompts submitted by an armed client are
        instrumented; every other prompt runs the original code path.

        Called on toggle rather than on queue, so there is no race with
        a prompt that starts executing the moment it is submitted.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        client_id = body.get("client_id")
        if not client_id:
            return web.json_response({"error": "client_id required"}, status=400)
        enabled = bool(body.get("enabled"))
        prof.arm(str(client_id), enabled)
        return web.json_response(prof.state_payload(None, str(client_id)))

    @routes.post("/bat/profiler/config")
    async def bat_profiler_config(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)
        # Note: enablement is NOT here — it is per-client, via /arm.
        # These are measurement-method settings that only take effect
        # inside an already-armed run.
        if "sync_cuda" in body:
            prof.CONFIG.sync_cuda = bool(body["sync_cuda"])
        if "reset_peak" in body:
            prof.CONFIG.reset_peak = bool(body["reset_peak"])
        if "sample_hz" in body:
            try:
                hz = float(body["sample_hz"])
                prof.CONFIG.sample_hz_active = max(0.5, min(20.0, hz))
            except Exception:
                pass
        logger.info("[BAT Profiler] config: sync_cuda=%s reset_peak=%s hz=%s",
                    prof.CONFIG.sync_cuda, prof.CONFIG.reset_peak,
                    prof.CONFIG.sample_hz_active)
        return web.json_response(prof.state_payload(None, body.get("client_id")))
