"""Cancellation support for the long-running BAT nodes.

ComfyUI's Cancel button sets a flag; the executor only acts on it *between*
nodes. A node that spends thirty seconds inside one call therefore cannot be
cancelled at all — the queue stops, the UI says "cancelling", and the artist
waits for the node to finish anyway. The fix is for the node itself to poll,
which is what `comfy.model_management.throw_exception_if_processing_interrupted`
is for: it raises `InterruptProcessingException`, which the executor already
knows how to unwind cleanly.

Nothing in this pack was polling, which is why Advanced Blend "took a while to
cancel". Any node with a per-frame or per-chunk loop should call `check()` once
per iteration — it is a flag read, far too cheap to matter next to a chunk of
image maths, and it turns a thirty-second wait into a sub-second one.

Imported defensively so the modules stay unit-testable outside a running
ComfyUI, where `comfy` is not importable at all.
"""

try:                                        # pragma: no cover - env dependent
    from comfy.model_management import throw_exception_if_processing_interrupted
except Exception:                           # pragma: no cover
    throw_exception_if_processing_interrupted = None


def check():
    """Raise if the user has hit Cancel. No-op outside ComfyUI."""
    if throw_exception_if_processing_interrupted is not None:
        throw_exception_if_processing_interrupted()
