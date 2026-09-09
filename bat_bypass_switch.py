"""
Bat Bypass Switch — graph-management node with no data path.
============================================================

What it is
----------
A node you drop into a template that carries a list of named toggles. Each
toggle owns a *selection* of nodes / subgraph nodes / backdrops captured from
the canvas, and clicking it flips every node in that selection between
``Always`` (mode 0) and ``Bypass`` (mode 4). It is the on-canvas equivalent of
"select these twelve nodes and hit Ctrl+B", turned into a labelled switch that
someone who did not build the workflow can find and understand.

Why the backend is a stub
-------------------------
Everything this node does happens in the browser, to the *graph*, before a
prompt is ever built — so there is nothing for Python to execute:

* ``RETURN_TYPES`` is empty and ``OUTPUT_NODE`` is left False, which means the
  node is never an execution root and never reachable from one. It therefore
  never runs, never appears in the executor's dependency walk, and costs
  nothing at queue time. Making it an OUTPUT_NODE would be actively harmful:
  output nodes are unconditional execution roots, which is exactly the thing
  that defeats caching elsewhere in this pack.
* The class still has to exist, because the frontend will not let you place a
  node whose ``class_type`` has no server-side definition — and, more
  importantly, because ``presets`` has to be a *declared widget* for its value
  to land in ``widgets_values`` and so be saved into the workflow JSON.

The ``presets`` widget
----------------------
One STRING widget holds the whole thing as JSON: the toggle list, their names,
their captured node/group ids, their on/off state and any exclusive-group
membership. It is hidden on the canvas (see ``web/bat_bypass_switch.js``) — the
node draws real toggle widgets instead — but keeping the state in a declared
widget rather than in ``node.properties`` buys three things for free:

* it round-trips through workflow save/load like any other widget value;
* it survives undo/redo, which in ComfyUI is a full ``loadGraphData`` and so
  only preserves what actually serialises;
* it is readable and diffable in the .json if anyone needs to audit or
  hand-repair a template.

It is deliberately single-line (``multiline`` False). A multiline STRING is
rendered as a DOM textarea, which is a great deal more work to hide reliably
than a plain text widget, and nobody is meant to type in here.

Position matters: ``presets`` is the only serialising widget on the node, and
litegraph writes ``widgets_values`` at each widget's *full array index*, so it
has to stay at index 0 with every dynamically-built toggle appended after it.
The toggles are all created with ``serialize = false`` for that reason.
"""

# Empty state, duplicated in the JS as `emptyState()`. Kept in sync by hand;
# if they ever disagree the JS normaliser wins, since it repairs whatever it
# is handed.
DEFAULT_PRESETS = '{"v":1,"exgroups":[],"presets":[]}'


class BatBypassSwitch:
    """Inert node. Its entire behaviour lives in web/bat_bypass_switch.js."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "presets": ("STRING", {
                    "default": DEFAULT_PRESETS,
                    "multiline": False,
                    "tooltip":
                        "The switch's toggles, as JSON: names, captured node "
                        "and backdrop ids, on/off state, exclusive groups. "
                        "Hidden on the node — use the buttons on the node "
                        "body. Saved into the workflow so a template keeps "
                        "its switches.",
                }),
            },
        }

    # No outputs and not an output node: nothing to execute, ever.
    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "BAT/workflow"
    DESCRIPTION = (
        "Named toggles that bypass/un-bypass saved selections of nodes, "
        "subgraphs and backdrops. Not wired into the graph — it manages it."
    )

    def noop(self, presets=DEFAULT_PRESETS):
        # Unreachable in practice (see the module docstring), but a FUNCTION
        # that does not exist is a load-time error in some ComfyUI builds.
        return ()
