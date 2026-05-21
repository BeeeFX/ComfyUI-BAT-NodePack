# 🦇 ComfyUI-BAT-NodePack

A small pack of batch / video utility nodes for ComfyUI. Mostly built around
sliding-context video models (WAN, VACE) and generic per-frame batch
manipulation.

These are personal tools that I (Baptiste) reach for in everyday workflows;
sharing them in case they're useful to others.

---

## Nodes

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Video Grid Split**       | `BAT/Video`    | Splits a video frame batch into a grid of sub-clips with optional overlap. Useful for tiled rendering where one input video maps to N output regions. |
| 🦇 **VACE Batch Tool**        | `BAT/VACE`     | Batch builder for VACE (Video Authoring Compositor) workflows. Composes per-frame image / mask / control inputs into a single stacked batch with optional fill colour and premultiplication. Adds `+ Keyframe` / `− Keyframe` buttons to the node for variable-length input editing. |
| 🦇 **WAN Context Calculator** | `BAT/WAN`      | Recommends optimal `num_frames` / `context_frames` / `stride` / `overlap` for the WAN sliding-context video model based on your input clip and target priorities (quality vs. speed). |
| 🦇 **WAN Batch Format**       | `BAT/WAN`      | Formats an input image batch into the WAN sliding-context structure. Pads / trims to target frame count, applies static-standard window timing, and emits a debug-window visualisation. |
| 🦇 **WAN Batch Crop**         | `BAT/WAN`      | The inverse of *WAN Batch Format* — crops a WAN-formatted batch back to its original frame count after generation, removing the start/end padding. |
| 🦇 **Video Batch Format**     | `BAT/Video`    | Generic video batch formatter — pads / trims / normalises an image batch to a target frame count using the selected model's temporal stride. |

All nodes register under `class_type` keys prefixed `Bat_…`, e.g.
`Bat_WanBatchFormat`. Display names start with 🦇 so they're easy to spot
in the *Add Node* menu.

---

## Install

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-NodePack.git
```

Restart ComfyUI. The nodes appear under `BAT/Video`, `BAT/VACE`, and
`BAT/WAN` in the right-click *Add Node* menu.

### ComfyUI-Manager

Not yet listed in the official registry. PRs welcome.

### Dependencies

No extra packages — uses `torch`, `PIL`, and `numpy` from ComfyUI's
environment.

---

## Workflow compatibility

These nodes were previously published internally under an `ETC_Tools`
pack with `Volt_*` class names. If you have old workflows that reference
those legacy class names, the BAT pack ships a frontend shim
(`web/js/bat-migrations.js`) that registers the `Volt_* → Bat_*` mapping
with a separate ETC suite migration tool. Without that companion tool
installed the shim no-ops silently; you'd need to drop the new BAT nodes
in by hand.

For fresh installs (the vast majority of public users) none of this
matters — just install and add nodes as usual.

---

## Author

[Baptiste](https://github.com/BeeeFX) · 2026

## Licence

MIT — see [LICENSE](./LICENSE). Use freely; attribution appreciated but
not required.
