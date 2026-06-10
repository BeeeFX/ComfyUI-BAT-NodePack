# 🦇 ComfyUI-BAT-NodePack

A small pack of batch / video utility nodes for ComfyUI. Mostly built around
sliding-context video models (WAN, VACE) and generic per-frame batch
manipulation, plus a handful of editor and I/O helpers.

These are personal tools that I (Baptiste) reach for in everyday workflows;
sharing them in case they're useful to others.

---

## Nodes

### Video & batch utilities

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Video Grid Split**       | `BAT/Video`    | Splits a video frame batch into a grid of sub-clips with optional overlap. Useful for tiled rendering where one input video maps to N output regions. |
| 🦇 **Video Batch Format**     | `BAT/Video`    | Generic video batch formatter — pads / trims / normalises an image batch to a target frame count using the selected model's temporal stride. |
| 🦇 **Video Loader**           | `BAT/Video`    | Standalone video reader. Picks a file (path or upload), decodes via `imageio_ffmpeg`, emits IMAGE batch + AUDIO + frame metadata. In-node preview + scrubber so artists can verify frame ranges before queuing. |
| 🦇 **Video Combine**          | `BAT/Video`    | Encodes an IMAGE batch (+ optional AUDIO) to a video file through ffmpeg. Format catalog in `bat_video_formats/*.json` covers H264 / H265 / VP9 / FFV1 / ProRes / GIF / WebP / EXR-sequence / PNG-sequence. Inline browser player with per-frame stepping, client-side hover thumbnails, save-frame-as-PNG, and smooth playhead via `requestVideoFrameCallback`. |
| 🦇 **Crop**                   | `BAT/Image`    | Crop an IMAGE batch to a rectangular region. Emits the cropped IMAGE + an Uncrop payload (original-size canvas + crop rect) so the inverse op can recompose without state shared through the graph. |
| 🦇 **Uncrop**                 | `BAT/Image`    | Pastes a (possibly resized / generated) IMAGE back at the rect recorded by *Crop*, with optional feather. The "inpaint just a region" companion. |

### VACE

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **VACE Batch Tool**        | `BAT/VACE`     | Batch builder for VACE (Video Authoring Compositor) workflows. Composes per-frame image / mask / control inputs into a single stacked batch with optional fill colour and premultiplication. Adds `+ Keyframe` / `− Keyframe` buttons to the node for variable-length input editing. |

### WAN sliding-context

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **WAN Context Calculator** | `BAT/WAN`      | Recommends optimal `num_frames` / `context_frames` / `stride` / `overlap` for the WAN sliding-context video model based on your input clip and target priorities (quality vs. speed). |
| 🦇 **WAN Batch Format**       | `BAT/WAN`      | Formats an input image batch into the WAN sliding-context structure. Pads / trims to target frame count, applies static-standard window timing, and emits a debug-window visualisation. |
| 🦇 **WAN Batch Crop**         | `BAT/WAN`      | The inverse of *WAN Batch Format* — crops a WAN-formatted batch back to its original frame count after generation, removing the start/end padding. |
| 🦇 **WAN Batch Frame Format** | `BAT/WAN`      | Per-frame variant of *WAN Batch Format*: emits a single frame with the right WAN window context applied around it. Handy for "preview one window before committing to a full render" or for ControlNet branches that only need one frame per window. |
| 🦇 **Wan Reference Aligner**  | `BAT/WAN`      | Aligns a reference image / batch to a WAN-formatted target. Handles the per-frame indexing math so a still ref (or short ref clip) broadcasts onto the same window layout as the target, ready to feed into a control branch. Debug overlay on the node visualises the detected window layout. |

### Editor & I/O helpers

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Points Editor**          | `BAT/Editor`   | In-node 2D points editor with image backdrop, point labels, multiple colour groups, and JSON serialisation. Used upstream of any node that wants explicit (x, y) point sets — typically ControlNet OpenPose / face-region nudging / per-region prompt masks. |
| 🦇 **VRI Frame Picker**       | `BAT/Video`    | Scrubs through a VRI (Volt Resource Identifier) clip in the node UI and emits the picked frame index as INT. Soft-couples to `ComfyUI-Volt_Loader` if installed; falls back to a plain file picker otherwise. |

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

Restart ComfyUI. The nodes appear under `BAT/Video`, `BAT/VACE`, `BAT/WAN`,
`BAT/Image`, and `BAT/Editor` in the right-click *Add Node* menu.

### ComfyUI-Manager

Not yet listed in the official registry. PRs welcome.

### Dependencies

No extra Python packages — uses `torch`, `PIL`, `numpy`, and
`imageio_ffmpeg` (already in ComfyUI's env). The Video Combine node
shells out to `ffmpeg` for encoding; if `imageio_ffmpeg.get_ffmpeg_exe()`
isn't usable on your box, the node falls back to whichever `ffmpeg` is
on `$PATH`.

---

## Workflow compatibility

These nodes were previously published internally under an `ETC_Tools`
pack with `Volt_*` class names. If you have old workflows that reference
those legacy class names, the BAT pack ships a frontend shim
(`web/bat-migrations.js`) that registers the `Volt_* → Bat_*` mapping
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
