# 🦇 ComfyUI-BAT-NodePack

A pack of batch / video utility nodes for ComfyUI. Built around
sliding-context video models (WAN, VACE), generic per-frame batch
manipulation, and a set of interactive on-node canvas editors — crop,
animated crop, grade, and roto — for VFX-style shot work.

These are personal tools that I (Baptiste) reach for in everyday workflows;
sharing them in case they're useful to others.

---

## Nodes

### Video & batch utilities

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Video Grid Split**       | `BAT/video`    | Splits a video frame batch into a grid of sub-clips with optional overlap. Useful for tiled rendering where one input video maps to N output regions. |
| 🦇 **Video Batch Format**     | `BAT/video`    | Generic video batch formatter — pads / trims / normalises an image batch to a target frame count using the selected model's temporal stride. |
| 🦇 **Video Loader**           | `BAT/video`    | Standalone video reader. Picks a file (path or upload), decodes via `imageio_ffmpeg`, emits IMAGE batch + AUDIO + frame metadata. In-node preview + scrubber so artists can verify frame ranges before queuing. |
| 🦇 **Video Combine**          | `BAT/video`    | Encodes an IMAGE batch (+ optional AUDIO) to a video file through ffmpeg. Format catalog in `bat_video_formats/*.json` covers H264 / H265 / VP9 / FFV1 / ProRes / GIF / WebP / EXR-sequence / PNG-sequence. Inline browser player with per-frame stepping, client-side hover thumbnails, save-frame-as-PNG, and smooth playhead via `requestVideoFrameCallback`. |
| 🦇 **Framehold**              | `BAT/video`    | Holds one frame of a batch across the whole batch — the frame-sequence equivalent of a freeze frame. |

### Canvas editors

Interactive nodes that draw a live preview on the node itself. All of them
share the display-zoom / pan, teardown, and layout modules described under
[Shared frontend modules](#shared-frontend-modules).

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Crop**                   | `BAT/image`    | Crop an IMAGE batch to a rectangular region, dragged directly on the node. Emits the cropped IMAGE + an Uncrop payload (original-size canvas + crop rect) so the inverse op can recompose without state shared through the graph. `constrain_to_canvas` controls whether the box may extend past the frame edge. |
| 🦇 **Animated Crop**          | `BAT/image`    | Keyframed *Crop* — set the box on separate frames and it interpolates between them, for following a subject across a shot. Same `constrain_to_canvas` toggle. |
| 🦇 **Uncrop**                 | `BAT/image`    | Pastes a (possibly resized / generated) IMAGE back at the rect recorded by *Crop*, with optional feather. Handles negative-origin rects from an unconstrained crop. The "inpaint just a region" companion. |
| 🦇 **Grade**                  | `BAT/image`    | Lift / gamma / gain / saturation grade with a live on-node preview — quick look adjustments without round-tripping through another app. |
| 🦇 **Animated Grade**         | `BAT/image`    | Keyframed *Grade*, for a look that changes across the shot. |
| 🦇 **Roto**                   | `BAT/image`    | Draw and animate bezier roto shapes on the node; outputs a MASK. |

### Masks

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Grow Mask**              | `BAT/mask`     | Dilates a MASK by N pixels. |
| 🦇 **Erode Mask**             | `BAT/mask`     | Erodes a MASK by N pixels. |

### VACE

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **VACE Batch Tool**        | `BAT/vace`     | Batch builder for VACE (Video Authoring Compositor) workflows. Composes per-frame image / mask / control inputs into a single stacked batch with optional fill colour and premultiplication. Adds `+ Keyframe` / `− Keyframe` buttons to the node for variable-length input editing. |

### WAN sliding-context

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **WAN Context Calculator** | `BAT/wan`      | Recommends optimal `num_frames` / `context_frames` / `stride` / `overlap` for the WAN sliding-context video model based on your input clip and target priorities (quality vs. speed). |
| 🦇 **WAN Batch Format**       | `BAT/wan`      | Formats an input image batch into the WAN sliding-context structure. Pads / trims to target frame count, applies static-standard window timing, and emits a debug-window visualisation. |
| 🦇 **WAN Batch Crop**         | `BAT/wan`      | The inverse of *WAN Batch Format* — crops a WAN-formatted batch back to its original frame count after generation, removing the start/end padding. |
| 🦇 **WAN Batch Frame Format** | `BAT/wan`      | Per-frame variant of *WAN Batch Format*: emits a single frame with the right WAN window context applied around it. Handy for "preview one window before committing to a full render" or for ControlNet branches that only need one frame per window. |
| 🦇 **Wan Reference Aligner**  | `BAT/wan`      | Aligns a reference image / batch to a WAN-formatted target. Handles the per-frame indexing math so a still ref (or short ref clip) broadcasts onto the same window layout as the target, ready to feed into a control branch. Debug overlay on the node visualises the detected window layout. |

### Editor & I/O helpers

| Display name                  | Category       | What it does |
|-------------------------------|----------------|---|
| 🦇 **Points Editor**          | `BAT/editors`  | In-node 2D points editor with image backdrop, point labels, multiple colour groups, and JSON serialisation. Used upstream of any node that wants explicit (x, y) point sets — typically ControlNet OpenPose / face-region nudging / per-region prompt masks. |
| 🦇 **BAT Frame Picker**       | `BAT/editors`  | Contact-sheet grid of every frame in a video *or* a `####` frame sequence, with filesystem autocomplete on the path. Click a cell to pick it; only that one frame is decoded at full resolution, so scrubbing a heavy EXR sequence stays cheap. |
| 🦇 **Filename Prefix**        | `BAT/io`       | Builds output filename prefixes from workflow context, for consistent naming across a render. |

All nodes register under `class_type` keys prefixed `Bat_…`, e.g.
`Bat_WanBatchFormat`. Display names start with 🦇 so they're easy to spot
in the *Add Node* menu.

---

## Video formats

*Video Combine* reads its encoder presets from JSON files in
`bat_video_formats/` — drop a new file in there to add a format, no Python
changes needed. Alongside the ffmpeg arguments, each preset can declare:

| Key | Meaning |
|---|---|
| `video_args` / `audio_args` | ffmpeg arguments, with `{widget}` placeholders |
| `widgets` | Widgets exposed on the node. Each supports `label` (friendly name) and `hidden`. |
| `derived` | Compute one widget from another. ProRes uses this so `pix_fmt` follows `profile` — picking 4444 automatically selects an alpha-capable pixel format instead of leaving it to the user. |
| `requires_even_dims` | Pad odd width/height, required by h264 / h265 / ProRes. |
| `metadata` | Metadata writer to use (e.g. `ffmpeg_mov`), so the workflow can be recovered from the written file. |
| `input_color_depth` | Request 16-bit input for formats that benefit, e.g. ProRes and EXR. |
| `browser_playable` | Whether the inline review player can play the result directly. |

---

## Shared frontend modules

Three modules under `web/` are shared by the canvas editors rather than
duplicated per node:

- **`bat_zoom_control.js`** — display-only zoom + pan for the on-node
  preview, so you can pull back and work on a crop or roto that extends past
  the frame edge. Purely a view transform; the backend only ever sees widget
  values, never this.
- **`bat_lifecycle.js`** — `onRemoved` teardown for editor resources
  (playback intervals, `ResizeObserver`s, `IntersectionObserver`s, RAF loops,
  window-level pointer listeners). Without it, deleting a node left all of
  that running against a detached DOM indefinitely.
- **`bat_node_layout.js`** — dual-mode DOM-widget sizing, so the editors lay
  out correctly under both ComfyUI Nodes 1.0 (litegraph canvas, sized from
  `node.size`) and Nodes 2.0 (`Comfy.VueNodes.Enabled`, which computes height
  bottom-up from each widget's `computeLayoutSize()`).

---

## Install

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-NodePack.git
```

Restart ComfyUI. The nodes appear under `BAT/video`, `BAT/vace`, `BAT/wan`,
`BAT/image`, `BAT/mask`, `BAT/io`, and `BAT/editors` in the right-click
*Add Node* menu.

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

### Removed nodes

**🦇 VRI Frame Picker** (`Bat_VriPicker`) has been removed. It was tied to
VRI (Volt Resource Identifier) pipeline paths, which meant nothing outside
our studio setup; that version now lives in the internal ETC suite. The
replacement here is **🦇 BAT Frame Picker** (`Bat_FramePicker`) — same
contact-sheet UI, but driven by an ordinary filesystem path or `####`
sequence pattern, so it works on any machine.

---

## Author

[Baptiste](https://github.com/BeeeFX) · 2026

## Licence

MIT — see [LICENSE](./LICENSE). Use freely; attribution appreciated but
not required.
