<p align="center">
  <img src="docs/assets/bat-banner.svg" alt="BAT — Video, animation and compositing tools for ComfyUI" width="100%">
</p>

# 🦇 ComfyUI-BAT-NodePack

BAT brings media loading, video tools, keyframe animation, and familiar compositing controls into your node graph. Bring in a clip or layered EXR, animate a crop, draw a mask, balance a grade, or blend an upscale back into your original footage—with interactive previews right on the nodes.

These are the tools I use in my own workflows. I'm **BeeeFX**, and I'm sharing them in case they make your next shot a little easier too.

**43 nodes · Interactive editors · BAT Profiler · MIT-licensed BAT code**

[Highlights](#highlights) · [All nodes](#all-nodes) · [Install](#install) · [Try it](#try-it) · [Help](#help)

## Highlights

### Start with the media you have.

**Loader** brings stills, numbered image sequences, folders of images, and movies into one node. Enter a path, then skip frames, sample every *n*th frame, or cap the batch. Movie audio and source alpha are available when present.

For a **layered EXR**, Loader can also hand off its passes and Cryptomatte data. Use **EXR Layer** to pick a named pass, or connect **Points Editor → Cryptomatte Matte** to click on objects and build a mask. EXR support needs the optional `OpenImageIO` package; see [installation](#install).

Prefer to choose a movie's in and out points by eye? Use the dedicated **Video Loader** below. It sits alongside Loader and offers a visual trim slider and player.

### Bring your footage in. Send the finished shot out.

**Video Loader + Video Combine** bookend your video workflow. Load a clip, scrub through it, and choose your in/out range with a visual trim control. Audio follows the trim, and sources with transparency can supply a mask.

When you're ready, export video, GIF, WebP, or an image sequence. Video Combine includes an inline player you can maximise, frame stepping, and a save-frame-as-PNG option for reviewing the result.

**Export choices:** H.264 · H.265 · VP9 · FFV1 · ProRes · GIF · WebP · EXR · PNG. Playback in the inline player depends on the browser and format.

![BAT Video Loader and Video Combine showing trim controls and an inline video preview](docs/assets/bat-video-loader-combine.png)

*Trim a clip in Video Loader, then review the export in Video Combine.*

### Animate the adjustment, frame by frame.

**Animated Crop, Animated Grade, and Roto** let you set keyframes directly in the node. Follow a subject with a moving crop, change the look over time, or draw and animate Bézier mask shapes.

Pair **Animated Crop** with **Uncrop** to work on a moving region, then put the processed result back into the original shot. Uncrop follows the saved crop positions and offers edge feathering to soften the join.

![Animated Crop and Animated Grade with keyframes visible in their on-node timelines](docs/assets/bat-animated-crop-grade.png)

*Move the crop and change the grade over time with keyframes right on the nodes.*

![BAT Roto shape connected to Animated Grade for a selective adjustment](docs/assets/bat-roto-masked-grade.png)

*Draw an animated Roto mask and use it to limit a grade to part of the shot.*

### A little compositing room inside your graph.

If you work in Nuke or another compositor, these tools should feel familiar:

- **Grade:** balance blackpoint, lift, gain, and gamma with a live preview.
- **Advanced Blend:** mix two images, or control their tone and fine detail separately—useful for dialling back an upscale's sharpening.
- **Layered Images:** stack up to eight images with blend modes, opacity, and masks.
- **Rescale:** compare detail at a fixed viewing size while choosing your working resolution.
- **HDR tools:** prepare exposure brackets, merge reconstructed passes, and blend recovered shadow/highlight information into a shot.
- **EXR Layer + Cryptomatte Matte:** pull passes and object masks from a layered EXR loaded by BAT.

### Point to an object. Get a tracked mask.

**SeC Segmenter** combines an on-node points editor with video object segmentation. Mark what to include or exclude, or draw a bounding box, then generate masks through the clip. Review them as an overlay before using them downstream.

Start with the segmenter's defaults. Add **SeC Advanced Params** when you want to choose the model, tracking direction, or memory settings. **SeC needs extra dependencies and model weights**—see [SeC setup](#sec-setup).

### Find where your workflow spends time and memory.

**BAT Profiler** lives in the sidebar and works across your graph, including nodes from other packs. Enable recording, run your workflow, then sort nodes by time, RAM, or VRAM and click a result to jump to that node.

Live memory charts show usage against your machine's capacity. Saved run history and **Copy report** help investigate slow runs or crashes, including the node that was running when a connection was lost. A lost run is a clue to investigate, not proof of an out-of-memory error.

### Still working with WAN VACE 2.1? This one's for you.

**VACE Batch Tool** is one of my most-used nodes. Assemble images and masks at chosen frame positions, add or remove keyframe inputs, and build the input batch for a WAN VACE 2.1 workflow in one place.

It's a specialised helper for that workflow, with fill colour and premultiplication controls when you need them.

## All nodes

Names below match the Add Node menu, with the 🦇 prefix omitted for readability. Browse the **BAT** categories or search for a node by name.

[Loading](#loading-media) · [Video](#video-and-frames) · [Animation](#crop-animation-and-masks) · [Compositing](#colour-and-compositing) · [SeC](#sec-masking) · [WAN / VACE](#wan-and-vace) · [Helpers](#workflow-helpers) · [Graph utilities](#graph-utilities)

### Loading media

| Node | Use it to… |
| --- | --- |
| **Loader** | Open a still, numbered sequence, image folder, or movie; get EXR layers and Cryptomatte data when available. |
| **EXR Layer** | Choose a named pass from Loader's layered EXR output. |
| **Cryptomatte Matte** | Use clicked points to turn EXR object IDs into a mask. |

Loader handles frame sampling; **Video Loader** below has the visual trim controls for movies. EXR loading requires `OpenImageIO`.

### Video and frames

| Node | Use it to… |
| --- | --- |
| **Video Loader** | Load and visually trim footage, with matching audio and source alpha where available. |
| **Video Combine** | Save a frame batch as video, animation, or an image sequence, with optional audio. |
| **BAT Frame Picker** | Pick one frame from a contact sheet of a video or a numbered image sequence. |
| **Framehold** | Freeze a chosen frame across the whole batch. |
| **Video Grid Split** | Divide a clip into grid regions, with optional overlap for tiled processing. |
| **Batch Format** | Adjust a batch to a valid frame count for WAN, Hunyuan, LTX 2.3, Cosmos, or MiniMax H3. |

### Crop, animation, and masks

| Node | Use it to… |
| --- | --- |
| **Crop** | Draw a crop directly on the image and save its position for Uncrop. |
| **Animated Crop** | Keyframe the crop region through a shot. |
| **Uncrop** | Place a processed crop back into its original image or shot, including animated crops. |
| **Animated Grade** | Keyframe colour adjustments over time. |
| **Roto** | Draw and animate Bézier shapes to create masks. |
| **Grow Mask** | Expand a mask's edges by a chosen number of pixels. |
| **Erode Mask** | Shrink a mask's edges by a chosen number of pixels. |

### Colour and compositing

| Node | Use it to… |
| --- | --- |
| **Grade** | Balance blackpoint, lift, gain, and gamma with a live preview. |
| **Advanced Blend** | Blend two images, with optional separate controls for tone and detail. |
| **Layered Images** | Composite up to eight layers using 19 blend modes, opacity, and masks. |
| **Rescale** | Resize images while comparing the result against the source at a fixed viewing size. |
| **Exposure Bracket** | Create multiple exposures to feed into separate LTX SDR-to-HDR passes. |
| **Exposure Merge** | Combine those reconstructed passes into one linear HDR result. |
| **HDR Tonal Composite** | Blend an LTX HDR reconstruction into the original's shadows and highlights. |

The HDR nodes are companions to an HDR reconstruction workflow; they don't run the LTX model themselves. For HDR export, use the linear output and a suitable format such as EXR. The display preview is for viewing, not a substitute for that linear data.

### SeC masking

| Node | Use it to… |
| --- | --- |
| **SeC Segmenter** | Mark an object on a frame and track its mask through a clip, with an overlay preview. |
| **SeC Advanced Params** | Set the segmenter's model, device, tracking direction, and memory options. |

### WAN and VACE

| Node | Use it to… |
| --- | --- |
| **VACE Batch Tool** | Arrange image and mask keyframes for WAN VACE 2.1 inputs. |
| **WAN Context Calculator** | Choose frame, context, stride, and overlap settings for sliding-context WAN workflows. |
| **WAN Batch Format** | Prepare a clip for a WAN sliding-context workflow, including padding. |
| **Batch Crop** | Remove the padding from a WAN-formatted result after generation. |
| **Wan Reference Aligner** | Match a reference image or clip to a WAN-formatted batch's window layout. |

### Workflow helpers

| Node | Use it to… |
| --- | --- |
| **Points Editor** | Place labelled points on an image for workflows that accept point coordinates. |
| **Filename Prefix** | Build consistent output names from workflow context. |
| **Bypass Switch** | Make named toggles for bypassing groups of nodes, subgraphs, or backdrops. |

### Graph utilities

Small helpers for inspecting values, formatting labels, and routing batches without an extra general-purpose node pack.

| Node | Use it to… |
| --- | --- |
| **Show Any** | See a value on the node and pass it onward. Turn the readout off when you no longer need that branch to run. |
| **Show Tensor Shape** | Inspect a batch's shape, size, range, and memory use while passing it onward. |
| **Convert Any** | Convert a value to text, integer, float, or boolean. |
| **Any to String** | Render a value as readable text, JSON, or its exact representation. |
| **Number to String** | Format frame numbers or versions with padding, precision, and optional text. |
| **Compare** | Compare two values to drive a boolean input. |
| **Index Switch** | Pick one of up to ten inputs; unused branches do not run. |
| **List Length** | Count items or frames in a list or batch. |
| **List Index** | Take one item or frame from a list or batch. |
| **List Batch** | Join two lists or matching image batches. |

**Also included, without adding a node:**

- **BAT Profiler:** per-node timing and memory inspection in the sidebar.
- **Canvas zoom:** zoom further out to see larger graphs. Adjust the minimum zoom under **Settings → 🦇 BAT → Canvas**.
- **Run selected outputs:** select output nodes and press **Alt+Enter** to queue just those branches. Change the shortcut under **Settings → Keybinding**.
- **Larger editors:** maximise the interactive editor when you need more room to work, then press **Esc** to return to the graph.

## Install

### Add the pack

Place this repository in your active ComfyUI installation's `custom_nodes` folder. With Git, run:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BeeeFX/ComfyUI-BAT-NodePack.git
```

Alternatively, use GitHub's **Code → Download ZIP**, extract it, and place the extracted folder inside `custom_nodes`. The pack's `__init__.py` should be directly inside that folder, not inside a second nested copy.

**Using ComfyUI Desktop?** Use the `custom_nodes` folder in Desktop's active ComfyUI installation; the path above is an example, not a fixed Desktop location.

Restart ComfyUI, then search the Add Node menu for **Loader**, **Animated Crop**, or another name above. BAT nodes have a 🦇 prefix.

Most tools use packages already supplied by ComfyUI. Video encoding uses FFmpeg through `imageio-ffmpeg`, with a system FFmpeg fallback. Optional `decord` can speed up video seeking; the loader can work without it.

To load **EXR files and their layers**, install `OpenImageIO` in the Python environment that runs ComfyUI (`python -m pip install OpenImageIO`), then restart. Loader's still, sequence, folder, and movie support works without it. See the [technical notes](docs/technical-notes.md#loading-media) for EXR overscan options.

### SeC setup

SeC is optional: the other BAT nodes can load without its extra packages or model.

1. Install the pack's requirements **using the Python environment that runs your ComfyUI**. From the pack folder, the command is:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Restart ComfyUI.
3. On its first run, SeC downloads the default **SeC-4B-fp16** checkpoint (about **7.35 GB**) into `ComfyUI/models/sams` if no model is installed. For manual setup, see the [model locations and downloads](bat_sec/NOTICE.md#models).

To control this, connect **SeC Advanced Params** and turn off `auto_download` before running. SeC uses CUDA when available; CPU inference is very slow. Model weights and dependencies are separate from the lightweight editing tools in the pack.

## Try it

These are small starting ideas to build in your own graph:

| I want to… | Start here |
| --- | --- |
| **Trim and export a clip** | Video Loader → Video Combine. Connect the images, frame rate, and audio if needed. |
| **Load a layered EXR pass** | Loader → EXR Layer → your processing nodes. Run once, then choose the pass from EXR Layer's list. |
| **Mask an EXR object** | Loader → Cryptomatte Matte; connect Loader's images to Points Editor as a background and its points to Cryptomatte Matte. |
| **Adjust a shot's colour** | Video Loader → Grade → Video Combine. Use Animated Grade for a changing look. |
| **Process a moving region** | Animated Crop → your processing nodes → Uncrop. Also connect Animated Crop's crop information to Uncrop. |
| **Create a tracked mask** | Video Loader → SeC Segmenter. Mark the object, run, and inspect the overlay. |
| **Reduce an upscale's harshness** | Connect the upscale and original to Advanced Blend, enable frequency separation, and compare the detail mix. |
| **Investigate a slow render** | Open BAT Profiler in the sidebar, enable recording, run the graph, and sort by time. |

## Help

<details>
<summary><strong>Installed, but can't find the nodes?</strong></summary>

Check that the pack is in the active installation's `custom_nodes` folder, restart ComfyUI, and reload the interface. Look at the startup log for a BAT import error. For SeC, install requirements in ComfyUI's own environment rather than an unrelated system Python.

</details>

<details>
<summary><strong>Opening an older BAT or Volt workflow?</strong></summary>

**Batch Format** replaces the former **Video Batch Format** and **WAN Batch Frame Format** nodes. **Batch Crop** is the current display name of **WAN Batch Crop**. **BAT Frame Picker** replaces the old studio-specific **VRI Frame Picker**.

Legacy `Volt_*` workflows have a migration shim for a separate ETC migration tool. Without that companion tool, replace legacy nodes manually; installing BAT alone does not automatically migrate them.

</details>

Found a problem or have a suggestion? [Open an issue](https://github.com/BeeeFX/ComfyUI-BAT-NodePack/issues) with the node name, what you expected, and what happened. A small example workflow or screenshot helps. For performance problems, BAT Profiler's **Copy report** is a useful starting point.

Looking for implementation details? The [technical notes archive](docs/technical-notes.md) preserves the previous README's deeper explanations and development history.

---

Made by [BeeeFX](https://github.com/BeeeFX). Personal tools, shared for your workflows.

BAT code is [MIT licensed](LICENSE). The bundled SeC stack has separate [third-party attribution and licensing](bat_sec/NOTICE.md); model weights are downloaded separately.
