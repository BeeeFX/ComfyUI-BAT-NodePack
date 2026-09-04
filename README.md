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
| 🦇 **HDR Tonal Composite**    | `BAT/image`    | Folds an LTX HDR reconstruction (`hdr_linear` from `LTXVHDRDecodePostprocess`) back into the original plate, but only in the tonal extremes — shadows and highlights gain the recovered range and detail while the midtones and the plate's colour stay exactly put. Keyed on tone, not space, so there are no matte edges. Two-stage transfer: level and local contrast move independently, so a clipped sky can gain structure with or without gaining brightness. Because a display has nothing above white, `highlight_headroom` (in stops) buys room for recovered highlights by moving the plate's white down — set it to 0 and the plate's whites come through exactly, at the cost of seeing nothing above them. Two outputs and they live in different worlds: `linear_out` is scene-linear and **unbounded** — whatever range the HDR carried, 50, 200, 1000 — for EXR and Nuke; `image_out` is a Rec.709/sRGB monitoring image in [0,1], because that is what a preview is. Nothing truncates the linear range by default — read it in Nuke as **linear**, not as a Rec.709 display transform (an inverse ODT clamps to [0,1] and will throw the highlights away). `plate_gamma_mode` covers srgb / rec709 / gamma_2_2 / gamma_2_4 / linear and must match how the plate is read in comp — sRGB and Rec.709 differ by up to 65% in the shadows, and the thresholds are keyed in that same encoding so `shadow_start 0.12` means "where this plate reads 0.12". `output_gamma_mode` defaults to `match_plate`, so with no HDR contribution the preview returns the plate untouched. `linear_out_primaries` (rec709 / acescg / aces2065-1) converts the gamut on the way out for ACES pipelines — matrices baked from the ACES primaries and cross-checked against OCIO's built-in config, with `OCIO_CONFIG_PATH` (or `$BAT_OCIO_CONFIG` / `$OCIO`) to derive them from a specific config instead. `hdr_ai` is always taken as scene-linear (which is what LTX's `hdr_linear` always is); the node warns if it peaks below 1.0, the signature of `tonemapped` having been wired by mistake. Carries up to 8 optional `hdr_ai_*` inputs behind +/- buttons, each emitting its own image_out/linear_out pair — so several LTX reconstructions of the same plate share one set of settings instead of one copy-pasted node per version. Live on-node canvas with a viewer exposure and a display transform (auto / OCIO / sRGB / Rec.709 / 2.2 / 2.4 / raw), both display-only — set `preview_ocio_view` and the real view transform is baked from your OCIO config into a 3D LUT so the canvas matches the viewer you actually grade on, which no gamma setting can do, so thresholds can be judged against what is actually above white and on the encode your monitor uses. |
| 🦇 **Advanced Blend**         | `BAT/image`    | Blends two plates, with optional frequency separation so tone and detail move independently. Nominally a general compositing blend — nine modes (`over` / `add` / `multiply` / `screen` / `overlay` / `soft_light` / `difference` / `min` / `max`), a `mix`, an optional mask — but the case it exists for is folding an upscale back over the original. A SEEDVR2 or similar refiner returns something genuinely more detailed *and* too sharp, and a plain cross-dissolve can't separate those: at 60% you take 60% of the structure along with 60% of the crunch. Splitting each plate at `split_radius` into a low band (tone, colour, large-scale structure) and a high band (edges, texture, grain) unwelds them — `low_mix 1` / `high_mix 0.4` keeps the upscale's resolution and drops most of its sharpening; `low_mix 0` / `high_mix 1` puts *only* A's detail onto B's tone. `soften_a` / `soften_b` Gaussian-blur either plate before any of that (a blur is a lowpass, so this is "take the highs out of A first" with a radius rather than a switch — good for killing an upscaler's grain floor at source). `detail_mode` picks `subtract` (high = x − blur, added back; the classic split) or `divide` (high = x ÷ blur, multiplied back; scale-invariant, so detail transfers into shadows without crushing them and into highlights without blowing them), with `detail_limit` to bound the ratio's fireflies. `detail_gain` scales the mixed band independently of whose detail it is. With separation ON, `blend_mode` applies to the **low** bands only and the high bands are always a lerp — a signed detail band has no meaningful black or white for `multiply`/`screen`/`overlay` to key off. Resolutions never match here, so `resize_mode` (default `match_a`, i.e. B is resampled up to the upscale's size) and `resize_filter` conform them first, and every radius is in working-resolution pixels. The resamplers are implemented in-node rather than borrowed from `comfy.utils.common_upscale`, whose `lanczos` path round-trips through PIL and so clamps to [0,1] and quantises to 8-bit — which would inject quantisation straight into the low band, the one artefact this node exists not to manufacture. **Nothing truncates range**: the resamplers, the Gaussian, the band maths and the `mix` lerp are all unclamped, `clamp_output` is OFF by default, and the `detail` / `difference` outputs are never clamped at all — so a scene-linear or HDR plate passes through with its values above white intact. (`over`/`add`/`min`/`max` are range-agnostic; the other five modes are defined against white = 1.0 and stop meaning what they mean above it.) Three outputs: the blend, the applied high band on 0.5 grey, and signed A−B on 0.5 grey. The node's face is a **plain blend by default**: only `blend_mode` and `mix` are visible, with everything else folded behind a collapsible **Advanced** section that starts closed. That uses the frontend's own advanced-widget mechanism, which both renderers support in 1.49.x but read differently — litegraph checks `widget.advanced`, the Vue renderer checks `widget.options.advanced` — so the node sets both, and litegraph *serialises* `showAdvanced`, so a workflow reopens with the section as you left it. The switch itself is a DOM button at the top of the preview widget, **not** a litegraph widget, and that is load-bearing rather than stylistic: an earlier build used a `serialize = false` button spliced in under `mix`, and although litegraph's `configure()` honours that flag in both directions, **copy/paste did not** — the button consumed `resize_mode`'s slot and every value behind it shifted up one, so `frequency_separation` read `true`, `split_radius` read `NaN`, and the dropdowns went blank. At least one of the several paths that consume `widgets_values` (litegraph `configure`, the Vue `widgetValueStore`, the clipboard deserialiser, `migrateWidgetsValues`, `fallbackWidgetsValuesNames`) ignores it, so the general rule is: **never put a non-serialising widget mid-list.** A DOM widget is immune by construction — `serialize: false` *and* last, so nothing can shift behind it. Nodes saved while that bug was live are repaired on load: the node snapshots its pristine defaults in `onNodeCreated` (which runs before `configure`, so they are the definition's own values, with no table to drift) and resets anything that cannot possibly be legitimate — a non-finite number, or a combo value that is not one of its own options — logging what it touched. A wrong-but-valid value is deliberately left alone. The Python coerces the same way on its side, so a corrupted workflow from any source cannot put a `NaN` into a blur radius. Live on-node canvas with Result / A / B / Detail / Diff views, hold-to-compare-B, zoom/pan with a reset button, a pixel probe, and a hover **?** listing every interaction (hold-to-compare in particular is invisible until someone tells you about it). The preview has **two layers**. The *draft* is the browser re-running the blend on a small lossless tile — instant, so it tracks a slider, and approximate, because its radii are scaled into tile space. The *full* layer is a request to `/bat/advanced_blend/render`, which runs the real `_core` on the real plates at **full resolution** and returns a PNG of exactly the region on screen: at fit that's the whole frame rendered properly and downscaled for transport, zoomed in it's native 1:1 pixels. It is not a mirror of the render — it *is* the render, so it's strictly more correct than the draft even at fit zoom. The trade is latency, not accuracy: the draft carries the drag and the truth lands ~300 ms after you stop. Measured on an A5000, a full 4K frame is ~165 ms and a 1:1 crop ~10 ms. It costs a CPU-side cache of the conformed preview frame between runs (~190 MB for a 4K pair, LRU-capped by `CACHE_MAX_BYTES`), and it needs a re-run if the upstream graph changes. Region renders are asserted **bit-identical** to the same crop of a full-frame render, which is what proves the blur context margin is right — too small a margin is correct in the middle of a region and wrong in a band around its edge, a seam that only shows once you pan. Making the *draft* full-resolution is not an option, and not for want of tuning: the tiles are lossless 16-bit because fine grain is exactly what this node manipulates and exactly what a lossy codec discards first (JPEG at q92–q98 measured destroying 75–105% of the high band's RMS), so the payload is entropy-bound — two full-res 4K tiles would be ~110 MB per execution, and the client-side blend over 8.3 M pixels ~1.8 billion operations. Beyond resolution, three things make it read sharp: the blend composites into an offscreen buffer blitted under a zoom/pan transform (rather than a canvas stretched by `object-fit`, which was most of the old softness), smoothing is off at or above 1:1, and the canvas backing store is sized in device pixels. There is deliberately **no viewer exposure** — range still passes through unclamped, but the display budget goes on detail. The draft blends at **full tile resolution, always**, and for this node that is closer to a correctness requirement than a preference: the preview exists to show what a change does to high-frequency detail, and downscaling destroys precisely that — a smooth quarter-res preview of a sharpening change shows nothing worth seeing. Two earlier attempts traded pixels away (a fixed half-res mip, then a mip chosen adaptively from measured repaint time) and both were wrong in the same way. **The correct degradation for a detail tool is to drop frames, not pixels.** The adaptive version also penalised the common case for nothing: the Gaussians are memoised on their radius, so only `split_radius`, `soften_a` and `soften_b` recompute them — every other slider (`mix`, `low_mix`, `high_mix`, `detail_gain`, `detail_limit`, the modes) is a cheap per-pixel pass over cached blurs and was being downscaled for no reason at all. Dropping frames only works off the main thread, since a 768×432 blend at a wide radius is hundreds of milliseconds of uninterruptible typed-array loop that would freeze the graph and the slider being dragged — so the blend runs in `bat_blend_worker.js`, requests coalesced to the newest, and a slow blend costs latency rather than responsiveness. **Full** also means the canvas never changes quality while you adjust a widget: the full-resolution render already on screen is *held* while its replacement computes, rather than dropping back to the draft, so there is no flicker between two sharpnesses mid-drag. It goes briefly stale instead of briefly soft, and the badge says which. Pan and zoom are the deliberate exception — they move the *region*, so the old render genuinely no longer fits and would be drawn over the wrong part of the picture; there it does fall back to the draft. That exception is the fragile half of the rule, so `shouldHoldFullRender` is extracted and tested rather than left inline. The `drag` dropdown offers Half and Quarter for weak hardware, but nothing selects them automatically. And the server layer used to refuse to help, because pure debouncing meant it only fired once you let go — precisely when you'd stopped needing it. It's now throttled as well as debounced, paced by its own measured round trip, so it refreshes mid-drag when it can keep up; zoomed in, where a 1:1 crop costs ~10 ms, the real render effectively follows your hand. Detail and Diff carry differences of a few code values, so they get an ×1/×4/×16/×64 amplification pivoted at mid grey — inspection only, and offered on those two views alone. `tests/verify_advanced_blend.py` runs the preview's JS through quickjs against the Python and asserts the two agree, so the canvas can't silently drift from the render. |
| 🦇 **Layered Images**         | `BAT/image`    | Stacks up to 8 images with a per-layer blend mode, opacity and optional mask — what *Advanced Blend* can't do, which is more than two things at once. `image_1` is the bottom; `image_N`/`mask_N` pairs appear as you wire them, one spare always ready, and a gap in the middle just closes rather than leaving a hole. Full W3C/Photoshop mode set (19), including the four **non-separable** colour modes — hue, saturation, colour, luminosity — which reason about the pixel as a colour rather than per channel. Compositing is the real W3C form with accumulated alpha, not a chain of lerps: `co = as(1−ab)Cs + as·ab·B(Cb,Cs) + (1−as)co`. That matters at the bottom of the stack — layer 1 composites over nothing, so its mask makes it genuinely semi-transparent, and the second output is that alpha. A lerp chain would have silently baked black into those pixels. Per-layer settings live in a **layers panel** on the node body (mode, a draggable opacity bar from `bat_scrub.js`, visibility, click-a-letter to solo), persisted as JSON in one hidden STRING widget the way *Roto* stores its shapes. Deliberately not per-layer widgets: a widget list that changes shape with the input count is a widget list whose saved values shift, which is exactly what corrupted *Advanced Blend* when its toggle was a widget. The panel lists layers top-first like a compositing app, and the bottom layer's mode is disabled with an explanation, since it has nothing beneath it to blend with. Same two-layer preview as *Advanced Blend* — a draft composited in a worker for instant feedback, the real thing rendered by Python at full resolution over the visible region a moment later — with view buttons that adapt to the stack: Result, A, B, C… one per connected layer, plus Alpha. Range-agnostic, `clamp_output` off by default. Note that `normal`, `plus`, `minus`, `darken` and `lighten` are the only modes that stay meaningful above white; the rest are defined against white = 1.0. |
| 🦇 **Exposure Bracket**       | `BAT/image`    | Splits a plate into N exposures for separate LTX SDR→HDR passes. Exposure is a multiply in linear light, then re-encoded and clamped, so on a scene-linear plate exposing down genuinely reveals what sits above white. Output slots appear and disappear with `count` and are labelled with their EV; `custom_stops` takes a typed list and overrides the generated layout. **Live on-node strip** previews every stop as a thumbnail, with a big view of whichever one is selected. It answers the question the widgets can't: `count`/`spacing`/`direction` describe a bracket in the abstract, but whether it's worth rendering depends on the plate — exposing down something already clipped flat at 1.0 recovers nothing, and eight LTX generations is an expensive way to find that out. Python ships a scene-linear tile with its above-white values intact and the strip re-derives each stop locally, so a plate with real range visibly gains highlight structure as the stops go down and a clipped one visibly doesn't. A `clip` toggle flags red where a stop has hit white and blue where it has hit black, with the percentages under each thumbnail; the usual verdict is "spacing is too wide" or "there's nothing up there". `plate_gamma_mode` repaints live (the tile ships as-received, so the strip applies the decode itself). `preview_frame` picks the frame — appended last in the widget list on purpose, since ComfyUI restores `widgets_values` positionally and inserting anywhere else would shift every saved workflow's values by one. It takes effect on the next Run (which frame is shipped is Python's decision), and the strip says so rather than appearing to ignore you. |
| 🦇 **Exposure Merge**         | `BAT/image`    | Recombines those passes into one linear HDR, weighting each pixel by how well exposed it was in that pass's own input — so each region comes from the pass that saw it best, and overlapping passes average out LTX's per-pass hallucination. Auto-aligns the passes by measuring their real ratio where they overlap, rather than trusting that LTX preserved the input exposure. Input slots track the connected *Exposure Bracket*: change its `count` and the merge grows or shrinks to match, each slot labelled with the EV it expects. It resolves the bracket through Reroutes, falls back to growing one spare slot at a time when nothing is connected, and never trims a slot that has a live link. **Live on-node canvas** re-runs the merge locally from per-pass tiles, so `align` / `reference` / `well_exposed_sigma` can be judged rather than guessed. The viewer exposure is essential rather than decorative here: the output is unbounded linear, so at 0 EV you are only looking at its [0,1] slice, and "did the highlights merge sensibly" is a question about what sits above that. A **Weights** view false-colours each pass and tints every pixel by the mix contributing to it — a region in one flat colour came from one pass alone (raise `well_exposed_sigma` if you wanted averaging there), and a near-black region means every pass was badly exposed and the merge is running on its weight floor. Per-pass **solo** buttons show each pass scaled by its own alignment factor, which is how you tell a levelling problem from a weighting one, and a pixel probe reads the merged value plus each pass's weight and percentage contribution. The alignment factors the render actually measured are shipped down to the canvas and used while the widgets still match them; once you move a slider it recomputes from the tile and labels the number an estimate, because `auto` measures over the whole frame and a tile is a few hundredths of the pixels. |
| 🦇 **Rescale**                | `BAT/image`    | Changes resolution, with a live preview that **does not change size on screen**. Built for one specific judgement: how far a plate can come down before an upscale starts to suffer. An upscale done on a smaller, perfectly sharp source beats one done on a bigger, slightly soft source, so the useful move is to scale down to just before detail loss becomes visible — and nothing in ComfyUI answers that, because every other preview scales *with* the image, so halving the resolution just makes the picture on the node smaller and tells you nothing about the pixels. This viewer inverts that: the picture is held at a constant size in your eye and only its resolution changes underneath, so pulling `scale` down shows detail going away rather than a thumbnail shrinking. Structurally, not by convention — the screen box is computed from the zoom and the node's size and nothing else, so no widget on the node can move it. Sizing modes: `factor`, `long_edge`, `short_edge`, `width`, `height`, `megapixels`, and `match_reference` (the only one that may change the aspect ratio); `multiple_of` snaps both axes for models that insist on 8 / 16 / 64, which is also the only thing that can shift the ratio by a fraction of a percent. Five unclamped float resamplers (`lanczos` / `bicubic` / `bilinear` / `area` / `nearest`), all of which widen the kernel when minifying so none of them alias — an aliasing preview would lie in the direction of "you lost more than you did". `clamp_output` is OFF by default, so a scene-linear plate keeps its values above white and a lanczos undershoot below black survives for you to look at. Outputs the image plus its `width`, `height` and applied `scale`, so a downstream upscaler can be driven from the number you settled on. The viewer is region-based: `Fit` / `1:1` / `2:1` / `4:1`, wheel zoom, drag to pan, ←/→ to step `preview_frame`, a draggable **wipe** against the unscaled source, and hold **B** to see the source at any time. `magnify` picks how the rescaled region gets back to screen size and is a viewer control rather than a widget because it is a display decision: `pixels` (nearest, the default) puts one output pixel in one visible block, so you are looking at the real pixel grid rather than at a resampler's opinion of it — which is the thing actually being decided — while `smooth` (lanczos) is the round-trip test of *information*, original versus round trip at the same size, where what looks missing is what the downscale genuinely threw away. The wipe divider carries a grip you can aim at, and both the grip and its grab zone come from one function, in canvas device pixels, with the grab zone specified in *displayed* pixels so it stays the same size under your finger at any graph zoom. That is not fussiness: the first version converted pointer events with `devicePixelRatio` alone, which is exactly right at graph zoom 100% and progressively offset either side of it — litegraph scales the node with a CSS transform that `clientWidth` knows nothing about — so the handle had to be aimed *next* to rather than *at*, by up to 100 px at 50% zoom, and at far-out zooms could not be hit at all. `tests/verify_rescale.py` now asserts a click on the drawn divider hits it at every graph zoom, and that the old conversion misses wherever it drifted further than the tolerance, so the bug cannot come back quietly. The wipe is also kept a few percent off either edge, because flush to one the handle is half outside the panel and there is nothing left to grab — and, being persisted, a workflow reopened in that state came back unusable; the extremes are what the compare button's `scaled` / `original` modes are for, one click away. The status bar warns when the graph is not at 100%, because a "1:1" view scaled by the canvas is 0.7 screen pixels per source pixel and a sharpness judgement made on it is optimistic. The readout carries an RMS and a PSNR of that round trip over the visible region, because "noticeable" is a judgement and a number is a useful second opinion (the eye tends to go before ~48 dB on grain and long after it on flat CG). Two layers, as in *Advanced Blend*: a **draft** that is this file scaling the tile it already has via two `drawImage` calls, and the **truth**, a POST to `/bat/rescale/render` that runs the node's own resampler on the cached frame at full resolution. Both halves of the comparison come back as **one** stacked PNG, which is what makes the wipe trustworthy rather than a transport micro-optimisation: they are framed from one snapped region on the server, so they cannot disagree by a subpixel and show a step at the divider that reads as detail — and the wipe then costs nothing, because it never leaves the browser. PNG, never JPEG: the subject is per-pixel sharpness and a DCT codec would put its own texture exactly where we are looking. Region renders are asserted **bit-identical** to the same crop of a full-frame resize, which is what proves the kernel context margins are right. Every request is stamped with the state it was made for and a response is only applied if that state is still current, which is what makes frame stepping behave: a render in flight when the frame changes used to be painted anyway when it landed, and only *then* was the new frame requested, so stepping showed each previous frame flash up before the one you asked for. A response is evidence about a moment, and the moment can be gone by the time it arrives. Changing frames also deliberately **holds** the picture on screen rather than dropping back to the draft — the draft is the whole-frame thumbnail of the frame *Python* shipped, so falling back to it flashed the anchor frame, a different picture entirely; holding means the framing stays put and only the content updates, the way a video scrubber behaves, with the badge naming which frame the pixels actually are while the new one is in flight. The cache holds one CPU-side frame per node (~99 MB for 4K, LRU-capped) plus a **weak** reference to the whole input batch: while the upstream output is still in ComfyUI's own execution cache the viewer can scrub to any frame at full resolution with no re-run and no extra memory, and when it has been freed (`--cache-none`, a graph edit) the frame stepper marks the number with an asterisk instead of pretending — "this is frame 40" is exactly the claim you are relying on when you pick a frame to judge on. |
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

Four modules under `web/` are shared by the canvas editors rather than
duplicated per node:

- **`bat_zoom_control.js`** — display-only zoom + pan for the on-node
  preview, so you can pull back and work on a crop or roto that extends past
  the frame edge. Purely a view transform; the backend only ever sees widget
  values, never this. Carries an explicit **reset-view** button (the percentage
  readout and a double-click on the image have always reset too, but neither
  announces itself), and marks its canvas for `bat_paste_guard.js` — a
  middle-drag to pan is the gesture most likely to trigger the X11 paste bug
  described there. `state.imgW` / `state.imgH` are part of its contract, not
  optional: `zoomTo()` reads them to anchor zoom-toward-cursor, and a host that
  leaves them unset gets a wheel that drifts the image instead of zooming into
  the point under the pointer. *Advanced Blend* also uses it for the opposite reason —
  to go past 1:1 and inspect real pixels — which works because it composites
  into an offscreen buffer and blits that under the transform, rather than
  letting CSS scale a canvas. Note the module's contract is in **CSS** pixels:
  it anchors zoom-toward-cursor by reading back `state.dispScale/offX/offY`, so
  a host that computes those in device pixels will zoom toward the wrong point
  on any HiDPI display.
- **`bat_scrub.js`** — a draggable value bar, to replace `<input type="range">`
  in the on-node panels. A native range is poor in a narrow column for three
  reasons: only the ~12px thumb is grabbable, there is no pointer capture so the
  drag dies the moment you slide out of a cramped row, and resolution is however
  wide the control happens to be (100 steps through 40 px is 2.5% per pixel).
  This makes the whole bar the target, captures the pointer for the gesture, and
  adds shift-drag for fine adjustment so precision is not bounded by width —
  plus double-click to reset, wheel to step, and arrows when focused. All its
  arithmetic is normalised against `getBoundingClientRect()`, which makes it
  immune to the graph zoom; Nodes 1.0 CSS-transforms DOM widgets by the canvas
  scale, so a control working in raw pixels would drift as you zoom. Nine files
  in the pack still use a native range and would all benefit.
- **`bat_blend_modes.js`** (with `bat_blend_modes.py`) — the W3C/Photoshop blend
  modes, separable and non-separable, as the single source for both sides. Two
  things in there look like mistakes and are not: the non-separable four use luma
  weights 0.3/0.59/0.11 (the spec's, and Photoshop's — *not* Rec.709, which the
  HDR nodes correctly use for a different job), and `clipColor` writes its
  corrections as `l + (c − l) * k` with `k` forced to 0 on a neutral colour.
  That second one is load-bearing: the spec divides by a span that goes to zero
  as a colour approaches neutral, and guarding that denominator with an epsilon
  amplifies a 1e-8 spread into a ~1e-2 error — the preview and the render
  disagreed by 0.16 on such a pixel before it was written this way.
- **`bat_layered_core.js`** / **`bat_layered_worker.js`** — the layered composite,
  DOM-free and off the main thread, on the same pattern as the blend worker.
- **`bat_blend_core.js`** — the Advanced Blend maths with no DOM and no ComfyUI:
  the blend modes, the reflect-padded Gaussian, the radius-keyed blur cache, the
  band split and the view→RGBA painter. Extracted so three consumers share one
  implementation — the editor (as a fallback), the blend worker (where it
  normally runs), and `tests/verify_advanced_blend.py`, which used to work by
  slicing the top off the editor file and now just imports it.
- **`bat_blend_worker.js`** — that core, off the main thread. Tiles are copied in
  once per execution; each repaint posts params and gets back a transferred RGBA
  buffer, so the per-frame cost back to the main thread is a pointer rather than
  1.3 MB. The float buffers the pixel probe needs only come back on a settled
  frame, since shipping 8 MB per frame to service a hover would cost more than
  the blend. Entirely optional: if the worker cannot be constructed the editor
  blends synchronously instead — slower, stuttery at wide radii, but correct.
- **`bat_paste_guard.js`** — one page-wide guard against a Linux-only bug. On
  X11 the middle mouse button pastes the PRIMARY selection, which the browser
  delivers as a `paste` event on the *document*, which ComfyUI answers by
  pasting your clipboard nodes — so a middle-click inside any on-node HTML
  editor dropped a copy of them on the graph. Litegraph's own canvas is safe
  (`processMouseDown` calls `preventDefault`), which is exactly why **DOM
  widgets were the hole**: a pointerdown that lands on one never reaches that
  handler. It cannot be fixed locally either — the paste comes from the
  selection machinery, not from our event, and arrives at the document rather
  than at our element — so the guard is a capture-phase document listener
  (capture beats ComfyUI's bubble-phase handler), armed only for 400 ms around a
  middle press or release inside a marked element, and installed once per page
  rather than once per node. Coverage is automatic: `addBatDOMWidget()` marks
  everything that goes through it, so all thirteen editors that use it are
  covered along with any written later. The points editor calls `addDOMWidget`
  directly and marks itself. *Filename Prefix* is deliberately **not** marked —
  it is a text editor, and middle-click paste into a text field is a real X11
  feature someone may actually want.
- **`bat_lifecycle.js`** — `onRemoved` teardown for editor resources
  (playback intervals, `ResizeObserver`s, `IntersectionObserver`s, RAF loops,
  window-level pointer listeners). Without it, deleting a node left all of
  that running against a detached DOM indefinitely.
- **`bat_node_layout.js`** — dual-mode DOM-widget sizing, so the editors lay
  out correctly under both ComfyUI Nodes 1.0 (litegraph canvas, sized from
  `node.size`) and Nodes 2.0 (`Comfy.VueNodes.Enabled`, which computes height
  bottom-up from each widget's `computeLayoutSize()`).
- **`bat_transfer.js`** — the display transfer functions (sRGB / BT.709 /
  gamma 2.2 / 2.4 / linear, both directions) as exact mirrors of the ones in
  `bat_hdr_tonal_composite.py`, which is where the authoritative versions live.
  Any preview showing what an exposure or a grade will look like has to run the
  same curve the render will. Two asymmetries in there look like bugs and are
  not: `linear` decodes as the identity but *encodes* through sRGB (linear has
  no display encoding of its own to offer), and the forward curves are not
  clamped to 1 on the way out. Used by *Exposure Bracket* and *Exposure Merge*;
  *HDR Tonal Composite* predates it and still carries its own copies.
- **`bat_hdr_preview.js`** (with `bat_hdr_preview.py`) — high-precision
  preview transport. The usual 8-bit JPEG thumbnail makes a 16-bit render and
  an 8-bit one arrive byte-identical, and clamps away anything above white
  entirely; this ships a small zlib'd 16-bit range-normalised tile alongside
  it so the preview maths runs on the source's real values. Used by *Grade*,
  *Animated Grade*, *HDR Tonal Composite*, *Advanced Blend*, *Exposure Bracket*
  and *Exposure Merge*. Several of those could not have a meaningful preview at
  all without it, since their whole subject is values a JPEG cannot represent —
  the bracket strip in particular exists to show whether there is anything above
  white to recover, and a clamped source answers that wrongly, always
  cheerfully, in the direction of "yes there is".

---

## Tests

`tests/` holds equivalence checks for the nodes whose live preview is a second
implementation of the render. Those nodes carry an implicit promise — that what
the artist dials in on the node body is what Run will produce — and two
hand-written implementations of the same algorithm drift the first time someone
edits one side. So the tests run both and diff them.

There is no Node.js on the render machines, but the `quickjs` wheel embeds a
full ES2020 engine in-process, which is enough: the numeric half of each
extension is pure, DOM-free, and can be sliced out of the file and evaluated
directly (sliced, not copied — a copy would drift exactly the way the test
exists to prevent).

```bash
pip install quickjs
python tests/verify_advanced_blend.py
python tests/verify_exposure_bracket.py
python tests/verify_rescale.py
```

`verify_advanced_blend.py` covers 43 parameter combinations across both detail
modes, every blend mode, masked and unmasked, SDR and HDR input. Tolerance is
1e-4 absolute; the observed worst case is ~1e-5, which is float32 noise (Python
accumulates the Gaussian in float32, JS in float64).

`verify_layered_images.py` covers all 19 blend modes over a grid that includes
every value the formulas branch on plus values outside [0,1], then 304 composite
configurations (19 modes × depths 1/2/3/5 × masked/opacity/disabled variants),
the alpha output, region independence, and `parse_layers` against malformed
state. Its tolerance is **depth-scaled**, which is a measurement rather than a
convenience: Python accumulates in float32 and JS in float64, so error compounds
per layer — 0, 3.7e-6, 1.6e-5, 1.1e-4, 1.75e-4 at depths 1/2/3/5/8, affecting one
pixel in 391. A single layer is required to be near-exact, and that is the
assertion carrying the weight: it proves the nineteen formulas are identical,
because a real difference shows at depth 1 as a residual orders of magnitude
larger. Writing this test first caught two genuine bugs in the Python before any
UI existed — a `ClipColor` that applied both corrections from the original colour
instead of sequentially, and the epsilon amplification described above.

`verify_exposure_bracket.py` covers the transfer functions in both directions
across all five gamma modes, `exposeToSdr` over a range of stops on input well
above white, and the merge's weighting, alignment and accumulation over 216
configurations (every `align` mode, both `reference` modes, several sigmas, 1–5
passes). Its tolerance is **relative**, which matters: a scene-linear plate
decoded through the sRGB curve reaches 1.5e6 on that test's inputs, where
float32's own epsilon is 0.18, so an absolute bound fails on arithmetic that is
perfectly correct. Both sides agree to ~1e-7 relative on the transfer functions
and ~2e-5 on the merge — the latter being float32 accumulation divided by a
weight total that legitimately gets down to 0.03.

Both scripts also evaluate their whole extension with the imports stubbed and
drive `beforeRegisterNodeDef`, so a syntax error or stale identifier in the
editor half fails here rather than showing up as a blank node in the browser.

`verify_rescale.py` covers three claims that are load-bearing enough to be
worth evidence rather than an argument. That a **region render is the render** —
asking the preview endpoint for a rectangle of destination pixels returns
exactly what cropping that rectangle out of a full-frame resize would, over
every filter, minifying and enlarging, at the frame edges and in the middle. Get
the kernel's context margin wrong and the result is correct in the interior and
wrong in a band around it, a seam that only appears once you pan, on the very
tool you are using to judge sharpness. That this module's resampler **is** the
pack's resampler, by diffing whole frames against `bat_advanced_blend._resize`.
And that the size readout is the size rendered, by driving `planSize()` in
quickjs against `plan_size()` over 5675 cases — including the rounding edges,
which is why both sides spell the rounding out as `floor(x + 0.5)` rather than
using their language's default (Python's `round()` is banker's rounding and JS's
`Math.round` is not, so a factor landing exactly on .5 would disagree between
the readout and the file that renders it). Box and nearest come out at exactly
zero; the interpolating kernels sit at ~2.5e-5, which is float32 matmul noise
between two different matrix shapes.

That test also documents one deliberate divergence from *Advanced Blend*: its
`area` box is closed on both sides (`|x| < 0.5`), so an output centre landing
exactly on an input pixel boundary — every third sample of a 2:3 resize, so
480→720 hits it constantly — zeroes both taps and falls through to a nearest
fallback whose rounding is not shift-invariant. *Rescale*'s box is half-open,
which removes the case rather than papering over it. The divergence is
**asserted** rather than skipped, so if that node is ever fixed the test says so
instead of quietly passing. (It is a live, if minor, latent bug in Advanced
Blend's `area` resize_filter when it is enlarging a plate; left alone rather
than changed under a node this one does not own.)

One thing the exposure test deliberately does not assert: `align="auto"`
measures the ratio between two passes over the whole frame in the render and
over a 256px tile in the preview, so those two cannot agree by construction.
The test drives both sides with the same buffer to isolate the arithmetic, which
is the part that can drift. The tile-vs-frame gap is why the render's real
alignment factors are shipped down to the canvas, and why the canvas labels the
number an estimate as soon as a widget makes it recompute.

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
