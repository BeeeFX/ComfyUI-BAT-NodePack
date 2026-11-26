/**
 * BAT NodePack — workflow migration registrations.
 *
 * Tells ETC_Core's migration registry which legacy `class_type` keys
 * map to which new BAT_* keys. Loaded automatically by ComfyUI via
 * the pack's WEB_DIRECTORY. When a workflow that still references the
 * old `Volt_*` keys is opened, ETC_Core's
 * `etc-node-migration.js` shows an in-UI popup offering one-click
 * replacement.
 *
 * If ETC_Core isn't installed, this script does nothing (the pack
 * still works for new workflows that already use Bat_* keys).
 */

console.log("[BAT.Migrations] module loaded");

// ─── Bat_BatchFormat merge (2026-09-02) ─────────────────────────────────────
//
// Target widget order:
//   0 model, 1 mode, 2 pad_method, 3 pad_position, 4 auto_target_frames,
//   5 target_num_frames, 6 pad_frames, 7 round_up, 8 grey_value
//
// auto_target_frames migrates as `false` in both directions: the predecessor
// nodes had no such toggle, so their typed target_num_frames is the user's
// actual intent and must keep being honoured.
//
// pad_frames migrates as 0 when the old mode was a nearest_* one. Both
// predecessors ignored it there (it was an arbitrary_pad-only field, often left
// at a stale value after a mode switch), but Bat_BatchFormat now adds it before
// the grid snap — carried across verbatim, 24 frames with a leftover 8 would
// come out 33 long instead of the 25 the old node produced.

const DEFAULT_MODEL = "WAN (4k+1)";

const padFramesFor = (oldMode, padFrames) =>
    (typeof oldMode === "string" && oldMode.startsWith("nearest")) ? 0 : padFrames;

// Old "Video Batch Format" model dropdown -> new model names.
const VIDEO_MODEL_MAP = {
    "WAN (stride 4)":     "WAN (4k+1)",
    "Hunyuan (stride 4)": "Hunyuan (4k+1)",
    "LTX (stride 8)":     "LTX 2.3 (8k+1)",
};

/**
 * Bat_VideoBatchFormat widget order was:
 *   0 model, 1 custom_temporal_stride, 2 mode, 3 pad_method, 4 pad_position,
 *   5 target_num_frames, 6 pad_frames, 7 round_up, 8 grey_value
 *
 * The custom stride is dropped — the merged node has no Custom entry, every
 * grid lives in its MODEL_SPECS table. A workflow that used Custom is resolved
 * by stride where we can (4 -> WAN, 8 -> LTX) and otherwise falls back to WAN
 * with a console warning, since silently picking the wrong grid would produce
 * a frame count the model rejects.
 */
function mapFromVideoBatchFormat(old) {
    const [model, customStride, mode, padMethod, padPosition,
           targetNumFrames, padFrames, roundUp, greyValue] = old;

    let newModel = VIDEO_MODEL_MAP[model];
    if (!newModel) {
        const stride = Number(customStride);
        if (stride === 4)      newModel = "WAN (4k+1)";
        else if (stride === 8) newModel = "LTX 2.3 (8k+1)";
        else {
            newModel = DEFAULT_MODEL;
            console.warn(
                `[BAT.Migrations] Video Batch Format used model='${model}' stride=${customStride}, ` +
                `which has no equivalent in Batch Format. Defaulted to '${DEFAULT_MODEL}' — check this node.`
            );
        }
    }

    return [newModel, mode, padMethod, padPosition, false,
            targetNumFrames, padFramesFor(mode, padFrames), roundUp, greyValue];
}

/**
 * Bat_WanBatchFrameFormat widget order was:
 *   0 mode, 1 pad_method, 2 pad_position, 3 target_num_frames,
 *   4 pad_frames, 5 round_up, 6 grey_value
 *
 * It had no model widget — the target grid was encoded in the mode value
 * ("nearest_wan_compatible" / the later "nearest_ltx23_compatible"), so the
 * model is recovered from mode and mode collapses to "nearest_compatible".
 * pad_method also loses its WAN-flavoured spelling.
 */
function mapFromWanFrameFormat(old) {
    const [mode, padMethod, padPosition, targetNumFrames,
           padFrames, roundUp, greyValue] = old;

    let newModel = DEFAULT_MODEL;
    let newMode = mode;
    if (mode === "nearest_wan_compatible") {
        newModel = "WAN (4k+1)";
        newMode = "nearest_compatible";
    } else if (mode === "nearest_ltx23_compatible") {
        newModel = "LTX 2.3 (8k+1)";
        newMode = "nearest_compatible";
    }

    const newPadMethod = padMethod === "wan_inpaint_grey" ? "grey_inpaint" : padMethod;

    return [newModel, newMode, newPadMethod, padPosition, false,
            targetNumFrames, padFramesFor(mode, padFrames), roundUp, greyValue];
}


// ─── Retiring ComfyUI-Easy-Use and comfyui-art-venture (2026-09-17) ─────────
//
// Both packs pip-install into the shared venv at node-execution time, and
// easy-use < 1.4.1 writes files anywhere on disk from `easy saveText`. The
// studio is removing them, so every node of theirs that was reachable from a
// workflow needs somewhere to land.
//
// Targets are split deliberately. Where core ComfyUI now covers the node, the
// migration points at the CORE class, not a BAT one — `StringToInt` becomes
// `ComfyNumberConvert`, the primitives become `Primitive*`. The migration
// engine only cares about class_type strings, so a target we do not own works
// exactly the same, and it is one less node for this pack to maintain.
//
// Verified against the nine studio workflows in the September audit: of all
// of these, only `easy showAnything` and `StringToInt` actually appear. The
// rest are registered so that a workflow nobody has opened yet still has a
// path forward, but their widget mappings are reasoned from the node
// definitions rather than observed on a real graph.

const PACK_CORE = "ComfyUI core";

// ─── ComfyUI_QwenVL (alexcong) → ComfyUI-QwenVL (1038lab) ───────────────────
//
// Two packs implemented the same model. The studio keeps 1038lab's because its
// `video` input is a piped IMAGE batch, where alexcong's takes a `video_path`
// STRING — a file already on disk, which cannot come from a Loader or a VAE
// decode. So the node that survives is the one that works inside a graph.
//
// The two nodes are shaped very differently (9 widgets vs 16), so this needs a
// real map rather than a pass-through. Target is the *Advanced* node: it is the
// only one of the pair that carries temperature, max_tokens and seed, all of
// which the studio workflows set.
//
// Prompt handling is the one piece of luck here. 1038lab splits the prompt into
// `preset_prompt` (a canned dropdown) + `custom_prompt` (free text), and
// AILab_QwenVL.py:368-370 reads:
//     prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
//     if custom_prompt and custom_prompt.strip(): prompt = custom_prompt.strip()
// custom_prompt wins outright when non-empty, so dropping alexcong's single
// `text` into custom_prompt reproduces the old behaviour exactly, whatever
// preset_prompt happens to be set to.

const QWEN_TARGET = "AILab_QwenVL_Advanced";

// alexcong "none"/"4bit"/"8bit" -> 1038lab's Quantization enum values.
const QWEN_QUANT = {
    "none": "None (FP16)",
    "4bit": "4-bit (VRAM-friendly)",
    "8bit": "8-bit (Balanced)",
};

// Every model alexcong offered that 1038lab also has. SkyCaptioner-V1 is the
// single one with no equivalent — it is not a Qwen-VL checkpoint at all.
const QWEN_MODELS = new Set([
    "Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-32B-Instruct", "Qwen3-VL-32B-Thinking",
]);
const QWEN_MODEL_FALLBACK = "Qwen2.5-VL-7B-Instruct";

// preset_prompt is inert once custom_prompt is set (see above), but it still
// has to hold a value the combo accepts or the node will not validate.
const QWEN_PRESET = "🖼️ Detailed Description";

const clamp = (v, lo, hi, dflt) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return dflt;
    return Math.min(hi, Math.max(lo, n));
};

/**
 * alexcong Qwen2.5VL widgets:
 *   0 text, 1 model, 2 quantization, 3 keep_model_loaded, 4 temperature,
 *   5 max_new_tokens, 6 seed, 7 control_after_generate, 8 video_path
 *
 * 1038lab AILab_QwenVL_Advanced widgets:
 *   0 model_name, 1 quantization, 2 attention_mode, 3 use_torch_compile,
 *   4 device, 5 preset_prompt, 6 custom_prompt, 7 max_tokens, 8 temperature,
 *   9 top_p, 10 num_beams, 11 repetition_penalty, 12 frame_count,
 *   13 keep_model_loaded, 14 seed, 15 control_after_generate
 */
function mapFromQwen25VL(old) {
    const [text, model, quant, keepLoaded, temperature,
           maxNewTokens, seed, controlAfterGenerate, videoPath] = old || [];

    let modelName = model;
    if (!QWEN_MODELS.has(model)) {
        console.warn(
            `[BAT.Migrations] Qwen2.5VL used model='${model}', which QwenVL ` +
            `(1038lab) does not provide. Defaulted to '${QWEN_MODEL_FALLBACK}' — ` +
            `check this node.`
        );
        modelName = QWEN_MODEL_FALLBACK;
    }

    if (videoPath && String(videoPath).trim()) {
        console.warn(
            `[BAT.Migrations] Qwen2.5VL had video_path='${videoPath}'. The ` +
            `replacement takes video as a piped IMAGE batch on its 'video' ` +
            `input, not a path — wire a loader into it.`
        );
    }

    return [
        modelName,
        QWEN_QUANT[quant] ?? "None (FP16)",
        "auto",                                   // attention_mode
        false,                                    // use_torch_compile
        "auto",                                   // device
        QWEN_PRESET,                              // preset_prompt (inert)
        text ?? "",                               // custom_prompt — wins outright
        clamp(maxNewTokens, 64, 4096, 512),       // max_tokens
        // alexcong allowed temperature 0; 1038lab's floor is 0.1.
        clamp(temperature, 0.1, 1.0, 0.6),
        0.9,                                      // top_p
        1,                                        // num_beams
        1.2,                                      // repetition_penalty
        16,                                       // frame_count
        keepLoaded ?? true,
        // alexcong's seed defaults to -1 ("random"); 1038lab's minimum is 1.
        clamp(seed, 1, 4294967295, 1),
        controlAfterGenerate ?? "fixed",
    ];
}

/**
 * Re-wire by input NAME, never by index.
 *
 * alexcong's node has `image` at slot 0, but a workflow that converted `seed`
 * to an input puts it at slot 1 — exactly where 1038lab's `video` input sits.
 * A positional map would silently land a seed link on the video socket, which
 * is the kind of failure nobody notices until the captions come back wrong.
 */
function qwenInputs(slot, name, oldNode, newNode) {
    const target = (newNode?.inputs || []).findIndex((i) => i && i.name === name);
    if (target >= 0) return target;
    console.warn(
        `[BAT.Migrations] Qwen2.5VL had a link on '${name}' (slot ${slot}) with ` +
        `no matching input on ${QWEN_TARGET}. That link is not carried over — ` +
        `reconnect it by hand.`
    );
    return -1;
}



// easy-use's two display nodes carry no widgets at all. BAT's equivalents put
// an `enabled` toggle at index 0, which has to be seeded true — otherwise a
// migrated readout comes back switched off and silently shows nothing.
const enableFirst = () => [true];

// Core's PrimitiveInt / PrimitiveFloat declare `control_after_generate`, which
// the frontend renders as a second widget; easy-use's Int/Float have only
// `value`. Pad rather than truncate: a trailing extra entry is ignored where
// the widget does not exist, while a missing one would leave the control
// unset on a node that does have it.
const padControl = (old) => [old?.[0], "fixed"];

// easy-use built 20 input slots on `anythingIndexSwitch` but clamped its index
// widget to 0-9, so only the first ten were ever reachable. Bat_IndexSwitch
// has exactly those ten. Anything wired beyond slot 9 was unreachable already,
// but say so rather than dropping it in silence.
function indexSwitchInputs(slot, name) {
    if (slot > 9) {
        console.warn(
            `[BAT.Migrations] easy anythingIndexSwitch had a link on '${name}' ` +
            `(slot ${slot}), which its own index widget could never select. ` +
            `Bat_Index Switch has 10 branches, so that link is not carried over.`
        );
    }
    return slot;
}

// ─── 🦇 Lines ← comfyui_controlnet_aux (2026-11-26) ───────────────────────────
//
// Bat_Lines widget order (append-only; tests/verify_lines.py pins it):
//   0 filter, 1 resolution, 2 output_size, 3 invert, 4 canny_low, 5 canny_high,
//   6 xdog_style, 7 xdog_sigma, 8 xdog_k, 9 xdog_sharpen, 10 xdog_threshold,
//   11 xdog_softness, 12 xdog_scribble_threshold, 13 standard_sigma,
//   14 standard_threshold, 15 teed_safe_steps, 16 hed_safe, 17 lineart_coarse,
//   18 mlsd_score, 19 mlsd_distance, 20 mlsd_line_width, 21 auto_download,
//   22 preview_frame
//
// Every migrated node gets output_size "processing": controlnet_aux returned
// its lines at the working resolution, and whatever is downstream was built
// around that size. Same weights and the same processing make Canny, XDoG
// scribble, standard lineart, lineart realistic / anime / manga, M-LSD and HED
// identical to before (tests/verify_lines*.py). TEED, PiDiNet and AnyLine
// change look, and say so in the console.

/** The full Bat_Lines widgets_values: defaults, then `over`. */
function linesValues(over) {
    const v = {
        filter: "canny", resolution: 512, output_size: "processing", invert: false,
        canny_low: 100, canny_high: 200, xdog_style: "soft", xdog_sigma: 0.8,
        xdog_k: 1.6, xdog_sharpen: 20.0, xdog_threshold: 0.3, xdog_softness: 3.0,
        xdog_scribble_threshold: 32, standard_sigma: 6.0, standard_threshold: 8.0,
        teed_safe_steps: 2, hed_safe: true, lineart_coarse: false, mlsd_score: 0.1,
        mlsd_distance: 0.1, mlsd_line_width: 1, auto_download: true, preview_frame: 0,
    };
    Object.assign(v, over || {});
    return [v.filter, v.resolution, v.output_size, v.invert, v.canny_low, v.canny_high,
            v.xdog_style, v.xdog_sigma, v.xdog_k, v.xdog_sharpen, v.xdog_threshold,
            v.xdog_softness, v.xdog_scribble_threshold, v.standard_sigma,
            v.standard_threshold, v.teed_safe_steps, v.hed_safe, v.lineart_coarse,
            v.mlsd_score, v.mlsd_distance, v.mlsd_line_width, v.auto_download,
            v.preview_frame];
}

/** controlnet_aux treats a resolution under 64 (or a non-number) as 512. */
function caResolution(v) {
    const n = Number(v);
    return Number.isFinite(n) && n >= 64 ? Math.round(n) : 512;
}

function caNumber(v, dflt) {
    const n = Number(v);
    return Number.isFinite(n) ? n : dflt;
}

function linesLookChanged(from, why) {
    console.warn(`[BAT.Migrations] ${from} → 🦇 Lines: ${why} Check the result before trusting it.`);
}

/** Old controlnet_aux values -> Bat_Lines values, by old class name. */
function mapToLines(from, old) {
    old = Array.isArray(old) ? old : [];
    switch (from) {
    case "CannyEdgePreprocessor":
        return linesValues({ filter: "canny", canny_low: caNumber(old[0], 100),
                             canny_high: caNumber(old[1], 200), resolution: caResolution(old[2]) });
    case "Scribble_XDoG_Preprocessor":
        return linesValues({ filter: "xdog", xdog_style: "scribble",
                             xdog_scribble_threshold: caNumber(old[0], 32), resolution: caResolution(old[1]) });
    case "LineartStandardPreprocessor":
        return linesValues({ filter: "lineart standard", standard_sigma: caNumber(old[0], 6.0),
                             standard_threshold: caNumber(old[1], 8), resolution: caResolution(old[2]) });
    case "LineArtPreprocessor":
        return linesValues({ filter: "lineart realistic", lineart_coarse: old[0] === "enable",
                             resolution: caResolution(old[1]) });
    case "AnimeLineArtPreprocessor":
        return linesValues({ filter: "lineart anime", resolution: caResolution(old[0]) });
    case "Manga2Anime_LineArt_Preprocessor":
        return linesValues({ filter: "lineart manga", resolution: caResolution(old[0]) });
    case "M-LSDPreprocessor":
        return linesValues({ filter: "mlsd", mlsd_score: caNumber(old[0], 0.1),
                             mlsd_distance: caNumber(old[1], 0.1), resolution: caResolution(old[2]) });
    case "HEDPreprocessor":
        return linesValues({ filter: "hed", hed_safe: old[0] !== "disable", resolution: caResolution(old[1]) });
    case "TEEDPreprocessor":
        linesLookChanged(from, "TEED now uses its author's MIT weights (controlnet_aux's copy is non-commercial), so the lines differ.");
        return linesValues({ filter: "teed", teed_safe_steps: caNumber(old[0], 2), resolution: caResolution(old[1]) });
    case "PiDiNetPreprocessor":
        linesLookChanged(from, "PiDiNet's licence is research-only, so this is now TEED soft edges.");
        return linesValues({ filter: "teed", teed_safe_steps: old[0] === "disable" ? 0 : 2,
                             resolution: caResolution(old[1]) });
    case "AnyLineArtPreprocessor_aux":
        linesLookChanged(from, "AnyLine's MTEED weights carry use restrictions, so this is now TEED soft edges.");
        return linesValues({ filter: "teed", resolution: caResolution(old[1]) });
    default:
        return null;
    }
}

/** AIO_Preprocessor: [preprocessor class name, resolution]. */
function mapFromAIO(old) {
    old = Array.isArray(old) ? old : [];
    const res = caResolution(old[1]);
    const mapped = mapToLines(String(old[0]), []);
    if (!mapped) {
        linesLookChanged("AIO_Preprocessor", `its preprocessor '${old[0]}' has no 🦇 Lines equivalent; set to Canny.`);
        return linesValues({ filter: "canny", resolution: res });
    }
    mapped[1] = res;   // AIO carries only the resolution; the rest stay at defaults
    return mapped;
}

/** Inputs by name: `resolution` may have been converted to a socket. */
function linesInputs(slot, name, oldNode, newNode) {
    const target = (newNode?.inputs || []).findIndex((i) => i && i.name === name);
    if (target >= 0) return target;
    console.warn(`[BAT.Migrations] ${oldNode?.type} had a link on '${name}' with no matching ` +
                 `input on 🦇 Lines; that link is not carried over.`);
    return -1;
}

const LINES_FROM = [
    "CannyEdgePreprocessor", "Scribble_XDoG_Preprocessor", "LineartStandardPreprocessor",
    "LineArtPreprocessor", "AnimeLineArtPreprocessor", "Manga2Anime_LineArt_Preprocessor",
    "M-LSDPreprocessor", "HEDPreprocessor", "TEEDPreprocessor", "PiDiNetPreprocessor",
    "AnyLineArtPreprocessor_aux",
];


const PACK = "BAT NodePack";
const MIGRATIONS = [
    { from: "Volt_VideoGridSplit",       to: "Bat_VideoGridSplit",       pack: PACK },
    { from: "Volt_VaceBatchTool",        to: "Bat_VaceBatchTool",        pack: PACK },
    { from: "Volt_WanContextCalculator", to: "Bat_WanContextCalculator", pack: PACK },
    { from: "Volt_WanBatchFormat",       to: "Bat_WanBatchFormat",       pack: PACK },
    { from: "Volt_WanBatchCrop",         to: "Bat_WanBatchCrop",         pack: PACK },
    // Bat_VideoBatchFormat and Bat_WanBatchFrameFormat were merged into the
    // single Bat_BatchFormat. Both had different widget orders and different
    // enum spellings, so each needs its own mapWidgetValues. Volt_* is the
    // pre-rename key for the video node — routed straight to the merged node
    // rather than hopping through a class that no longer exists.
    { from: "Volt_VideoBatchFormat",     to: "Bat_BatchFormat", pack: PACK, mapWidgetValues: mapFromVideoBatchFormat },
    { from: "Bat_VideoBatchFormat",      to: "Bat_BatchFormat", pack: PACK, mapWidgetValues: mapFromVideoBatchFormat },
    { from: "Bat_WanBatchFrameFormat",   to: "Bat_BatchFormat", pack: PACK, mapWidgetValues: mapFromWanFrameFormat },
    // ── easy-use → BAT (no core equivalent exists) ──────────────────────
    { from: "easy showAnything",        to: "Bat_ShowAny",          pack: PACK, mapWidgetValues: enableFirst },
    { from: "easy showTensorShape",     to: "Bat_ShowTensorShape",  pack: PACK, mapWidgetValues: enableFirst },
    { from: "easy convertAnything",     to: "Bat_ConvertAny",       pack: PACK },
    { from: "easy compare",             to: "Bat_Compare",          pack: PACK },
    { from: "easy lengthAnything",      to: "Bat_ListLength",       pack: PACK },
    { from: "easy indexAnything",       to: "Bat_ListIndex",        pack: PACK },
    { from: "easy batchAnything",       to: "Bat_ListBatch",        pack: PACK },
    { from: "easy anythingIndexSwitch", to: "Bat_IndexSwitch",      pack: PACK, mapInputs: indexSwitchInputs },

    // ── art-venture / easy-use → core ComfyUI ───────────────────────────
    // StringToInt's lone output is INT. ComfyNumberConvert emits FLOAT on
    // slot 0 and INT on slot 1, so the output link has to move across.
    { from: "StringToInt",     to: "ComfyNumberConvert", pack: PACK_CORE, mapOutputs: () => 1 },
    { from: "StringToNumber",  to: "ComfyNumberConvert", pack: PACK_CORE },
    { from: "BooleanPrimitive", to: "PrimitiveBoolean",  pack: PACK_CORE },
    { from: "easy string",     to: "PrimitiveString",    pack: PACK_CORE },
    { from: "easy boolean",    to: "PrimitiveBoolean",   pack: PACK_CORE },
    { from: "easy int",        to: "PrimitiveInt",       pack: PACK_CORE, mapWidgetValues: padControl },
    { from: "easy float",      to: "PrimitiveFloat",     pack: PACK_CORE, mapWidgetValues: padControl },
    // ── ComfyUI_QwenVL (alexcong) → ComfyUI-QwenVL (1038lab) ────────────
    { from: "Qwen2.5VL", to: QWEN_TARGET, pack: "ComfyUI-QwenVL (1038lab)",
      mapWidgetValues: mapFromQwen25VL, mapInputs: qwenInputs },

    // comfyui_controlnet_aux line extractors -> 🦇 Lines (2026-11-26).
    ...LINES_FROM.map((from) => ({
        from, to: "Bat_Lines", pack: PACK,
        mapWidgetValues: (old) => mapToLines(from, old), mapInputs: linesInputs,
    })),
    { from: "AIO_Preprocessor", to: "Bat_Lines", pack: PACK,
      mapWidgetValues: mapFromAIO, mapInputs: linesInputs },
];

/**
 * Register straight away if ETC_Core's engine is already up; queue otherwise.
 *
 * This used to poll for the registry and give up after 20 x 50ms. ETC_Core's
 * import chain (etc-core.js -> events -> ui -> fetch -> paths -> the migration
 * engine) can take longer than that second on a cold NFS-served load, and
 * whenever it did, every migration below was dropped for the whole session --
 * a workflow full of legacy nodes then opened with no popup at all, which is
 * indistinguishable from the pack not being installed.
 *
 * The queue is drained by etc-node-migration.js as soon as it evaluates, so
 * load order no longer matters and there is nothing to time out.
 */
function registerAll() {
    const reg = window.ETC?.registerNodeMigration;
    if (typeof reg === "function") {
        for (const m of MIGRATIONS) reg(m);
        console.log(`[BAT.Migrations] registered ${MIGRATIONS.length} migrations`);
        return true;
    }
    window.ETC = window.ETC || {};
    window.ETC.pendingNodeMigrations = window.ETC.pendingNodeMigrations || [];
    window.ETC.pendingNodeMigrations.push(...MIGRATIONS);
    console.log(`[BAT.Migrations] ETC registry not up yet; queued ${MIGRATIONS.length} migrations`);
    return false;
}

registerAll();
