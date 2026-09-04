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

const DEFAULT_MODEL = "WAN (4k+1)";

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
            targetNumFrames, padFrames, roundUp, greyValue];
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
            targetNumFrames, padFrames, roundUp, greyValue];
}

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
];

function registerAll() {
    const reg = window.ETC?.registerNodeMigration;
    if (typeof reg !== "function") return false;
    for (const m of MIGRATIONS) reg(m);
    console.log(`[BAT.Migrations] registered ${MIGRATIONS.length} migrations`);
    return true;
}

// Try immediately (covers the case where ETC_Core's module ran first).
if (!registerAll()) {
    // ETC_Core's etc-node-migration.js hasn't executed yet. Retry on a
    // microtask, then on a few short timeouts. Module load order between
    // unrelated extensions isn't guaranteed.
    console.log("[BAT.Migrations] ETC registry not ready, will retry");
    let attempts = 0;
    const tick = () => {
        if (registerAll()) return;
        if (++attempts < 20) setTimeout(tick, 50);
        else console.warn("[BAT.Migrations] ETC_Core never appeared; migrations not registered");
    };
    queueMicrotask(tick);
}
