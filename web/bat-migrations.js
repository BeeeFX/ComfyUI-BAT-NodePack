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

const PACK = "BAT NodePack";
const MIGRATIONS = [
    { from: "Volt_VideoGridSplit",       to: "Bat_VideoGridSplit",       pack: PACK },
    { from: "Volt_VaceBatchTool",        to: "Bat_VaceBatchTool",        pack: PACK },
    { from: "Volt_WanContextCalculator", to: "Bat_WanContextCalculator", pack: PACK },
    { from: "Volt_WanBatchFormat",       to: "Bat_WanBatchFormat",       pack: PACK },
    { from: "Volt_WanBatchCrop",         to: "Bat_WanBatchCrop",         pack: PACK },
    { from: "Volt_VideoBatchFormat",     to: "Bat_VideoBatchFormat",     pack: PACK },
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
