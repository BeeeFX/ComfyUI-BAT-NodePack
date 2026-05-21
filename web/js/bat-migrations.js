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

import { app } from "/scripts/app.js";

app.registerExtension({
    name: "BAT.Migrations",
    async setup() {
        const reg = window.ETC?.registerNodeMigration;
        if (typeof reg !== "function") {
            // ETC_Core not installed (or loaded later than expected).
            // BAT still works for fresh workflows; legacy workflows
            // would just render the old nodes as missing.
            return;
        }
        const PACK = "BAT NodePack";
        reg({ from: "Volt_VideoGridSplit",       to: "Bat_VideoGridSplit",       pack: PACK });
        reg({ from: "Volt_VaceBatchTool",        to: "Bat_VaceBatchTool",        pack: PACK });
        reg({ from: "Volt_WanContextCalculator", to: "Bat_WanContextCalculator", pack: PACK });
        reg({ from: "Volt_WanBatchFormat",       to: "Bat_WanBatchFormat",       pack: PACK });
        reg({ from: "Volt_WanBatchCrop",         to: "Bat_WanBatchCrop",         pack: PACK });
        reg({ from: "Volt_VideoBatchFormat",     to: "Bat_VideoBatchFormat",     pack: PACK });
    },
});
