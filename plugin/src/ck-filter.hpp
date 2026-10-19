// The "CorridorKey" OBS video filter (contracts/filter-ui.md, data-model "FilterSettings").
#pragma once

#include <obs-module.h>

namespace ck {

void register_filter();
// obs-websocket vendor requests (e2e tests and diagnostics): "corridorkey" / GetFilterStatus.
void register_websocket_vendor();
// Joins workers of destroyed filters (called from obs_module_unload).
void shutdown_filters();

} // namespace ck
