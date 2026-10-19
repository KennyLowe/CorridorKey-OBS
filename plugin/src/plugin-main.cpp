/*
corridorkey-obs: live CorridorKey green-screen keying filter for OBS Studio
Copyright (C) 2026 Kenny Lowe

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>

#include "ck-filter.hpp"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "CorridorKey live green-screen keying. Powered by CorridorKey by Corridor Digital "
	       "(CC BY-NC-SA 4.0). Not affiliated.";
}

bool obs_module_load(void)
{
	ck::register_filter();
	blog(LOG_INFO, "[corridorkey] loaded (version %s)", PLUGIN_VERSION);
	return true;
}

void obs_module_post_load(void)
{
	ck::register_websocket_vendor();
}

void obs_module_unload(void)
{
	ck::shutdown_filters();
	blog(LOG_INFO, "[corridorkey] unloaded");
}
