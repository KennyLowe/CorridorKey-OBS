// Machine-level plugin config (data-model "Machine-level plugin config"): not per filter.
// Stored as JSON in OBS's module config folder (obs_module_config_path("config.json")).
// Thread-safe.
#pragma once

#include <string>
#include <vector>

namespace ck::config {

void init(const std::string &path);
std::string licence_acknowledged(const std::string &model_id); // ISO date, "" if not
void acknowledge_licence(const std::string &model_id);
// seconds since the epoch of the last check-models run (0 = never)
long long last_model_check();
void set_model_check(long long when, const std::vector<std::string> &newer);
std::vector<std::string> newer_models();

} // namespace ck::config
