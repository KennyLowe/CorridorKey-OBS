// Quality settings and the Auto rule (research D7, data-model "PerfTable").
#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ck {

struct Quality {
	std::string id; // "1024x576_r0"
	int w = 0, h = 0;
	bool refiner = false;
	int pixels() const { return w * h; }
};

// The 10 measured settings, lightest first.
const std::vector<Quality> &all_qualities();
std::optional<Quality> quality_from_id(const std::string &id);
std::string quality_label(const Quality &q); // "1024×576, refiner off"

// Live mode for a source: camera class by height (<=720, <=1080, else 2160) and fps class
// (<=20, <=30, else 60). Combinations the benchmark didn't define map to the next heavier
// measured mode (720p60 -> 1080p60, 1080p20 -> 1080p30, 2160p20 -> 2160p30).
std::string live_mode_id(int height, double fps);

struct PerfConfig {
	std::string quality;
	double p95_ms_load = 0;
	bool fits = false;
};
struct PerfMode {
	std::string mode_id;
	double budget_ms = 0;
	std::vector<PerfConfig> configs;
};
struct PerfGpu {
	std::string gpu_name, driver, run_id;
	std::vector<PerfMode> modes;
};
struct PerfTable {
	std::vector<std::string> generated_from;
	std::vector<PerfGpu> gpus;
};

// Parses perf-table.json (contracts/perf-table.schema.json). nullopt on malformed input.
std::optional<PerfTable> parse_perf_table(const std::string &json_text);

struct AutoChoice {
	std::string quality;      // what to run now ("" = nothing prepared yet)
	std::string best_overall; // the best setting for this GPU/mode if everything were prepared
	bool unknown_gpu = false; // no measured results for this GPU: fastest setting + warning
	bool needs_prepare = false; // best_overall isn't prepared (offer to prepare it)
	std::string note;         // human-readable explanation
};

// Auto rule: exact gpu_name match; among settings that fit the mode's budget, the largest
// internal pixel count (ties: lowest p95). Only *prepared* settings are chosen to run; if no
// prepared setting fits, the fastest prepared one runs. Unknown GPU -> 512_r0 + warning.
AutoChoice auto_select(const PerfTable &t, const std::string &gpu_name, const std::string &mode_id,
		       const std::set<std::string> &prepared);

} // namespace ck
