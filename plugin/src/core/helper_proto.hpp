// JSON-lines protocol of the ckbench helper commands (contracts/prepare-cli.md):
// prepare, fetch-weights and check-models all emit {"event":"progress",...} lines and exactly one
// final {"event":"result",...} line. Pure C++ (unit-tested).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ck {

struct HelperEvent {
	enum class Kind { Progress, Result } kind = Kind::Progress;
	// progress
	std::string stage;
	int pct = 0;
	std::string msg;
	int64_t bytes = -1, total = -1; // fetch-weights only
	// result
	bool ok = false;
	std::string error; // error code when !ok
	std::string key, engine_path, trt_version;  // prepare
	double build_s = 0;
	std::string weights_path, sha256, model_id; // fetch-weights
	std::string current;                        // check-models
	std::vector<std::string> newer;             // check-models: newer model ids
	std::string raw;                            // the line itself
};

// nullopt for a blank, non-JSON or unknown line (ignored by the client, logged for diagnosis).
std::optional<HelperEvent> parse_helper_line(const std::string &line);

} // namespace ck
