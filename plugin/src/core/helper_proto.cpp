#include "core/helper_proto.hpp"

#include <nlohmann/json.hpp>

namespace ck {

using json = nlohmann::json;

static std::string str(const json &j, const char *k)
{
	auto it = j.find(k);
	return it != j.end() && it->is_string() ? it->get<std::string>() : std::string();
}

std::optional<HelperEvent> parse_helper_line(const std::string &line)
{
	auto first = line.find_first_not_of(" \t\r\n");
	if (first == std::string::npos || line[first] != '{')
		return std::nullopt;
	try {
		json j = json::parse(line);
		if (!j.is_object())
			return std::nullopt;
		HelperEvent e;
		e.raw = line;
		const std::string ev = str(j, "event");
		if (ev == "progress") {
			e.kind = HelperEvent::Kind::Progress;
			e.stage = str(j, "stage");
			e.pct = j.at("pct").get<int>();
			e.msg = str(j, "msg");
			if (j.contains("bytes"))
				e.bytes = j["bytes"].get<int64_t>();
			if (j.contains("total"))
				e.total = j["total"].get<int64_t>();
			return e;
		}
		if (ev == "result") {
			e.kind = HelperEvent::Kind::Result;
			e.ok = j.at("ok").get<bool>();
			e.error = str(j, "error");
			e.msg = str(j, "msg");
			e.key = str(j, "key");
			e.engine_path = str(j, "engine_path");
			e.trt_version = str(j, "trt_version");
			if (j.contains("build_s") && j["build_s"].is_number())
				e.build_s = j["build_s"].get<double>();
			e.weights_path = str(j, "weights_path");
			e.sha256 = str(j, "sha256");
			e.model_id = str(j, "model_id");
			e.current = str(j, "current");
			if (j.contains("newer") && j["newer"].is_array())
				for (const auto &n : j["newer"])
					if (n.is_object())
						e.newer.push_back(str(n, "model_id"));
			return e;
		}
	} catch (const std::exception &) {
	}
	return std::nullopt;
}

} // namespace ck
