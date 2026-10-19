#include "core/cache_key.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace ck {

using json = nlohmann::json;
namespace fs = std::filesystem;

static fs::path u8(const std::string &s)
{
	return fs::path((const char8_t *)s.c_str());
}

static json components_json(const KeyComponents &c)
{
	// nlohmann's default object type is a std::map: keys come out sorted, like sort_keys=True.
	return json{{"gpu_uuid", c.gpu_uuid},     {"driver_version", c.driver_version}, {"trt_version", c.trt_version},
		    {"weights_sha256", c.weights_sha256}, {"internal_w", c.internal_w},     {"internal_h", c.internal_h},
		    {"refiner", c.refiner}};
}

std::string canonical_json(const KeyComponents &c)
{
	return components_json(c).dump(); // compact: separators (",", ":")
}

std::string cache_key(const KeyComponents &c)
{
	return sha256_hex(canonical_json(c));
}

std::optional<Manifest> read_manifest(const std::string &path)
{
	std::error_code ec;
	if (!fs::exists(u8(path), ec))
		return Manifest{};
	std::ifstream f(u8(path), std::ios::binary);
	if (!f)
		return std::nullopt;
	std::stringstream ss;
	ss << f.rdbuf();
	try {
		json doc = json::parse(ss.str());
		Manifest m;
		for (const auto &e : doc.at("entries")) {
			CacheEntry ce;
			ce.key = e.at("key").get<std::string>();
			ce.comp.gpu_uuid = e.at("gpu_uuid").get<std::string>();
			ce.comp.driver_version = e.at("driver_version").get<std::string>();
			ce.comp.trt_version = e.at("trt_version").get<std::string>();
			ce.comp.weights_sha256 = e.at("weights_sha256").get<std::string>();
			ce.comp.internal_w = e.at("internal_w").get<int>();
			ce.comp.internal_h = e.at("internal_h").get<int>();
			ce.comp.refiner = e.at("refiner").get<bool>();
			ce.engine_path = e.at("engine_path").get<std::string>();
			ce.onnx_path = e.value("onnx_path", "");
			ce.created = e.value("created", "");
			ce.build_s = e.value("build_s", 0.0);
			ce.gpu_name = e.value("gpu_name", "");
			m.entries.push_back(std::move(ce));
		}
		return m;
	} catch (const std::exception &) {
		return std::nullopt;
	}
}

bool write_manifest(const std::string &path, const Manifest &m)
{
	json entries = json::array();
	for (const auto &e : m.entries) {
		json j = components_json(e.comp);
		j["key"] = e.key;
		j["engine_path"] = e.engine_path;
		j["onnx_path"] = e.onnx_path;
		j["created"] = e.created;
		j["build_s"] = e.build_s;
		j["gpu_name"] = e.gpu_name;
		entries.push_back(std::move(j));
	}
	json doc{{"schema_version", "1.0"}, {"entries", entries}};
	auto tmp = u8(path + ".tmp");
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f)
			return false;
		f << doc.dump(2) << "\n";
		if (!f)
			return false;
	}
	std::error_code ec;
	fs::rename(tmp, u8(path), ec); // MoveFileEx(REPLACE_EXISTING) on Windows: atomic on one volume
	return !ec;
}

static std::string first_difference(const KeyComponents &a, const KeyComponents &b)
{
	if (a.gpu_uuid != b.gpu_uuid)
		return "GPU changed";
	if (a.driver_version != b.driver_version)
		return "driver changed (" + a.driver_version + " -> " + b.driver_version + ")";
	if (a.trt_version != b.trt_version)
		return "TensorRT changed (" + a.trt_version + " -> " + b.trt_version + ")";
	if (a.weights_sha256 != b.weights_sha256)
		return "weights changed";
	return "key mismatch";
}

LookupResult lookup(const Manifest &m, const KeyComponents &want, bool check_file_exists)
{
	LookupResult best;
	const std::string want_key = cache_key(want);
	for (const auto &e : m.entries) {
		if (e.comp.internal_w != want.internal_w || e.comp.internal_h != want.internal_h ||
		    e.comp.refiner != want.refiner)
			continue;
		if (e.comp == want && e.key == want_key) {
			std::error_code ec;
			if (check_file_exists && !fs::exists(u8(e.engine_path), ec))
				continue;
			return LookupResult{Lookup::Hit, &e, {}};
		}
		best = LookupResult{Lookup::Stale, nullptr, first_difference(e.comp, want)};
	}
	return best;
}

} // namespace ck
