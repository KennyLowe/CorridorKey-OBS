// Engine cache key, manifest and staleness (data-model "EngineCacheEntry").
//
// The key is sha256 over the canonical JSON of the key components (keys sorted, no spaces),
// exactly what Python's json.dumps(c, sort_keys=True, separators=(",", ":")) produces, so the
// plugin and `ckbench prepare` agree (shared fixture: tests/data/manifest_fixture.json).
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ck {

std::string sha256_hex(const void *data, size_t len);
std::string sha256_hex(const std::string &s);
// Streams the file; nullopt if it can't be read.
std::optional<std::string> sha256_file(const std::string &path);

struct KeyComponents {
	std::string gpu_uuid;       // "GPU-xxxxxxxx-..." as NVML reports it
	std::string driver_version; // "576.80"
	std::string trt_version;    // "10.16.1.11"
	std::string weights_sha256;
	int internal_w = 0;
	int internal_h = 0;
	bool refiner = false;
	bool operator==(const KeyComponents &) const = default;
};

std::string canonical_json(const KeyComponents &c);
std::string cache_key(const KeyComponents &c);

struct CacheEntry {
	std::string key;
	KeyComponents comp;
	std::string engine_path;
	std::string onnx_path;
	std::string created;
	double build_s = 0;
	std::string gpu_name;
};

struct Manifest {
	std::vector<CacheEntry> entries;
};

// Missing file -> empty manifest. Corrupt file -> nullopt (the caller treats it as empty and
// re-prepares; the helper rewrites it atomically).
std::optional<Manifest> read_manifest(const std::string &path);
// Atomic: writes <path>.tmp then renames over <path>.
bool write_manifest(const std::string &path, const Manifest &m);

enum class Lookup { Hit, Miss, Stale };
struct LookupResult {
	Lookup kind = Lookup::Miss;
	const CacheEntry *entry = nullptr; // set for Hit
	std::string why;                   // for Stale: which component differs
};
// Finds the entry for `want` (same size/refiner/weights). Hit only if every key component matches
// the current environment, the stored key is consistent, and the engine file exists. An entry for
// the same quality and weights whose GPU/driver/TensorRT differ is reported as Stale.
LookupResult lookup(const Manifest &m, const KeyComponents &want, bool check_file_exists = true);

} // namespace ck
