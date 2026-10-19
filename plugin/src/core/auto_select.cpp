#include "core/auto_select.hpp"

#include <algorithm>
#include <limits>

#include <nlohmann/json.hpp>

namespace ck {

using json = nlohmann::json;

const std::vector<Quality> &all_qualities()
{
	static const std::vector<Quality> v = {
		{"512_r0", 512, 512, false},       {"512_r1", 512, 512, true},       {"768x448_r0", 768, 448, false},
		{"768x448_r1", 768, 448, true},    {"768_r0", 768, 768, false},      {"768_r1", 768, 768, true},
		{"1024x576_r0", 1024, 576, false}, {"1024x576_r1", 1024, 576, true}, {"1024_r0", 1024, 1024, false},
		{"1024_r1", 1024, 1024, true},
	};
	return v;
}

std::optional<Quality> quality_from_id(const std::string &id)
{
	for (const auto &q : all_qualities())
		if (q.id == id)
			return q;
	return std::nullopt;
}

std::string quality_label(const Quality &q)
{
	return std::to_string(q.w) + "\xC3\x97" + std::to_string(q.h) + (q.refiner ? ", refiner on" : ", refiner off");
}

std::string live_mode_id(int height, double fps)
{
	int res = height <= 720 ? 720 : height <= 1080 ? 1080 : 2160;
	int rate = fps <= 20.5 ? 20 : fps <= 30.5 ? 30 : 60;
	if (res == 720 && rate == 60)
		res = 1080;
	if (res != 720 && rate == 20)
		rate = 30;
	return (res == 2160 ? std::string("2160p") : std::to_string(res) + "p") + std::to_string(rate);
}

std::optional<PerfTable> parse_perf_table(const std::string &text)
{
	try {
		json doc = json::parse(text);
		if (doc.at("schema_version").get<std::string>() != "1.0")
			return std::nullopt;
		PerfTable t;
		t.generated_from = doc.at("generated_from").get<std::vector<std::string>>();
		for (const auto &g : doc.at("gpus")) {
			PerfGpu pg;
			pg.gpu_name = g.at("gpu_name").get<std::string>();
			pg.driver = g.at("driver").get<std::string>();
			pg.run_id = g.at("run_id").get<std::string>();
			for (const auto &m : g.at("modes")) {
				PerfMode pm;
				pm.mode_id = m.at("mode_id").get<std::string>();
				pm.budget_ms = m.at("budget_ms").get<double>();
				for (const auto &c : m.at("configs")) {
					PerfConfig pc;
					pc.quality = c.at("quality").get<std::string>();
					pc.p95_ms_load = c.at("p95_ms_load").get<double>();
					pc.fits = c.at("fits").get<bool>();
					if (quality_from_id(pc.quality))
						pm.configs.push_back(pc);
				}
				pg.modes.push_back(std::move(pm));
			}
			t.gpus.push_back(std::move(pg));
		}
		return t;
	} catch (const std::exception &) {
		return std::nullopt;
	}
}

static int pixels(const std::string &id)
{
	auto q = quality_from_id(id);
	return q ? q->pixels() : 0;
}

// Larger is better: more pixels, then lower p95.
static bool better(const PerfConfig &a, const PerfConfig &b)
{
	if (pixels(a.quality) != pixels(b.quality))
		return pixels(a.quality) > pixels(b.quality);
	return a.p95_ms_load < b.p95_ms_load;
}

AutoChoice auto_select(const PerfTable &t, const std::string &gpu_name, const std::string &mode_id,
		       const std::set<std::string> &prepared)
{
	AutoChoice c;
	const PerfMode *mode = nullptr;
	for (const auto &g : t.gpus)
		if (g.gpu_name == gpu_name)
			for (const auto &m : g.modes)
				if (m.mode_id == mode_id)
					mode = &m;

	if (!mode || mode->configs.empty()) {
		c.unknown_gpu = true;
		c.best_overall = "512_r0";
		c.quality = prepared.count("512_r0") ? "512_r0" : "";
		c.needs_prepare = c.quality.empty();
		c.note = "No measured results for " + gpu_name + " at " + mode_id +
			 ": using the fastest setting (512\xC3\x97" "512, refiner off). It may not keep up.";
		return c;
	}

	const PerfConfig *best = nullptr, *fastest = nullptr;
	for (const auto &pc : mode->configs) {
		if (pc.fits && (!best || better(pc, *best)))
			best = &pc;
		if (!fastest || pc.p95_ms_load < fastest->p95_ms_load)
			fastest = &pc;
	}
	if (!best) {
		best = fastest;
		c.note = "No measured setting fits " + mode_id + " on this GPU; using the fastest.";
	}
	c.best_overall = best->quality;

	// What to run now: the best prepared setting that fits; else the fastest prepared one.
	const PerfConfig *run = nullptr;
	for (const auto &pc : mode->configs)
		if (pc.fits && prepared.count(pc.quality) && (!run || better(pc, *run)))
			run = &pc;
	if (!run)
		for (const auto &pc : mode->configs)
			if (prepared.count(pc.quality) && (!run || pc.p95_ms_load < run->p95_ms_load))
				run = &pc;
	if (run)
		c.quality = run->quality;
	c.needs_prepare = !prepared.count(c.best_overall);
	return c;
}

} // namespace ck
