#include <catch2/catch_test_macros.hpp>

#include "core/auto_select.hpp"

using namespace ck;

static const char *kTable = R"({
  "schema_version": "1.0",
  "generated_from": ["run-3060"],
  "gpus": [
    {"gpu_name": "NVIDIA GeForce RTX 3060", "driver": "576.80", "run_id": "run-3060",
     "modes": [
       {"mode_id": "1080p30", "budget_ms": 33.333,
        "configs": [
          {"quality": "512_r0", "p95_ms_load": 15.7, "fits": true},
          {"quality": "512_r1", "p95_ms_load": 28.1, "fits": true},
          {"quality": "768x448_r0", "p95_ms_load": 19.5, "fits": true},
          {"quality": "768x448_r1", "p95_ms_load": 30.6, "fits": true},
          {"quality": "768_r0", "p95_ms_load": 32.8, "fits": true},
          {"quality": "1024x576_r0", "p95_ms_load": 30.1, "fits": true},
          {"quality": "1024_r0", "p95_ms_load": 54.1, "fits": false}
        ]},
       {"mode_id": "1080p60", "budget_ms": 16.667,
        "configs": [
          {"quality": "512_r0", "p95_ms_load": 15.7, "fits": true},
          {"quality": "768x448_r0", "p95_ms_load": 19.5, "fits": false}
        ]},
       {"mode_id": "2160p60", "budget_ms": 16.667,
        "configs": [
          {"quality": "512_r0", "p95_ms_load": 32.1, "fits": false},
          {"quality": "768x448_r0", "p95_ms_load": 35.0, "fits": false}
        ]}
     ]}
  ]
})";

TEST_CASE("quality ids", "[auto]")
{
	CHECK(all_qualities().size() == 10);
	auto q = quality_from_id("1024x576_r1");
	REQUIRE(q);
	CHECK(q->w == 1024);
	CHECK(q->h == 576);
	CHECK(q->refiner);
	auto s = quality_from_id("768_r0");
	REQUIRE(s);
	CHECK(s->w == 768);
	CHECK(s->h == 768);
	CHECK_FALSE(quality_from_id("2048_r0"));
	CHECK_FALSE(quality_from_id("auto"));
	CHECK(quality_label(*q) == "1024\xC3\x97" "576, refiner on");
}

TEST_CASE("live mode detection", "[auto]")
{
	CHECK(live_mode_id(1080, 30) == "1080p30");
	CHECK(live_mode_id(1080, 29.97) == "1080p30");
	CHECK(live_mode_id(1080, 59.94) == "1080p60");
	CHECK(live_mode_id(720, 20) == "720p20");
	CHECK(live_mode_id(720, 30) == "720p30");
	CHECK(live_mode_id(720, 60) == "1080p60");
	CHECK(live_mode_id(1080, 20) == "1080p30");
	CHECK(live_mode_id(2160, 30) == "2160p30");
	CHECK(live_mode_id(2160, 60) == "2160p60");
	CHECK(live_mode_id(1440, 30) == "2160p30");
	CHECK(live_mode_id(480, 30) == "720p30");
	CHECK(live_mode_id(960, 25) == "1080p30");
}

TEST_CASE("perf table parses", "[auto]")
{
	auto t = parse_perf_table(kTable);
	REQUIRE(t);
	REQUIRE(t->gpus.size() == 1);
	CHECK(t->gpus[0].modes.size() == 3);
	CHECK(t->gpus[0].modes[0].configs[5].quality == "1024x576_r0");
	CHECK_FALSE(parse_perf_table("{not json"));
	CHECK_FALSE(parse_perf_table(R"({"schema_version":"1.0"})"));
}

TEST_CASE("auto picks the largest fitting prepared setting", "[auto]")
{
	auto t = *parse_perf_table(kTable);
	const std::string g = "NVIDIA GeForce RTX 3060";
	auto all = std::set<std::string>{"512_r0", "512_r1", "768x448_r0", "768x448_r1", "768_r0", "1024x576_r0", "1024_r0"};
	auto c = auto_select(t, g, "1080p30", all);
	// 1024x576 (589,824 px) beats 768x768 (589,824 px)? equal pixels -> lowest p95 wins
	CHECK(c.quality == "1024x576_r0");
	CHECK(c.best_overall == "1024x576_r0");
	CHECK_FALSE(c.needs_prepare);
	CHECK_FALSE(c.unknown_gpu);

	c = auto_select(t, g, "1080p60", all);
	CHECK(c.quality == "512_r0");
}

TEST_CASE("auto never picks an unprepared setting but offers the better one", "[auto]")
{
	auto t = *parse_perf_table(kTable);
	const std::string g = "NVIDIA GeForce RTX 3060";
	auto c = auto_select(t, g, "1080p30", {"512_r0", "768x448_r0"});
	CHECK(c.quality == "768x448_r0");
	CHECK(c.best_overall == "1024x576_r0");
	CHECK(c.needs_prepare);

	c = auto_select(t, g, "1080p30", {});
	CHECK(c.quality.empty());
	CHECK(c.best_overall == "1024x576_r0");
	CHECK(c.needs_prepare);

	// prepared but too slow for the mode: run the fastest prepared one
	c = auto_select(t, g, "1080p30", {"1024_r0"});
	CHECK(c.quality == "1024_r0");
	c = auto_select(t, g, "1080p30", {"1024_r0", "512_r1"});
	CHECK(c.quality == "512_r1");
}

TEST_CASE("nothing fits the mode: fastest measured setting", "[auto]")
{
	auto t = *parse_perf_table(kTable);
	auto c = auto_select(t, "NVIDIA GeForce RTX 3060", "2160p60", {"512_r0", "768x448_r0"});
	CHECK(c.best_overall == "512_r0");
	CHECK(c.quality == "512_r0");
	CHECK_FALSE(c.note.empty());
}

TEST_CASE("unknown GPU or unmeasured mode -> 512_r0 with a warning", "[auto]")
{
	auto t = *parse_perf_table(kTable);
	auto c = auto_select(t, "NVIDIA RTX A2000 12GB", "1080p30", {"512_r0", "1024x576_r0"});
	CHECK(c.unknown_gpu);
	CHECK(c.quality == "512_r0");
	CHECK(c.best_overall == "512_r0");
	CHECK_FALSE(c.note.empty());

	c = auto_select(t, "NVIDIA RTX A2000 12GB", "1080p30", {"1024x576_r0"});
	CHECK(c.quality.empty());
	CHECK(c.needs_prepare);

	c = auto_select(t, "NVIDIA GeForce RTX 3060", "720p20", {"512_r0"});
	CHECK(c.unknown_gpu); // no measurement for that mode on this GPU
	CHECK(c.quality == "512_r0");
}
