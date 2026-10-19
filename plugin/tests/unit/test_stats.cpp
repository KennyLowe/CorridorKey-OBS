#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/stats.hpp"

using namespace ck;
using Catch::Approx;

TEST_CASE("percentile matches numpy linear interpolation", "[stats]")
{
	std::deque<double> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	CHECK(percentile(v, 50) == Approx(5.5));
	CHECK(percentile(v, 95) == Approx(9.55));
	CHECK(percentile(v, 0) == Approx(1));
	CHECK(percentile(v, 100) == Approx(10));
	CHECK(percentile({}, 50) == 0);
	CHECK(percentile({4.0}, 95) == Approx(4.0));
}

TEST_CASE("worker time median and p95 over a rolling window", "[stats]")
{
	Stats s(120);
	for (int i = 0; i < 1000; i++)
		s.add_worker_ms(100.0); // pushed out of the window
	for (int i = 1; i <= 120; i++)
		s.add_worker_ms((double)i);
	auto r = s.readout(33.333);
	CHECK(r.samples == 120);
	CHECK(r.worker_ms_median == Approx(60.5));
	CHECK(r.worker_ms_p95 == Approx(114.05));
	CHECK(r.behind); // 60.5 ms > 33.3 ms
}

TEST_CASE("skipped rate counts the window's frames", "[stats]")
{
	Stats s(100);
	for (int i = 0; i < 100; i++)
		s.add_frame(true, i % 50 == 0 ? 1 : 0); // 2 skipped of 100
	auto r = s.readout(33.333);
	CHECK(r.submitted == 100);
	CHECK(r.skipped == 2);
	CHECK(r.skipped_pct == Approx(2.0));
	for (int i = 0; i < 100; i++)
		s.add_frame(true, 0);
	r = s.readout(33.333);
	CHECK(r.skipped == 0);
	CHECK(r.skipped_pct == Approx(0.0));
}

TEST_CASE("added latency = pairing delay x frame + one frame", "[stats]")
{
	Stats s;
	s.add_pairing_delay(2);
	s.add_pairing_delay(2);
	s.add_pairing_delay(3);
	auto r = s.readout(1000.0 / 30.0);
	CHECK(r.pairing_delay_frames == Approx(2.0));
	CHECK(r.added_latency_ms == Approx(100.0));
	CHECK(r.added_latency_frames == Approx(3.0));
}

TEST_CASE("empty window is all zero and not behind", "[stats]")
{
	Stats s;
	auto r = s.readout(16.7);
	CHECK(r.samples == 0);
	CHECK(r.skipped_pct == 0);
	CHECK_FALSE(r.behind);
	CHECK(r.added_latency_ms == 0);
}

TEST_CASE("readout text", "[stats]")
{
	Readout r;
	r.samples = 10;
	r.worker_ms_median = 23.14;
	r.skipped_pct = 0.2;
	r.added_latency_ms = 62.4;
	r.added_latency_frames = 1.87;
	CHECK(format_readout(r) == "23.1 ms, 0.2% skipped, +62 ms (1.9 frames)");
}
