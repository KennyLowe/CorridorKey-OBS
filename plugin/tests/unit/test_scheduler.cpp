#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <set>
#include <thread>
#include <vector>

#include "core/scheduler.hpp"

using namespace ck;
using namespace std::chrono_literals;
using In = Scheduler::In;
using Out = Scheduler::Out;

static int submit(Scheduler &s, uint64_t id)
{
	int slot = s.begin_submit();
	s.end_submit(slot, id, (int64_t)id * 1000);
	return slot;
}

TEST_CASE("submit uses free slots and never blocks", "[scheduler]")
{
	Scheduler s;
	std::set<int> used;
	for (uint64_t i = 0; i < 4; i++)
		used.insert(submit(s, i));
	CHECK(used.size() == 4);
	CHECK(s.skipped() == 0);
	// Ring full of submitted frames: the oldest submitted one is dropped and reused.
	int slot0 = -1;
	for (int i = 0; i < 4; i++)
		if (s.input_state(i) == In::Submitted)
			slot0 = slot0 < 0 ? i : slot0;
	int reused = submit(s, 4);
	CHECK(s.skipped() == 1);
	CHECK(s.submitted() == 5);
	CHECK(s.input_state(reused) == In::Submitted);
}

TEST_CASE("the oldest submitted slot is the one dropped", "[scheduler]")
{
	Scheduler s;
	int a = submit(s, 10);
	submit(s, 11);
	submit(s, 12);
	submit(s, 13);
	int r = submit(s, 14);
	CHECK(r == a); // frame 10 was the oldest
	auto j = s.take(0ms);
	REQUIRE(j);
	CHECK(j->frame_id == 14);
}

TEST_CASE("worker takes the newest and drops older submitted frames", "[scheduler]")
{
	Scheduler s;
	submit(s, 1);
	submit(s, 2);
	int newest = submit(s, 3);
	auto j = s.take(0ms);
	REQUIRE(j);
	CHECK(j->slot == newest);
	CHECK(j->frame_id == 3);
	CHECK(j->ts_ns == 3000);
	CHECK(s.skipped() == 2);
	CHECK(s.input_state(newest) == In::InFlight);
	int free_count = 0;
	for (int i = 0; i < 4; i++)
		free_count += s.input_state(i) == In::Free;
	CHECK(free_count == 3);
	CHECK_FALSE(s.take(0ms)); // nothing left
}

TEST_CASE("in-flight and writing slots are never handed out by submit", "[scheduler]")
{
	Scheduler s;
	submit(s, 1);
	auto j = s.take(0ms);
	REQUIRE(j);
	int w = s.begin_submit(); // writing, not yet submitted
	for (uint64_t i = 2; i < 20; i++) {
		int slot = submit(s, i);
		CHECK(slot != j->slot);
		CHECK(slot != w);
	}
	s.end_submit(w, 100, 0);
	s.release_input(j->slot);
	CHECK(s.input_state(j->slot) == In::Free);
}

TEST_CASE("abort_submit frees the slot", "[scheduler]")
{
	Scheduler s;
	int w = s.begin_submit();
	s.abort_submit(w);
	CHECK(s.input_state(w) == In::Free);
	CHECK_FALSE(s.take(0ms));
}

TEST_CASE("outputs: done -> displayed, previous displayed freed", "[scheduler]")
{
	Scheduler s;
	CHECK_FALSE(s.poll_output());
	submit(s, 7);
	auto j = s.take(0ms);
	int o = s.begin_output();
	s.complete_output(o, *j, 12.5);
	s.release_input(j->slot);
	auto r = s.poll_output();
	REQUIRE(r);
	CHECK(r->out_slot == o);
	CHECK(r->frame_id == 7);
	CHECK(r->worker_ms == 12.5);
	CHECK(s.output_state(o) == Out::Displayed);
	CHECK_FALSE(s.poll_output()); // nothing newer

	// The worker never writes into the displayed slot.
	submit(s, 8);
	auto j2 = s.take(0ms);
	int o2 = s.begin_output();
	CHECK(o2 != o);
	s.complete_output(o2, *j2, 10.0);
	auto r2 = s.poll_output();
	REQUIRE(r2);
	CHECK(r2->frame_id == 8);
	CHECK(s.output_state(o) == Out::Free);
	CHECK(s.output_state(o2) == Out::Displayed);
}

TEST_CASE("an undisplayed done output is superseded by a newer one", "[scheduler]")
{
	Scheduler s;
	submit(s, 1);
	auto j1 = s.take(0ms);
	int o1 = s.begin_output();
	s.complete_output(o1, *j1, 1.0);
	s.release_input(j1->slot);
	submit(s, 2);
	auto j2 = s.take(0ms);
	int o2 = s.begin_output(); // the done one is reused (none displayed yet, either is allowed)
	s.complete_output(o2, *j2, 1.0);
	s.release_input(j2->slot);
	CHECK(s.superseded() == 1);
	auto r = s.poll_output();
	REQUIRE(r);
	CHECK(r->frame_id == 2);
}

TEST_CASE("take times out and stop wakes the worker", "[scheduler]")
{
	Scheduler s;
	auto t0 = std::chrono::steady_clock::now();
	CHECK_FALSE(s.take(20ms));
	CHECK(std::chrono::steady_clock::now() - t0 >= 15ms);
	std::thread t([&] { std::this_thread::sleep_for(20ms); s.stop(); });
	t0 = std::chrono::steady_clock::now();
	CHECK_FALSE(s.take(5000ms));
	CHECK(std::chrono::steady_clock::now() - t0 < 2000ms);
	t.join();
	CHECK(s.stopped());
}

TEST_CASE("reset frees every slot", "[scheduler]")
{
	Scheduler s;
	submit(s, 1);
	submit(s, 2);
	auto j = s.take(0ms);
	int o = s.begin_output();
	s.complete_output(o, *j, 1.0);
	s.reset();
	for (int i = 0; i < Scheduler::kInputs; i++)
		CHECK(s.input_state(i) == In::Free);
	for (int i = 0; i < Scheduler::kOutputs; i++)
		CHECK(s.output_state(i) == Out::Free);
	CHECK_FALSE(s.poll_output());
}

TEST_CASE("concurrent render and worker keep the invariants", "[scheduler]")
{
	Scheduler s;
	std::atomic<bool> done{false};
	std::atomic<uint64_t> keyed{0}, displayed{0};
	std::atomic<int> violations{0};
	const uint64_t frames = 20000;

	std::thread worker([&] {
		uint64_t last = 0;
		while (true) {
			auto j = s.take(50ms);
			if (!j) {
				if (s.stopped())
					break;
				continue;
			}
			if (j->frame_id < last)
				violations++; // the worker must only move forward in time
			last = j->frame_id;
			int o = s.begin_output();
			if (s.output_state(o) != Out::Writing)
				violations++;
			std::this_thread::yield();
			s.complete_output(o, *j, 1.0);
			s.release_input(j->slot);
			keyed++;
		}
	});

	uint64_t last_disp = 0;
	for (uint64_t f = 1; f <= frames; f++) {
		int slot = s.begin_submit();
		if (s.input_state(slot) != In::Writing)
			violations++;
		s.end_submit(slot, f, (int64_t)f);
		if (auto r = s.poll_output()) {
			if (r->frame_id <= last_disp)
				violations++; // displayed outputs only move forward
			last_disp = r->frame_id;
			displayed++;
		}
	}
	s.stop();
	worker.join();
	done = true;
	CHECK(violations == 0);
	CHECK(keyed > 0);
	CHECK(displayed > 0);
	// Every submitted frame is either keyed, skipped, or still waiting in the ring.
	uint64_t waiting = 0;
	for (int i = 0; i < Scheduler::kInputs; i++)
		waiting += s.input_state(i) == In::Submitted;
	CHECK(keyed + s.skipped() + waiting == frames);
}
