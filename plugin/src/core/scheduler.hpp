// Input/output ring between the render thread and the inference worker (data-model
// "Scheduler ring"). Thread-safe. The render thread never blocks here except on the internal
// mutex, which is only ever held for a few instructions.
//
// Input slots (4): free -> writing (render copies a frame in) -> submitted -> in_flight (worker)
// -> free (worker releases it once its GPU work on the slot has finished).
//   - begin_submit takes a free slot; if none is free it drops the oldest submitted slot (counted
//     as skipped) and reuses it. It never blocks.
//   - take() gives the worker the newest submitted slot and drops older submitted ones (skipped).
// Output slots (2): free -> writing (worker) -> done -> displayed (render) -> free when a newer
//   output is displayed. A done output that is overtaken by a newer one before the render picks
//   it up is "superseded" (counted separately).
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>

namespace ck {

class Scheduler {
public:
	static constexpr int kInputs = 4;
	static constexpr int kOutputs = 2;

	enum class In { Free, Writing, Submitted, InFlight };
	enum class Out { Free, Writing, Done, Displayed };

	struct Job {
		int slot;
		uint64_t frame_id; // render frame counter at submit
		int64_t ts_ns;     // frame timestamp
	};
	struct Result {
		int out_slot;
		uint64_t frame_id; // the input frame the output was made from
		int64_t ts_ns;
		double worker_ms;
	};

	// ---- render thread
	// Returns the input slot to write the new frame into (never blocks, never fails).
	int begin_submit();
	void end_submit(int slot, uint64_t frame_id, int64_t ts_ns);
	void abort_submit(int slot);
	// If a newer output is done, mark it displayed (the previous displayed one becomes free).
	std::optional<Result> poll_output();

	// ---- worker thread
	// Waits up to `timeout` for a submitted slot. Returns nullopt on timeout or after stop().
	std::optional<Job> take(std::chrono::milliseconds timeout);
	void release_input(int slot);
	// The output slot to write into: never the displayed one.
	int begin_output();
	void complete_output(int out_slot, const Job &job, double worker_ms);
	void abort_output(int out_slot);

	// ---- any thread
	void stop();
	bool stopped() const;
	uint64_t skipped() const;    // input frames dropped without being keyed
	uint64_t superseded() const; // outputs produced but never displayed
	uint64_t submitted() const;
	In input_state(int slot) const;
	Out output_state(int slot) const;
	// Return every slot to free (e.g. after a resize); keeps the counters.
	void reset();

private:
	mutable std::mutex mu_;
	std::condition_variable cv_;
	std::array<In, kInputs> in_{};
	std::array<Job, kInputs> in_job_{};
	std::array<uint64_t, kInputs> in_seq_{}; // submission order
	std::array<Out, kOutputs> out_{};
	std::array<Result, kOutputs> out_res_{};
	std::array<uint64_t, kOutputs> out_seq_{};
	uint64_t seq_ = 0;
	uint64_t skipped_ = 0, superseded_ = 0, submitted_ = 0;
	bool stop_ = false;
};

} // namespace ck
