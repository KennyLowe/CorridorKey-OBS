#include "core/scheduler.hpp"

namespace ck {

int Scheduler::begin_submit()
{
	std::lock_guard lk(mu_);
	submitted_++;
	for (int i = 0; i < kInputs; i++) {
		if (in_[i] == In::Free) {
			in_[i] = In::Writing;
			return i;
		}
	}
	// No free slot: drop the oldest submitted frame and reuse its slot.
	int oldest = -1;
	for (int i = 0; i < kInputs; i++)
		if (in_[i] == In::Submitted && (oldest < 0 || in_seq_[i] < in_seq_[oldest]))
			oldest = i;
	// With one worker at most one slot is in flight and the render writes one at a time, so with
	// four slots a submitted one always exists here.
	if (oldest < 0)
		oldest = 0;
	skipped_++;
	in_[oldest] = In::Writing;
	return oldest;
}

void Scheduler::end_submit(int slot, uint64_t frame_id, int64_t ts_ns)
{
	{
		std::lock_guard lk(mu_);
		in_[slot] = In::Submitted;
		in_job_[slot] = Job{slot, frame_id, ts_ns};
		in_seq_[slot] = ++seq_;
	}
	cv_.notify_one();
}

void Scheduler::abort_submit(int slot)
{
	std::lock_guard lk(mu_);
	if (in_[slot] == In::Writing) {
		in_[slot] = In::Free;
		submitted_--;
	}
}

std::optional<Scheduler::Result> Scheduler::poll_output()
{
	std::lock_guard lk(mu_);
	int done = -1;
	for (int i = 0; i < kOutputs; i++)
		if (out_[i] == Out::Done && (done < 0 || out_seq_[i] > out_seq_[done]))
			done = i;
	if (done < 0)
		return std::nullopt;
	for (int i = 0; i < kOutputs; i++)
		if (i != done && out_[i] == Out::Displayed)
			out_[i] = Out::Free;
	out_[done] = Out::Displayed;
	return out_res_[done];
}

std::optional<Scheduler::Job> Scheduler::take(std::chrono::milliseconds timeout)
{
	std::unique_lock lk(mu_);
	auto has_work = [&] {
		if (stop_)
			return true;
		for (auto s : in_)
			if (s == In::Submitted)
				return true;
		return false;
	};
	if (!cv_.wait_for(lk, timeout, has_work) || stop_)
		return std::nullopt;
	int newest = -1;
	for (int i = 0; i < kInputs; i++)
		if (in_[i] == In::Submitted && (newest < 0 || in_seq_[i] > in_seq_[newest]))
			newest = i;
	for (int i = 0; i < kInputs; i++) {
		if (i != newest && in_[i] == In::Submitted) {
			in_[i] = In::Free; // stale: never queue (FR-004)
			skipped_++;
		}
	}
	in_[newest] = In::InFlight;
	return in_job_[newest];
}

void Scheduler::release_input(int slot)
{
	std::lock_guard lk(mu_);
	if (in_[slot] == In::InFlight)
		in_[slot] = In::Free;
}

int Scheduler::begin_output()
{
	std::lock_guard lk(mu_);
	int pick = -1;
	for (int i = 0; i < kOutputs; i++)
		if (out_[i] == Out::Free) {
			pick = i;
			break;
		}
	if (pick < 0) {
		// Overwrite an undisplayed result (the render hasn't picked it up yet).
		for (int i = 0; i < kOutputs; i++)
			if (out_[i] == Out::Done) {
				pick = i;
				superseded_++;
				break;
			}
	}
	if (pick < 0)
		pick = 0; // unreachable with one worker: at most one output is displayed
	out_[pick] = Out::Writing;
	return pick;
}

void Scheduler::complete_output(int out_slot, const Job &job, double worker_ms)
{
	std::lock_guard lk(mu_);
	// Any older done output is now superseded.
	for (int i = 0; i < kOutputs; i++)
		if (i != out_slot && out_[i] == Out::Done) {
			out_[i] = Out::Free;
			superseded_++;
		}
	out_[out_slot] = Out::Done;
	out_res_[out_slot] = Result{out_slot, job.frame_id, job.ts_ns, worker_ms};
	out_seq_[out_slot] = ++seq_;
}

void Scheduler::abort_output(int out_slot)
{
	std::lock_guard lk(mu_);
	if (out_[out_slot] == Out::Writing)
		out_[out_slot] = Out::Free;
}

void Scheduler::stop()
{
	{
		std::lock_guard lk(mu_);
		stop_ = true;
	}
	cv_.notify_all();
}

bool Scheduler::stopped() const
{
	std::lock_guard lk(mu_);
	return stop_;
}

uint64_t Scheduler::skipped() const
{
	std::lock_guard lk(mu_);
	return skipped_;
}

uint64_t Scheduler::superseded() const
{
	std::lock_guard lk(mu_);
	return superseded_;
}

uint64_t Scheduler::submitted() const
{
	std::lock_guard lk(mu_);
	return submitted_;
}

Scheduler::In Scheduler::input_state(int slot) const
{
	std::lock_guard lk(mu_);
	return in_[slot];
}

Scheduler::Out Scheduler::output_state(int slot) const
{
	std::lock_guard lk(mu_);
	return out_[slot];
}

void Scheduler::reset()
{
	std::lock_guard lk(mu_);
	in_.fill(In::Free);
	out_.fill(Out::Free);
}

} // namespace ck
