#include "core/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ck {

double percentile(std::deque<double> v, double p)
{
	if (v.empty())
		return 0.0;
	std::sort(v.begin(), v.end());
	double rank = p / 100.0 * (double)(v.size() - 1);
	size_t lo = (size_t)std::floor(rank);
	size_t hi = std::min(lo + 1, v.size() - 1);
	double f = rank - (double)lo;
	return v[lo] + (v[hi] - v[lo]) * f;
}

template<class D> static void push(D &d, typename D::value_type v, size_t cap)
{
	d.push_back(v);
	while (d.size() > cap)
		d.pop_front();
}

void Stats::add_worker_ms(double ms)
{
	push(worker_ms_, ms, window_);
}

void Stats::add_frame(bool submitted, uint64_t skipped_delta)
{
	push(frames_, {(uint8_t)(submitted ? 1 : 0), skipped_delta}, window_);
}

void Stats::add_pairing_delay(double frames)
{
	push(delay_, frames, window_);
}

Readout Stats::readout(double frame_ms) const
{
	Readout r;
	r.samples = worker_ms_.size();
	r.worker_ms_median = percentile(worker_ms_, 50);
	r.worker_ms_p95 = percentile(worker_ms_, 95);
	for (auto &[sub, skip] : frames_) {
		r.submitted += sub;
		r.skipped += skip;
	}
	r.skipped_pct = r.submitted ? 100.0 * (double)r.skipped / (double)r.submitted : 0.0;
	if (!delay_.empty()) {
		r.pairing_delay_frames = percentile(delay_, 50);
		r.added_latency_ms = r.pairing_delay_frames * frame_ms + frame_ms;
		r.added_latency_frames = frame_ms > 0 ? r.added_latency_ms / frame_ms : 0.0;
	}
	r.behind = r.samples > 0 && r.worker_ms_median > frame_ms;
	return r;
}

void Stats::clear()
{
	worker_ms_.clear();
	delay_.clear();
	frames_.clear();
}

std::string format_readout(const Readout &r)
{
	char buf[160];
	snprintf(buf, sizeof(buf), "%.1f ms, %.1f%% skipped, +%.0f ms (%.1f frames)", r.worker_ms_median, r.skipped_pct,
		 r.added_latency_ms, r.added_latency_frames);
	return buf;
}

} // namespace ck
