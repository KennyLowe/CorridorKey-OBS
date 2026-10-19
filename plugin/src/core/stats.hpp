// Rolling performance window per filter instance (data-model "Readout", research D8).
// Not thread-safe: fed and read on the render thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>

namespace ck {

struct Readout {
	size_t samples = 0;           // worker-time samples in the window
	double worker_ms_median = 0;  // keying time (worker, GPU-complete)
	double worker_ms_p95 = 0;
	uint64_t skipped = 0;         // input frames skipped in the window
	uint64_t submitted = 0;       // input frames submitted in the window
	double skipped_pct = 0;       // skipped / submitted * 100
	double pairing_delay_frames = 0; // median render frames from submit to display
	double added_latency_ms = 0;  // pairing_delay x frame_ms (measured: see stats.cpp)
	double added_latency_frames = 0;
	bool behind = false;          // worker median slower than the frame time
};

class Stats {
public:
	explicit Stats(size_t window = 120) : window_(window) {}

	void add_worker_ms(double ms);
	// One render frame: was a frame submitted, and how many inputs were skipped since the last call.
	void add_frame(bool submitted, uint64_t skipped_delta);
	void add_pairing_delay(double frames);
	Readout readout(double frame_ms) const;
	void clear();

private:
	size_t window_;
	std::deque<double> worker_ms_;
	std::deque<double> delay_;
	std::deque<std::pair<uint8_t, uint64_t>> frames_; // (submitted, skipped)
};

// Percentile with linear interpolation between closest ranks (numpy's default).
double percentile(std::deque<double> v, double p);
// "Keying — 1024×576, 23.1 ms, 0.2% skipped, +62 ms (1.9 frames)" style fragment.
std::string format_readout(const Readout &r);

} // namespace ck
