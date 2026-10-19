// Filter state machine (data-model "FilterState"). Pure C++: no OBS, no CUDA.
// Not thread-safe by itself: the filter owns one instance and calls it from the render thread
// (worker events are handed over through atomics/the scheduler first).
#pragma once

#include <string>

namespace ck {

enum class State { NoWeights, Downloading, Unsupported, Preparing, Ready, Keying, Behind, Fallback, Failed };
enum class FallbackMode { ColorKey, LastMatte, Passthrough };
// What the render thread draws this frame.
enum class Output { Composite, ColorKey, LastMatte, Passthrough };

const char *to_string(State s);
const char *to_string(FallbackMode m);
FallbackMode fallback_mode_from_string(const std::string &s); // unknown -> ColorKey

class StateMachine {
public:
	State state() const { return state_; }
	// Human-readable cause for Unsupported / Fallback / Failed, and the preparing stage.
	const std::string &reason() const { return reason_; }
	int progress() const { return progress_; }

	// Each event returns true when the state (or its reason) changed, so callers can log it.
	bool unsupported(const std::string &why);
	bool no_weights();
	// Weights download (in-plugin, via the helper). Only shown when nothing is keying: a download
	// for a model switch runs in the background while the current engine keeps keying.
	bool downloading(int pct);
	bool download_failed(const std::string &error);
	bool preparing(int pct, const std::string &stage);
	bool prepare_failed(const std::string &error);
	bool engine_ready(); // cache hit or preparation finished
	// A new matte was produced at now_ms.
	bool matte_arrived(double now_ms);
	bool worker_error(const std::string &error);
	// Once per render frame. `behind`: the worker is slower than the frame time over the window.
	bool tick(double now_ms, double fallback_timeout_ms, bool behind);

	// The output for the current state. LastMatte needs a matte; without one it degrades to
	// pass-through.
	Output output(FallbackMode mode, bool have_last_matte) const;

private:
	bool set(State s, const std::string &why = {});
	State state_ = State::NoWeights;
	std::string reason_;
	int progress_ = 0;
	double last_matte_ms_ = 0.0;
	bool have_matte_time_ = false;
};

} // namespace ck
