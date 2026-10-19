#include "core/state.hpp"

namespace ck {

const char *to_string(State s)
{
	switch (s) {
	case State::NoWeights:
		return "no_weights";
	case State::Downloading:
		return "downloading";
	case State::Unsupported:
		return "unsupported";
	case State::Preparing:
		return "preparing";
	case State::Ready:
		return "ready";
	case State::Keying:
		return "keying";
	case State::Behind:
		return "behind";
	case State::Fallback:
		return "fallback";
	case State::Failed:
		return "failed";
	}
	return "?";
}

const char *to_string(FallbackMode m)
{
	switch (m) {
	case FallbackMode::ColorKey:
		return "colorkey";
	case FallbackMode::LastMatte:
		return "last_matte";
	case FallbackMode::Passthrough:
		return "passthrough";
	}
	return "?";
}

FallbackMode fallback_mode_from_string(const std::string &s)
{
	if (s == "last_matte")
		return FallbackMode::LastMatte;
	if (s == "passthrough")
		return FallbackMode::Passthrough;
	return FallbackMode::ColorKey;
}

bool StateMachine::set(State s, const std::string &why)
{
	if (s == state_ && why == reason_)
		return false;
	state_ = s;
	reason_ = why;
	return true;
}

bool StateMachine::unsupported(const std::string &why)
{
	return set(State::Unsupported, why);
}

bool StateMachine::no_weights()
{
	have_matte_time_ = false;
	return set(State::NoWeights);
}

static bool idle(State s)
{
	return s == State::NoWeights || s == State::Downloading || s == State::Failed;
}

bool StateMachine::downloading(int pct)
{
	if (!idle(state_))
		return false; // a model switch downloads in the background; the current key carries on
	bool changed = pct != progress_ || state_ != State::Downloading;
	progress_ = pct;
	return set(State::Downloading, "downloading weights") || changed;
}

bool StateMachine::download_failed(const std::string &error)
{
	if (!idle(state_))
		return false;
	have_matte_time_ = false;
	return set(State::NoWeights, error);
}

bool StateMachine::preparing(int pct, const std::string &stage)
{
	bool changed = pct != progress_ || state_ != State::Preparing;
	progress_ = pct;
	have_matte_time_ = false;
	return set(State::Preparing, stage) || changed;
}

bool StateMachine::prepare_failed(const std::string &error)
{
	return set(State::Failed, error);
}

bool StateMachine::engine_ready()
{
	have_matte_time_ = false;
	return set(State::Ready);
}

bool StateMachine::matte_arrived(double now_ms)
{
	switch (state_) {
	case State::Ready:
	case State::Fallback:
		last_matte_ms_ = now_ms;
		have_matte_time_ = true;
		return set(State::Keying);
	case State::Keying:
	case State::Behind:
		last_matte_ms_ = now_ms;
		have_matte_time_ = true;
		return false;
	default:
		// No weights / preparing / unsupported / failed: a stray matte (e.g. from an engine being
		// replaced) must not resurrect the key.
		return false;
	}
}

bool StateMachine::worker_error(const std::string &error)
{
	switch (state_) {
	case State::Ready:
	case State::Keying:
	case State::Behind:
	case State::Fallback:
		return set(State::Fallback, error);
	default:
		return false;
	}
}

bool StateMachine::tick(double now_ms, double fallback_timeout_ms, bool behind)
{
	if (state_ != State::Keying && state_ != State::Behind)
		return false;
	if (have_matte_time_ && now_ms - last_matte_ms_ > fallback_timeout_ms)
		return set(State::Fallback, "timeout");
	return set(behind ? State::Behind : State::Keying);
}

Output StateMachine::output(FallbackMode mode, bool have_last_matte) const
{
	switch (state_) {
	case State::NoWeights:
	case State::Downloading:
		return Output::Passthrough;
	case State::Keying:
	case State::Behind:
		return Output::Composite;
	default:
		break;
	}
	switch (mode) {
	case FallbackMode::ColorKey:
		return Output::ColorKey;
	case FallbackMode::LastMatte:
		return have_last_matte ? Output::LastMatte : Output::Passthrough;
	case FallbackMode::Passthrough:
		return Output::Passthrough;
	}
	return Output::Passthrough;
}

} // namespace ck
