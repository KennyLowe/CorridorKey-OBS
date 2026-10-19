#include <catch2/catch_test_macros.hpp>

#include "core/state.hpp"

using namespace ck;

TEST_CASE("starts with no weights and passes through", "[state]")
{
	StateMachine sm;
	CHECK(sm.state() == State::NoWeights);
	CHECK(sm.output(FallbackMode::ColorKey, false) == Output::Passthrough);
	CHECK(sm.output(FallbackMode::LastMatte, true) == Output::Passthrough);
}

TEST_CASE("no_weights -> preparing -> ready -> keying", "[state]")
{
	StateMachine sm;
	CHECK(sm.preparing(10, "onnx"));
	CHECK(sm.state() == State::Preparing);
	CHECK(sm.progress() == 10);
	CHECK(sm.reason() == "onnx");
	CHECK(sm.output(FallbackMode::ColorKey, false) == Output::ColorKey);
	CHECK(sm.preparing(45, "engine")); // progress changes are reported
	CHECK_FALSE(sm.preparing(45, "engine"));
	CHECK(sm.engine_ready());
	CHECK(sm.state() == State::Ready);
	CHECK(sm.output(FallbackMode::Passthrough, false) == Output::Passthrough);
	CHECK(sm.matte_arrived(1000.0));
	CHECK(sm.state() == State::Keying);
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::Composite);
}

TEST_CASE("cache hit goes straight to ready", "[state]")
{
	StateMachine sm;
	CHECK(sm.engine_ready());
	CHECK(sm.state() == State::Ready);
}

TEST_CASE("preparing -> failed keeps the fallback output", "[state]")
{
	StateMachine sm;
	sm.preparing(0, "verify");
	CHECK(sm.prepare_failed("weights_invalid: not a CorridorKey checkpoint"));
	CHECK(sm.state() == State::Failed);
	CHECK(sm.reason() == "weights_invalid: not a CorridorKey checkpoint");
	CHECK(sm.output(FallbackMode::ColorKey, false) == Output::ColorKey);
	CHECK(sm.output(FallbackMode::Passthrough, false) == Output::Passthrough);
}

TEST_CASE("keying <-> behind follows the worker speed", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(0.0);
	CHECK(sm.tick(10.0, 250.0, true));
	CHECK(sm.state() == State::Behind);
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::Composite);
	CHECK_FALSE(sm.tick(20.0, 250.0, true));
	sm.matte_arrived(30.0);
	CHECK(sm.state() == State::Behind); // a matte doesn't clear "behind"; the window does
	CHECK(sm.tick(40.0, 250.0, false));
	CHECK(sm.state() == State::Keying);
}

TEST_CASE("no matte for the timeout -> fallback, then recovers on the next matte", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(1000.0);
	CHECK_FALSE(sm.tick(1249.0, 250.0, false));
	CHECK(sm.state() == State::Keying);
	CHECK(sm.tick(1251.0, 250.0, false));
	CHECK(sm.state() == State::Fallback);
	CHECK(sm.reason() == "timeout");
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::ColorKey);
	CHECK(sm.output(FallbackMode::LastMatte, true) == Output::LastMatte);
	CHECK(sm.output(FallbackMode::LastMatte, false) == Output::Passthrough);
	CHECK(sm.output(FallbackMode::Passthrough, true) == Output::Passthrough);
	CHECK(sm.matte_arrived(1300.0));
	CHECK(sm.state() == State::Keying);
}

TEST_CASE("worker error -> immediate fallback, recovers on the next matte", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(0.0);
	CHECK(sm.worker_error("enqueueV3 failed"));
	CHECK(sm.state() == State::Fallback);
	CHECK(sm.reason() == "enqueueV3 failed");
	CHECK_FALSE(sm.tick(5.0, 250.0, false)); // stays in fallback
	CHECK(sm.matte_arrived(10.0));
	CHECK(sm.state() == State::Keying);
}

TEST_CASE("worker error before the first matte also falls back", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	CHECK(sm.worker_error("engine load failed"));
	CHECK(sm.state() == State::Fallback);
}

TEST_CASE("timeout does not apply before the first matte", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	CHECK_FALSE(sm.tick(100000.0, 250.0, false));
	CHECK(sm.state() == State::Ready);
}

TEST_CASE("unsupported from any state uses the fallback mode", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(0.0);
	CHECK(sm.unsupported("nvinfer_10.dll not found"));
	CHECK(sm.state() == State::Unsupported);
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::ColorKey);
	CHECK_FALSE(sm.unsupported("nvinfer_10.dll not found"));
	CHECK_FALSE(sm.matte_arrived(10.0)); // no recovery from unsupported by a stray matte
	CHECK(sm.state() == State::Unsupported);
	CHECK_FALSE(sm.tick(1000.0, 250.0, false));
}

TEST_CASE("weights removed -> no_weights passes through", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(0.0);
	CHECK(sm.no_weights());
	CHECK(sm.state() == State::NoWeights);
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::Passthrough);
}

TEST_CASE("names", "[state]")
{
	CHECK(std::string(to_string(State::Behind)) == "behind");
	CHECK(std::string(to_string(State::NoWeights)) == "no_weights");
	CHECK(fallback_mode_from_string("last_matte") == FallbackMode::LastMatte);
	CHECK(fallback_mode_from_string("passthrough") == FallbackMode::Passthrough);
	CHECK(fallback_mode_from_string("colorkey") == FallbackMode::ColorKey);
	CHECK(fallback_mode_from_string("bogus") == FallbackMode::ColorKey);
	CHECK(std::string(to_string(FallbackMode::LastMatte)) == "last_matte");
}

TEST_CASE("no_weights -> downloading -> preparing (T011a)", "[state]")
{
	StateMachine sm;
	CHECK(sm.downloading(0));
	CHECK(sm.state() == State::Downloading);
	CHECK(sm.output(FallbackMode::ColorKey, false) == Output::Passthrough);
	CHECK(sm.downloading(42));
	CHECK(sm.progress() == 42);
	CHECK_FALSE(sm.downloading(42));
	CHECK(sm.preparing(0, "verify"));
	CHECK(sm.state() == State::Preparing);
	CHECK(std::string(to_string(State::Downloading)) == "downloading");
}

TEST_CASE("a failed download returns to no_weights with the error", "[state]")
{
	StateMachine sm;
	sm.downloading(10);
	CHECK(sm.download_failed("network: cannot reach huggingface.co"));
	CHECK(sm.state() == State::NoWeights);
	CHECK(sm.reason() == "network: cannot reach huggingface.co");
	CHECK(sm.output(FallbackMode::ColorKey, true) == Output::Passthrough);
}

TEST_CASE("a download while keying (switching model) doesn't stop the key", "[state]")
{
	StateMachine sm;
	sm.engine_ready();
	sm.matte_arrived(0.0);
	CHECK_FALSE(sm.downloading(5));
	CHECK(sm.state() == State::Keying);
	CHECK_FALSE(sm.download_failed("network"));
	CHECK(sm.state() == State::Keying);
}
