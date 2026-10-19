// Border-median calibration vs bench/ckbench/hint.py estimate_key_colour on real frames of the
// pexels-athlete clip (fixture: `python -m ckbench.parity calibration`).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "../gpu/npy.hpp"
#include "core/calibrate.hpp"

using namespace ck;

TEST_CASE("matches hint.py estimate_key_colour on pexels-athlete", "[calibrate]")
{
	const std::string dir = std::string(CK_REPO_ROOT) + "/outputs/parity/calibration/";
	auto frames = npy::load(dir + "frames.npy"); // [N,H,W,3] uint8 RGB
	std::ifstream mf(dir + "meta.json");
	REQUIRE(mf.good());
	auto meta = nlohmann::json::parse(mf);
	REQUIRE(frames.shape.size() == 4);
	const int n = (int)frames.shape[0], h = (int)frames.shape[1], w = (int)frames.shape[2];
	Calibrator c(meta["border_px"].get<int>());
	std::vector<uint8_t> rgba((size_t)w * h * 4);
	for (int i = 0; i < n; i++) {
		const uint8_t *src = frames.u8() + (size_t)i * w * h * 3;
		for (size_t p = 0; p < (size_t)w * h; p++) {
			rgba[p * 4 + 0] = src[p * 3 + 0];
			rgba[p * 4 + 1] = src[p * 3 + 1];
			rgba[p * 4 + 2] = src[p * 3 + 2];
			rgba[p * 4 + 3] = 255;
		}
		c.add_frame(rgba.data(), w, h, (size_t)w * 4);
	}
	CHECK(c.frames() == n);
	std::string err;
	auto key = c.key(err);
	INFO(err);
	REQUIRE(key);
	for (int ch = 0; ch < 3; ch++)
		CHECK((*key)[ch] == Catch::Approx(meta["key"][ch].get<double>()).margin(1e-6));
}

TEST_CASE("a frame with no green excess can't be calibrated", "[calibrate]")
{
	std::vector<uint8_t> grey(64 * 64 * 4, 128);
	Calibrator c(8);
	c.add_frame(grey.data(), 64, 64, 64 * 4);
	std::string err;
	CHECK_FALSE(c.key(err));
	CHECK(err.find("green") != std::string::npos);
}

TEST_CASE("no frames: no key", "[calibrate]")
{
	Calibrator c;
	std::string err;
	CHECK_FALSE(c.key(err));
}
