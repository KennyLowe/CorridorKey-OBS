// Screen-colour calibration (research D6, T032): the per-channel median of an outer border band
// over N frames, like bench/ckbench/hint.py estimate_key_colour (same band: the top and bottom
// `border` rows at full width plus the left and right `border` columns at full height, so the
// corners count twice; median = the lower middle value, as torch.median). Exact for 8-bit input
// via per-channel histograms. Pure C++.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ck {

class Calibrator {
public:
	explicit Calibrator(int border = 32) : border_(border) {}
	// RGBA8 (or RGBX) rows with a pitch in bytes.
	void add_frame(const uint8_t *rgba, int w, int h, size_t pitch);
	int frames() const { return frames_; }
	// The screen colour in [0,1], or nullopt + err when it has no green excess (not a green screen).
	std::optional<std::array<float, 3>> key(std::string &err) const;

private:
	int border_;
	int frames_ = 0;
	std::array<std::array<uint64_t, 256>, 3> hist_{};
	uint64_t count_ = 0;
};

} // namespace ck
