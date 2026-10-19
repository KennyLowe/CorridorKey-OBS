#include "core/calibrate.hpp"

#include <algorithm>
#include <cstdio>

namespace ck {

static constexpr float kMinKeyDiff = 0.02f; // hint.py MIN_KEY_DIFF

void Calibrator::add_frame(const uint8_t *rgba, int w, int h, size_t pitch)
{
	const int b = std::min({border_, w, h});
	auto add = [&](int x, int y) {
		const uint8_t *p = rgba + (size_t)y * pitch + (size_t)x * 4;
		hist_[0][p[0]]++;
		hist_[1][p[1]]++;
		hist_[2][p[2]]++;
		count_++;
	};
	for (int y = 0; y < b; y++) // top band, full width
		for (int x = 0; x < w; x++)
			add(x, y);
	for (int y = h - b; y < h; y++) // bottom band
		for (int x = 0; x < w; x++)
			add(x, y);
	for (int y = 0; y < h; y++) { // left and right bands, full height (corners again)
		for (int x = 0; x < b; x++)
			add(x, y);
		for (int x = w - b; x < w; x++)
			add(x, y);
	}
	frames_++;
}

std::optional<std::array<float, 3>> Calibrator::key(std::string &err) const
{
	if (!count_) {
		err = "no frames captured";
		return std::nullopt;
	}
	std::array<float, 3> k{};
	const uint64_t target = (count_ - 1) / 2; // lower median, like torch.median
	for (int c = 0; c < 3; c++) {
		uint64_t cum = 0;
		for (int v = 0; v < 256; v++) {
			cum += hist_[c][v];
			if (cum > target) {
				k[c] = (float)v / 255.0f;
				break;
			}
		}
	}
	if (k[1] - std::max(k[0], k[2]) < kMinKeyDiff) {
		char buf[160];
		snprintf(buf, sizeof(buf), "the border colour (%.2f, %.2f, %.2f) has no green excess; is this a green screen?",
			 k[0], k[1], k[2]);
		err = buf;
		return std::nullopt;
	}
	return k;
}

} // namespace ck
