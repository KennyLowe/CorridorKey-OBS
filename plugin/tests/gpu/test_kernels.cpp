// CUDA kernels vs the feature 001 spike's own Python (fixtures from `python -m ckbench.parity
// kernels`: hint.py chroma_hint / calibrated_hint, pipeline._down / _up, float32 on the CPU).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gpu/kernels.cuh"
#include "npy.hpp"

using namespace ck::gpu;

static const std::string kDir = std::string(CK_REPO_ROOT) + "/outputs/parity/kernels/";

#define CU(x) REQUIRE((x) == cudaSuccess)

template<class T> struct Dev {
	T *p = nullptr;
	explicit Dev(size_t n) { REQUIRE(cudaMalloc(&p, n * sizeof(T)) == cudaSuccess); }
	~Dev() { cudaFree(p); }
};

struct Case {
	std::string name;
	int W, H, w, h;
};

static std::vector<Case> cases()
{
	std::ifstream f(kDir + "meta.json");
	REQUIRE(f.good());
	auto j = nlohmann::json::parse(f);
	std::vector<Case> v;
	for (auto &c : j["cases"])
		v.push_back({c["name"], c["W"], c["H"], c["w"], c["h"]});
	return v;
}

struct Diff {
	double max = 0, mean = 0;
};
static Diff diff(const float *a, const float *b, size_t n)
{
	Diff d;
	for (size_t i = 0; i < n; i++) {
		double e = std::fabs((double)a[i] - (double)b[i]);
		d.max = std::max(d.max, e);
		d.mean += e;
	}
	d.mean /= (double)n;
	return d;
}

static std::vector<float> run_hint(const Case &c, const npy::Array &rgba, HintParams p)
{
	size_t n = (size_t)c.W * c.H;
	Dev<uint8_t> d_rgba(n * 4);
	Dev<float> tmp(n), mask(n);
	cudaStream_t s;
	CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
	CU(cudaMemcpyAsync(d_rgba.p, rgba.u8(), n * 4, cudaMemcpyHostToDevice, s));
	CU(hint(d_rgba.p, (size_t)c.W * 4, c.W, c.H, p, tmp.p, mask.p, s));
	std::vector<float> out(n);
	CU(cudaMemcpyAsync(out.data(), mask.p, n * 4, cudaMemcpyDeviceToHost, s));
	CU(cudaStreamSynchronize(s));
	cudaStreamDestroy(s);
	return out;
}

TEST_CASE("hint (fixed key) matches chroma_hint", "[kernels]")
{
	for (auto &c : cases()) {
		auto rgba = npy::load(kDir + c.name + "_rgba.npy");
		auto ref = npy::load(kDir + c.name + "_hint_fixed.npy");
		auto out = run_hint(c, rgba, HintParams{1.0f, 0.05f, 0.15f});
		auto d = diff(out.data(), ref.f32(), out.size());
		INFO(c.name << " max " << d.max << " mean " << d.mean);
		CHECK(d.max <= 1e-3);
	}
}

TEST_CASE("hint (calibrated key) matches calibrated_hint", "[kernels]")
{
	const float key[3] = {0.20f, 0.70f, 0.25f};
	float kd = key[1] - std::max(key[0], key[2]);
	for (auto &c : cases()) {
		auto rgba = npy::load(kDir + c.name + "_rgba.npy");
		auto ref = npy::load(kDir + c.name + "_hint_cal.npy");
		auto out = run_hint(c, rgba, HintParams{kd, 0.30f, 0.70f});
		auto d = diff(out.data(), ref.f32(), out.size());
		INFO(c.name << " max " << d.max << " mean " << d.mean);
		CHECK(d.max <= 1e-3);
	}
}

TEST_CASE("preprocess matches the spike's model input", "[kernels]")
{
	for (auto &c : cases()) {
		auto rgba = npy::load(kDir + c.name + "_rgba.npy");
		auto hint_ref = npy::load(kDir + c.name + "_hint_fixed.npy");
		auto ref = npy::load(kDir + c.name + "_input.npy");
		REQUIRE(ref.shape == std::vector<size_t>{1, 4, (size_t)c.h, (size_t)c.w});
		size_t n = (size_t)c.W * c.H, m = (size_t)4 * c.w * c.h;
		Dev<uint8_t> d_rgba(n * 4);
		Dev<float> d_mask(n), d_out(m), scratch(preprocess_scratch_floats(c.W, c.H, c.w, c.h));
		cudaStream_t s;
		CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
		CU(cudaMemcpyAsync(d_rgba.p, rgba.u8(), n * 4, cudaMemcpyHostToDevice, s));
		// feed the reference hint so this test isolates the resize + normalisation
		CU(cudaMemcpyAsync(d_mask.p, hint_ref.f32(), n * 4, cudaMemcpyHostToDevice, s));
		CU(preprocess(d_rgba.p, (size_t)c.W * 4, d_mask.p, c.W, c.H, d_out.p, c.w, c.h, scratch.p, s));
		std::vector<float> out(m);
		CU(cudaMemcpyAsync(out.data(), d_out.p, m * 4, cudaMemcpyDeviceToHost, s));
		CU(cudaStreamSynchronize(s));
		cudaStreamDestroy(s);
		// RGB channels are normalised (divided by std ~0.22), so allow 1e-3 in [0,1] units there.
		size_t plane = (size_t)c.w * c.h;
		auto drgb = diff(out.data(), ref.f32(), 3 * plane);
		auto dh = diff(out.data() + 3 * plane, ref.f32() + 3 * plane, plane);
		INFO(c.name << " rgb max " << drgb.max << " hint max " << dh.max);
		CHECK(drgb.max <= 1e-3 / 0.224);
		CHECK(dh.max <= 1e-3);
	}
}

TEST_CASE("postprocess matches the spike's RGBA8 output", "[kernels]")
{
	for (auto &c : cases()) {
		auto alpha = npy::load(kDir + c.name + "_alpha.npy");
		auto fg = npy::load(kDir + c.name + "_fg.npy");
		auto ref = npy::load(kDir + c.name + "_out.npy");
		size_t n = (size_t)c.W * c.H, m = (size_t)c.w * c.h;
		Dev<float> d_a(m), d_fg(3 * m);
		Dev<uint8_t> d_out(n * 4);
		cudaStream_t s;
		CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
		CU(cudaMemcpyAsync(d_a.p, alpha.f32(), m * 4, cudaMemcpyHostToDevice, s));
		CU(cudaMemcpyAsync(d_fg.p, fg.f32(), 3 * m * 4, cudaMemcpyHostToDevice, s));
		CU(postprocess(d_a.p, d_fg.p, c.w, c.h, d_out.p, (size_t)c.W * 4, c.W, c.H, s));
		std::vector<uint8_t> out(n * 4);
		CU(cudaMemcpyAsync(out.data(), d_out.p, n * 4, cudaMemcpyDeviceToHost, s));
		CU(cudaStreamSynchronize(s));
		cudaStreamDestroy(s);
		int maxd = 0;
		size_t off1 = 0;
		for (size_t i = 0; i < n * 4; i++) {
			int d = std::abs((int)out[i] - (int)ref.u8()[i]);
			maxd = std::max(maxd, d);
			off1 += d > 0;
		}
		INFO(c.name << " max LSB diff " << maxd << ", differing bytes " << off1);
		// 1e-3 in [0,1] is a quarter of an 8-bit step: only rounding ties may differ, by 1.
		CHECK(maxd <= 1);
		CHECK((double)off1 / (double)(n * 4) < 1e-3);
	}
}
