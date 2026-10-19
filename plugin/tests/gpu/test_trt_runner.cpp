// TensorRT runner parity: a fixed 512x512 model input (built by the spike's pipeline) run through
// the engine from `ckbench prepare` must match the spike's PyTorch fp16 output within the
// feature 001 drift bounds (fixtures: `python -m ckbench.parity runner`).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/cache_key.hpp"
#include "gpu/dll_loader.hpp"
#include "gpu/trt_runner.hpp"
#include "npy.hpp"

using namespace ck::gpu;

static const std::string kDir = std::string(CK_REPO_ROOT) + "/outputs/parity/runner/";

TEST_CASE("a missing TensorRT folder is reported, not a crash", "[trt]")
{
	auto err = load_tensorrt("C:/definitely/not/here");
	CHECK(err.find("nvinfer_10.dll not found") != std::string::npos);
	if (!tensorrt_loaded()) {
		std::string e2;
		CHECK_FALSE(TrtRunner::load("x.engine", e2));
		CHECK(e2 == "TensorRT is not loaded");
	}
}

TEST_CASE("TensorRT 10.16.1 loads from the configured folder", "[trt]")
{
	REQUIRE(load_tensorrt(CK_TRT_DIR) == "");
	CHECK(tensorrt_loaded());
	CHECK(tensorrt_version() == kRequiredTrtVersion);
	CHECK(load_tensorrt(CK_TRT_DIR) == ""); // idempotent
	CHECK(load_tensorrt("C:/Windows").find("already loaded") != std::string::npos);
}

TEST_CASE("device identity matches what ckbench prepare recorded", "[trt]")
{
	auto d = device_info(0);
	CHECK(d.uuid.rfind("GPU-", 0) == 0);
	CHECK(d.uuid.size() == 40);
	CHECK_FALSE(d.driver_version.empty());
	const char *la = std::getenv("LOCALAPPDATA");
	REQUIRE(la);
	auto m = ck::read_manifest(std::string(la) + "/corridorkey-obs/engines/manifest.json");
	REQUIRE(m);
	REQUIRE_FALSE(m->entries.empty());
	const auto &e = m->entries[0].comp;
	CHECK(e.gpu_uuid == d.uuid);
	CHECK(e.driver_version == d.driver_version);
	REQUIRE(load_tensorrt(CK_TRT_DIR) == "");
	CHECK(e.trt_version == tensorrt_version());
	CHECK(m->entries[0].gpu_name == d.name);
}

TEST_CASE("engine output matches the PyTorch spike within the 001 drift bounds", "[trt]")
{
	REQUIRE(load_tensorrt(CK_TRT_DIR) == "");
	std::ifstream mf(kDir + "meta.json");
	REQUIRE(mf.good());
	auto meta = nlohmann::json::parse(mf);
	std::string err;
	auto r = TrtRunner::load(meta["engine_path"].get<std::string>(), err);
	INFO(err);
	REQUIRE(r);
	CHECK(r->width() == 512);
	CHECK(r->height() == 512);

	auto in = npy::load(kDir + "input.npy");
	auto a_ref = npy::load(kDir + "alpha.npy");
	auto fg_ref = npy::load(kDir + "fg.npy");
	size_t plane = 512 * 512;
	REQUIRE(in.count() == 4 * plane);
	cudaStream_t s;
	REQUIRE(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess);
	REQUIRE(cudaMemcpyAsync(r->input(), in.f32(), 4 * plane * 4, cudaMemcpyHostToDevice, s) == cudaSuccess);
	REQUIRE(r->enqueue(s) == "");
	std::vector<float> a(plane), fg(3 * plane);
	REQUIRE(cudaMemcpyAsync(a.data(), r->alpha(), plane * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess);
	REQUIRE(cudaMemcpyAsync(fg.data(), r->fg(), 3 * plane * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess);
	REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
	cudaStreamDestroy(s);

	double a_mae = 0, fg_sum = 0;
	size_t fg_n = 0;
	for (size_t i = 0; i < plane; i++) {
		a_mae += std::fabs(a[i] - a_ref.f32()[i]);
		if (a_ref.f32()[i] > 0.1f)
			for (int c = 0; c < 3; c++) {
				fg_sum += std::fabs(fg[c * plane + i] - fg_ref.f32()[c * plane + i]);
				fg_n++;
			}
	}
	a_mae /= (double)plane;
	double fg_mae = fg_n ? fg_sum / (double)fg_n : 0.0;
	INFO("alpha MAE " << a_mae << ", fg MAE (alpha > 0.1) " << fg_mae << " over " << fg_n / 3 << " px");
	CHECK(fg_n > 0);
	CHECK(a_mae <= 1e-3);
	CHECK(fg_mae <= 1e-3);
	WARN("runner parity: alpha MAE " << a_mae << ", fg MAE (alpha > 0.1) " << fg_mae);
}
