#include "gpu/trt_runner.hpp"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

#include <NvInferRuntime.h>

#include "gpu/dll_loader.hpp"

namespace ck::gpu {
namespace {

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char *msg) noexcept override
	{
		if (severity <= Severity::kERROR) {
			std::lock_guard lk(mu);
			last = msg;
		}
	}
	std::string take()
	{
		std::lock_guard lk(mu);
		return std::exchange(last, {});
	}
	std::mutex mu;
	std::string last;
};

Logger &logger()
{
	static Logger l;
	return l;
}

} // namespace

struct TrtRunner::Impl {
	std::unique_ptr<nvinfer1::IRuntime> runtime;
	std::unique_ptr<nvinfer1::ICudaEngine> engine;
	std::unique_ptr<nvinfer1::IExecutionContext> ctx;
};

TrtRunner::~TrtRunner()
{
	if (impl_) {
		impl_->ctx.reset();
		impl_->engine.reset();
		impl_->runtime.reset();
	}
	cudaFree(in_);
	cudaFree(alpha_);
	cudaFree(fg_);
}

std::unique_ptr<TrtRunner> TrtRunner::load(const std::string &engine_path, std::string &err)
{
	if (!tensorrt_loaded()) {
		err = "TensorRT is not loaded";
		return nullptr;
	}
	std::ifstream f(std::filesystem::path((const char8_t *)engine_path.c_str()), std::ios::binary);
	if (!f) {
		err = "cannot open engine " + engine_path;
		return nullptr;
	}
	std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	std::unique_ptr<TrtRunner> r(new TrtRunner());
	r->impl_ = std::make_unique<Impl>();
	auto &im = *r->impl_;
	im.runtime.reset(nvinfer1::createInferRuntime(logger()));
	if (!im.runtime) {
		err = "createInferRuntime failed";
		return nullptr;
	}
	im.engine.reset(im.runtime->deserializeCudaEngine(blob.data(), blob.size()));
	if (!im.engine) {
		err = "failed to deserialize " + engine_path + ": " + logger().take();
		return nullptr;
	}
	im.ctx.reset(im.engine->createExecutionContext());
	if (!im.ctx) {
		err = "createExecutionContext failed (out of GPU memory?): " + logger().take();
		return nullptr;
	}
	// Expected I/O: input [1,4,h,w], alpha [1,1,h,w], fg [1,3,h,w], all float32 (the ONNX export
	// is fp32 at the boundary; FP16 is internal). Anything else fails closed.
	int h = 0, w = 0;
	bool have_in = false, have_a = false, have_fg = false;
	for (int i = 0; i < im.engine->getNbIOTensors(); i++) {
		const char *name = im.engine->getIOTensorName(i);
		auto dims = im.engine->getTensorShape(name);
		if (im.engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT || dims.nbDims != 4 || dims.d[0] != 1) {
			err = std::string("unexpected engine tensor ") + name + " (need float32 NCHW batch 1)";
			return nullptr;
		}
		std::string n = name;
		if (im.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
			if (dims.d[1] != 4) {
				err = "engine input must have 4 channels";
				return nullptr;
			}
			h = (int)dims.d[2];
			w = (int)dims.d[3];
			have_in = true;
		} else if (n == "alpha" && dims.d[1] == 1) {
			have_a = true;
		} else if (n == "fg" && dims.d[1] == 3) {
			have_fg = true;
		}
	}
	if (!have_in || !have_a || !have_fg || w <= 0 || h <= 0) {
		err = "engine is not a CorridorKey engine (need input, alpha, fg)";
		return nullptr;
	}
	r->w_ = w;
	r->h_ = h;
	size_t plane = (size_t)w * h * sizeof(float);
	if (cudaMalloc(&r->in_, 4 * plane) != cudaSuccess || cudaMalloc(&r->alpha_, plane) != cudaSuccess ||
	    cudaMalloc(&r->fg_, 3 * plane) != cudaSuccess) {
		cudaGetLastError();
		err = "out of GPU memory allocating engine buffers";
		return nullptr;
	}
	for (int i = 0; i < im.engine->getNbIOTensors(); i++) {
		const char *name = im.engine->getIOTensorName(i);
		std::string n = name;
		void *p = im.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? (void *)r->in_
			  : n == "alpha"                                                    ? (void *)r->alpha_
											    : (void *)r->fg_;
		if (!im.ctx->setTensorAddress(name, p)) {
			err = "setTensorAddress failed for " + n;
			return nullptr;
		}
	}
	return r;
}

std::string TrtRunner::enqueue(cudaStream_t stream)
{
	if (!impl_->ctx->enqueueV3(stream)) {
		auto m = logger().take();
		return "enqueueV3 failed" + (m.empty() ? std::string() : ": " + m);
	}
	return {};
}

} // namespace ck::gpu
