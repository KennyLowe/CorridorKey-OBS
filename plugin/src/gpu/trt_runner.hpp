// TensorRT engine runner (research D2): deserialise one engine, bind float32 device buffers,
// enqueueV3 on a caller-owned stream. Requires load_tensorrt() to have succeeded first.
// Owned and used by one thread (the worker); never touched by the render thread.
#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

namespace ck::gpu {

class TrtRunner {
public:
	~TrtRunner();
	// nullptr + err on failure (missing file, wrong TensorRT, unexpected I/O, out of memory).
	static std::unique_ptr<TrtRunner> load(const std::string &engine_path_utf8, std::string &err);

	int width() const { return w_; }
	int height() const { return h_; }
	// Device buffers owned by the runner: input [1,4,h,w], alpha [1,1,h,w], fg [1,3,h,w] (float32).
	float *input() const { return in_; }
	const float *alpha() const { return alpha_; }
	const float *fg() const { return fg_; }

	// Queues inference on `stream`; returns "" or an error.
	std::string enqueue(cudaStream_t stream);

private:
	TrtRunner() = default;
	struct Impl;
	std::unique_ptr<Impl> impl_;
	int w_ = 0, h_ = 0;
	float *in_ = nullptr, *alpha_ = nullptr, *fg_ = nullptr;
};

} // namespace ck::gpu
