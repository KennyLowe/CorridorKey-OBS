// Per-filter inference worker (research D4, T017). One thread per filter instance owns its CUDA
// stream, the TensorRT engine and the frame buffers. It also does all slow setup (loading
// TensorRT, hashing the weights, reading the engine cache, loading engines, running the prepare
// helper) so neither OBS's render thread nor its UI thread ever waits on it.
//
// The render thread talks to it through:
//   - the Scheduler (frames in, mattes out; never blocks),
//   - request(): the wanted configuration (settings, frame size), and
//   - status(): a snapshot of what the worker has (engine, buffers, errors, preparation).
// Errors never throw into OBS: they are reported through status().
#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "core/auto_select.hpp"
#include "core/scheduler.hpp"
#include "gpu/dll_loader.hpp"
#include "gpu/kernels.cuh"

namespace ck {

namespace gpu {
class TrtRunner;
}

enum class DebugFail { None, WorkerError, Stall };
DebugFail debug_fail_from_string(const std::string &s);

struct WorkerRequest {
	std::string trt_dir;
	std::string weights_path;
	std::string quality = "auto"; // "auto" or a quality id
	std::string helper_cmd;       // command line prefix used to run `prepare`
	std::string cache_dir;        // engine cache (manifest.json)
	std::string perf_table_path;
	int frame_w = 0, frame_h = 0; // source size (render thread)
	double fps = 30.0;
	gpu::HintParams hint;
	DebugFail debug = DebugFail::None;
	uint64_t prepare_clicks = 0; // "Prepare selected" presses
	bool operator==(const WorkerRequest &) const = default;
};

enum class Phase { Starting, NoWeights, Unsupported, Preparing, Failed, Ready };
const char *to_string(Phase p);

struct WorkerStatus {
	uint64_t seq = 0; // bumps on every change
	Phase phase = Phase::Starting;
	std::string reason; // why unsupported/failed/no weights; the preparing stage
	int progress = 0;   // preparing %
	bool preparing_in_background = false; // a preparation runs while the current engine keys
	std::string preparing_quality;
	std::string quality; // engine in use ("" = none)
	int engine_w = 0, engine_h = 0;
	std::string gpu_name;
	bool auto_mode = false;
	AutoChoice auto_choice;
	std::set<std::string> prepared;
	std::string mode_id;
	// frame buffers for this size (0 = none); the render thread uses them only when they match
	// the size it last requested
	int frame_w = 0, frame_h = 0;
	uint64_t buffers_gen = 0;
	// worker errors while keying (render turns each new one into a fallback)
	uint64_t error_seq = 0;
	std::string last_error;
	uint64_t jobs = 0;
};

// Device buffers shared with the render thread (allocated and freed by the worker only).
struct FrameBuffers {
	int w = 0, h = 0;
	size_t pitch = 0;
	uint8_t *in[Scheduler::kInputs] = {};   // RGBA8 camera frames (render copies in)
	uint8_t *out[Scheduler::kOutputs] = {}; // RGBA8 fg + alpha (render copies out)
	cudaEvent_t in_ready[Scheduler::kInputs] = {};    // recorded by render after its copy in
	cudaEvent_t out_consumed[Scheduler::kOutputs] = {}; // recorded by render after its copy out
	uint64_t gen = 0;
};

class Worker {
public:
	Worker(int cuda_device, std::string name);
	~Worker(); // stops and joins (waits for at most one in-flight inference)

	Scheduler &scheduler() { return sched_; }
	void request(const WorkerRequest &r);
	WorkerRequest current_request() const;
	WorkerStatus status() const;
	uint64_t status_seq() const { return status_seq_.load(std::memory_order_acquire); }
	// Buffers for (w, h) if allocated, else false.
	bool buffers(int w, int h, FrameBuffers &out) const;
	// Stop without joining (the destructor joins). Safe from any thread.
	void stop();

private:
	void run();
	void reconcile();
	void process(const Scheduler::Job &job);
	void set_status(const std::function<void(WorkerStatus &)> &f);
	void report_error(const std::string &msg);
	bool realloc_buffers(int w, int h);
	void free_buffers();

	const int dev_;
	const std::string name_;
	Scheduler sched_;
	std::thread thread_;
	std::atomic<bool> stop_{false};

	mutable std::mutex req_mu_;
	WorkerRequest req_;
	uint64_t req_seq_ = 0;

	mutable std::mutex st_mu_;
	WorkerStatus st_;
	std::atomic<uint64_t> status_seq_{0};

	mutable std::mutex buf_mu_;
	FrameBuffers bufs_;

	// worker-thread-only state
	cudaStream_t stream_ = nullptr;
	cudaEvent_t done_ = nullptr;
	float *tmp_ = nullptr, *mask_ = nullptr, *scratch_ = nullptr;
	size_t scratch_floats_ = 0;
	std::unique_ptr<gpu::TrtRunner> runner_;
	std::string runner_quality_, runner_path_;
	std::string trt_tried_dir_, trt_err_;
	std::string weights_sha_, weights_sha_for_;
	int64_t weights_stamp_ = 0;
	WorkerRequest seen_;
	uint64_t seen_seq_ = ~0ull;
	std::string failed_key_; // quality+weights that failed to prepare/load (don't retry in a loop)
	uint64_t failed_clicks_ = 0;
	gpu::HintParams hint_;
	DebugFail debug_ = DebugFail::None;
	std::chrono::steady_clock::time_point last_full_{};
	gpu::DeviceInfo dev_info_;
	std::string table_path_;
	std::optional<PerfTable> table_;
};

} // namespace ck
