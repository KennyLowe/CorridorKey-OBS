#include "worker.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "ck-log.hpp"
#include "core/cache_key.hpp"
#include "gpu/dll_loader.hpp"
#include "gpu/trt_runner.hpp"

namespace ck {

namespace fs = std::filesystem;
using clock_ = std::chrono::steady_clock;

DebugFail debug_fail_from_string(const std::string &s)
{
	if (s == "worker_error")
		return DebugFail::WorkerError;
	if (s == "stall")
		return DebugFail::Stall;
	return DebugFail::None;
}

const char *to_string(Phase p)
{
	switch (p) {
	case Phase::Starting:
		return "starting";
	case Phase::NoWeights:
		return "no_weights";
	case Phase::Unsupported:
		return "unsupported";
	case Phase::Preparing:
		return "preparing";
	case Phase::Failed:
		return "failed";
	case Phase::Ready:
		return "ready";
	}
	return "?";
}

static fs::path u8path(const std::string &s)
{
	return fs::path((const char8_t *)s.c_str());
}

Worker::Worker(int cuda_device, std::string name) : dev_(cuda_device), name_(std::move(name))
{
	thread_ = std::thread([this] { run(); });
}

Worker::~Worker()
{
	stop();
	if (thread_.joinable())
		thread_.join();
}

void Worker::stop()
{
	stop_ = true;
	sched_.stop();
}

void Worker::request(const WorkerRequest &r)
{
	std::lock_guard lk(req_mu_);
	if (r == req_)
		return;
	req_ = r;
	req_seq_++;
}

WorkerRequest Worker::current_request() const
{
	std::lock_guard lk(req_mu_);
	return req_;
}

WorkerStatus Worker::status() const
{
	std::lock_guard lk(st_mu_);
	return st_;
}

bool Worker::buffers(int w, int h, FrameBuffers &out) const
{
	std::lock_guard lk(buf_mu_);
	if (bufs_.w != w || bufs_.h != h || !bufs_.in[0])
		return false;
	out = bufs_;
	return true;
}

void Worker::set_status(const std::function<void(WorkerStatus &)> &f)
{
	std::lock_guard lk(st_mu_);
	WorkerStatus before = st_;
	f(st_);
	st_.seq = before.seq + 1;
	status_seq_.store(st_.seq, std::memory_order_release);
}

void Worker::report_error(const std::string &msg)
{
	set_status([&](WorkerStatus &s) {
		s.error_seq++;
		s.last_error = msg;
	});
}

// ------------------------------------------------------------------------------- buffers

void Worker::free_buffers()
{
	std::lock_guard lk(buf_mu_);
	for (auto &p : bufs_.in)
		cudaFree(std::exchange(p, nullptr));
	for (auto &p : bufs_.out)
		cudaFree(std::exchange(p, nullptr));
	for (auto &e : bufs_.in_ready)
		if (e)
			cudaEventDestroy(std::exchange(e, nullptr));
	for (auto &e : bufs_.out_consumed)
		if (e)
			cudaEventDestroy(std::exchange(e, nullptr));
	bufs_.w = bufs_.h = 0;
	cudaFree(std::exchange(tmp_, nullptr));
	cudaFree(std::exchange(mask_, nullptr));
}

bool Worker::realloc_buffers(int w, int h)
{
	// The render thread stops using the old buffers as soon as it requests a new size, and
	// cudaFree waits for any of its copies still queued on the GPU.
	free_buffers();
	sched_.reset();
	FrameBuffers b;
	b.w = w;
	b.h = h;
	b.pitch = (size_t)w * 4;
	bool ok = true;
	for (auto &p : b.in)
		ok = ok && cudaMalloc(&p, b.pitch * h) == cudaSuccess;
	for (auto &p : b.out)
		ok = ok && cudaMalloc(&p, b.pitch * h) == cudaSuccess;
	for (auto &e : b.in_ready)
		ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
	for (auto &e : b.out_consumed)
		ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
	ok = ok && cudaMalloc(&tmp_, sizeof(float) * w * h) == cudaSuccess;
	ok = ok && cudaMalloc(&mask_, sizeof(float) * w * h) == cudaSuccess;
	uint64_t gen;
	{
		std::lock_guard lk(buf_mu_);
		gen = bufs_.gen + 1;
		bufs_ = b;
		bufs_.gen = gen;
	}
	if (!ok) {
		cudaGetLastError();
		free_buffers();
		log(LOG_ERROR, "%s: out of GPU memory allocating %dx%d frame buffers", name_.c_str(), w, h);
		set_status([&](WorkerStatus &s) {
			s.phase = Phase::Failed;
			s.reason = "out of GPU memory for the frame buffers";
			s.frame_w = s.frame_h = 0;
		});
		return false;
	}
	set_status([&](WorkerStatus &s) {
		s.frame_w = w;
		s.frame_h = h;
		s.buffers_gen = gen;
	});
	return true;
}

// ------------------------------------------------------------------------------- setup

static std::optional<PerfTable> load_perf_table(const std::string &path)
{
	std::ifstream f(u8path(path), std::ios::binary);
	if (!f)
		return std::nullopt;
	std::stringstream ss;
	ss << f.rdbuf();
	return parse_perf_table(ss.str());
}

void Worker::reconcile()
{
	static const auto kRecheck = std::chrono::seconds(2);

	WorkerRequest r;
	uint64_t rs;
	{
		std::lock_guard lk(req_mu_);
		r = req_;
		rs = req_seq_;
	}
	hint_ = r.hint;
	debug_ = r.debug;

	if (r.frame_w > 0 && r.frame_h > 0) {
		std::unique_lock lk(buf_mu_);
		bool differ = bufs_.w != r.frame_w || bufs_.h != r.frame_h;
		lk.unlock();
		if (differ)
			realloc_buffers(r.frame_w, r.frame_h);
	}

	WorkerStatus cur = status();
	bool changed = rs != seen_seq_;
	bool due = clock_::now() - last_full_ > kRecheck && (cur.phase != Phase::Ready || r.quality == "auto");
	if (!changed && !due)
		return;
	seen_seq_ = rs;
	last_full_ = clock_::now();

	auto drop_engine = [&] {
		runner_.reset();
		runner_quality_.clear();
		runner_path_.clear();
	};
	auto set_phase = [&](Phase p, const std::string &why) {
		set_status([&](WorkerStatus &s) {
			s.phase = p;
			s.reason = why;
			if (p != Phase::Ready) {
				s.quality.clear();
				s.engine_w = s.engine_h = 0;
			}
		});
	};

	// 1. TensorRT (once per folder)
	if (r.trt_dir != trt_tried_dir_) {
		trt_tried_dir_ = r.trt_dir;
		trt_err_ = gpu::load_tensorrt(r.trt_dir);
		if (!trt_err_.empty())
			log(LOG_WARNING, "%s: TensorRT unavailable: %s", name_.c_str(), trt_err_.c_str());
		else
			log(LOG_INFO, "%s: TensorRT %s loaded from %s", name_.c_str(), gpu::tensorrt_version().c_str(),
			    r.trt_dir.c_str());
	}
	if (!trt_err_.empty()) {
		drop_engine();
		set_phase(Phase::Unsupported, trt_err_);
		return;
	}
	if (dev_info_.cuda_device != dev_)
		dev_info_ = gpu::device_info(dev_);
	const gpu::DeviceInfo &dev = dev_info_;
	if (dev.sm_major < 8 || dev.driver_version.empty()) {
		drop_engine();
		set_phase(Phase::Unsupported, dev.driver_version.empty() ? "NVIDIA driver (NVML) not available"
									 : dev.name + " is not an RTX (Ampere or newer) GPU");
		return;
	}
	set_status([&](WorkerStatus &s) { s.gpu_name = dev.name; });

	// 2. weights
	if (r.weights_path.empty()) {
		drop_engine();
		set_phase(Phase::NoWeights, "Weights not set");
		return;
	}
	std::error_code ec;
	auto wp = u8path(r.weights_path);
	if (!fs::is_regular_file(wp, ec)) {
		drop_engine();
		set_phase(Phase::NoWeights, "Weights file not found: " + r.weights_path);
		return;
	}
	int64_t stamp = (int64_t)fs::file_size(wp, ec) ^ (int64_t)fs::last_write_time(wp, ec).time_since_epoch().count();
	if (r.weights_path != weights_sha_for_ || stamp != weights_stamp_) {
		if (cur.phase != Phase::Ready)
			set_phase(Phase::Starting, "checking weights");
		auto sha = sha256_file(r.weights_path);
		if (!sha) {
			drop_engine();
			set_phase(Phase::NoWeights, "Cannot read the weights file: " + r.weights_path);
			return;
		}
		weights_sha_ = *sha;
		weights_sha_for_ = r.weights_path;
		weights_stamp_ = stamp;
	}

	// 3. engine cache: which settings are prepared for this GPU, driver, TensorRT and weights
	auto manifest = read_manifest(r.cache_dir + "/manifest.json");
	if (!manifest)
		manifest = Manifest{};
	KeyComponents base{dev.uuid, dev.driver_version, gpu::tensorrt_version(), weights_sha_, 0, 0, false};
	std::set<std::string> prepared;
	std::string stale_note;
	auto engine_for = [&](const Quality &q, LookupResult *res = nullptr) {
		KeyComponents c = base;
		c.internal_w = q.w;
		c.internal_h = q.h;
		c.refiner = q.refiner;
		auto lr = lookup(*manifest, c);
		if (res)
			*res = lr;
		return lr;
	};
	for (const auto &q : all_qualities()) {
		LookupResult lr = engine_for(q);
		if (lr.kind == Lookup::Hit)
			prepared.insert(q.id);
		else if (lr.kind == Lookup::Stale && stale_note.empty())
			stale_note = lr.why;
	}

	// 4. which setting
	std::string q = r.quality;
	AutoChoice choice;
	std::string mode = live_mode_id(r.frame_h > 0 ? r.frame_h : 1080, r.fps);
	if (r.quality == "auto") {
		if (table_path_ != r.perf_table_path) {
			table_path_ = r.perf_table_path;
			table_ = load_perf_table(r.perf_table_path);
			if (!table_)
				log(LOG_WARNING, "%s: cannot read perf table %s", name_.c_str(), r.perf_table_path.c_str());
		}
		choice = auto_select(table_ ? *table_ : PerfTable{}, dev.name, mode, prepared);
		q = choice.quality;
	}
	set_status([&](WorkerStatus &s) {
		s.prepared = prepared;
		s.auto_mode = r.quality == "auto";
		s.auto_choice = choice;
		s.mode_id = mode;
	});

	auto want = quality_from_id(q.empty() ? choice.best_overall : q);
	if (!want) {
		drop_engine();
		set_phase(Phase::Failed, "Unknown quality setting: " + r.quality);
		return;
	}
	if (!prepared.count(want->id)) {
		drop_engine();
		std::string why = quality_label(*want) + " is not prepared for this GPU" +
				  (stale_note.empty() ? "" : " (cached engine is stale: " + stale_note + ")") +
				  ". Run: ckbench prepare --size " + std::to_string(want->w) + "x" + std::to_string(want->h) +
				  " --refiner " + (want->refiner ? "1" : "0");
		set_phase(Phase::Failed, why);
		return;
	}

	// 5. load the engine (on this thread; the render thread keeps falling back meanwhile)
	LookupResult hit;
	engine_for(*want, &hit);
	const std::string path = hit.entry->engine_path;
	if (runner_ && runner_path_ == path) {
		if (cur.phase != Phase::Ready || cur.quality != want->id)
			set_status([&](WorkerStatus &s) {
				s.phase = Phase::Ready;
				s.reason.clear();
				s.quality = want->id;
				s.engine_w = runner_->width();
				s.engine_h = runner_->height();
			});
		return;
	}
	const std::string fkey = path;
	if (fkey == failed_key_ && !changed)
		return; // already failed; wait for a settings change
	drop_engine();
	auto t0 = clock_::now();
	std::string err;
	runner_ = gpu::TrtRunner::load(path, err);
	if (!runner_) {
		failed_key_ = fkey;
		log(LOG_ERROR, "%s: engine load failed: %s", name_.c_str(), err.c_str());
		set_phase(Phase::Failed, "Engine load failed: " + err);
		return;
	}
	failed_key_.clear();
	runner_quality_ = want->id;
	runner_path_ = path;
	double ms = std::chrono::duration<double, std::milli>(clock_::now() - t0).count();
	log(LOG_INFO, "%s: engine %s loaded in %.0f ms (%s)", name_.c_str(), want->id.c_str(), ms, path.c_str());
	set_status([&](WorkerStatus &s) {
		s.phase = Phase::Ready;
		s.reason.clear();
		s.quality = want->id;
		s.engine_w = runner_->width();
		s.engine_h = runner_->height();
	});
}

// ------------------------------------------------------------------------------- jobs

void Worker::process(const Scheduler::Job &job)
{
	if (debug_ == DebugFail::Stall) { // SC-004 test hook: the worker stops producing mattes
		sched_.release_input(job.slot);
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		return;
	}
	if (debug_ == DebugFail::WorkerError) { // SC-004 test hook: every job fails
		sched_.release_input(job.slot);
		report_error("debug_fail=worker_error: simulated keyer failure");
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		return;
	}
	FrameBuffers b;
	{
		std::lock_guard lk(buf_mu_);
		b = bufs_;
	}
	if (!runner_ || !b.in[0]) {
		sched_.release_input(job.slot);
		return;
	}
	const int w = runner_->width(), h = runner_->height();
	size_t need = gpu::preprocess_scratch_floats(b.w, b.h, w, h);
	if (need > scratch_floats_) {
		cudaFree(std::exchange(scratch_, nullptr));
		scratch_floats_ = 0;
		if (cudaMalloc(&scratch_, need * sizeof(float)) != cudaSuccess) {
			cudaGetLastError();
			sched_.release_input(job.slot);
			report_error("out of GPU memory (preprocess scratch)");
			return;
		}
		scratch_floats_ = need;
	}

	auto t0 = clock_::now();
	cudaError_t e = cudaStreamWaitEvent(stream_, b.in_ready[job.slot], 0);
	if (e == cudaSuccess)
		e = gpu::hint(b.in[job.slot], b.pitch, b.w, b.h, hint_, tmp_, mask_, stream_);
	if (e == cudaSuccess)
		e = gpu::preprocess(b.in[job.slot], b.pitch, mask_, b.w, b.h, runner_->input(), w, h, scratch_, stream_);
	std::string err;
	if (e == cudaSuccess)
		err = runner_->enqueue(stream_);
	int o = -1;
	if (e == cudaSuccess && err.empty()) {
		o = sched_.begin_output();
		e = cudaStreamWaitEvent(stream_, b.out_consumed[o], 0);
		if (e == cudaSuccess)
			e = gpu::postprocess(runner_->alpha(), runner_->fg(), w, h, b.out[o], b.pitch, b.w, b.h, stream_);
	}
	if (e == cudaSuccess && err.empty())
		e = cudaEventRecord(done_, stream_);
	if (e == cudaSuccess && err.empty())
		e = cudaEventSynchronize(done_); // blocking-sync event: this thread sleeps, it doesn't spin
	if (e == cudaSuccess)
		e = cudaGetLastError();
	double ms = std::chrono::duration<double, std::milli>(clock_::now() - t0).count();
	sched_.release_input(job.slot);
	if (e != cudaSuccess || !err.empty()) {
		if (o >= 0)
			sched_.abort_output(o);
		if (err.empty())
			err = std::string("CUDA error: ") + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")";
		report_error(err);
		return;
	}
	sched_.complete_output(o, job, ms);
	set_status([](WorkerStatus &s) { s.jobs++; });
}

void Worker::run()
{
	if (cudaSetDevice(dev_) != cudaSuccess ||
	    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
	    cudaEventCreateWithFlags(&done_, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess) {
		cudaGetLastError();
		set_status([](WorkerStatus &s) {
			s.phase = Phase::Unsupported;
			s.reason = "cannot initialise CUDA on the OBS GPU";
		});
		return;
	}
	while (!stop_) {
		try {
			reconcile();
			auto job = sched_.take(std::chrono::milliseconds(20));
			if (job && !stop_)
				process(*job);
		} catch (const std::exception &ex) { // never let anything escape into OBS
			report_error(std::string("worker exception: ") + ex.what());
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}
	runner_.reset();
	free_buffers();
	cudaFree(std::exchange(scratch_, nullptr));
	if (done_)
		cudaEventDestroy(done_);
	if (stream_)
		cudaStreamDestroy(stream_);
}

} // namespace ck
