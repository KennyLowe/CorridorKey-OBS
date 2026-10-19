#include "worker.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "ck-log.hpp"
#include "plugin_config.hpp"
#include "prepare_client.hpp"
#include "core/cache_key.hpp"
#include "gpu/dll_loader.hpp"
#include "gpu/trt_runner.hpp"

namespace ck {

namespace fs = std::filesystem;
using clock_ = std::chrono::steady_clock;

std::atomic<int> g_heavy_cuda{0};

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
	case Phase::Downloading:
		return "downloading";
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
	HeavyCuda heavy;
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

struct ModelInfo {
	bool found = false;
	std::string sha256;
	long long size = 400000000;
};

// The pinned official model (models/manifest.json, shipped in the plugin's data folder).
static ModelInfo read_model(const std::string &manifest_path, const std::string &model_id)
{
	ModelInfo m;
	std::ifstream f(u8path(manifest_path), std::ios::binary);
	if (!f)
		return m;
	try {
		auto j = nlohmann::json::parse(f);
		for (const auto &e : j.at("models"))
			if (e.at("model_id").get<std::string>() == model_id) {
				m.found = true;
				m.sha256 = e.at("sha256").get<std::string>();
				m.size = e.at("size").get<long long>();
			}
	} catch (...) {
	}
	return m;
}

// "CorridorKey_v1.0" -> "CorridorKey v1.0"
static std::string display_name(std::string id)
{
	for (auto &c : id)
		if (c == '_')
			c = ' ';
	return id;
}

static std::optional<PerfTable> load_perf_table(const std::string &path)
{
	std::ifstream f(u8path(path), std::ios::binary);
	if (!f)
		return std::nullopt;
	std::stringstream ss;
	ss << f.rdbuf();
	return parse_perf_table(ss.str());
}

// Helpers: fetch-weights (download), prepare (engine build), check-models (daily notice).
void Worker::poll_helpers(const WorkerRequest &)
{
	if (fetch_) {
		for (const auto &e : fetch_->poll()) {
			if (e.kind == HelperEvent::Kind::Progress)
				set_status([&](WorkerStatus &s) {
					s.progress = e.pct;
					if (e.total > 0) {
						s.dl_bytes = e.bytes;
						s.dl_total = e.total;
					}
				});
		}
		if (fetch_->finished()) {
			auto res = fetch_->result();
			if (res && res->ok) {
				fetch_error_.clear();
				log(LOG_INFO, "%s: weights downloaded and verified: %s (sha256 %s)", name_.c_str(),
				    res->weights_path.c_str(), res->sha256.c_str());
			} else {
				fetch_error_ = res ? "Download failed (" + res->error + "): " + res->msg +
							     " Press \"Download weights\" to retry."
						   : "Download failed: the helper exited with code " +
							     std::to_string(fetch_->exit_code()) + " (see " + fetch_->log_path() + ")";
				log(LOG_ERROR, "%s: %s", name_.c_str(), fetch_error_.c_str());
			}
			std::string lp = fetch_->log_path();
			set_status([&](WorkerStatus &s) { s.last_helper_log = lp; });
			fetch_.reset();
			force_recheck_ = true;
		}
	}
	if (prep_) {
		const std::string label = quality_label(*quality_from_id(prep_quality_));
		for (const auto &e : prep_->poll())
			if (e.kind == HelperEvent::Kind::Progress)
				set_status([&](WorkerStatus &s) {
					s.progress = e.pct;
					s.reason = "preparing " + label + " \xE2\x80\x94 " +
						   (e.stage == "engine" ? std::string("building engine")
						    : e.stage == "onnx" ? std::string("exporting ONNX")
									: e.stage);
				});
		if (prep_->finished()) {
			auto res = prep_->result();
			if (res && res->ok) {
				prep_fail_key_.clear();
				prep_error_.clear();
				log(LOG_INFO, "%s: prepared %s in %.0f s: %s", name_.c_str(), prep_quality_.c_str(), res->build_s,
				    res->engine_path.c_str());
			} else {
				prep_fail_key_ = prep_key_;
				prep_error_ = res ? res->error + ": " + res->msg
						  : "the helper exited with code " + std::to_string(prep_->exit_code()) + " (see " +
							    prep_->log_path() + ")";
				log(LOG_ERROR, "%s: preparing %s failed: %s", name_.c_str(), prep_quality_.c_str(),
				    prep_error_.c_str());
			}
			std::string lp = prep_->log_path();
			set_status([&](WorkerStatus &s) {
				s.last_helper_log = lp;
				s.preparing_in_background = false;
				s.preparing_quality.clear();
			});
			prep_.reset();
			force_recheck_ = true;
		}
	}
	if (check_ && check_->finished()) {
		auto res = check_->result();
		if (res && res->ok) {
			config::set_model_check((long long)std::time(nullptr), res->newer);
			if (!res->newer.empty())
				log(LOG_INFO, "%s: newer CorridorKey model available: %s (not switched)", name_.c_str(),
				    res->newer.front().c_str());
		} else {
			log(LOG_INFO, "%s: model check failed (non-fatal): %s", name_.c_str(), res ? res->msg.c_str() : "?");
		}
		check_.reset();
		set_status([&](WorkerStatus &s) { s.newer_models = config::newer_models(); });
	}
}

// At most once a day, and never while keying (FR-006a).
void Worker::maybe_check_models(const WorkerRequest &r, bool keying)
{
	if (check_ || keying || r.helper_cmd.empty())
		return;
	long long now = (long long)std::time(nullptr);
	if (now - config::last_model_check() < 24 * 3600)
		return;
	config::set_model_check(now, config::newer_models()); // throttle even if the helper fails
	std::vector<std::string> args{"check-models", "--json"};
	if (!r.manifest_path.empty())
		args.insert(args.end(), {"--manifest", r.manifest_path});
	if (const char *api = std::getenv("CK_MODELS_API"))
		args.insert(args.end(), {"--api-url", api});
	std::string err;
	check_ = HelperProcess::start(r.helper_cmd, args, "check-models", err);
	if (!check_)
		log(LOG_INFO, "%s: model check not started: %s", name_.c_str(), err.c_str());
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
	// Button presses count only from this session on.
	if (download_handled_ == ~0ull)
		download_handled_ = r.download_clicks;
	if (prepare_handled_ == ~0ull)
		prepare_handled_ = r.prepare_clicks;

	if (r.frame_w > 0 && r.frame_h > 0) {
		std::unique_lock lk(buf_mu_);
		bool differ = bufs_.w != r.frame_w || bufs_.h != r.frame_h;
		lk.unlock();
		if (differ)
			realloc_buffers(r.frame_w, r.frame_h);
	}

	poll_helpers(r);

	WorkerStatus cur = status();
	bool changed = rs != seen_seq_ || force_recheck_;
	bool busy = fetch_ || prep_;
	bool due = clock_::now() - last_full_ > (busy ? std::chrono::milliseconds(250)
						      : std::chrono::duration_cast<std::chrono::milliseconds>(kRecheck)) &&
		   (cur.phase != Phase::Ready || r.quality == "auto" || busy);
	if (!changed && !due)
		return;
	force_recheck_ = false;
	seen_seq_ = rs;
	last_full_ = clock_::now();
	maybe_check_models(r, cur.phase == Phase::Ready && runner_ != nullptr);

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
	auto set_model = [&](const std::string &label, bool verified) {
		set_status([&](WorkerStatus &s) {
			s.model_label = label;
			s.model_verified = verified;
			s.newer_models = config::newer_models();
		});
	};

	// 1. TensorRT (once per folder)
	if (r.trt_dir != trt_tried_dir_) {
		HeavyCuda heavy;
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

	// 2. weights: present, or downloaded on request (in-plugin, via the helper; FR-006)
	const ModelInfo model = read_model(r.manifest_path, r.model_id);
	const std::string model_name = display_name(r.model_id);
	if (r.download_clicks != download_handled_) {
		download_handled_ = r.download_clicks;
		if (!r.licence_ok) {
			fetch_error_ = "Accept the CorridorKey licence (tick the box under Model) before downloading.";
		} else if (!fetch_) {
			std::vector<std::string> args{"fetch-weights", "--model", r.model_id, "--dest-dir", r.weights_dir,
						      "--json"};
			if (!r.manifest_path.empty())
				args.insert(args.end(), {"--manifest", r.manifest_path});
			std::string err;
			fetch_ = HelperProcess::start(r.helper_cmd, args, "fetch-weights", err);
			if (!fetch_) {
				fetch_error_ = err;
			} else {
				fetch_error_.clear();
				log(LOG_INFO, "%s: downloading %s: %s", name_.c_str(), r.model_id.c_str(),
				    fetch_->command_line().c_str());
				set_status([](WorkerStatus &s) {
					s.progress = 0;
					s.dl_bytes = s.dl_total = 0;
				});
			}
		}
	}
	std::error_code ec;
	auto wp = u8path(r.weights_path);
	const bool have_weights = !r.weights_path.empty() && fs::is_regular_file(wp, ec);
	if (fetch_ && !(have_weights && runner_)) {
		WorkerStatus st = status();
		char buf[160];
		snprintf(buf, sizeof(buf), "downloading %d%% (%.0f of %.0f MB)", st.progress, (double)st.dl_bytes / 1e6,
			 (double)(st.dl_total ? st.dl_total : model.size) / 1e6);
		set_model(model_name + ", " + buf, false);
		drop_engine();
		set_phase(Phase::Downloading, buf);
		return;
	}
	if (!have_weights) {
		drop_engine();
		set_model(model_name + ", not downloaded", false);
		std::string why = !fetch_error_.empty() ? fetch_error_
				  : r.weights_path.empty() ? std::string("Weights not set")
							   : "Weights not downloaded: click \"Download weights\" (about " +
								     std::to_string((model.size + 500000) / 1000000) + " MB)";
		set_phase(Phase::NoWeights, why);
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
	const bool verified = model.found && weights_sha_ == model.sha256;
	set_model(verified ? model_name + ", verified"
			   : "unverified model (sha256 " + weights_sha_.substr(0, 12) + "\xE2\x80\xA6 is not in the manifest)",
		  verified);

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
		// An unmeasured (unverified) model: Auto uses the fastest setting until it's benchmarked.
		choice = auto_select(table_ && verified ? *table_ : PerfTable{}, dev.name, mode, prepared);
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

	// 4b. preparation (FR-007): on a miss, or when "Prepare selected" is pressed
	auto start_prepare = [&](const Quality &pq) {
		std::vector<std::string> args{"prepare",     "--weights", r.weights_path,
					      "--size",      std::to_string(pq.w) + "x" + std::to_string(pq.h),
					      "--refiner",   pq.refiner ? "1" : "0",
					      "--cache-dir", r.cache_dir,
					      "--json"};
		std::string err;
		prep_ = HelperProcess::start(r.helper_cmd, args, "prepare-" + pq.id, err);
		if (!prep_) {
			prep_fail_key_ = pq.id + weights_sha_;
			prep_error_ = err;
			return;
		}
		prep_quality_ = pq.id;
		prep_key_ = pq.id + weights_sha_;
		log(LOG_INFO, "%s: preparing %s%s: %s", name_.c_str(), pq.id.c_str(),
		    stale_note.empty() ? "" : (" (cached engine is stale: " + stale_note + ")").c_str(),
		    prep_->command_line().c_str());
		const bool bg = runner_ != nullptr;
		set_status([&](WorkerStatus &s) {
			s.progress = 0;
			s.preparing_quality = pq.id;
			s.preparing_in_background = bg;
		});
	};
	if (r.prepare_clicks != prepare_handled_) {
		prepare_handled_ = r.prepare_clicks;
		auto pq = quality_from_id(r.quality == "auto" ? choice.best_overall : r.quality);
		if (pq && !prep_ && !prepared.count(pq->id)) {
			prep_fail_key_.clear();
			start_prepare(*pq);
		}
	}
	if (!prepared.count(want->id)) {
		const std::string key = want->id + weights_sha_;
		if (!prep_ && key != prep_fail_key_)
			start_prepare(*want);
		if (prep_) {
			if (!runner_) {
				WorkerStatus st = status();
				set_phase(Phase::Preparing, st.reason.rfind("preparing", 0) == 0
								    ? st.reason
								    : "preparing " + quality_label(*want) + " \xE2\x80\x94 starting");
			}
			return; // the current engine (if any) keeps keying meanwhile
		}
		drop_engine();
		if (prep_error_.rfind("weights_invalid", 0) == 0) { // not usable weights: pass through (FR-011)
			set_model("invalid weights file", false);
			set_phase(Phase::NoWeights, "Invalid weights file: " + prep_error_);
			return;
		}
		set_phase(Phase::Failed, "Preparing " + quality_label(*want) + " failed: " + prep_error_ +
						 " (press \"Prepare selected\" to retry)");
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
	HeavyCuda heavy; // until the warm-up below has finished
	drop_engine();
	set_phase(Phase::Starting, "loading engine " + want->id); // the render thread stops using CUDA meanwhile
	auto t0 = clock_::now();
	std::string err;
	runner_ = gpu::TrtRunner::load(path, err);
	if (!runner_) {
		failed_key_ = fkey;
		log(LOG_ERROR, "%s: engine load failed: %s", name_.c_str(), err.c_str());
		set_phase(Phase::Failed, "Engine load failed: " + err);
		return;
	}
	// Warm up: one full pass (hint, preprocess, TensorRT, postprocess) so CUDA's lazy module loading
	// and TensorRT's first-run setup happen now, not while the render thread is mapping textures.
	{
		FrameBuffers b;
		{
			std::lock_guard lk(buf_mu_);
			b = bufs_;
		}
		cudaError_t e = cudaSuccess;
		if (b.in[0] && tmp_ && mask_) {
			size_t need = gpu::preprocess_scratch_floats(b.w, b.h, runner_->width(), runner_->height());
			if (need > scratch_floats_) {
				cudaFree(std::exchange(scratch_, nullptr));
				scratch_floats_ = cudaMalloc(&scratch_, need * sizeof(float)) == cudaSuccess ? need : 0;
			}
			if (scratch_floats_) {
				e = gpu::hint(b.in[0], b.pitch, b.w, b.h, hint_, tmp_, mask_, stream_);
				if (e == cudaSuccess)
					e = gpu::preprocess(b.in[0], b.pitch, mask_, b.w, b.h, runner_->input(), runner_->width(),
							    runner_->height(), scratch_, stream_);
			}
		}
		std::string werr = e == cudaSuccess ? runner_->enqueue(stream_) : "";
		if (e == cudaSuccess && werr.empty() && b.out[0])
			e = gpu::postprocess(runner_->alpha(), runner_->fg(), runner_->width(), runner_->height(), b.out[0],
					     b.pitch, b.w, b.h, stream_);
		cudaError_t se = cudaStreamSynchronize(stream_);
		if (e != cudaSuccess || se != cudaSuccess || !werr.empty()) {
			cudaGetLastError();
			log(LOG_WARNING, "%s: engine warm-up failed: %s", name_.c_str(),
			    werr.empty() ? cudaGetErrorString(e != cudaSuccess ? e : se) : werr.c_str());
		}
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
		HeavyCuda heavy;
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
	std::optional<HeavyCuda> heavy_init(std::in_place);
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
	heavy_init.reset();
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
	fetch_.reset(); // kills the helper process trees
	prep_.reset();
	check_.reset();
	HeavyCuda heavy_teardown;
	runner_.reset();
	free_buffers();
	cudaFree(std::exchange(scratch_, nullptr));
	if (done_)
		cudaEventDestroy(done_);
	if (stream_)
		cudaStreamDestroy(stream_);
}

} // namespace ck
