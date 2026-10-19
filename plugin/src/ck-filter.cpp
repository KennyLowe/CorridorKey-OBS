/*
corridorkey-obs: the "CorridorKey" video filter.
Copyright (C) 2026 Kenny Lowe. GPL-2.0-or-later (see plugin/LICENSE).

Render-thread rules (Constitution III, research D4):
  - video_render never waits on inference. Per frame it renders the source into a texrender,
    copies that into a CUDA-registered texture, queues an async copy into a free ring slot (map ->
    cudaMemcpy2DFromArrayAsync -> event -> unmap) and submits it to the scheduler, which never
    blocks. If the worker has finished a newer matte, it queues an async copy of it into the
    output texture. Then it draws: the CorridorKey composite, or the fallback.
  - No CUDA allocation, free or synchronisation happens on the render thread in steady state.
    Texture (re)registration happens only when the source size changes.
  - The fallback (colour key / last matte / pass-through) is a plain OBS shader and does not
    depend on CUDA or the worker.
*/
#include "ck-filter.hpp"

#include <graphics/graphics.h>
#include <graphics/vec4.h>
#include <util/platform.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <functional>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "ck-log.hpp"
#include "plugin_config.hpp"
#include "core/auto_select.hpp"
#include "core/state.hpp"
#include "core/stats.hpp"
#include "gpu/interop.hpp"
#include "third_party/obs-websocket-api.h"
#include "worker.hpp"

#ifndef CK_BENCH_DIR
#define CK_BENCH_DIR ""
#endif

namespace ck {
namespace {

constexpr const char *kWeightsUrl = "https://huggingface.co/nikopueringer/CorridorKey_v1.0";
constexpr const char *kLicenceUrl = "https://github.com/nikopueringer/CorridorKey";
constexpr const char *kAttribution =
	"Powered by CorridorKey by Corridor Digital (CC BY-NC-SA 4.0). Not affiliated.";

// ------------------------------------------------------------------------------- settings

struct Settings {
	std::string model_id = "CorridorKey_v1.0";
	bool licence_ack = false;
	long long download_requested = 0, prepare_requested = 0;
	std::string weights_path;
	std::string quality = "auto";
	std::string hint_mode = "fixed"; // fixed / calibrated / manual
	float hint_rgb[3] = {0.0f, 1.0f, 0.0f};
	float hint_lo = 0.30f, hint_hi = 0.70f;
	FallbackMode fallback = FallbackMode::ColorKey;
	int fallback_timeout_ms = 250;
	bool show_hint = false;
	std::string helper_cmd;
	std::string trt_dir;
	DebugFail debug = DebugFail::None;

	gpu::HintParams hint() const
	{
		if (hint_mode == "fixed")
			return gpu::HintParams{1.0f, 0.05f, 0.15f}; // the spike's chroma_hint
		float kd = hint_rgb[1] - std::max(hint_rgb[0], hint_rgb[2]);
		float lo = std::clamp(hint_lo, 0.0f, 0.99f);
		float hi = std::clamp(hint_hi, lo + 0.01f, 1.0f);
		return gpu::HintParams{std::max(kd, 0.02f), lo, hi};
	}
};

std::string default_helper_cmd()
{
	std::string bench = CK_BENCH_DIR;
	return bench.empty() ? "uv run ckbench" : "uv run --project \"" + bench + "\" ckbench";
}

std::string default_cache_dir()
{
	if (const char *d = std::getenv("CK_ENGINE_CACHE")) // tests
		return d;
	const char *la = std::getenv("LOCALAPPDATA");
	return std::string(la ? la : ".") + "\\corridorkey-obs\\engines";
}

// Where "Download weights" puts the file (CK_WEIGHTS_DIR overrides it, for tests).
std::string weights_dir()
{
	if (const char *d = std::getenv("CK_WEIGHTS_DIR"))
		return d;
	const char *la = std::getenv("LOCALAPPDATA");
	return std::string(la ? la : ".") + "\\corridorkey-obs\\weights";
}

std::string default_weights_path(const std::string &model_id)
{
	return weights_dir() + "\\" + model_id + ".safetensors";
}

// The model manifest shipped beside the plugin (CK_MODEL_MANIFEST overrides it, for tests).
std::string manifest_path()
{
	if (const char *m = std::getenv("CK_MODEL_MANIFEST"))
		return m;
	char *p = obs_module_file("manifest.json");
	std::string s = p ? p : "";
	bfree(p);
	return s;
}

void color_from_obs(long long c, float rgb[3])
{
	// obs colour property: 0xAABBGGRR
	rgb[0] = (float)(c & 0xff) / 255.0f;
	rgb[1] = (float)((c >> 8) & 0xff) / 255.0f;
	rgb[2] = (float)((c >> 16) & 0xff) / 255.0f;
}

long long color_to_obs(const float rgb[3])
{
	auto b = [](float v) { return (long long)std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f); };
	return 0xff000000LL | (b(rgb[2]) << 16) | (b(rgb[1]) << 8) | b(rgb[0]);
}

// ------------------------------------------------------------------------------- the filter

struct Snapshot {
	State state = State::NoWeights;
	std::string reason;
	int progress = 0;
	Readout readout;
	WorkerStatus worker;
	std::string status_text;
	double render_us_median = 0, render_us_p95 = 0, render_us_p99 = 0, render_us_max = 0;
	uint64_t render_frames = 0, mattes = 0, skipped_total = 0, submitted_total = 0, fallbacks = 0;
	uint64_t worker_errors = 0;
	double frame_ms = 0;
	int src_w = 0, src_h = 0;
	std::string output;
	double last_matte_age_ms = -1;
};

struct Filter {
	obs_source_t *ctx = nullptr;
	std::string name;

	std::mutex set_mu;
	Settings set;
	uint64_t set_gen = 1;

	int cuda_dev = -1;
	std::string unsupported;
	std::unique_ptr<Worker> worker;

	// ---- render thread only
	gs_texrender_t *tr = nullptr;
	gs_effect_t *fx_composite = nullptr, *fx_colorkey = nullptr;
	gs_texture_t *in_tex = nullptr, *out_tex = nullptr;
	std::unique_ptr<gpu::InteropTexture> in_io, out_io;
	int tex_w = 0, tex_h = 0;
	cudaStream_t copy_stream = nullptr;
	FrameBuffers bufs;
	bool have_bufs = false;
	uint64_t bufs_gen_seen = 0;
	StateMachine sm;
	Stats stats;
	bool have_matte = false;
	uint64_t tick = 0, rendered_tick = ~0ull, frame_no = 0;
	uint64_t worker_seq_seen = ~0ull, error_seq_seen = 0, skipped_seen = 0;
	WorkerStatus ws;
	uint64_t req_gen_sent = 0;
	int req_w = 0, req_h = 0;
	Settings rs; // render-thread copy of the settings
	uint64_t rs_gen = 0;
	std::deque<double> render_us;
	double render_us_max = 0;
	uint64_t mattes = 0, fallbacks = 0, worker_errors = 0;
	double last_matte_ms = 0;
	uint64_t last_snapshot_ns = 0, last_summary_ns = 0;
	Output last_output = Output::Passthrough;

	std::mutex snap_mu;
	Snapshot snap;
};

std::mutex g_reg_mu;
std::map<obs_source_t *, Filter *> g_filters;
std::mutex g_reap_mu;
std::vector<std::thread> g_reapers;

double now_ms()
{
	return (double)os_gettime_ns() / 1e6;
}

double obs_frame_ms()
{
	obs_video_info ovi{};
	if (obs_get_video_info(&ovi) && ovi.fps_num)
		return 1000.0 * (double)ovi.fps_den / (double)ovi.fps_num;
	return 1000.0 / 30.0;
}

std::string state_text(const Filter &f, const Snapshot &s)
{
	char buf[512];
	const auto &w = s.worker;
	std::string q = w.quality.empty() ? std::string() : quality_label(*quality_from_id(w.quality));
	std::string txt;
	switch (s.state) {
	case State::Keying:
	case State::Behind:
		snprintf(buf, sizeof(buf), "Keying \xE2\x80\x94 %dx%d, %s%s", w.engine_w, w.engine_h,
			 format_readout(s.readout).c_str(), s.state == State::Behind ? " \xE2\x80\x94 falling behind" : "");
		txt = buf;
		break;
	case State::Preparing:
		snprintf(buf, sizeof(buf), "Preparing \xE2\x80\x94 %s %d%%", s.reason.c_str(), s.progress);
		txt = buf;
		break;
	case State::Downloading:
		txt = "Downloading weights \xE2\x80\x94 " + w.model_label;
		break;
	case State::Ready:
		txt = "Ready (" + q + ") \xE2\x80\x94 waiting for frames";
		break;
	case State::Fallback:
		txt = "Fallback (" + std::string(to_string(f.rs.fallback)) + "): " + s.reason;
		break;
	case State::Unsupported:
		txt = "Unsupported: " + s.reason;
		break;
	case State::Failed:
		txt = "Failed: " + s.reason;
		break;
	case State::NoWeights:
		txt = s.reason.empty() ? "Weights not set" : s.reason;
		break;
	}
	if (w.preparing_in_background && !w.preparing_quality.empty())
		txt += "\nPreparing " + quality_label(*quality_from_id(w.preparing_quality)) + " in the background (" +
		       std::to_string(w.progress) + "%); the current setting keeps keying.";
	if (w.auto_mode && !w.mode_id.empty()) {
		txt += "\nAuto (" + w.mode_id + "): " +
		       (w.auto_choice.quality.empty() ? std::string("nothing prepared yet")
						      : quality_label(*quality_from_id(w.auto_choice.quality)));
		if (w.auto_choice.needs_prepare && !w.auto_choice.best_overall.empty())
			txt += "; best for this GPU: " + quality_label(*quality_from_id(w.auto_choice.best_overall)) +
			       " (needs preparing)";
		if (!w.auto_choice.note.empty())
			txt += "\n" + w.auto_choice.note;
	}
	return txt;
}

// ------------------------------------------------------------------------------- settings io

void read_settings(Filter *f, obs_data_t *s)
{
	Settings n;
	n.model_id = obs_data_get_string(s, "model_id");
	if (n.model_id.empty())
		n.model_id = "CorridorKey_v1.0";
	n.licence_ack = obs_data_get_bool(s, "licence_ack");
	n.download_requested = obs_data_get_int(s, "download_requested");
	n.prepare_requested = obs_data_get_int(s, "prepare_requested");
	if (n.licence_ack)
		config::acknowledge_licence(n.model_id);
	n.weights_path = obs_data_get_string(s, "weights_path");
	n.quality = obs_data_get_string(s, "quality");
	if (n.quality.empty())
		n.quality = "auto";
	n.hint_mode = obs_data_get_string(s, "hint_mode");
	color_from_obs(obs_data_get_int(s, "hint_color"), n.hint_rgb);
	n.hint_lo = (float)obs_data_get_double(s, "hint_lo");
	n.hint_hi = (float)obs_data_get_double(s, "hint_hi");
	n.fallback = fallback_mode_from_string(obs_data_get_string(s, "fallback_mode"));
	n.fallback_timeout_ms = (int)std::clamp<long long>(obs_data_get_int(s, "fallback_timeout_ms"), 50, 2000);
	n.show_hint = obs_data_get_bool(s, "show_hint");
	n.helper_cmd = obs_data_get_string(s, "helper_cmd");
	n.trt_dir = obs_data_get_string(s, "trt_dir");
	n.debug = debug_fail_from_string(obs_data_get_string(s, "debug_fail"));
	std::lock_guard lk(f->set_mu);
	bool weights_changed = n.weights_path != f->set.weights_path;
	f->set = n;
	f->set_gen++;
	if (weights_changed)
		log(LOG_INFO, "%s: weights %s", f->name.c_str(), n.weights_path.empty() ? "(none)" : n.weights_path.c_str());
	if (n.debug != DebugFail::None)
		log(LOG_INFO, "%s: debug_fail=%s", f->name.c_str(), obs_data_get_string(s, "debug_fail"));
}

WorkerRequest make_request(const Settings &s, int w, int h, double fps)
{
	WorkerRequest r;
	r.trt_dir = s.trt_dir;
	r.weights_path = s.weights_path;
	r.quality = s.quality;
	r.helper_cmd = s.helper_cmd;
	r.cache_dir = default_cache_dir();
	char *p = obs_module_file("perf-table.json");
	r.perf_table_path = p ? p : "";
	bfree(p);
	r.frame_w = w;
	r.frame_h = h;
	r.fps = fps;
	r.hint = s.hint();
	r.debug = s.debug;
	r.model_id = s.model_id;
	r.manifest_path = manifest_path();
	r.weights_dir = weights_dir();
	r.licence_ok = !config::licence_acknowledged(s.model_id).empty();
	r.download_clicks = (uint64_t)s.download_requested;
	r.prepare_clicks = (uint64_t)s.prepare_requested;
	return r;
}

// ------------------------------------------------------------------------------- render thread

void destroy_textures(Filter *f)
{
	if (f->copy_stream)
		cudaStreamSynchronize(f->copy_stream); // our own short copies only
	f->in_io.reset();
	f->out_io.reset();
	if (f->in_tex)
		gs_texture_destroy(std::exchange(f->in_tex, nullptr));
	if (f->out_tex)
		gs_texture_destroy(std::exchange(f->out_tex, nullptr));
	f->tex_w = f->tex_h = 0;
	f->have_bufs = false;
	f->have_matte = false;
}

bool ensure_textures(Filter *f, int w, int h)
{
	if (f->tex_w == w && f->tex_h == h && f->in_io && f->out_io)
		return true;
	destroy_textures(f);
	const uint64_t t0 = os_gettime_ns();
	f->in_tex = gs_texture_create(w, h, GS_RGBA_UNORM, 1, nullptr, 0);
	f->out_tex = gs_texture_create(w, h, GS_RGBA_UNORM, 1, nullptr, 0);
	if (!f->in_tex || !f->out_tex)
		return false;
	std::string err;
	f->in_io = gpu::InteropTexture::register_texture((ID3D11Texture2D *)gs_texture_get_obj(f->in_tex),
							 gpu::InteropTexture::Use::Read, err);
	if (f->in_io)
		f->out_io = gpu::InteropTexture::register_texture((ID3D11Texture2D *)gs_texture_get_obj(f->out_tex),
								  gpu::InteropTexture::Use::Write, err);
	if (!f->in_io || !f->out_io) {
		log(LOG_ERROR, "%s: CUDA interop registration failed: %s", f->name.c_str(), err.c_str());
		f->unsupported = "CUDA cannot share OBS's textures: " + err;
		destroy_textures(f);
		return false;
	}
	f->tex_w = w;
	f->tex_h = h;
	log(LOG_INFO, "%s: source %dx%d, interop textures registered in %.1f ms", f->name.c_str(), w, h,
	    (double)(os_gettime_ns() - t0) / 1e6);
	return true;
}

gs_texture_t *render_source(Filter *f, obs_source_t *target, obs_source_t *parent, uint32_t w, uint32_t h)
{
	if (!f->tr)
		f->tr = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
	gs_texrender_reset(f->tr);
	const uint32_t flags = obs_source_get_output_flags(target);
	const bool custom_draw = (flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
	const bool async = (flags & OBS_SOURCE_ASYNC) != 0;
	if (!gs_texrender_begin(f->tr, w, h))
		return nullptr;
	vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	if (target == parent && !custom_draw && !async)
		obs_source_default_render(target);
	else
		obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(f->tr);
	return gs_texrender_get_texture(f->tr);
}

void log_state_change(Filter *f, State before)
{
	if (f->sm.state() == before && f->sm.state() != State::Preparing)
		return;
	if (f->sm.state() == State::Fallback && before != State::Fallback)
		f->fallbacks++;
	char age[64] = "";
	if (f->sm.state() == State::Fallback && f->last_matte_ms > 0)
		snprintf(age, sizeof(age), ", last matte %.0f ms ago", now_ms() - f->last_matte_ms);
	log(f->sm.state() == State::Fallback || f->sm.state() == State::Failed || f->sm.state() == State::Unsupported
		    ? LOG_WARNING
		    : LOG_INFO,
	    "%s: state %s -> %s%s%s (t=%.1f ms%s)", f->name.c_str(), to_string(before), to_string(f->sm.state()),
	    f->sm.reason().empty() ? "" : ": ", f->sm.reason().c_str(), now_ms(), age);
}

// Apply the worker's status to the state machine.
void sync_worker_status(Filter *f)
{
	if (!f->worker)
		return;
	uint64_t seq = f->worker->status_seq();
	if (seq == f->worker_seq_seen)
		return;
	f->worker_seq_seen = seq;
	f->ws = f->worker->status();
	const auto &w = f->ws;
	State before = f->sm.state();
	bool changed = false;
	switch (w.phase) {
	case Phase::Unsupported:
		changed = f->sm.unsupported(w.reason);
		break;
	case Phase::NoWeights:
		changed = f->sm.no_weights();
		if (!w.reason.empty())
			changed = f->sm.download_failed(w.reason) || changed;
		break;
	case Phase::Downloading:
		changed = f->sm.downloading(w.progress);
		break;
	case Phase::Starting:
		changed = f->sm.preparing(0, w.reason.empty() ? "loading" : w.reason);
		break;
	case Phase::Preparing:
		changed = f->sm.preparing(w.progress, w.reason);
		break;
	case Phase::Failed:
		changed = f->sm.prepare_failed(w.reason);
		break;
	case Phase::Ready:
		if (f->sm.state() == State::NoWeights || f->sm.state() == State::Unsupported ||
		    f->sm.state() == State::Preparing || f->sm.state() == State::Failed)
			changed = f->sm.engine_ready();
		break;
	}
	if (w.error_seq != f->error_seq_seen) {
		f->error_seq_seen = w.error_seq;
		f->worker_errors++;
		changed = f->sm.worker_error(w.last_error) || changed;
		if (f->worker_errors < 20 || f->worker_errors % 100 == 0)
			log(LOG_ERROR, "%s: keyer error (#%llu): %s", f->name.c_str(), (unsigned long long)f->worker_errors,
			    w.last_error.c_str());
	}
	if (changed)
		log_state_change(f, before);
}

// The keying half of a frame: submit this frame, collect a finished matte.
void pipeline_step(Filter *f, gs_texture_t *src, int w, int h, double frame_ms)
{
	bool submitted = false;
	uint64_t frame = ++f->frame_no;
	if (f->worker && f->cuda_dev >= 0 && f->unsupported.empty()) {
		// tell the worker what we need (only when something changed)
		if (f->rs_gen != f->req_gen_sent || f->req_w != w || f->req_h != h) {
			f->worker->request(make_request(f->rs, w, h, 1000.0 / frame_ms));
			f->req_gen_sent = f->rs_gen;
			f->req_w = w;
			f->req_h = h;
		}
		// No CUDA call at all on the render thread until the worker has a warmed-up engine: engine
		// loading and CUDA's lazy module loading take driver locks that would stall this thread.
		const bool worker_ready = f->ws.phase == Phase::Ready;
		if (worker_ready) {
			const uint64_t ts0 = os_gettime_ns();
			cudaSetDevice(f->cuda_dev);
			if (!f->copy_stream) {
				if (cudaStreamCreateWithFlags(&f->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
					cudaGetLastError();
					f->unsupported = "cannot create a CUDA stream on the OBS GPU";
				}
				log(LOG_INFO, "%s: render-thread CUDA setup %.1f ms", f->name.c_str(),
				    (double)(os_gettime_ns() - ts0) / 1e6);
			}
		}
		if (worker_ready && f->unsupported.empty() && ensure_textures(f, w, h)) {
			if (!f->have_bufs || f->ws.buffers_gen != f->bufs_gen_seen) {
				f->have_bufs = f->worker->buffers(w, h, f->bufs);
				f->bufs_gen_seen = f->ws.buffers_gen;
			}
			Scheduler &sched = f->worker->scheduler();
			const bool keyable = f->have_bufs && f->ws.phase == Phase::Ready &&
					     (f->sm.state() == State::Ready || f->sm.state() == State::Keying ||
					      f->sm.state() == State::Behind || f->sm.state() == State::Fallback);
			if (keyable) {
				int slot = sched.begin_submit();
				gs_copy_texture(f->in_tex, src);
				cudaError_t e = f->in_io->copy_to_linear(f->bufs.in[slot], f->bufs.pitch, f->copy_stream,
									  f->bufs.in_ready[slot]);
				if (e == cudaSuccess) {
					sched.end_submit(slot, frame, (int64_t)os_gettime_ns());
					submitted = true;
				} else {
					cudaGetLastError();
					sched.abort_submit(slot);
					State before = f->sm.state();
					if (f->sm.worker_error(std::string("frame copy failed: ") + cudaGetErrorName(e)))
						log_state_change(f, before);
				}
			}
			if (f->have_bufs) {
				if (auto r = sched.poll_output()) {
					cudaError_t e = f->out_io->copy_from_linear(f->bufs.out[r->out_slot], f->bufs.pitch,
										    f->copy_stream,
										    f->bufs.out_consumed[r->out_slot]);
					if (e == cudaSuccess) {
						f->have_matte = true;
						f->mattes++;
						f->last_matte_ms = now_ms();
						f->stats.add_worker_ms(r->worker_ms);
						f->stats.add_pairing_delay((double)(frame - r->frame_id));
						State before = f->sm.state();
						if (f->sm.matte_arrived(now_ms()))
							log_state_change(f, before);
					} else {
						cudaGetLastError();
					}
				}
			}
			uint64_t sk = sched.skipped();
			f->stats.add_frame(submitted, sk - f->skipped_seen);
			f->skipped_seen = sk;
		}
	}
	if (!f->unsupported.empty() && f->sm.state() != State::Unsupported) {
		State before = f->sm.state();
		f->sm.unsupported(f->unsupported);
		log_state_change(f, before);
	}
	Readout ro = f->stats.readout(frame_ms);
	State before = f->sm.state();
	if (f->sm.tick(now_ms(), (double)f->rs.fallback_timeout_ms, ro.behind))
		log_state_change(f, before);
}

void draw(Filter *f, gs_effect_t *fx, const char *tech, gs_texture_t *image, gs_texture_t *keyed, int w, int h)
{
	const bool prev = gs_framebuffer_srgb_enabled();
	gs_enable_framebuffer_srgb(false);
	if (gs_eparam_t *p = gs_effect_get_param_by_name(fx, "image"); p && image)
		gs_effect_set_texture(p, image);
	if (gs_eparam_t *p = gs_effect_get_param_by_name(fx, "keyed"); p && keyed)
		gs_effect_set_texture(p, keyed);
	if (fx == f->fx_colorkey) {
		gpu::HintParams hp = f->rs.hint();
		vec2 texel;
		vec2_set(&texel, 1.0f / (float)w, 1.0f / (float)h);
		gs_effect_set_vec2(gs_effect_get_param_by_name(fx, "texel"), &texel);
		gs_effect_set_float(gs_effect_get_param_by_name(fx, "kd"), hp.kd);
		gs_effect_set_float(gs_effect_get_param_by_name(fx, "lo"), hp.lo);
		gs_effect_set_float(gs_effect_get_param_by_name(fx, "hi"), hp.hi);
	}
	while (gs_effect_loop(fx, tech))
		gs_draw_sprite(image ? image : keyed, 0, (uint32_t)w, (uint32_t)h);
	gs_enable_framebuffer_srgb(prev);
}

void update_snapshot(Filter *f, double frame_ms, int w, int h, Output out)
{
	uint64_t now = os_gettime_ns();
	bool summary = now - f->last_summary_ns > 60ull * 1000000000ull;
	if (now - f->last_snapshot_ns < 200ull * 1000000ull && !summary)
		return;
	f->last_snapshot_ns = now;
	Snapshot s;
	s.state = f->sm.state();
	s.reason = f->sm.reason();
	s.progress = f->sm.progress();
	s.readout = f->stats.readout(frame_ms);
	s.worker = f->ws;
	s.render_us_median = percentile(f->render_us, 50);
	s.render_us_p95 = percentile(f->render_us, 95);
	s.render_us_p99 = percentile(f->render_us, 99);
	s.render_us_max = f->render_us_max;
	s.render_frames = f->frame_no;
	s.mattes = f->mattes;
	s.fallbacks = f->fallbacks;
	s.worker_errors = f->worker_errors;
	if (f->worker) {
		s.skipped_total = f->worker->scheduler().skipped();
		s.submitted_total = f->worker->scheduler().submitted();
	}
	s.frame_ms = frame_ms;
	s.last_matte_age_ms = f->last_matte_ms > 0 ? now_ms() - f->last_matte_ms : -1;
	s.src_w = w;
	s.src_h = h;
	static const char *names[] = {"composite", "colorkey", "last_matte", "passthrough"};
	s.output = names[(int)out];
	s.status_text = state_text(*f, s);
	if (summary) {
		f->last_summary_ns = now;
		log(LOG_INFO,
		    "%s: 60 s summary: state %s, %s, keying %.1f ms median / %.1f ms p95, %llu/%llu frames skipped, "
		    "render-thread cost %.0f us median / %.0f us p99 / %.0f us max",
		    f->name.c_str(), to_string(s.state), s.worker.quality.empty() ? "no engine" : s.worker.quality.c_str(),
		    s.readout.worker_ms_median, s.readout.worker_ms_p95, (unsigned long long)s.skipped_total,
		    (unsigned long long)s.submitted_total, s.render_us_median, s.render_us_p99, s.render_us_max);
	}
	std::lock_guard lk(f->snap_mu);
	f->snap = std::move(s);
}

void filter_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<Filter *>(data);
	const uint64_t t0 = os_gettime_ns();
	obs_source_t *target = obs_filter_get_target(f->ctx);
	obs_source_t *parent = obs_filter_get_parent(f->ctx);
	const uint32_t w = target ? obs_source_get_base_width(target) : 0;
	const uint32_t h = target ? obs_source_get_base_height(target) : 0;
	if (!target || !parent || !w || !h || !f->fx_composite || !f->fx_colorkey) {
		obs_source_skip_video_filter(f->ctx);
		return;
	}
	{
		std::lock_guard lk(f->set_mu);
		if (f->rs_gen != f->set_gen) {
			f->rs = f->set;
			f->rs_gen = f->set_gen;
		}
	}
	const double frame_ms = obs_frame_ms();
	const bool first = f->rendered_tick != f->tick; // studio mode: key once per frame
	gs_texture_t *src = nullptr;
	if (first) {
		f->rendered_tick = f->tick;
		src = render_source(f, target, parent, w, h);
		if (!src) {
			obs_source_skip_video_filter(f->ctx);
			return;
		}
		sync_worker_status(f);
		pipeline_step(f, src, (int)w, (int)h, frame_ms);
	} else {
		src = gs_texrender_get_texture(f->tr);
	}

	Output out = f->sm.output(f->rs.fallback, f->have_matte && f->out_tex);
	if ((out == Output::Composite || out == Output::LastMatte) && !(f->out_tex && f->have_matte))
		out = Output::Passthrough;
	if (f->rs.show_hint)
		draw(f, f->fx_colorkey, "DrawHint", src, nullptr, (int)w, (int)h);
	else if (out == Output::Composite)
		draw(f, f->fx_composite, "Draw", nullptr, f->out_tex, (int)w, (int)h);
	else if (out == Output::LastMatte)
		draw(f, f->fx_composite, "LastMatte", src, f->out_tex, (int)w, (int)h);
	else if (out == Output::ColorKey)
		draw(f, f->fx_colorkey, "Draw", src, nullptr, (int)w, (int)h);
	else
		draw(f, obs_get_base_effect(OBS_EFFECT_DEFAULT), "Draw", src, nullptr, (int)w, (int)h);
	if (out != f->last_output && first) {
		log(LOG_INFO, "%s: output %d -> %d", f->name.c_str(), (int)f->last_output, (int)out);
		f->last_output = out;
	}

	if (first) {
		double us = (double)(os_gettime_ns() - t0) / 1000.0;
		f->render_us.push_back(us);
		if (f->render_us.size() > 600)
			f->render_us.pop_front();
		f->render_us_max = std::max(f->render_us_max, us);
		update_snapshot(f, frame_ms, (int)w, (int)h, out);
	}
}

// ------------------------------------------------------------------------------- obs_source_info

const char *filter_name(void *)
{
	return obs_module_text("CorridorKey");
}

void filter_defaults(obs_data_t *s)
{
	obs_data_set_default_string(s, "model_id", "CorridorKey_v1.0");
	obs_data_set_default_bool(s, "licence_ack", false);
	obs_data_set_default_int(s, "download_requested", 0);
	obs_data_set_default_int(s, "prepare_requested", 0);
	obs_data_set_default_string(s, "weights_path", default_weights_path("CorridorKey_v1.0").c_str());
	obs_data_set_default_string(s, "quality", "auto");
	obs_data_set_default_string(s, "hint_mode", "fixed");
	float green[3] = {0.0f, 1.0f, 0.0f};
	obs_data_set_default_int(s, "hint_color", color_to_obs(green));
	obs_data_set_default_double(s, "hint_lo", 0.30);
	obs_data_set_default_double(s, "hint_hi", 0.70);
	obs_data_set_default_string(s, "fallback_mode", "colorkey");
	obs_data_set_default_int(s, "fallback_timeout_ms", 250);
	obs_data_set_default_bool(s, "show_hint", false);
	obs_data_set_default_string(s, "helper_cmd", default_helper_cmd().c_str());
	obs_data_set_default_string(s, "trt_dir", "C:\\TensorRT-10.16.1.11\\bin");
	obs_data_set_default_string(s, "debug_fail", "none");
}

void filter_update(void *data, obs_data_t *s)
{
	read_settings(static_cast<Filter *>(data), s);
}

void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	auto *f = new Filter();
	f->ctx = source;
	f->name = std::string("'") + obs_source_get_name(source) + "'";
	read_settings(f, settings);

	obs_enter_graphics();
	char *p = obs_module_file("effects/composite.effect");
	char *errors = nullptr;
	f->fx_composite = gs_effect_create_from_file(p, &errors);
	if (!f->fx_composite)
		log(LOG_ERROR, "%s: composite.effect failed: %s", f->name.c_str(), errors ? errors : "?");
	bfree(errors);
	bfree(p);
	errors = nullptr;
	p = obs_module_file("effects/colorkey.effect");
	f->fx_colorkey = gs_effect_create_from_file(p, &errors);
	if (!f->fx_colorkey)
		log(LOG_ERROR, "%s: colorkey.effect failed: %s", f->name.c_str(), errors ? errors : "?");
	bfree(errors);
	bfree(p);
	std::string err;
	if (gs_get_device_type() != GS_DEVICE_DIRECT3D_11)
		err = "OBS is not using Direct3D 11";
	else
		f->cuda_dev = gpu::cuda_device_for_d3d11((ID3D11Device *)gs_get_device_obj(), err);
	obs_leave_graphics();

	if (f->cuda_dev < 0) {
		f->unsupported = err;
		f->sm.unsupported(err);
		log(LOG_WARNING, "%s: unsupported: %s (the source passes through / colour key)", f->name.c_str(),
		    err.c_str());
	} else {
		f->worker = std::make_unique<Worker>(f->cuda_dev, f->name);
		Settings s;
		{
			std::lock_guard lk(f->set_mu);
			s = f->set;
		}
		f->worker->request(make_request(s, 0, 0, 1000.0 / obs_frame_ms()));
		log(LOG_INFO, "%s: created on CUDA device %d", f->name.c_str(), f->cuda_dev);
	}
	{
		std::lock_guard lk(g_reg_mu);
		g_filters[source] = f;
	}
	return f;
}

void filter_destroy(void *data)
{
	auto *f = static_cast<Filter *>(data);
	{
		std::lock_guard lk(g_reg_mu);
		g_filters.erase(f->ctx);
	}
	if (f->worker)
		f->worker->stop();
	obs_enter_graphics();
	destroy_textures(f);
	if (f->copy_stream)
		cudaStreamDestroy(f->copy_stream);
	if (f->tr)
		gs_texrender_destroy(f->tr);
	if (f->fx_composite)
		gs_effect_destroy(f->fx_composite);
	if (f->fx_colorkey)
		gs_effect_destroy(f->fx_colorkey);
	obs_leave_graphics();
	if (f->worker) {
		// Join off whatever thread OBS destroys us on: the worker may be inside an inference.
		std::lock_guard lk(g_reap_mu);
		g_reapers.emplace_back([w = std::move(f->worker)]() mutable { w.reset(); });
	}
	log(LOG_INFO, "%s: destroyed", f->name.c_str());
	delete f;
}

void filter_tick(void *data, float)
{
	static_cast<Filter *>(data)->tick++;
}

bool refresh_clicked(obs_properties_t *, obs_property_t *, void *)
{
	return true; // rebuild the properties so the status line refreshes
}

// Buttons bump a counter setting (also settable over obs-websocket, which can't press filter
// buttons); the worker acts on each increment.
void bump(Filter *f, const char *key, const std::function<void(obs_data_t *)> &extra = {})
{
	obs_data_t *s = obs_source_get_settings(f->ctx);
	obs_data_set_int(s, key, obs_data_get_int(s, key) + 1);
	if (extra)
		extra(s);
	obs_source_update(f->ctx, s);
	obs_data_release(s);
}

bool download_clicked(obs_properties_t *, obs_property_t *, void *data)
{
	auto *f = static_cast<Filter *>(data);
	bump(f, "download_requested", [](obs_data_t *s) {
		std::string id = obs_data_get_string(s, "model_id");
		obs_data_set_string(s, "weights_path", default_weights_path(id.empty() ? "CorridorKey_v1.0" : id).c_str());
	});
	return true;
}

bool prepare_clicked(obs_properties_t *, obs_property_t *, void *data)
{
	bump(static_cast<Filter *>(data), "prepare_requested");
	return true;
}

obs_properties_t *filter_properties(void *data)
{
	auto *f = static_cast<Filter *>(data);
	Snapshot s;
	{
		std::lock_guard lk(f->snap_mu);
		s = f->snap;
	}
	obs_properties_t *props = obs_properties_create();

	std::string status = s.status_text.empty() ? (f->unsupported.empty() ? "Starting" : "Unsupported: " + f->unsupported)
						   : s.status_text;
	obs_properties_add_text(props, "status_info", status.c_str(), OBS_TEXT_INFO);
	obs_properties_add_button(props, "refresh", "Refresh status", refresh_clicked);

	// Model (contracts/filter-ui.md, revised 2026-10-19)
	obs_properties_t *mp = obs_properties_create();
	std::string model_id;
	{
		std::lock_guard lk(f->set_mu);
		model_id = f->set.model_id;
	}
	std::string label = s.worker.model_label.empty() ? model_id : s.worker.model_label;
	obs_properties_add_text(mp, "model_info", ("Model: " + label).c_str(), OBS_TEXT_INFO);
	std::string acked = config::licence_acknowledged(model_id);
	if (acked.empty()) {
		obs_properties_add_bool(mp, "licence_ack",
					"I accept the CorridorKey licence: CC BY-NC-SA 4.0 plus Corridor Digital's terms "
					"(non-commercial, attribution, share-alike)");
	} else {
		obs_properties_add_text(mp, "licence_info", ("Licence accepted on " + acked + " (CC BY-NC-SA 4.0 + Corridor Digital terms)").c_str(),
					OBS_TEXT_INFO);
	}
	obs_property_t *lic = obs_properties_add_button(mp, "licence_link", "Read the CorridorKey licence", nullptr);
	obs_property_button_set_type(lic, OBS_BUTTON_URL);
	obs_property_button_set_url(lic, (char *)kLicenceUrl);
	obs_properties_add_button(mp, "download", "Download weights (about 400 MB, from Corridor Digital's Hugging Face)",
				  download_clicked);
	obs_properties_add_path(mp, "weights_path", "Use a local file\xE2\x80\xA6", OBS_PATH_FILE,
				"CorridorKey weights (*.safetensors);;All files (*.*)", nullptr);
	obs_property_t *get = obs_properties_add_button(mp, "get_weights", "Model page (official source)", nullptr);
	obs_property_button_set_type(get, OBS_BUTTON_URL);
	obs_property_button_set_url(get, (char *)kWeightsUrl);
	for (const auto &n : s.worker.newer_models)
		obs_properties_add_text(mp, ("newer_" + n).c_str(),
					("Newer model available: " + n +
					 ". Not switched: update this plugin's model manifest (git pull) to pin its hash, "
					 "then choose it and press Download weights.").c_str(),
					OBS_TEXT_INFO);
	obs_property_t *mid = obs_properties_add_text(mp, "model_id", "Model id", OBS_TEXT_DEFAULT);
	obs_property_set_visible(mid, false);
	obs_properties_add_group(props, "model_group", "Model", OBS_GROUP_NORMAL, mp);

	obs_property_t *q = obs_properties_add_list(props, "quality", "Quality", OBS_COMBO_TYPE_LIST,
						    OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(q, "Auto", "auto");
	for (const auto &qq : all_qualities()) {
		std::string ql = quality_label(qq);
		if (!s.worker.prepared.count(qq.id))
			ql += " (needs preparing)";
		obs_property_list_add_string(q, ql.c_str(), qq.id.c_str());
	}
	obs_properties_add_button(props, "prepare", "Prepare selected", prepare_clicked);

	obs_properties_t *hint = obs_properties_create();
	obs_property_t *hm = obs_properties_add_list(hint, "hint_mode", "Screen colour", OBS_COMBO_TYPE_LIST,
						     OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(hm, "Fixed green (default key)", "fixed");
	obs_property_list_add_string(hm, "Auto-calibrated", "calibrated");
	obs_property_list_add_string(hm, "Manual", "manual");
	obs_properties_add_color(hint, "hint_color", "Screen colour");
	obs_properties_add_float_slider(hint, "hint_lo", "Low threshold", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(hint, "hint_hi", "High threshold", 0.0, 1.0, 0.01);
	obs_properties_add_bool(hint, "show_hint", "Show hint (rough mask)");
	obs_properties_add_group(props, "hint_group", "Screen colour (hint)", OBS_GROUP_NORMAL, hint);

	obs_properties_t *fb = obs_properties_create();
	obs_property_t *fm = obs_properties_add_list(fb, "fallback_mode", "Fallback", OBS_COMBO_TYPE_LIST,
						     OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(fm, "Colour key (rough matte)", "colorkey");
	obs_property_list_add_string(fm, "Last good matte", "last_matte");
	obs_property_list_add_string(fm, "Pass-through", "passthrough");
	obs_properties_add_int_slider(fb, "fallback_timeout_ms", "Fallback after (ms without a new matte)", 50, 2000, 10);
	obs_properties_add_group(props, "fallback_group", "Fallback", OBS_GROUP_NORMAL, fb);

	obs_properties_t *adv = obs_properties_create();
	obs_properties_add_text(adv, "helper_cmd", "Helper command", OBS_TEXT_DEFAULT);
	obs_properties_add_path(adv, "trt_dir", "TensorRT folder (nvinfer_10.dll)", OBS_PATH_DIRECTORY, nullptr, nullptr);
	obs_property_t *dbg = obs_properties_add_list(adv, "debug_fail", "debug_fail", OBS_COMBO_TYPE_LIST,
						      OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(dbg, "none", "none");
	obs_property_list_add_string(dbg, "worker_error", "worker_error");
	obs_property_list_add_string(dbg, "stall", "stall");
	obs_property_set_visible(dbg, false); // SC-004 test hook only
	obs_properties_add_group(props, "advanced_group", "Advanced", OBS_GROUP_NORMAL, adv);

	obs_properties_add_text(props, "attribution", kAttribution, OBS_TEXT_INFO);
	return props;
}

// ------------------------------------------------------------------------------- websocket vendor

void ws_get_status(obs_data_t *req, obs_data_t *resp, void *)
{
	const char *src_name = obs_data_get_string(req, "sourceName");
	const char *flt_name = obs_data_get_string(req, "filterName");
	obs_source_t *src = obs_get_source_by_name(src_name);
	obs_source_t *flt = src ? obs_source_get_filter_by_name(src, flt_name) : nullptr;
	Snapshot s;
	bool found = false;
	{
		std::lock_guard lk(g_reg_mu);
		auto it = flt ? g_filters.find(flt) : g_filters.end();
		if (it != g_filters.end()) {
			std::lock_guard lk2(it->second->snap_mu);
			s = it->second->snap;
			found = true;
		}
	}
	obs_source_release(flt);
	obs_source_release(src);
	obs_data_set_bool(resp, "found", found);
	if (!found)
		return;
	obs_data_set_string(resp, "state", to_string(s.state));
	obs_data_set_string(resp, "reason", s.reason.c_str());
	obs_data_set_int(resp, "progress", s.progress);
	obs_data_set_string(resp, "status", s.status_text.c_str());
	obs_data_set_string(resp, "output", s.output.c_str());
	obs_data_set_string(resp, "workerPhase", to_string(s.worker.phase));
	obs_data_set_string(resp, "quality", s.worker.quality.c_str());
	obs_data_set_int(resp, "engineWidth", s.worker.engine_w);
	obs_data_set_int(resp, "engineHeight", s.worker.engine_h);
	obs_data_set_string(resp, "modeId", s.worker.mode_id.c_str());
	obs_data_set_string(resp, "autoQuality", s.worker.auto_choice.quality.c_str());
	obs_data_set_string(resp, "autoBest", s.worker.auto_choice.best_overall.c_str());
	obs_data_set_bool(resp, "autoUnknownGpu", s.worker.auto_choice.unknown_gpu);
	obs_data_set_string(resp, "gpuName", s.worker.gpu_name.c_str());
	std::string prepared;
	for (const auto &p : s.worker.prepared)
		prepared += (prepared.empty() ? "" : ",") + p;
	obs_data_set_string(resp, "prepared", prepared.c_str());
	obs_data_set_double(resp, "workerMsMedian", s.readout.worker_ms_median);
	obs_data_set_double(resp, "workerMsP95", s.readout.worker_ms_p95);
	obs_data_set_double(resp, "skippedPct", s.readout.skipped_pct);
	obs_data_set_double(resp, "pairingDelayFrames", s.readout.pairing_delay_frames);
	obs_data_set_double(resp, "addedLatencyMs", s.readout.added_latency_ms);
	obs_data_set_double(resp, "addedLatencyFrames", s.readout.added_latency_frames);
	obs_data_set_bool(resp, "behind", s.readout.behind);
	obs_data_set_double(resp, "renderUsMedian", s.render_us_median);
	obs_data_set_double(resp, "renderUsP95", s.render_us_p95);
	obs_data_set_double(resp, "renderUsP99", s.render_us_p99);
	obs_data_set_double(resp, "renderUsMax", s.render_us_max);
	obs_data_set_int(resp, "renderFrames", (long long)s.render_frames);
	obs_data_set_int(resp, "mattes", (long long)s.mattes);
	obs_data_set_int(resp, "skippedTotal", (long long)s.skipped_total);
	obs_data_set_int(resp, "submittedTotal", (long long)s.submitted_total);
	obs_data_set_int(resp, "fallbacks", (long long)s.fallbacks);
	obs_data_set_int(resp, "workerErrors", (long long)s.worker_errors);
	obs_data_set_double(resp, "frameMs", s.frame_ms);
	obs_data_set_double(resp, "lastMatteAgeMs", s.last_matte_age_ms);
	obs_data_set_string(resp, "modelLabel", s.worker.model_label.c_str());
	obs_data_set_bool(resp, "modelVerified", s.worker.model_verified);
	obs_data_set_int(resp, "downloadBytes", s.worker.dl_bytes);
	obs_data_set_int(resp, "downloadTotal", s.worker.dl_total);
	std::string newer;
	for (const auto &n : s.worker.newer_models)
		newer += (newer.empty() ? "" : ",") + n;
	obs_data_set_string(resp, "newerModels", newer.c_str());
	obs_data_set_bool(resp, "preparingInBackground", s.worker.preparing_in_background);
	obs_data_set_string(resp, "preparingQuality", s.worker.preparing_quality.c_str());
	obs_data_set_int(resp, "workerProgress", s.worker.progress);
	obs_data_set_int(resp, "sourceWidth", s.src_w);
	obs_data_set_int(resp, "sourceHeight", s.src_h);
}

} // namespace

void register_filter()
{
	obs_source_info info = {};
	info.id = "corridorkey_filter";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = filter_name;
	info.create = filter_create;
	info.destroy = filter_destroy;
	info.update = filter_update;
	info.get_defaults = filter_defaults;
	info.get_properties = filter_properties;
	info.video_tick = filter_tick;
	info.video_render = filter_render;
	obs_register_source(&info);
}

void register_websocket_vendor()
{
	obs_websocket_vendor v = obs_websocket_register_vendor("corridorkey");
	if (!v) {
		log(LOG_INFO, "obs-websocket not available; vendor requests disabled");
		return;
	}
	obs_websocket_vendor_register_request(v, "GetFilterStatus", ws_get_status, nullptr);
}

void shutdown_filters()
{
	std::vector<std::thread> t;
	{
		std::lock_guard lk(g_reap_mu);
		t.swap(g_reapers);
	}
	for (auto &th : t)
		if (th.joinable())
			th.join();
}

} // namespace ck
