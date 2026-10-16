"""Speed + VRAM benchmark loop (US1/US2/US4; research D7)."""
from __future__ import annotations

import contextlib
import gc
import time
import traceback

import torch

from . import results as R
from . import vram
from .config import Configuration
from .frames import benchmark_frame
from .pipeline import Pipeline
from .timing import run_timed, spread, summarise_run


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def bench_key(e: dict) -> tuple:
    return (e["config"]["config_id"], e["camera_res"])


def build_keyer(cfg: Configuration, doc: dict):
    if cfg.runtime == "torch":
        from .runtime_torch import TorchKeyer

        return TorchKeyer(cfg.size, cfg.refiner)
    from .export import engine_path
    from .runtime_trt import TrtKeyer

    return TrtKeyer(engine_path(cfg.internal_w, cfg.internal_h, cfg.refiner), cfg.size)


def trt_precheck(cfg: Configuration, doc: dict) -> tuple[str, str] | None:
    """For TRT configs, map a failed/missing export to a benchmark status."""
    if cfg.runtime != "trt":
        return None
    from .export import engine_path

    for ex in doc.get("exports", []):
        if (ex["internal_w"], ex["internal_h"]) == cfg.size and ex.get("refiner", True) == cfg.refiner:
            if ex["onnx_status"] != "ok":
                return "export_failed", ex.get("onnx_reason", "ONNX export failed")
            if ex["engine_status"] != "ok":
                return "build_failed", ex.get("engine_reason", "engine build failed")
    if not engine_path(cfg.internal_w, cfg.internal_h, cfg.refiner).exists():
        return "not_attempted", "no TensorRT engine built for this size (run `ckbench export`)"
    return None


def run_config(doc: dict, cfg: Configuration, cameras: list[str], runs: int, frames: int, warmup: int,
               load_ctx=None, log=print) -> list[dict]:
    """Benchmark one configuration at each camera resolution. Never raises for OOM/build errors."""
    frame, frame_src = benchmark_frame()
    entries: list[dict] = []

    def record(e: dict) -> None:
        R.upsert(doc["benchmarks"], e, bench_key)
        R.save(doc)
        entries.append(e)

    pre = trt_precheck(cfg, doc)
    if pre:
        for cam in cameras:
            record({"config": cfg.to_dict(), "camera_res": cam, "status": pre[0], "status_reason": pre[1]})
        log(f"  {cfg.config_id}: {pre[0]} ({pre[1][:120]})")
        return entries

    keyer = None
    try:
        _free()
        keyer = build_keyer(cfg, doc)
        if getattr(keyer, "posembed_interpolated", None) is not None:
            cfg = Configuration.from_dict({**cfg.to_dict(), "posembed_interpolated": keyer.posembed_interpolated})
    except Exception as e:  # build failure / OOM at build: recorded, not raised
        status = "oom" if _is_oom(e) else "not_attempted"
        reason = f"keyer build failed: {type(e).__name__}: {e}"[:2000]
        del keyer
        _free()
        for cam in cameras:
            record({"config": cfg.to_dict(), "camera_res": cam, "status": status, "status_reason": reason})
        log(f"  {cfg.config_id}: {status} at build ({type(e).__name__})")
        return entries

    for cam in cameras:
        _free()
        torch.cuda.reset_peak_memory_stats()
        run_entries = []
        try:
            ctx = load_ctx() if load_ctx else contextlib.nullcontext()
            with ctx as load_info:
                snap0 = vram.snapshot()
                with vram.VramSampler() as sampler:
                    pipe = Pipeline(keyer, cam, frame)
                    for i in range(runs):
                        t0 = time.perf_counter()
                        raw = run_timed(pipe, warmup=warmup, frames=frames)
                        run_entries.append(summarise_run(raw))
                        log(f"  {cfg.config_id} @{cam} run {i + 1}/{runs}: model median "
                            f"{run_entries[-1]['model_ms']['median']:.2f} ms, e2e median "
                            f"{run_entries[-1]['e2e_ms']['median']:.2f} ms ({time.perf_counter() - t0:.0f}s)",
                            flush=True)
                    del pipe
                snap1 = vram.snapshot()
            mpct, _ = spread([r["model_ms"]["median"] for r in run_entries])
            epct, _ = spread([r["e2e_ms"]["median"] for r in run_entries])
            pct = max(mpct, epct)
            peak, method = sampler.peak_mib()
            e = {
                "config": cfg.to_dict(), "camera_res": cam, "status": "ok", "runs": run_entries,
                "median_spread_pct": pct, "spread_flag": bool(pct > 10.0),
                "model_spread_pct": mpct, "e2e_spread_pct": epct,
                "peak_vram_mib": round(peak, 1), "vram_method": method,
                "clock_start_mhz": snap0["sm_clock_mhz"], "clock_end_mhz": snap1["sm_clock_mhz"],
                "temp_end_c": snap1["temp_c"], "frames_per_run": frames, "warmup": warmup,
                "input_frame": frame_src,
                "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if cfg.runtime == "torch":
                e["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if load_info:
                e["load_info"] = load_info
            record(e)
        except Exception as ex:
            status = "oom" if _is_oom(ex) else "not_attempted"
            reason = f"{type(ex).__name__}: {ex}"[:2000]
            if status != "oom":
                reason += "\n" + traceback.format_exc()[-1500:]
            _free()
            record({"config": cfg.to_dict(), "camera_res": cam, "status": status, "status_reason": reason,
                    "runs": run_entries})
            log(f"  {cfg.config_id} @{cam}: {status} ({type(ex).__name__}: {str(ex)[:200]})")
    if hasattr(keyer, "close"):
        keyer.close()
    del keyer
    _free()
    return entries
