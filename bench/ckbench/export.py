"""ONNX export + TensorRT fp16 engine build per size, and drift vs PyTorch (US2; research D3).

Each export and each engine build runs in its own subprocess (``python -m ckbench.export ...``)
so a crash or out-of-memory in one size can't take the benchmark down, and so the peak RAM of
each step can be measured on its own (psutil peak working set of the worker process).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .results import OUTPUTS_DIR

ONNX_DIR = OUTPUTS_DIR / "onnx"
ENGINE_DIR = OUTPUTS_DIR / "engines"
OPSET = 18
WORKSPACE_GIB = 8
DRIFT_FRAMES = 20


def onnx_path(w: int, h: int, refiner: bool) -> Path:
    return ONNX_DIR / f"ck_{w}x{h}_ref{int(refiner)}.onnx"


def engine_path(w: int, h: int, refiner: bool) -> Path:
    return ENGINE_DIR / f"ck_{w}x{h}_ref{int(refiner)}.engine"


def _peak_ram_gib() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None) or getattr(mi, "peak_pagefile", None) or mi.rss
    return round(peak / 2**30, 2)


# ----------------------------------------------------------------------------- workers

def _worker_onnx(w: int, h: int, refiner: bool) -> dict:
    """Build the fp32 model on CPU and export it with a static [1,4,h,w] input."""
    import torch

    from .runtime_torch import _vendor_import

    CorridorKeyEngine, _ = _vendor_import()
    from .runtime_torch import TorchKeyer

    t0 = time.perf_counter()
    keyer = TorchKeyer((w, h), refiner, device="cpu")
    model = keyer.model.float().eval()  # no refiner_scale hook is ever registered: plain forward

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return out["alpha"], out["fg"]

    x = torch.zeros(1, 4, h, w)
    p = onnx_path(w, h, refiner)
    p.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(Wrapper(model).eval(), (x,), str(p), opset_version=OPSET, dynamo=False,
                          input_names=["input"], output_names=["alpha", "fg"], do_constant_folding=True)
    return {"onnx_status": "ok", "onnx_s": round(time.perf_counter() - t0, 1),
            "onnx_peak_ram_gib": _peak_ram_gib(), "onnx_path": str(p),
            "onnx_size_mib": round(sum(f.stat().st_size for f in p.parent.glob(p.stem + "*")) / 2**20, 1)}


def _worker_engine(w: int, h: int, refiner: bool) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    src = onnx_path(w, h, refiner)
    if not parser.parse_from_file(str(src)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        return {"engine_status": "failed", "engine_reason": f"ONNX parse failed: {errs}"[:2000]}
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GIB << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    t0 = time.perf_counter()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        return {"engine_status": "failed", "engine_reason": "build_serialized_network returned None (see log)",
                "engine_build_s": round(time.perf_counter() - t0, 1)}
    dst = engine_path(w, h, refiner)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(ser)
    return {"engine_status": "ok", "engine_build_s": round(time.perf_counter() - t0, 1),
            "engine_size_mib": round(dst.stat().st_size / 2**20, 1), "engine_peak_ram_gib": _peak_ram_gib()}


def _run_worker(step: str, w: int, h: int, refiner: bool, log) -> dict:
    out_json = OUTPUTS_DIR / "logs" / f"{step}_{w}x{h}_ref{int(refiner)}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.unlink(missing_ok=True)
    log_file = out_json.with_suffix(".log")
    cmd = [sys.executable, "-m", "ckbench.export", step, str(w), str(h), str(int(refiner)), str(out_json)]
    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if out_json.exists():
        return json.loads(out_json.read_text())
    tail = log_file.read_text(encoding="utf-8", errors="replace")[-1500:]
    key = "onnx" if step == "onnx" else "engine"
    log(f"    {step} worker exited {r.returncode} after {time.perf_counter() - t0:.0f}s")
    return {f"{key}_status": "failed", f"{key}_reason": f"worker exit code {r.returncode}: {tail}"}


def _worker_main(argv: list[str]) -> None:
    step, w, h, ref, out = argv[0], int(argv[1]), int(argv[2]), bool(int(argv[3])), Path(argv[4])
    try:
        res = _worker_onnx(w, h, ref) if step == "onnx" else _worker_engine(w, h, ref)
    except Exception as e:
        import traceback

        key = "onnx" if step == "onnx" else "engine"
        res = {f"{key}_status": "failed", f"{key}_reason": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"[:3000]}
        if step == "onnx":
            res["onnx_peak_ram_gib"] = _peak_ram_gib()
    out.write_text(json.dumps(res))


# ----------------------------------------------------------------------------- drift

def drift_frames(n: int = DRIFT_FRAMES):
    """n fixed host RGB uint8 frames + a description of where they came from."""
    from .frames import CLIPS_OUT, PREFERRED, read_rgb, synthetic_frame
    from .results import REPO_ROOT

    for cid in PREFERRED:
        files = sorted((CLIPS_OUT / cid).glob("*.png"))[:n]
        if len(files) == n:
            return [read_rgb(f) for f in files], f"{cid} frames 0-{n - 1} (normalised 1080p)"
    for sub in ("Godiva Close", "Locked Backdrop"):
        files = sorted((REPO_ROOT / "footage" / "hcw" / sub).glob("*.png"))[:n]
        if len(files) == n:
            return [read_rgb(f) for f in files], f"HCW {sub} frames 0-{n - 1} (raw)"
    return [synthetic_frame(seed=i) for i in range(n)], f"synthetic frames seeds 0-{n - 1}"


def measure_drift(w: int, h: int, refiner: bool) -> dict:
    """Max/mean |TRT - torch fp16| of raw network alpha/fg at internal size over fixed frames."""
    import gc

    import torch

    from .pipeline import _down
    from .hint import chroma_hint
    from .runtime_torch import TorchKeyer
    from .runtime_trt import TrtKeyer

    frames, src = drift_frames()

    def inputs(keyer):
        for f in frames:
            rgb = torch.from_numpy(f).cuda().permute(2, 0, 1).to(torch.float16) / 255
            hint = chroma_hint(rgb)
            yield keyer.prepare(_down(rgb[None], h, w), _down(hint[None, None], h, w))

    ref = []
    tk = TorchKeyer((w, h), refiner)
    with torch.inference_mode():
        for inp in inputs(tk):
            a, f = tk.run(inp)
            ref.append((a.float().cpu(), f.float().cpu()))
    del tk
    gc.collect()
    torch.cuda.empty_cache()

    stats = {"alpha_mae": 0.0, "alpha_max": 0.0, "fg_mae": 0.0, "fg_max": 0.0}
    rk = TrtKeyer(engine_path(w, h, refiner), (w, h))
    with torch.inference_mode():
        for (ra, rf), inp in zip(ref, inputs(rk)):
            a, f = rk.run(inp)
            da = (a.float().cpu() - ra).abs()
            df = (f.float().cpu() - rf).abs()
            stats["alpha_mae"] += float(da.mean()) / len(ref)
            stats["fg_mae"] += float(df.mean()) / len(ref)
            stats["alpha_max"] = max(stats["alpha_max"], float(da.max()))
            stats["fg_max"] = max(stats["fg_max"], float(df.max()))
    rk.close()
    del rk
    gc.collect()
    torch.cuda.empty_cache()
    return {"drift": stats, "drift_frames": src}


# ----------------------------------------------------------------------------- driver

def export_one(w: int, h: int, refiner: bool, log=print) -> dict:
    entry: dict = {"internal_w": w, "internal_h": h, "refiner": refiner, "opset": OPSET,
                   "workspace_gib": WORKSPACE_GIB}
    log(f"[export] {w}x{h} ref{int(refiner)}: ONNX ...")
    entry.update(_run_worker("onnx", w, h, refiner, log))
    if entry["onnx_status"] != "ok":
        entry.update(engine_status="not_attempted", engine_reason="ONNX export failed")
        log(f"    ONNX failed: {entry.get('onnx_reason', '')[:200]}")
        return entry
    log(f"    ONNX ok in {entry.get('onnx_s')}s, peak RAM {entry.get('onnx_peak_ram_gib')} GiB; building engine ...")
    entry.update(_run_worker("engine", w, h, refiner, log))
    if entry["engine_status"] != "ok":
        log(f"    engine failed: {entry.get('engine_reason', '')[:200]}")
        return entry
    log(f"    engine ok in {entry.get('engine_build_s')}s ({entry.get('engine_size_mib')} MiB); measuring drift ...")
    try:
        entry.update(measure_drift(w, h, refiner))
        log(f"    drift {entry['drift']}")
    except Exception as e:
        entry["drift_error"] = f"{type(e).__name__}: {e}"[:1000]
        log(f"    drift failed: {entry['drift_error'][:200]}")
    return entry


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
