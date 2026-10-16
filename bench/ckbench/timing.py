"""Timing helpers: statistics, run-to-run spread, and the warm-up + timed loop (research D7)."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

SPREAD_FLAG_PCT = 10.0  # SC-004


def stats(samples) -> dict:
    a = np.asarray(list(samples), dtype=np.float64)
    if a.size == 0:
        raise ValueError("stats() needs at least one sample")
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def spread(run_medians) -> tuple[float, bool]:
    """Run-to-run spread of medians: (max-min)/min*100, flagged when > 10% (SC-004)."""
    m = [float(x) for x in run_medians]
    if len(m) < 2:
        return 0.0, False
    pct = (max(m) - min(m)) / min(m) * 100.0
    return pct, bool(pct > SPREAD_FLAG_PCT + 1e-9)


def run_timed(fn: Callable[[], dict], warmup: int = 50, frames: int = 300) -> dict:
    """Call ``fn`` ``warmup`` times untimed, then ``frames`` times timed.

    ``fn`` processes one frame and returns a dict of per-stage GPU times in ms (from CUDA events);
    the stage named ``infer`` is the model time. End-to-end time is wall-clock (perf_counter)
    around ``fn``, which must itself finish with ``torch.cuda.synchronize()`` (pipeline.py does),
    so the wall time covers all queued GPU work.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e2e: list[float] = []
    st: dict[str, list[float]] = {}
    for _ in range(frames):
        t0 = time.perf_counter()
        stages = fn()
        e2e.append((time.perf_counter() - t0) * 1000.0)
        for k, v in stages.items():
            st.setdefault(k, []).append(v)
    return {"e2e": e2e, "model": st.get("infer", []), "stages": st}


def summarise_run(raw: dict) -> dict:
    """Turn one run_timed() result into a results.json run entry (stage_ms = per-stage medians)."""
    return {
        "model_ms": stats(raw["model"]),
        "e2e_ms": stats(raw["e2e"]),
        "stage_ms": {k: float(np.median(v)) for k, v in raw["stages"].items()},
    }
