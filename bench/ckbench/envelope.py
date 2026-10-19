"""Latency model, operating envelope and verdict (US4/US5; research D12-D14; data-model.md).

Interpretations (stated in the report):
- A configuration's time for a mode is its *worst run's* p95 end-to-end ms under the concurrent
  NVENC load (load = nvenc1080p30) at that mode's camera resolution.
- "Internal resolution >= 768" (the gate) means both internal sides are >= 768.
- A2000 figures are ESTIMATES: the 3060 time x 1.25 (memory-bandwidth ratio, optimistic) to
  x 1.59 (FP32 throughput ratio). A2000 status is "supported" only when the slow bound fits the
  budget. "borderline" means the range straddles it.
- Added latency = measured end-to-end wall median (upload, hint, resizes, inference, download)
  + one frame period of queueing (asynchronous pipeline, Constitution III).
"""
from __future__ import annotations

from .config import LIVE_MODES, LiveMode

A2000_RANGE = (1.25, 1.59)
GATE_MODE = "1080p30"
GATE_MS = 20.0
GATE_MIN_INTERNAL = 768
LOAD = "nvenc1080p30"
_RES_ORDER = {"720p": 0, "1080p": 1, "2160p": 2}


def worst_p95(entry: dict) -> float:
    return max(r["e2e_ms"]["p95"] for r in entry["runs"])


def e2e_median(entry: dict) -> float:
    meds = sorted(r["e2e_ms"]["median"] for r in entry["runs"])
    return meds[len(meds) // 2]


def latency_ms(entry: dict, fps: float) -> dict:
    ms = e2e_median(entry) + 1000.0 / fps
    stages = {}
    for r in entry["runs"]:
        for k, v in r.get("stage_ms", {}).items():
            stages.setdefault(k, []).append(v)
    return {"ms": ms, "frames_at_30": ms / (1000 / 30), "frames_at_60": ms / (1000 / 60),
            "queue_ms": 1000.0 / fps, "e2e_median_ms": e2e_median(entry),
            "stage_ms": {k: sorted(v)[len(v) // 2] for k, v in stages.items()}}


def idle_id(config_id: str) -> str:
    return config_id.replace(f"-{LOAD}", "-idle")


def quality_status(quality: list[dict], config_id: str) -> str:
    qs = [q for q in quality if q["config_id"] == idle_id(config_id)]
    if not qs:
        return "missing"
    v = [q["owner_verdict"] for q in qs]
    if "fail" in v:
        return "fail"
    if "pending" in v:
        return "pending"
    return "pass"


def _quality_ok(status: str, allow_pending: bool) -> bool:
    return status == "pass" or (allow_pending and status == "pending")


def _pixels(e: dict) -> int:
    return e["config"]["internal_w"] * e["config"]["internal_h"]


def _candidates(benchmarks: list[dict], camera_res: str) -> list[dict]:
    return [b for b in benchmarks if b["status"] == "ok" and b["config"]["load"] == LOAD
            and b["camera_res"] == camera_res and b.get("runs")]


def _dominated(mode: LiveMode, failed: list[LiveMode]) -> bool:
    return any(_RES_ORDER[f.camera_res] <= _RES_ORDER[mode.camera_res] and f.fps <= mode.fps for f in failed)


def _a2000(p95: float, budget: float) -> dict:
    lo, hi = p95 * A2000_RANGE[0], p95 * A2000_RANGE[1]
    return {"e2e_ms_range": [lo, hi], "borderline": bool(lo <= budget < hi),
            "status": "supported" if hi <= budget else "unsupported"}


def envelope(benchmarks: list[dict], quality: list[dict], allow_pending: bool = False) -> list[dict]:
    out = []
    failed_3060: list[LiveMode] = []
    failed_a2000: list[LiveMode] = []
    for mode in LIVE_MODES:
        budget = mode.budget_ms
        cands = _candidates(benchmarks, mode.camera_res)
        entry: dict = {"mode_id": mode.mode_id, "budget_ms": budget}

        # --- RTX 3060 (measured)
        if _dominated(mode, failed_3060):
            r = {"status": "not_attempted", "reason": "a lighter mode at the same or lower fps already failed"}
        elif not cands:
            r = {"status": "not_attempted", "reason": f"no load measurement at {mode.camera_res}"}
        else:
            scored = [(b, worst_p95(b), quality_status(quality, b["config"]["config_id"])) for b in cands]
            ok = [s for s in scored if s[1] <= budget and _quality_ok(s[2], allow_pending)]
            if ok:
                b, p95, qs = max(ok, key=lambda s: (_pixels(s[0]), -s[1]))
                r = {"status": "supported"}
            else:
                b, p95, qs = min(scored, key=lambda s: s[1])
                fast = [s for s in scored if s[1] <= budget]
                r = {"status": "unsupported",
                     "reason": ("fits the budget but quality not passed" if fast else "no configuration within budget")}
                failed_3060.append(mode)
            r.update(best_config_id=b["config"]["config_id"], e2e_ms_p95=p95, quality=qs)
        entry["rtx3060"] = r

        # --- A2000 (ESTIMATE)
        if _dominated(mode, failed_a2000):
            a = {"status": "not_attempted", "reason": "a lighter mode at the same or lower fps already failed"}
        elif not cands:
            a = {"status": "not_attempted", "reason": f"no load measurement at {mode.camera_res}"}
        else:
            prio = {"supported": 0, "borderline": 1, "unsupported": 2}
            best = None
            for b in cands:
                qs = quality_status(quality, b["config"]["config_id"])
                if not _quality_ok(qs, allow_pending):
                    continue
                est = _a2000(worst_p95(b), budget)
                k = "supported" if est["status"] == "supported" else ("borderline" if est["borderline"] else "unsupported")
                # fitting (or borderline): prefer the largest internal size; otherwise the fastest
                rank = (prio[k], -_pixels(b) if k != "unsupported" else 0, worst_p95(b))
                if best is None or rank < best[0]:
                    best = (rank, b, est)
            if best is None:
                a = {"status": "unsupported", "reason": "no quality-passed configuration"}
            else:
                _, b, est = best
                a = {**est, "best_config_id": b["config"]["config_id"], "method": "ESTIMATE x1.25-x1.59 of 3060 p95"}
            if a["status"] != "supported":
                failed_a2000.append(mode)
        entry["a2000_estimate"] = a
        out.append(entry)
    return out


def verdict(benchmarks: list[dict], quality: list[dict], env: list[dict], allow_pending: bool = False) -> dict:
    by = {e["mode_id"]: e for e in env}
    gate = []
    for b in _candidates(benchmarks, "1080p"):
        c = b["config"]
        if min(c["internal_w"], c["internal_h"]) < GATE_MIN_INTERNAL:
            continue
        p95 = worst_p95(b)
        if p95 <= GATE_MS and _quality_ok(quality_status(quality, c["config_id"]), allow_pending):
            gate.append((b, p95))
    if gate:
        b, p95 = max(gate, key=lambda s: (_pixels(s[0]), -s[1]))
        cid = b["config"]["config_id"]
        v = {"verdict": "GO", "deciding_config_id": cid,
             "reasoning": (f"1080p30 gate met on the RTX 3060: {cid} runs at {p95:.1f} ms p95 end-to-end "
                           f"under a concurrent 1080p30 NVENC encode (<= {GATE_MS:.0f} ms) at internal "
                           f"resolution >= {GATE_MIN_INTERNAL}, with owner-accepted quality."),
             "deciding_mode": GATE_MODE}
        est = _a2000(p95, by[GATE_MODE]["budget_ms"])
        lo, hi = est["e2e_ms_range"]
        v["a2000_risk"] = (f"A2000 ESTIMATE for {cid} at 1080p30: {lo:.1f}-{hi:.1f} ms vs the "
                           f"{by[GATE_MODE]['budget_ms']:.1f} ms frame budget"
                           + (" (BORDERLINE)" if est["borderline"] else
                              ("" if est["status"] == "supported" else " (EXCEEDS BUDGET)"))
                           + f"; the {GATE_MS:.0f} ms headroom target is "
                           + ("met" if hi <= GATE_MS else "not met by the conservative bound")
                           + ". Must be confirmed on the A2000.")
        return v

    supported = [e for e in env if e["rtx3060"]["status"] == "supported"]
    if supported:
        order = {m.mode_id: i for i, m in enumerate(LIVE_MODES)}
        # Decide on the target mode (FR-015) when it is supported, else the heaviest supported mode.
        target = by.get(GATE_MODE)
        top = target if target and target["rtx3060"]["status"] == "supported" else             max(supported, key=lambda e: order[e["mode_id"]])
        r = top["rtx3060"]
        what = ("The target mode" if top["mode_id"] == GATE_MODE else "The heaviest live mode the RTX 3060 sustains")
        v = {"verdict": "CONDITIONAL-GO", "deciding_config_id": r["best_config_id"], "deciding_mode": top["mode_id"],
             "reasoning": (f"The 1080p30 gate (<= {GATE_MS:.0f} ms p95 at internal >= {GATE_MIN_INTERNAL} on both "
                           f"sides, under encode load, quality accepted) is not met. {what}, {top['mode_id']}, is "
                           f"sustained with {r['best_config_id']} at {r['e2e_ms_p95']:.1f} ms p95 (budget "
                           f"{top['budget_ms']:.1f} ms) but without the gate's headroom. Supported modes on the "
                           f"RTX 3060: " + ", ".join(e["mode_id"] for e in supported) + ".")}
        a = top["a2000_estimate"]
        if a.get("e2e_ms_range"):
            lo, hi = a["e2e_ms_range"]
            v["a2000_risk"] = (f"A2000 ESTIMATE for {top['mode_id']}: {lo:.1f}-{hi:.1f} ms vs {top['budget_ms']:.1f} ms "
                               f"budget: {a['status']}" + (" (BORDERLINE)" if a.get("borderline") else "")
                               + ". Must be confirmed on the A2000.")
        else:
            v["a2000_risk"] = f"A2000 ESTIMATE for {top['mode_id']}: {a['status']} ({a.get('reason', '')})."
        return v

    return {"verdict": "NO-GO",
            "reasoning": ("No live mode from 720p20 upwards is sustained on the RTX 3060 under encode load "
                          "with owner-accepted quality."),
            "a2000_risk": "The A2000 is estimated to be 1.25-1.59x slower, so it cannot do better."}
