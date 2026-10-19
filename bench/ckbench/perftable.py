"""`ckbench perf-table`: committed benchmark runs -> the OBS plugin's Auto table (feature 002).

For each GPU, one run is used: the one with the most completed TensorRT load measurements (ties:
the latest run id). For each live mode, every TensorRT configuration measured under the NVENC load
at that mode's camera resolution is listed with its worst run's p95 end-to-end time (the same
statistic the 001 envelope uses). ``fits`` = that p95 is within the mode's budget and the owner
didn't fail its quality. Configurations that aren't plugin quality settings (2048, PyTorch, idle)
are left out. Output validates against ``perf-table.schema.json``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

from .config import LIVE_MODES
from .envelope import LOAD, quality_status, worst_p95
from .results import REPO_ROOT, RESULTS_DIR

SCHEMA_PATH = Path(__file__).resolve().parent / "perf-table.schema.json"
DEFAULT_OUT = REPO_ROOT / "plugin" / "data" / "perf-table.json"
QUALITY_IDS = ("512_r0", "512_r1", "768x448_r0", "768x448_r1", "768_r0", "768_r1",
               "1024x576_r0", "1024x576_r1", "1024_r0", "1024_r1")


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def quality_id(config_id: str) -> str | None:
    """'trt-fp16-1024x576-ref0-nvenc1080p30' -> '1024x576_r0'; None if not a plugin setting."""
    parts = config_id.split("-")
    if len(parts) < 4 or parts[0] != "trt":
        return None
    w, h = parts[2].split("x")
    size = w if w == h else f"{w}x{h}"
    q = f"{size}_r{parts[3][3:]}"
    return q if q in QUALITY_IDS else None


def _load_ok(doc: dict) -> list[dict]:
    return [b for b in doc.get("benchmarks", []) if b.get("status") == "ok" and b["config"]["load"] == LOAD
            and b.get("runs") and quality_id(b["config"]["config_id"])]


def _gpu_entry(doc: dict) -> dict:
    env = doc["environment"]
    measured = _load_ok(doc)
    modes = []
    for mode in LIVE_MODES:
        configs = []
        for b in measured:
            if b["camera_res"] != mode.camera_res:
                continue
            p95 = worst_p95(b)
            qs = quality_status(doc.get("quality", []), b["config"]["config_id"])
            configs.append({"quality": quality_id(b["config"]["config_id"]), "p95_ms_load": round(p95, 2),
                            "fits": bool(p95 <= mode.budget_ms and qs != "fail")})
        if configs:
            configs.sort(key=lambda c: QUALITY_IDS.index(c["quality"]))
            modes.append({"mode_id": mode.mode_id, "budget_ms": mode.budget_ms, "configs": configs})
    return {"gpu_name": env["gpu_name"], "driver": env["gpu_driver"], "run_id": env["run_id"], "modes": modes}


def build_table(docs: list[dict]) -> dict:
    by_gpu: dict[str, dict] = {}
    for d in docs:
        g = d["environment"]["gpu_name"]
        cur = by_gpu.get(g)
        score = (len(_load_ok(d)), d["environment"]["run_id"])
        if cur is None or score > (len(_load_ok(cur)), cur["environment"]["run_id"]):
            by_gpu[g] = d
    chosen = [d for d in by_gpu.values() if _load_ok(d)]
    chosen.sort(key=lambda d: d["environment"]["run_id"])
    table = {"schema_version": "1.0", "generated_from": [d["environment"]["run_id"] for d in chosen],
             "gpus": [_gpu_entry(d) for d in chosen]}
    jsonschema.Draft202012Validator(schema()).validate(table)
    return table


def load_runs(run_ids: list[str] | None = None) -> list[dict]:
    dirs = [RESULTS_DIR / r for r in run_ids] if run_ids else sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    return [json.loads((d / "results.json").read_text(encoding="utf-8")) for d in dirs if (d / "results.json").exists()]


def write(table: dict, out: Path = DEFAULT_OUT) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return out
