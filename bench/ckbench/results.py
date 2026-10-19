"""results.json load/save with schema validation on every save."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
OUTPUTS_DIR = REPO_ROOT / "outputs"
SCHEMA_PATH = Path(__file__).resolve().parent / "results.schema.json"


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(doc: dict) -> None:
    jsonschema.Draft202012Validator(_schema()).validate(doc)


def run_dir(run_id: str) -> Path:
    return RESULTS_DIR / run_id


def latest_run_id() -> str | None:
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(p.name for p in RESULTS_DIR.iterdir() if (p / "results.json").exists())
    return runs[-1] if runs else None


def load(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "results.json").read_text(encoding="utf-8"))


def save(doc: dict) -> Path:
    validate(doc)
    d = run_dir(doc["environment"]["run_id"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / "results.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def new_doc(environment: dict) -> dict:
    return {"schema_version": "1.0", "environment": environment, "benchmarks": []}


def upsert(items: list, new: dict, key) -> None:
    """Replace the entry with the same key(new), or append it."""
    k = key(new)
    for i, it in enumerate(items):
        if key(it) == k:
            items[i] = new
            return
    items.append(new)
