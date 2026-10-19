"""ckbench perf-table: committed results.json runs -> plugin/data/perf-table.json (feature 002, T012)."""
import json
from pathlib import Path

import jsonschema
import pytest

from ckbench import perftable as PT

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUN_3060 = "20261016T221252Z-c16425"


def bench(cid_size, ref, cam, p95s, status="ok", runtime="trt", load="nvenc1080p30"):
    w, h = cid_size
    cid = f"{runtime}-fp16-{w}x{h}-ref{int(ref)}-{load}"
    e = {"config": {"config_id": cid, "runtime": runtime, "internal_w": w, "internal_h": h, "refiner": ref,
                    "load": load, "precision": "fp16"}, "camera_res": cam, "status": status}
    if status == "ok":
        e["runs"] = [{"e2e_ms": {"p95": p, "median": p - 1}} for p in p95s]
    return e


def doc(run_id, gpu, benchmarks, quality=None):
    return {"schema_version": "1.0", "environment": {"run_id": run_id, "gpu_name": gpu, "gpu_driver": "576.80"},
            "benchmarks": benchmarks, "quality": quality or []}


def test_quality_id_mapping():
    assert PT.quality_id("trt-fp16-1024x576-ref0-nvenc1080p30") == "1024x576_r0"
    assert PT.quality_id("trt-fp16-512x512-ref1-idle") == "512_r1"
    assert PT.quality_id("trt-fp16-768x768-ref0-nvenc1080p30") == "768_r0"
    assert PT.quality_id("trt-fp16-2048x2048-ref0-nvenc1080p30") is None  # not a live setting
    assert PT.quality_id("torch-fp16-512x512-ref0-nvenc1080p30") is None  # plugin runs TensorRT only


def test_table_from_one_run():
    d = doc("r1", "GPU X", [
        bench((512, 512), False, "1080p", [14.0, 15.7, 15.1]),
        bench((1024, 576), False, "1080p", [30.1, 29.0]),
        bench((1024, 1024), False, "1080p", [54.0]),
        bench((512, 512), False, "720p", [14.2]),
        bench((768, 768), True, "1080p", [], status="not_attempted"),
        bench((512, 512), False, "1080p", [9.0], load="idle"),        # idle: ignored
        bench((512, 512), False, "1080p", [9.0], runtime="torch"),    # torch: ignored
    ])
    t = PT.build_table([d])
    jsonschema.Draft202012Validator(PT.schema()).validate(t)
    assert t["generated_from"] == ["r1"]
    (g,) = t["gpus"]
    assert g["gpu_name"] == "GPU X" and g["driver"] == "576.80" and g["run_id"] == "r1"
    modes = {m["mode_id"]: m for m in g["modes"]}
    assert set(modes) == {"720p20", "720p30", "1080p30", "1080p60"}
    c = {x["quality"]: x for x in modes["1080p30"]["configs"]}
    assert c["512_r0"] == {"quality": "512_r0", "p95_ms_load": 15.7, "fits": True}  # worst run's p95
    assert c["1024x576_r0"]["fits"] is True
    assert c["1024_r0"]["fits"] is False
    assert "768_r1" not in c
    c60 = {x["quality"]: x for x in modes["1080p60"]["configs"]}
    assert c60["512_r0"]["fits"] is True and c60["1024x576_r0"]["fits"] is False
    assert modes["1080p60"]["budget_ms"] == pytest.approx(1000 / 60)


def test_failed_quality_never_fits():
    q = [{"config_id": "trt-fp16-512x512-ref0-idle", "clip_id": "c", "owner_verdict": "fail"}]
    t = PT.build_table([doc("r1", "GPU X", [bench((512, 512), False, "1080p", [10.0])], q)])
    c = t["gpus"][0]["modes"][0]["configs"][0]
    assert c["fits"] is False


def test_one_run_per_gpu_most_complete_wins():
    full = doc("20260101T000000Z-aaaaaa", "GPU X", [bench((512, 512), False, "1080p", [15.0]),
                                                    bench((1024, 576), False, "1080p", [30.0])])
    partial = doc("20260202T000000Z-bbbbbb", "GPU X", [bench((1024, 576), False, "1080p", [31.5])])
    other = doc("20260303T000000Z-cccccc", "GPU Y", [bench((512, 512), False, "1080p", [20.0])])
    t = PT.build_table([full, partial, other])
    assert [g["gpu_name"] for g in t["gpus"]] == ["GPU X", "GPU Y"]
    assert t["gpus"][0]["run_id"] == "20260101T000000Z-aaaaaa"
    assert t["generated_from"] == ["20260101T000000Z-aaaaaa", "20260303T000000Z-cccccc"]


def test_real_3060_run():
    t = PT.build_table([json.loads((RESULTS / RUN_3060 / "results.json").read_text())])
    jsonschema.Draft202012Validator(PT.schema()).validate(t)
    (g,) = t["gpus"]
    assert g["gpu_name"] == "NVIDIA GeForce RTX 3060"
    m = {x["mode_id"]: {c["quality"]: c for c in x["configs"]} for x in g["modes"]}
    # matches the 001 report's envelope: 1080p30 sustained by 1024x576 at 30.1 ms p95
    assert m["1080p30"]["1024x576_r0"]["fits"] and m["1080p30"]["1024x576_r0"]["p95_ms_load"] == pytest.approx(30.1, abs=0.05)
    assert m["1080p60"]["512_r0"]["fits"] and not m["1080p60"]["768x448_r0"]["fits"]
    assert not any(c["fits"] for c in m["2160p60"].values())


def test_cli_writes_the_file(tmp_path):
    from ckbench import cli

    out = tmp_path / "perf-table.json"
    rc = cli.main(["perf-table", "--out", str(out), "--runs", RUN_3060])
    assert rc == 0
    t = json.loads(out.read_text())
    jsonschema.Draft202012Validator(PT.schema()).validate(t)
    assert t["generated_from"] == [RUN_3060]
