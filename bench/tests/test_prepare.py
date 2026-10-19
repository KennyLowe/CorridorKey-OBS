"""ckbench prepare (contracts/prepare-cli.md): JSON-lines progress, one final result, cache manifest
shared with the plugin, atomic writes, idempotent cache hits and --prune. The ONNX export and the
engine build are injected fakes here; the real ones are exercised by the plugin's parity test."""
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ckbench import prepare as P

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "plugin" / "tests" / "data" / "manifest_fixture.json"

ENV = P.Identity(gpu_name="NVIDIA GeForce RTX 3060", gpu_uuid="GPU-0763dee5-2e3a-a1b2-955d-4600c81fd6c0",
                 driver_version="576.80", trt_version=P.TRT_VERSION)


def fake_weights(path: Path, ok: bool = True) -> Path:
    t = {"x": np.zeros(1, np.float32)}
    if ok:
        t = {k: np.zeros(1, np.float32) for k in P.REQUIRED_KEYS}
    save_file(t, str(path))
    return path


class FakeBuilders:
    def __init__(self, onnx_ok=True, engine_ok=True, engine_reason="build_serialized_network returned None"):
        self.calls = []
        self.onnx_ok, self.engine_ok, self.engine_reason = onnx_ok, engine_ok, engine_reason

    def onnx(self, weights, w, h, refiner, onnx_path, log_path, beat):
        self.calls.append("onnx")
        beat()
        if not self.onnx_ok:
            return {"onnx_status": "failed", "onnx_reason": "RuntimeError: boom"}
        Path(onnx_path).write_bytes(b"onnx")
        return {"onnx_status": "ok", "onnx_s": 1.0}

    def engine(self, w, h, refiner, onnx_path, engine_path, log_path, beat):
        self.calls.append("engine")
        beat()
        if not self.engine_ok:
            return {"engine_status": "failed", "engine_reason": self.engine_reason}
        Path(engine_path).write_bytes(b"engine")
        return {"engine_status": "ok", "engine_build_s": 2.0}


def run(tmp_path, weights, builders, size=(512, 512), refiner=False, env=ENV, prune=False):
    lines = []
    rc = P.prepare(str(weights), size[0], size[1], refiner, str(tmp_path / "cache"), lines.append,
                   identity=lambda: env, builders=builders, prune=prune)
    return rc, [json.loads(json.dumps(x)) for x in lines]


# ---------------------------------------------------------------------------- key and manifest

def test_key_matches_the_plugin_fixture():
    doc = json.loads(FIXTURE.read_text())
    e = doc["entries"][0]
    comp = {k: e[k] for k in P.KEY_FIELDS}
    assert P.cache_key(comp) == e["key"]
    assert P.canonical(comp).startswith('{"driver_version":"576.80","gpu_uuid":')


def test_manifest_roundtrip_is_atomic(tmp_path):
    m = tmp_path / "manifest.json"
    P.write_manifest(m, {"schema_version": "1.0", "entries": [{"key": "k"}]})
    assert not (tmp_path / "manifest.json.tmp").exists()
    assert P.read_manifest(m)["entries"] == [{"key": "k"}]


def test_corrupt_or_missing_manifest_reads_empty(tmp_path):
    m = tmp_path / "manifest.json"
    assert P.read_manifest(m) == {"schema_version": "1.0", "entries": []}
    m.write_text('{"entries": [')
    assert P.read_manifest(m) == {"schema_version": "1.0", "entries": []}


# ---------------------------------------------------------------------------- weights

def test_weights_verified(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    assert len(P.verify_weights(w)) == 64


def test_not_a_corridorkey_checkpoint(tmp_path):
    with pytest.raises(P.PrepareError) as e:
        P.verify_weights(fake_weights(tmp_path / "other.safetensors", ok=False))
    assert e.value.code == "weights_invalid"


def test_garbage_file(tmp_path):
    g = tmp_path / "w.safetensors"
    g.write_bytes(b"\x00" * 100)
    with pytest.raises(P.PrepareError) as e:
        P.verify_weights(g)
    assert e.value.code == "weights_invalid"


def test_missing_weights_file(tmp_path):
    with pytest.raises(P.PrepareError) as e:
        P.verify_weights(tmp_path / "nope.safetensors")
    assert e.value.code == "weights_invalid"


# ---------------------------------------------------------------------------- prepare flow

def test_success_events_and_manifest(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    b = FakeBuilders()
    rc, ev = run(tmp_path, w, b, size=(1024, 576))
    assert rc == 0
    assert b.calls == ["onnx", "engine"]
    assert all(e["event"] in ("progress", "result") for e in ev)
    assert [e["event"] for e in ev].count("result") == 1 and ev[-1]["event"] == "result"
    stages = [e["stage"] for e in ev if e["event"] == "progress"]
    assert stages[0] == "verify" and "onnx" in stages and "engine" in stages and stages[-1] == "done"
    pcts = [e["pct"] for e in ev if e["event"] == "progress"]
    assert pcts == sorted(pcts) and pcts[-1] == 100 and all(0 <= p <= 100 for p in pcts)
    r = ev[-1]
    assert r["ok"] is True and r["trt_version"] == P.TRT_VERSION and Path(r["engine_path"]).exists()
    assert r["engine_path"].endswith("ck_1024x576_ref0.engine")
    man = P.read_manifest(tmp_path / "cache" / "manifest.json")
    (e,) = man["entries"]
    assert e["key"] == r["key"]
    assert e["internal_w"] == 1024 and e["internal_h"] == 576 and e["refiner"] is False
    assert e["gpu_uuid"] == ENV.gpu_uuid and e["driver_version"] == "576.80"
    assert e["weights_sha256"] == P.verify_weights(w)
    assert e["key"] == P.cache_key({k: e[k] for k in P.KEY_FIELDS})
    assert e["gpu_name"] == ENV.gpu_name and "created" in e and "build_s" in e


def test_cache_hit_is_idempotent(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    run(tmp_path, w, FakeBuilders())
    b = FakeBuilders()
    rc, ev = run(tmp_path, w, b)
    assert rc == 0 and b.calls == []
    assert ev[-1]["ok"] is True
    assert len(P.read_manifest(tmp_path / "cache" / "manifest.json")["entries"]) == 1


def test_stale_entry_is_rebuilt_and_replaced(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    run(tmp_path, w, FakeBuilders())
    newer = P.Identity(**{**ENV._asdict(), "driver_version": "581.15"})
    b = FakeBuilders()
    rc, ev = run(tmp_path, w, b, env=newer)
    assert rc == 0 and b.calls == ["onnx", "engine"]
    entries = P.read_manifest(tmp_path / "cache" / "manifest.json")["entries"]
    assert {e["driver_version"] for e in entries} == {"576.80", "581.15"}


def test_engine_file_missing_means_rebuild(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    _, ev = run(tmp_path, w, FakeBuilders())
    Path(ev[-1]["engine_path"]).unlink()
    b = FakeBuilders()
    rc, _ = run(tmp_path, w, b)
    assert rc == 0 and b.calls == ["onnx", "engine"]
    assert len(P.read_manifest(tmp_path / "cache" / "manifest.json")["entries"]) == 1


@pytest.mark.parametrize("builders,code", [
    (FakeBuilders(onnx_ok=False), "export_failed"),
    (FakeBuilders(engine_ok=False), "build_failed"),
    (FakeBuilders(engine_ok=False, engine_reason="worker exit code 3: std::bad_alloc out of memory"), "oom"),
])
def test_failures(tmp_path, builders, code):
    w = fake_weights(tmp_path / "w.safetensors")
    rc, ev = run(tmp_path, w, builders)
    assert rc != 0
    assert ev[-1] == {"event": "result", "ok": False, "error": code, "msg": ev[-1]["msg"]}
    assert P.read_manifest(tmp_path / "cache" / "manifest.json")["entries"] == []


def test_invalid_weights_result(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors", ok=False)
    b = FakeBuilders()
    rc, ev = run(tmp_path, w, b)
    assert rc != 0 and b.calls == [] and ev[-1]["error"] == "weights_invalid"


def test_trt_version_mismatch(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    other = P.Identity(**{**ENV._asdict(), "trt_version": "10.13.0.35"})
    b = FakeBuilders()
    rc, ev = run(tmp_path, w, b, env=other)
    assert rc != 0 and b.calls == [] and ev[-1]["error"] == "trt_version_mismatch"


def test_unwritable_cache_is_io_error(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    (tmp_path / "cache").write_text("a file, not a folder")
    rc, ev = run(tmp_path, w, FakeBuilders())
    assert rc != 0 and ev[-1]["error"] == "io"


def test_prune_removes_other_environments(tmp_path):
    w = fake_weights(tmp_path / "w.safetensors")
    old = P.Identity(**{**ENV._asdict(), "driver_version": "560.00"})
    _, ev_old = run(tmp_path, w, FakeBuilders(), env=old)
    old_engine = Path(ev_old[-1]["engine_path"])
    assert old_engine.exists()
    rc, ev = run(tmp_path, w, FakeBuilders(), prune=True)
    assert rc == 0
    entries = P.read_manifest(tmp_path / "cache" / "manifest.json")["entries"]
    assert [e["driver_version"] for e in entries] == ["576.80"]
    assert not old_engine.exists()


def test_cli_wires_the_contract(tmp_path, monkeypatch, capsys):
    from ckbench import cli

    w = fake_weights(tmp_path / "w.safetensors")
    seen = {}

    def fake_prepare(weights, iw, ih, refiner, cache_dir, emit, prune=False, **kw):
        seen.update(weights=weights, size=(iw, ih), refiner=refiner, cache_dir=cache_dir, prune=prune)
        emit({"event": "result", "ok": True, "key": "k", "engine_path": "e", "trt_version": P.TRT_VERSION,
              "build_s": 0.0})
        return 0

    monkeypatch.setattr(P, "prepare", fake_prepare)
    rc = cli.main(["prepare", "--weights", str(w), "--size", "768x448", "--refiner", "1",
                   "--cache-dir", str(tmp_path / "c"), "--json", "--prune"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert seen == {"weights": str(w), "size": (768, 448), "refiner": True, "cache_dir": str(tmp_path / "c"),
                    "prune": True}
    assert json.loads(out[-1])["event"] == "result"
