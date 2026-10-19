"""ckbench fetch-weights / check-models (feature 002, T011a; contracts/prepare-cli.md).

Unit tests use a local HTTP server (with HTTP Range support) serving a small fake weights file
through a test manifest whose base_url points at it. No network. The real download is one opt-in
test: `uv run pytest -m network`.
"""
import hashlib
import http.server
import json
import os
import threading
from pathlib import Path

import pytest

from ckbench import fetch as F

REPO = Path(__file__).resolve().parents[2]
PAYLOAD = os.urandom(300_000)
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    payload = PAYLOAD
    status_override = None
    drop_after = None  # close the connection after this many bytes (simulated network failure)
    requests = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        Handler.requests.append((self.path, self.headers.get("Range")))
        if Handler.status_override:
            self.send_response(Handler.status_override)
            self.end_headers()
            return
        if not self.path.startswith("/org/fake/resolve/rev1/fake.safetensors"):
            self.send_response(404)
            self.end_headers()
            return
        data, start = Handler.payload, 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if Handler.drop_after is not None:
            self.wfile.write(body[:Handler.drop_after])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    Handler.payload, Handler.status_override, Handler.drop_after, Handler.requests = PAYLOAD, None, None, []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def manifest(tmp_path, base_url, sha=SHA, size=len(PAYLOAD)) -> Path:
    m = {"schema_version": "1.0", "models": [{
        "model_id": "Fake_v1.0", "hf_repo": "org/fake", "revision": "rev1", "file": "fake.safetensors",
        "size": size, "sha256": sha, "licence": "test", "licence_url": "http://example.invalid",
        "screen_color": "green", "base_url": base_url}]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p


def run(tmp_path, man, model="Fake_v1.0"):
    ev = []
    rc = F.fetch_weights(str(man), model, str(tmp_path / "weights"), ev.append)
    return rc, ev


# ---------------------------------------------------------------------------- manifest

def test_repo_manifest_pins_the_measured_model():
    m = F.load_manifest(REPO / "models" / "manifest.json")
    (e,) = m["models"]
    assert e["model_id"] == "CorridorKey_v1.0"
    assert e["sha256"] == "74d614f7d92fc559a118c30a7deadedc3cacd8ef83dcb85a030d0bed7af8b20b"
    assert e["size"] == 398849256
    assert F.source_url(e) == ("https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/"
                               "f6386ddf042d8e92aeb5fd16cb9b101cff508195/CorridorKey_v1.0.safetensors")


def test_url_comes_only_from_the_manifest(tmp_path):
    e = F.load_manifest(manifest(tmp_path, "http://127.0.0.1:1"))["models"][0]
    assert F.source_url(e) == "http://127.0.0.1:1/org/fake/resolve/rev1/fake.safetensors"


@pytest.mark.parametrize("url,ok", [
    ("https://huggingface.co/x", True), ("https://cdn-lfs.huggingface.co/x", True),
    ("https://cas-bridge.xethub.hf.co/x", True), ("https://cdn-lfs-us-1.hf.co/x", True),
    ("https://evil.example.com/huggingface.co", False), ("http://huggingface.co/x", False),
    ("https://huggingface.co.evil.com/x", False),
])
def test_redirect_host_allowlist(url, ok):
    assert F.allowed_redirect(url, "https://huggingface.co") is ok


# ---------------------------------------------------------------------------- fetch

def test_download_verify_rename(tmp_path, server):
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc == 0
    assert ev[-1] == {"event": "result", "ok": True, "weights_path": str(tmp_path / "weights" / "fake.safetensors"),
                      "sha256": SHA, "model_id": "Fake_v1.0"}
    assert Path(ev[-1]["weights_path"]).read_bytes() == PAYLOAD
    assert not list((tmp_path / "weights").glob("*.part"))
    stages = [e["stage"] for e in ev if e["event"] == "progress"]
    assert stages[0] == "download" and "verify" in stages and stages[-1] == "done"
    prog = [e for e in ev if e["event"] == "progress" and e["stage"] == "download"]
    assert prog[-1]["bytes"] == len(PAYLOAD) and prog[-1]["total"] == len(PAYLOAD)
    assert [e["event"] for e in ev].count("result") == 1


def test_cache_hit_does_not_download(tmp_path, server):
    man = manifest(tmp_path, server)
    run(tmp_path, man)
    Handler.requests.clear()
    rc, ev = run(tmp_path, man)
    assert rc == 0 and ev[-1]["ok"] and Handler.requests == []


def test_resume_with_range(tmp_path, server):
    d = tmp_path / "weights"
    d.mkdir()
    (d / "fake.safetensors.part").write_bytes(PAYLOAD[:100_000])
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc == 0
    assert Handler.requests[0][1] == "bytes=100000-"
    assert (d / "fake.safetensors").read_bytes() == PAYLOAD


def test_hash_mismatch_deletes_the_part_file(tmp_path, server):
    rc, ev = run(tmp_path, manifest(tmp_path, server, sha="0" * 64))
    assert rc != 0 and ev[-1]["error"] == "hash_mismatch"
    assert not list((tmp_path / "weights").glob("*"))


def test_size_mismatch_is_a_hash_mismatch(tmp_path, server):
    rc, ev = run(tmp_path, manifest(tmp_path, server, size=len(PAYLOAD) + 1))
    assert rc != 0 and ev[-1]["error"] == "hash_mismatch"
    assert not (tmp_path / "weights" / "fake.safetensors").exists()


def test_http_error(tmp_path, server):
    Handler.status_override = 403
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc != 0 and ev[-1]["error"] == "http" and "403" in ev[-1]["msg"]


def test_connection_drop_keeps_part_for_resume(tmp_path, server):
    Handler.drop_after = 50_000
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc != 0 and ev[-1]["error"] == "network"
    assert not (tmp_path / "weights" / "fake.safetensors").exists()
    part = tmp_path / "weights" / "fake.safetensors.part"
    assert part.exists() and part.stat().st_size == 50_000
    Handler.drop_after = None
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc == 0 and Handler.requests[-1][1] == "bytes=50000-"


def test_offline(tmp_path):
    rc, ev = run(tmp_path, manifest(tmp_path, "http://10.255.255.1:9"))  # unroutable
    assert rc != 0 and ev[-1]["error"] == "network"


def test_unknown_model(tmp_path, server):
    rc, ev = run(tmp_path, manifest(tmp_path, server), model="Nope")
    assert rc != 0 and ev[-1]["error"] == "unknown_model"


def test_unwritable_destination(tmp_path, server):
    (tmp_path / "weights").write_text("a file")
    rc, ev = run(tmp_path, manifest(tmp_path, server))
    assert rc != 0 and ev[-1]["error"] in ("disk", "io")


# ---------------------------------------------------------------------------- check-models

def test_check_models_reports_only_newer_corridorkey_versions():
    listing = json.loads((Path(__file__).parent / "fixtures" / "hf_models_nikopueringer.json").read_text())
    man = F.load_manifest(REPO / "models" / "manifest.json")
    r = F.newer_models(man, listing)
    assert r["current"] == "CorridorKey_v1.0"
    assert [n["model_id"] for n in r["newer"]] == ["CorridorKey_v1.1"]  # Blue excluded, v0.9 older
    assert r["newer"][0]["hf_repo"] == "nikopueringer/CorridorKey_v1.1"
    assert r["newer"][0]["last_modified"] == "2026-10-01T10:00:00.000Z"
    assert F.newer_models(man, [x for x in listing if "v1.1" not in x["id"]])["newer"] == []


def test_check_models_network_failure_is_non_fatal():
    ev = []
    rc = F.check_models(str(REPO / "models" / "manifest.json"), ev.append, api_url="http://10.255.255.1:9/api")
    assert rc != 0 and ev[-1] == {"event": "result", "ok": False, "error": "network", "msg": ev[-1]["msg"]}


def test_cli_fetch_and_check(tmp_path, server, capsys, monkeypatch):
    from ckbench import cli

    man = manifest(tmp_path, server)
    rc = cli.main(["fetch-weights", "--manifest", str(man), "--model", "Fake_v1.0", "--dest-dir",
                   str(tmp_path / "w"), "--json"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and json.loads(lines[-1])["ok"] is True
    listing = (Path(__file__).parent / "fixtures" / "hf_models_nikopueringer.json").read_text()
    monkeypatch.setattr(F, "_get_json", lambda url, timeout: json.loads(listing))
    rc = cli.main(["check-models", "--manifest", str(REPO / "models" / "manifest.json"), "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and out["ok"] and out["newer"][0]["model_id"] == "CorridorKey_v1.1"


@pytest.mark.network
def test_real_download_matches_the_pinned_sha(tmp_path):
    ev = []
    rc = F.fetch_weights(str(REPO / "models" / "manifest.json"), "CorridorKey_v1.0", str(tmp_path), ev.append)
    assert rc == 0, ev[-1]
    assert ev[-1]["sha256"] == "74d614f7d92fc559a118c30a7deadedc3cacd8ef83dcb85a030d0bed7af8b20b"
