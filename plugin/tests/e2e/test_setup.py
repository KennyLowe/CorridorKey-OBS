"""US3 first-run setup in real OBS (T026): download -> verify -> prepare -> key without leaving OBS,
restart within 10 s, wrong weights, OBS killed mid-preparation, bad/offline downloads, and the
newer-model notice. Each scenario runs its own OBS session with test-only environment overrides
(CK_WEIGHTS_DIR, CK_ENGINE_CACHE, CK_MODEL_MANIFEST, CK_MODELS_API) so the real weights folder and
engine cache are never touched.

The real download (scenario A) fetches the official 399 MB weights from Hugging Face.
"""
import hashlib
import http.server
import json
import os
import shutil
import threading
import time
from pathlib import Path

import psutil
import pytest

import media
import obs_harness as H
from conftest import record

WORK = H.REPO / "outputs" / "e2e" / "setup"
FIXTURES = H.REPO / "bench" / "tests" / "fixtures"
PLUGIN_CFG = H.CFG / "plugin_config" / "corridorkey-obs" / "config.json"
SRC = "cam"


def fresh(name: str) -> Path:
    d = WORK / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def session(env: dict) -> H.Obs:
    H.write_test_config(H.ARTIFACTS / "rec")
    PLUGIN_CFG.unlink(missing_ok=True)  # licence acknowledgement + model-check throttle start clean
    o = H.Obs(extra_env=env).start()
    o.add_color_background("bg", (128, 128, 128))
    o.add_media(SRC, media.clip_mp4("hcw-locked-backdrop"))
    return o


def setup_settings(**over) -> dict:
    s = {"trt_dir": H.TRT_DIR, "quality": "auto", "fallback_mode": "colorkey", "debug_fail": "none"}
    s.update(over)
    return s


def keying(s):
    return s.get("found") and s["state"] in ("keying", "behind")


def helper_procs(marker: str) -> list:
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if marker in " ".join(p.info["cmdline"] or []):
                out.append(p)
        except (psutil.Error, TypeError):
            pass
    return out


# ----------------------------------------------------------------------------- A: real first run

def test_a_first_run_download_prepare_key_then_restart():
    wdir, cache = fresh("A-weights"), fresh("A-engines")
    env = {"CK_WEIGHTS_DIR": wdir, "CK_ENGINE_CACHE": cache}
    o = session(env)
    try:
        t0 = time.time()
        o.add_filter(SRC, setup_settings())
        s = o.wait_status(SRC, lambda s: s["state"] == "no_weights" and "Download weights" in s["reason"], 30)
        assert s["output"] == "passthrough", s
        assert o.screenshot(SRC)[..., 3].min() == 255

        # without the licence acknowledgement the download is refused
        o.set_filter(SRC, {"download_requested": 1})
        s = o.wait_status(SRC, lambda s: "licence" in s["reason"].lower(), 10)
        assert not list(wdir.iterdir())

        o.set_filter(SRC, {"licence_ack": True, "download_requested": 2})
        o.wait_status(SRC, lambda s: s["state"] == "downloading", 20)
        fps = []
        timeline = []
        st = None
        while True:  # downloading -> preparing -> keying; OBS keeps rendering meanwhile
            st = o.status(SRC)
            fps.append(o.stats())
            timeline.append({"t": round(time.time() - t0, 1), "state": st["state"], "progress": st["workerProgress"],
                             "render_skipped": fps[-1]["renderSkippedFrames"], "render_us_max": st["renderUsMax"],
                             "render_us_p99": st["renderUsP99"]})
            if keying(st) or st["state"] == "failed" or time.time() - t0 > 900:
                break
            time.sleep(2)
        elapsed = time.time() - t0
        assert keying(st), st
        assert st["modelVerified"] and "verified" in st["modelLabel"], st
        assert (wdir / "CorridorKey_v1.0.safetensors").exists() and not list(wdir.glob("*.part"))
        skipped = fps[-1]["renderSkippedFrames"] - fps[0]["renderSkippedFrames"]
        fps_vals = [f["activeFps"] for f in fps]
        record("sc003_first_run", {"seconds_add_filter_to_key": round(elapsed, 1), "quality": st["quality"],
                                   "render_skipped_during_setup": skipped, "active_fps_min": min(fps_vals),
                                   "active_fps_mean": sum(fps_vals) / len(fps_vals), "timeline": timeline})
        assert elapsed < 15 * 60
        assert skipped == 0 and min(fps_vals) >= 29.7

        # a wrong weights file: specific error, pass-through
        bad = wdir / "not-corridorkey.safetensors"
        bad.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      " + os.urandom(1000))
        o.set_filter(SRC, {"weights_path": str(bad)})
        s = o.wait_status(SRC, lambda s: s["state"] == "no_weights" and "weights_invalid" in s["reason"], 120)
        assert s["output"] == "passthrough"
        record("wrong_weights", {"status": s["status"]})
        o.set_filter(SRC, {"weights_path": str(wdir / "CorridorKey_v1.0.safetensors")})
        o.wait_status(SRC, keying, 60)
    finally:
        o.close()

    # restart: keying within 10 s, no re-preparation (SC-003 second half)
    o = session(env)
    try:
        t0 = time.time()
        o.add_filter(SRC, setup_settings())
        s = o.wait_status(SRC, keying, 30, poll=0.1)
        restart_s = time.time() - t0
        assert not any("preparing " in l and ": uv" in l for l in o.ck_log())
        record("sc003_restart", {"seconds_to_key": round(restart_s, 2), "quality": s["quality"]})
        assert restart_s < 10
    finally:
        o.close()


# ----------------------------------------------------------------------------- B: killed mid-preparation

def test_b_obs_killed_during_preparation_restarts_cleanly():
    cache = fresh("B-engines")
    env = {"CK_ENGINE_CACHE": cache}
    settings = setup_settings(weights_path=str(H.WEIGHTS), quality="768x448_r0")
    o = session(env)
    try:
        o.add_filter(SRC, settings)
        o.wait_status(SRC, lambda s: s["state"] == "preparing" and s["workerProgress"] >= 25, 180)
        assert helper_procs(str(cache))
    finally:
        o.kill()  # no clean shutdown
    time.sleep(3)
    assert helper_procs(str(cache)) == []  # the Job object took the helper tree down with OBS
    man = cache / "manifest.json"
    if man.exists():
        json.loads(man.read_text())  # never half-written
    o = session(env)
    try:
        o.add_filter(SRC, settings)
        s = o.wait_status(SRC, keying, 600)
        assert s["quality"] == "768x448_r0"
        entries = json.loads(man.read_text())["entries"]
        assert [e["internal_w"] for e in entries] == [768]
        record("killed_during_prepare", {"recovered": True, "entries": len(entries)})
    finally:
        o.close()


# ----------------------------------------------------------------------------- C: fixture server

PAYLOAD = os.urandom(200_000)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/models"):
            body = (FIXTURES / "hf_models_nikopueringer.json").read_bytes()
        elif "/resolve/" in self.path:
            body = PAYLOAD
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fixture_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_c_bad_download_offline_and_newer_model_notice(fixture_server):
    wdir = fresh("C-weights")
    real = json.loads((H.REPO / "models" / "manifest.json").read_text())["models"][0]
    man = WORK / "C-manifest.json"
    man.write_text(json.dumps({"schema_version": "1.0", "models": [
        real,
        {**real, "model_id": "Bad_v1.0", "hf_repo": "org/bad", "revision": "r", "file": "bad.safetensors",
         "size": len(PAYLOAD), "sha256": "0" * 64, "base_url": fixture_server},
        {**real, "model_id": "Offline_v1.0", "hf_repo": "org/off", "revision": "r", "file": "off.safetensors",
         "base_url": "http://10.255.255.1:9"},
    ]}))
    env = {"CK_WEIGHTS_DIR": wdir, "CK_MODEL_MANIFEST": man, "CK_MODELS_API": f"{fixture_server}/api/models"}
    o = session(env)
    try:
        # the real model, already keying from a local file
        o.add_filter(SRC, setup_settings(weights_path=str(H.WEIGHTS), quality="512_r0"))
        o.wait_status(SRC, keying, 120)
        base = H.measure(o, 3)

        # hash mismatch: specific error, nothing partial kept, pass-through
        o.add_media("bad", media.clip_mp4("hcw-locked-backdrop"))
        o.add_filter("bad", setup_settings(model_id="Bad_v1.0", licence_ack=True,
                                           weights_path=str(wdir / "Bad_v1.0.safetensors")))
        o.set_filter("bad", {"download_requested": 1})
        s = o.wait_status("bad", lambda s: s["state"] == "no_weights" and "hash_mismatch" in s["reason"], 60)
        assert s["output"] == "passthrough" and "retry" in s["reason"].lower()
        assert list(wdir.iterdir()) == []
        record("download_hash_mismatch", {"status": s["status"]})

        # offline: network error with retry; OBS unaffected
        o.add_media("off", media.clip_mp4("hcw-locked-backdrop"))
        o.add_filter("off", setup_settings(model_id="Offline_v1.0", licence_ack=True,
                                           weights_path=str(wdir / "Offline_v1.0.safetensors")))
        o.set_filter("off", {"download_requested": 1})
        s = o.wait_status("off", lambda s: s["state"] == "no_weights" and "network" in s["reason"], 60)
        m = H.measure(o, 3)
        assert m["render_skipped"] == 0 and abs(m["active_fps_mean"] - base["active_fps_mean"]) < 0.3
        assert s["output"] == "passthrough"
        record("download_offline", {"status": s["status"], "fps": m})

        # newer model in the (fixture) model list: notice shown, nothing switched
        s = o.wait_status(SRC, lambda s: "CorridorKey_v1.1" in s["newerModels"], 60)
        assert keying(s) and s["modelVerified"] and "CorridorKey v1.0" in s["modelLabel"]
        assert "CorridorKeyBlue" not in s["newerModels"]
        record("newer_model_notice", {"newer": s["newerModels"], "model": s["modelLabel"], "state": s["state"]})
    finally:
        o.close()
