"""US2 + US4 in real OBS 32.2.2 (T022, T023): live key, never break the stream.

Runs under a concurrent NVENC recording (Simple output, NVENC H.264 at 6 Mb/s) for realism.
SC-002: OBS's active fps with the filter stays within 1% of the rate without it and the render
thread skips no frames. SC-004: forced failures fall back within the timeout and recover.
"""
import time

import cv2
import numpy as np
import pytest

import media
import obs_harness as H
from conftest import log_events, now_s, record

SRC = "clip"
FPS_TOL = 0.01
MEASURE_S = 30


def keying(s):
    return s.get("found") and s["state"] in ("keying", "behind")


def wait_quality(o, q, timeout=60, src=SRC, flt="CorridorKey"):
    return o.wait_status(src, lambda s: keying(s) and s["quality"] == q, timeout, flt=flt)


@pytest.fixture(scope="module")
def scene(obs_session):
    o = obs_session
    o.add_color_background("bg", (128, 128, 128))
    o.add_media(SRC, media.clip_mp4("hcw-locked-backdrop"))
    o.add_filter(SRC, H.default_filter_settings(quality="512_r0"))
    o.enable_filter(SRC, False)
    time.sleep(2)
    o.start_record()
    time.sleep(3)
    yield o
    o.stop_record()


def test_a_fps_unaffected_under_nvenc(scene):
    o = scene
    assert o.send("GetRecordStatus")["outputActive"]
    assert any("nvenc" in l.lower() for l in o.log_text().splitlines()[-200:]), "recording is not using NVENC"
    base = H.measure(o, MEASURE_S)
    rows = {"baseline (filter disabled)": base}
    o.enable_filter(SRC, True)
    for q in ("512_r0", "1024x576_r0"):
        o.set_filter(SRC, {"quality": q})
        wait_quality(o, q)
        time.sleep(3)
        m = H.measure(o, MEASURE_S)
        st = o.status(SRC)
        m.update(quality=q, worker_ms_median=st["workerMsMedian"], worker_ms_p95=st["workerMsP95"],
                 skipped_pct=st["skippedPct"], added_latency_ms=st["addedLatencyMs"],
                 added_latency_frames=st["addedLatencyFrames"], render_us_median=st["renderUsMedian"],
                 render_us_p95=st["renderUsP95"], render_us_p99=st["renderUsP99"], state=st["state"])
        rows[q] = m
    record("sc002_fps", rows)
    for q in ("512_r0", "1024x576_r0"):
        m = rows[q]
        assert abs(m["active_fps_mean"] - base["active_fps_mean"]) <= FPS_TOL * base["active_fps_mean"], rows
        assert m["render_skipped"] == 0, rows
        assert m["state"] in ("keying", "behind"), rows


def _fallback_after(o, setting, expect_reason, timeout_ms=250):
    o.set_filter(SRC, {"quality": "512_r0", "fallback_timeout_ms": timeout_ms})
    wait_quality(o, "512_r0")
    time.sleep(1.5)
    t_set = now_s()
    o.set_filter(SRC, {"debug_fail": setting})
    s = o.wait_status(SRC, lambda s: s["state"] == "fallback", 5)
    assert expect_reason in s["reason"], s
    assert s["output"] == "colorkey", s
    ev = log_events(o, r"debug_fail=" + setting, t_set - 1)
    fb = log_events(o, r"-> fallback", t_set - 1)
    assert ev and fb, (ev, fb)
    delay_ms = (fb[0][0] - ev[0][0]) * 1000
    fps_during = H.measure(o, 5)
    o.set_filter(SRC, {"debug_fail": "none"})
    t_clear = now_s()
    o.wait_status(SRC, keying, 10)
    rec = log_events(o, r"fallback -> keying", t_clear - 1)
    recover_ms = (rec[0][0] - t_clear) * 1000 if rec else None
    return delay_ms, fb[0][1], fps_during, recover_ms


def test_b_worker_error_falls_back_and_recovers(scene):
    delay_ms, line, fps, recover_ms = _fallback_after(scene, "worker_error", "simulated keyer failure")
    record("sc004_worker_error", {"fallback_delay_ms": delay_ms, "log": line, "fps": fps, "recover_ms": recover_ms})
    # immediate on error: within the timeout + one frame of the failure being injected
    assert delay_ms <= 250 + 34, line
    assert fps["render_skipped"] == 0
    assert abs(fps["active_fps_mean"] - 30.0) <= 0.3


def test_c_stall_times_out_and_recovers(scene):
    delay_ms, line, fps, recover_ms = _fallback_after(scene, "stall", "timeout")
    age = float(line.split("last matte ")[1].split(" ms")[0])
    record("sc004_stall", {"fallback_delay_ms": delay_ms, "last_matte_age_ms": age, "log": line, "fps": fps,
                           "recover_ms": recover_ms})
    # the fallback appears at the first frame after the timeout has elapsed since the last matte
    assert 250 <= age <= 250 + 34, line
    assert fps["render_skipped"] == 0
    assert abs(fps["active_fps_mean"] - 30.0) <= 0.3


def test_d_missing_tensorrt_folder_is_unsupported(scene):
    o = scene
    o.set_filter(SRC, {"trt_dir": r"C:\nonexistent\TensorRT\bin"})
    s = o.wait_status(SRC, lambda s: s["state"] == "unsupported", 10)
    assert "nvinfer_10.dll not found" in s["reason"]
    assert s["output"] == "colorkey"
    img = o.screenshot(SRC)
    assert img[..., 3].mean() < 250  # the colour key removes the screen
    m = H.measure(o, 5)
    assert o.alive() and m["render_skipped"] == 0
    record("sc004_missing_trt", {"status": s["status"], "fps": m})
    o.set_filter(SRC, {"trt_dir": H.TRT_DIR})
    o.wait_status(SRC, keying, 30)


def test_e_weights_removed_passes_through(scene):
    o = scene
    for path in ("", r"C:\nonexistent\CorridorKey_v1.0.safetensors"):
        o.set_filter(SRC, {"weights_path": path})
        s = o.wait_status(SRC, lambda s: s["state"] == "no_weights", 10)
        assert s["output"] == "passthrough", s
        img = o.screenshot(SRC)
        assert img[..., 3].min() == 255  # untouched: fully opaque
    record("weights_removed", {"status": s["status"]})
    o.set_filter(SRC, {"weights_path": str(H.WEIGHTS)})
    o.wait_status(SRC, keying, 30)


def test_f_overload_two_instances_fps_holds(scene):
    o = scene
    o.add_media("clip2", media.clip_mp4("hcw-godiva-close"))
    o.add_filter("clip2", H.default_filter_settings(quality="1024_r1"))
    o.set_filter(SRC, {"quality": "1024_r1"})
    o.wait_status(SRC, lambda s: s["quality"] == "1024_r1" and s["mattes"] > 0, 60)
    o.wait_status("clip2", lambda s: s["quality"] == "1024_r1" and s.get("mattes", 0) > 0, 60)
    time.sleep(3)
    m = H.measure(o, 20)
    s1, s2 = o.status(SRC), o.status("clip2")
    record("sc004_overload", {"fps": m, "clip": {k: s1[k] for k in ("state", "workerMsMedian", "skippedPct", "renderUsP99")},
                              "clip2": {k: s2[k] for k in ("state", "workerMsMedian", "skippedPct", "renderUsP99")}})
    # the GPU can't key two 1024x1024+refiner sources at 30 fps: both fall behind or back...
    assert s1["state"] in ("behind", "fallback") and s2["state"] in ("behind", "fallback"), (s1, s2)
    assert s1["skippedPct"] > 0 or s1["state"] == "fallback"
    # ...but OBS keeps rendering at its frame rate
    assert m["render_skipped"] == 0, m
    assert abs(m["active_fps_mean"] - 30.0) <= 0.3, m
    o.remove_filter("clip2")
    o.send("RemoveInput", {"inputName": "clip2"})
    o.set_filter(SRC, {"quality": "1024x576_r0"})


@pytest.mark.parametrize("clip_id", ["hcw-locked-backdrop", "hcw-godiva-close", "pixabay-hair"])
def test_g_reference_stills_for_owner_review(scene, clip_id):
    """T023: a keyed still of each reference clip over grey (git-ignored; never committed)."""
    o = scene
    name = f"still-{clip_id}"
    o.add_media(name, media.clip_mp4(clip_id))
    o.add_filter(name, H.default_filter_settings(quality="1024x576_r0"))
    o.wait_status(name, lambda s: keying(s) and s["mattes"] > 10, 60)
    time.sleep(1)
    img = o.screenshot(name, 1920, 1080).astype(np.float32)
    a = img[..., 3:4] / 255.0
    comp = img[..., :3] * a + 128.0 * (1 - a)
    out = H.ARTIFACTS / "stills" / f"{clip_id}-1024x576_r0-over-grey.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR))
    assert 0.02 < a.mean() < 0.98  # something was keyed, something kept
    o.remove_filter(name)
    o.send("RemoveInput", {"inputName": name})
