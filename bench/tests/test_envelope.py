import pytest

from ckbench.envelope import A2000_RANGE, envelope, latency_ms, quality_status, verdict, worst_p95


def bench(runtime, w, h, ref, cam, p95s, load="nvenc1080p30", status="ok", median=None):
    cid = f"{runtime}-fp16-{w}x{h}-ref{int(ref)}-{load}"
    runs = [{"model_ms": {"mean": p * 0.8, "median": p * 0.8, "p95": p * 0.85},
             "e2e_ms": {"mean": median or p * 0.9, "median": median or p * 0.9, "p95": p},
             "stage_ms": {"upload": 1.0, "hint": 0.5, "resize_in": 0.5, "infer": p * 0.8, "resize_out": 0.5,
                          "download": 1.0}} for p in p95s]
    cfg = {"config_id": cid, "runtime": runtime, "precision": "fp16", "internal_w": w, "internal_h": h,
           "refiner": ref, "load": load}
    e = {"config": cfg, "camera_res": cam, "status": status}
    if status == "ok":
        e["runs"] = runs
    return e


def q(config_id, clip, verdict="pass", flags=()):
    return {"config_id": config_id, "clip_id": clip, "edge_alpha_mae": 0, "edge_fg_mae": 0, "flicker_excess": 0,
            "flags": list(flags), "owner_verdict": verdict}


def qall(idle_id, verdict="pass", clips=("c1", "c2", "c3")):
    return [q(idle_id, c, verdict) for c in clips]


def by_mode(env):
    return {e["mode_id"]: e for e in env}


def test_worst_p95_and_latency():
    e = bench("trt", 768, 768, False, "1080p", [10.0, 12.0, 11.0], median=9.0)
    assert worst_p95(e) == 12.0
    lat = latency_ms(e, fps=30)
    assert lat["ms"] == pytest.approx(9.0 + 1000 / 30)
    assert lat["frames_at_30"] == pytest.approx(lat["ms"] / (1000 / 30))
    assert lat["frames_at_60"] == pytest.approx(lat["ms"] / (1000 / 60))


def test_quality_status_maps_load_id_to_idle():
    quality = qall("trt-fp16-768x768-ref0-idle")
    assert quality_status(quality, "trt-fp16-768x768-ref0-nvenc1080p30") == "pass"
    quality[1]["owner_verdict"] = "pending"
    assert quality_status(quality, "trt-fp16-768x768-ref0-nvenc1080p30") == "pending"
    quality[0]["owner_verdict"] = "fail"
    assert quality_status(quality, "trt-fp16-768x768-ref0-nvenc1080p30") == "fail"
    assert quality_status([], "trt-fp16-768x768-ref0-nvenc1080p30") == "missing"


def test_go_verdict():
    b = [bench(r, s, s, False, cam, [15.0])
         for r in ("trt",) for s in (512, 768) for cam in ("720p", "1080p", "2160p")]
    quality = qall("trt-fp16-512x512-ref0-idle") + qall("trt-fp16-768x768-ref0-idle")
    env = envelope(b, quality)
    m = by_mode(env)
    assert m["1080p30"]["rtx3060"]["status"] == "supported"
    assert m["1080p30"]["rtx3060"]["best_config_id"] == "trt-fp16-768x768-ref0-nvenc1080p30"
    assert m["1080p60"]["rtx3060"]["status"] == "supported"  # 15 <= 16.67
    v = verdict(b, quality, env)
    assert v["verdict"] == "GO"
    assert v["deciding_config_id"] == "trt-fp16-768x768-ref0-nvenc1080p30"


def test_gate_needs_768_and_20ms():
    # only 512 is fast enough -> 1080p30 supported but the gate (>= 768 at <= 20 ms) is not met
    b = [bench("trt", 512, 512, False, "1080p", [18.0]), bench("trt", 768, 768, False, "1080p", [25.0]),
         bench("trt", 512, 512, False, "720p", [17.0]), bench("trt", 768, 768, False, "720p", [24.0])]
    quality = qall("trt-fp16-512x512-ref0-idle") + qall("trt-fp16-768x768-ref0-idle")
    env = envelope(b, quality)
    v = verdict(b, quality, env)
    assert by_mode(env)["1080p30"]["rtx3060"]["status"] == "supported"
    assert v["verdict"] == "CONDITIONAL-GO"


def test_conditional_go_and_not_attempted_propagation():
    b = [bench("torch", 512, 512, False, "720p", [45.0]), bench("torch", 512, 512, False, "1080p", [48.0]),
         bench("torch", 512, 512, False, "2160p", [60.0])]
    quality = qall("torch-fp16-512x512-ref0-idle")
    env = envelope(b, quality)
    m = by_mode(env)
    assert m["720p20"]["rtx3060"]["status"] == "supported"      # 45 <= 50
    assert m["720p30"]["rtx3060"]["status"] == "unsupported"    # 45 > 33.3
    for mid in ("1080p30", "1080p60", "2160p30", "2160p60"):
        assert m[mid]["rtx3060"]["status"] == "not_attempted", mid
    assert verdict(b, quality, env)["verdict"] == "CONDITIONAL-GO"


def test_not_attempted_only_for_same_or_higher_fps_and_res():
    # 720p30 fails, but 720p20 is lighter in fps and stays evaluated
    b = [bench("torch", 512, 512, False, "720p", [40.0])]
    env = by_mode(envelope(b, qall("torch-fp16-512x512-ref0-idle")))
    assert env["720p20"]["rtx3060"]["status"] == "supported"
    assert env["720p30"]["rtx3060"]["status"] == "unsupported"
    assert env["1080p30"]["rtx3060"]["status"] == "not_attempted"


def test_no_go():
    b = [bench("torch", 512, 512, False, cam, [90.0]) for cam in ("720p", "1080p", "2160p")]
    quality = qall("torch-fp16-512x512-ref0-idle")
    env = envelope(b, quality)
    assert by_mode(env)["720p20"]["rtx3060"]["status"] == "unsupported"
    assert all(e["rtx3060"]["status"] != "supported" for e in env)
    v = verdict(b, quality, env)
    assert v["verdict"] == "NO-GO"


def test_quality_fail_blocks_support():
    b = [bench("trt", 768, 768, False, "720p", [10.0]), bench("trt", 768, 768, False, "1080p", [10.0])]
    quality = qall("trt-fp16-768x768-ref0-idle", verdict="fail")
    env = by_mode(envelope(b, quality))
    assert env["720p20"]["rtx3060"]["status"] == "unsupported"
    assert verdict(b, quality, list(env.values()))["verdict"] == "NO-GO"


def test_pending_blocks_unless_allowed():
    b = [bench("trt", 768, 768, False, "720p", [10.0])]
    quality = qall("trt-fp16-768x768-ref0-idle", verdict="pending")
    assert by_mode(envelope(b, quality))["720p20"]["rtx3060"]["status"] == "unsupported"
    env = by_mode(envelope(b, quality, allow_pending=True))
    assert env["720p20"]["rtx3060"]["status"] == "supported"
    assert env["720p20"]["rtx3060"]["quality"] == "pending"


def test_idle_and_failed_entries_ignored():
    b = [bench("trt", 768, 768, False, "720p", [10.0], load="idle"),
         bench("trt", 1024, 1024, False, "720p", [10.0], status="oom")]
    quality = qall("trt-fp16-768x768-ref0-idle") + qall("trt-fp16-1024x1024-ref0-idle")
    env = by_mode(envelope(b, quality))
    assert env["720p20"]["rtx3060"]["status"] == "not_attempted"  # no load measurement


def test_a2000_estimate_and_borderline():
    assert A2000_RANGE == (1.25, 1.59)
    # 3060 p95 25 ms at 720p: A2000 range [31.25, 39.75]
    b = [bench("trt", 768, 768, False, "720p", [25.0])]
    env = by_mode(envelope(b, qall("trt-fp16-768x768-ref0-idle")))
    a20, a30 = env["720p20"]["a2000_estimate"], env["720p30"]["a2000_estimate"]
    assert a20["e2e_ms_range"] == pytest.approx([31.25, 39.75])
    assert a20["status"] == "supported" and a20["borderline"] is False      # 39.75 <= 50
    assert a30["borderline"] is True                                        # 31.25 <= 33.3 < 39.75
    assert a30["status"] == "unsupported"                                   # conservative bound decides
    assert env["720p30"]["rtx3060"]["status"] == "supported"


def test_go_with_borderline_a2000_records_risk():
    b = [bench("trt", 768, 768, False, cam, [19.0]) for cam in ("720p", "1080p")]
    quality = qall("trt-fp16-768x768-ref0-idle")
    env = envelope(b, quality)
    v = verdict(b, quality, env)
    assert v["verdict"] == "GO"
    assert v.get("a2000_risk")  # 23.75-30.2 ms vs the 20 ms gate / 33.3 ms budget must be called out


def test_non_square_768x448_cannot_meet_gate_but_supports_lower_modes():
    # spec Session 2026-10-17: "internal >= 768" means BOTH sides; 768x448 is fast but not gate-eligible
    b = [bench("trt", 768, 448, False, cam, [15.0]) for cam in ("720p", "1080p")]
    quality = qall("trt-fp16-768x448-ref0-idle")
    env = envelope(b, quality)
    m = by_mode(env)
    assert m["1080p30"]["rtx3060"]["status"] == "supported"
    assert m["1080p30"]["rtx3060"]["best_config_id"] == "trt-fp16-768x448-ref0-nvenc1080p30"
    v = verdict(b, quality, env)
    assert v["verdict"] == "CONDITIONAL-GO"
