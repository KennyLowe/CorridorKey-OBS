"""results.json -> REPORT.md (US5; FR-009, FR-010, FR-016)."""
from __future__ import annotations

import json

from .config import LIVE_MODES
from .envelope import (A2000_RANGE, GATE_MIN_INTERNAL, GATE_MS, LOAD, envelope, latency_ms, verdict, worst_p95)
from .quality import EDGE_ALPHA_MAX, EDGE_FG_MAX, FLICKER_MAX

PRIOR_EVIDENCE = (
    "Before this spike, no frame rates had been published by CorridorKey's authors. Third-party ports "
    "reported, unverified: an After Effects port on an RTX 4090 (fp16, 1080p source, including I/O and "
    "post-processing) at about 187 ms/frame at 512 without the refiner, 230 ms with it, 286 ms at 1024 and "
    "612 ms at 2048; and a Nuke TensorRT port on an RTX A5000 (fp16, 2048) at about 300 ms/frame. Those "
    "figures made the 1080p30 gate unlikely before any measurement here; they include unknown host "
    "overhead and none were taken on this hardware."
)

GATE_TEXT = (
    f"**Gate (spec, Constitution I and II):** on the RTX 3060, some configuration at internal resolution "
    f">= {GATE_MIN_INTERNAL} (both sides) processes a 1080p frame end-to-end in <= {GATE_MS:.0f} ms (p95, "
    f"worst of 3 runs) while a sustained 1080p30 NVENC encode runs on the same GPU, with owner-accepted "
    f"quality and flicker on every reference clip. The margin below the 33.3 ms frame is headroom for the "
    f"slower A2000 and for OBS's own work. \"Internal resolution >= {GATE_MIN_INTERNAL}\" means both sides are "
    f">= {GATE_MIN_INTERNAL} (owner decision, 2026-10-17): non-square settings such as 768x448 cannot meet the "
    f"gate, but can still support lower live modes in the envelope (CONDITIONAL-GO)."
)

DEVIATIONS = [
    "TensorRT 10.16.1.11 instead of the newest 11.x: TensorRT 11 removed the weakly-typed `BuilderFlag.FP16` "
    "build the plan specifies (only strongly-typed networks remain).",
    "ffmpeg 7.1.1 (Gyan build) instead of the latest: ffmpeg 8.1+ requires NVENC API 13.1 (driver >= 610) and "
    "cannot open h264_nvenc on driver 576.80.",
    "Speed benchmarks always use a deterministic synthetic 1080p green-screen frame (tasks.md preferred "
    "pexels-athlete frame 0 when present) so idle/load and torch/TRT entries see identical input; the "
    "networks have static shapes, so content does not change the work.",
    "Peak VRAM comes from NVML device-used memory minus a baseline taken before the process initialised CUDA: "
    "NVML returns no per-process memory under Windows WDDM. Other processes' activity adds noise.",
    "Added latency uses the measured end-to-end wall median (which includes the two resize stages) plus one "
    "frame of queueing, slightly more conservative than research D12's stage sum.",
    "The alpha hint for the two Pexels clips is calibrated per clip (screen colour = median of a 32 px border "
    "band; difference key normalised to that colour): they are ungraded flat footage that the fixed hint "
    "(research D8) does not recognise as green screen (owner decision, 2026-10-17). All other clips use the "
    "fixed hint. Each clip's results entry records its hint, key colour and border diagnostic.",
    "Square configurations use CorridorKeyEngine's own bicubic position-embedding resize; the 16:9 variants "
    "(research D2) use the same resize to a rectangular token grid.",
]

LIMITATIONS = [
    "OBS's own compositing and rendering load is not simulated; the concurrent load is a 1080p30 NVENC encode only.",
    "Reference clips are public stock footage; results on the owner's own studio (screen, lighting, camera) may differ.",
    "The RTX A2000 was not measured. Every A2000 figure is an ESTIMATE scaled from the RTX 3060.",
    "The Hollywood Camera Work plates are below 1080p (1280x1080 and 960x720) and are upscaled for the quality run.",
    "The alpha hint is a simple chroma key (research D8); hint failures (e.g. green clothing) affect the key and "
    "are not corrected.",
    "Eager PyTorch only (no torch.compile: MSVC is not installed); upstream reports ~30% from compiling.",
]


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _speed_rows(benchmarks):
    rows = ["| runtime | internal | refiner | load | camera | status | model median / p95 ms | "
            "e2e median / p95 ms (worst run) | peak VRAM MiB | spread % | clock start/end MHz |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    key = lambda b: (b["config"]["load"], b["config"]["runtime"], b["camera_res"],  # noqa: E731
                     b["config"]["internal_w"] * b["config"]["internal_h"], b["config"]["refiner"])
    for b in sorted(benchmarks, key=key):
        c = b["config"]
        size = f"{c['internal_w']}x{c['internal_h']}"
        if b["status"] != "ok":
            reason = (b.get("status_reason") or "").splitlines()[0][:90] if b.get("status_reason") else ""
            rows.append(f"| {c['runtime']} | {size} | {'on' if c['refiner'] else 'off'} | {c['load']} | "
                        f"{b['camera_res']} | **{b['status']}** {reason} | - | - | - | - | - |")
            continue
        runs = b["runs"]
        mm = sorted(r["model_ms"]["median"] for r in runs)[len(runs) // 2]
        mp = max(r["model_ms"]["p95"] for r in runs)
        em = sorted(r["e2e_ms"]["median"] for r in runs)[len(runs) // 2]
        flag = " **FLAG**" if b.get("spread_flag") else ""
        rows.append(f"| {c['runtime']} | {size} | {'on' if c['refiner'] else 'off'} | {c['load']} | {b['camera_res']} "
                    f"| ok | {mm:.1f} / {mp:.1f} | {em:.1f} / {worst_p95(b):.1f} | {_f(b.get('peak_vram_mib'), 0)} "
                    f"| {_f(b.get('median_spread_pct'))}{flag} | {b.get('clock_start_mhz')}/{b.get('clock_end_mhz')} |")
    return rows


def _slowdown_rows(benchmarks):
    idle = {(b["config"]["config_id"].replace("-idle", ""), b["camera_res"]): b
            for b in benchmarks if b["status"] == "ok" and b["config"]["load"] == "idle"}
    rows = ["| configuration | camera | idle e2e median ms | load e2e median ms | slowdown |", "|---|---|---|---|---|"]
    n = 0
    for b in benchmarks:
        if b["status"] != "ok" or b["config"]["load"] != LOAD:
            continue
        k = (b["config"]["config_id"].replace(f"-{LOAD}", ""), b["camera_res"])
        if k not in idle:
            continue
        mi = sorted(r["e2e_ms"]["median"] for r in idle[k]["runs"])[1 if len(idle[k]["runs"]) > 2 else 0]
        ml = sorted(r["e2e_ms"]["median"] for r in b["runs"])[1 if len(b["runs"]) > 2 else 0]
        rows.append(f"| {k[0]} | {k[1]} | {mi:.1f} | {ml:.1f} | {(ml / mi - 1) * 100:+.1f}% |")
        n += 1
    return rows if n else ["(no load measurements with a matching idle measurement)"]


def _quality_reference_run() -> str | None:
    """The latest committed run that has owner quality verdicts (the RTX 3060 run today)."""
    from .results import RESULTS_DIR

    for d in sorted((p for p in RESULTS_DIR.iterdir() if (p / "results.json").exists()), reverse=True):
        if json.loads((d / "results.json").read_text(encoding="utf-8")).get("quality"):
            return d.name
    return None


def render(doc: dict, allow_pending: bool, quality_from: dict | None = None) -> tuple[str, dict, list, list]:
    benchmarks = doc.get("benchmarks", [])
    quality = doc.get("quality", [])
    note = None
    if quality_from is not None and not quality:
        quality = quality_from.get("quality", [])
        note = (f"Quality verdicts borrowed from run {quality_from['environment']['run_id']} "
                f"({quality_from['environment']['gpu_name']}), matched by configuration: the model and its "
                "outputs are the same on any GPU; only speed is measured here.")
    speed_only = not quality
    if speed_only:
        ref = _quality_reference_run()
        note = ("Speed-only run: quality not run on this machine" + (f"; see run {ref}" if ref else "") +
                ". Configurations are judged on timing alone below.")
    env = envelope(benchmarks, quality, allow_pending, quality_missing_ok=speed_only)
    v = verdict(benchmarks, quality, env, allow_pending, quality_missing_ok=speed_only)
    gpu = doc["environment"]["gpu_name"]
    if "3060" not in gpu:
        v["reasoning"] = v["reasoning"].replace("RTX 3060", gpu)
    v["prior_evidence_note"] = PRIOR_EVIDENCE
    v["draft"] = bool(allow_pending)
    e = doc["environment"]
    L = []
    if allow_pending:
        L += ["> **DRAFT — owner review pending.** Quality verdicts still `pending` are treated as passes "
              "below, so this verdict is provisional.", ""]
    L += [f"# CorridorKey live-keying feasibility: {v['verdict']}", "",
          f"Run `{e['run_id']}`, {e['gpu_name']}, driver {e['gpu_driver']}.", "",
          f"**Verdict: {v['verdict']}.** {v['reasoning']}", ""]
    if v.get("a2000_risk"):
        L += [f"**A2000 (ESTIMATE):** {v['a2000_risk']}", ""]
    if note:
        L += [f"> {note}", ""]
    L += ["## Prior evidence", "", PRIOR_EVIDENCE, "", "## Gate", "", GATE_TEXT, ""]

    L += ["## Operating envelope", "",
          f"Each mode's budget is 1000/fps ms of end-to-end time under the encode load, with no extra margin. "
          f"A2000 columns are an **ESTIMATE (x{A2000_RANGE[0]}-x{A2000_RANGE[1]} of the 3060)**: the 3060 time "
          f"scaled by the memory-bandwidth ratio (360/288 GB/s, optimistic) to the FP32 throughput ratio "
          f"(12.7/8.0 TFLOPS, conservative). The A2000 counts as supported only when the conservative bound "
          f"fits, and as borderline when the range straddles the budget. Once a mode fails, heavier modes at "
          f"the same or higher fps are not attempted. A mode's best configuration is the one with the largest "
          f"internal pixel count that fits with quality passed (ties: lowest p95); the refiner setting is not "
          f"ranked. When nothing fits, the fastest configuration is shown. Load runs (T032) covered every "
          f"idle-ok configuration whose idle 1080p e2e median was <= 60 ms (the loosest budget, 50 ms, + 20%); the "
          f"rest are recorded as not attempted - exceeds budget.", "",
          f"| mode | budget ms | {gpu} (measured) | best config | p95 ms | quality | A2000 ESTIMATE | A2000 range ms |",
          "|---|---|---|---|---|---|---|---|"]
    for en in env:
        r, a = en["rtx3060"], en["a2000_estimate"]
        rng = a.get("e2e_ms_range")
        a_s = a["status"] + (" (borderline)" if a.get("borderline") else "") +             (f" ({a['best_config_id']})" if a.get("best_config_id") else "")
        L.append(f"| {en['mode_id']} | {en['budget_ms']:.1f} | {r['status']}"
                 + (f" ({r['reason']})" if r.get("reason") and r["status"] != "supported" else "")
                 + f" | {r.get('best_config_id', '-')} | {_f(r.get('e2e_ms_p95'))} | "
                 + ("not run" if r.get("quality") == "not_run" else r.get("quality", "-")) + " | "
                 f"{a_s} | {('%.1f-%.1f' % tuple(rng)) if rng else '-'} |")
    L.append("")

    L += ["## Speed and memory", "",
          "Model time is CUDA-event time of the network alone. End-to-end time is wall-clock per frame: pinned "
          "upload of a uint8 camera frame, chroma-key hint, resize to internal size, network, resize back, RGBA8 "
          "pack and download. Each configuration was run 3 times (50 warm-up and 300 timed frames per run); "
          "spread is the max-min of run medians as a % of the min (flagged above 10%, SC-004).", ""]
    L += _speed_rows(benchmarks) + [""]
    L += ["### Slowdown under the NVENC encode load", ""] + _slowdown_rows(benchmarks) + [""]

    exports = doc.get("exports", [])
    L += ["## TensorRT conversion and drift", ""]
    if exports:
        L += ["| internal | refiner | ONNX | ONNX peak RAM GiB | engine | build s | engine MiB | drift alpha MAE / max | "
              "drift fg MAE / max | drift fg MAE / max where alpha > 0.1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for x in sorted(exports, key=lambda x: (x["internal_w"] * x["internal_h"], x.get("refiner", True))):
            d = x.get("drift") or {}
            reason = ""
            if x["onnx_status"] != "ok":
                reason = " " + (x.get("onnx_reason") or "").splitlines()[0][:80]
            elif x["engine_status"] != "ok":
                reason = " " + (x.get("engine_reason") or "").splitlines()[0][:80]
            L.append(f"| {x['internal_w']}x{x['internal_h']} | {'on' if x.get('refiner', True) else 'off'} | "
                     f"{x['onnx_status']} | {_f(x.get('onnx_peak_ram_gib'))} | {x['engine_status']}{reason} | "
                     f"{_f(x.get('engine_build_s'), 0)} | {_f(x.get('engine_size_mib'), 0)} | "
                     f"{_f(d.get('alpha_mae'), 4)} / {_f(d.get('alpha_max'), 3)} | "
                     f"{_f(d.get('fg_mae'), 4)} / {_f(d.get('fg_max'), 3)} | "
                     f"{_f(d.get('fg_mae_alpha_gt_0_1'), 4)} / {_f(d.get('fg_max_alpha_gt_0_1'), 3)} |")
        L += ["", "Drift is |TensorRT - PyTorch fp16| of the raw network outputs at the internal size, over 20 "
              "fixed frames (" + (next((x.get("drift_frames") for x in exports if x.get("drift_frames")), "-")) + ").", ""]
    else:
        L += ["Not run.", ""]

    clips = doc.get("clips", [])
    L += ["## Reference clips", "", "| clip | focus | native | frames | hint (border mean) | licence | source |",
          "|---|---|---|---|---|---|---|"]
    for c in clips:
        h = c.get("hint", "fixed")
        if h == "calibrated":
            h += (f" key {tuple(c['hint_key_colour'])} ({c['hint_border_mean']}; fixed hint was "
                  f"{c['hint_diagnostic_fixed']['hint_border_mean']})")
        else:
            h += f" ({c.get('hint_border_mean', '-')})"
        L.append(f"| {c['clip_id']} | {c['focus']} | {c.get('native_res', '-')} @ {c.get('native_fps', '-')} | "
                 f"{c.get('frames', '-')} | {h} | {c['licence']} (checked {c.get('licence_checked_on', '-')}) | "
                 f"{c.get('page_url', c['source_url'])} |")
    L += ["", "Clips are fetched from their source (or downloaded by hand for the Hollywood Camera Work plates), "
          "never committed or rehosted, and normalised to 1920x1080 at 30 fps (aspect kept, padded with screen "
          "green), up to 300 frames. No stills derived from them are committed.", ""]

    L += ["## Quality and flicker", "",
          f"Against the PyTorch 2048 refiner-on reference on the same frames. Flag thresholds were declared before "
          f"any quality run: edge alpha MAE > {EDGE_ALPHA_MAX}, edge FG MAE > {EDGE_FG_MAX}, flicker excess > "
          f"{FLICKER_MAX}. Flags only direct attention; the owner's visual verdict decides, and passing a "
          f"flagged entry needs a written reason.", ""]
    if quality:
        L += ["| config | clip | edge alpha MAE | edge FG MAE | flicker excess | flags | owner verdict | reason |",
              "|---|---|---|---|---|---|---|---|"]
        for q in sorted(quality, key=lambda q: (q["config_id"], q["clip_id"])):
            L.append(f"| {q['config_id']} | {q['clip_id']} | {q['edge_alpha_mae']:.4f} | {q['edge_fg_mae']:.4f} | "
                     f"{q['flicker_excess']:.4f} | {', '.join(q['flags']) or '-'} | {q['owner_verdict']} | "
                     f"{q.get('override_reason') or q.get('review_note') or ''} |")
        L.append("")
    else:
        L += ["Not run.", ""]

    L += ["## Added latency", "",
          "Estimated delay the keyer adds = measured end-to-end median + one frame of queueing (asynchronous, "
          "off the render thread).", "",
          "| mode | configuration | e2e median ms | + queue ms | added ms | frames @30 | frames @60 |",
          "|---|---|---|---|---|---|---|"]
    lat_out = []
    by_id = {(b["config"]["config_id"], b["camera_res"]): b for b in benchmarks if b["status"] == "ok"}
    for en in env:
        m = next(mm for mm in LIVE_MODES if mm.mode_id == en["mode_id"])
        cid = en["rtx3060"].get("best_config_id")
        b = by_id.get((cid, m.camera_res)) if cid else None
        if not b:
            continue
        lat = latency_ms(b, m.fps)
        lat_out.append({"mode_id": m.mode_id, "config_id": cid, **lat})
        L.append(f"| {m.mode_id} | {cid} | {lat['e2e_median_ms']:.1f} | {lat['queue_ms']:.1f} | {lat['ms']:.1f} | "
                 f"{lat['frames_at_30']:.2f} | {lat['frames_at_60']:.2f} |")
    L.append("")

    L += ["## Environment", "", "| item | value |", "|---|---|"]
    for k in ("gpu_name", "gpu_driver", "gpu_vram_mib", "gpu_max_sm_clock_mhz", "gpu_power_limit_w", "os", "cpu",
              "ram_gib", "python", "torch_cuda", "cudnn", "tensorrt_version", "ffmpeg_version", "corridorkey_sha",
              "weights_file", "weights_sha256"):
        if k in e:
            L.append(f"| {k} | {e[k]} |")
    L.append("| packages | " + ", ".join(f"{k} {v}" for k, v in e["packages"].items()) + " |")
    L.append("")

    notes = doc.get("notes", {})
    if notes.get("reproducibility"):
        L += ["## Reproducibility (SC-005)", "", notes["reproducibility"], ""]
    if notes.get("schedule"):
        L += ["## Schedule (SC-006)", "", notes["schedule"], ""]
    L += ["## Deviations from the plan", ""] + [f"- {d}" for d in DEVIATIONS + notes.get("deviations", [])] + [""]
    L += ["## Limitations", ""] + [f"- {x}" for x in LIMITATIONS] + [""]
    L += ["## Reproduce", "", "See [`bench/README.md`](../../README.md): `bench\\setup.ps1`, then `uv run ckbench all` "
          "(or the individual commands). `results.json` in this folder holds every number above.", ""]
    return "\n".join(L), v, env, lat_out


def pending_candidates(doc: dict) -> list[str]:
    """Quality entries that feed a load-measured configuration and are still pending."""
    load_idle_ids = {b["config"]["config_id"].replace(f"-{LOAD}", "-idle") for b in doc.get("benchmarks", [])
                     if b["status"] == "ok" and b["config"]["load"] == LOAD}
    return [f"{q['config_id']} x {q['clip_id']}" for q in doc.get("quality", [])
            if q["owner_verdict"] == "pending" and q["config_id"] in load_idle_ids]
