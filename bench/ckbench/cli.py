"""`ckbench` command line (contracts/cli.md). Exit codes: 0 ok, 1 failed precondition, 2 partial."""
from __future__ import annotations

import argparse
import sys

from . import results as R

EXIT_OK, EXIT_PRECONDITION, EXIT_PARTIAL = 0, 1, 2


class Precondition(Exception):
    pass


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _sizes(s: str) -> list[tuple[int, int]]:
    out = []
    for x in _csv(s):
        if "x" in x:
            w, h = x.split("x")
            out.append((int(w), int(h)))
        else:
            out.append((int(x), int(x)))
    return out


def _onoff(s: str) -> list[bool]:
    m = {"on": True, "off": False}
    return [m[x] for x in _csv(s)]


def _load_doc(args) -> dict:
    run_id = args.run_id or R.latest_run_id()
    if not run_id:
        raise Precondition("no run found; run `ckbench env --new-run` first")
    return R.load(run_id)


# --------------------------------------------------------------------------- commands

def cmd_env(args) -> int:
    from . import env

    ok, err = env.nvenc_ok()
    import torch

    if not torch.cuda.is_available():
        raise Precondition("CUDA not available")
    environment = env.capture(None if (args.new_run or not args.run_id) else args.run_id)
    if args.run_id and not args.new_run and (R.run_dir(args.run_id) / "results.json").exists():
        doc = R.load(args.run_id)
        doc["environment"] = environment
    else:
        doc = R.new_doc(environment)
    p = R.save(doc)
    e = environment
    print(f"run_id          {e['run_id']}")
    print(f"gpu             {e['gpu_name']} ({e['gpu_vram_mib']} MiB), driver {e['gpu_driver']}")
    print(f"torch           {e['packages']['torch']} (CUDA {e['torch_cuda']})")
    print(f"tensorrt        {e['tensorrt_version']}")
    print(f"corridorkey     {e['corridorkey_sha']}")
    print(f"weights sha256  {e['weights_sha256']}")
    print(f"ffmpeg          {e['ffmpeg_version']}  (h264_nvenc {'OK' if ok else 'FAILED'})")
    print(f"results         {p}")
    if not ok:
        print(f"h264_nvenc check failed: {err}", file=sys.stderr)
        return EXIT_PRECONDITION
    return EXIT_OK


VARIANT_16X9 = [(1024, 576), (768, 448)]  # research D2; 768x432 rounded up to a multiple of 32
LOAD_ALIASES = {"idle": "idle", "nvenc": "nvenc1080p30", "nvenc1080p30": "nvenc1080p30"}


BEST_PRUNE_MS = 50.0 * 1.2  # loosest live-mode budget (720p20) with 20% slack, idle 1080p e2e median


def best_configs(doc: dict, runtimes: list[str]) -> tuple[list, list]:
    """T032 selection for load runs: every idle-ok configuration (each runtime x size x refiner)
    whose idle 1080p e2e median is within BEST_PRUNE_MS. Load only adds time, so the others
    cannot sustain any live mode; they are returned separately to be recorded as not_attempted."""
    from .config import Configuration
    from .envelope import e2e_median

    keep, pruned = [], []
    for b in doc.get("benchmarks", []):
        c = b["config"]
        if b["status"] != "ok" or c["load"] != "idle" or b["camera_res"] != "1080p" or c["runtime"] not in runtimes:
            continue
        cfg = Configuration.from_dict(c).with_load("nvenc1080p30")
        med = e2e_median(b)
        (keep if med <= BEST_PRUNE_MS else pruned).append((cfg, med))
    return keep, pruned


def cmd_speed(args) -> int:
    from .bench_speed import run_config
    from .config import Configuration

    doc = _load_doc(args)
    sizes = _sizes(args.sizes) + (VARIANT_16X9 if args.variant16x9 else [])
    loads = [LOAD_ALIASES[x] for x in _csv(args.load)]
    cameras = _csv(args.camera)
    partial = False
    for load in loads:
        load_ctx = None
        if load != "idle":
            from .load import nvenc_load

            load_ctx = nvenc_load
        if args.best and load != "idle":
            keep, pruned = best_configs(doc, _csv(args.runtime))
            for cfg, med in pruned:
                for cam in cameras:
                    R.upsert(doc["benchmarks"], {
                        "config": cfg.to_dict(), "camera_res": cam, "status": "not_attempted",
                        "status_reason": (f"not attempted - exceeds budget: idle 1080p e2e median {med:.1f} ms is "
                                          f"above {BEST_PRUNE_MS:.0f} ms (loosest budget 50 ms + 20%)")},
                        lambda e: (e["config"]["config_id"], e["camera_res"]))
                print(f"[speed] {cfg.config_id}: not attempted (idle e2e {med:.1f} ms)", flush=True)
            R.save(doc)
            configs = [c for c, _ in keep]
        else:
            configs = [Configuration(runtime=rt, internal_w=w, internal_h=h, refiner=ref, load=load)
                       for rt in _csv(args.runtime) for (w, h) in sizes for ref in _onoff(args.refiner)]
        for cfg in configs:
            print(f"[speed] {cfg.config_id} cameras={cameras}", flush=True)
            for e in run_config(doc, cfg, cameras, args.runs, args.frames, args.warmup, load_ctx=load_ctx):
                partial |= e["status"] != "ok"
    return EXIT_PARTIAL if partial else EXIT_OK


def cmd_export(args) -> int:
    from .export import export_one

    doc = _load_doc(args)
    sizes = _sizes(args.sizes) + (VARIANT_16X9 if args.variant16x9 else [])
    sizes.sort(key=lambda s: s[0] * s[1])  # 2048 last: it needs the most RAM and time
    doc.setdefault("exports", [])
    partial = False
    for (w, h) in sizes:
        for ref in _onoff(args.refiner):
            e = export_one(w, h, ref)
            R.upsert(doc["exports"], e, lambda x: (x["internal_w"], x["internal_h"], x.get("refiner", True)))
            R.save(doc)
            partial |= e["onnx_status"] != "ok" or e["engine_status"] != "ok"
    return EXIT_PARTIAL if partial else EXIT_OK


def cmd_clips(args) -> int:
    from .clips import load_manifest, normalise

    doc = _load_doc(args)
    doc.setdefault("clips", [])
    missing = 0
    for clip in load_manifest(args.manifest):
        try:
            e = normalise(clip)
            R.upsert(doc["clips"], e, lambda c: c["clip_id"])
        except FileNotFoundError as ex:
            print(f"  {clip['clip_id']}: MISSING ({ex})", file=sys.stderr)
            missing += 1
        R.save(doc)
    if len(doc["clips"]) < 3:
        raise Precondition("fewer than 3 reference clips available (FR-006)")
    return EXIT_PARTIAL if missing else EXIT_OK


def cmd_quality(args) -> int:
    from .bench_quality import run_quality

    doc = _load_doc(args)
    failures = run_quality(doc, _csv(args.configs) if args.configs else None)
    print("comparison material: outputs/compare/index.html")
    return EXIT_PARTIAL if failures else EXIT_OK


def cmd_review(args) -> int:
    from .review import ReviewError, apply_review

    doc = _load_doc(args)
    try:
        q = apply_review(doc, args.config, args.clip, args.verdict, args.reason)
    except ReviewError as e:
        raise Precondition(str(e))
    R.save(doc)
    print(f"{args.config} x {args.clip}: {q['owner_verdict']}")
    return EXIT_OK


def cmd_report(args) -> int:
    from .report import pending_candidates, render

    doc = _load_doc(args)
    pend = pending_candidates(doc)
    if pend and not args.allow_pending:
        raise Precondition(f"{len(pend)} candidate quality reviews are pending (e.g. {pend[0]}); record them "
                           "with `ckbench review` or use --allow-pending for a draft")
    text, v, env, lat = render(doc, args.allow_pending)
    doc["envelope"], doc["verdict"], doc["latency"] = env, v, lat
    R.save(doc)
    p = R.run_dir(doc["environment"]["run_id"]) / "REPORT.md"
    p.write_text(text, encoding="utf-8")
    print(f"{v['verdict']}{' (DRAFT)' if args.allow_pending else ''}: {p}")
    return EXIT_OK


def cmd_all(args) -> int:
    """env -> clips -> export -> speed idle -> speed nvenc (best) -> quality -> report (draft)."""
    def ns(**kw):
        return argparse.Namespace(run_id=None, **kw)

    rc = cmd_env(ns(new_run=True))
    if rc:
        return rc
    worst = EXIT_OK
    steps = [
        (cmd_clips, dict(manifest=None)),
        (cmd_export, dict(sizes="512,768,1024,2048", refiner="on,off", variant16x9=False)),
        (cmd_speed, dict(runtime="torch,trt", sizes="512,768,1024,2048", refiner="on,off", load="idle",
                         camera="1080p", runs=3, frames=300, warmup=50, variant16x9=True, best=False)),
        (cmd_speed, dict(runtime="torch,trt", sizes="", refiner="on,off", load="nvenc",
                         camera="720p,1080p,2160p", runs=3, frames=300, warmup=50, variant16x9=False, best=True)),
        (cmd_quality, dict(configs=None)),
        (cmd_report, dict(allow_pending=True)),
    ]
    for fn, kw in steps:
        print(f"== ckbench all: {fn.__name__[4:]}", flush=True)
        rc = fn(ns(**kw))
        if rc == EXIT_PRECONDITION:
            return rc
        worst = max(worst, rc)
    return worst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ckbench", description="CorridorKey live-keying feasibility benchmark")
    p.add_argument("--run-id", help="results run id (default: latest)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("env", help="capture environment; verify CUDA, weights, NVENC")
    s.add_argument("--new-run", action="store_true")
    s.set_defaults(fn=cmd_env)

    s = sub.add_parser("clips", help="fetch/normalise reference clips")
    s.add_argument("--manifest", default=None)
    s.set_defaults(fn=cmd_clips)

    s = sub.add_parser("export", help="ONNX export + TensorRT engine build per size")
    s.add_argument("--sizes", default="512,768,1024,2048")
    s.add_argument("--refiner", default="on,off")
    s.add_argument("--variant16x9", action="store_true")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("speed", help="timing + VRAM benchmarks")
    s.add_argument("--runtime", default="torch")
    s.add_argument("--sizes", default="512,768,1024,2048")
    s.add_argument("--refiner", default="on,off")
    s.add_argument("--load", default="idle")
    s.add_argument("--camera", default="1080p")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--frames", type=int, default=300)
    s.add_argument("--warmup", type=int, default=50)
    s.add_argument("--variant16x9", action="store_true")
    s.add_argument("--best", action="store_true",
                   help="with --load nvenc: the idle-ok configurations that can still fit a live mode (T032)")
    s.set_defaults(fn=cmd_speed)

    s = sub.add_parser("quality", help="quality + flicker vs the 2048 reference")
    s.add_argument("--configs", default=None)
    s.set_defaults(fn=cmd_quality)

    s = sub.add_parser("review", help="record the owner's visual verdict")
    s.add_argument("--config", required=True)
    s.add_argument("--clip", required=True)
    s.add_argument("--verdict", required=True, choices=["pass", "fail"])
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("report", help="envelope + verdict + REPORT.md")
    s.add_argument("--allow-pending", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("all", help="env -> clips -> export -> speed -> quality -> report (draft)")
    s.set_defaults(fn=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    from . import vram

    vram.capture_baseline()  # before anything initialises CUDA in this process
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except Precondition as e:
        print(f"ckbench: {e}", file=sys.stderr)
        return EXIT_PRECONDITION


if __name__ == "__main__":
    sys.exit(main())
