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


def _not_implemented(args) -> int:
    raise Precondition(f"`{args.cmd}` not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ckbench", description="CorridorKey live-keying feasibility benchmark")
    p.add_argument("--run-id", help="results run id (default: latest)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("env", help="capture environment; verify CUDA, weights, NVENC")
    s.add_argument("--new-run", action="store_true")
    s.set_defaults(fn=cmd_env)

    s = sub.add_parser("clips", help="fetch/normalise reference clips")
    s.add_argument("--manifest", default=None)
    s.set_defaults(fn=_not_implemented)

    s = sub.add_parser("export", help="ONNX export + TensorRT engine build per size")
    s.add_argument("--sizes", default="512,768,1024,2048")
    s.add_argument("--refiner", default="on,off")
    s.add_argument("--variant16x9", action="store_true")
    s.set_defaults(fn=_not_implemented)

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
    s.set_defaults(fn=_not_implemented)

    s = sub.add_parser("quality", help="quality + flicker vs the 2048 reference")
    s.add_argument("--configs", default=None)
    s.set_defaults(fn=_not_implemented)

    s = sub.add_parser("review", help="record the owner's visual verdict")
    s.add_argument("--config", required=True)
    s.add_argument("--clip", required=True)
    s.add_argument("--verdict", required=True, choices=["pass", "fail"])
    s.add_argument("--reason", default=None)
    s.set_defaults(fn=_not_implemented)

    s = sub.add_parser("report", help="envelope + verdict + REPORT.md")
    s.add_argument("--allow-pending", action="store_true")
    s.set_defaults(fn=_not_implemented)

    s = sub.add_parser("all", help="env -> clips -> export -> speed -> quality -> report (draft)")
    s.set_defaults(fn=_not_implemented)
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
