"""`ckbench prepare`: weights -> ONNX -> TensorRT engine for one quality setting, cached per GPU,
driver, TensorRT version and weights (feature 002, contracts/prepare-cli.md).

The OBS plugin runs this as a child process and reads JSON lines from stdout:

    {"event":"progress","stage":"verify|onnx|engine|done","pct":0-100,"msg":"..."}
    {"event":"result","ok":true,"key":"<sha256>","engine_path":"...","trt_version":"10.16.1.11","build_s":312.4}
    {"event":"result","ok":false,"error":"weights_invalid|oom|export_failed|build_failed|trt_version_mismatch|io","msg":"..."}

Exactly one ``result`` line, always last; exit code 0 only when ok. The ONNX export and the engine
build each run in their own subprocess (``python -m ckbench.export``, as in the benchmark) with
their output going to log files in the cache folder, so stdout carries only the protocol.

The cache key is sha256 over the canonical JSON of the key components (sorted keys, compact), the
same string the plugin's C++ ``cache_key()`` hashes (shared fixture:
plugin/tests/data/manifest_fixture.json). ``manifest.json`` is replaced atomically (temp file +
``os.replace``) and only after the engine is complete, so a killed run never leaves a half entry.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

TRT_VERSION = "10.16.1.11"
KEY_FIELDS = ("gpu_uuid", "driver_version", "trt_version", "weights_sha256", "internal_w", "internal_h", "refiner")
# Present in the CorridorKey v1.0 checkpoint (GreenFormer: Hiera encoder, decoders, CNN refiner).
REQUIRED_KEYS = ("encoder.model.pos_embed", "alpha_decoder.bn.weight", "refiner.stem.0.weight")
DEFAULT_CACHE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "corridorkey-obs" / "engines"

# Progress bands per stage (typical times on an RTX 3060 at 512-1024: ONNX 8-15 s, engine 2-4 min).
BANDS = {"verify": (0, 5), "onnx": (5, 20), "engine": (20, 99)}
EXPECTED_S = {"onnx": 15.0, "engine": 180.0}


class Identity(NamedTuple):
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    trt_version: str


class PrepareError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


# ----------------------------------------------------------------------------- key + manifest

def canonical(comp: dict) -> str:
    return json.dumps({k: comp[k] for k in KEY_FIELDS}, sort_keys=True, separators=(",", ":"))


def cache_key(comp: dict) -> str:
    return hashlib.sha256(canonical(comp).encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict:
    """Missing or corrupt -> empty (a corrupt one is rewritten on the next successful build)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
            return doc
    except (OSError, ValueError):
        pass
    return {"schema_version": "1.0", "entries": []}


def write_manifest(path: Path, doc: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- environment

def identity() -> Identity:
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def s(v):
        return v.decode() if isinstance(v, bytes) else v

    try:
        import tensorrt

        trt_version = tensorrt.__version__
    except Exception as e:  # reported as trt_version_mismatch
        trt_version = f"unavailable: {e}"
    return Identity(gpu_name=s(pynvml.nvmlDeviceGetName(h)), gpu_uuid=s(pynvml.nvmlDeviceGetUUID(h)),
                    driver_version=s(pynvml.nvmlSystemGetDriverVersion()), trt_version=trt_version)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weights(path) -> str:
    """sha256 of a loadable CorridorKey checkpoint, else PrepareError(weights_invalid)."""
    p = Path(path)
    if not p.is_file():
        raise PrepareError("weights_invalid", f"weights file not found: {p}")
    try:
        from safetensors import safe_open

        with safe_open(str(p), framework="numpy") as f:
            keys = set(f.keys())
    except Exception as e:
        raise PrepareError("weights_invalid", f"not a safetensors file: {type(e).__name__}: {e}"[:400])
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise PrepareError("weights_invalid",
                           f"not a CorridorKey checkpoint (missing {', '.join(missing)}); download "
                           "CorridorKey_v1.0.safetensors from the official CorridorKey source")
    return sha256_file(p)


# ----------------------------------------------------------------------------- real builders

class SubprocessBuilders:
    """Runs the benchmark's export workers (ckbench.export) with the cache paths."""

    @staticmethod
    def _run(step: str, w: int, h: int, refiner: bool, a: str, b: str, log_path: Path, beat) -> dict:
        out_json = log_path.with_suffix(".json")
        out_json.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "ckbench.export", step, str(w), str(h), str(int(refiner)), str(out_json), a, b]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, cwd=str(Path(__file__).parents[1]))
            while proc.poll() is None:
                beat()
                time.sleep(2.0)
        if out_json.exists():
            return json.loads(out_json.read_text())
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        key = "onnx" if step == "onnx" else "engine"
        return {f"{key}_status": "failed", f"{key}_reason": f"worker exit code {proc.returncode}: {tail}"}

    def onnx(self, weights, w, h, refiner, onnx_path, log_path, beat):
        return self._run("onnx", w, h, refiner, str(weights), str(onnx_path), Path(log_path), beat)

    def engine(self, w, h, refiner, onnx_path, engine_path, log_path, beat):
        return self._run("engine", w, h, refiner, str(onnx_path), str(engine_path), Path(log_path), beat)


def _is_oom(reason: str) -> bool:
    r = reason.lower()
    return any(s in r for s in ("out of memory", "memoryerror", "bad_alloc", "cudaerrormemoryallocation",
                                "insufficient memory", "outofmemory"))


# ----------------------------------------------------------------------------- prepare

def _prune(cache: Path, man: dict, ident: Identity) -> dict:
    keep = []
    for e in man["entries"]:
        same_env = (e.get("gpu_uuid") == ident.gpu_uuid and e.get("driver_version") == ident.driver_version
                    and e.get("trt_version") == ident.trt_version)
        if same_env and Path(e.get("engine_path", "")).exists():
            keep.append(e)
            continue
        d = Path(e.get("engine_path", "")).parent
        if d.exists() and d.parent == cache:
            shutil.rmtree(d, ignore_errors=True)
    return {**man, "entries": keep}


def prepare(weights: str, w: int, h: int, refiner: bool, cache_dir: str, emit: Callable[[dict], None],
            prune: bool = False, identity: Callable[[], Identity] = identity, builders=None) -> int:
    """Returns the exit code; emits progress dicts and exactly one final result dict."""
    builders = builders or SubprocessBuilders()
    t_start = time.perf_counter()
    last_pct = [0]

    def progress(stage: str, pct: float, msg: str = "") -> None:
        pct = int(max(last_pct[0], min(100, pct)))
        last_pct[0] = pct
        emit({"event": "progress", "stage": stage, "pct": pct, "msg": msg})

    def fail(code: str, msg: str) -> int:
        emit({"event": "result", "ok": False, "error": code, "msg": msg})
        return 1

    def heartbeat(stage: str):
        lo, hi = BANDS[stage]
        t0 = time.perf_counter()

        def beat():
            # creeps towards the end of the band without reaching it
            frac = 1.0 - 0.5 ** ((time.perf_counter() - t0) / EXPECTED_S[stage])
            progress(stage, lo + (hi - lo) * frac * 0.95, f"{stage}: {time.perf_counter() - t0:.0f} s")
        return beat

    try:
        progress("verify", 0, "checking weights")
        ident = identity()
        if ident.trt_version != TRT_VERSION:
            return fail("trt_version_mismatch",
                        f"TensorRT {TRT_VERSION} is required (found {ident.trt_version}); run bench/setup.ps1")
        wsha = verify_weights(weights)
        comp = {"gpu_uuid": ident.gpu_uuid, "driver_version": ident.driver_version, "trt_version": ident.trt_version,
                "weights_sha256": wsha, "internal_w": int(w), "internal_h": int(h), "refiner": bool(refiner)}
        key = cache_key(comp)
        cache = Path(cache_dir)
        try:
            cache.mkdir(parents=True, exist_ok=True)
            mpath = cache / "manifest.json"
            man = read_manifest(mpath)
            if prune:
                man = _prune(cache, man, ident)
                write_manifest(mpath, man)
        except OSError as e:
            return fail("io", f"cache folder {cache} is not writable: {e}")
        progress("verify", BANDS["verify"][1], f"weights sha256 {wsha[:12]}")

        for e in man["entries"]:
            if e.get("key") == key and Path(e.get("engine_path", "")).exists():
                progress("done", 100, "cached")
                emit({"event": "result", "ok": True, "key": key, "engine_path": e["engine_path"],
                      "trt_version": ident.trt_version, "build_s": e.get("build_s", 0.0)})
                return 0

        d = cache / key[:16]
        stem = f"ck_{w}x{h}_ref{int(refiner)}"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return fail("io", f"cannot create {d}: {e}")
        onnx_p, engine_p = d / f"{stem}.onnx", d / f"{stem}.engine"

        progress("onnx", BANDS["onnx"][0], "exporting ONNX")
        r = builders.onnx(str(weights), w, h, refiner, str(onnx_p), str(d / f"{stem}.onnx.log"), heartbeat("onnx"))
        if r.get("onnx_status") != "ok":
            reason = str(r.get("onnx_reason", "unknown"))
            return fail("oom" if _is_oom(reason) else "export_failed", reason[-600:])

        progress("engine", BANDS["engine"][0], "building TensorRT engine (minutes)")
        t_build = time.perf_counter()
        r = builders.engine(w, h, refiner, str(onnx_p), str(engine_p), str(d / f"{stem}.engine.log"),
                            heartbeat("engine"))
        if r.get("engine_status") != "ok" or not engine_p.exists():
            reason = str(r.get("engine_reason", "engine file missing"))
            return fail("oom" if _is_oom(reason) else "build_failed", reason[-600:])
        build_s = round(time.perf_counter() - t_build, 1)

        entry = {"key": key, **comp, "engine_path": str(engine_p), "onnx_path": str(onnx_p),
                 "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 "build_s": build_s, "gpu_name": ident.gpu_name}
        try:
            man = read_manifest(mpath)  # re-read: another prepare may have finished meanwhile
            man["entries"] = [e for e in man["entries"] if e.get("key") != key] + [entry]
            man["schema_version"] = "1.0"
            write_manifest(mpath, man)
        except OSError as e:
            return fail("io", f"cannot write {mpath}: {e}")
        progress("done", 100, f"engine ready in {time.perf_counter() - t_start:.0f} s")
        emit({"event": "result", "ok": True, "key": key, "engine_path": str(engine_p),
              "trt_version": ident.trt_version, "build_s": build_s})
        return 0
    except PrepareError as e:
        return fail(e.code, e.msg)
    except OSError as e:
        return fail("io", f"{type(e).__name__}: {e}")
    except Exception as e:  # never leave the plugin without a result line
        return fail("build_failed", f"{type(e).__name__}: {e}"[:600])


def json_emitter(stream=None) -> Callable[[dict], None]:
    lock = threading.Lock()
    stream = stream or sys.stdout

    def emit(ev: dict) -> None:
        with lock:
            stream.write(json.dumps(ev) + "\n")
            stream.flush()
    return emit


def text_emitter(ev: dict) -> None:
    if ev["event"] == "progress":
        print(f"[prepare] {ev['stage']:<6} {ev['pct']:3d}%  {ev['msg']}", flush=True)
    elif ev["ok"]:
        print(f"[prepare] ok: {ev['engine_path']} (key {ev['key'][:12]}, build {ev['build_s']} s)", flush=True)
    else:
        print(f"[prepare] FAILED {ev['error']}: {ev['msg']}", file=sys.stderr, flush=True)


def text_emitter_fetch(ev: dict) -> None:
    if ev["event"] == "progress":
        print(f"[fetch] {ev['stage']:<8} {ev['pct']:3d}%  {ev.get('bytes', 0) / 2**20:.0f} of "
              f"{ev.get('total', 0) / 2**20:.0f} MiB", flush=True)
    elif ev["ok"]:
        print(f"[fetch] ok: {ev['weights_path']} (sha256 {ev['sha256'][:12]})", flush=True)
    else:
        print(f"[fetch] FAILED {ev['error']}: {ev['msg']}", file=sys.stderr, flush=True)
