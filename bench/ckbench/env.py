"""Environment capture (data-model.md "Environment"; FR-009, FR-013)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .results import BENCH_DIR

VENDOR_DIR = BENCH_DIR / "vendor" / "CorridorKey"
WEIGHTS_FILE = "CorridorKey_v1.0.safetensors"
WEIGHTS_PATH = VENDOR_DIR / "CorridorKeyModule" / "checkpoints" / WEIGHTS_FILE

PINNED_PACKAGES = (
    "torch", "torchvision", "timm", "safetensors", "onnx", "tensorrt-cu12", "tensorrt-cu12-libs",
    "tensorrt-cu12-bindings", "nvidia-ml-py", "numpy", "opencv-python-headless", "jsonschema",
    "psutil", "pytest",
)


def new_run_id() -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    h = hashlib.sha1(f"{now.isoformat()}-{platform.node()}-{os.getpid()}".encode()).hexdigest()[:6]
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + h


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    return str(links) if links.exists() else None


def ffmpeg_version() -> str:
    exe = ffmpeg_exe()
    if not exe:
        return "not found"
    out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0].strip() if out else "unknown"


def nvenc_ok() -> tuple[bool, str]:
    exe = ffmpeg_exe()
    if not exe:
        return False, "ffmpeg not found"
    r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=1920x1080:rate=30", "-t", "1", "-c:v", "h264_nvenc",
                        "-preset", "p5", "-b:v", "6M", "-f", "null", "-"], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[-400:]


def corridorkey_sha() -> str:
    r = subprocess.run(["git", "-C", str(VENDOR_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CorridorKey checkout not found at {VENDOR_DIR}; run bench/setup.ps1")
    return r.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def packages() -> dict:
    out = {}
    for name in PINNED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def capture(run_id: str | None = None) -> dict:
    import psutil
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    drv = pynvml.nvmlSystemGetDriverVersion()
    drv = drv.decode() if isinstance(drv, bytes) else drv

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f"weights not found at {WEIGHTS_PATH}; run bench/setup.ps1")

    try:
        import tensorrt
        trt_version = tensorrt.__version__
    except Exception as e:  # recorded, not fatal
        trt_version = f"unavailable: {e}"

    import torch

    return {
        "run_id": run_id or new_run_id(),
        "gpu_name": name,
        "gpu_driver": drv,
        "gpu_vram_mib": int(pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20),
        "gpu_max_sm_clock_mhz": int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
        "gpu_power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "python": sys.version.split()[0],
        "packages": packages(),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "corridorkey_sha": corridorkey_sha(),
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": sha256_file(WEIGHTS_PATH),
        "ffmpeg_version": ffmpeg_version(),
        "tensorrt_version": trt_version,
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
