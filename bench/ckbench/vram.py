"""NVML helpers: background VRAM sampler, clock/temperature snapshots (research D7).

On Windows (WDDM) NVML does not report per-process memory (usedGpuMemory is None), so the
sampler records device-wide used memory. ``peak_mib()`` returns the peak minus a baseline taken
*before this process initialised CUDA* (``capture_baseline()``), i.e. the memory this process
added: CUDA context + allocations. Other processes changing their usage add noise; the
method is recorded alongside each result as ``vram_method``.
"""
from __future__ import annotations

import os
import threading

import pynvml

_inited = False
_baseline_mib: float | None = None


def _h(index: int = 0):
    global _inited
    if not _inited:
        pynvml.nvmlInit()
        _inited = True
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def device_used_mib() -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(_h()).used / 2**20


def capture_baseline() -> float:
    """Call before any CUDA initialisation in this process."""
    global _baseline_mib
    if _baseline_mib is None:
        _baseline_mib = device_used_mib()
    return _baseline_mib


def process_used_mib() -> float | None:
    pid = os.getpid()
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_h()):
            if p.pid == pid and p.usedGpuMemory is not None:
                return p.usedGpuMemory / 2**20
    except pynvml.NVMLError:
        pass
    return None


def snapshot() -> dict:
    h = _h()
    return {
        "sm_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
        "temp_c": int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
    }


def encoder_utilisation() -> int:
    return int(pynvml.nvmlDeviceGetEncoderUtilization(_h())[0])


def gpu_utilisation() -> int:
    return int(pynvml.nvmlDeviceGetUtilizationRates(_h()).gpu)


class VramSampler:
    """Samples memory every ``interval_s`` in a background thread while active."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_device_mib = 0.0
        self.peak_process_mib: float | None = None
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _sample(self) -> None:
        self.peak_device_mib = max(self.peak_device_mib, device_used_mib())
        p = process_used_mib()
        if p is not None:
            self.peak_process_mib = max(self.peak_process_mib or 0.0, p)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramSampler":
        _h()
        self._sample()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join()
        self._sample()

    def peak_mib(self) -> tuple[float, str]:
        if self.peak_process_mib is not None:
            return self.peak_process_mib, "nvml_process"
        base = _baseline_mib if _baseline_mib is not None else 0.0
        return self.peak_device_mib - base, "nvml_device_minus_pre_cuda_baseline"
