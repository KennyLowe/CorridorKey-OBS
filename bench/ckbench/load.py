"""Concurrent 1080p30 NVENC encode as a stand-in for an OBS stream (US4; research D9)."""
from __future__ import annotations

import contextlib
import subprocess
import threading
import time

from . import vram
from .env import ffmpeg_exe

FFMPEG_ARGS = ["-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
               "testsrc2=size=1920x1080:rate=30", "-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "6M",
               "-f", "null", "-"]


class LoadError(RuntimeError):
    pass


@contextlib.contextmanager
def nvenc_load(settle_s: float = 3.0):
    """Start the encode, wait, verify it is alive and NVENC is busy; yield info; kill on exit."""
    exe = ffmpeg_exe()
    if not exe:
        raise LoadError("ffmpeg not found")
    p = subprocess.Popen([exe, *FFMPEG_ARGS], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    enc_samples: list[int] = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            enc_samples.append(vram.encoder_utilisation())
            stop.wait(0.5)

    t = threading.Thread(target=sample, daemon=True)
    try:
        time.sleep(settle_s)
        if p.poll() is not None:
            raise LoadError(f"ffmpeg exited early ({p.returncode}): {p.stderr.read().decode(errors='replace')[-500:]}")
        # A real-time 1080p30 encode keeps NVENC only ~10% busy and NVML's utilisation counter often
        # reads 0 between bursts, so liveness is checked on the encoder session statistics instead.
        sessions, fps, _ = vram.encoder_stats()
        if sessions < 1:
            raise LoadError("NVML reports no active NVENC session with ffmpeg running")
        t.start()
        info = {"command": "ffmpeg " + " ".join(FFMPEG_ARGS), "encoder_sessions": sessions,
                "encoder_avg_fps_at_start": fps, "encoder_util_at_start_pct": vram.encoder_utilisation()}
        yield info
        if p.poll() is not None:
            raise LoadError(f"ffmpeg died during the measurement (exit {p.returncode})")
        info["encoder_util_mean_pct"] = round(sum(enc_samples) / max(1, len(enc_samples)), 1)
        info["encoder_util_min_pct"] = min(enc_samples) if enc_samples else None
    finally:
        stop.set()
        if t.is_alive():
            t.join()
        if p.poll() is None:
            p.kill()
            p.wait()
