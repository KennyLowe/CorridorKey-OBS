"""Test media for the e2e suite (all in the git-ignored outputs/, never committed).

- Reference clips: normalised by the feature 001 code (ckbench.clips.normalise, 1080p30 PNGs in
  outputs/clips/<id>/) and encoded to a looping H.264 file in outputs/e2e/<id>.mp4. The results
  docs are not touched.
- A frame-counter test pattern (SC-005): each frame shows its index as a binary bar code (16
  blocks) on a green screen, so a screenshot can be decoded exactly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from obs_harness import REPO

OUT = REPO / "outputs" / "e2e"
COUNTER_BITS = 16
COUNTER_FRAMES = 900  # 30 s at 30 fps


def _ffmpeg() -> str:
    from ckbench.env import ffmpeg_exe

    return ffmpeg_exe() or "ffmpeg"


def clip_mp4(clip_id: str) -> Path:
    from ckbench.clips import load_manifest, normalise
    from ckbench.frames import CLIPS_OUT

    out = OUT / f"{clip_id}.mp4"
    if out.exists():
        return out
    frames = CLIPS_OUT / clip_id
    if not list(frames.glob("*.png")):
        (clip,) = [c for c in load_manifest() if c["clip_id"] == clip_id]
        normalise(clip)
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-i",
                    str(frames / "%05d.png"), "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv", str(out)], check=True)
    return out


def counter_frame(i: int, w: int = 1920, h: int = 1080) -> np.ndarray:
    """RGB uint8: green screen, a grey 'subject' box (so the key keeps it), and the frame index as
    16 black/white blocks across the middle of the box."""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (40, 180, 60)
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 8, 7 * w // 8
    img[y0:y1, x0:x1] = (128, 128, 128)
    bw = (x1 - x0) // COUNTER_BITS
    for b in range(COUNTER_BITS):
        on = (i >> (COUNTER_BITS - 1 - b)) & 1
        img[y0 + 40:y1 - 40, x0 + b * bw + 8:x0 + (b + 1) * bw - 8] = 255 if on else 0
    return img


def decode_counter(rgba: np.ndarray) -> int | None:
    """Frame index from a (possibly scaled) screenshot of a counter frame; None if unreadable."""
    h, w = rgba.shape[:2]
    y = h // 2
    x0, x1 = w // 8, 7 * w // 8
    bw = (x1 - x0) / COUNTER_BITS
    v = 0
    for b in range(COUNTER_BITS):
        cx = int(x0 + (b + 0.5) * bw)
        px = rgba[y - 2:y + 3, cx - 2:cx + 3, :3].astype(float).mean()
        if 60 < px < 195:
            return None
        v = (v << 1) | (1 if px >= 195 else 0)
    return v


def counter_mp4() -> Path:
    out = OUT / "counter.mp4"
    if out.exists():
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "1920x1080", "-framerate", "30", "-i", "-", "-c:v", "libx264", "-crf", "8",
                          "-g", "30", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-color_range", "tv", str(out)], stdin=subprocess.PIPE)
    for i in range(COUNTER_FRAMES):
        p.stdin.write(counter_frame(i).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed writing the counter pattern")
    return out
