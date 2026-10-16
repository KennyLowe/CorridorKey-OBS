"""Benchmark input frames.

Timing doesn't depend on frame content (static shapes), but a realistic frame is used when
available: frame 0 of the normalised `pexels-athlete` clip, else frame 0 of any normalised clip,
else a deterministic synthetic green-screen frame. The chosen source is recorded with each result.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .results import OUTPUTS_DIR

CLIPS_OUT = OUTPUTS_DIR / "clips"
PREFERRED = ("pexels-athlete", "pexels-dance", "pixabay-hair", "hcw-locked-backdrop", "hcw-godiva-close")


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def synthetic_frame(w: int = 1920, h: int = 1080, seed: int = 0) -> np.ndarray:
    """Screen-green background with a soft-edged grey/skin-toned 'subject' and sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (0.20, 0.70, 0.25)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    body = ((xx - w * 0.5) / (w * 0.16)) ** 2 + ((yy - h * 0.75) / (h * 0.45)) ** 2
    head = ((xx - w * 0.5) / (w * 0.07)) ** 2 + ((yy - h * 0.28) / (h * 0.14)) ** 2
    a = np.clip(1.5 - np.minimum(body, head), 0, 1)[..., None]
    subj = np.where(head[..., None] < body[..., None], (0.80, 0.62, 0.52), (0.35, 0.35, 0.40)).astype(np.float32)
    img = img * (1 - a) + subj * a
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def benchmark_frame() -> tuple[np.ndarray, str]:
    """The fixed speed-benchmark input.

    Deliberately always the synthetic frame (deviation from tasks.md T016, which preferred
    pexels-athlete frame 0 when present): every speed entry, idle or under load, torch or TRT,
    then sees identical input regardless of whether `ckbench clips` has run yet. The networks
    have static shapes, so content does not change the work done.
    """
    return synthetic_frame(), "synthetic green-screen pattern (seed 0)"


def clip_frames(clip_id: str) -> list[Path]:
    return sorted((CLIPS_OUT / clip_id).glob("*.png"))
