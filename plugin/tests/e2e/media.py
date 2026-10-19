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


# --- Hard keying case with a known matte (hybrid key test) ----------------------------------------

HARD_W, HARD_H = 1920, 1080
HARD_REGIONS = ("subject", "screen", "edge", "green_cloth", "shadow", "strands", "glass")


def _reference_key(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CorridorKey at 2048 with the refiner (feature 001's quality reference) on one frame:
    (alpha [H,W], fg [H,W,3]) in [0,1]."""
    import torch

    from ckbench.pipeline import key_tensor
    from ckbench.runtime_torch import TorchKeyer

    keyer = TorchKeyer((2048, 2048), refiner=True)
    try:
        rgb = torch.from_numpy(rgb_u8).cuda().permute(2, 0, 1).float() / 255.0
        alpha, fg, _ = key_tensor(keyer, rgb)
        return alpha.clamp(0, 1).cpu().numpy(), fg.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    finally:
        keyer.close()


def hard_case() -> tuple[Path, Path]:
    """A static 3 s clip of a real subject over a synthetic hard green screen, and its true matte.

    The subject (alpha + unmixed foreground) is CorridorKey's own best key of a Godiva Close frame.
    It is composited over a screen with uneven light (left half darker), a dark diagonal crease, and
    thin dark strands and a translucent pale panel on the screen; a patch of the dress is recoloured
    green-ish (what a colour key punches holes in). Returns (mp4, npz with 'alpha' and region masks).
    """
    import cv2

    from ckbench.frames import CLIPS_OUT

    mp4, gt = OUT / "hard-case.mp4", OUT / "hard-case-truth.npz"
    if mp4.exists() and gt.exists():
        return mp4, gt
    src = cv2.cvtColor(cv2.imread(str(CLIPS_OUT / "hcw-godiva-close" / "00100.png")), cv2.COLOR_BGR2RGB)
    a, fg = _reference_key(src)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Screen: green, darker to the left, with a dark diagonal crease.
    screen = np.empty((h, w, 3), np.float32)
    screen[:] = (0.16, 0.72, 0.24)
    light = 0.55 + 0.45 * (xx / w)
    crease = np.abs((xx - 0.35 * w) - 0.6 * (yy - 0.5 * h)) < 45
    light = np.where(crease, light * 0.45, light)
    screen *= light[..., None]

    # Green-ish cloth: recolour solid subject pixels inside a box on the dress.
    cloth = np.zeros((h, w), bool)
    cloth[int(0.62 * h):int(0.82 * h), int(0.44 * w):int(0.56 * w)] = True
    cloth &= a > 0.98
    lum = fg[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    fg = np.where(cloth[..., None], np.stack([0.62 * lum, 0.95 * lum, 0.7 * lum], -1).clip(0, 1), fg)  # sage green

    # Strands and a translucent panel on the screen side (left of the subject).
    layer = np.zeros((h, w), np.uint8)
    for i in range(7):
        x0 = int(0.12 * w + i * 22)
        cv2.line(layer, (x0, int(0.15 * h)), (x0 + 60, int(0.55 * h)), 255, 1, cv2.LINE_AA)
    strands_a = (layer.astype(np.float32) / 255.0) * 0.85
    glass = np.zeros((h, w), np.float32)
    glass[int(0.62 * h):int(0.9 * h), int(0.08 * w):int(0.22 * w)] = 0.35
    extra_fg = np.where((strands_a > 0)[..., None], np.float32(0.12), np.float32(0.9))  # dark strands, pale glass
    extra_a = np.maximum(strands_a, glass) * (a < 0.02)

    # Composite: extra layer over the screen, subject over that.
    behind = screen * (1 - extra_a[..., None]) + extra_fg * extra_a[..., None]
    alpha = a + (1 - a) * extra_a
    img = fg * a[..., None] + behind * (1 - a[..., None])
    frame = (img.clip(0, 1) * 255 + 0.5).astype(np.uint8)

    edge = cv2.dilate(((alpha > 0.02) & (alpha < 0.98)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    regions = {
        "subject": (a > 0.98) & ~edge & ~cloth,
        "screen": (alpha < 0.02) & ~edge & ~crease,
        "edge": edge & (a > 0.02),
        "green_cloth": cloth & ~edge,
        "shadow": crease & (alpha < 0.02),
        "strands": strands_a > 0.3,
        "glass": (glass > 0) & (strands_a == 0) & (a < 0.02),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(gt, alpha=alpha.astype(np.float32), **regions)
    np.savez_compressed(OUT / "hard-case-layers.npz", a=a.astype(np.float32), fg=fg.astype(np.float32),
                        behind=behind.astype(np.float32), extra_a=extra_a.astype(np.float32))
    p = subprocess.Popen([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-framerate", "30", "-i", "-", "-c:v", "libx264", "-crf", "8",
                          "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-color_range", "tv", str(mp4)], stdin=subprocess.PIPE)
    for _ in range(90):
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed writing the hard case")
    cv2.imwrite(str(OUT / "hard-case-frame.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return mp4, gt


# --- Moving hard case: the subject slides sideways; a frame counter strip maps screenshots to frames --

MOVE_FRAMES = 90
MOVE_AMP_PX = 95  # x offset = 95 sin(2 pi i / 30): up to ~20 px per frame at 30 fps
STRIP = (1040, 1064, 40, 24)  # y0, y1, x0, block width: 16 blocks along the bottom-left of the screen


def move_dx(i: int) -> int:
    return int(round(MOVE_AMP_PX * np.sin(2 * np.pi * i / 30)))


def _shift(x: np.ndarray, dx: int) -> np.ndarray:
    out = np.zeros_like(x)
    if dx >= 0:
        out[:, dx:] = x[:, :x.shape[1] - dx]
    else:
        out[:, :dx] = x[:, -dx:]
    return out


def moving_frame(i: int, layers) -> tuple[np.ndarray, np.ndarray]:
    """(RGB uint8 frame, true alpha) of moving frame i."""
    dx = move_dx(i)
    a, fg = _shift(layers["a"], dx), _shift(layers["fg"], dx)
    behind, extra_a = layers["behind"], layers["extra_a"] * (a < 0.02)
    alpha = a + (1 - a) * extra_a
    img = fg * a[..., None] + behind * (1 - a[..., None])
    frame = (img.clip(0, 1) * 255 + 0.5).astype(np.uint8)
    y0, y1, x0, bw = STRIP
    for b in range(COUNTER_BITS):
        on = (i >> (COUNTER_BITS - 1 - b)) & 1
        frame[y0:y1, x0 + b * bw:x0 + (b + 1) * bw - 4] = 255 if on else 0
    return frame, alpha


def decode_strip(rgb: np.ndarray) -> int | None:
    y0, y1, x0, bw = STRIP
    y = (y0 + y1) // 2
    v = 0
    for b in range(COUNTER_BITS):
        cx = x0 + b * bw + (bw - 4) // 2
        px = rgb[y - 3:y + 4, cx - 3:cx + 4, :3].astype(float).mean()
        if 60 < px < 195:
            return None
        v = (v << 1) | (1 if px >= 195 else 0)
    return v


def hard_case_moving():
    """(mp4, layers) of the moving hard case (hard_case() must be built first; it writes the layers)."""
    hard_case()
    layers = dict(np.load(OUT / "hard-case-layers.npz"))
    mp4 = OUT / "hard-case-moving.mp4"
    if mp4.exists():
        return mp4, layers
    p = subprocess.Popen([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{HARD_W}x{HARD_H}", "-framerate", "30", "-i", "-", "-c:v", "libx264", "-crf", "8",
                          "-g", "30", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                          "-color_trc", "bt709", "-color_range", "tv", str(mp4)], stdin=subprocess.PIPE)
    for i in range(MOVE_FRAMES):
        p.stdin.write(moving_frame(i, layers)[0].tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed writing the moving hard case")
    return mp4, layers


def no_screen_mp4() -> Path:
    """A static 1080p grey frame with a darker box: no green screen, so calibration must fail."""
    out = OUT / "no-screen.mp4"
    if out.exists():
        return out
    OUT.mkdir(parents=True, exist_ok=True)
    img = np.full((1080, 1920, 3), 120, np.uint8)
    img[300:800, 700:1200] = (90, 80, 70)
    p = subprocess.Popen([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "1920x1080", "-framerate", "30", "-i", "-", "-c:v", "libx264", "-crf", "8",
                          "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for _ in range(60):
        p.stdin.write(img.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed writing the no-screen clip")
    return out
