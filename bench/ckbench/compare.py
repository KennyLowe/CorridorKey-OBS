"""Side-by-side comparison material for the owner's visual review (US3; FR-007).

Per configuration x clip, written to outputs/compare/<config_id>/:
- <clip>_grey.mp4 and <clip>_checker.mp4: 3-up (reference composite | configuration composite |
  |delta alpha| x4 heatmap), each panel 1280x720, 30 fps, libx264 CRF 16;
- <clip>_sheet.png: a contact sheet of 6 evenly spaced frames, 3-up per row, panels
  at 960x540.
Composites use the straight sRGB foreground over 50% grey or a 0.15/0.55 checkerboard.
These files are derived from the reference footage and are git-ignored (never committed).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .env import ffmpeg_exe
from .results import OUTPUTS_DIR

COMPARE_DIR = OUTPUTS_DIR / "compare"
PANEL_W, PANEL_H = 1280, 720
SHEET_W, SHEET_H = 960, 540
SHEET_FRAMES = 6


def checkerboard(h: int, w: int, size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.where(((yy // size) + (xx // size)) % 2 == 0, 0.15, 0.55).astype(np.float32)
    return np.repeat(c[..., None], 3, axis=2)


def composite(alpha: np.ndarray, fg_chw: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """alpha [H,W], fg [3,H,W] sRGB straight, bg [H,W,3] -> uint8 RGB [H,W,3]."""
    a = alpha[..., None]
    out = fg_chw.transpose(1, 2, 0) * a + bg * (1.0 - a)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def heatmap(alpha: np.ndarray, ref_alpha: np.ndarray) -> np.ndarray:
    d = np.clip(np.abs(alpha - ref_alpha) * 4.0, 0, 1)
    bgr = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _rs(x: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)


def panel_inputs(alpha, fg, ref_alpha, ref_fg, w, h):
    """Downscale once per frame to panel size: (a, f_hwc, ra, rf_hwc, heat)."""
    a, ra = _rs(alpha, w, h), _rs(ref_alpha, w, h)
    f = _rs(np.ascontiguousarray(fg.transpose(1, 2, 0)), w, h)
    rf = _rs(np.ascontiguousarray(ref_fg.transpose(1, 2, 0)), w, h)
    return a, f, ra, rf, heatmap(a, ra)


def _comp_hwc(a, f_hwc, bg):
    """bg + a * (fg - bg), via OpenCV (multithreaded); saturating conversion to uint8."""
    a3 = cv2.merge([a, a, a])
    out = cv2.add(bg, cv2.multiply(a3, cv2.subtract(f_hwc, bg)))
    return cv2.convertScaleAbs(out, alpha=255.0)


def three_up_from(pi, bg, cfg_label: str) -> np.ndarray:
    a, f, ra, rf, heat = pi
    panels = [_label(_comp_hwc(ra, rf, bg), "reference 2048"), _label(_comp_hwc(a, f, bg), cfg_label),
              _label(heat.copy(), "|d alpha| x4")]
    return np.concatenate(panels, axis=1)


def three_up(alpha, fg, ref_alpha, ref_fg, bg, w, h, cfg_label: str) -> np.ndarray:
    """Downscale to panel size first, then composite (bg must already be [h,w,3])."""
    return three_up_from(panel_inputs(alpha, fg, ref_alpha, ref_fg, w, h), bg, cfg_label)


class _Mp4:
    def __init__(self, path: Path, w: int, h: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


class CompareWriter:
    """Streams one configuration x clip into the two videos and the contact sheet."""

    def __init__(self, config_id: str, clip_id: str, n_frames: int, h: int = 1080, w: int = 1920):
        d = COMPARE_DIR / config_id
        self.cfg_label = config_id
        self.bgs = {"grey": np.full((PANEL_H, PANEL_W, 3), 0.5, np.float32),
                    "checker": checkerboard(PANEL_H, PANEL_W, 48)}
        self.sheet_bg = checkerboard(SHEET_H, SHEET_W, 36)
        self.vids = {k: _Mp4(d / f"{clip_id}_{k}.mp4", 3 * PANEL_W, PANEL_H) for k in self.bgs}
        self.sheet_idx = set(np.linspace(0, n_frames - 1, SHEET_FRAMES).round().astype(int).tolist())
        self.sheet_rows: list[np.ndarray] = []
        self.sheet_path = d / f"{clip_id}_sheet.png"
        self.t = 0

    def add(self, alpha, fg, ref_alpha, ref_fg) -> None:
        pi = panel_inputs(alpha, fg, ref_alpha, ref_fg, PANEL_W, PANEL_H)
        for k, bg in self.bgs.items():
            self.vids[k].write(three_up_from(pi, bg, self.cfg_label))
        if self.t in self.sheet_idx:
            row = three_up(alpha, fg, ref_alpha, ref_fg, self.sheet_bg, SHEET_W, SHEET_H,
                           f"{self.cfg_label}  f{self.t}")
            self.sheet_rows.append(row)
        self.t += 1

    def close(self) -> None:
        for v in self.vids.values():
            v.close()
        if self.sheet_rows:
            sheet = np.concatenate(self.sheet_rows, axis=0)
            cv2.imwrite(str(self.sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def write_index(quality: list[dict]) -> Path:
    """outputs/compare/index.html: one row per config x clip with metrics, flags and links."""
    rows = []
    for q in sorted(quality, key=lambda q: (q["clip_id"], q["config_id"])):
        c, k = q["config_id"], q["clip_id"]
        fl = ", ".join(q["flags"]) or "-"
        rows.append(
            f"<tr><td>{k}</td><td>{c}</td><td>{q['edge_alpha_mae']:.4f}</td><td>{q['edge_fg_mae']:.4f}</td>"
            f"<td>{q['flicker_excess']:.4f}</td><td class='{'flag' if q['flags'] else ''}'>{fl}</td>"
            f"<td>{q['owner_verdict']}</td><td><a href='{c}/{k}_grey.mp4'>grey</a> | "
            f"<a href='{c}/{k}_checker.mp4'>checker</a> | <a href='{c}/{k}_sheet.png'>sheet</a></td></tr>")
    html = ("<!doctype html><meta charset='utf-8'><title>ckbench comparison</title>"
            "<style>body{font-family:sans-serif}td,th{padding:3px 8px;border-bottom:1px solid #ccc}"
            ".flag{color:#b00;font-weight:bold}</style><h1>ckbench quality review</h1>"
            "<p>Flag thresholds: edge alpha MAE &gt; 0.030, edge FG MAE &gt; 0.040, flicker excess &gt; 0.010. "
            "Record verdicts with <code>uv run ckbench review --config ID --clip ID --verdict pass|fail "
            "[--reason TEXT]</code>.</p><table><tr><th>clip</th><th>config</th><th>edge &alpha; MAE</th>"
            "<th>edge FG MAE</th><th>flicker excess</th><th>flags</th><th>verdict</th><th>material</th></tr>"
            + "".join(rows) + "</table>")
    p = COMPARE_DIR / "index.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
