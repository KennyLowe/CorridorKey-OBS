"""Quality and flicker metrics against the 2048 reference (research D11; FR-006, FR-007a).

All values are on a 0-1 scale and pixel-weighted across every frame of a clip.

- edge band: reference alpha in (0.02, 0.98), dilated by 8 px (square kernel), at 1080p.
- edge_alpha_mae: mean |alpha - ref_alpha| inside the band.
- edge_fg_mae: mean |fg - ref_fg| (all RGB channels) inside the band where ref_alpha > 0.1.
- flicker_excess: mean |a_t - a_(t-1)| for the configuration minus the same for the reference,
  over the union of the two frames' bands.

Flag thresholds were declared in the plan before any quality run and are constants on purpose.
"""
from __future__ import annotations

import cv2
import numpy as np

EDGE_ALPHA_MAX = 0.030
EDGE_FG_MAX = 0.040
FLICKER_MAX = 0.010
BAND_LO, BAND_HI, BAND_DILATE_PX, FG_MIN_ALPHA = 0.02, 0.98, 8, 0.1
_KERNEL = np.ones((2 * BAND_DILATE_PX + 1, 2 * BAND_DILATE_PX + 1), np.uint8)


def edge_band(ref_alpha: np.ndarray) -> np.ndarray:
    soft = ((ref_alpha > BAND_LO) & (ref_alpha < BAND_HI)).astype(np.uint8)
    return cv2.dilate(soft, _KERNEL).astype(bool)


def edge_alpha_mae(alpha: np.ndarray, ref_alpha: np.ndarray, band: np.ndarray) -> float:
    n = int(band.sum())
    return float(np.abs(alpha - ref_alpha)[band].sum() / n) if n else 0.0


def _fg_mask(ref_alpha: np.ndarray, band: np.ndarray) -> np.ndarray:
    return band & (ref_alpha > FG_MIN_ALPHA)


def edge_fg_mae(fg: np.ndarray, ref_fg: np.ndarray, ref_alpha: np.ndarray, band: np.ndarray) -> float:
    """fg/ref_fg: [3,H,W]."""
    m = _fg_mask(ref_alpha, band)
    n = int(m.sum())
    return float(np.abs(fg - ref_fg)[:, m].sum() / (3 * n)) if n else 0.0


def _temporal(seq, bands) -> tuple[float, int]:
    s, n = 0.0, 0
    for t in range(1, len(seq)):
        b = bands[t] | bands[t - 1]
        s += float(np.abs(seq[t] - seq[t - 1])[b].sum())
        n += int(b.sum())
    return s, n


def flicker_excess(cfg_seq, ref_seq, bands) -> float:
    cs, n = _temporal(cfg_seq, bands)
    rs, _ = _temporal(ref_seq, bands)
    return (cs - rs) / n if n else 0.0


def flags(e_alpha: float, e_fg: float, flick: float) -> list[str]:
    out = []
    if e_alpha > EDGE_ALPHA_MAX:
        out.append("edge_alpha")
    if e_fg > EDGE_FG_MAX:
        out.append("edge_fg")
    if flick > FLICKER_MAX:
        out.append("flicker")
    return out


class ClipMetrics:
    """Streaming version of the metrics above (frames arrive one at a time)."""

    def __init__(self) -> None:
        self.ea_sum = self.ea_n = 0.0
        self.ef_sum = self.ef_n = 0.0
        self.fc_sum = self.fr_sum = self.f_n = 0.0
        self._prev = None  # (cfg_alpha, ref_alpha, band)
        self.frames = 0

    def update(self, alpha, fg, ref_alpha, ref_fg) -> None:
        band = edge_band(ref_alpha)
        self.ea_sum += float(np.abs(alpha - ref_alpha)[band].sum())
        self.ea_n += int(band.sum())
        m = _fg_mask(ref_alpha, band)
        self.ef_sum += float(np.abs(fg - ref_fg)[:, m].sum())
        self.ef_n += 3 * int(m.sum())
        if self._prev is not None:
            pa, pr, pb = self._prev
            b = band | pb
            self.fc_sum += float(np.abs(alpha - pa)[b].sum())
            self.fr_sum += float(np.abs(ref_alpha - pr)[b].sum())
            self.f_n += int(b.sum())
        self._prev = (alpha, ref_alpha, band)
        self.frames += 1

    def result(self) -> dict:
        ea = self.ea_sum / self.ea_n if self.ea_n else 0.0
        ef = self.ef_sum / self.ef_n if self.ef_n else 0.0
        fx = (self.fc_sum - self.fr_sum) / self.f_n if self.f_n else 0.0
        return {"edge_alpha_mae": ea, "edge_fg_mae": ef, "flicker_excess": fx, "flags": flags(ea, ef, fx),
                "edge_band_px_per_frame": round(self.ea_n / self.frames) if self.frames else 0}
