import numpy as np
import pytest

from ckbench.quality import (
    EDGE_ALPHA_MAX, EDGE_FG_MAX, FLICKER_MAX, ClipMetrics, edge_alpha_mae, edge_band, edge_fg_mae, flags,
    flicker_excess,
)


def test_thresholds_declared():
    assert (EDGE_ALPHA_MAX, EDGE_FG_MAX, FLICKER_MAX) == (0.030, 0.040, 0.010)


def test_edge_band_dilates_by_8px():
    ref = np.zeros((64, 64), np.float32)
    ref[:, 32:] = 1.0
    ref[:, 32] = 0.5  # a one-pixel soft column
    band = edge_band(ref)
    assert band[:, 32].all()
    assert band[:, 24].all() and band[:, 40].all()  # 8 px either side
    assert not band[:, 23].any() and not band[:, 41].any()


def test_edge_band_empty_for_hard_matte():
    ref = np.zeros((16, 16), np.float32)
    ref[:, 8:] = 1.0
    assert not edge_band(ref).any()


def test_edge_alpha_mae_inside_band_only():
    ref = np.full((10, 10), 0.5, np.float32)
    band = np.zeros((10, 10), bool)
    band[:5] = True
    a = ref.copy()
    a[:5] += 0.1   # error inside band
    a[5:] += 0.9   # huge error outside band is ignored
    assert edge_alpha_mae(a, ref, band) == pytest.approx(0.1, abs=1e-6)
    assert edge_alpha_mae(a, ref, np.zeros_like(band)) == 0.0


def test_edge_fg_mae_requires_ref_alpha_above_0_1():
    ref_a = np.full((4, 4), 0.5, np.float32)
    ref_a[:, :2] = 0.05  # too transparent: excluded
    band = np.ones((4, 4), bool)
    ref_fg = np.zeros((3, 4, 4), np.float32)
    fg = ref_fg.copy()
    fg[:, :, :2] = 1.0   # excluded columns
    fg[:, :, 2:] = 0.2
    assert edge_fg_mae(fg, ref_fg, ref_a, band) == pytest.approx(0.2, abs=1e-6)


def test_flicker_excess():
    band = np.ones((2, 2), bool)
    ref_seq = [np.full((2, 2), 0.5, np.float32)] * 3           # perfectly stable reference
    cfg_seq = [np.full((2, 2), v, np.float32) for v in (0.5, 0.52, 0.5)]
    assert flicker_excess(cfg_seq, ref_seq, [band] * 3) == pytest.approx(0.02, abs=1e-6)
    assert flicker_excess(ref_seq, ref_seq, [band] * 3) == pytest.approx(0.0)


def test_flags():
    assert flags(0.0, 0.0, 0.0) == []
    assert flags(0.031, 0.0, 0.0) == ["edge_alpha"]
    assert flags(0.030, 0.040, 0.010) == []  # thresholds are exclusive
    assert flags(0.1, 0.05, 0.02) == ["edge_alpha", "edge_fg", "flicker"]


def test_clip_metrics_accumulator_matches_functions():
    rng = np.random.default_rng(0)
    ref_a = [np.clip(rng.random((32, 32)).astype(np.float32), 0, 1) for _ in range(3)]
    cfg_a = [np.clip(r + 0.01, 0, 1) for r in ref_a]
    ref_fg = [rng.random((3, 32, 32)).astype(np.float32) for _ in range(3)]
    cfg_fg = [f + 0.02 for f in ref_fg]
    m = ClipMetrics()
    for ca, cf, ra, rf in zip(cfg_a, cfg_fg, ref_a, ref_fg):
        m.update(ca, cf, ra, rf)
    out = m.result()
    bands = [edge_band(r) for r in ref_a]
    # pixel-weighted over all frames
    num = sum(np.abs(c - r)[b].sum() for c, r, b in zip(cfg_a, ref_a, bands))
    den = sum(b.sum() for b in bands)
    assert out["edge_alpha_mae"] == pytest.approx(num / den, rel=1e-5)
    assert out["edge_fg_mae"] == pytest.approx(0.02, abs=1e-4)
    assert out["flicker_excess"] == pytest.approx(flicker_excess(cfg_a, ref_a, bands), abs=1e-6)
