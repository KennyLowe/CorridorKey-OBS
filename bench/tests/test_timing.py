import numpy as np
import pytest

from ckbench.timing import spread, stats


def test_stats_known_values():
    s = stats(list(range(1, 101)))  # 1..100
    assert s["mean"] == pytest.approx(50.5)
    assert s["median"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(np.percentile(np.arange(1, 101), 95))


def test_stats_single_and_empty():
    assert stats([7.0]) == {"mean": 7.0, "median": 7.0, "p95": 7.0}
    with pytest.raises(ValueError):
        stats([])


def test_spread_pct_and_flag():
    pct, flag = spread([10.0, 10.5, 11.0])
    assert pct == pytest.approx(10.0)
    assert flag is False  # exactly 10% is not > 10%
    pct, flag = spread([10.0, 11.5])
    assert pct == pytest.approx(15.0) and flag is True
    pct, flag = spread([20.0])
    assert pct == 0.0 and flag is False
