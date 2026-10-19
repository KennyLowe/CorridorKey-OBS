import pytest

from ckbench.review import ReviewError, apply_review


def _doc():
    return {"quality": [
        {"config_id": "a", "clip_id": "c1", "edge_alpha_mae": 0.05, "edge_fg_mae": 0.0, "flicker_excess": 0.0,
         "flags": ["edge_alpha"], "owner_verdict": "pending"},
        {"config_id": "b", "clip_id": "c1", "edge_alpha_mae": 0.0, "edge_fg_mae": 0.0, "flicker_excess": 0.0,
         "flags": [], "owner_verdict": "pending"},
    ]}


def test_pass_unflagged_without_reason():
    d = _doc()
    apply_review(d, "b", "c1", "pass", None)
    assert d["quality"][1]["owner_verdict"] == "pass"


def test_pass_flagged_without_reason_rejected():
    d = _doc()
    with pytest.raises(ReviewError):
        apply_review(d, "a", "c1", "pass", None)
    with pytest.raises(ReviewError):
        apply_review(d, "a", "c1", "pass", "   ")
    assert d["quality"][0]["owner_verdict"] == "pending"


def test_pass_flagged_with_reason():
    d = _doc()
    apply_review(d, "a", "c1", "pass", "error confined to motion-blurred hand")
    assert d["quality"][0]["owner_verdict"] == "pass"
    assert d["quality"][0]["override_reason"] == "error confined to motion-blurred hand"


def test_fail_needs_no_reason_and_clears_stale_override():
    d = _doc()
    apply_review(d, "a", "c1", "pass", "x")
    apply_review(d, "a", "c1", "fail", None)
    assert d["quality"][0]["owner_verdict"] == "fail"
    assert "override_reason" not in d["quality"][0]


def test_unknown_entry_rejected():
    with pytest.raises(ReviewError):
        apply_review(_doc(), "zzz", "c1", "pass", None)
    with pytest.raises(ReviewError):
        apply_review(_doc(), "a", "c1", "maybe", "x")
