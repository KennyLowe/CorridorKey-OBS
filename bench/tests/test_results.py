import copy

import jsonschema
import pytest

from ckbench.results import validate

ENV = {
    "run_id": "20261016T000000Z-abcdef", "gpu_name": "NVIDIA GeForce RTX 3060", "gpu_driver": "576.80",
    "python": "3.12.12", "packages": {"torch": "2.8.0+cu128"},
    "corridorkey_sha": "97e55a453060745bead1befd293f6e523c4b845c", "weights_sha256": "a" * 64,
}
DOC = {"schema_version": "1.0", "environment": ENV, "benchmarks": []}
Q = {"config_id": "torch-fp16-768x768-ref1-idle", "clip_id": "c", "edge_alpha_mae": 0.05,
     "edge_fg_mae": 0.01, "flicker_excess": 0.0, "flags": ["edge_alpha"], "owner_verdict": "pending"}


def test_valid_doc_passes():
    validate(DOC)


def test_flagged_pass_without_reason_fails():
    d = copy.deepcopy(DOC)
    d["quality"] = [dict(Q, owner_verdict="pass")]
    with pytest.raises(jsonschema.ValidationError):
        validate(d)


def test_flagged_pass_with_reason_ok():
    d = copy.deepcopy(DOC)
    d["quality"] = [dict(Q, owner_verdict="pass", override_reason="edge error is in background noise")]
    validate(d)


def test_bad_sha_fails():
    d = copy.deepcopy(DOC)
    d["environment"] = dict(ENV, corridorkey_sha="nope")
    with pytest.raises(jsonschema.ValidationError):
        validate(d)
