"""Owner visual review (FR-007a): record pass/fail; passing a flagged entry needs a written reason."""
from __future__ import annotations


class ReviewError(ValueError):
    pass


def apply_review(doc: dict, config_id: str, clip_id: str, verdict: str, reason: str | None) -> dict:
    if verdict not in ("pass", "fail"):
        raise ReviewError(f"verdict must be pass or fail, got {verdict!r}")
    for q in doc.get("quality", []):
        if q["config_id"] == config_id and q["clip_id"] == clip_id:
            break
    else:
        raise ReviewError(f"no quality entry for config {config_id!r} and clip {clip_id!r}")
    reason = (reason or "").strip()
    if verdict == "pass" and q["flags"] and not reason:
        raise ReviewError(f"{config_id} on {clip_id} is flagged ({', '.join(q['flags'])}); "
                          "passing it requires --reason (FR-007a)")
    q["owner_verdict"] = verdict
    if reason:
        q["override_reason" if verdict == "pass" and q["flags"] else "review_note"] = reason
    if verdict == "fail" or not q["flags"]:
        q.pop("override_reason", None)
    return q
