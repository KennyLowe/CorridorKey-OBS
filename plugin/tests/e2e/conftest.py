import json
import re
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import obs_harness as H  # noqa: E402

RESULTS = H.ARTIFACTS / "e2e-results.json"


def record(key: str, value) -> None:
    """Measured numbers for plugin/TEST-RESULTS.md (git-ignored artefact)."""
    H.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=2))


_TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d): (.*)$")


def log_events(o: "H.Obs", pattern: str, after_s: float = 0.0) -> list[tuple[float, str]]:
    """[(seconds since midnight, line)] of corridorkey log lines matching a regex."""
    out = []
    for line in o.log_text().splitlines():
        m = _TS.match(line)
        if not m or "[corridorkey]" not in line:
            continue
        t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        if t >= after_s and re.search(pattern, m[5]):
            out.append((t, m[5]))
    return out


def now_s() -> float:
    lt = time.localtime()
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec + (time.time() % 1)


@pytest.fixture(scope="module")
def obs_session():
    H.write_test_config(H.ARTIFACTS / "rec")
    o = H.Obs().start()
    try:
        yield o
    finally:
        o.close()
        assert o.proc is None
