"""Running ckbench on a second machine behind a corporate network (feature 002, issue #2):
speed-only runs, borrowed quality verdicts, clip download failures, clip hashes, and git's
"dubious ownership" refusal."""
import copy
import hashlib
import json
import subprocess
import urllib.error
from pathlib import Path

import pytest

from ckbench import cli, clips, env, envelope as E, perftable as PT, report

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUN_3060 = "20261016T221252Z-c16425"


def doc_3060():
    return json.loads((RESULTS / RUN_3060 / "results.json").read_text())


def speed_only(doc):
    d = copy.deepcopy(doc)
    d.pop("quality", None)
    d.pop("clips", None)
    d["environment"] = dict(d["environment"], run_id="20261020T000000Z-a2000x", gpu_name="NVIDIA RTX A2000 12GB",
                            gpu_driver="597.16")
    return d


# ---------------------------------------------------------------------------- envelope / report

def test_speed_only_envelope_uses_timing():
    d = speed_only(doc_3060())
    env_ = E.envelope(d["benchmarks"], [], allow_pending=True, quality_missing_ok=True)
    m = {e["mode_id"]: e["rtx3060"] for e in env_}
    assert m["1080p30"]["status"] == "supported"
    assert m["1080p30"]["quality"] == "not_run"
    # without the flag a run with no quality can't support anything (the 001 behaviour)
    strict = {e["mode_id"]: e["rtx3060"] for e in E.envelope(d["benchmarks"], [], allow_pending=True)}
    assert strict["1080p30"]["status"] != "supported"


def test_report_for_a_speed_only_run_says_quality_was_not_run():
    d = speed_only(doc_3060())
    text, v, env_, _ = report.render(d, allow_pending=True)
    assert "quality not run on this machine" in text
    assert f"see run {RUN_3060}" in text
    assert "quality not passed" not in text
    assert "NVIDIA RTX A2000 12GB (measured)" in text
    assert v["verdict"] in ("GO", "CONDITIONAL-GO")
    assert "RTX 3060" not in v["reasoning"]


def test_report_can_borrow_quality_verdicts():
    d = speed_only(doc_3060())
    text, v, env_, _ = report.render(d, allow_pending=False, quality_from=doc_3060())
    m = {e["mode_id"]: e["rtx3060"] for e in env_}
    assert m["1080p30"]["quality"] == "pass"
    assert f"quality verdicts borrowed from run {RUN_3060}".lower() in text.lower()


def test_perf_table_fits_from_timing_when_quality_is_missing():
    t = PT.build_table([speed_only(doc_3060())])
    (g,) = t["gpus"]
    assert g["gpu_name"] == "NVIDIA RTX A2000 12GB"
    m = {x["mode_id"]: {c["quality"]: c for c in x["configs"]} for x in g["modes"]}
    assert m["1080p30"]["1024x576_r0"]["fits"] is True  # 30.1 ms, no quality entries: timing decides


def test_all_speed_only_skips_clips_and_quality(monkeypatch):
    calls = []
    for name in ("cmd_env", "cmd_clips", "cmd_export", "cmd_speed", "cmd_quality", "cmd_report"):
        monkeypatch.setattr(cli, name, (lambda n: lambda a: calls.append((n, vars(a))) or 0)(name))
    assert cli.main(["all", "--speed-only"]) == 0
    names = [c[0] for c in calls]
    assert names == ["cmd_env", "cmd_export", "cmd_speed", "cmd_speed", "cmd_report"]
    assert calls[-1][1]["allow_pending"] is True


# ---------------------------------------------------------------------------- clips

def test_blocked_clip_download_gives_the_manual_route(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise urllib.error.HTTPError("https://www.pexels.com/download/video/1/", 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    clip = {"clip_id": "pexels-x", "source_url": "https://www.pexels.com/download/video/1/",
            "page_url": "https://www.pexels.com/video/x-1/", "local_path": "footage/public/pexels-x.mp4"}
    with pytest.raises(FileNotFoundError) as e:
        clips.fetch(clip, tmp_path / "pexels-x.mp4")
    msg = str(e.value)
    assert "503" in msg and "https://www.pexels.com/video/x-1/" in msg
    assert "footage/public/pexels-x.mp4" in msg and "another machine" in msg and "browser" in msg


def test_clip_sha256_mismatch_warns_but_continues(tmp_path, capsys):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"abc")
    assert clips.check_sha256({"clip_id": "c", "sha256": hashlib.sha256(b"abc").hexdigest()}, f) is True
    assert clips.check_sha256({"clip_id": "c", "sha256": "0" * 64}, f) is False
    assert "WARNING" in capsys.readouterr().err
    assert clips.check_sha256({"clip_id": "c"}, f) is None  # no hash pinned: nothing to check


def test_manifest_pins_the_public_clip_hashes():
    m = {c["clip_id"]: c for c in clips.load_manifest()}
    assert m["pexels-athlete"]["sha256"].startswith("6f7ce28d")
    assert m["pexels-dance"]["sha256"].startswith("71e2c2b2")
    assert m["pixabay-hair"]["sha256"].startswith("7ff23da9")


# ---------------------------------------------------------------------------- git

def test_corridorkey_sha_missing_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "VENDOR_DIR", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="not found.*setup.ps1"):
        env.corridorkey_sha()


def test_corridorkey_sha_dubious_ownership_prints_the_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "VENDOR_DIR", tmp_path)
    err = (f"fatal: detected dubious ownership in repository at '{tmp_path.as_posix()}'\n"
           "To add an exception for this directory, call:\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", err))
    with pytest.raises(RuntimeError) as e:
        env.corridorkey_sha()
    msg = str(e.value)
    assert "dubious ownership" in msg
    assert f"git config --global --add safe.directory {tmp_path.as_posix()}" in msg


def test_corridorkey_sha_other_git_error_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "VENDOR_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "fatal: not a git repository"))
    with pytest.raises(RuntimeError, match="not a git repository"):
        env.corridorkey_sha()
