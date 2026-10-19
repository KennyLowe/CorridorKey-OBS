"""bench/setup-net.ps1: corporate-network robustness of setup.ps1 (feature 002).

Fake curl / git / uv executables emulate the failures seen on the owner's A2000 machine:
- schannel CRYPT_E_NO_REVOCATION_CHECK (curl exit 35; git clone fails), which works with
  --ssl-no-revoke / http.schannelCheckRevoke=false;
- uv failing on certificates until UV_NATIVE_TLS=1.
"""
import hashlib
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
NET = BENCH / "setup-net.ps1"
REVOKE = "curl: (35) schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012)"


def ps(script: str, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                           f". '{NET}'; {script}"], cwd=cwd, capture_output=True, text=True, env=env)


def fake(tmp: Path, name: str, body: str) -> Path:
    p = tmp / f"{name}.cmd"
    p.write_text("@echo off\r\n" + textwrap.dedent(body).replace("\n", "\r\n"))
    return p


@pytest.fixture
def payload(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(5000))
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def curl_fake(tmp, src, revocation: bool):
    # copies src to the -o target; fails like schannel unless --ssl-no-revoke is given
    return fake(tmp, "curl", f"""
        set OUT=
        set NOREVOKE=0
        :loop
        if "%~1"=="" goto done
        if "%~1"=="-o" set OUT=%~2
        if "%~1"=="--ssl-no-revoke" set NOREVOKE=1
        shift
        goto loop
        :done
        if "{int(revocation)}"=="1" if "%NOREVOKE%"=="0" goto revoke
        if not "%OUT%"=="" copy /y "{src}" "%OUT%" >nul
        exit /b 0
        :revoke
        echo {REVOKE} 1>&2
        exit /b 35
    """)


def test_download_retries_without_revocation_check(tmp_path, payload):
    src, sha = payload
    exe = curl_fake(tmp_path, src, revocation=True)
    out = tmp_path / "w.bin"
    r = ps(f"Get-VerifiedFile -Url 'https://example.invalid/w' -Out '{out}' -Sha256 '{sha}' -CurlExe '{exe}'", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.read_bytes() == src.read_bytes()
    assert "revocation" in (r.stdout + r.stderr).lower() and "--ssl-no-revoke" in r.stdout + r.stderr


def test_download_without_problem_does_not_disable_revocation(tmp_path, payload):
    src, sha = payload
    exe = curl_fake(tmp_path, src, revocation=False)
    out = tmp_path / "w.bin"
    r = ps(f"Get-VerifiedFile -Url 'https://example.invalid/w' -Out '{out}' -Sha256 '{sha}' -CurlExe '{exe}'", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ssl-no-revoke" not in r.stdout + r.stderr


def test_download_hash_mismatch_fails_and_deletes(tmp_path, payload):
    src, _ = payload
    exe = curl_fake(tmp_path, src, revocation=False)
    out = tmp_path / "w.bin"
    r = ps(f"Get-VerifiedFile -Url 'https://example.invalid/w' -Out '{out}' -Sha256 '{'0' * 64}' -CurlExe '{exe}'", tmp_path)
    assert r.returncode != 0
    assert "sha256" in (r.stdout + r.stderr).lower()
    assert not out.exists() and not Path(str(out) + ".part").exists()


def test_other_curl_failures_are_not_retried(tmp_path):
    exe = fake(tmp_path, "curl", "echo curl: (6) Could not resolve host 1>&2\nexit /b 6\n")
    r = ps(f"Get-VerifiedFile -Url 'https://example.invalid/w' -Out '{tmp_path / 'w.bin'}' -Sha256 '{'0' * 64}' -CurlExe '{exe}'",
           tmp_path)
    assert r.returncode != 0 and "ssl-no-revoke" not in r.stdout + r.stderr


def test_git_clone_retries_with_schannel_revocation_off_for_that_command_only(tmp_path):
    log = tmp_path / "git.log"
    exe = fake(tmp_path, "git", f"""
        echo %* >> "{log}"
        echo %* | findstr /C:"http.schannelCheckRevoke=false" >nul && exit /b 0
        echo fatal: unable to access 'https://github.com/x/': schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012) 1>&2
        exit /b 128
    """)
    r = ps(f"Invoke-GitNet -GitExe '{exe}' clone --quiet https://github.com/x/y.git '{tmp_path / 'y'}'", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and "-c http.schannelCheckRevoke=false clone" in calls[1]
    assert "revocation" in (r.stdout + r.stderr).lower()


def test_uv_sync_retries_with_system_certs(tmp_path):
    # current uv (0.13.1 on the A2000 machine) reads UV_SYSTEM_CERTS; older uv (0.7) reads UV_NATIVE_TLS
    exe = fake(tmp_path, "uv", """
        if "%UV_SYSTEM_CERTS%"=="1" if "%UV_NATIVE_TLS%"=="1" exit /b 0
        echo error: Request failed after 3 retries: invalid peer certificate: UnknownIssuer 1>&2
        exit /b 2
    """)
    r = ps(f"Invoke-UvSync -UvExe '{exe}'", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "UV_SYSTEM_CERTS" in r.stdout + r.stderr and "UV_NATIVE_TLS" in r.stdout + r.stderr


def test_uv_sync_other_failure_is_not_retried(tmp_path):
    exe = fake(tmp_path, "uv", "echo error: No solution found 1>&2\nexit /b 1\n")
    r = ps(f"Invoke-UvSync -UvExe '{exe}'", tmp_path)
    assert r.returncode != 0 and "UV_SYSTEM_CERTS" not in r.stdout


def test_git_head_dubious_ownership_prints_the_fix(tmp_path):
    exe = fake(tmp_path, "git", """
        echo fatal: detected dubious ownership in repository at 'C:/git/corridorkey/bench/vendor/CorridorKey' 1>&2
        echo To add an exception for this directory, call: 1>&2
        exit /b 128
    """)
    r = ps(f"Get-GitHead -Dir '{tmp_path}' -GitExe '{exe}'", tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "git config --global --add safe.directory C:/git/corridorkey/bench/vendor/CorridorKey" in out


def test_git_head_ok(tmp_path):
    exe = fake(tmp_path, "git", "echo 97e55a453060745bead1befd293f6e523c4b845c\nexit /b 0\n")
    r = ps(f"Get-GitHead -Dir '{tmp_path}' -GitExe '{exe}'", tmp_path)
    assert r.returncode == 0 and "97e55a453060745bead1befd293f6e523c4b845c" in r.stdout
