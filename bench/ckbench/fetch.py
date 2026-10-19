"""`ckbench fetch-weights` and `ckbench check-models` (feature 002, research D3a,
contracts/prepare-cli.md). The OBS plugin runs these as child processes, so all network code and
its failure modes stay out of the OBS process.

fetch-weights:
- The URL is built only from the model manifest (models/manifest.json):
  ``<base_url>/<hf_repo>/resolve/<revision>/<file>`` with base_url https://huggingface.co (a test
  manifest may point base_url at a local server). Redirects are followed only to Hugging Face hosts
  over https (huggingface.co, hf.co and their subdomains, which serve the LFS/Xet content).
- It downloads to ``<dest>/<file>.part`` (resuming with HTTP Range), verifies the size and sha256
  against the manifest, then renames atomically. A mismatch deletes the part file. An existing file
  with the right hash is a cache hit.

check-models:
- Lists https://huggingface.co/api/models?author=nikopueringer and reports CorridorKey_v<version>
  repos newer than the newest model in the manifest. CorridorKeyBlue* (blue screen) is out of
  scope and never reported. It only reports: switching models is always an explicit user action.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .results import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
DEFAULT_BASE = "https://huggingface.co"
MODELS_API = "https://huggingface.co/api/models?author=nikopueringer"
DEFAULT_DEST = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "corridorkey-obs" / "weights"
TIMEOUT_S = 10.0
CHUNK = 1 << 20
UA = "corridorkey-obs/0.1 (+https://github.com/KennyLowe/CorridorKey-OBS)"
_VERSION = re.compile(r"^CorridorKey_v(\d+(?:\.\d+)*)$")


class FetchError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_model(manifest: dict, model_id: str) -> dict:
    for m in manifest.get("models", []):
        if m["model_id"] == model_id:
            return m
    raise FetchError("unknown_model", f"{model_id} is not in the model manifest")


def source_url(entry: dict) -> str:
    base = entry.get("base_url", DEFAULT_BASE).rstrip("/")
    return f"{base}/{entry['hf_repo']}/resolve/{entry['revision']}/{entry['file']}"


def _is_hf_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in ("huggingface.co", "hf.co"))


def allowed_redirect(url: str, base_url: str) -> bool:
    u, b = urlparse(url), urlparse(base_url)
    if _is_hf_host(b.hostname or ""):
        return u.scheme == "https" and _is_hf_host(u.hostname or "")
    return u.scheme == b.scheme and u.netloc == b.netloc  # local test server


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, base_url: str):
        self.base_url = base_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_redirect(newurl, self.base_url):
            raise FetchError("http", f"refusing redirect to a non-Hugging Face host: {urlparse(newurl).netloc}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener(base_url: str):
    return urllib.request.build_opener(_Redirects(base_url))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK * 4), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(manifest_path: str, model_id: str, dest_dir: str, emit: Callable[[dict], None],
                  timeout: float = TIMEOUT_S) -> int:
    def fail(code: str, msg: str) -> int:
        emit({"event": "result", "ok": False, "error": code, "msg": msg})
        return 1

    try:
        entry = find_model(load_manifest(manifest_path), model_id)
        size, sha = int(entry["size"]), entry["sha256"].lower()
        dest = Path(dest_dir)
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return fail("io", f"cannot create {dest}: {e}")
        final = dest / entry["file"]
        part = dest / (entry["file"] + ".part")
        if final.exists() and final.stat().st_size == size:
            emit({"event": "progress", "stage": "verify", "pct": 99, "bytes": size, "total": size})
            if _sha256(final) == sha:
                emit({"event": "progress", "stage": "done", "pct": 100, "bytes": size, "total": size})
                emit({"event": "result", "ok": True, "weights_path": str(final), "sha256": sha, "model_id": model_id})
                return 0

        url = source_url(entry)
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        emit({"event": "progress", "stage": "download", "pct": int(have * 98 / size), "bytes": have, "total": size})
        if have < size:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                resp = _opener(entry.get("base_url", DEFAULT_BASE)).open(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                return fail("http", f"HTTP {e.code} from {urlparse(e.url or url).netloc}: {e.reason}")
            except FetchError:
                raise
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
                return fail("network", f"cannot reach {urlparse(url).netloc}: {getattr(e, 'reason', e)}")
            if have and resp.status != 206:
                have = 0  # server ignored the range: start again
            # the server's own idea of the full file size (to tell "short file" from "cut off")
            declared = None
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                declared = int(cr.rsplit("/", 1)[1])
            elif resp.headers.get("Content-Length", "").isdigit():
                declared = have + int(resp.headers["Content-Length"])
            last_emit = 0.0
            try:
                with resp, open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        now = time.monotonic()
                        if now - last_emit > 0.25:
                            last_emit = now
                            emit({"event": "progress", "stage": "download", "pct": int(min(have, size) * 98 / size),
                                  "bytes": have, "total": size})
            except (socket.timeout, ConnectionError, http.client.IncompleteRead, urllib.error.URLError) as e:
                return fail("network", f"download interrupted at {have} of {size} bytes: {e}; retry to resume")
            except OSError as e:
                code = "disk" if getattr(e, "errno", None) == 28 else "io"
                return fail(code, f"cannot write {part}: {e}")
            emit({"event": "progress", "stage": "download", "pct": int(min(have, size) * 98 / size), "bytes": have,
                  "total": size})
            if have < size and declared is not None and have >= declared:
                part.unlink(missing_ok=True)
                return fail("hash_mismatch", f"the server's file is {declared} bytes, the manifest says {size}; "
                                             "the partial file was deleted")
            if have < size:
                return fail("network", f"download ended early at {have} of {size} bytes; retry to resume")

        emit({"event": "progress", "stage": "verify", "pct": 98, "bytes": size, "total": size})
        got_size = part.stat().st_size
        got = _sha256(part) if got_size == size else None
        if got != sha:
            part.unlink(missing_ok=True)
            return fail("hash_mismatch", (f"size {got_size} != {size}" if got is None else f"sha256 {got} != {sha}")
                        + "; the partial file was deleted")
        os.replace(part, final)
        emit({"event": "progress", "stage": "done", "pct": 100, "bytes": size, "total": size})
        emit({"event": "result", "ok": True, "weights_path": str(final), "sha256": sha, "model_id": model_id})
        return 0
    except FetchError as e:
        return fail(e.code, e.msg)
    except OSError as e:
        return fail("io", f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- check-models

def _version(model_id: str):
    m = _VERSION.match(model_id)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def newer_models(manifest: dict, listing: list[dict]) -> dict:
    known = [m["model_id"] for m in manifest["models"] if _version(m["model_id"])]
    current = max(known, key=_version)
    newer = []
    for item in listing:
        repo = item.get("id", "")
        name = repo.split("/")[-1]
        v = _version(name)  # CorridorKeyBlue_* and anything else don't match
        if v and v > _version(current):
            newer.append({"model_id": name, "hf_repo": repo, "last_modified": item.get("lastModified", "")})
    newer.sort(key=lambda n: _version(n["model_id"]))
    return {"current": current, "newer": newer}


def _get_json(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with _opener(DEFAULT_BASE).open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_models(manifest_path: str, emit: Callable[[dict], None], api_url: str = MODELS_API,
                 timeout: float = TIMEOUT_S) -> int:
    try:
        man = load_manifest(manifest_path)
        listing = _get_json(api_url, timeout)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError, ValueError) as e:
        emit({"event": "result", "ok": False, "error": "network", "msg": f"model check failed: {e}"})
        return 1
    emit({"event": "result", "ok": True, **newer_models(man, listing)})
    return 0
