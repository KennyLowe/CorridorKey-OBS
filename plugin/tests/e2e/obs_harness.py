r"""Drive a real OBS Studio 32.2.2 over obs-websocket for the plugin's end-to-end tests.

The harness never touches the owner's own OBS installation, settings or running instance:
- it runs a private *portable* copy of the installed OBS 32.2.2 binaries in the git-ignored
  outputs/obs-portable/ (made with `robocopy "C:\Program Files\obs-studio" outputs\obs-portable /E`),
  started with --portable, so its configuration lives in outputs/obs-portable/config/ and the
  per-machine plugin folder (C:\ProgramData\obs-studio\plugins) is not scanned;
- the plugin under test is loaded from the build's rundir through OBS_PLUGINS_PATH /
  OBS_PLUGINS_DATA_PATH (the same DLL and data that scripts/install.ps1 installs);
- it uses a throwaway profile and scene collection ("CKTest") and an obs-websocket server on
  localhost:CK_OBS_WS_PORT (default 4466) with a password from CK_OBS_WS_PASSWORD.

Run from the repo root:  uv run --project bench pytest plugin/tests/e2e
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(__file__).resolve().parent / ".artifacts"
PORTABLE = Path(os.environ.get("CK_OBS_PORTABLE", str(REPO / "outputs" / "obs-portable")))
OBS_DIR = PORTABLE / "bin" / "64bit"
OBS_EXE = OBS_DIR / "obs64.exe"
CFG = PORTABLE / "config" / "obs-studio"
PLUGIN_RUNDIR = Path(os.environ.get("CK_PLUGIN_RUNDIR", str(REPO / "plugin" / "build_x64" / "rundir" / "RelWithDebInfo")))


def _guard_cfg() -> None:
    """Never write OBS configuration outside the private portable copy (the owner's OBS lives in
    %APPDATA%\\obs-studio and must not be touched)."""
    appdata = Path(os.environ.get("APPDATA", "")).resolve()
    if not CFG.resolve().is_relative_to(PORTABLE.resolve()) or CFG.resolve().is_relative_to(appdata):
        raise RuntimeError(f"refusing to write OBS config at {CFG}: not inside {PORTABLE}")
PROFILE = "CKTest"
COLLECTION = "CKTest"
SCENE = "CK Scene"
PORT = int(os.environ.get("CK_OBS_WS_PORT", "4466"))
TRT_DIR = r"C:\TensorRT-10.16.1.11\bin"
WEIGHTS = REPO / "bench" / "vendor" / "CorridorKey" / "CorridorKeyModule" / "checkpoints" / "CorridorKey_v1.0.safetensors"
FILTER_ID = "corridorkey_filter"


def password() -> str:
    pw = os.environ.get("CK_OBS_WS_PASSWORD")
    if not pw:
        pw = secrets.token_hex(12)
        os.environ["CK_OBS_WS_PASSWORD"] = pw
    return pw


# ----------------------------------------------------------------------------- config

def backup_config() -> None:  # the portable copy has its own config: nothing to back up
    pass


def restore_config() -> None:
    pass


def write_test_config(rec_dir: Path) -> None:
    _guard_cfg()
    ws = CFG / "plugin_config" / "obs-websocket" / "config.json"
    ws.parent.mkdir(parents=True, exist_ok=True)
    ws.write_text(json.dumps({"alerts_enabled": False, "auth_required": True, "first_load": False,
                              "server_enabled": True, "server_password": password(), "server_port": PORT}))
    prof = CFG / "basic" / "profiles" / PROFILE
    prof.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    (prof / "basic.ini").write_text(
        "[General]\nName=CKTest\n\n"
        "[Video]\nBaseCX=1920\nBaseCY=1080\nOutputCX=1920\nOutputCY=1080\nFPSType=1\nFPSInt=30\n"
        "ColorFormat=NV12\nColorSpace=709\nColorRange=Partial\n\n"
        "[Output]\nMode=Simple\n\n"
        f"[SimpleOutput]\nFilePath={rec_dir.as_posix()}\nRecFormat2=mkv\nVBitrate=6000\nABitrate=160\n"
        "StreamEncoder=nvenc\nRecQuality=Stream\nNVENCPreset2=p5\nUseAdvanced=false\n", encoding="utf-8")
    scenes = CFG / "basic" / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    (scenes / f"{COLLECTION}.json").write_text(json.dumps({
        "name": COLLECTION, "current_scene": SCENE, "current_program_scene": SCENE,
        "scene_order": [{"name": SCENE}], "groups": [], "transitions": [], "modules": {},
        "sources": [{"id": "scene", "versioned_id": "scene", "name": SCENE, "uuid": "c0ffee00-0000-4000-8000-000000000001",
                     "settings": {"items": [], "id_counter": 0, "custom_size": False}, "enabled": True, "flags": 0,
                     "mixers": 0, "volume": 1.0, "filters": []}],
    }), encoding="utf-8")
    # no first-run wizard, no exit confirmation
    ui = CFG / "user.ini"
    text = ui.read_text(encoding="utf-8-sig") if ui.exists() else "[General]\n"
    for key, val in (("FirstRun", "true"), ("ConfirmOnExit", "false")):
        if re.search(rf"^{key}=.*$", text, flags=re.M):
            text = re.sub(rf"^{key}=.*$", f"{key}={val}", text, flags=re.M)
        else:
            text = text.replace("[General]\n", f"[General]\n{key}={val}\n", 1)
    ui.write_text(text, encoding="utf-8")


def remove_test_profile() -> None:
    _guard_cfg()
    shutil.rmtree(CFG / "basic" / "profiles" / PROFILE, ignore_errors=True)
    (CFG / "basic" / "scenes" / f"{COLLECTION}.json").unlink(missing_ok=True)
    for bak in (CFG / "basic" / "scenes").glob(f"{COLLECTION}.json.bak*"):
        bak.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- windows

_user32 = ctypes.windll.user32
WM_CLOSE = 0x0010


def _windows_of(pid: int) -> list[tuple[int, str, bool]]:
    out = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        p = wt.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid:
            buf = ctypes.create_unicode_buffer(256)
            _user32.GetWindowTextW(hwnd, buf, 256)
            out.append((hwnd, buf.value, bool(_user32.IsWindowVisible(hwnd))))
        return True

    _user32.EnumWindows(cb, 0)
    return out


# ----------------------------------------------------------------------------- OBS

@dataclass
class Obs:
    proc: subprocess.Popen | None = None
    cl: object = None
    log_path: Path | None = None
    started_at: float = 0.0
    extra_args: list = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)

    def start(self, timeout: float = 60.0) -> "Obs":
        import obsws_python as obs

        logs = CFG / "logs"
        before = set(logs.glob("*.txt")) if logs.exists() else set()
        if not OBS_EXE.exists():
            raise RuntimeError(f"portable OBS missing at {PORTABLE}: copy C:/Program Files/obs-studio there (robocopy /E)")
        args = [str(OBS_EXE), "--portable", "--profile", PROFILE, "--collection", COLLECTION, "--minimize-to-tray",
                "--disable-shutdown-check", "--disable-updater", "--multi", *self.extra_args]
        env = {**os.environ, "OBS_PLUGINS_PATH": str(PLUGIN_RUNDIR), "OBS_PLUGINS_DATA_PATH": str(PLUGIN_RUNDIR),
               **{k: str(v) for k, v in self.extra_env.items()}}
        self.started_at = time.time()
        self.proc = subprocess.Popen(args, cwd=str(OBS_DIR), env=env)
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            try:
                self.cl = obs.ReqClient(host="localhost", port=PORT, password=password(), timeout=10)
                self.cl.send("GetSceneList", {}, raw=True)  # 207 "not ready" until OBS has loaded
                break
            except Exception as e:  # not up yet
                last = e
                time.sleep(1.0)
        else:
            self.kill()
            raise RuntimeError(f"OBS websocket did not come up: {last}")
        after = set(logs.glob("*.txt"))
        new = sorted(after - before, key=lambda p: p.stat().st_mtime)
        self.log_path = new[-1] if new else max(after, key=lambda p: p.stat().st_mtime)
        return self

    def log_text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace") if self.log_path else ""

    def ck_log(self) -> list[str]:
        return [l for l in self.log_text().splitlines() if "[corridorkey]" in l]

    def close(self, timeout: float = 30.0) -> None:
        if not self.proc:
            return
        try:
            if self.cl is not None:
                try:
                    if self.cl.get_record_status().output_active:
                        self.cl.stop_record()
                        time.sleep(1.0)
                except Exception:
                    pass
                try:
                    self.cl.disconnect()
                except Exception:
                    pass
            for hwnd, title, _ in _windows_of(self.proc.pid):
                if title.startswith("OBS ") and " - Profile: " in title:
                    _user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
        self.proc = None
        self.cl = None

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None
        self.cl = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # --------------------------------------------------------------------- requests
    def send(self, req: str, data: dict | None = None):
        return self.cl.send(req, data or {}, raw=True)

    def stats(self) -> dict:
        return self.send("GetStats")

    def status(self, source: str, flt: str = "CorridorKey") -> dict:
        r = self.send("CallVendorRequest", {"vendorName": "corridorkey", "requestType": "GetFilterStatus",
                                            "requestData": {"sourceName": source, "filterName": flt}})
        return r["responseData"]

    def wait_status(self, source: str, pred, timeout: float, flt: str = "CorridorKey", poll: float = 0.05) -> dict:
        t0 = time.time()
        s = None
        while time.time() - t0 < timeout:
            s = self.status(source, flt)
            if pred(s):
                return s
            time.sleep(poll)
        raise TimeoutError(f"status condition not met in {timeout}s; last: {s}")

    def add_color_background(self, name: str = "bg", rgb=(128, 128, 128)) -> None:
        r, g, b = rgb
        self.send("CreateInput", {"sceneName": SCENE, "inputName": name, "inputKind": "color_source_v3",
                                  "inputSettings": {"color": 0xFF000000 | (b << 16) | (g << 8) | r,
                                                    "width": 1920, "height": 1080}})

    def add_media(self, name: str, path: Path, loop: bool = True) -> None:
        self.send("CreateInput", {"sceneName": SCENE, "inputName": name, "inputKind": "ffmpeg_source",
                                  "inputSettings": {"local_file": str(path), "looping": loop, "is_local_file": True,
                                                    "restart_on_activate": False, "hw_decode": False,
                                                    "close_when_inactive": False}})

    def add_filter(self, source: str, settings: dict, name: str = "CorridorKey") -> None:
        self.send("CreateSourceFilter", {"sourceName": source, "filterName": name, "filterKind": FILTER_ID,
                                         "filterSettings": settings})

    def set_filter(self, source: str, settings: dict, name: str = "CorridorKey") -> None:
        self.send("SetSourceFilterSettings", {"sourceName": source, "filterName": name,
                                              "filterSettings": settings, "overlay": True})

    def enable_filter(self, source: str, on: bool, name: str = "CorridorKey") -> None:
        self.send("SetSourceFilterEnabled", {"sourceName": source, "filterName": name, "filterEnabled": on})

    def remove_filter(self, source: str, name: str = "CorridorKey") -> None:
        self.send("RemoveSourceFilter", {"sourceName": source, "filterName": name})

    def screenshot(self, source: str, w: int = 480, h: int = 270) -> np.ndarray:
        """RGBA uint8 [h, w, 4] of a source (its filters applied)."""
        import cv2

        r = self.send("GetSourceScreenshot", {"sourceName": source, "imageFormat": "png", "imageWidth": w,
                                              "imageHeight": h})
        data = base64.b64decode(r["imageData"].split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    def start_record(self) -> None:
        self.send("StartRecord")

    def stop_record(self) -> None:
        try:
            self.send("StopRecord")
        except Exception:
            pass


def default_filter_settings(**over) -> dict:
    s = {"weights_path": str(WEIGHTS), "trt_dir": TRT_DIR, "quality": "512_r0", "fallback_mode": "colorkey",
         "fallback_timeout_ms": 250, "debug_fail": "none", "hint_mode": "fixed", "show_hint": False}
    s.update(over)
    return s


def measure(o: Obs, seconds: float) -> dict:
    """OBS render statistics over a window: active fps (mean of samples), render frames skipped."""
    a = o.stats()
    fps = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        fps.append(o.stats()["activeFps"])
    b = o.stats()
    return {"seconds": seconds, "active_fps_mean": float(np.mean(fps)), "active_fps_min": float(np.min(fps)),
            "render_skipped": b["renderSkippedFrames"] - a["renderSkippedFrames"],
            "render_total": b["renderTotalFrames"] - a["renderTotalFrames"],
            "output_skipped": b["outputSkippedFrames"] - a["outputSkippedFrames"],
            "avg_frame_render_ms": b["averageFrameRenderTime"]}
