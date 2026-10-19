"""Reference clips: fetch, probe, and normalise to 1080p30 PNG sequences (US3; research D10)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .env import ffmpeg_exe
from .frames import CLIPS_OUT
from .results import BENCH_DIR, REPO_ROOT

MANIFEST = BENCH_DIR / "clips.toml"
TARGET_W, TARGET_H, TARGET_FPS, MAX_FRAMES = 1920, 1080, 30, 300
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


def load_manifest(path: Path | None = None) -> list[dict]:
    with open(path or MANIFEST, "rb") as f:
        return tomllib.load(f)["clip"]


def _ffprobe() -> str:
    exe = ffmpeg_exe()
    return str(Path(exe).with_name("ffprobe.exe")) if exe and Path(exe).with_name("ffprobe.exe").exists() else "ffprobe"


def check_sha256(clip: dict, path: Path) -> bool | None:
    """Optional `sha256` in clips.toml: warn (don't fail) on a mismatch. None = nothing pinned."""
    want = clip.get("sha256")
    if not want:
        return None
    import hashlib
    import sys

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != want.lower():
        print(f"WARNING: {clip['clip_id']}: {path} has sha256 {h.hexdigest()}, clips.toml pins {want}. "
              "The source may have changed the file; results may not be comparable with other runs.",
              file=sys.stderr)
        return False
    return True


def fetch(clip: dict, dst: Path) -> None:
    """Download a public clip, or raise FileNotFoundError with the manual route (bot protection and
    corporate proxies often block these downloads)."""
    try:
        download(clip["source_url"], dst)
    except (urllib.error.URLError, OSError) as e:  # HTTPError is a URLError
        what = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else f"{type(e).__name__}: {e}"
        raise FileNotFoundError(
            f"{clip['clip_id']}: download failed ({what}) from {clip['source_url']}. Download it manually: open "
            f"{clip['page_url']} in a browser and save the video as {clip['local_path']} (in the repository), or "
            f"copy that file from another machine that has it.") from None


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dst)


def probe_video(path: Path) -> tuple[str, float, str]:
    out = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,color_primaries", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return f"{s['width']}x{s['height']}", round(float(num) / float(den), 3), s.get("color_primaries", "unknown")


def _frame_tensors(folder: Path, step: int = 10):
    import cv2
    import torch

    for f in sorted(folder.glob("*.png"))[::step]:
        yield torch.from_numpy(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255


def hint_diagnostic(folder: Path, hint_fn=None) -> dict:
    """Mean hint (1 = subject) over a 32 px border of frames 0/mid/last, where the screen usually
    is. Values near 1 mean the hint does not recognise the screen."""
    import cv2
    import torch

    from .hint import chroma_hint

    hint_fn = hint_fn or chroma_hint
    files = sorted(folder.glob("*.png"))
    vals, frame_means = [], []
    for f in (files[0], files[len(files) // 2], files[-1]):
        rgb = torch.from_numpy(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        m = hint_fn(rgb)
        b = torch.cat([m[:32].flatten(), m[-32:].flatten(), m[:, :32].flatten(), m[:, -32:].flatten()])
        vals.append(float(b.mean()))
        frame_means.append(float(m.mean()))
    return {"hint_border_mean": round(sum(vals) / len(vals), 3),
            "hint_frame_mean": round(sum(frame_means) / len(frame_means), 3)}


def hint_fn_for(clip_entry: dict):
    """The hint function a clip's results entry asks for (fixed unless calibrated)."""
    from functools import partial

    from .hint import calibrated_hint, chroma_hint

    if clip_entry.get("hint", "fixed") == "calibrated":
        return partial(calibrated_hint, key=tuple(clip_entry["hint_key_colour"]))
    return chroma_hint


def png_sequence_input(folder: Path) -> tuple[list[str], str, int]:
    """ffmpeg input args for a numbered PNG sequence; returns (args, native_res, count)."""
    pngs = sorted(folder.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no PNGs in {folder}")
    m = re.match(r"^(.*?)(\d+)\.png$", pngs[0].name)
    if not m:
        raise ValueError(f"cannot parse frame number from {pngs[0].name}")
    prefix, digits = m.group(1), m.group(2)
    pattern = str(folder / f"{prefix}%0{len(digits)}d.png")
    import cv2

    h, w = cv2.imread(str(pngs[0])).shape[:2]
    args = ["-framerate", str(TARGET_FPS), "-start_number", str(int(digits)), "-i", pattern]
    return args, f"{w}x{h}", len(pngs)


def _screen_green(input_args: list[str]) -> str:
    """Median colour of the first frame's border, as an ffmpeg hex colour for padding."""
    raw = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *input_args, "-frames:v", "1",
                          "-vf", "scale=320:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 320, 3)
    border = np.concatenate([px[:4].reshape(-1, 3), px[-4:].reshape(-1, 3), px[:, :4].reshape(-1, 3),
                             px[:, -4:].reshape(-1, 3)])
    r, g, b = np.median(border, axis=0).astype(int)
    return f"0x{r:02X}{g:02X}{b:02X}"


def normalise(clip: dict, log=print) -> dict:
    """Fetch (if needed) and normalise one clip; returns its results.json entry."""
    local = REPO_ROOT / clip["local_path"]
    if clip["source_kind"] == "video":
        if not local.exists():
            if clip["manual_download"]:
                raise FileNotFoundError(f"{local} missing: download manually from {clip['page_url']}")
            log(f"  downloading {clip['clip_id']} from {clip['source_url']}")
            fetch(clip, local)
        check_sha256(clip, local)
        native_res, native_fps, primaries = probe_video(local)
        input_args = ["-ss", str(clip.get("segment_start_s", 0.0)), "-i", str(local)]
        available = None
    else:
        if not local.exists():
            raise FileNotFoundError(f"{local} missing: download manually from {clip['page_url']}")
        input_args, native_res, available = png_sequence_input(local)
        native_fps = float(TARGET_FPS)  # image sequences have no fps; treated as 30 (research D10)
        primaries = "unknown (PNG, assumed sRGB)"

    out = CLIPS_OUT / clip["clip_id"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pad = _screen_green(input_args)
    # BT.2020-tagged 8-bit SDR sources are converted to BT.709/sRGB primaries (transfer assumed BT.709).
    conv = "colorspace=all=bt709:iall=bt2020:itrc=bt709:fast=0," if primaries == "bt2020" else ""
    vf = (f"{conv}fps={TARGET_FPS},scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color={pad},format=rgb24")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *input_args, "-vf", vf,
           "-frames:v", str(MAX_FRAMES), "-start_number", "0", str(out / "%05d.png")]
    subprocess.run(cmd, check=True)
    n = len(list(out.glob("*.png")))
    log(f"  {clip['clip_id']}: {native_res} @ {native_fps} -> {n} frames at 1080p30 (pad {pad})")
    entry = {k: clip[k] for k in ("clip_id", "focus", "source_url", "licence", "licence_checked_on",
                                  "manual_download", "commit_stills_allowed", "source_kind")}
    entry.update(page_url=clip["page_url"], native_res=native_res, native_fps=native_fps,
                 segment_start_s=float(clip.get("segment_start_s", 0.0)), frames=n,
                 local_path=clip["local_path"], pad_colour=pad, colour_primaries=primaries,
                 colour_conversion=("bt2020->bt709 (ffmpeg colorspace)" if conv else "none"),
                 hint=clip.get("hint", "fixed"), **hint_diagnostic(out))
    if entry["hint"] == "calibrated":
        from functools import partial

        from .hint import BORDER_PX, CAL_T0, CAL_T1, calibrated_hint, estimate_key_colour

        key = estimate_key_colour(_frame_tensors(out))
        entry["hint_key_colour"] = [round(v, 4) for v in key]
        entry["hint_calibration"] = (f"median RGB of a {BORDER_PX} px border band over every 10th frame; "
                                     f"key = (g - max(r,b)) / screen difference, smoothstep {CAL_T0}-{CAL_T1}, 5x5 blur")
        entry["hint_diagnostic_fixed"] = {k: entry.pop(k) for k in ("hint_border_mean", "hint_frame_mean")}
        entry.update(hint_diagnostic(out, partial(calibrated_hint, key=key)))
        log(f"  {clip['clip_id']}: calibrated key colour {entry['hint_key_colour']}; border hint "
            f"{entry['hint_diagnostic_fixed']['hint_border_mean']} (fixed) -> {entry['hint_border_mean']} (calibrated)")
    if available is not None:
        entry["frames_available"] = available
    return entry
