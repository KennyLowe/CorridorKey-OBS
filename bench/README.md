# ckbench: CorridorKey live-keying feasibility benchmark

`ckbench` measures whether [CorridorKey](https://github.com/nikopueringer/CorridorKey), Corridor
Digital's neural green-screen keyer, can key live video in real time on a given NVIDIA GPU. It
times the model, in PyTorch and as a TensorRT fp16 engine, at internal resolutions 512, 768, 1024
and 2048, with and without the refiner, both idle and while an NVENC stream encode runs. It scores
quality and flicker against the 2048 reference on public green-screen clips. It then writes a
`results.json` and a `REPORT.md` with a GO / CONDITIONAL-GO / NO-GO verdict.

Results from the development machine (RTX 3060 12 GB) are in [`results/`](results/).

## Prerequisites

- Windows 10/11 x64 with an NVIDIA GPU and a driver that supports CUDA 12.8 (R570 or newer).
- [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched by uv if needed) and git.
- ffmpeg with `h264_nvenc`, on `PATH`:
  `winget install --id Gyan.FFmpeg -e --version 7.1.1`.
  ffmpeg 8.1 and later need NVENC API 13.1, which needs driver 610 or newer. On older drivers they
  list `h264_nvenc` but can't open it. `setup.ps1` actually opens the encoder to check this.
- About 40 GB free disk. Building the 2048 TensorRT engine needs a lot of RAM (the ONNX trace
  was reported to need ~140 GB). If it fails, that's recorded and the benchmark carries on.
- Two Hollywood Camera Work plates, which you download yourself because their licence forbids
  rehosting: "Locked Backdrop" and "Godiva Close" from
  <https://www.hollywoodcamerawork.com/green-screen-plates.html>. Unzip the PNG sequences into
  `footage/hcw/Locked Backdrop/` and `footage/hcw/Godiva Close/` at the repository root.

Nothing in this folder redistributes CorridorKey or its weights. `setup.ps1` fetches both from
their official sources into the git-ignored `bench/vendor/`. CorridorKey and its weights are
licensed CC BY-NC-SA 4.0 by Corridor Digital.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File bench\setup.ps1
```

The script is idempotent. It:

- clones CorridorKey at the pinned commit;
- downloads `CorridorKey_v1.0.safetensors` from Hugging Face at a pinned revision;
- runs `uv sync` (exact pins in `pyproject.toml` / `uv.lock`);
- checks CUDA and NVENC;
- warns if the reference plates are missing.

## Running

From `bench/`:

```powershell
uv run pytest                      # unit tests for metrics, hint, statistics, envelope/verdict
uv run ckbench env --new-run       # new results run; records GPU, driver, versions, SHAs
uv run ckbench speed --runtime torch --load idle --camera 1080p
uv run ckbench export              # ONNX + TensorRT fp16 engines per size (slow at 2048)
uv run ckbench speed --runtime trt --load idle --camera 1080p
uv run ckbench clips               # normalise reference clips to 1080p30
uv run ckbench quality             # metrics + comparison material in ../outputs/compare/
uv run ckbench review --config <id> --clip <id> --verdict pass|fail [--reason "..."]
uv run ckbench speed --runtime torch,trt --load nvenc --camera 720p,1080p,2160p
uv run ckbench report              # REPORT.md (refuses while owner reviews are pending)
```

`uv run ckbench all` runs the whole chain in one go and writes a draft report.

Exit codes: `0` success; `1` failed precondition; `2` partial (some configurations OOM'd or failed
to export; recorded in `results.json`).

## Confirming on another GPU (e.g. the RTX A2000)

1. Install the prerequisites above, clone this repository and copy in the HCW plates.
2. Run `bench\setup.ps1`.
3. Run `uv run ckbench all`. Engines are built for *that* GPU, because TensorRT engines are not
   portable between GPU models.
4. Compare its `results/<run-id>/REPORT.md` with the RTX 3060 report. Its tables use the same
   format.

## Layout

| Path | What |
|---|---|
| `ckbench/` | harness source (`cli.py` is the entry point) |
| `tests/` | pytest unit tests for the deterministic parts |
| `clips.toml` | reference clip manifest: source, licence, segment |
| `results/<run-id>/` | committed `results.json` + `REPORT.md` |
| `vendor/` | CorridorKey checkout + weights (git-ignored) |
| `../outputs/` | ONNX, engines, normalised clips, comparison videos (git-ignored) |
