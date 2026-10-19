# Test results: corridorkey-obs v0.1.0-rc.1 (feature 002)

Run on 2026-10-19 on the development machine. **Every number here is from the RTX 3060.** Nothing
was measured on the RTX A2000 yet (owner tasks: `docs/A2000-RUN.md`, the SC-001 soak with
`tests/e2e/soak.py`).

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB (sm_86), driver **576.80**, HAGS on |
| CPU / RAM | Intel Xeon W-2235, 224 GB |
| OS | Windows 11 Enterprise 10.0.26200 (25H2) |
| OBS | **32.2.2** (plugin built against the OBS 31.1.1 headers pinned by obs-plugintemplate) |
| TensorRT | **10.16.1.11** (CUDA 12.9 build); engines built by `ckbench prepare` with the TensorRT Python package 10.16.1.11 |
| CUDA | 12.9.1 toolkit (nvcc 12.9.86), cudart static |
| Compiler | MSVC 14.44 (VS 2022 Build Tools 17.14), Windows SDK 10.0.22621, CMake 4.4.4 |
| Tests | Catch2 3.8.1; pytest 9.1.1; obsws-python 1.8.0 |
| e2e OBS | a private portable copy of the installed OBS 32.2.2 (`outputs/obs-portable`, `--portable`), 1920×1080 @ 30 fps, Simple output recording with NVENC H.264 6 Mb/s during the SC-002/SC-004 tests |

## Suites

| Suite | Command | Result |
|---|---|---|
| Unit (Catch2): state machine, scheduler ring, stats, cache key + manifest, Auto, helper protocol, calibration | `scripts\build.ps1` (ctest) | **53 test cases, 334 assertions, all pass** |
| GPU (Catch2): CUDA kernels vs the spike's Python, TensorRT runner parity, GPU identity, D3D11↔CUDA interop | same | **12 test cases, all pass** (ctest total 65/65) |
| bench pytest: prepare, perf-table, fetch-weights, check-models, setup network handling, second-machine robustness, 001 suites | `uv run pytest` in `bench/` | **120 passed** |
| bench pytest, opt-in network: real 399 MB weights download, sha256 matches the manifest | `uv run pytest -m network` | **1 passed** (8.7 s) |
| e2e in real OBS (pytest + obs-websocket) | `uv run --project bench pytest plugin/tests/e2e` | **15 passed** (13 min 37 s) |

After the full e2e run, two status-text strings were corrected. Those are the unverified-model Auto
note and a missing full stop in download errors. Then ctest (65/65) and the e2e scenario that shows
those texts (`test_setup.py::test_c`) were re-run and pass.

## Measurements

### SC-002: OBS frame rate with the filter (1080p30, NVENC recording running, 30 s windows)

| | active fps (mean / min) | render-skipped frames | keying ms median / p95 | filter skipped | render-thread cost per frame: median / p95 / p99 / max (last 600 frames) |
|---|---|---|---|---|---|
| filter disabled | 30.00 / 30.00 | 0 | — | — | — |
| 512_r0 | 30.00 / 30.00 | **0** | 12.4 / 13.3 | 0.0% | 0.94 / 1.42 / 1.78 / 11.6 ms |
| 1024x576_r0 (Auto on 1080p30) | 30.00 / 30.00 | **0** | 26.1 / 27.2 | 0.0% | 0.91 / 1.45 / 2.12 / 6.9 ms |

SC-002 passes (within 1%; actually identical) on the RTX 3060. The plan's render-thread target was
≤ 1 ms. The median meets it, but single calls of 7–12 ms occur, rarely (the maximum over 600 frames): map/unmap
occasionally waiting on the GPU. That is under half a frame and caused no skipped frame, but it
is above the target, and it's reported here as such.

### SC-004: forced failures (fallback default: colour key; timeout 250 ms)

| Case | Result |
|---|---|
| `debug_fail=worker_error` | fallback 67 ms after the setting changed (immediate on the first failed job); recovers in 0.3 s when cleared; 30 fps, 0 render-skipped |
| `debug_fail=stall` | fallback when the last matte was **266 ms** old (timeout 250 ms + under 1 frame); recovers in 0.4 s; 30 fps, 0 render-skipped |
| TensorRT folder missing | *Unsupported: nvinfer_10.dll not found in …*, colour-key output, OBS alive, 0 render-skipped |
| weights cleared / missing file | *Weights not set* / not found, pass-through (alpha 255 everywhere) |
| overload: two 1024_r1 instances | both behind/fallback (82% inputs skipped, newest-frame-only), OBS at 30.00 fps, 0 render-skipped |
| invalid weights file | *Invalid weights file: weights_invalid: …*, pass-through |
| download hash mismatch | specific error, nothing partial kept, pass-through |
| download offline | `network` error with retry, OBS at 30 fps, 0 render-skipped |
| OBS killed during preparation | helper tree killed with OBS (Job object); next start re-prepares cleanly; manifest valid |

GPU memory exhaustion wasn't forced as its own test. It surfaces through the same paths: engine
load failure → failed, CUDA error in a job → fallback.

### SC-003: first run and restart

- No weights, add the filter, accept the licence, Download weights (399 MB from Hugging Face,
  verified), prepare 1024x576_r0 (Auto), keying: **205 s** in the final run (179 s earlier), with
  **0 render-skipped frames and 30.00 fps throughout**. The worst render call during setup was
  4.2 ms after the gate described below.
- Restart OBS: keying again after **4.6 s** (no re-preparation).

### SC-005: added latency

Measured on OBS's NVENC-recorded program output with a frame-counter pattern, the filter enabled
mid-recording, all ~297 recorded frames decoded: the filter adds **1.0 frame** at 512_r0 and at
1024x576_r0. The Status line shows **+33 ms (1.0 frames)**. Agreement is exact (SC-005: ±1 frame).

### US6: calibration on pexels-athlete

With "Show hint", the rough mask's mean over a 32 px border (screen) is 0.97 with the fixed key and
**0.16–0.23 after calibration** (two runs at different clip positions; threshold 0.25; the spike's
Python gave 0.969 → 0.161).

### Kernels and runner vs the feature 001 Python

- hint (fixed and calibrated), preprocess (antialiased resize + ImageNet normalisation) and
  postprocess (bilinear up, RGBA8 pack) match `hint.py` / `pipeline.py` within 1e-3 (postprocess ≤ 1
  LSB).
- TensorRT 512_r0 vs the PyTorch fp16 spike output: alpha MAE **0.00020**, fg MAE (alpha > 0.1)
  **0.00018** (bound 0.001).
- Engine preparation on the 3060: 512_r0 145 s, 1024x576_r0 ~3 min, 1024_r1 322 s.

## Not tested here (owner)

- RTX A2000: benchmark (T028), SC-001 60-minute soak (T036), SC-002 on the A2000.
- SC-006: owner comparison of the live key against the 001 output (T037). Stills for review are in
  `tests/e2e/.artifacts/stills/` (git-ignored: derived from licensed footage).
