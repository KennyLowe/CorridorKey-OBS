# CorridorKey-OBS

Live neural green-screen keying for [OBS Studio](https://obsproject.com/), built on
[CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer / Corridor Digital.

> **Status: v0.1.0-rc.4 owner test build of the OBS filter** (feature 002). `plugin/` is a native
> OBS Studio 32 filter, "CorridorKey". It keys a green-screen source live with a TensorRT engine on a
> worker thread, falls back to a colour key whenever keying can't keep up, and downloads the official
> weights and prepares the engine without leaving OBS. On the RTX 3060 at 1080p30 with an NVENC
> recording running:
>
> - OBS holds 30.00 fps with the filter, with 0 render-skipped frames.
> - Keying takes about 26 ms per frame at 1024×576.
> - The key adds 1 frame of latency, measured on the recorded output.
>
> Install guide: [`docs/INSTALL.md`](docs/INSTALL.md). Test results:
> [`plugin/TEST-RESULTS.md`](plugin/TEST-RESULTS.md). The RTX A2000 (production card) is not measured
> yet: [`docs/A2000-RUN.md`](docs/A2000-RUN.md).
>
> **Feasibility spike (feature 001): CONDITIONAL-GO.** On an RTX 3060 with a concurrent 1080p30
> NVENC encode:
>
> - The 1080p30 gate (≤ 20 ms with both internal sides ≥ 768) is not met: 768×768 takes 32.8 ms p95.
> - 1080p30 is still sustained at 1024×576 (30.1 ms p95 of a 33.3 ms budget).
> - Report: [`bench/results/20261016T221252Z-c16425/REPORT.md`](bench/results/20261016T221252Z-c16425/REPORT.md).
>   The harness in [`bench/`](bench/) reproduces it on any NVIDIA GPU.

## Goals

- An OBS video filter that keys a green-screen source with CorridorKey in real time.
- Inference off OBS's render thread, with graceful fallback so the stream never stalls.
- NVIDIA GPUs on Windows first (TensorRT, engines built on your own GPU).

## Model weights

This repository does **not** include CorridorKey weights. Get them from the
[official CorridorKey repository](https://github.com/nikopueringer/CorridorKey) under its license.

## Licensing

The repository is licensed by part:

| Part | Licence | File |
|---|---|---|
| `plugin/` (the OBS filter; links OBS's GPL `libobs`) | GPL-2.0-or-later | [`plugin/LICENSE`](plugin/LICENSE) |
| `bench/` (benchmark and model-preparation helper), docs and results | CC BY-NC-SA 4.0 | [`LICENSE`](LICENSE) |
| `models/manifest.json` (pins the official weights: Hugging Face repo, revision, sha256) | CC BY-NC-SA 4.0 | [`LICENSE`](LICENSE) |
| CorridorKey weights | their own licence (CC BY-NC-SA 4.0 + Corridor Digital's terms) | not in this repository |

`plugin/` contains no CorridorKey code or weights. It loads a TensorRT engine that the user
prepares on their own GPU from weights they downloaded themselves, using the `bench/` helper.
The weights are never shipped or downloaded by this project.

## Attribution

CorridorKey is © Corridor Digital / Niko Pueringer and licensed under CC BY-NC-SA 4.0 with
additional terms from its author. This project is free and non-commercial. It may not be sold or
offered as a paid inference service. This project is not affiliated with or endorsed by Corridor
Digital.
