# CorridorKey-OBS

Live neural green-screen keying for [OBS Studio](https://obsproject.com/), built on
[CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer / Corridor Digital.

> **Status: feasibility spike complete. Verdict: CONDITIONAL-GO.** CorridorKey was designed for
> offline VFX work. The spike measured whether it can key live video on an RTX 3060, with a
> TensorRT fp16 engine and a concurrent 1080p30 NVENC encode:
>
> - **1080p30 gate not met.** The gate needs ≤ 20 ms/frame with both internal sides ≥ 768. The best
>   eligible configuration (768×768) takes 32.8 ms p95.
> - **1080p30 is still sustained** without that headroom: 1024×576 internal, 30.1 ms p95 end-to-end
>   (33.3 ms budget).
> - **Supported live modes on the RTX 3060:** 720p20 through 2160p30, with smaller internal sizes
>   for the faster modes. 2160p60 is not supported.
> - **RTX A2000:** not measured. The *estimate* supports up to 1080p30 at 768×448, which must be
>   confirmed on the card.
>
> Full report, numbers and method:
> [`bench/results/20261016T221252Z-c16425/REPORT.md`](bench/results/20261016T221252Z-c16425/REPORT.md).
> The harness is in [`bench/`](bench/). It reproduces the results on any NVIDIA GPU, including the A2000.

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
