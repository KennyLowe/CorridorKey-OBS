# CorridorKey-OBS

Live neural green-screen keying for [OBS Studio](https://obsproject.com/), built on
[CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer / Corridor Digital.

> **Status: feasibility spike.** CorridorKey was designed for offline VFX work. This project is
> first measuring whether it can run fast enough for live video on consumer and workstation
> NVIDIA GPUs (RTX 3060 / RTX A2000). An OBS plugin will only be built if the benchmarks pass.

## Goals

- An OBS video filter that keys a green-screen source with CorridorKey in real time.
- Inference off OBS's render thread, with graceful fallback so the stream never stalls.
- NVIDIA GPUs on Windows first (TensorRT, engines built on your own GPU).

## Model weights

This repository does **not** include CorridorKey weights. Get them from the
[official CorridorKey repository](https://github.com/nikopueringer/CorridorKey) under its license.

## License and attribution

CorridorKey is © Corridor Digital / Niko Pueringer and licensed under CC BY-NC-SA 4.0 with
additional terms from its author. This project is a free, non-commercial derivative released under
the same license ([LICENSE](LICENSE)). It may not be sold or offered as a paid inference service.
This project is not affiliated with or endorsed by Corridor Digital.
