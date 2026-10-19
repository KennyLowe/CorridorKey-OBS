> **DRAFT — owner review pending.** Quality verdicts still `pending` are treated as passes below, so this verdict is provisional.

# CorridorKey live-keying feasibility: CONDITIONAL-GO

Run `20261019T122416Z-e45fdf`, NVIDIA RTX A2000 12GB, driver 597.16.

**Verdict: CONDITIONAL-GO.** The 1080p30 gate (<= 20 ms p95 at internal >= 768 on both sides, under encode load, quality accepted) is not met. The target mode, 1080p30, is sustained with trt-fp16-768x448-ref0-nvenc1080p30 at 26.1 ms p95 (budget 33.3 ms) but without the gate's headroom. Supported modes on the NVIDIA RTX A2000 12GB: 720p20, 720p30, 1080p30.

**A2000 (ESTIMATE):** A2000 ESTIMATE for 1080p30: 25.9-33.0 ms vs 33.3 ms budget: supported. Must be confirmed on the A2000.

> Quality verdicts borrowed from run 20261016T221252Z-c16425 (NVIDIA GeForce RTX 3060), matched by configuration: the model and its outputs are the same on any GPU; only speed is measured here.

## Prior evidence

Before this spike, no frame rates had been published by CorridorKey's authors. Third-party ports reported, unverified: an After Effects port on an RTX 4090 (fp16, 1080p source, including I/O and post-processing) at about 187 ms/frame at 512 without the refiner, 230 ms with it, 286 ms at 1024 and 612 ms at 2048; and a Nuke TensorRT port on an RTX A5000 (fp16, 2048) at about 300 ms/frame. Those figures made the 1080p30 gate unlikely before any measurement here; they include unknown host overhead and none were taken on this hardware.

## Gate

**Gate (spec, Constitution I and II):** on the RTX 3060, some configuration at internal resolution >= 768 (both sides) processes a 1080p frame end-to-end in <= 20 ms (p95, worst of 3 runs) while a sustained 1080p30 NVENC encode runs on the same GPU, with owner-accepted quality and flicker on every reference clip. The margin below the 33.3 ms frame is headroom for the slower A2000 and for OBS's own work. "Internal resolution >= 768" means both sides are >= 768 (owner decision, 2026-10-17): non-square settings such as 768x448 cannot meet the gate, but can still support lower live modes in the envelope (CONDITIONAL-GO).

## Operating envelope

Each mode's budget is 1000/fps ms of end-to-end time under the encode load, with no extra margin. A2000 columns are an **ESTIMATE (x1.25-x1.59 of the 3060)**: the 3060 time scaled by the memory-bandwidth ratio (360/288 GB/s, optimistic) to the FP32 throughput ratio (12.7/8.0 TFLOPS, conservative). The A2000 counts as supported only when the conservative bound fits, and as borderline when the range straddles the budget. Once a mode fails, heavier modes at the same or higher fps are not attempted. A mode's best configuration is the one with the largest internal pixel count that fits with quality passed (ties: lowest p95); the refiner setting is not ranked. When nothing fits, the fastest configuration is shown. Load runs (T032) covered every idle-ok configuration whose idle 1080p e2e median was <= 60 ms (the loosest budget, 50 ms, + 20%); the rest are recorded as not attempted - exceeds budget.

| mode | budget ms | NVIDIA RTX A2000 12GB (measured) | best config | p95 ms | quality | A2000 ESTIMATE | A2000 range ms |
|---|---|---|---|---|---|---|---|
| 720p20 | 50.0 | supported | trt-fp16-1024x576-ref0-nvenc1080p30 | 43.0 | pass | supported (trt-fp16-768x448-ref0-nvenc1080p30) | 29.8-37.9 |
| 720p30 | 33.3 | supported | trt-fp16-768x448-ref0-nvenc1080p30 | 23.9 | pass | supported (trt-fp16-512x512-ref0-nvenc1080p30) | 23.7-30.2 |
| 1080p30 | 33.3 | supported | trt-fp16-768x448-ref0-nvenc1080p30 | 26.1 | pass | supported (trt-fp16-512x512-ref0-nvenc1080p30) | 25.9-33.0 |
| 1080p60 | 16.7 | unsupported (no configuration within budget) | trt-fp16-512x512-ref0-nvenc1080p30 | 20.8 | pass | unsupported (trt-fp16-512x512-ref0-nvenc1080p30) | 25.9-33.0 |
| 2160p30 | 33.3 | unsupported (no configuration within budget) | trt-fp16-512x512-ref0-nvenc1080p30 | 35.3 | pass | unsupported (trt-fp16-512x512-ref0-nvenc1080p30) | 44.1-56.1 |
| 2160p60 | 16.7 | not_attempted (a lighter mode at the same or lower fps already failed) | - | - | - | not_attempted | - |

## Speed and memory

Model time is CUDA-event time of the network alone. End-to-end time is wall-clock per frame: pinned upload of a uint8 camera frame, chroma-key hint, resize to internal size, network, resize back, RGBA8 pack and download. Each configuration was run 3 times (50 warm-up and 300 timed frames per run); spread is the max-min of run medians as a % of the min (flagged above 10%, SC-004).

| runtime | internal | refiner | load | camera | status | model median / p95 ms | e2e median / p95 ms (worst run) | peak VRAM MiB | spread % | clock start/end MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| torch | 512x512 | off | idle | 1080p | ok | 34.0 / 36.8 | 39.5 / 42.5 | 783 | 2.0 | 780/1200 |
| torch | 512x512 | on | idle | 1080p | ok | 86.1 / 98.0 | 91.4 / 103.8 | 1053 | 3.4 | 1200/1320 |
| torch | 768x448 | off | idle | 1080p | ok | 43.8 / 54.8 | 49.6 / 62.9 | 1046 | 3.8 | 780/1185 |
| torch | 768x448 | on | idle | 1080p | ok | 113.8 / 120.5 | 119.4 / 125.9 | 1368 | 0.3 | 780/1200 |
| torch | 768x768 | off | idle | 1080p | ok | 79.1 / 81.9 | 85.3 / 88.3 | 1239 | 0.9 | 780/1140 |
| torch | 1024x576 | off | idle | 1080p | ok | 76.7 / 80.2 | 82.9 / 87.3 | 1376 | 0.3 | 780/1117 |
| torch | 768x768 | on | idle | 1080p | ok | 206.6 / 221.6 | 212.7 / 228.7 | 1880 | 0.5 | 780/1185 |
| torch | 1024x576 | on | idle | 1080p | ok | 199.8 / 203.7 | 205.7 / 209.6 | 1956 | 0.2 | 487/1140 |
| torch | 1024x1024 | off | idle | 1080p | ok | 156.3 / 159.8 | 163.2 / 166.7 | 1823 | 0.2 | 780/1125 |
| torch | 1024x1024 | on | idle | 1080p | ok | 381.1 / 387.5 | 387.7 / 394.1 | 2872 | 0.4 | 780/1117 |
| torch | 2048x2048 | off | idle | 1080p | ok | 1127.3 / 1184.4 | 1138.5 / 1195.8 | 5874 | 1.0 | 555/1095 |
| torch | 2048x2048 | on | idle | 1080p | ok | 2040.0 / 2204.9 | 2050.7 / 2217.6 | 9949 | 2.4 | 637/1200 |
| trt | 512x512 | off | idle | 1080p | ok | 15.5 / 18.1 | 21.1 / 24.4 | 759 | 1.8 | 780/1177 |
| trt | 512x512 | on | idle | 1080p | ok | 54.1 / 62.0 | 59.9 / 67.8 | 1071 | 1.7 | 360/1102 |
| trt | 768x448 | off | idle | 1080p | ok | 20.3 / 21.4 | 25.8 / 26.9 | 737 | 0.2 | 780/1117 |
| trt | 768x448 | on | idle | 1080p | ok | 38.3 / 43.8 | 44.1 / 49.9 | 802 | 9.5 | 780/1080 |
| trt | 768x768 | off | idle | 1080p | ok | 42.2 / 44.2 | 48.7 / 50.4 | 864 | 1.0 | 780/1162 |
| trt | 1024x576 | off | idle | 1080p | ok | 41.7 / 44.1 | 48.4 / 50.2 | 873 | 0.7 | 780/1170 |
| trt | 768x768 | on | idle | 1080p | ok | 75.7 / 79.7 | 83.0 / 86.7 | 978 | 1.9 | 780/1110 |
| trt | 1024x576 | on | idle | 1080p | ok | 75.9 / 78.8 | 82.6 / 85.1 | 976 | 0.6 | 780/1117 |
| trt | 1024x1024 | off | idle | 1080p | ok | 86.4 / 89.7 | 94.5 / 97.1 | 1108 | 0.5 | 780/1095 |
| trt | 1024x1024 | on | idle | 1080p | ok | 148.8 / 161.0 | 156.2 / 169.5 | 1313 | 12.2 **FLAG** | 780/1050 |
| trt | 2048x2048 | off | idle | 1080p | ok | 739.0 / 748.7 | 751.7 / 761.5 | 2923 | 0.2 | 780/1125 |
| trt | 2048x2048 | on | idle | 1080p | ok | 990.9 / 1001.7 | 1003.5 / 1015.1 | 3656 | 0.1 | 780/1110 |
| torch | 512x512 | off | nvenc1080p30 | 1080p | ok | 32.8 / 33.8 | 37.9 / 39.2 | 2399 | 0.3 | 1200/1185 |
| torch | 512x512 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.4 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 1080p | ok | 42.0 / 43.1 | 47.5 / 48.5 | 2451 | 0.2 | 1200/1170 |
| torch | 768x448 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 119.4 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.3 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.9 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 212.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 205.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 163.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 387.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1138.5 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 2050.7 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 512x512 | off | nvenc1080p30 | 2160p | ok | 32.6 / 33.1 | 52.2 / 53.0 | 2711 | 0.2 | 1200/1200 |
| torch | 512x512 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.4 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 2160p | ok | 41.3 / 42.5 | 60.8 / 61.3 | 2770 | 0.1 | 1200/1192 |
| torch | 768x448 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 119.4 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.3 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.9 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 212.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 205.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 163.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 387.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1138.5 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 2050.7 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 512x512 | off | nvenc1080p30 | 720p | ok | 32.8 / 35.8 | 35.5 / 39.7 | 2430 | 0.5 | 1200/1185 |
| torch | 512x512 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.4 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 720p | ok | 42.1 / 43.2 | 45.0 / 46.1 | 2437 | 0.2 | 1200/1155 |
| torch | 768x448 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 119.4 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.3 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.9 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 212.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 205.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 163.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 387.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1138.5 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 2050.7 ms is above 60 ms (loosest bu | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 1080p | ok | 14.4 / 15.3 | 20.3 / 20.8 | 2205 | 0.7 | 1200/1147 |
| trt | 512x512 | on | nvenc1080p30 | 1080p | ok | 27.2 / 30.0 | 32.5 / 35.8 | 2308 | 2.1 | 1200/1185 |
| trt | 768x448 | off | nvenc1080p30 | 1080p | ok | 20.2 / 20.7 | 25.6 / 26.1 | 2249 | 0.6 | 1200/1147 |
| trt | 768x448 | on | nvenc1080p30 | 1080p | ok | 38.2 / 39.4 | 43.8 / 45.0 | 2311 | 0.4 | 1200/1095 |
| trt | 768x768 | off | nvenc1080p30 | 1080p | ok | 38.1 / 39.3 | 44.1 / 45.3 | 2389 | 0.4 | 1200/1057 |
| trt | 1024x576 | off | nvenc1080p30 | 1080p | ok | 38.0 / 39.2 | 44.0 / 45.1 | 2394 | 0.6 | 1200/1087 |
| trt | 768x768 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 83.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x576 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.6 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 94.5 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 156.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 751.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1003.5 ms is above 60 ms (loosest bu | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 2160p | ok | 14.2 / 15.3 | 34.1 / 35.3 | 2663 | 1.2 | 1200/1185 |
| trt | 512x512 | on | nvenc1080p30 | 2160p | ok | 27.0 / 27.6 | 46.6 / 48.0 | 2709 | 1.9 | 1200/1162 |
| trt | 768x448 | off | nvenc1080p30 | 2160p | ok | 19.7 / 20.2 | 38.6 / 39.8 | 2705 | 0.2 | 1200/1192 |
| trt | 768x448 | on | nvenc1080p30 | 2160p | ok | 37.1 / 38.2 | 57.1 / 57.9 | 2767 | 0.2 | 1200/1087 |
| trt | 768x768 | off | nvenc1080p30 | 2160p | ok | 36.8 / 38.0 | 56.9 / 57.7 | 2825 | 0.2 | 1200/1140 |
| trt | 1024x576 | off | nvenc1080p30 | 2160p | ok | 36.9 / 38.0 | 56.8 / 57.7 | 2812 | 0.2 | 1200/1147 |
| trt | 768x768 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 83.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x576 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.6 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 94.5 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 156.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 751.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1003.5 ms is above 60 ms (loosest bu | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 720p | ok | 14.5 / 15.6 | 17.8 / 19.0 | 2117 | 0.8 | 1200/1140 |
| trt | 512x512 | on | nvenc1080p30 | 720p | ok | 26.9 / 29.4 | 29.9 / 34.0 | 2252 | 2.6 | 1200/1185 |
| trt | 768x448 | off | nvenc1080p30 | 720p | ok | 20.5 / 20.9 | 23.4 / 23.9 | 2153 | 0.3 | 1200/1125 |
| trt | 768x448 | on | nvenc1080p30 | 720p | ok | 38.2 / 39.4 | 41.3 / 42.5 | 2194 | 0.4 | 1200/1080 |
| trt | 768x768 | off | nvenc1080p30 | 720p | ok | 38.3 / 39.7 | 41.8 / 43.2 | 2302 | 0.6 | 1200/1050 |
| trt | 1024x576 | off | nvenc1080p30 | 720p | ok | 38.1 / 39.3 | 41.6 / 43.0 | 2302 | 0.2 | 1200/1072 |
| trt | 768x768 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 83.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x576 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 82.6 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 94.5 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 1024x1024 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 156.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 751.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1003.5 ms is above 60 ms (loosest bu | - | - | - | - | - |

### Slowdown under the NVENC encode load

| configuration | camera | idle e2e median ms | load e2e median ms | slowdown |
|---|---|---|---|---|
| trt-fp16-512x512-ref1 | 1080p | 59.9 | 32.5 | -45.8% |
| trt-fp16-512x512-ref0 | 1080p | 21.1 | 20.3 | -4.1% |
| trt-fp16-768x768-ref0 | 1080p | 48.7 | 44.1 | -9.3% |
| trt-fp16-1024x576-ref0 | 1080p | 48.4 | 44.0 | -9.2% |
| trt-fp16-768x448-ref1 | 1080p | 44.1 | 43.8 | -0.8% |
| trt-fp16-768x448-ref0 | 1080p | 25.8 | 25.6 | -0.5% |
| torch-fp16-512x512-ref0 | 1080p | 39.5 | 37.9 | -3.9% |
| torch-fp16-768x448-ref0 | 1080p | 49.6 | 47.5 | -4.2% |

## TensorRT conversion and drift

| internal | refiner | ONNX | ONNX peak RAM GiB | engine | build s | engine MiB | drift alpha MAE / max | drift fg MAE / max | drift fg MAE / max where alpha > 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 512x512 | off | ok | 2.1 | ok | 103 | 141 | 0.0002 / 0.020 | 0.0016 / 0.256 | 0.0002 / 0.012 |
| 512x512 | on | ok | 2.4 | ok | 156 | 276 | 0.0000 / 0.005 | 0.0002 / 0.060 | 0.0002 / 0.003 |
| 768x448 | off | ok | 2.2 | ok | 110 | 145 | 0.0002 / 0.023 | 0.0015 / 0.793 | 0.0002 / 0.011 |
| 768x448 | on | ok | 2.4 | ok | 163 | 144 | 0.0001 / 0.008 | 0.0005 / 0.121 | 0.0003 / 0.004 |
| 768x768 | off | ok | 2.9 | ok | 117 | 146 | 0.0002 / 0.021 | 0.0016 / 0.826 | 0.0003 / 0.015 |
| 1024x576 | off | ok | 2.9 | ok | 131 | 149 | 0.0001 / 0.022 | 0.0015 / 0.948 | 0.0002 / 0.008 |
| 768x768 | on | ok | 3.2 | ok | 201 | 147 | 0.0001 / 0.010 | 0.0005 / 0.143 | 0.0003 / 0.005 |
| 1024x576 | on | ok | 3.2 | ok | 192 | 147 | 0.0001 / 0.011 | 0.0003 / 0.148 | 0.0003 / 0.008 |
| 1024x1024 | off | ok | 3.5 | ok | 148 | 153 | 0.0001 / 0.019 | 0.0014 / 0.942 | 0.0002 / 0.005 |
| 1024x1024 | on | ok | 3.7 | ok | 260 | 154 | 0.0001 / 0.010 | 0.0003 / 0.132 | 0.0002 / 0.007 |
| 2048x2048 | off | ok | 6.8 | ok | 462 | 195 | 0.0001 / 0.006 | 0.0015 / 0.913 | 0.0002 / 0.006 |
| 2048x2048 | on | ok | 7.8 | ok | 873 | 196 | 0.0001 / 0.005 | 0.0001 / 0.103 | 0.0002 / 0.005 |

Drift is |TensorRT - PyTorch fp16| of the raw network outputs at the internal size, over 20 fixed frames (synthetic frames seeds 0-19).

## Reference clips

| clip | focus | native | frames | hint (border mean) | licence | source |
|---|---|---|---|---|---|---|

Clips are fetched from their source (or downloaded by hand for the Hollywood Camera Work plates), never committed or rehosted, and normalised to 1920x1080 at 30 fps (aspect kept, padded with screen green), up to 300 frames. No stills derived from them are committed.

## Quality and flicker

Against the PyTorch 2048 refiner-on reference on the same frames. Flag thresholds were declared before any quality run: edge alpha MAE > 0.03, edge FG MAE > 0.04, flicker excess > 0.01. Flags only direct attention; the owner's visual verdict decides, and passing a flagged entry needs a written reason.

| config | clip | edge alpha MAE | edge FG MAE | flicker excess | flags | owner verdict | reason |
|---|---|---|---|---|---|---|---|
| torch-fp16-1024x1024-ref0-idle | hcw-godiva-close | 0.0603 | 0.0390 | -0.0067 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref0-idle | hcw-locked-backdrop | 0.0385 | 0.0143 | -0.0048 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref0-idle | pexels-athlete | 0.0465 | 0.0143 | -0.0104 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref0-idle | pexels-dance | 0.0329 | 0.0202 | -0.0051 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref0-idle | pixabay-hair | 0.0369 | 0.0356 | -0.0009 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref1-idle | hcw-godiva-close | 0.0534 | 0.0270 | -0.0041 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref1-idle | hcw-locked-backdrop | 0.0231 | 0.0103 | -0.0027 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref1-idle | pexels-athlete | 0.0450 | 0.0087 | -0.0127 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref1-idle | pexels-dance | 0.0224 | 0.0097 | -0.0041 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x1024-ref1-idle | pixabay-hair | 0.0272 | 0.0238 | 0.0008 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref0-idle | hcw-godiva-close | 0.0647 | 0.0413 | -0.0089 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref0-idle | hcw-locked-backdrop | 0.0413 | 0.0157 | -0.0057 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref0-idle | pexels-athlete | 0.0579 | 0.0157 | -0.0158 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref0-idle | pexels-dance | 0.0432 | 0.0220 | -0.0080 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref0-idle | pixabay-hair | 0.0420 | 0.0415 | -0.0020 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref1-idle | hcw-godiva-close | 0.0567 | 0.0291 | -0.0056 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref1-idle | hcw-locked-backdrop | 0.0248 | 0.0115 | -0.0035 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref1-idle | pexels-athlete | 0.0563 | 0.0103 | -0.0178 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref1-idle | pexels-dance | 0.0304 | 0.0119 | -0.0062 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-1024x576-ref1-idle | pixabay-hair | 0.0315 | 0.0270 | 0.0003 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref0-idle | hcw-godiva-close | 0.0195 | 0.0209 | -0.0018 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref0-idle | hcw-locked-backdrop | 0.0204 | 0.0070 | -0.0022 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref0-idle | pexels-athlete | 0.0161 | 0.0090 | 0.0025 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref0-idle | pexels-dance | 0.0137 | 0.0133 | -0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref0-idle | pixabay-hair | 0.0178 | 0.0195 | -0.0009 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref1-idle | hcw-godiva-close | 0.0000 | 0.0000 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref1-idle | hcw-locked-backdrop | 0.0000 | 0.0000 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref1-idle | pexels-athlete | 0.0000 | 0.0000 | 0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref1-idle | pexels-dance | 0.0000 | 0.0000 | 0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-2048x2048-ref1-idle | pixabay-hair | 0.0000 | 0.0000 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref0-idle | hcw-godiva-close | 0.1214 | 0.0546 | -0.0135 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref0-idle | hcw-locked-backdrop | 0.0563 | 0.0208 | -0.0070 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref0-idle | pexels-athlete | 0.0929 | 0.0204 | -0.0300 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-512x512-ref0-idle | pexels-dance | 0.0746 | 0.0279 | -0.0122 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-512x512-ref0-idle | pixabay-hair | 0.0687 | 0.0620 | -0.0036 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref1-idle | hcw-godiva-close | 0.1032 | 0.0430 | -0.0088 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref1-idle | hcw-locked-backdrop | 0.0375 | 0.0167 | -0.0055 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-512x512-ref1-idle | pexels-athlete | 0.0909 | 0.0141 | -0.0315 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-512x512-ref1-idle | pexels-dance | 0.0529 | 0.0169 | -0.0091 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-512x512-ref1-idle | pixabay-hair | 0.0445 | 0.0468 | -0.0007 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref0-idle | hcw-godiva-close | 0.0857 | 0.0474 | -0.0114 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref0-idle | hcw-locked-backdrop | 0.0504 | 0.0177 | -0.0067 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref0-idle | pexels-athlete | 0.0763 | 0.0181 | -0.0238 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x448-ref0-idle | pexels-dance | 0.0597 | 0.0249 | -0.0109 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x448-ref0-idle | pixabay-hair | 0.0535 | 0.0515 | -0.0029 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref1-idle | hcw-godiva-close | 0.0739 | 0.0358 | -0.0074 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref1-idle | hcw-locked-backdrop | 0.0322 | 0.0136 | -0.0048 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x448-ref1-idle | pexels-athlete | 0.0750 | 0.0124 | -0.0256 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x448-ref1-idle | pexels-dance | 0.0425 | 0.0145 | -0.0083 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x448-ref1-idle | pixabay-hair | 0.0373 | 0.0351 | -0.0000 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref0-idle | hcw-godiva-close | 0.0802 | 0.0450 | -0.0092 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref0-idle | hcw-locked-backdrop | 0.0468 | 0.0162 | -0.0058 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref0-idle | pexels-athlete | 0.0634 | 0.0166 | -0.0178 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x768-ref0-idle | pexels-dance | 0.0465 | 0.0228 | -0.0077 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x768-ref0-idle | pixabay-hair | 0.0485 | 0.0450 | -0.0017 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref1-idle | hcw-godiva-close | 0.0695 | 0.0338 | -0.0057 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref1-idle | hcw-locked-backdrop | 0.0298 | 0.0126 | -0.0039 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| torch-fp16-768x768-ref1-idle | pexels-athlete | 0.0625 | 0.0109 | -0.0198 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x768-ref1-idle | pexels-dance | 0.0322 | 0.0124 | -0.0060 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| torch-fp16-768x768-ref1-idle | pixabay-hair | 0.0340 | 0.0328 | 0.0005 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref0-idle | hcw-godiva-close | 0.0603 | 0.0390 | -0.0067 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref0-idle | hcw-locked-backdrop | 0.0385 | 0.0142 | -0.0048 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref0-idle | pexels-athlete | 0.0465 | 0.0143 | -0.0104 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref0-idle | pexels-dance | 0.0329 | 0.0202 | -0.0051 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref0-idle | pixabay-hair | 0.0368 | 0.0357 | -0.0009 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref1-idle | hcw-godiva-close | 0.0533 | 0.0270 | -0.0041 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref1-idle | hcw-locked-backdrop | 0.0231 | 0.0103 | -0.0027 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref1-idle | pexels-athlete | 0.0450 | 0.0087 | -0.0126 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref1-idle | pexels-dance | 0.0224 | 0.0097 | -0.0041 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x1024-ref1-idle | pixabay-hair | 0.0273 | 0.0239 | 0.0008 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref0-idle | hcw-godiva-close | 0.0647 | 0.0413 | -0.0089 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref0-idle | hcw-locked-backdrop | 0.0413 | 0.0157 | -0.0057 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref0-idle | pexels-athlete | 0.0579 | 0.0157 | -0.0158 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref0-idle | pexels-dance | 0.0432 | 0.0220 | -0.0080 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref0-idle | pixabay-hair | 0.0420 | 0.0415 | -0.0020 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref1-idle | hcw-godiva-close | 0.0566 | 0.0291 | -0.0056 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref1-idle | hcw-locked-backdrop | 0.0248 | 0.0115 | -0.0035 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref1-idle | pexels-athlete | 0.0562 | 0.0103 | -0.0177 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref1-idle | pexels-dance | 0.0303 | 0.0119 | -0.0062 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-1024x576-ref1-idle | pixabay-hair | 0.0315 | 0.0270 | 0.0003 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref0-idle | hcw-godiva-close | 0.0194 | 0.0209 | -0.0018 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref0-idle | hcw-locked-backdrop | 0.0204 | 0.0069 | -0.0022 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref0-idle | pexels-athlete | 0.0162 | 0.0090 | 0.0025 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref0-idle | pexels-dance | 0.0138 | 0.0133 | -0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref0-idle | pixabay-hair | 0.0178 | 0.0195 | -0.0009 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref1-idle | hcw-godiva-close | 0.0003 | 0.0003 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref1-idle | hcw-locked-backdrop | 0.0002 | 0.0001 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref1-idle | pexels-athlete | 0.0003 | 0.0002 | 0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref1-idle | pexels-dance | 0.0002 | 0.0001 | 0.0000 | - | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-2048x2048-ref1-idle | pixabay-hair | 0.0002 | 0.0003 | 0.0000 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref0-idle | hcw-godiva-close | 0.1214 | 0.0546 | -0.0135 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref0-idle | hcw-locked-backdrop | 0.0563 | 0.0208 | -0.0070 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref0-idle | pexels-athlete | 0.0929 | 0.0203 | -0.0299 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-512x512-ref0-idle | pexels-dance | 0.0746 | 0.0279 | -0.0122 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-512x512-ref0-idle | pixabay-hair | 0.0688 | 0.0619 | -0.0036 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref1-idle | hcw-godiva-close | 0.1032 | 0.0430 | -0.0087 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref1-idle | hcw-locked-backdrop | 0.0375 | 0.0167 | -0.0055 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-512x512-ref1-idle | pexels-athlete | 0.0908 | 0.0141 | -0.0315 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-512x512-ref1-idle | pexels-dance | 0.0529 | 0.0169 | -0.0091 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-512x512-ref1-idle | pixabay-hair | 0.0445 | 0.0466 | -0.0007 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref0-idle | hcw-godiva-close | 0.0857 | 0.0474 | -0.0114 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref0-idle | hcw-locked-backdrop | 0.0504 | 0.0176 | -0.0067 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref0-idle | pexels-athlete | 0.0763 | 0.0181 | -0.0238 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x448-ref0-idle | pexels-dance | 0.0597 | 0.0249 | -0.0108 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x448-ref0-idle | pixabay-hair | 0.0532 | 0.0519 | -0.0029 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref1-idle | hcw-godiva-close | 0.0738 | 0.0358 | -0.0074 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref1-idle | hcw-locked-backdrop | 0.0321 | 0.0136 | -0.0048 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x448-ref1-idle | pexels-athlete | 0.0750 | 0.0124 | -0.0255 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x448-ref1-idle | pexels-dance | 0.0425 | 0.0145 | -0.0083 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x448-ref1-idle | pixabay-hair | 0.0373 | 0.0352 | -0.0000 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref0-idle | hcw-godiva-close | 0.0802 | 0.0450 | -0.0092 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref0-idle | hcw-locked-backdrop | 0.0468 | 0.0162 | -0.0058 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref0-idle | pexels-athlete | 0.0634 | 0.0166 | -0.0178 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x768-ref0-idle | pexels-dance | 0.0465 | 0.0228 | -0.0077 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x768-ref0-idle | pixabay-hair | 0.0485 | 0.0451 | -0.0017 | edge_alpha, edge_fg | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref1-idle | hcw-godiva-close | 0.0695 | 0.0338 | -0.0057 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref1-idle | hcw-locked-backdrop | 0.0298 | 0.0126 | -0.0039 | - | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |
| trt-fp16-768x768-ref1-idle | pexels-athlete | 0.0625 | 0.0109 | -0.0198 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x768-ref1-idle | pexels-dance | 0.0322 | 0.0124 | -0.0060 | edge_alpha | pass | Owner visual review 2026-10-19: edges look great at normal viewing size. |
| trt-fp16-768x768-ref1-idle | pixabay-hair | 0.0340 | 0.0329 | 0.0005 | edge_alpha | pass | Owner visual review 2026-10-17: edges look great at normal viewing size. |

## Added latency

Estimated delay the keyer adds = measured end-to-end median + one frame of queueing (asynchronous, off the render thread).

| mode | configuration | e2e median ms | + queue ms | added ms | frames @30 | frames @60 |
|---|---|---|---|---|---|---|
| 720p20 | trt-fp16-1024x576-ref0-nvenc1080p30 | 41.6 | 50.0 | 91.6 | 2.75 | 5.49 |
| 720p30 | trt-fp16-768x448-ref0-nvenc1080p30 | 23.4 | 33.3 | 56.7 | 1.70 | 3.40 |
| 1080p30 | trt-fp16-768x448-ref0-nvenc1080p30 | 25.6 | 33.3 | 59.0 | 1.77 | 3.54 |
| 1080p60 | trt-fp16-512x512-ref0-nvenc1080p30 | 20.3 | 16.7 | 36.9 | 1.11 | 2.22 |
| 2160p30 | trt-fp16-512x512-ref0-nvenc1080p30 | 34.1 | 33.3 | 67.4 | 2.02 | 4.05 |

## Environment

| item | value |
|---|---|
| gpu_name | NVIDIA RTX A2000 12GB |
| gpu_driver | 597.16 |
| gpu_vram_mib | 12282 |
| gpu_max_sm_clock_mhz | 2100 |
| gpu_power_limit_w | 70.0 |
| os | Windows-11-10.0.22631-SP0 |
| cpu | Intel64 Family 6 Model 143 Stepping 8, GenuineIntel |
| ram_gib | 31.2 |
| python | 3.12.15 |
| torch_cuda | 12.8 |
| cudnn | 91002 |
| tensorrt_version | 10.16.1.11 |
| ffmpeg_version | ffmpeg version 7.1.1-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers |
| corridorkey_sha | 97e55a453060745bead1befd293f6e523c4b845c |
| weights_file | CorridorKey_v1.0.safetensors |
| weights_sha256 | 74d614f7d92fc559a118c30a7deadedc3cacd8ef83dcb85a030d0bed7af8b20b |
| packages | torch 2.8.0+cu128, torchvision 0.23.0+cu128, timm 1.0.26, safetensors 0.7.0, onnx 1.23.2, tensorrt-cu12 10.16.1.11, tensorrt-cu12-libs 10.16.1.11, tensorrt-cu12-bindings 10.16.1.11, nvidia-ml-py 13.615.71, numpy 2.4.2, opencv-python-headless 4.13.0.92, jsonschema 4.26.0, psutil 7.2.2, pytest 9.1.1 |

## Deviations from the plan

- TensorRT 10.16.1.11 instead of the newest 11.x: TensorRT 11 removed the weakly-typed `BuilderFlag.FP16` build the plan specifies (only strongly-typed networks remain).
- ffmpeg 7.1.1 (Gyan build) instead of the latest: ffmpeg 8.1+ requires NVENC API 13.1 (driver >= 610) and cannot open h264_nvenc on driver 576.80.
- Speed benchmarks always use a deterministic synthetic 1080p green-screen frame (tasks.md preferred pexels-athlete frame 0 when present) so idle/load and torch/TRT entries see identical input; the networks have static shapes, so content does not change the work.
- Peak VRAM comes from NVML device-used memory minus a baseline taken before the process initialised CUDA: NVML returns no per-process memory under Windows WDDM. Other processes' activity adds noise.
- Added latency uses the measured end-to-end wall median (which includes the two resize stages) plus one frame of queueing, slightly more conservative than research D12's stage sum.
- The alpha hint for the two Pexels clips is calibrated per clip (screen colour = median of a 32 px border band; difference key normalised to that colour): they are ungraded flat footage that the fixed hint (research D8) does not recognise as green screen (owner decision, 2026-10-17). All other clips use the fixed hint. Each clip's results entry records its hint, key colour and border diagnostic.
- Square configurations use CorridorKeyEngine's own bicubic position-embedding resize; the 16:9 variants (research D2) use the same resize to a rectangular token grid.

## Limitations

- OBS's own compositing and rendering load is not simulated; the concurrent load is a 1080p30 NVENC encode only.
- Reference clips are public stock footage; results on the owner's own studio (screen, lighting, camera) may differ.
- The RTX A2000 was not measured. Every A2000 figure is an ESTIMATE scaled from the RTX 3060.
- The Hollywood Camera Work plates are below 1080p (1280x1080 and 960x720) and are upscaled for the quality run.
- The alpha hint is a simple chroma key (research D8); hint failures (e.g. green clothing) affect the key and are not corrected.
- Eager PyTorch only (no torch.compile: MSVC is not installed); upstream reports ~30% from compiling.

## Reproduce

See [`bench/README.md`](../../README.md): `bench\setup.ps1`, then `uv run ckbench all` (or the individual commands). `results.json` in this folder holds every number above.
