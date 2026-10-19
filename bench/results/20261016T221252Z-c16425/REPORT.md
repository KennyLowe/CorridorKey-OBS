# CorridorKey live-keying feasibility: CONDITIONAL-GO

Run `20261016T221252Z-c16425`, NVIDIA GeForce RTX 3060, driver 576.80.

**Verdict: CONDITIONAL-GO.** The 1080p30 gate (<= 20 ms p95 at internal >= 768 on both sides, under encode load, quality accepted) is not met. The target mode, 1080p30, is sustained with trt-fp16-1024x576-ref0-nvenc1080p30 at 30.1 ms p95 (budget 33.3 ms) but without the gate's headroom. Supported modes on the RTX 3060: 720p20, 720p30, 1080p30, 1080p60, 2160p30.

**A2000 (ESTIMATE):** A2000 ESTIMATE for 1080p30: 24.3-30.9 ms vs 33.3 ms budget: supported. Must be confirmed on the A2000.

## Prior evidence

Before this spike, no frame rates had been published by CorridorKey's authors. Third-party ports reported, unverified: an After Effects port on an RTX 4090 (fp16, 1080p source, including I/O and post-processing) at about 187 ms/frame at 512 without the refiner, 230 ms with it, 286 ms at 1024 and 612 ms at 2048; and a Nuke TensorRT port on an RTX A5000 (fp16, 2048) at about 300 ms/frame. Those figures made the 1080p30 gate unlikely before any measurement here; they include unknown host overhead and none were taken on this hardware.

## Gate

**Gate (spec, Constitution I and II):** on the RTX 3060, some configuration at internal resolution >= 768 (both sides) processes a 1080p frame end-to-end in <= 20 ms (p95, worst of 3 runs) while a sustained 1080p30 NVENC encode runs on the same GPU, with owner-accepted quality and flicker on every reference clip. The margin below the 33.3 ms frame is headroom for the slower A2000 and for OBS's own work. "Internal resolution >= 768" means both sides are >= 768 (owner decision, 2026-10-17): non-square settings such as 768x448 cannot meet the gate, but can still support lower live modes in the envelope (CONDITIONAL-GO).

## Operating envelope

Each mode's budget is 1000/fps ms of end-to-end time under the encode load, with no extra margin. A2000 columns are an **ESTIMATE (x1.25-x1.59 of the 3060)**: the 3060 time scaled by the memory-bandwidth ratio (360/288 GB/s, optimistic) to the FP32 throughput ratio (12.7/8.0 TFLOPS, conservative). The A2000 counts as supported only when the conservative bound fits, and as borderline when the range straddles the budget. Once a mode fails, heavier modes at the same or higher fps are not attempted. A mode's best configuration is the one with the largest internal pixel count that fits with quality passed (ties: lowest p95); the refiner setting is not ranked. When nothing fits, the fastest configuration is shown. Load runs (T032) covered every idle-ok configuration whose idle 1080p e2e median was <= 60 ms (the loosest budget, 50 ms, + 20%); the rest are recorded as not attempted - exceeds budget.

| mode | budget ms | RTX 3060 (measured) | best config | p95 ms | quality | A2000 ESTIMATE | A2000 range ms |
|---|---|---|---|---|---|---|---|
| 720p20 | 50.0 | supported | trt-fp16-768x768-ref0-nvenc1080p30 | 27.7 | pass | supported (trt-fp16-768x768-ref0-nvenc1080p30) | 34.6-44.0 |
| 720p30 | 33.3 | supported | trt-fp16-768x768-ref0-nvenc1080p30 | 27.7 | pass | supported (trt-fp16-768x448-ref0-nvenc1080p30) | 21.7-27.6 |
| 1080p30 | 33.3 | supported | trt-fp16-1024x576-ref0-nvenc1080p30 | 30.1 | pass | supported (trt-fp16-768x448-ref0-nvenc1080p30) | 24.3-30.9 |
| 1080p60 | 16.7 | supported | trt-fp16-512x512-ref0-nvenc1080p30 | 15.7 | pass | unsupported (trt-fp16-512x512-ref0-nvenc1080p30) | 19.6-25.0 |
| 2160p30 | 33.3 | supported | trt-fp16-512x512-ref0-nvenc1080p30 | 32.1 | pass | unsupported (trt-fp16-512x512-ref0-nvenc1080p30) | 40.1-51.0 |
| 2160p60 | 16.7 | unsupported (no configuration within budget) | trt-fp16-512x512-ref0-nvenc1080p30 | 32.1 | pass | not_attempted | - |

## Speed and memory

Model time is CUDA-event time of the network alone. End-to-end time is wall-clock per frame: pinned upload of a uint8 camera frame, chroma-key hint, resize to internal size, network, resize back, RGBA8 pack and download. Each configuration was run 3 times (50 warm-up and 300 timed frames per run); spread is the max-min of run medians as a % of the min (flagged above 10%, SC-004).

| runtime | internal | refiner | load | camera | status | model median / p95 ms | e2e median / p95 ms (worst run) | peak VRAM MiB | spread % | clock start/end MHz |
|---|---|---|---|---|---|---|---|---|---|---|
| torch | 512x512 | off | idle | 1080p | ok | 31.3 / 43.4 | 37.0 / 49.2 | 787 | 3.5 | 1837/1942 |
| torch | 512x512 | on | idle | 1080p | ok | 73.2 / 87.4 | 79.0 / 93.9 | 1068 | 1.5 | 1807/1942 |
| torch | 768x448 | off | idle | 1080p | ok | 32.4 / 47.5 | 38.2 / 53.2 | 822 | 1.0 | 1185/1935 |
| torch | 768x448 | on | idle | 1080p | ok | 85.4 / 99.5 | 91.2 / 105.3 | 1162 | 0.4 | 1522/1935 |
| torch | 1024x576 | off | idle | 1080p | ok | 56.1 / 60.2 | 62.2 / 66.7 | 1189 | 0.3 | 390/1935 |
| torch | 768x768 | off | idle | 1080p | ok | 57.0 / 59.6 | 63.1 / 65.9 | 1201 | 0.6 | 1942/1927 |
| torch | 768x768 | on | idle | 1080p | ok | 149.0 / 153.1 | 155.1 / 159.3 | 1796 | 0.7 | 495/1942 |
| torch | 1024x576 | on | idle | 1080p | ok | 147.8 / 151.1 | 153.8 / 157.3 | 1769 | 0.1 | 412/1935 |
| torch | 1024x1024 | off | idle | 1080p | ok | 108.2 / 110.2 | 114.7 / 117.5 | 1764 | 0.1 | 525/1897 |
| torch | 1024x1024 | on | idle | 1080p | ok | 271.8 / 290.2 | 278.3 / 297.9 | 3402 | 0.8 | 660/1935 |
| torch | 2048x2048 | off | idle | 1080p | ok | 778.0 / 788.3 | 787.2 / 798.1 | 5807 | 2.8 | 442/1852 |
| torch | 2048x2048 | on | idle | 1080p | ok | 1396.3 / 1431.9 | 1405.4 / 1442.5 | 9793 | 0.3 | 465/1935 |
| trt | 512x512 | off | idle | 1080p | ok | 9.0 / 9.7 | 14.6 / 15.8 | 597 | 0.4 | 1957/1942 |
| trt | 512x512 | on | idle | 1080p | ok | 17.2 / 18.0 | 22.9 / 23.9 | 647 | 0.1 | 1807/1957 |
| trt | 768x448 | off | idle | 1080p | ok | 12.4 / 12.9 | 18.1 / 18.8 | 637 | 0.0 | 1942/1927 |
| trt | 768x448 | on | idle | 1080p | ok | 23.4 / 24.7 | 29.2 / 31.0 | 701 | 0.3 | 1942/1852 |
| trt | 768x768 | off | idle | 1080p | ok | 22.5 / 23.4 | 28.5 / 29.4 | 783 | 0.2 | 1942/1875 |
| trt | 1024x576 | off | idle | 1080p | ok | 22.6 / 23.4 | 28.6 / 29.4 | 783 | 0.1 | 1942/1890 |
| trt | 768x768 | on | idle | 1080p | ok | 41.1 / 42.4 | 47.2 / 48.5 | 895 | 0.4 | 1957/1860 |
| trt | 1024x576 | on | idle | 1080p | ok | 41.4 / 42.8 | 47.5 / 49.0 | 895 | 0.0 | 1927/1867 |
| trt | 1024x1024 | off | idle | 1080p | ok | 45.7 / 46.8 | 52.2 / 53.4 | 1030 | 0.1 | 1942/1822 |
| trt | 1024x1024 | on | idle | 1080p | ok | 78.7 / 80.1 | 85.1 / 86.7 | 1207 | 0.1 | 1942/1822 |
| trt | 2048x2048 | off | idle | 1080p | ok | 394.8 / 396.1 | 404.1 / 405.4 | 2794 | 0.1 | 1942/1710 |
| trt | 2048x2048 | on | idle | 1080p | ok | 527.7 / 529.1 | 536.9 / 538.3 | 3562 | 0.1 | 1942/1725 |
| torch | 512x512 | off | nvenc1080p30 | 1080p | ok | 32.1 / 47.4 | 38.0 / 53.4 | 1006 | 4.2 | 1807/1950 |
| torch | 512x512 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 79.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 1080p | ok | 33.4 / 45.6 | 39.3 / 51.7 | 1045 | 1.8 | 1807/1950 |
| torch | 768x448 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 62.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 63.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 155.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 153.8 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 114.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 278.3 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 787.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1405.4 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 512x512 | off | nvenc1080p30 | 2160p | ok | 26.0 / 40.5 | 48.0 / 63.2 | 1296 | 1.9 | 1807/1950 |
| torch | 512x512 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 79.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 2160p | ok | 32.4 / 39.1 | 53.8 / 60.9 | 1380 | 1.8 | 1807/1950 |
| torch | 768x448 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 62.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 63.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 155.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 153.8 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 114.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 278.3 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 787.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1405.4 ms is above 60 ms (loosest bu | - | - | - | - | - |
| torch | 512x512 | off | nvenc1080p30 | 720p | ok | 36.4 / 58.2 | 39.6 / 64.2 | 1012 | 4.6 | 1807/1950 |
| torch | 512x512 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 79.0 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x448 | off | nvenc1080p30 | 720p | ok | 33.5 / 49.0 | 36.6 / 52.2 | 1036 | 1.7 | 1807/1957 |
| torch | 768x448 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 91.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 1024x576 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 62.2 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 63.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| torch | 768x768 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 155.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x576 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 153.8 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 114.7 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 1024x1024 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 278.3 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 787.2 ms is above 60 ms (loosest bud | - | - | - | - | - |
| torch | 2048x2048 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 1405.4 ms is above 60 ms (loosest bu | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 1080p | ok | 8.9 / 9.9 | 14.9 / 15.7 | 806 | 0.5 | 1807/1950 |
| trt | 512x512 | on | nvenc1080p30 | 1080p | ok | 17.9 / 20.9 | 23.7 / 28.1 | 1101 | 1.1 | 1807/1905 |
| trt | 768x448 | off | nvenc1080p30 | 1080p | ok | 12.5 / 13.4 | 18.8 / 19.5 | 817 | 0.2 | 1807/1935 |
| trt | 768x448 | on | nvenc1080p30 | 1080p | ok | 24.0 / 24.5 | 29.9 / 30.6 | 879 | 0.3 | 1807/1890 |
| trt | 768x768 | off | nvenc1080p30 | 1080p | ok | 23.1 / 23.7 | 29.3 / 32.8 | 972 | 0.5 | 1807/1897 |
| trt | 1024x576 | off | nvenc1080p30 | 1080p | ok | 23.1 / 23.9 | 29.3 / 30.1 | 949 | 0.3 | 1807/1852 |
| trt | 768x768 | on | nvenc1080p30 | 1080p | ok | 42.4 / 43.3 | 48.7 / 49.7 | 1103 | 0.2 | 1807/1912 |
| trt | 1024x576 | on | nvenc1080p30 | 1080p | ok | 42.1 / 43.0 | 48.4 / 49.4 | 1063 | 0.2 | 1807/1890 |
| trt | 1024x1024 | off | nvenc1080p30 | 1080p | ok | 46.4 / 47.3 | 53.2 / 54.1 | 1250 | 0.2 | 1807/1822 |
| trt | 1024x1024 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 404.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 1080p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 536.9 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 2160p | ok | 9.0 / 10.1 | 31.1 / 32.1 | 1266 | 0.7 | 1807/1950 |
| trt | 512x512 | on | nvenc1080p30 | 2160p | ok | 17.9 / 19.1 | 39.6 / 42.2 | 1415 | 0.3 | 1807/1905 |
| trt | 768x448 | off | nvenc1080p30 | 2160p | ok | 12.4 / 13.3 | 33.9 / 35.0 | 1271 | 0.5 | 1807/1950 |
| trt | 768x448 | on | nvenc1080p30 | 2160p | ok | 23.9 / 24.4 | 45.2 / 46.0 | 1335 | 0.1 | 1807/1897 |
| trt | 768x768 | off | nvenc1080p30 | 2160p | ok | 23.0 / 24.0 | 44.2 / 46.0 | 1428 | 0.8 | 1807/1942 |
| trt | 1024x576 | off | nvenc1080p30 | 2160p | ok | 23.1 / 23.4 | 44.1 / 45.0 | 1412 | 0.2 | 1807/1935 |
| trt | 768x768 | on | nvenc1080p30 | 2160p | ok | 41.8 / 42.9 | 63.3 / 64.3 | 1536 | 0.2 | 1807/1860 |
| trt | 1024x576 | on | nvenc1080p30 | 2160p | ok | 41.9 / 42.7 | 63.4 / 64.0 | 1509 | 0.1 | 1807/1860 |
| trt | 1024x1024 | off | nvenc1080p30 | 2160p | ok | 46.0 / 46.8 | 67.9 / 68.6 | 1659 | 0.1 | 1807/1912 |
| trt | 1024x1024 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 404.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 2160p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 536.9 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 512x512 | off | nvenc1080p30 | 720p | ok | 9.3 / 10.7 | 12.5 / 14.2 | 726 | 2.9 | 1807/1935 |
| trt | 512x512 | on | nvenc1080p30 | 720p | ok | 18.0 / 23.3 | 21.1 / 27.9 | 800 | 1.6 | 1807/1920 |
| trt | 768x448 | off | nvenc1080p30 | 720p | ok | 12.5 / 13.3 | 16.0 / 17.4 | 664 | 1.3 | 1807/1935 |
| trt | 768x448 | on | nvenc1080p30 | 720p | ok | 24.1 / 24.5 | 27.2 / 28.3 | 791 | 0.2 | 1807/1875 |
| trt | 768x768 | off | nvenc1080p30 | 720p | ok | 23.2 / 23.7 | 26.5 / 27.7 | 888 | 0.1 | 1807/1882 |
| trt | 1024x576 | off | nvenc1080p30 | 720p | ok | 23.2 / 23.7 | 26.6 / 27.7 | 865 | 0.3 | 1807/1882 |
| trt | 768x768 | on | nvenc1080p30 | 720p | ok | 42.4 / 43.4 | 45.9 / 47.3 | 1019 | 0.6 | 1807/1882 |
| trt | 1024x576 | on | nvenc1080p30 | 720p | ok | 42.3 / 43.6 | 46.3 / 50.6 | 1008 | 1.0 | 1807/1882 |
| trt | 1024x1024 | off | nvenc1080p30 | 720p | ok | 46.5 / 47.4 | 50.6 / 51.7 | 1167 | 0.4 | 1807/1875 |
| trt | 1024x1024 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 85.1 ms is above 60 ms (loosest budg | - | - | - | - | - |
| trt | 2048x2048 | off | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 404.1 ms is above 60 ms (loosest bud | - | - | - | - | - |
| trt | 2048x2048 | on | nvenc1080p30 | 720p | **not_attempted** not attempted - exceeds budget: idle 1080p e2e median 536.9 ms is above 60 ms (loosest bud | - | - | - | - | - |

### Slowdown under the NVENC encode load

| configuration | camera | idle e2e median ms | load e2e median ms | slowdown |
|---|---|---|---|---|
| torch-fp16-512x512-ref0 | 1080p | 37.0 | 38.0 | +2.8% |
| torch-fp16-768x448-ref0 | 1080p | 38.2 | 39.3 | +2.9% |
| trt-fp16-512x512-ref1 | 1080p | 22.9 | 23.7 | +3.5% |
| trt-fp16-512x512-ref0 | 1080p | 14.6 | 14.9 | +1.6% |
| trt-fp16-768x768-ref1 | 1080p | 47.2 | 48.7 | +3.2% |
| trt-fp16-768x768-ref0 | 1080p | 28.5 | 29.3 | +2.8% |
| trt-fp16-1024x1024-ref0 | 1080p | 52.2 | 53.2 | +2.0% |
| trt-fp16-1024x576-ref1 | 1080p | 47.5 | 48.4 | +2.0% |
| trt-fp16-1024x576-ref0 | 1080p | 28.6 | 29.3 | +2.2% |
| trt-fp16-768x448-ref1 | 1080p | 29.2 | 29.9 | +2.3% |
| trt-fp16-768x448-ref0 | 1080p | 18.1 | 18.8 | +3.6% |

## TensorRT conversion and drift

| internal | refiner | ONNX | ONNX peak RAM GiB | engine | build s | engine MiB | drift alpha MAE / max | drift fg MAE / max | drift fg MAE / max where alpha > 0.1 |
|---|---|---|---|---|---|---|---|---|---|
| 512x512 | off | ok | 2.0 | ok | 123 | 142 | 0.0002 / 0.034 | 0.0004 / 0.955 | 0.0003 / 0.014 |
| 512x512 | on | ok | 2.4 | ok | 176 | 143 | 0.0002 / 0.049 | 0.0002 / 0.181 | 0.0002 / 0.018 |
| 768x448 | off | ok | 2.2 | ok | 124 | 143 | 0.0002 / 0.019 | 0.0004 / 0.616 | 0.0002 / 0.010 |
| 768x448 | on | ok | 2.4 | ok | 176 | 144 | 0.0002 / 0.024 | 0.0002 / 0.198 | 0.0002 / 0.013 |
| 768x768 | off | ok | 2.9 | ok | 128 | 146 | 0.0001 / 0.009 | 0.0004 / 0.961 | 0.0003 / 0.008 |
| 1024x576 | off | ok | 2.9 | ok | 134 | 146 | 0.0001 / 0.009 | 0.0004 / 0.763 | 0.0002 / 0.005 |
| 768x768 | on | ok | 3.2 | ok | 201 | 147 | 0.0001 / 0.023 | 0.0002 / 0.173 | 0.0002 / 0.012 |
| 1024x576 | on | ok | 3.2 | ok | 197 | 147 | 0.0002 / 0.024 | 0.0002 / 0.085 | 0.0002 / 0.008 |
| 1024x1024 | off | ok | 3.5 | ok | 163 | 153 | 0.0001 / 0.008 | 0.0003 / 0.936 | 0.0002 / 0.004 |
| 1024x1024 | on | ok | 3.8 | ok | 239 | 153 | 0.0001 / 0.013 | 0.0002 / 0.108 | 0.0002 / 0.006 |
| 2048x2048 | off | ok | 5.9 | ok | 335 | 197 | 0.0001 / 0.019 | 0.0005 / 0.942 | 0.0002 / 0.010 |
| 2048x2048 | on | ok | 6.0 | ok | 575 | 199 | 0.0001 / 0.032 | 0.0002 / 0.190 | 0.0002 / 0.032 |

Drift is |TensorRT - PyTorch fp16| of the raw network outputs at the internal size, over 20 fixed frames (HCW Godiva Close frames 0-19 (raw)).

## Reference clips

| clip | focus | native | frames | hint (border mean) | licence | source |
|---|---|---|---|---|---|---|
| pexels-athlete | motion | 3840x2160 @ 60.0 | 300 | calibrated key (0.3725, 0.4275, 0.2275) (0.161; fixed hint was 0.969) | Pexels License (free use and modification; no redistribution on stock platforms) (checked 2026-10-16) | https://www.pexels.com/video/dynamic-athlete-in-action-against-green-screen-31916831/ |
| pexels-dance | motion | 3840x2160 @ 60.0 | 300 | calibrated key (0.3804, 0.4353, 0.2353) (0.067; fixed hint was 0.965) | Pexels License (free use and modification; no redistribution on stock platforms) (checked 2026-10-16) | https://www.pexels.com/video/expressive-dance-on-minimalist-green-screen-31916827/ |
| pixabay-hair | hair | 3840x2160 @ 24.0 | 300 | fixed (0.132) | Pixabay Content License (free use and adaptation; no standalone redistribution) (checked 2026-10-16) | https://pixabay.com/videos/man-hair-green-screen-chroma-key-190965/ |
| hcw-locked-backdrop | hair | 1280x1080 @ 30.0 | 80 | fixed (0.219) | Hollywood Camera Work plate licence (use and post results anywhere; no rehosting; no sale) (checked 2026-10-16) | https://www.hollywoodcamerawork.com/green-screen-plates.html |
| hcw-godiva-close | transparency | 960x720 @ 30.0 | 300 | fixed (0.139) | Hollywood Camera Work plate licence (use and post results anywhere; no rehosting; no sale) (checked 2026-10-16) | https://www.hollywoodcamerawork.com/green-screen-plates.html |

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
| 720p20 | trt-fp16-768x768-ref0-nvenc1080p30 | 26.5 | 50.0 | 76.5 | 2.30 | 4.59 |
| 720p30 | trt-fp16-768x768-ref0-nvenc1080p30 | 26.5 | 33.3 | 59.9 | 1.80 | 3.59 |
| 1080p30 | trt-fp16-1024x576-ref0-nvenc1080p30 | 29.3 | 33.3 | 62.6 | 1.88 | 3.76 |
| 1080p60 | trt-fp16-512x512-ref0-nvenc1080p30 | 14.9 | 16.7 | 31.5 | 0.95 | 1.89 |
| 2160p30 | trt-fp16-512x512-ref0-nvenc1080p30 | 31.1 | 33.3 | 64.4 | 1.93 | 3.87 |
| 2160p60 | trt-fp16-512x512-ref0-nvenc1080p30 | 31.1 | 16.7 | 47.8 | 1.43 | 2.87 |

## Environment

| item | value |
|---|---|
| gpu_name | NVIDIA GeForce RTX 3060 |
| gpu_driver | 576.80 |
| gpu_vram_mib | 12288 |
| gpu_max_sm_clock_mhz | 2130 |
| gpu_power_limit_w | 170.0 |
| os | Windows-11-10.0.26200-SP0 |
| cpu | Intel64 Family 6 Model 85 Stepping 7, GenuineIntel |
| ram_gib | 223.7 |
| python | 3.12.12 |
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
- Encode-load liveness check: the first load pass verified the encode by requiring NVML encoder utilisation > 0 in 5 samples over 1 s; a real-time 1080p30 encode keeps NVENC only ~7-14% busy and the counter often reads 0, so 4 configuration x camera entries were wrongly recorded as not_attempted. The check now uses NVML encoder session statistics (an active session) and those 4 entries were re-measured; the other load entries ran with the encode alive and are unchanged.

## Limitations

- OBS's own compositing and rendering load is not simulated; the concurrent load is a 1080p30 NVENC encode only.
- Reference clips are public stock footage; results on the owner's own studio (screen, lighting, camera) may differ.
- The RTX A2000 was not measured. Every A2000 figure is an ESTIMATE scaled from the RTX 3060.
- The Hollywood Camera Work plates are below 1080p (1280x1080 and 960x720) and are upscaled for the quality run.
- The alpha hint is a simple chroma key (research D8); hint failures (e.g. green clothing) affect the key and are not corrected.
- Eager PyTorch only (no torch.compile: MSVC is not installed); upstream reports ~30% from compiling.

## Reproduce

See [`bench/README.md`](../../README.md): `bench\setup.ps1`, then `uv run ckbench all` (or the individual commands). `results.json` in this folder holds every number above.
