## Hybrid key: OBS chroma key for the interior, CorridorKey only for the edge

### Why

Tested rc.3 on the RTX A2000 machine with a clean, evenly lit green screen and a Sony ZV-E10 through a Cam Link (1920×1080, 50 fps, cropped to 1120×650, canvas 1280×720 at 25 fps, virtual camera). With **Use source colour** on, the picture is sharp, but OBS's built-in **Chroma Key** filter looked better, including on a glass of water moved across the screen. That is the easy case for a classic key: full resolution, per-pixel, no latency.

CorridorKey's model is limited on this GPU, and its matte is a few frames old when applied.

### Measurements (A2000, from the perf table and live logs)

| Quality | p95 ms | Fits 33 ms (30 fps bucket) | Live result |
|---|---|---|---|
| 512×512, refiner off | 20.8 | yes | not tried |
| 768×448, refiner off | 26.1 | yes | 18.0 ms median / 23.9 ms p95, 0 fallbacks once settled |
| 768×448, refiner on | 45.0 | no | 43–44 ms, 94 of 6392 frames skipped, repeated fallback |
| 1024×576, refiner off | 45.1 | no | 37.7–41 ms, 60 of 1273 skipped, flips keying/fallback about every 0.4 s |
| 768×768 / 1024×1024 | ≥45 | no | not usable |

So the only real-time setting on the A2000 is 768×448 with the refiner off. The model's strengths (hair, motion blur, translucency) appear at higher resolution and with the refiner, which don't fit the budget here.

### Cause of the softness fixed in rc.3

The model's colour output was bilinearly enlarged and drawn as-is, so the camera's pixels were never used. "Use source colour" fixed that. The matte is still 768×448 enlarged to the source, so edges are soft.

### Proposal

Use the model only where the classic key is weak, and keep the classic key's sharp, immediate interior.

1. Add an opt-in mode, for example **Hybrid (chroma key + CorridorKey edge)**.
2. Compute a classic key matte at full source resolution in the shader (reuse `colorkey.effect` and the calibrated screen colour, which already exist for the fallback and the hint).
3. Take the **interior** (alpha near 0 or 1 in the chroma key) straight from the chroma key.
4. Take the **uncertain band** (a few pixels either side of the chroma key's edge, plus any region where the two mattes disagree strongly) from the model's alpha and unmixed foreground.
5. Colour: live frame with despill, as in `PSCompositeSource`; model foreground only in the band.
6. Keep the existing fallbacks.

Expected benefits: sharp interior and edges that follow motion with no lag, because the classic key is computed on the current frame. The model still handles hair and translucent fringes.

### Open questions for the build

- How wide should the uncertain band be, and should it scale with source size? A fixed pixel width is simplest.
- How to mix at the band boundary without a visible seam (smoothstep on the band mask).
- Lag: the model matte is old by a few frames, so in the band it can mismatch the current frame during fast motion. Option: only trust the model in the band where the chroma key is also uncertain.
- Whether to expose the chroma key's similarity and smoothness as sliders, or derive them from the calibrated hint (`hint_lo` / `hint_hi`).
- How to test. A harder comparison is needed: creased or unevenly lit screen, shadows, green-ish clothing, a glass of water, fine hair. The easy case above is where the classic key already wins, so it won't show any benefit.

### Acceptance

- Opt-in, default off. Existing behaviour and tests unchanged.
- A new e2e test alongside `plugin/tests/e2e/test_source_colour.py` comparing hybrid against chroma key and against plain source-colour on a synthetic hard case (uneven screen, thin strands, translucent object).
- Render-thread cost stays near 1 ms at 1280×720 on the 3060 and the A2000.
- Document the setting in `docs/INSTALL.md`.

### Related

- rc.3 "Use source colour" (merged in #5).
- Earlier finding: the grid pattern seen over flat areas at 768×448 was never traced (it could be the model's patch grid, the refiner or the resize). The hybrid would hide it in the interior. A debug dump of the model's matte and foreground before the resize would settle it.
