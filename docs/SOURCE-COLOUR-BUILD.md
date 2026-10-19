# Build and test the "Use source colour" toggle (run on the RTX 3060 dev machine)

Branch: `source-colour-toggle` (based on `v0.1.0-rc.2`). The A2000 machine has no build toolchain
(no CMake, no `build_x64`), so the change is built here and handed back.

## Why

On the A2000 the key is good but the picture is soft and blocky. `k_postprocess`
(`plugin/src/gpu/kernels.cu`) bilinearly enlarges the model's 768×448 foreground **colour** and
alpha to the source size, and `PSComposite` draws only that, so the camera's own pixels are never
used. The A2000 can't run a bigger model within the frame budget (1024×576 refiner off measured
39–41 ms p95 against a 40 ms budget at 25 fps, and the filter kept dropping to the fallback).

## What changed (2 files, no GPU kernel or model change)

- `plugin/data/effects/composite.effect`: new `DrawSource` technique. The matte comes from the
  model. The colour comes from the live frame with a light green-spill clamp
  (`g = min(g, max(r, b))`). Where alpha is partial, the model's unmixed foreground takes over.
- `plugin/src/ck-filter.cpp`: new `source_colour` setting (default **off**, so existing behaviour
  and tests are unchanged) and a checkbox under Quality. When on, the composite draw uses
  `DrawSource` and passes the live frame (`src`) as `image`.

Known trade-off: the matte is a few frames old when paired, but the colour is from the current
frame, so on fast movement the outline can trail slightly. The `LastMatte` fallback already works
this way.

## Steps

```powershell
git fetch origin
git checkout source-colour-toggle
cd plugin
.\scripts\build.ps1          # configure + build + ctest (unit and GPU suites)
```

Expected: build succeeds and ctest passes as before (65/65 in `plugin/TEST-RESULTS.md`).
Nothing in the tests covers the new setting, so a pass only means nothing else broke.

### Try it on the 3060 first (OBS closed)

```powershell
.\scripts\install.ps1
```

Open OBS, open the CorridorKey filter, tick **Use source colour**. Compare with it off. Check:

1. Detail on glasses, hair, fabric and fingers is sharp with it on.
2. No green fringe at the edges, and no dark or light halo around the outline.
3. Move quickly: how far does the edge trail?
4. The render-thread cost in the log's `60 s summary` line is about the same (it adds one texture
   read per pixel).
5. Un-ticking returns the old look, with no restart.

### Package for the A2000

```powershell
.\scripts\package.ps1 -Version v0.1.0-rc.2-srccolour
```

This writes `plugin\release\corridorkey-obs-v0.1.0-rc.2-srccolour-windows-x64.zip` and `.sha256`.
The zip is git-ignored, so copy it to the A2000 machine another way, or attach it to a GitHub
release or a draft pre-release. The script is not changed; it only needs the version string.

## Hand back

Make any shader or code tweaks you want (the `edge` weight and the despill line in
`PSCompositeSource` are the obvious things to tune), then:

```powershell
git add -A
git commit -m "plugin: <what you changed>"
git push origin source-colour-toggle
```

Tell me it's pushed. On the A2000 machine I'll pull the branch and install from the zip you built
(or from your pushed files if you attach the zip to a release).

## If it doesn't build

- `cmake` not found: `scripts\build.ps1` falls back to `C:\Program Files\CMake\bin\cmake.exe`.
- Anything else: paste the last 40 lines of the build output.
