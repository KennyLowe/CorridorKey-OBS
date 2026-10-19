# Builds the owner test release zip: corridorkey-obs-<version>-windows-x64.zip + .sha256 in plugin\release\.
# Layout: corridorkey-obs\ (as installed under C:\ProgramData\obs-studio\plugins\), install.ps1,
# INSTALL.md, A2000-RUN.md, LICENSE (GPL-2.0, the plugin part). No weights, ONNX, engines,
# TensorRT/CUDA DLLs or footage (checked below).
param([string]$Version = 'v0.1.0-rc.4', [string]$Config = 'RelWithDebInfo')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $root
$rundir = Join-Path $root "build_x64\rundir\$Config"
$dll = Join-Path $rundir 'corridorkey-obs.dll'
if (-not (Test-Path $dll)) { throw "not built: $dll (run scripts\build.ps1)" }

$out = Join-Path $root 'release'
$stage = Join-Path $out "stage-$Version"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$plug = Join-Path $stage 'corridorkey-obs'
New-Item -ItemType Directory -Force (Join-Path $plug 'bin\64bit') | Out-Null
Copy-Item $dll (Join-Path $plug 'bin\64bit')
$pdb = Join-Path $rundir 'corridorkey-obs.pdb'
if (Test-Path $pdb) { Copy-Item $pdb (Join-Path $plug 'bin\64bit') }
Copy-Item -Recurse (Join-Path $rundir 'corridorkey-obs') (Join-Path $plug 'data')
Copy-Item (Join-Path $root 'packaging\install.ps1') $stage
Copy-Item (Join-Path $repo 'docs\INSTALL.md') $stage
Copy-Item (Join-Path $repo 'docs\A2000-RUN.md') $stage
Copy-Item (Join-Path $root 'LICENSE') $stage

foreach ($need in 'data\perf-table.json', 'data\manifest.json', 'data\effects\composite.effect', 'data\effects\colorkey.effect') {
    if (-not (Test-Path (Join-Path $plug $need))) { throw "missing $need" }
}
$bad = Get-ChildItem -Recurse -File $stage | Where-Object {
    $_.Extension -in '.safetensors', '.onnx', '.engine', '.plan', '.pt', '.pth', '.mp4', '.mov', '.png', '.npy' -or
    $_.Name -match '^(nvinfer|nvonnxparser|cudart|cublas|cudnn)'
}
if ($bad) { throw "forbidden files in the package: $($bad.FullName -join ', ')" }
$dlls = Get-ChildItem -Recurse -File $stage -Filter *.dll
if ($dlls.Count -ne 1) { throw "expected exactly one DLL (the plugin), found: $($dlls.Name -join ', ')" }

$zip = Join-Path $out "corridorkey-obs-$Version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
$sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
Set-Content -NoNewline -Encoding ascii -Path "$zip.sha256" -Value "$sha  $(Split-Path -Leaf $zip)`n"
Write-Host "package: $zip"
Write-Host "sha256:  $sha"
