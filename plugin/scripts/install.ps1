# Copy the built plugin into OBS's per-machine plugin folder (OBS 32 layout):
#   C:\ProgramData\obs-studio\plugins\corridorkey-obs\bin\64bit\corridorkey-obs.dll (+ .pdb)
#   C:\ProgramData\obs-studio\plugins\corridorkey-obs\data\...
# Close OBS first. No TensorRT/CUDA DLLs are copied: the plugin loads nvinfer_10.dll from the
# TensorRT folder configured in the filter (default C:\TensorRT-10.16.1.11\bin).
param([string]$Config = 'RelWithDebInfo')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "build_x64\$Config"
$dll = Join-Path $build 'corridorkey-obs.dll'
if (-not (Test-Path $dll)) { throw "not built: $dll (run scripts\build.ps1)" }
$dest = Join-Path $env:ProgramData 'obs-studio\plugins\corridorkey-obs'
$bin = Join-Path $dest 'bin\64bit'
New-Item -ItemType Directory -Force $bin | Out-Null
Copy-Item $dll $bin -Force
$pdb = Join-Path $build 'corridorkey-obs.pdb'
if (Test-Path $pdb) { Copy-Item $pdb $bin -Force }
$data = Join-Path $dest 'data'
if (Test-Path $data) { Remove-Item -Recurse -Force $data }
Copy-Item (Join-Path $root 'data') $data -Recurse -Force
Copy-Item (Join-Path (Split-Path -Parent $root) 'models\manifest.json') (Join-Path $data 'manifest.json') -Force
Write-Host "installed to $dest"
