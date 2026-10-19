# Installs the CorridorKey OBS filter from this release zip (see INSTALL.md).
# Copies .\corridorkey-obs\ to C:\ProgramData\obs-studio\plugins\corridorkey-obs\
# (OBS 32 layout: bin\64bit\corridorkey-obs.dll, data\ beside bin). Run as Administrator, OBS closed.
$ErrorActionPreference = 'Stop'
$src = Join-Path $PSScriptRoot 'corridorkey-obs'
if (-not (Test-Path (Join-Path $src 'bin\64bit\corridorkey-obs.dll'))) {
    throw "corridorkey-obs\bin\64bit\corridorkey-obs.dll not found next to this script; unzip the whole release first."
}
$obs = Get-Process obs64 -ErrorAction SilentlyContinue
if ($obs) {
    Write-Host "OBS is running (obs64.exe, PID $($obs.Id -join ', ')). Close OBS, then run this again." -ForegroundColor Yellow
    exit 1
}
$dest = Join-Path $env:ProgramData 'obs-studio\plugins\corridorkey-obs'
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
Copy-Item -Recurse $src $dest
Write-Host "Installed to $dest"
Get-ChildItem -Recurse -File $dest | ForEach-Object { Write-Host "  $($_.FullName.Substring($dest.Length + 1))" }
if (-not (Test-Path 'C:\TensorRT-10.16.1.11\bin\nvinfer_10.dll')) {
    Write-Host "Note: C:\TensorRT-10.16.1.11\bin\nvinfer_10.dll not found. Install TensorRT 10.16.1.11 (INSTALL.md) or set the TensorRT folder in the filter." -ForegroundColor Yellow
}
Write-Host "Start OBS and add the CorridorKey filter to a video source (INSTALL.md, step 4)."
