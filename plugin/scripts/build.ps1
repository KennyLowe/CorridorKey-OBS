# Configure, build and test the plugin (no CI; owner machines only, FR-015).
# Usage: .\scripts\build.ps1 [-Config RelWithDebInfo] [-TensorRTRoot C:\TensorRT-10.16.1.11] [-NoTest]
param(
    [string]$Config = 'RelWithDebInfo',
    [string]$TensorRTRoot = 'C:\TensorRT-10.16.1.11',
    [switch]$NoTest
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    # A shell opened before the CUDA install doesn't have CUDA_PATH yet (MSBuild's CUDA targets need it).
    foreach ($v in 'CUDA_PATH', 'CUDA_PATH_V12_9') {
        if (-not [Environment]::GetEnvironmentVariable($v)) {
            [Environment]::SetEnvironmentVariable($v, [Environment]::GetEnvironmentVariable($v, 'Machine'))
        }
    }
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if (-not $cmake) { $cmake = 'C:\Program Files\CMake\bin\cmake.exe' }
    $ctest = Join-Path (Split-Path $cmake) 'ctest.exe'
    & $cmake --preset windows-x64 "-DTensorRT_ROOT=$TensorRTRoot"
    if ($LASTEXITCODE) { throw "configure failed ($LASTEXITCODE)" }
    & $cmake --build --preset windows-x64 --config $Config
    if ($LASTEXITCODE) { throw "build failed ($LASTEXITCODE)" }
    if (-not $NoTest) {
        & $ctest --preset windows-x64 -C $Config
        if ($LASTEXITCODE) { throw "tests failed ($LASTEXITCODE)" }
    }
}
finally { Pop-Location }
