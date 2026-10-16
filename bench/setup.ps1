<#
.SYNOPSIS
  Prepare the ckbench environment (idempotent).

  - clones CorridorKey at a pinned commit into bench/vendor/CorridorKey
  - downloads the official CorridorKey v1.0 weights from Hugging Face (never committed)
  - creates the uv environment with exact pins (uv sync)
  - checks CUDA, ffmpeg + h264_nvenc, and the manually downloaded HCW plates

  Run from anywhere:  powershell -ExecutionPolicy Bypass -File bench\setup.ps1
#>
$ErrorActionPreference = 'Stop'

$CorridorKeyRepo = 'https://github.com/nikopueringer/CorridorKey'
$CorridorKeySha  = '97e55a453060745bead1befd293f6e523c4b845c'   # main HEAD on 2026-10-16
$HfRepo          = 'nikopueringer/CorridorKey_v1.0'
$HfRevision      = 'f6386ddf042d8e92aeb5fd16cb9b101cff508195'   # HF repo revision on 2026-10-16
$WeightsFile     = 'CorridorKey_v1.0.safetensors'

$Bench    = $PSScriptRoot
$RepoRoot = Split-Path $Bench -Parent
$Vendor   = Join-Path $Bench 'vendor'
$CkDir    = Join-Path $Vendor 'CorridorKey'
$CkptDir  = Join-Path $CkDir 'CorridorKeyModule\checkpoints'
$Weights  = Join-Path $CkptDir $WeightsFile

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

# 1. CorridorKey at the pinned SHA
Step "CorridorKey @ $CorridorKeySha"
if (-not (Test-Path (Join-Path $CkDir '.git'))) {
    New-Item -ItemType Directory -Force $Vendor | Out-Null
    git clone --quiet $CorridorKeyRepo $CkDir
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
}
$head = (git -C $CkDir rev-parse HEAD).Trim()
if ($head -ne $CorridorKeySha) {
    git -C $CkDir fetch --quiet origin
    git -C $CkDir checkout --quiet $CorridorKeySha
    if ($LASTEXITCODE -ne 0) { throw "could not check out $CorridorKeySha" }
}
Write-Host "    CorridorKey SHA: $((git -C $CkDir rev-parse HEAD).Trim())"

# 2. Weights (official source; licence CC BY-NC-SA 4.0; never committed)
Step "Weights $WeightsFile"
New-Item -ItemType Directory -Force $CkptDir | Out-Null
if (-not (Test-Path $Weights)) {
    $url = "https://huggingface.co/$HfRepo/resolve/$HfRevision/$WeightsFile"
    Write-Host "    downloading $url"
    curl.exe -L --fail --silent --show-error -o "$Weights.part" $url
    if ($LASTEXITCODE -ne 0) { throw 'weights download failed' }
    Move-Item -Force "$Weights.part" $Weights
}
Write-Host "    sha256: $((Get-FileHash -Algorithm SHA256 $Weights).Hash.ToLower())"

# 3. Python environment
Step 'uv sync'
Push-Location $Bench
try {
    uv sync
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }

    # 4. CUDA
    Step 'CUDA check'
    uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('    ', torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed' }
} finally { Pop-Location }

# 5. ffmpeg + NVENC (actually opens the encoder: the binary may list h264_nvenc
#    yet require a newer driver than installed)
Step 'ffmpeg h264_nvenc check'
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ff) { throw 'ffmpeg not on PATH. Install: winget install --id Gyan.FFmpeg -e --version 7.1.1' }
Write-Host "    $((& ffmpeg -hide_banner -version | Select-Object -First 1))"
& ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 1 -c:v h264_nvenc -preset p5 -b:v 6M -f null - 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw 'h264_nvenc failed to open. ffmpeg >= 8.1 needs NVENC API 13.1 (driver >= 610); use Gyan.FFmpeg 7.1.1 on older drivers.'
}
Write-Host '    h264_nvenc OK'

# 6. Manually downloaded reference plates (warn only; licence forbids rehosting)
Step 'Hollywood Camera Work plates'
foreach ($p in @('footage\hcw\Locked Backdrop', 'footage\hcw\Godiva Close')) {
    $full = Join-Path $RepoRoot $p
    if (Test-Path $full) { Write-Host "    found $p ($((Get-ChildItem $full -Filter *.png).Count) png)" }
    else { Warn "$p missing: download from https://www.hollywoodcamerawork.com/green-screen-plates.html and unzip there" }
}

Step 'setup complete'
