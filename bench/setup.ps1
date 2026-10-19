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
$Bench    = $PSScriptRoot
$RepoRoot = Split-Path $Bench -Parent

# The official model (repo, revision, file, size, sha256) comes from the committed model manifest,
# the same file the OBS plugin's in-app download uses.
$Model       = (Get-Content -Raw (Join-Path $RepoRoot 'models\manifest.json') | ConvertFrom-Json).models |
               Where-Object model_id -eq 'CorridorKey_v1.0'
if (-not $Model) { throw 'CorridorKey_v1.0 not found in models\manifest.json' }
$HfRepo      = $Model.hf_repo
$HfRevision  = $Model.revision
$WeightsFile = $Model.file
$Vendor   = Join-Path $Bench 'vendor'
$CkDir    = Join-Path $Vendor 'CorridorKey'
$CkptDir  = Join-Path $CkDir 'CorridorKeyModule\checkpoints'
$Weights  = Join-Path $CkptDir $WeightsFile

# Downloads/clones that survive corporate networks (revocation-check failures, TLS inspection).
. (Join-Path $PSScriptRoot 'setup-net.ps1')

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }

# 1. CorridorKey at the pinned SHA
Step "CorridorKey @ $CorridorKeySha"
if (-not (Test-Path (Join-Path $CkDir '.git'))) {
    New-Item -ItemType Directory -Force $Vendor | Out-Null
    Invoke-GitNet clone --quiet $CorridorKeyRepo $CkDir | Out-Null
}
$head = Get-GitHead -Dir $CkDir
if ($head -ne $CorridorKeySha) {
    Push-Location $CkDir
    try { Invoke-GitNet fetch --quiet origin | Out-Null } finally { Pop-Location }
    git -C $CkDir checkout --quiet $CorridorKeySha
    if ($LASTEXITCODE -ne 0) { throw "could not check out $CorridorKeySha" }
}
Write-Host "    CorridorKey SHA: $(Get-GitHead -Dir $CkDir)"

# 2. Weights (official source; licence CC BY-NC-SA 4.0; never committed)
Step "Weights $WeightsFile"
New-Item -ItemType Directory -Force $CkptDir | Out-Null
if (-not (Test-Path $Weights)) {
    $url = "https://huggingface.co/$HfRepo/resolve/$HfRevision/$WeightsFile"
    Write-Host "    downloading $url (about $([math]::Round($Model.size / 1MB)) MB)"
    Get-VerifiedFile -Url $url -Out $Weights -Sha256 $Model.sha256
}
$sha = (Get-FileHash -Algorithm SHA256 $Weights).Hash.ToLower()
if ($sha -ne $Model.sha256) { throw "weights at $Weights have sha256 $sha, the manifest pins $($Model.sha256)" }
Write-Host "    sha256: $sha (matches models\manifest.json)"

# 3. Python environment
Step 'uv sync'
Push-Location $Bench
try {
    Invoke-UvSync

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
