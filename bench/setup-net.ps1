# Network helpers for setup.ps1 that cope with corporate networks (dot-source this file).
#
# - schannel certificate-revocation failures (CRYPT_E_NO_REVOCATION_CHECK / CRYPT_E_REVOCATION_OFFLINE:
#   the revocation server is unreachable, common behind corporate proxies): retried once with
#   revocation checking off for that one command (curl --ssl-no-revoke; git -c
#   http.schannelCheckRevoke=false; global git config is never changed). Integrity is still
#   guaranteed by the pinned sha256 (downloads) and the pinned commit (git).
# - uv failing on certificates (a TLS-inspecting proxy with a corporate root CA): retried once with
#   UV_SYSTEM_CERTS=1 (current uv, e.g. 0.13.1) and UV_NATIVE_TLS=1 (older uv) so uv uses the Windows
#   certificate store.

$script:RevocationPattern = 'CRYPT_E_NO_REVOCATION_CHECK|CRYPT_E_REVOCATION_OFFLINE'

function Get-VerifiedFile {
    param(
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Out,
        [Parameter(Mandatory)] [string]$Sha256,
        [string]$CurlExe = 'curl.exe'
    )
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $part = "$Out.part"
        Remove-Item -Force -ErrorAction SilentlyContinue $part
        & $CurlExe -L --fail --show-error --progress-bar -o $part $Url
        $rc = $LASTEXITCODE
        if ($rc -eq 35) {
            # Exit 35 is any TLS connect error: look at the message before turning anything off.
            $probe = (& $CurlExe -sS -I -o NUL $Url 2>&1 | Out-String)
            if ($probe -match $script:RevocationPattern) {
                Write-Warning ("The certificate revocation check failed ($($Matches[0])): this network blocks the " +
                    "revocation server. Retrying once with curl --ssl-no-revoke; the file is still verified " +
                    "against its pinned sha256.")
                & $CurlExe -L --fail --show-error --progress-bar --ssl-no-revoke -o $part $Url
                $rc = $LASTEXITCODE
            }
        }
        if ($rc -ne 0) {
            Remove-Item -Force -ErrorAction SilentlyContinue $part
            throw "download failed (curl exit $rc): $Url"
        }
        $got = (Get-FileHash -Algorithm SHA256 $part).Hash.ToLower()
        if ($got -ne $Sha256.ToLower()) {
            Remove-Item -Force -ErrorAction SilentlyContinue $part
            throw "sha256 mismatch for ${Url}: got $got, expected $($Sha256.ToLower()); the download was deleted"
        }
        Move-Item -Force $part $Out
    } finally {
        $ErrorActionPreference = $eap
    }
}

function Invoke-GitNet {
    param(
        [string]$GitExe = 'git',
        [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Rest
    )
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = (& $GitExe @Rest 2>&1 | Out-String)
        $rc = $LASTEXITCODE
        if ($rc -ne 0 -and $out -match $script:RevocationPattern) {
            Write-Warning ("git's certificate revocation check failed ($($Matches[0])): this network blocks the " +
                "revocation server. Retrying once with -c http.schannelCheckRevoke=false for this command only " +
                "(global git config unchanged); the checkout is still pinned to an exact commit.")
            $out = (& $GitExe -c http.schannelCheckRevoke=false @Rest 2>&1 | Out-String)
            $rc = $LASTEXITCODE
        }
        if ($rc -ne 0) { throw "git $($Rest -join ' ') failed (exit $rc): $out" }
        $out
    } finally {
        $ErrorActionPreference = $eap
    }
}

function Invoke-UvSync {
    param([string]$UvExe = 'uv')
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = (& $UvExe sync 2>&1 | Tee-Object -Variable lines | Out-String)
        $lines | ForEach-Object { Write-Host $_ }
        $rc = $LASTEXITCODE
        if ($rc -ne 0 -and $out -match 'certificate|UnknownIssuer|CERTIFICATE_VERIFY_FAILED|invalid peer') {
            Write-Warning ("uv could not verify a TLS certificate (a TLS-inspecting proxy?). Retrying once with " +
                "UV_SYSTEM_CERTS=1 and UV_NATIVE_TLS=1 so uv uses the Windows certificate store, which has " +
                "the corporate root.")
            $env:UV_SYSTEM_CERTS = '1'
            $env:UV_NATIVE_TLS = '1'
            & $UvExe sync
            $rc = $LASTEXITCODE
        }
        if ($rc -ne 0) { throw "uv sync failed (exit $rc)" }
    } finally {
        $ErrorActionPreference = $eap
    }
}

# HEAD of a checkout. git refuses folders owned by another account ("dubious ownership", e.g. a
# clone made from an elevated prompt); print the exact fix instead of changing global git config.
function Get-GitHead {
    param([Parameter(Mandatory)] [string]$Dir, [string]$GitExe = 'git')
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = (& $GitExe -C $Dir rev-parse HEAD 2>&1 | Out-String)
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $eap
    }
    if ($rc -ne 0) {
        if ($out -match "dubious ownership in repository at '([^']+)'") {
            $p = $Matches[1].Replace([string][char]92, '/')
            Write-Host "git refuses to read $p (""dubious ownership"": the folder is owned by another account, e.g. Administrators). Fix, then re-run setup.ps1:" -ForegroundColor Yellow
            Write-Host "    git config --global --add safe.directory $p"
            throw "git refuses to read $p (dubious ownership); see the fix above"
        }
        throw "git rev-parse failed in ${Dir}: $out"
    }
    $out.Trim()
}
