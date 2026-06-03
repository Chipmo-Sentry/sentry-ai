<#
.SYNOPSIS
  Manage this AI node's pairing with the backend WITHOUT reinstalling.
  Once paired the token lives in app\.env and survives restarts/updates;
  use this only to re-pair (new code) or unpair.
.EXAMPLE
  .\pairing.ps1 status
  .\pairing.ps1 pair       # prompts for a 6-digit code from superadmin
  .\pairing.ps1 unpair
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet('status', 'pair', 'unpair')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
$envPath = Join-Path $cfg.AppSrc '.env'

function Read-Env {
    $h = [ordered]@{}
    if (Test-Path $envPath) {
        foreach ($line in (Get-Content $envPath)) {
            if ($line -match '^\s*([^=#]+)=(.*)$') { $h[$Matches[1].Trim()] = $Matches[2] }
        }
    }
    return $h
}

function Write-Env($h) {
    $out = foreach ($k in $h.Keys) { "$k=$($h[$k])" }
    Set-Content -Path $envPath -Value $out -Encoding utf8
}

function Resolve-Uv {
    $rt = Join-Path $cfg.ConfigDir 'runtime.txt'
    if (Test-Path $rt) {
        foreach ($line in (Get-Content $rt)) {
            if ($line -match '^UV_EXE=(.+)$') { return $Matches[1].Trim() }
        }
    }
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }
    throw "uv not found. Re-run setup once, or install uv."
}

function Restart-Ai {
    $svc = "$($cfg.ServicePrefix)ai"
    if (Get-Service -Name $svc -ErrorAction SilentlyContinue) {
        Write-Host "Restarting $svc ..." -ForegroundColor Cyan
        Restart-Service -Name $svc -Force -ErrorAction Stop
        Write-Host "  restarted." -ForegroundColor Green
    }
    else {
        Write-Host "  ($svc not installed yet - run setup first)" -ForegroundColor DarkGray
    }
}

$envv = Read-Env
$nodeId = if ($envv.Contains('AI_NODE_ID')) { $envv['AI_NODE_ID'] } else { '' }
$paired = [bool]$nodeId

switch ($Action) {
    'status' {
        if ($paired) {
            Write-Host "Paired" -ForegroundColor Green
            Write-Host "  node id : $nodeId"
            Write-Host "  backend : $($envv['SENTRY_BACKEND_URL'])"
            Write-Host ""
            Write-Host "Pairing is saved - it survives restarts & updates. Re-pair only if you" -ForegroundColor DarkGray
            Write-Host "moved the node or revoked it in superadmin." -ForegroundColor DarkGray
        }
        else {
            Write-Host "Not paired" -ForegroundColor Yellow
            Write-Host "  Generate a code in superadmin (AI server -> Холболтын код), then run:"
            Write-Host "    .\pairing.ps1 pair"
        }
    }
    'pair' {
        if ($paired) {
            Write-Host "Already paired (node $nodeId). Re-pairing replaces it." -ForegroundColor Yellow
        }
        $backend = $envv['SENTRY_BACKEND_URL']
        $inB = Read-Host "Backend URL [$backend]"
        if ($inB) { $backend = $inB }
        if (-not $backend) { throw "Backend URL required." }
        $code = Read-Host "6-digit pairing code (superadmin -> AI server -> Холболтын код)"
        if (-not ($code -match '^\d{6}$')) { throw "Code must be exactly 6 digits." }

        $uvExe = Resolve-Uv
        Push-Location $cfg.AppSrc
        try {
            & $uvExe run python -m sentry_ai.pair --code $code --backend $backend --env-file '.env'
            if ($LASTEXITCODE -ne 0) { throw "Pairing failed (exit $LASTEXITCODE). The code expires after 30 min." }
        }
        finally { Pop-Location }
        Restart-Ai
        Write-Host ""
        Write-Host "Paired. It should appear online under 'AI server' in superadmin." -ForegroundColor Green
    }
    'unpair' {
        if (-not $paired) {
            Write-Host "Not paired - nothing to do." -ForegroundColor Yellow
        }
        else {
            $confirm = Read-Host "Unpair node $nodeId? Type 'unpair' to confirm"
            if ($confirm -ne 'unpair') {
                Write-Host "Cancelled."
            }
            else {
                $envv.Remove('AI_NODE_ID')
                $envv.Remove('SENTRY_BACKEND_SERVICE_TOKEN')
                Write-Env $envv
                Restart-Ai
                Write-Host ""
                Write-Host "Unpaired locally. Also revoke it in superadmin (AI server -> Цуцлах)." -ForegroundColor Green
            }
        }
    }
}

if ($Action -ne 'status') { Write-Host ""; Read-Host "Press Enter to close" | Out-Null }
