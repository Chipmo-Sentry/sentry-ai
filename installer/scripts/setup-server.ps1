<#
.SYNOPSIS
  Post-install setup for the Chipmo Sentry AI server. Ensures uv, syncs the
  sentry-ai venv (downloads torch), writes runtime config, PAIRS the node with
  the backend using the 6-digit code, and installs the Windows services.

.NOTES
  First run needs internet (uv sync pulls torch/ultralytics, ~2-3 GB) and may
  take 10-20 minutes. On an update, leave -PairingCode blank to keep the
  existing pairing.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BackendUrl,
    [string]$OllamaUrl = 'http://localhost:11434',
    [string]$PairingCode = '',
    [string]$PublicUrl = '',
    [string]$TunnelName = ''
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
New-Item -ItemType Directory -Force -Path $cfg.LogDir, $cfg.RunDir, $cfg.ConfigDir | Out-Null

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

# 1. Ensure uv ---------------------------------------------------------------
Write-Step "Checking for uv..."
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uv) {
    Write-Step "uv not found - installing (astral.sh)..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    $uv = (Get-Command uv -ErrorAction SilentlyContinue)
    if (-not $uv) { throw "uv install failed. Install uv manually and re-run setup-server.ps1." }
}
$uvExe = $uv.Source
Write-Host "    uv: $uvExe"

# 2. Sync the sentry-ai venv -------------------------------------------------
Write-Step "Syncing sentry-ai dependencies (downloads torch, can take 10-20 min)..."
Push-Location $cfg.AppSrc
try {
    & $uvExe sync --no-dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit $LASTEXITCODE)." }
}
finally { Pop-Location }

# 3. Write base runtime config (NOT the token/node id - pairing fills those) -
Write-Step "Writing base config..."
$envPath = Join-Path $cfg.AppSrc '.env'
$baseKeys = @{
    'ENVIRONMENT'          = 'production'
    'DEBUG'                = 'false'
    'LOG_LEVEL'            = 'INFO'
    'HOST'                 = $cfg.AiHost
    'PORT'                 = "$($cfg.AiPort)"
    'OLLAMA_BASE_URL'      = $OllamaUrl
    'DEFAULT_PROVIDER'     = 'qwen3-vl-4b'
    'INFERENCE_TIMEOUT_SEC' = '30'
    'SENTRY_BACKEND_URL'   = $BackendUrl
}
# Preserve existing lines (esp. SENTRY_BACKEND_SERVICE_TOKEN + AI_NODE_ID on update).
$existing = @{}
if (Test-Path $envPath) {
    foreach ($line in (Get-Content $envPath)) {
        if ($line -match '^\s*([^=#]+)=(.*)$') { $existing[$Matches[1].Trim()] = $Matches[2] }
    }
}
foreach ($k in $baseKeys.Keys) { $existing[$k] = $baseKeys[$k] }
$out = foreach ($k in $existing.Keys) { "$k=$($existing[$k])" }
Set-Content -Path $envPath -Value $out -Encoding utf8

# 4. Pair with the backend ---------------------------------------------------
if ($PairingCode) {
    Write-Step "Pairing with backend (code $PairingCode)..."
    Push-Location $cfg.AppSrc
    try {
        $pairArgs = @('run', 'python', '-m', 'sentry_ai.pair', '--code', $PairingCode, '--backend', $BackendUrl, '--env-file', '.env')
        if ($PublicUrl) { $pairArgs += @('--public-url', $PublicUrl) }
        & $uvExe @pairArgs
        if ($LASTEXITCODE -ne 0) { throw "pairing failed (exit $LASTEXITCODE). Check the code (it expires after 30 min)." }
    }
    finally { Pop-Location }
}
elseif (-not $existing.ContainsKey('AI_NODE_ID')) {
    throw "No pairing code given and this node isn't paired yet. Generate a code in superadmin (AI servers page) and re-run with it."
}
else {
    Write-Step "No code given - keeping existing pairing ($($existing['AI_NODE_ID']))."
}

# Persist uv path for the service installer.
Set-Content -Path (Join-Path $cfg.ConfigDir 'runtime.txt') -Value @("UV_EXE=$uvExe") -Encoding ascii

# 5. Install services --------------------------------------------------------
Write-Step "Installing Windows services..."
$installArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'install-services.ps1'))
if ($TunnelName) { $installArgs += @('-TunnelName', $TunnelName) }
& powershell @installArgs

Write-Step "Done."
Write-Host ""
Write-Host "Chipmo Sentry AI server installed + paired." -ForegroundColor Green
Write-Host "  It should now appear under 'AI servers' in the superadmin dashboard." -ForegroundColor Green
Write-Host "  Manage:  $($cfg.AppRoot)\scripts\server-control.ps1 status|health|logs" -ForegroundColor Green
Write-Host "  Ensure Ollama is running:  ollama serve   (ollama pull qwen3-vl:4b-instruct)" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to close"
