<#
.SYNOPSIS
  Post-install setup for the Chipmo Sentry AI server. Run by the installer (or
  manually, elevated). Ensures uv, syncs the sentry-ai venv (downloads torch),
  writes the runtime .env, and installs the Windows services.

.NOTES
  This step needs internet (uv sync pulls torch/ultralytics, ~2-3 GB) and may
  take 10-20 minutes the first time. A console window stays open so you can
  watch progress.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BackendUrl,
    [Parameter(Mandatory = $true)][string]$Token,
    [string]$OllamaUrl = 'http://localhost:11434',
    [string]$TunnelName = ''
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
New-Item -ItemType Directory -Force -Path $cfg.LogDir, $cfg.RunDir, $cfg.ConfigDir | Out-Null

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

# 1. Ensure uv is available -------------------------------------------------
Write-Step "Checking for uv..."
$uv = (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $uv) {
    Write-Step "uv not found - installing (astral.sh)..."
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    $uv = (Get-Command uv -ErrorAction SilentlyContinue)
    if (-not $uv) { throw "uv install failed. Install uv manually (https://docs.astral.sh/uv) and re-run setup-server.ps1." }
}
$uvExe = $uv.Source
Write-Host "    uv: $uvExe"

# 2. Sync the sentry-ai venv (this is the long step) ------------------------
Write-Step "Syncing sentry-ai dependencies (this downloads torch, can take 10-20 min)..."
Push-Location $cfg.AppSrc
try {
    & $uvExe sync --no-dev
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit $LASTEXITCODE)." }
}
finally { Pop-Location }

# 3. Write runtime .env (sentry-ai reads .env from its project dir) ---------
Write-Step "Writing runtime config..."
$envPath = Join-Path $cfg.AppSrc '.env'
$envBody = @"
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
HOST=$($cfg.AiHost)
PORT=$($cfg.AiPort)

OLLAMA_BASE_URL=$OllamaUrl
DEFAULT_PROVIDER=minicpm-v-2.6
INFERENCE_TIMEOUT_SEC=30

SENTRY_BACKEND_URL=$BackendUrl
SENTRY_BACKEND_SERVICE_TOKEN=$Token
"@
Set-Content -Path $envPath -Value $envBody -Encoding utf8
Write-Host "    wrote $envPath"

# Persist the resolved uv path + tunnel name for the service installer.
Set-Content -Path (Join-Path $cfg.ConfigDir 'runtime.txt') `
    -Value @("UV_EXE=$uvExe", "TUNNEL_NAME=$TunnelName") -Encoding ascii

# 4. Install the Windows services ------------------------------------------
Write-Step "Installing Windows services..."
$installArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'install-services.ps1'))
if ($TunnelName) { $installArgs += @('-TunnelName', $TunnelName) }
& powershell @installArgs

Write-Step "Done."
Write-Host ""
Write-Host "Chipmo Sentry AI server installed." -ForegroundColor Green
Write-Host "  Manage:  $($cfg.AppRoot)\scripts\server-control.ps1 status|health|logs|start|stop|restart" -ForegroundColor Green
Write-Host "  Make sure Ollama is running:  ollama serve   (model: ollama pull minicpm-v:8b)" -ForegroundColor Yellow
if (-not $TunnelName) {
    Write-Host "  Tunnel NOT configured. To expose to the Railway backend, set up a" -ForegroundColor Yellow
    Write-Host "  cloudflared tunnel (config\cloudflared.yml) then re-run install-services -TunnelName <name>." -ForegroundColor Yellow
}
Write-Host ""
Read-Host "Press Enter to close"
