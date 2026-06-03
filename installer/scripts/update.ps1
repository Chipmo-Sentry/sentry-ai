<#
.SYNOPSIS
  Update the Chipmo Sentry AI server to the latest published release.

.DESCRIPTION
  Downloads the newest ChipmoSentryAi-Setup.exe from GitHub Releases and runs
  it. Re-installing over the top updates the bundled binaries + sentry-ai
  source and re-runs setup (uv sync picks up new deps, services are
  reinstalled). The installer pre-fills your existing backend URL / token /
  Ollama URL from the current config, so just click through.

  Compares the latest release tag with the installed version and skips the
  download if you are already current (pass -Force to reinstall anyway).
.EXAMPLE
  .\update.ps1
  .\update.ps1 -Force
#>
[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\server.config.ps1"

$repo = 'Chipmo-Sentry/sentry-ai'
$asset = 'ChipmoSentryAi-Setup.exe'

function Get-InstalledVersion {
    # __version__ in the bundled source.
    $init = Join-Path $ServerConfig.AppSrc 'src\sentry_ai\__init__.py'
    if (Test-Path $init) {
        $m = Select-String -Path $init -Pattern '__version__\s*=\s*"([^"]+)"'
        if ($m) { return $m.Matches[0].Groups[1].Value }
    }
    return '0.0.0'
}

Write-Host "==> Checking latest release of $repo ..." -ForegroundColor Cyan
$installed = Get-InstalledVersion
Write-Host "    installed: $installed"

$latestTag = $null
try {
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" `
        -Headers @{ 'User-Agent' = 'ChipmoSentryAi-Updater' } -TimeoutSec 15
    $latestTag = "$($rel.tag_name)".TrimStart('v')
    Write-Host "    latest:    $latestTag"
}
catch {
    Write-Warning "Could not query GitHub releases: $($_.Exception.Message)"
}

if (-not $Force -and $latestTag -and ($latestTag -eq $installed)) {
    Write-Host "Already up to date ($installed). Use -Force to reinstall." -ForegroundColor Green
    return
}

$url = "https://github.com/$repo/releases/latest/download/$asset"
$dest = Join-Path $env:TEMP $asset
Write-Host "==> Downloading $url" -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $dest

Write-Host "==> Launching installer (admin)..." -ForegroundColor Cyan
# Elevate; the wizard pre-fills your existing config.
Start-Process -FilePath $dest -Verb RunAs
Write-Host "Follow the installer to finish the update." -ForegroundColor Green
