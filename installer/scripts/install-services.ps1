<#
.SYNOPSIS
  Install the Chipmo Sentry AI server components as Windows services (NSSM):
  ingest (MediaMTX), ai (sentry-ai uvicorn), and optionally tunnel (cloudflared).
  Run ELEVATED. Boot-start + auto-restart on crash.
.EXAMPLE
  .\install-services.ps1 -TunnelName sentry-ingest
  .\install-services.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param([string]$TunnelName = '')

. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
New-Item -ItemType Directory -Force -Path $cfg.LogDir | Out-Null

$nssm = $cfg.NssmExe
if (-not (Test-Path $nssm)) { throw "nssm.exe not found at $nssm" }

# Resolve uv full path (services run as LocalSystem - PATH may differ).
$uvExe = $null
$runtimeFile = Join-Path $cfg.ConfigDir 'runtime.txt'
if (Test-Path $runtimeFile) {
    $line = (Get-Content $runtimeFile | Where-Object { $_ -like 'UV_EXE=*' } | Select-Object -First 1)
    if ($line) { $uvExe = $line.Substring(7) }
}
if (-not $uvExe -or -not (Test-Path $uvExe)) {
    $c = Get-Command uv -ErrorAction SilentlyContinue
    if ($c) { $uvExe = $c.Source }
}
if (-not $uvExe) { throw "uv.exe not found - run setup-server.ps1 first." }

function Get-Spec([string]$name) {
    switch ($name) {
        'ingest' {
            return @{ Exe = $cfg.MediaMtxExe; Args = ('"' + $cfg.MediaMtxConfig + '"'); Wd = $cfg.BinDir }
        }
        'ai' {
            return @{
                Exe  = $uvExe
                Args = ('run --project "' + $cfg.AppSrc + '" uvicorn sentry_ai.main:app --host ' + $cfg.AiHost + ' --port ' + $cfg.AiPort)
                Wd   = $cfg.AppSrc
            }
        }
        'tunnel' {
            return @{
                Exe  = $cfg.CloudflaredExe
                Args = ('--config "' + $cfg.CloudflaredCfg + '" tunnel run ' + $TunnelName)
                Wd   = $cfg.BinDir
            }
        }
        'tunnels' {
            # Quick-tunnel supervisor: keeps 2 trycloudflare tunnels alive +
            # re-points Railway when a URL changes (needs config\railway-token.txt).
            $ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
            $script = Join-Path $PSScriptRoot 'tunnel-supervisor.ps1'
            return @{
                Exe  = $ps
                Args = ('-NoProfile -ExecutionPolicy Bypass -File "' + $script + '"')
                Wd   = $cfg.AppRoot
            }
        }
    }
}

function Install-One([string]$name) {
    $svc = "$($cfg.ServicePrefix)$name"
    $spec = Get-Spec $name
    $out = Join-Path $cfg.LogDir "$name.out.log"
    $err = Join-Path $cfg.LogDir "$name.err.log"
    if ($PSCmdlet.ShouldProcess($svc, "nssm install")) {
        # Remove any stale instance first (idempotent re-install).
        & $nssm stop $svc 2>$null
        & $nssm remove $svc confirm 2>$null
        & $nssm install $svc $spec.Exe $spec.Args
        & $nssm set $svc AppDirectory $spec.Wd
        & $nssm set $svc AppStdout $out
        & $nssm set $svc AppStderr $err
        & $nssm set $svc Start SERVICE_AUTO_START
        & $nssm set $svc AppExit Default Restart
        & $nssm set $svc AppRestartDelay 5000
        & $nssm set $svc DisplayName "Chipmo Sentry AI - $name"
        & $nssm start $svc
        Write-Host "  installed + started $svc" -ForegroundColor Green
    }
    else {
        Write-Host ("  [WhatIf] nssm install {0} {1} {2}" -f $svc, $spec.Exe, $spec.Args) -ForegroundColor DarkGray
    }
}

# ingest + ai (long-running) + tunnels (quick-tunnel supervisor + Railway re-wire).
$targets = @('ingest', 'ai', 'tunnels')
if ($TunnelName) { $targets += 'tunnel' }  # optional NAMED tunnel (needs a domain)

foreach ($t in $targets) { Install-One $t }

# The login-startup shortcut is now redundant (services boot-start everything);
# remove it so the two don't double-start the tunnels.
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) 'ChipmoSentryAI.lnk'
if (Test-Path $lnk) {
    Remove-Item $lnk -Force -ErrorAction SilentlyContinue
    Write-Host "  removed redundant login-startup shortcut" -ForegroundColor DarkGray
}

Write-Host "Services ready (ingest, ai, tunnels). Manage via services.msc or" -ForegroundColor Cyan
Write-Host "  nssm start/stop $($cfg.ServicePrefix)<name>   (e.g. $($cfg.ServicePrefix)tunnels)" -ForegroundColor Cyan
Write-Host "Live-video re-wire needs config\railway-token.txt (the project token)." -ForegroundColor Yellow
