<#
.SYNOPSIS
  Stop + remove the Chipmo Sentry AI server services. Run ELEVATED.
  Called by the installer's uninstaller, or manually.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
$nssm = $cfg.NssmExe
if (-not (Test-Path $nssm)) {
    $c = Get-Command nssm.exe -ErrorAction SilentlyContinue
    if ($c) { $nssm = $c.Source } else { Write-Warning "nssm.exe not found; nothing to remove."; return }
}

foreach ($name in $ServerComponents) {
    $svc = "$($cfg.ServicePrefix)$name"
    if ($PSCmdlet.ShouldProcess($svc, "nssm remove")) {
        & $nssm stop $svc 2>$null
        & $nssm remove $svc confirm 2>$null
        Write-Host "  removed $svc" -ForegroundColor Green
    }
}
