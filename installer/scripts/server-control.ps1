<#
.SYNOPSIS
  Manage the installed Chipmo Sentry AI server services.
.EXAMPLE
  .\server-control.ps1 status
  .\server-control.ps1 health
  .\server-control.ps1 start          # all
  .\server-control.ps1 restart ai
  .\server-control.ps1 logs ingest
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet('start', 'stop', 'restart', 'status', 'health', 'logs')]
    [string]$Action = 'status',
    [Parameter(Position = 1)][ValidateSet('all', 'ingest', 'ai', 'tunnel')]
    [string]$Component = 'all'
)

. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig

function Targets { if ($Component -eq 'all') { return $ServerComponents } else { return @($Component) } }
function SvcName([string]$n) { return "$($cfg.ServicePrefix)$n" }

function Svc-Exists([string]$n) {
    return [bool](Get-Service -Name (SvcName $n) -ErrorAction SilentlyContinue)
}

function Do-Service([string]$verb, [string]$n) {
    $svc = SvcName $n
    if (-not (Svc-Exists $n)) { Write-Host ("  {0,-7} not installed" -f $n) -ForegroundColor DarkGray; return }
    try {
        if ($verb -eq 'start') { Start-Service $svc -ErrorAction Stop }
        elseif ($verb -eq 'stop') { Stop-Service $svc -Force -ErrorAction Stop }
        Write-Host ("  {0,-7} {1}" -f $n, $verb) -ForegroundColor Green
    }
    catch { Write-Host ("  {0,-7} {1} failed: {2}" -f $n, $verb, $_.Exception.Message) -ForegroundColor Red }
}

function Show-Status {
    Write-Host "Services:" -ForegroundColor Cyan
    foreach ($n in $ServerComponents) {
        if (-not (Svc-Exists $n)) { Write-Host ("  {0,-7} (not installed)" -f $n) -ForegroundColor DarkGray; continue }
        $s = Get-Service (SvcName $n)
        $color = if ($s.Status -eq 'Running') { 'Green' } else { 'Yellow' }
        Write-Host ("  {0,-7} {1}" -f $n, $s.Status) -ForegroundColor $color
    }
    $o = Get-Service -Name 'Ollama*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($o) { Write-Host ("  {0,-7} {1} (external)" -f 'ollama', $o.Status) -ForegroundColor DarkCyan }
}

function Test-Endpoint([string]$url) {
    try { $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3; return ($r.StatusCode -lt 500) }
    catch { if ($_.Exception.Response) { return $true }; return $false }
}

function Show-Health {
    $probes = @{
        ingest = "http://127.0.0.1:$($cfg.MediaMtxApiPort)/v3/config/global/get"
        ai     = "http://$($cfg.AiHost):$($cfg.AiPort)/healthz"
        ollama = "http://127.0.0.1:$($cfg.OllamaPort)/api/tags"
    }
    Write-Host "Health:" -ForegroundColor Cyan
    foreach ($k in @('ollama', 'ingest', 'ai')) {
        $ok = Test-Endpoint $probes[$k]
        $tag = if ($ok) { 'OK  ' } else { 'DOWN' }
        $color = if ($ok) { 'Green' } else { 'Red' }
        Write-Host ("  {0,-7} {1}  {2}" -f $k, $tag, $probes[$k]) -ForegroundColor $color
    }
}

function Show-Logs([string]$n) {
    if ($n -eq 'all') { Write-Host "Pick one: .\server-control.ps1 logs ai" -ForegroundColor Yellow; return }
    foreach ($suffix in @('err', 'out')) {
        $f = Join-Path $cfg.LogDir "$n.$suffix.log"
        if (Test-Path $f) { Write-Host "==== $f (last 30) ====" -ForegroundColor Cyan; Get-Content $f -Tail 30 }
    }
}

switch ($Action) {
    'start' { foreach ($t in Targets) { Do-Service 'start' $t } }
    'stop' { foreach ($t in Targets) { Do-Service 'stop' $t } }
    'restart' { foreach ($t in Targets) { Do-Service 'stop' $t }; Start-Sleep 1; foreach ($t in Targets) { Do-Service 'start' $t } }
    'status' { Show-Status }
    'health' { Show-Health }
    'logs' { Show-Logs $Component }
}
