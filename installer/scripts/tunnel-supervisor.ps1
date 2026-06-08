<#
.SYNOPSIS
  Long-running supervisor for the two cloudflared Quick Tunnels (HLS :8888,
  AI :8001) that expose this AI center to the Railway-hosted backend/frontend.

  Quick-Tunnel URLs are EPHEMERAL (change every cloudflared restart). This script
  keeps both tunnels alive and, whenever a URL changes, re-points Railway:
    frontend NEXT_PUBLIC_MEDIAMTX_HLS_BASE / _WHEP_BASE  -> HLS tunnel
    backend  SENTRY_AI_URL                               -> AI tunnel
  and redeploys both (NEXT_PUBLIC_* are build-time). Designed to run as an NSSM
  service (foreground, never exits) so it boot-starts + auto-restarts.

  Railway re-wire needs the project token in config\railway-token.txt. Without it
  the tunnels still stay up; only the Railway re-wire is skipped (logged).
#>
[CmdletBinding()]
param([int]$IntervalSec = 30)

. "$PSScriptRoot\server.config.ps1"
$cfg = $ServerConfig
New-Item -ItemType Directory -Force -Path $cfg.LogDir, $cfg.RunDir | Out-Null

$CF = $cfg.CloudflaredExe
$RW = @{
  Project  = '2fa5edf9-cc0e-4d92-8189-721f8748ef96'
  Env      = '67a3c209-fd62-4266-b400-59289d5166ef'
  Backend  = '938ab195-99f9-4736-a1f4-6eb355c7314f'
  Frontend = 'd2712eff-5d44-4799-8325-117bb06a5bee'
  Api      = 'https://backboard.railway.com/graphql/v2'
}
$BackendRepo  = 'Chipmo-Sentry/sentry-backend'
$FrontendRepo = 'Chipmo-Sentry/sentry-frontend'
$tokenFile    = Join-Path $cfg.ConfigDir 'railway-token.txt'
$stateFile    = Join-Path $cfg.RunDir 'tunnel-urls.txt'

function Log($msg) {
  $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Write-Host $line
  Add-Content -Path (Join-Path $cfg.LogDir 'tunnels.log') -Value $line -ErrorAction SilentlyContinue
}

# A tunnel: a cloudflared --url child whose PID + URL we track.
function Get-Pid($pidFile) {
  if (Test-Path $pidFile) {
    $procId = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { return [int]$procId }
  }
  return $null
}

function Read-Url($logFile) {
  foreach ($f in @($logFile, "$logFile.err")) {
    if (Test-Path $f) {
      $m = Select-String -Path $f -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($m) { return $m.Matches[0].Value }
    }
  }
  return $null
}

# Ensure a cloudflared quick tunnel for $localPort is running; return its URL.
function Ensure-Tunnel([int]$localPort, [string]$tag) {
  $pidFile = Join-Path $cfg.RunDir "cf-$tag.pid"
  $logFile = Join-Path $cfg.LogDir "cf-$tag.out"
  $procId = Get-Pid $pidFile
  if (-not $procId) {
    # (re)start: clear the old log so we read the NEW url
    Remove-Item $logFile, "$logFile.err" -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $CF `
      -ArgumentList ("tunnel --url http://localhost:$localPort --no-autoupdate") `
      -WorkingDirectory $cfg.BinDir -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    Set-Content -Path $pidFile -Value $p.Id -Encoding ascii
    Log "cf-$tag started (pid $($p.Id)) for :$localPort"
    Start-Sleep -Seconds 6
  }
  $url = $null
  for ($i = 0; $i -lt 10 -and -not $url; $i++) { $url = Read-Url $logFile; if (-not $url) { Start-Sleep 2 } }
  return $url
}

function Get-LatestSha([string]$repo) {
  try {
    $r = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits/main" `
      -Headers @{ 'User-Agent' = 'chipmo-supervisor'; 'Accept' = 'application/vnd.github+json' } -TimeoutSec 20
    return $r.sha
  } catch { Log "sha fetch failed for $repo : $($_.Exception.Message)"; return $null }
}

function Rw-Upsert([string]$tok, [string]$svc, [string]$name, [string]$val) {
  $q = 'mutation($i:VariableUpsertInput!){ variableUpsert(input:$i) }'
  $body = @{ query = $q; variables = @{ i = @{ projectId = $RW.Project; environmentId = $RW.Env; serviceId = $svc; name = $name; value = $val } } } | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $RW.Api -Method Post -Headers @{ 'Project-Access-Token' = $tok } -ContentType 'application/json' -Body $body | Out-Null
}

function Rw-Deploy([string]$tok, [string]$svc, [string]$repo) {
  $sha = Get-LatestSha $repo
  if (-not $sha) { return }
  $q = 'mutation($s:String!,$e:String!,$c:String!){ serviceInstanceDeployV2(serviceId:$s,environmentId:$e,commitSha:$c) }'
  $body = @{ query = $q; variables = @{ s = $svc; e = $RW.Env; c = $sha } } | ConvertTo-Json -Depth 6
  Invoke-RestMethod -Uri $RW.Api -Method Post -Headers @{ 'Project-Access-Token' = $tok } -ContentType 'application/json' -Body $body | Out-Null
}

function Rewire([string]$hls, [string]$ai) {
  if (-not (Test-Path $tokenFile)) {
    Log "URLs changed but config\railway-token.txt missing - set Railway envs manually: HLS=$hls AI=$ai"
    return
  }
  # Take only the first whitespace-delimited token, so a trailing name/comment
  # in the file (e.g. '<uuid> "Claude_AI_Predator 4"') doesn't corrupt it.
  $tok = ((Get-Content $tokenFile -Raw) -split '\s+' | Where-Object { $_ } | Select-Object -First 1)
  try {
    Rw-Upsert $tok $RW.Frontend 'NEXT_PUBLIC_MEDIAMTX_HLS_BASE'  $hls
    Rw-Upsert $tok $RW.Frontend 'NEXT_PUBLIC_MEDIAMTX_WHEP_BASE' $hls
    Rw-Upsert $tok $RW.Backend  'SENTRY_AI_URL'                  $ai
    Rw-Deploy $tok $RW.Backend  $BackendRepo
    Rw-Deploy $tok $RW.Frontend $FrontendRepo
    Log "Railway re-wired: HLS=$hls AI=$ai (backend+frontend redeploying)"
  } catch { Log "Railway re-wire failed: $($_.Exception.Message)" }
}

Log "tunnel-supervisor starting (interval ${IntervalSec}s)"
$lastKey = if (Test-Path $stateFile) { (Get-Content $stateFile -Raw).Trim() } else { '' }

while ($true) {
  try {
    $hls = Ensure-Tunnel $cfg.MediaMtxHlsPort 'hls'
    $ai  = Ensure-Tunnel $cfg.AiPort 'ai'
    if ($hls -and $ai) {
      $key = "$hls|$ai"
      if ($key -ne $lastKey) {
        Log "tunnel URLs: HLS=$hls AI=$ai (changed)"
        Rewire $hls $ai
        Set-Content -Path $stateFile -Value $key -Encoding ascii
        $lastKey = $key
      }
    } else {
      Log "waiting for tunnel URL(s): HLS=$hls AI=$ai"
    }
  } catch {
    Log "supervisor loop error: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $IntervalSec
}
