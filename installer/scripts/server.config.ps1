# Shared config for the installed Chipmo Sentry AI server.
# Dot-sourced by the control + service scripts. Layout (under {app}):
#   app\        sentry-ai source (uv project, .venv lives here)
#   bin\        mediamtx.exe, cloudflared.exe, nssm.exe, mediamtx.yml
#   config\     cloudflared.yml (tunnel)
#   scripts\    these scripts
#   logs\       per-component stdout/stderr
#
# AppRoot is resolved from this script's location (scripts\ -> parent).

$here = $PSScriptRoot
$AppRoot = Split-Path -Parent $here

$ServerConfig = @{
    AppRoot        = $AppRoot
    AppSrc         = Join-Path $AppRoot 'app'           # sentry-ai uv project
    BinDir         = Join-Path $AppRoot 'bin'
    ConfigDir      = Join-Path $AppRoot 'config'
    LogDir         = Join-Path $AppRoot 'logs'
    RunDir         = Join-Path $AppRoot 'run'

    MediaMtxExe    = Join-Path $AppRoot 'bin\mediamtx.exe'
    MediaMtxConfig = Join-Path $AppRoot 'bin\mediamtx.yml'
    CloudflaredExe = Join-Path $AppRoot 'bin\cloudflared.exe'
    CloudflaredCfg = Join-Path $AppRoot 'config\cloudflared.yml'
    NssmExe        = Join-Path $AppRoot 'bin\nssm.exe'

    AiHost          = '127.0.0.1'
    AiPort          = 8001
    OllamaPort      = 11434
    MediaMtxApiPort = 9997
    MediaMtxHlsPort = 8888

    ServicePrefix  = 'ChipmoSentryAi-'
    TunnelName     = 'sentry-ingest'   # overwritten by setup if user sets one
}

# Components this installer manages. 'tunnel' is optional (only if a cloudflared
# tunnel name is configured). Ollama is the user's own install - not managed.
$ServerComponents = @('ingest', 'ai', 'tunnel')
