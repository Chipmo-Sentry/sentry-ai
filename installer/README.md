# Chipmo Sentry AI — server installer

`ChipmoSentryAi-Setup.exe` turns a Windows machine (e.g. the Predator laptop
with an RTX GPU) into the AI "center" that connects **out** to the Railway-
hosted backend. One installer sets up three Windows services:

| Service | What |
|---|---|
| `ChipmoSentryAi-ingest` | MediaMTX video fan-out (RTSP→WebRTC/HLS + recording) |
| `ChipmoSentryAi-ai` | sentry-ai live worker + VLM verify (uvicorn) |
| `ChipmoSentryAi-tunnel` | cloudflared (optional — exposes ingest + ai to Railway) |

**Ollama is NOT installed** by this — install it yourself
(https://ollama.com/download, then `ollama pull minicpm-v:8b`). The installer
just points sentry-ai at it via the Ollama URL you provide.

## Get the installer
Push a version tag matching `src/sentry_ai/__init__.py`:
```
git tag v0.1.0 && git push origin v0.1.0
```
GitHub Actions (`.github/workflows/release.yml`) downloads MediaMTX +
cloudflared + NSSM, compiles the Inno Setup script, and publishes
`ChipmoSentryAi-Setup.exe` to the release. Download it from the Releases page.

> Build locally instead: install [Inno Setup 6], drop `mediamtx.exe`,
> `cloudflared.exe`, `nssm.exe` into `installer\bin\`, then
> `iscc /DAppVersion=0.1.0 installer\sentry-ai-server.iss` →
> `dist\ChipmoSentryAi-Setup.exe`.

## Install (on the AI machine)
First, in **superadmin → AI сервер → Холболтын код** generate a **6-digit
pairing code**. Then run the .exe **as administrator** (services need it). The
wizard asks for:
- **Railway backend URL** — e.g. `https://sentry-backend-xxxx.up.railway.app`
- **Pairing code** — the 6 digits from superadmin (blank on an update keeps the
  existing pairing)
- **Ollama URL** — default `http://localhost:11434`
- **cloudflared tunnel name** — optional; leave blank to skip the tunnel service

On Finish it runs first-time setup (visible console): ensures `uv`, runs
`uv sync` (downloads torch/ultralytics — **10–20 min, needs internet**), writes
`app\.env`, **pairs with the backend** (the returned `ai_node` JWT becomes the
node's credential), and installs the services. The node then appears under
**AI сервер** in superadmin (online status, telemetry, config, revoke).

## Manage
```powershell
# From <install>\scripts (or Start Menu shortcuts):
.\server-control.ps1 status     # service states
.\server-control.ps1 health     # HTTP probes (ollama, ingest, ai)
.\server-control.ps1 restart ai
.\server-control.ps1 logs ai
```
Services also appear in `services.msc` as `ChipmoSentryAi-*` (boot-start,
auto-restart on crash). Logs: `<install>\logs\<component>.out/err.log`.

## Updates
```powershell
<install>\scripts\update.ps1        # or Start Menu "Check for updates"
```
It compares the installed `__version__` with the latest GitHub release, and if
newer downloads + launches `ChipmoSentryAi-Setup.exe`. Re-installing over the
top refreshes the bundled binaries + source and re-runs setup (`uv sync` picks
up new deps, services reinstall). The wizard **pre-fills your existing backend
URL / token / Ollama URL** from the current config, so just click through.
A new release is cut by pushing a version tag (see "Get the installer").

## Railway backend env (so it can drive this center)
| Env | Value |
|---|---|
| `SENTRY_AI_URL` | `https://ai.sentry.chipmo.mn` (your tunnel hostname) |
| `MEDIAMTX_API_URL` | `https://mtxapi.sentry.chipmo.mn` |
| `MEDIAMTX_RTSP_URL` | `rtsp://127.0.0.1:8554` |

Pairing means `LIVE_METADATA_SHARED_SECRET` is **no longer required** for paired
nodes — the backend accepts the node's `ai_node` JWT on the live-metadata path.

Frontend: `NEXT_PUBLIC_MEDIAMTX_HLS_BASE` / `_WHEP_BASE` → your `media.*` /
`whep.*` tunnel hostnames.

## cloudflared tunnel
One-time, signed in to Cloudflare (see `config\cloudflared.config.yml.example`
for the exact commands), then put the real config at `<install>\config\
cloudflared.yml` and install with the tunnel name.

## Troubleshooting — `ai` service won't start
The `ai` service runs `uv run` as **LocalSystem** by default. If LocalSystem
can't reach the `uv` install or the project `.venv`, run the service as your
own user instead:
```powershell
<install>\bin\nssm.exe set ChipmoSentryAi-ai ObjectName ".\YourUser" "YourPassword"
<install>\bin\nssm.exe restart ChipmoSentryAi-ai
```
Check `<install>\logs\ai.err.log` for the actual error.

## ⚠ Known gap — L5 breach clip cut
On a live threshold breach the **backend** cuts the clip from its
`MEDIAMTX_RECORDINGS_DIR`. The Railway backend can't read this machine's local
`recordings\`, so saved-clip-on-breach won't work in the split topology until
the cut moves to a local control plane (sentry-ingest `/v1/cut`, TODO). Live
view + alerts still work.
