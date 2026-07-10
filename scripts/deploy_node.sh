#!/usr/bin/env bash
# Deploy the latest sentry-ai main to the vast.ai GPU node and restart the
# supervised service. Run FROM THE NODE (ssh in first) or pipe via ssh:
#
#   ssh -i ~/.ssh/id_ed25519 -p 30186 root@221.150.153.47 'bash -s' < scripts/deploy_node.sh
#
# Safe: touches only /workspace/sentry-ai + the sentry-ai supervisor program
# (mediamtx and vllm keep running; live analysis gap is a few seconds).
set -euo pipefail

cd /workspace/sentry-ai
echo "== before: $(git log --oneline -1)"
git fetch origin
git checkout main
git pull --ff-only origin main
echo "== after:  $(git log --oneline -1)"

# --inexact: add/upgrade lockfile deps WITHOUT removing extraneous packages —
# the node venv carries hand-installed GPU extras (cu128 torch + nvidia-* libs)
# that a plain `uv sync` strips, breaking `import torch` (learned 2026-07-10).
uv sync --inexact --frozen 2>/dev/null || uv sync --inexact

supervisorctl restart sentry-ai
sleep 25

echo "== supervisor:"
supervisorctl status
echo "== demographics + worker log lines (first run downloads ~48 MB of ONNX):"
tail -n 200 /workspace/logs/sentry-ai*.log 2>/dev/null \
  | grep -E "demographics|camera.yolo_stats|uvicorn|Application startup" | tail -n 25
