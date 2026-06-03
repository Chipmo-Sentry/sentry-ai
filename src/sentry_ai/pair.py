"""Pair this AI node with the backend using a 6-digit code from superadmin.

Run on the installed machine (the installer calls this):

    python -m sentry_ai.pair --code 123456 \
        --backend https://sentry-backend-xxxx.up.railway.app \
        --public-url https://ai.sentry.chipmo.mn

On success it writes SENTRY_BACKEND_URL, SENTRY_BACKEND_SERVICE_TOKEN (the
returned ai_node JWT) and AI_NODE_ID into the target .env file, so the running
service picks them up on (re)start.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import httpx

from sentry_ai import __version__


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        pass
    return None


def _update_env(env_path: Path, values: dict[str, str]) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in values:
            out.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            out.append(line)
    for k, v in values.items():
        if k not in seen:
            out.append(f"{k}={v}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentry-ai pair")
    p.add_argument("--code", required=True, help="6-digit pairing code from superadmin")
    p.add_argument("--backend", required=True, help="Backend base URL (Railway)")
    p.add_argument("--name", default=None, help="Friendly node name")
    p.add_argument("--public-url", default=None, help="This node's public sentry-ai URL")
    p.add_argument("--env-file", default=".env", help="Path to .env to update")
    args = p.parse_args(argv)

    backend = args.backend.rstrip("/")
    payload = {
        "code": args.code.strip(),
        "name": args.name or socket.gethostname(),
        "hostname": socket.gethostname(),
        "public_url": args.public_url,
        "version": __version__,
        "gpu": _gpu_name(),
    }
    try:
        resp = httpx.post(f"{backend}/api/v1/ai-nodes/pair", json=payload, timeout=20.0)
    except httpx.HTTPError as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        return 2
    if resp.status_code != 200:
        print(f"ERROR: pairing failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return 1

    data = resp.json()
    token = data["ai_node_token"]
    node_id = data["ai_node_id"]
    _update_env(
        Path(args.env_file),
        {
            "SENTRY_BACKEND_URL": backend,
            "SENTRY_BACKEND_SERVICE_TOKEN": token,
            "AI_NODE_ID": node_id,
        },
    )
    print(f"Paired successfully. Node id: {node_id}")
    print(f"Wrote credentials to {args.env_file}. Restart the sentry-ai service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
