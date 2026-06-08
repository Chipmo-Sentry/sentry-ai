"""Entrypoint for the standalone heartbeat process.

Spawned by the app lifespan via `python -m sentry_ai.heartbeat_cli`. Runs the
heartbeat loop in its own OS process so the GPU live-workers in the main app
cannot starve or wedge it (see heartbeat.py for why an in-process thread fails).
Reads worker counts from the main app over HTTP.
"""

from sentry_ai.heartbeat import run_forever
from sentry_ai.logging_setup import configure_logging


def main() -> None:
    configure_logging()
    run_forever()


if __name__ == "__main__":
    main()
