"""pair._update_env — .env key upsert (pure, no network)."""

from __future__ import annotations

from pathlib import Path

from sentry_ai.pair import _update_env


def test_creates_env_when_missing(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    _update_env(p, {"SENTRY_BACKEND_URL": "https://b", "AI_NODE_ID": "n1"})
    text = p.read_text(encoding="utf-8")
    assert "SENTRY_BACKEND_URL=https://b" in text
    assert "AI_NODE_ID=n1" in text


def test_replaces_existing_key_preserves_others(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text(
        "OLLAMA_BASE_URL=http://localhost:11434\nSENTRY_BACKEND_SERVICE_TOKEN=old\n",
        encoding="utf-8",
    )
    _update_env(p, {"SENTRY_BACKEND_SERVICE_TOKEN": "newjwt"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert "OLLAMA_BASE_URL=http://localhost:11434" in lines
    assert "SENTRY_BACKEND_SERVICE_TOKEN=newjwt" in lines
    # No duplicate token line.
    assert sum(1 for ln in lines if ln.startswith("SENTRY_BACKEND_SERVICE_TOKEN=")) == 1


def test_appends_new_keys(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n", encoding="utf-8")
    _update_env(p, {"AI_NODE_ID": "abc"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert "FOO=bar" in lines
    assert "AI_NODE_ID=abc" in lines
