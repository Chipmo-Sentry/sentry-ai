"""Unit tests for the Jinja prompt loader."""

from sentry_ai.pipeline.prompt import render_prompt


def test_render_verify_v1_default_context() -> None:
    rendered = render_prompt("verify_v1.j2")
    assert "browsing" in rendered
    assert "cart_pickup" in rendered
    assert "pocket_conceal" in rendered
    assert "bag_conceal" in rendered
    assert "other" in rendered
    # Default store context substituted
    assert "жирийн ритейл" in rendered


def test_render_verify_v1_custom_store_context() -> None:
    rendered = render_prompt(
        "verify_v1.j2",
        store_context="Дэлгүүр: хүүхдийн дэлгүүр, family group түгээмэл.",
    )
    assert "хүүхдийн дэлгүүр" in rendered
    # Default fallback should be absent when custom context provided
    assert "жирийн ритейл" not in rendered
