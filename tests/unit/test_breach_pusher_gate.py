from sentry_ai.live_worker.breach_pusher import _should_push_theft_clip
from sentry_ai.schemas.vlm_output import Category, VLMOutput


def test_only_theft_or_attempt_verdicts_are_pushed() -> None:
    assert _should_push_theft_clip(
        VLMOutput(actions=[Category.pocket_conceal], confidence=0.50, reasoning="оролдсон")
    )
    assert _should_push_theft_clip(
        VLMOutput(actions=[Category.bag_conceal], confidence=0.72, reasoning="цүнхэнд нуув")
    )


def test_benign_or_weak_verdicts_are_not_pushed() -> None:
    assert not _should_push_theft_clip(
        VLMOutput(actions=[Category.browsing], confidence=0.95, reasoning="үзэж байна")
    )
    assert not _should_push_theft_clip(
        VLMOutput(actions=[Category.cart_pickup], confidence=0.95, reasoning="ил тавив")
    )
    assert not _should_push_theft_clip(
        VLMOutput(actions=[Category.pocket_conceal], confidence=0.49, reasoning="сул")
    )
