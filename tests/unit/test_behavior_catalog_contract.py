"""Behavior catalog drift-guard (T17 #2).

The behavior dimension/sequence keys are duplicated as string literals across
three repos and synced BY HAND:

  * sentry-ai      src/sentry_ai/live_worker/behavior.py   (DEFAULT_WEIGHTS,
                   DEFAULT_SEQUENCES — this repo)
  * sentry-backend src/sentry_backend/api/v1/behaviors.py  (BUILTIN_META,
                   SEQUENCE_META)
  * sentry-superadmin BehaviorsPage (display only)

KEEP IN SYNC: sentry-backend has the MIRROR of this file at
tests/unit/test_behavior_catalog_contract.py with the IDENTICAL literal lists
below. If you add/remove/rename a key, update BOTH test files plus
sentry-backend BUILTIN_META/SEQUENCE_META and the superadmin BehaviorsPage.
"""

from __future__ import annotations

from sentry_ai.live_worker.behavior import DEFAULT_SEQUENCES, DEFAULT_WEIGHTS

# Dimension keys with a live detector in sentry-ai (DEFAULT_WEIGHTS).
# Mirrored in sentry-backend test as AI_DIMENSION_KEYS.
AI_DIMENSION_KEYS: list[str] = [
    "bag_interaction",
    "body_block",
    "crouch",
    "item_pickup",
    "loitering",
    "looking_around",
    "pocket_interaction",
    "rapid_movement",
    "wrist_to_torso",
]

# Sequence rule keys (DEFAULT_SEQUENCES). Mirrored in sentry-backend
# SEQUENCE_META and its contract test.
SEQUENCE_KEYS: list[str] = [
    "concealment_sequence",
    "seq_loiter_pickup_conceal",
    "seq_look_pickup",
    "seq_pickup_bag",
    "seq_pickup_wrist",
    "seq_pickup_wrist_bag",
]

_SYNC_MSG = (
    "Behavior catalog key set changed! Update sentry-backend BUILTIN_META/"
    "SEQUENCE_META (api/v1/behaviors.py), the superadmin BehaviorsPage, AND the"
    " mirrored lists in BOTH test_behavior_catalog_contract.py files"
    " (sentry-ai + sentry-backend) together."
)


def test_dimension_keys_match_contract() -> None:
    assert sorted(DEFAULT_WEIGHTS) == AI_DIMENSION_KEYS, _SYNC_MSG


def test_sequence_keys_match_contract() -> None:
    assert sorted(r.key for r in DEFAULT_SEQUENCES) == SEQUENCE_KEYS, _SYNC_MSG


def test_sequence_patterns_reference_known_tokens() -> None:
    # Pattern steps must be dimension keys or the two composite tokens the
    # sequence engine resolves ("concealment", "conceal_hide").
    allowed = set(AI_DIMENSION_KEYS) | {"concealment", "conceal_hide"}
    for rule in DEFAULT_SEQUENCES:
        for step in rule.pattern:
            assert step in allowed, _SYNC_MSG
