"""Zone geometry (docs/29 P1c): compile + point-in-polygon in normalized space."""

from __future__ import annotations

from sentry_ai.live_worker.zones import compile_zones, point_in_poly, zones_at

# A unit square split: left half = shelf, a small exit box top-right.
_SHELF = {"type": "shelf", "points": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]}
_EXIT = {"type": "exit", "points": [[0.8, 0.0], [1.0, 0.0], [1.0, 0.2], [0.8, 0.2]]}


def test_compile_groups_by_type() -> None:
    c = compile_zones([_SHELF, _EXIT])
    assert set(c) == {"shelf", "exit"}
    assert len(c["shelf"][0]) == 4


def test_compile_skips_malformed() -> None:
    c = compile_zones(
        [
            {"type": "shelf", "points": [[0.1, 0.1], [0.2, 0.2]]},  # <3 points
            {"type": "aisle", "points": [[0, 0], [1, 0], [0.5, 1]]},  # unknown type
            {"type": "exit", "points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]]},  # out of range pt
            "not a dict",
        ]
    )
    # The exit polygon drops only its out-of-range vertex (2.0), leaving 2 → skipped.
    assert c == {}


def test_compile_none_is_empty() -> None:
    assert compile_zones(None) == {}


def test_point_in_poly_basic_square() -> None:
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_poly(0.5, 0.5, sq) is True
    assert point_in_poly(1.5, 0.5, sq) is False
    assert point_in_poly(-0.1, 0.5, sq) is False


def test_zones_at_returns_matching_types() -> None:
    c = compile_zones([_SHELF, _EXIT])
    assert zones_at(0.25, 0.5, c) == {"shelf"}  # left half
    assert zones_at(0.9, 0.1, c) == {"exit"}  # exit box
    assert zones_at(0.7, 0.7, c) == set()  # neither


def test_zones_at_empty_when_no_zones() -> None:
    assert zones_at(0.5, 0.5, {}) == set()
