"""Per-camera detection zones (docs/29) — point-in-polygon in normalized space.

The backend forwards each camera's zones to /v1/live/start: a list of
{type, points:[[x,y],...]} polygons in NORMALIZED 0-1 image coords (drawn on the
agent-pc). The worker tests each tracked person's foot point — normalized by the
live frame's own width/height (docs/29 risk #3: denorm against the real frame, not
the camera's nominal resolution) — against these polygons to decide which zone
type(s) the person is currently standing in.

Pure + Tk/cv2-free so it unit-tests headlessly.
"""

from __future__ import annotations

# type -> list of polygons; each polygon is a list of (x, y) in 0-1.
CompiledZones = dict[str, list[list[tuple[float, float]]]]

_VALID_TYPES = frozenset({"exit", "shelf", "checkout", "entrance"})


def compile_zones(zones: list[dict[str, object]] | None) -> CompiledZones:
    """Group raw zone dicts into {type: [polygon, ...]} of (x,y) tuples.

    Lenient: a malformed entry (missing type/points, <3 points, non-numeric or
    out-of-range coord) is skipped, never raised — the backend already validates
    on write, but a partial/garbled payload must not crash the worker."""
    out: CompiledZones = {}
    for z in zones or []:
        if not isinstance(z, dict):
            continue
        ztype = z.get("type")
        if ztype not in _VALID_TYPES:
            continue
        pts_raw = z.get("points")
        if not isinstance(pts_raw, list) or len(pts_raw) < 3:
            continue
        # Reject the WHOLE polygon if ANY vertex is malformed or out of [0,1] —
        # dropping just the bad vertex would silently RESHAPE the operator's zone
        # into a different region (a phantom exit/shelf zone causing false alerts).
        # A wrong zone is worse than a missing one; the backend validates on write,
        # so an out-of-range vertex only reaches here via a garbled payload.
        poly: list[tuple[float, float]] = []
        valid = True
        for p in pts_raw:
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                valid = False
                break
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                valid = False
                break
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                valid = False
                break
            poly.append((x, y))
        if valid and len(poly) >= 3:
            out.setdefault(str(ztype), []).append(poly)
    return out


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (even-odd rule). `poly` is a list of
    (x,y) vertices in the same coordinate space as (x,y)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def zones_at(x: float, y: float, compiled: CompiledZones) -> set[str]:
    """Return the set of zone TYPES whose any polygon contains the point (x,y)
    in normalized 0-1 space (e.g. {"exit"} or {"shelf","checkout"})."""
    hit: set[str] = set()
    for ztype, polys in compiled.items():
        if any(point_in_poly(x, y, poly) for poly in polys):
            hit.add(ztype)
    return hit
