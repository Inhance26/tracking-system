"""Aisle zone definitions and point-in-polygon assignment.

Zones are stored in NORMALISED coordinates (0.0 - 1.0) so a zone drawn on a
640x360 snapshot still works if you later feed a 1920x1080 stream.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Sequence, Tuple

# A pleasant, high-contrast palette for aisle overlays (BGR for OpenCV).
DEFAULT_COLORS = [
    (255, 176, 59),   # blue-ish
    (94, 197, 116),   # green
    (86, 122, 240),   # red/coral
    (216, 130, 240),  # violet
    (60, 200, 226),   # amber
    (200, 200, 120),  # teal
]


@dataclass
class Zone:
    id: str
    name: str
    points: List[List[float]] = field(default_factory=list)  # normalised [[x, y], ...]
    color: List[int] = field(default_factory=lambda: [255, 176, 59])  # BGR

    def to_pixels(self, width: int, height: int) -> List[Tuple[int, int]]:
        return [(int(round(x * width)), int(round(y * height))) for x, y in self.points]

    def contains(self, nx: float, ny: float) -> bool:
        return point_in_polygon(nx, ny, self.points)


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Ray-casting test. `poly` is a sequence of (x, y) in the same units as x/y."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        # Does the horizontal ray at `y` cross the edge j->i?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x_cross > x:
                inside = not inside
        j = i
    return inside


def assign_zone(zones: Sequence[Zone], nx: float, ny: float) -> str | None:
    """Return the id of the first zone containing the point, else None.

    Zones are tested in order, so if you draw overlapping zones the earlier one
    in zones.json wins.
    """
    for z in zones:
        if z.contains(nx, ny):
            return z.id
    return None


def load_zones(path: str) -> List[Zone]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    zones_raw = raw.get("zones", raw) if isinstance(raw, dict) else raw
    zones: List[Zone] = []
    for i, z in enumerate(zones_raw):
        zones.append(
            Zone(
                id=str(z.get("id") or f"zone_{i + 1}"),
                name=str(z.get("name") or f"Zone {i + 1}"),
                points=[[float(p[0]), float(p[1])] for p in z.get("points", [])],
                color=list(z.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            )
        )
    return zones


def save_zones(path: str, zones: Sequence[Zone]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"zones": [asdict(z) for z in zones]}, fh, indent=2)
    os.replace(tmp, path)


def zones_from_payload(payload) -> List[Zone]:
    """Build zones from JSON posted by the browser editor (validates shape)."""
    items = payload.get("zones", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("expected a list of zones")
    zones: List[Zone] = []
    for i, z in enumerate(items):
        pts = z.get("points", [])
        if len(pts) < 3:
            raise ValueError(f"zone {i + 1} needs at least 3 points")
        clean = []
        for p in pts:
            x, y = float(p[0]), float(p[1])
            clean.append([min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)])
        zones.append(
            Zone(
                id=str(z.get("id") or f"zone_{i + 1}"),
                name=str(z.get("name") or f"Zone {i + 1}")[:40],
                points=clean,
                color=list(z.get("color") or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
            )
        )
    return zones
