"""Per-zone dwell-time accounting.

The tracker already knows how long each person has been in their current zone.
This turns that into zone-level statistics that survive people coming and
going:

  visits          completed stays (someone entered the zone, then left it)
  avg_visit_s     mean length of a completed stay
  max_visit_s     longest completed stay
  occupancy_s     cumulative person-seconds - two people for a minute is 120
  longest_now_s   the longest stay currently in progress

A "visit" closes when the person moves to a different zone, or when the tracker
loses them for longer than `grace` seconds. The grace period matters: detectors
drop people for a frame or two behind a pillar all the time, and without it a
single stay would be chopped into a dozen fragments.

Everything lives in memory and resets when the app restarts. Pass a CSV path or
a store.VisitStore to keep a durable record of completed visits; the two are
independent, so you can have either, both, or neither.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class _OpenVisit:
    zone_id: str
    since: float
    last_seen: float


@dataclass
class ZoneTotals:
    visits: int = 0
    total_visit_s: float = 0.0
    max_visit_s: float = 0.0
    occupancy_s: float = 0.0        # cumulative person-seconds in the zone

    @property
    def avg_visit_s(self) -> float:
        return self.total_visit_s / self.visits if self.visits else 0.0


class ZoneDwell:
    def __init__(self, grace: float = 2.0, csv_path: Optional[str] = None,
                 store=None) -> None:
        self.grace = grace
        self.csv_path = csv_path
        # A store.VisitStore, or None. Kept as a plain attribute rather than
        # imported here so dwell.py still has no dependencies of its own.
        self.store = store
        self._lock = threading.Lock()
        self._open: Dict[int, _OpenVisit] = {}
        self._totals: Dict[str, ZoneTotals] = {}
        self._last_tick: Optional[float] = None
        if csv_path and not os.path.exists(csv_path):
            self._write_csv_header()

    # ------------------------------------------------------------------
    def update(self, people: Sequence[dict], now: Optional[float] = None) -> None:
        """people: dicts with 'id' and 'zone' (zone id or None), as built by the
        pipeline each frame."""
        now = now or time.time()
        with self._lock:
            dt = 0.0 if self._last_tick is None else max(0.0, now - self._last_tick)
            self._last_tick = now

            seen = set()
            for p in people:
                pid, zid = int(p["id"]), p.get("zone")
                seen.add(pid)
                if zid is None:
                    # Left the zones entirely - close any open visit.
                    self._close(pid, now)
                    continue

                # Accumulate occupancy for every person standing in a zone.
                self._totals.setdefault(zid, ZoneTotals()).occupancy_s += dt

                cur = self._open.get(pid)
                if cur is None:
                    self._open[pid] = _OpenVisit(zid, now, now)
                elif cur.zone_id != zid:
                    self._close(pid, now)
                    self._open[pid] = _OpenVisit(zid, now, now)
                else:
                    cur.last_seen = now

            # Anyone missing for longer than the grace period has left.
            for pid in [i for i, v in self._open.items()
                        if i not in seen and now - v.last_seen > self.grace]:
                self._close(pid, now)

    def _close(self, pid: int, now: float) -> None:
        v = self._open.pop(pid, None)
        if v is None:
            return
        duration = max(0.0, v.last_seen - v.since)
        if duration < 0.5:            # ignore a single-frame flicker
            return
        t = self._totals.setdefault(v.zone_id, ZoneTotals())
        t.visits += 1
        t.total_visit_s += duration
        t.max_visit_s = max(t.max_visit_s, duration)
        if self.csv_path:
            self._append_csv(pid, v.zone_id, v.since, v.last_seen, duration)
        if self.store is not None:
            # add_visit() swallows its own errors, for the same reason the CSV
            # append does: logging must never take the tracker down.
            self.store.add_visit(pid, v.zone_id, v.since, v.last_seen, duration)

    def close_open(self, now: Optional[float] = None) -> int:
        """Close every visit still in progress, and return how many.

        Called when the tracker shuts down. Without it, everyone standing in a
        zone at that moment simply vanishes: their visit is only ever written
        when it closes, so the last stay of every person on screen - often the
        longest one of the session - would be lost from the CSV and the
        database alike.
        """
        now = now or time.time()
        with self._lock:
            pids = list(self._open)
            for pid in pids:
                self._close(pid, now)
            return len(pids)

    # ------------------------------------------------------------------
    def snapshot(self, now: Optional[float] = None) -> Dict[str, dict]:
        """Per-zone stats keyed by zone id, safe to serialise."""
        now = now or time.time()
        with self._lock:
            longest_now: Dict[str, float] = {}
            for v in self._open.values():
                d = max(0.0, now - v.since)
                if d > longest_now.get(v.zone_id, 0.0):
                    longest_now[v.zone_id] = d
            out = {}
            for zid, t in self._totals.items():
                out[zid] = {
                    "visits": t.visits,
                    "avg_visit_s": round(t.avg_visit_s, 1),
                    "max_visit_s": round(t.max_visit_s, 1),
                    "occupancy_s": round(t.occupancy_s, 1),
                    "longest_now_s": round(longest_now.get(zid, 0.0), 1),
                }
            for zid, d in longest_now.items():
                out.setdefault(zid, {"visits": 0, "avg_visit_s": 0.0, "max_visit_s": 0.0,
                                     "occupancy_s": 0.0})["longest_now_s"] = round(d, 1)
            return out

    def zone_seconds_for(self, pid: int, now: Optional[float] = None) -> float:
        """How long this person has been in their current zone."""
        now = now or time.time()
        with self._lock:
            v = self._open.get(int(pid))
            return max(0.0, now - v.since) if v else 0.0

    # ------------------------------------------------------------------
    def _write_csv_header(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["entered_at", "left_at", "person_id", "zone_id", "seconds"])

    def _append_csv(self, pid: int, zid: str, since: float,
                    until: float, duration: float) -> None:
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)),
                    pid, zid, round(duration, 1),
                ])
        except OSError:
            pass   # logging must never take the tracker down


def format_duration(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"
