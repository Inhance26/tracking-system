"""A small IoU + centroid tracker.

Used when the detector does not supply its own track IDs (the HOG and demo
backends). With the YOLO backend, Ultralytics' ByteTrack supplies IDs and this
tracker is bypassed.

It is deliberately simple: greedy IoU matching, a distance fallback for people
who move fast between frames, and an age-out so someone who leaves the frame
stops being counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


@dataclass
class Track:
    id: int
    bbox: BBox
    conf: float = 0.0
    hits: int = 1
    age: int = 0                     # frames since last matched detection
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    zone_id: Optional[str] = None
    zone_since: float = field(default_factory=time.time)

    @property
    def dwell_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def zone_seconds(self) -> float:
        return max(0.0, self.last_seen - self.zone_since)


class SimpleTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age: int = 30,
        min_hits: int = 3,
        max_center_dist: float = 0.12,  # normalised frame widths
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_center_dist = max_center_dist
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Tuple[BBox, float]]) -> List[Track]:
        """detections: sequence of ((x1, y1, x2, y2) normalised, confidence)."""
        now = time.time()
        track_ids = list(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        matched: Dict[int, int] = {}  # track_id -> detection index

        # Score every (track, detection) pair, then take them greedily best-first.
        pairs: List[Tuple[float, int, int]] = []
        for tid in track_ids:
            tb = self._tracks[tid].bbox
            tc = centroid(tb)
            for di, (db, _conf) in enumerate(detections):
                score = iou(tb, db)
                if score < self.iou_threshold:
                    dc = centroid(db)
                    dist = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5
                    if dist > self.max_center_dist:
                        continue
                    # Distance fallback scores below any real IoU match.
                    score = max(0.0, (1.0 - dist / self.max_center_dist)) * self.iou_threshold
                pairs.append((score, tid, di))

        for score, tid, di in sorted(pairs, key=lambda p: -p[0]):
            if score <= 0 or tid in matched or di not in unmatched_dets:
                continue
            matched[tid] = di
            unmatched_dets.discard(di)

        for tid, di in matched.items():
            bbox, conf = detections[di]
            t = self._tracks[tid]
            t.bbox = bbox
            t.conf = conf
            t.hits += 1
            t.age = 0
            t.last_seen = now

        for tid in track_ids:
            if tid not in matched:
                self._tracks[tid].age += 1

        for di in sorted(unmatched_dets):
            bbox, conf = detections[di]
            t = Track(id=self._next_id, bbox=bbox, conf=conf)
            self._tracks[self._next_id] = t
            self._next_id += 1

        for tid in [t for t, tr in self._tracks.items() if tr.age > self.max_age]:
            del self._tracks[tid]

        return self.confirmed()

    def confirmed(self) -> List[Track]:
        """Tracks steady enough to show and count."""
        return [t for t in self._tracks.values() if t.hits >= self.min_hits and t.age == 0]


class PassthroughTracker:
    """Wraps detector-supplied IDs (YOLO/ByteTrack) in the same Track objects,
    so zone dwell timing works identically across backends."""

    def __init__(self, max_age: int = 30) -> None:
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}

    def update(self, detections: Sequence[Tuple[BBox, float, int]]) -> List[Track]:
        now = time.time()
        seen = set()
        for bbox, conf, tid in detections:
            seen.add(tid)
            t = self._tracks.get(tid)
            if t is None:
                t = Track(id=tid, bbox=bbox, conf=conf)
                self._tracks[tid] = t
            else:
                t.bbox = bbox
                t.conf = conf
                t.hits += 1
                t.age = 0
                t.last_seen = now
        for tid, t in list(self._tracks.items()):
            if tid not in seen:
                t.age += 1
                if t.age > self.max_age:
                    del self._tracks[tid]
        return [t for t in self._tracks.values() if t.age == 0]

    def confirmed(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.age == 0]
