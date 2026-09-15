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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]

# How many frames of pose history a track carries. sequences.py will read
# windows out of this in a later phase; here it just has to be long enough to
# cover the longest window a downstream ST-GCN would want (100 frames is the
# usual ceiling) without letting a long-lived track grow without bound.
POSE_HISTORY = 128


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

    # --- pose (phase 1) ----------------------------------------------------
    # Whatever the model produced on the most recent matched frame, normalised
    # like bbox. None on non-pose backends, and on frames where the detector
    # found a box but no skeleton.
    keypoints: Optional[List[List[float]]] = None
    kp_conf: Optional[List[float]] = None
    # pose.PoseQuality for that same frame, or None. `pose_usable` is the
    # field a downstream sequence writer should filter on.
    pose_quality: Optional[Any] = None
    # Rolling history, appended only on frames where a pose was attached.
    # Entries are (frame_no, keypoints, kp_conf, usable).
    pose_history: Deque = field(default_factory=lambda: deque(maxlen=POSE_HISTORY))

    @property
    def pose_usable(self) -> bool:
        return bool(self.pose_quality is not None and self.pose_quality.usable)

    def attach_pose(self, pose, quality, frame_no: int) -> None:
        """Record one frame's skeleton against this track.

        Called from pipeline.py once the tracker has decided which detection
        this track matched, so the pose lands on the persistent ID rather than
        on a per-frame box. `pose` is the (kp_xy, kp_conf) pair the detector
        emitted, or None.
        """
        if pose is None:
            self.keypoints = self.kp_conf = self.pose_quality = None
            return
        self.keypoints, self.kp_conf = pose
        self.pose_quality = quality
        self.pose_history.append(
            (frame_no, self.keypoints, self.kp_conf, self.pose_usable))

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
        max_coast: int = 0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_center_dist = max_center_dist
        self.max_coast = max_coast
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        # track id -> the pose of the detection it matched this frame.
        self.matched_pose: Dict[int, object] = {}

    def update(self, detections: Sequence[Tuple[BBox, float]],
               keypoints: Optional[Sequence] = None) -> List[Track]:
        """detections: sequence of ((x1, y1, x2, y2) normalised, confidence).

        `keypoints`, when given, is index-aligned with `detections`. Which
        detection a track matched is known only in here, so the pose is stashed
        on `matched_pose` for pipeline.py to attach to the right ID.
        """
        now = time.time()
        self.matched_pose = {}
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
            if keypoints is not None and di < len(keypoints):
                self.matched_pose[tid] = keypoints[di]

        for tid in track_ids:
            if tid not in matched:
                self._tracks[tid].age += 1

        for di in sorted(unmatched_dets):
            bbox, conf = detections[di]
            t = Track(id=self._next_id, bbox=bbox, conf=conf)
            self._tracks[self._next_id] = t
            if keypoints is not None and di < len(keypoints):
                self.matched_pose[self._next_id] = keypoints[di]
            self._next_id += 1

        for tid in [t for t, tr in self._tracks.items() if tr.age > self.max_age]:
            del self._tracks[tid]

        return self.confirmed()

    def confirmed(self) -> List[Track]:
        """Tracks steady enough to show and count.

        `age <= max_coast` keeps someone in the headcount for a few frames
        after the detector loses them. Detectors drop people behind a pillar or
        a passing forklift constantly, and counting strictly on age == 0 makes
        the number visibly jitter (4 - 3 - 4) when nobody actually moved.
        """
        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.age <= self.max_coast]


class PassthroughTracker:
    """Wraps detector-supplied IDs (YOLO/ByteTrack) in the same Track objects,
    so zone dwell timing works identically across backends."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 max_coast: int = 0) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_coast = max_coast
        self._tracks: Dict[int, Track] = {}
        # track id -> the pose of the detection it matched this frame.
        self.matched_pose: Dict[int, object] = {}

    def update(self, detections: Sequence[Tuple[BBox, float, int]],
               keypoints: Optional[Sequence] = None) -> List[Track]:
        """`keypoints`, when given, is index-aligned with `detections`.

        Alignment is free on this path: the pose model emits boxes, IDs and
        keypoints as rows of one result, and detector.py keeps them in step
        through de-duplication.
        """
        now = time.time()
        self.matched_pose = {}
        seen = set()
        for i, (bbox, conf, tid) in enumerate(detections):
            seen.add(tid)
            if keypoints is not None and i < len(keypoints):
                self.matched_pose[tid] = keypoints[i]
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
        return self.confirmed()

    def confirmed(self) -> List[Track]:
        """Same confirmation rule as SimpleTracker.

        This used to return every track with age == 0, which meant min_hits was
        silently ignored on the YOLO backend: one frame of a ByteTrack id on a
        shadow or a stack of pipes was enough to bump the headcount.
        """
        return [t for t in self._tracks.values()
                if t.hits >= self.min_hits and t.age <= self.max_coast]
