"""The capture -> detect -> track -> assign-zone -> annotate loop.

Runs on a background thread and publishes two things behind a lock:
  * the latest annotated frame, already JPEG-encoded (for the MJPEG stream)
  * the latest stats dict (for /api/stats)

The web server only ever reads those two, so a slow browser never stalls
video processing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

import pose as pose_mod
from detector import build_detector
from dwell import ZoneDwell, format_duration
from store import VisitStore
from tracker import PassthroughTracker, SimpleTracker
from zones import Zone, assign_zone, load_zones

FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_COLOR = (120, 235, 120)   # BGR, green like the reference overlay
LINGER_COLOR = (70, 190, 245)    # amber - someone lingering past --long-dwell
TEXT_COLOR = (20, 20, 20)
# Skeletons are drawn in two colours so the quality gate is visible on the
# video itself: white means the pose passed and a behaviour model could use it,
# grey means it was found but rejected as too small or too uncertain.
POSE_OK_COLOR = (235, 235, 235)
POSE_WEAK_COLOR = (110, 110, 110)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.raw_jpeg: Optional[bytes] = None   # un-annotated, for the zone editor
        self.stats: Dict = {
            "running": False, "fps": 0.0, "total_in_store": 0,
            "zones": [], "people": [], "unassigned": 0, "frame": 0,
        }
        self.frame_size = (0, 0)

    def publish(self, jpeg: bytes, raw_jpeg: bytes, stats: Dict, size) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.raw_jpeg = raw_jpeg
            self.stats = stats
            self.frame_size = size

    def read_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.jpeg

    def read_raw_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.raw_jpeg

    def read_stats(self) -> Dict:
        with self.lock:
            return dict(self.stats)


def open_capture(source: str) -> cv2.VideoCapture:
    """Accepts a file path, an rtsp:// URL, or a webcam index like '0'."""
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


class Pipeline(threading.Thread):
    daemon = True

    def __init__(self, cfg, state: SharedState) -> None:
        super().__init__(name="pipeline")
        self.cfg = cfg
        self.state = state
        self.zones: List[Zone] = load_zones(cfg.zones_path)
        self._zones_lock = threading.Lock()
        self._stop = threading.Event()
        self._fps_hist = deque(maxlen=30)
        db_path = getattr(cfg, "db", None)
        self.store = (VisitStore(db_path, camera_id=getattr(cfg, "camera_id", "cam1"))
                      if db_path else None)
        self.dwell = ZoneDwell(csv_path=getattr(cfg, "dwell_csv", None),
                               store=self.store)

    # -- zones can be re-saved from the browser editor while we run -----------
    def reload_zones(self) -> None:
        with self._zones_lock:
            self.zones = load_zones(self.cfg.zones_path)

    def current_zones(self) -> List[Zone]:
        with self._zones_lock:
            return list(self.zones)

    def stop(self) -> None:
        self._stop.set()

    # -----------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        detector = build_detector(cfg)
        supplies_ids = getattr(detector, "name", "") == "yolo"
        # Set by the detector from the loaded model's task, not from the flag,
        # so this is true only when keypoints will really arrive.
        pose_on = bool(getattr(detector, "pose", False))
        coast = int(getattr(cfg, "count_coast", 0))
        simple = SimpleTracker(max_age=cfg.max_age, min_hits=cfg.min_hits,
                               max_coast=coast)
        passthrough = PassthroughTracker(max_age=cfg.max_age, min_hits=cfg.min_hits,
                                         max_coast=coast)

        cap = open_capture(cfg.source)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        is_file = not str(cfg.source).isdigit() and "://" not in str(cfg.source)
        target_dt = (1.0 / src_fps) if (is_file and 0 < src_fps < 120) else 0.0

        frame_no = 0
        skip = max(1, int(cfg.frame_skip))
        prev_start = None

        while not self._stop.is_set():
            t0 = time.time()
            if prev_start is not None:
                period = t0 - prev_start
                self._fps_hist.append(1.0 / period if period > 0 else 0.0)
            prev_start = t0
            ok, frame = cap.read()
            if not ok:
                if is_file and cfg.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_no += 1
            if cfg.width and frame.shape[1] != cfg.width:
                h = int(frame.shape[0] * cfg.width / frame.shape[1])
                frame = cv2.resize(frame, (cfg.width, h))
            h, w = frame.shape[:2]

            if frame_no % skip == 0 or frame_no == 1:
                boxes, ids, kpts = detector.detect(frame)
                if supplies_ids:
                    # ByteTrack occasionally reports boxes before it has assigned
                    # IDs (ids is None). Skip those few frames rather than mixing
                    # two ID spaces - the same people reappear a frame later.
                    pairs = [(b, c, i) for (b, c), i in zip(boxes, ids or [])]
                    active = passthrough
                    tracks = passthrough.update(pairs, keypoints=kpts)
                else:
                    active = simple
                    tracks = simple.update(boxes, keypoints=kpts)
                if pose_on:
                    self._attach_poses(tracks, active.matched_pose, frame_no, h)
            else:
                # A skipped frame produces no new pose. Tracks keep the last one
                # they were given rather than appending a duplicate to their
                # history, which would put a stall into the sequence a
                # behaviour model later reads.
                tracks = (passthrough if supplies_ids else simple).confirmed()

            zones = self.current_zones()
            zone_counts = {z.id: 0 for z in zones}
            unassigned = 0
            people = []
            now = time.time()

            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                # Feet, not the middle of the body: which floor tile they stand on.
                foot_x, foot_y = (x1 + x2) / 2.0, y2
                zid = assign_zone(zones, foot_x, foot_y)
                if zid != t.zone_id:
                    t.zone_id = zid
                    t.zone_since = now
                if zid is None:
                    unassigned += 1
                else:
                    zone_counts[zid] += 1
                person = {
                    "id": int(t.id),
                    "zone": zid,
                    "dwell_s": round(t.dwell_seconds, 1),
                    "zone_s": round(t.zone_seconds, 1),
                    "bbox": [round(float(v), 4) for v in t.bbox],
                }
                if pose_on:
                    # A quality summary only. The 17 keypoints themselves are
                    # deliberately NOT published here - /api/stats is polled by
                    # every open dashboard, and shipping full skeletons through
                    # it would cost far more bandwidth than the dashboard has
                    # any use for. sequences.py writes them to disk in a later
                    # phase, which is where a behaviour model should read them.
                    q = t.pose_quality
                    person["pose"] = q.as_dict() if q is not None else None
                people.append(person)

            self.dwell.update(people, now)
            dwell_stats = self.dwell.snapshot(now)
            zone_time = {p["id"]: self.dwell.zone_seconds_for(p["id"], now)
                         for p in people}
            for p in people:
                p["zone_s"] = round(zone_time[p["id"]], 1)

            annotated = self._annotate(frame, tracks, zones, zone_counts,
                                       len(tracks), zone_time)

            fps = (sum(self._fps_hist) / len(self._fps_hist)) if self._fps_hist else 0.0

            pose_block = None
            if pose_on:
                with_pose = [t for t in tracks if t.pose_quality is not None]
                usable = [t for t in with_pose if t.pose_usable]
                pose_block = {
                    "enabled": True,
                    "tracked": len(tracks),
                    "with_pose": len(with_pose),
                    "usable": len(usable),
                    "min_height_px": round(float(self.cfg.pose_min_height), 1),
                    "min_kp_conf": round(float(self.cfg.pose_min_kp_conf), 2),
                    "min_core_visible": int(self.cfg.pose_min_core),
                }

            stats = {
                "running": True,
                "fps": round(fps, 1),
                "total_in_store": len(tracks),
                "unassigned": unassigned,
                "frame": frame_no,
                "pose": pose_block,
                "timestamp": now,
                "long_dwell_s": float(getattr(self.cfg, "long_dwell", 0) or 0),
                "zones": [
                    {
                        "id": z.id,
                        "name": z.name,
                        "count": zone_counts.get(z.id, 0),
                        "color": [z.color[2], z.color[1], z.color[0]],  # RGB for the browser
                        **dwell_stats.get(z.id, {
                            "visits": 0, "avg_visit_s": 0.0, "max_visit_s": 0.0,
                            "occupancy_s": 0.0, "longest_now_s": 0.0,
                        }),
                    }
                    for z in zones
                ],
                "people": sorted(people, key=lambda p: -p["zone_s"])[:60],
            }

            enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
            ok1, buf1 = cv2.imencode(".jpg", annotated, enc)
            ok2, buf2 = cv2.imencode(".jpg", frame, enc)
            if ok1 and ok2:
                self.state.publish(buf1.tobytes(), buf2.tobytes(), stats, (w, h))

            if target_dt:
                sleep = target_dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        cap.release()
        # Anyone still standing in a zone has a visit that has not been written
        # yet. Close them before the store goes away, or the last - and often
        # longest - stay of every person on screen is lost.
        closed = self.dwell.close_open()
        if self.store is not None:
            self.store.close()
            print(f"  [store] {self.store.rows_written} zone visits written to "
                  f"{self.cfg.db}  ({closed} still open at shutdown)")
        s = self.state.read_stats()
        s["running"] = False
        self.state.publish(self.state.read_jpeg(), self.state.read_raw_jpeg(),
                           s, self.state.frame_size)

    # -----------------------------------------------------------------------
    def _attach_poses(self, tracks, matched_pose, frame_no: int, frame_h: int) -> None:
        """Hang this frame's skeletons on the track IDs they belong to.

        `matched_pose` is track id -> the detector's (keypoints, confidences)
        for the detection that track matched. The tracker fills it in because
        it is the only place that knows which detection went with which ID;
        doing it here instead would mean re-deriving a matching that has
        already been done.

        frame_h is the height of the frame the MODEL saw, after any --width
        resize, so the gate measures the pixels actually available to it.
        """
        for t in tracks:
            p = matched_pose.get(int(t.id))
            quality = None
            if p is not None:
                quality = pose_mod.assess(
                    t.bbox, p[1], frame_h,
                    min_height_px=self.cfg.pose_min_height,
                    min_kp_conf=self.cfg.pose_min_kp_conf,
                    min_core_visible=self.cfg.pose_min_core,
                )
            t.attach_pose(p, quality, frame_no)

    # -----------------------------------------------------------------------
    def _annotate(self, frame, tracks, zones, zone_counts, total, zone_time=None):
        h, w = frame.shape[:2]
        out = frame.copy()

        # Translucent zone fills, then crisp outlines on top.
        if zones:
            overlay = out.copy()
            for z in zones:
                pts = np.array(z.to_pixels(w, h), dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], tuple(int(c) for c in z.color))
            cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

            for z in zones:
                pts = np.array(z.to_pixels(w, h), dtype=np.int32)
                if len(pts) < 3:
                    continue
                color = tuple(int(c) for c in z.color)
                cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
                n = zone_counts.get(z.id, 0)
                label = f"{z.name}: {n}"
                anchor = pts[pts[:, 1].argmin()]
                _draw_label(out, label, (int(anchor[0]), max(18, int(anchor[1]) - 8)), color)

        zone_time = zone_time or {}
        long_dwell = float(getattr(self.cfg, "long_dwell", 120) or 0)
        for t in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in
                              (t.bbox[0] * w, t.bbox[1] * h, t.bbox[2] * w, t.bbox[3] * h)]
            secs = zone_time.get(int(t.id), 0.0)
            # Amber once someone has been standing in one zone a long time.
            lingering = t.zone_id is not None and long_dwell and secs >= long_dwell
            colour = LINGER_COLOR if lingering else PERSON_COLOR
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
            cv2.circle(out, ((x1 + x2) // 2, y2), 3, colour, -1, cv2.LINE_AA)
            if t.keypoints:
                _draw_skeleton(out, t.keypoints, t.kp_conf, w, h, t.pose_usable,
                               float(getattr(self.cfg, "pose_min_kp_conf",
                                             pose_mod.MIN_KP_CONF)))
            label = f"#{t.id}  {format_duration(secs)}" if t.zone_id else f"#{t.id}"
            _draw_label(out, label, (x1, max(16, y1 - 6)), colour)

        banner = f"PEOPLE: {total}"
        cv2.rectangle(out, (0, 0), (w, 34), (28, 26, 24), -1)
        cv2.putText(out, banner, (12, 24), FONT, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
        return out


def _draw_skeleton(img, keypoints, kp_conf, w: int, h: int,
                   usable: bool, min_kp_conf: float) -> None:
    """Draw one person's bones and joints over the frame.

    Joints below `min_kp_conf` are not drawn, and a bone is drawn only when
    both of its ends are. A skeleton with limbs missing is an honest picture of
    what the model found; one drawn through low-confidence guesses looks
    convincing and is not - which matters here, because the whole point of
    phase 1 is judging by eye whether these poses are good enough to use.
    """
    conf = kp_conf or [1.0] * len(keypoints)
    colour = POSE_OK_COLOR if usable else POSE_WEAK_COLOR
    pts = []
    for i, (nx, ny) in enumerate(keypoints):
        seen = i < len(conf) and conf[i] >= min_kp_conf
        pts.append((int(round(nx * w)), int(round(ny * h))) if seen else None)

    for a, b in pose_mod.SKELETON:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], colour, 1, cv2.LINE_AA)
    for pt in pts:
        if pt:
            cv2.circle(img, pt, 2, colour, -1, cv2.LINE_AA)


def _fmt_dwell(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        return f"{m // 60}h {m % 60}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _draw_label(img, text: str, org, color) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, 0.48, 1)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 8))
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 8, y + base - 1), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
