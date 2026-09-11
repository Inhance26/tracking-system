#!/usr/bin/env python3
"""Measure how many people the tracker reports, so you can tune it with numbers.

The headcount on the dashboard is the end of a chain - model -> confidence ->
de-duplication -> confirmation -> coasting - and any link can be the one that
is off. This runs the real pipeline logic over a clip and reports the count
distribution, so you can compare settings instead of eyeballing the video.

  # what the current settings actually report
  python calibrate.py --frames 400

  # sweep a few confidence thresholds
  python calibrate.py --frames 400 --sweep-conf 0.15,0.25,0.35

  # is the bigger model worth it?
  python calibrate.py --frames 400 --sweep-weights yolov8n.pt,yolov8x.pt

  # check against a known truth for the stretch you sampled
  python calibrate.py --frames 400 --expect 4

Sampling every frame is wasteful for a survey; --stride 5 covers five times
the wall-clock for the same inference budget.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from detector import build_detector  # noqa: E402
from tracker import PassthroughTracker, SimpleTracker  # noqa: E402


class _Cfg:
    """Just enough of app.Config for build_detector()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def run(args, weights: str, conf: float) -> dict:
    cfg = _Cfg(detector=args.detector, weights=weights, conf=conf, imgsz=args.imgsz,
               device=args.device, dedupe_ios=args.dedupe_ios,
               yolox_model=os.path.join(HERE, "models", "yolox_s.onnx"),
               runpod_endpoint=os.environ.get("RUNPOD_ENDPOINT_ID"),
               runpod_api_key=os.environ.get("RUNPOD_API_KEY"),
               runpod_timeout=10.0, demo_detections="")
    detector = build_detector(cfg)
    supplies_ids = getattr(detector, "name", "") == "yolo"
    tracker = (PassthroughTracker(max_age=args.max_age, min_hits=args.min_hits,
                                  max_coast=args.count_coast)
               if supplies_ids else
               SimpleTracker(max_age=args.max_age, min_hits=args.min_hits,
                             max_coast=args.count_coast))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")

    raw_counts, tracked_counts = [], []
    read = 0
    while len(tracked_counts) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if args.stride > 1 and read % args.stride:
            continue
        if args.width and frame.shape[1] != args.width:
            h = int(frame.shape[0] * args.width / frame.shape[1])
            frame = cv2.resize(frame, (args.width, h))

        boxes, ids = detector.detect(frame)
        if supplies_ids:
            tracks = tracker.update(
                [(b, c, i) for (b, c), i in zip(boxes, ids or [])])
        else:
            tracks = tracker.update(boxes)

        raw_counts.append(len(boxes))
        tracked_counts.append(len(tracks))
    cap.release()

    return {"raw": raw_counts, "tracked": tracked_counts}


def report(label: str, res: dict, expect: int | None, stride: int = 1) -> None:
    raw, tracked = res["raw"], res["tracked"]
    if not tracked:
        print(f"{label}: no frames read")
        return
    hist = Counter(tracked)
    spread = " ".join(f"{n}x{hist[n]}" for n in sorted(hist))
    # How often the number changes between consecutive frames: the jitter the
    # user actually sees on the dashboard. Only meaningful at --stride 1 -
    # sampled frames further apart differ because people genuinely moved.
    flips = sum(1 for a, b in zip(tracked, tracked[1:]) if a != b)
    jitter = (f"changes {100*flips/max(1,len(tracked)-1):4.1f}%  "
              if stride == 1 else "")
    line = (f"{label:<34} boxes~{statistics.mean(raw):4.1f}   "
            f"counted~{statistics.mean(tracked):4.1f}  "
            f"range {min(tracked)}-{max(tracked)}  "
            f"{jitter} [{spread}]")
    if expect is not None:
        acc = 100 * sum(1 for c in tracked if c == expect) / len(tracked)
        line += f"  exact={acc:.0f}%"
    print(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=os.path.join(HERE, "cctv_footage.mp4"))
    p.add_argument("--detector", default="yolo",
                   choices=["yolo", "runpod", "yolox", "hog"])
    # Defaults mirror app.Config so a bare run measures what the app does.
    p.add_argument("--weights", default="yolov8s.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--frames", type=int, default=300, help="frames to sample")
    p.add_argument("--stride", type=int, default=1,
                   help="sample every Nth frame to cover more of the clip")
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--count-coast", type=int, default=5)
    p.add_argument("--dedupe-ios", type=float, default=0.6)
    p.add_argument("--expect", type=int, default=None,
                   help="known headcount for the sampled stretch; reports how "
                        "often the tracker hits it exactly")
    p.add_argument("--sweep-conf", default=None, help="e.g. 0.15,0.25,0.35")
    p.add_argument("--sweep-weights", default=None, help="e.g. yolov8n.pt,yolov8x.pt")
    a = p.parse_args(argv)

    confs = ([float(c) for c in a.sweep_conf.split(",")]
             if a.sweep_conf else [a.conf])
    weights = (a.sweep_weights.split(",") if a.sweep_weights else [a.weights])

    print(f"source {os.path.basename(a.source)}  imgsz {a.imgsz}  "
          f"min_hits {a.min_hits}  coast {a.count_coast}  dedupe {a.dedupe_ios}")
    print(f"sampling {a.frames} frames, stride {a.stride}\n")
    for w in weights:
        for c in confs:
            report(f"{os.path.basename(w)} conf={c}", run(a, w, c), a.expect,
                   a.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
