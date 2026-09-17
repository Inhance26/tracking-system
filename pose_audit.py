#!/usr/bin/env python3
"""Measure whether pose is usable on your footage, before you build on it.

The pose model returns 17 keypoints for every person it finds. It does not tell
you which of those skeletons are worth anything. On this clip the people at the
back of the shop are ~12x40 px (see the note in app.py), and a skeleton fitted
to a 40 px-tall blob is noise in the shape of a person. Feed those into an
action-recognition model and you are training it on invented motion.

So this reports, from your own footage rather than from a default:

  * the distribution of person heights, in pixels
  * how many joints the model actually finds at each of those sizes
  * what fraction of detections survive the gate at a range of thresholds

and writes pose_check.jpg so you can look at the skeletons yourself.

  python pose_audit.py
  python pose_audit.py --frames 300 --stride 5
  python pose_audit.py --sweep-min-height 40,60,80,120
  python pose_audit.py --pose-weights yolo11m-pose.pt

Read the height table first. The column that matters is "core joints": once it
falls below about 8 of 12, the skeletons at that size are guesses. Put
--pose-min-height at the boundary where that happens and pass the same number
to app.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pose as pose_mod  # noqa: E402
from detector import build_detector  # noqa: E402

OUT_IMAGE = os.path.join(HERE, "pose_check.jpg")

# Buckets for the height table. Open-ended at the top.
BUCKETS = (0, 40, 60, 80, 120, 200, 10 ** 6)


class _Cfg:
    """Just enough of app.Config for build_detector()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class Sample:
    """One detected person on one frame."""

    __slots__ = ("height_px", "visible_core", "mean_core_conf", "conf")

    def __init__(self, height_px, visible_core, mean_core_conf, conf):
        self.height_px = height_px
        self.visible_core = visible_core
        self.mean_core_conf = mean_core_conf
        self.conf = conf


def collect(args) -> tuple[List[Sample], Optional[tuple], int]:
    """Run the pose model over sampled frames, gathering one Sample per person.

    Returns (samples, best_frame, frames_read) where best_frame is the frame
    with the most people on it, kept for the annotated JPEG.
    """
    cfg = _Cfg(detector="yolo", weights=args.pose_weights, conf=args.conf,
               imgsz=args.imgsz, device=args.device, dedupe_ios=args.dedupe_ios,
               pose=True, pose_weights=args.pose_weights,
               yolox_model="", runpod_endpoint=None, runpod_api_key=None,
               runpod_timeout=10.0, demo_detections="")
    detector = build_detector(cfg)
    if not getattr(detector, "pose", False):
        raise SystemExit(
            f"'{args.pose_weights}' is not a pose model - it returns no "
            "keypoints.\nUse a -pose checkpoint, e.g. yolo11s-pose.pt."
        )

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")

    samples: List[Sample] = []
    best = None
    best_count = -1
    read = sampled = 0

    while sampled < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if args.stride > 1 and read % args.stride:
            continue
        sampled += 1
        if args.width and frame.shape[1] != args.width:
            h = int(frame.shape[0] * args.width / frame.shape[1])
            frame = cv2.resize(frame, (args.width, h))
        fh = frame.shape[0]

        boxes, _ids, kpts = detector.detect(frame)
        for i, (bbox, conf) in enumerate(boxes):
            p = kpts[i] if kpts and i < len(kpts) else None
            kp_conf = p[1] if p else None
            q = pose_mod.assess(bbox, kp_conf, fh,
                                min_height_px=args.min_height,
                                min_kp_conf=args.min_kp_conf,
                                min_core_visible=args.min_core)
            samples.append(Sample(q.height_px, q.visible_core,
                                  q.mean_core_conf, conf))

        if len(boxes) > best_count:
            best_count = len(boxes)
            best = (frame.copy(), boxes, kpts)

    cap.release()
    return samples, best, sampled


def height_table(samples: List[Sample], args) -> None:
    """How many joints the model finds, broken down by how big the person is.

    This is the table the whole script exists for: it shows where skeleton
    quality falls off, which is what the gate should be set to.
    """
    print("  person height      n    core joints found   mean conf   passes gate")
    print("  " + "-" * 68)
    for lo, hi in zip(BUCKETS, BUCKETS[1:]):
        in_bucket = [s for s in samples if lo <= s.height_px < hi]
        if not in_bucket:
            continue
        n = len(in_bucket)
        joints = sum(s.visible_core for s in in_bucket) / n
        conf = sum(s.mean_core_conf for s in in_bucket) / n
        passing = sum(1 for s in in_bucket
                      if s.height_px >= args.min_height
                      and s.visible_core >= args.min_core)
        label = f"{lo}-{hi}px" if hi < 10 ** 6 else f"{lo}px+"
        # 12 is len(pose.CORE_KEYPOINTS) - the face joints are never counted.
        print(f"  {label:<14} {n:>5}    {joints:>5.1f} / 12          "
              f"{conf:>5.2f}      {100 * passing / n:>5.0f}%")


def sweep(samples: List[Sample], heights: List[float], min_core: int) -> None:
    print("\n  if --pose-min-height were:")
    total = len(samples)
    for h in heights:
        keep = sum(1 for s in samples
                   if s.height_px >= h and s.visible_core >= min_core)
        print(f"    {h:>6.0f} px   {keep:>5} of {total} detections usable "
              f"({100 * keep / max(1, total):.0f}%)")


def draw_best(best, args) -> None:
    """Save the busiest sampled frame with its skeletons drawn.

    Pass/fail is drawn in the colour the app uses, so this picture and the live
    video agree about which people the gate is keeping.
    """
    if best is None:
        return
    frame, boxes, kpts = best
    h, w = frame.shape[:2]
    kept = 0
    for i, (bbox, _conf) in enumerate(boxes):
        p = kpts[i] if kpts and i < len(kpts) else None
        q = pose_mod.assess(bbox, p[1] if p else None, h,
                            min_height_px=args.min_height,
                            min_kp_conf=args.min_kp_conf,
                            min_core_visible=args.min_core)
        colour = (235, 235, 235) if q.usable else (110, 110, 110)
        kept += 1 if q.usable else 0
        x1, y1, x2, y2 = [int(round(v * s)) for v, s in
                          zip(bbox, (w, h, w, h))]
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (120, 235, 120) if q.usable else (90, 90, 90), 1)
        if not p:
            continue
        pts = []
        for j, (nx, ny) in enumerate(p[0]):
            seen = j < len(p[1]) and p[1][j] >= args.min_kp_conf
            pts.append((int(round(nx * w)), int(round(ny * h))) if seen else None)
        for a, b in pose_mod.SKELETON:
            if pts[a] and pts[b]:
                cv2.line(frame, pts[a], pts[b], colour, 1, cv2.LINE_AA)
        for pt in pts:
            if pt:
                cv2.circle(frame, pt, 2, colour, -1, cv2.LINE_AA)

    cv2.putText(frame, f"{kept}/{len(boxes)} poses pass the gate", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(OUT_IMAGE, frame)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=os.path.join(HERE, "my_clip.mp4"))
    p.add_argument("--pose-weights", default="yolo11s-pose.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--dedupe-ios", type=float, default=0.6)
    p.add_argument("--frames", type=int, default=60,
                   help="frames to sample (pose is slower than detection, and "
                        "much slower on CPU - start small)")
    p.add_argument("--stride", type=int, default=10,
                   help="sample every Nth frame, to cover more of the clip")
    # Defaults mirror pose.py so a bare run measures what the app would do.
    p.add_argument("--min-height", type=float, default=pose_mod.MIN_HEIGHT_PX)
    p.add_argument("--min-kp-conf", type=float, default=pose_mod.MIN_KP_CONF)
    p.add_argument("--min-core", type=int, default=pose_mod.MIN_CORE_VISIBLE)
    p.add_argument("--sweep-min-height", default="40,60,80,120,200",
                   help="comma-separated heights to report pass rates for")
    a = p.parse_args(argv)

    if not os.path.exists(a.source):
        print(f"\n  Video file not found: {a.source}\n")
        return 1

    print(f"\n  source {os.path.basename(a.source)}  model {a.pose_weights}  "
          f"imgsz {a.imgsz}  width {a.width}")
    print(f"  sampling {a.frames} frames, stride {a.stride}  "
          f"(gate: >={a.min_height:.0f}px and >={a.min_core}/12 joints)\n")

    samples, best, sampled = collect(a)
    if not samples:
        print("  No people detected at all. Try --conf 0.15, or check the clip.")
        return 1

    heights = sorted(s.height_px for s in samples)
    usable = [s for s in samples
              if s.height_px >= a.min_height and s.visible_core >= a.min_core]

    print(f"  {len(samples)} detections over {sampled} frames "
          f"({len(samples) / sampled:.1f} per frame)")
    print(f"  heights: smallest {heights[0]:.0f}px  "
          f"median {heights[len(heights) // 2]:.0f}px  "
          f"tallest {heights[-1]:.0f}px\n")

    height_table(samples, a)
    sweep(samples, [float(x) for x in a.sweep_min_height.split(",")], a.min_core)

    pct = 100 * len(usable) / len(samples)
    print(f"\n  At the current gate, {len(usable)} of {len(samples)} "
          f"detections ({pct:.0f}%) are usable for behaviour data.")
    if pct < 25:
        print("  That is low. Either most of your people are too far from this")
        print("  camera for pose, or the gate is stricter than it needs to be -")
        print("  the height table above says which.")

    draw_best(best, a)
    print(f"  Annotated frame written to {os.path.basename(OUT_IMAGE)} "
          "(white = passes, grey = rejected)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
