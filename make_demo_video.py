#!/usr/bin/env python3
"""Generate a synthetic overhead 'store' video plus matching ground-truth boxes.

This exists so you can see the whole system working before wiring a real camera
or installing YOLO:

    python make_demo_video.py
    python app.py --source demo_store.mp4 --detector demo

It writes demo_store.mp4, demo_detections.json and a demo zones.json laid out to
match the aisles in the video.
"""

from __future__ import annotations

import json
import math
import os
import random

import cv2
import numpy as np

W, H = 960, 540
FPS = 20
SECONDS = 30
FRAMES = FPS * SECONDS
HERE = os.path.dirname(os.path.abspath(__file__))

FLOOR = (206, 208, 210)
SHELF = (96, 104, 118)
SHELF_TOP = (128, 138, 152)

# Three shelf racks -> three walking aisles between/beside them.
RACKS = [(150, 90, 250, 460), (390, 90, 490, 460), (630, 90, 730, 460)]
AISLE_X = [110, 330, 570, 810]          # walkable lanes (centre x)
AISLE_SPAN = [(60, 148), (252, 388), (492, 628), (732, 900)]


class Walker:
    """Strolls up and down one lane, occasionally switching lanes via the top
    or bottom cross-aisle, and pausing to 'browse'."""

    def __init__(self, rng: random.Random, lane: int | None = None) -> None:
        self.rng = rng
        self.lane = rng.randrange(len(AISLE_X)) if lane is None else lane
        self.x = float(AISLE_X[self.lane] + rng.uniform(-18, 18))
        self.y = float(rng.uniform(120, 430))
        self.vy = rng.choice([-1, 1]) * rng.uniform(0.9, 2.2)
        self.pause = 0
        self.cross = None          # target x while moving along a cross-aisle
        self.w = rng.uniform(30, 38)
        self.h = rng.uniform(62, 76)
        self.phase = rng.uniform(0, math.tau)

    def step(self) -> None:
        if self.pause > 0:
            self.pause -= 1
            self.x += math.sin(self.phase) * 0.25
            self.phase += 0.09
            return

        if self.cross is not None:
            self.x += math.copysign(2.4, self.cross - self.x)
            if abs(self.x - self.cross) < 3:
                self.x = self.cross
                self.cross = None
                self.vy = -self.vy
            return

        self.y += self.vy
        self.x += math.sin(self.phase) * 0.6
        self.phase += 0.05

        if self.y < 105 or self.y > 445:                 # reached a cross-aisle
            self.y = max(105, min(445, self.y))
            if self.rng.random() < 0.6:
                self.lane = self.rng.randrange(len(AISLE_X))
                self.cross = float(AISLE_X[self.lane] + self.rng.uniform(-14, 14))
            else:
                self.vy = -self.vy
        elif self.rng.random() < 0.012:
            self.pause = self.rng.randint(20, 90)        # browsing a shelf

    def bbox(self):
        return (self.x - self.w / 2, self.y - self.h / 2,
                self.x + self.w / 2, self.y + self.h / 2)


def draw_store(frame: np.ndarray) -> None:
    frame[:] = FLOOR
    for i in range(0, W, 48):                            # floor tiling
        cv2.line(frame, (i, 0), (i, H), (196, 198, 200), 1)
    for j in range(0, H, 48):
        cv2.line(frame, (0, j), (W, j), (196, 198, 200), 1)

    cv2.rectangle(frame, (0, 0), (W - 1, 78), (176, 180, 186), -1)     # entrance strip
    cv2.rectangle(frame, (0, 468), (W - 1, H - 1), (176, 180, 186), -1)
    cv2.putText(frame, "ENTRANCE", (24, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (86, 92, 102), 2, cv2.LINE_AA)
    cv2.putText(frame, "CHECKOUT", (W - 220, 508), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (86, 92, 102), 2, cv2.LINE_AA)

    for (x1, y1, x2, y2) in RACKS:
        cv2.rectangle(frame, (x1, y1), (x2, y2), SHELF, -1)
        cv2.rectangle(frame, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), SHELF_TOP, -1)
        for y in range(y1 + 20, y2 - 10, 42):            # stock rows
            cv2.line(frame, (x1 + 10, y), (x2 - 10, y), (108, 116, 130), 2)


def draw_person(frame: np.ndarray, w: Walker) -> None:
    cx, cy = int(w.x), int(w.y)
    cv2.ellipse(frame, (cx, cy + 6), (int(w.w * 0.52), int(w.h * 0.30)),
                0, 0, 360, (60, 64, 72), -1, cv2.LINE_AA)   # shoulders, seen from above
    cv2.circle(frame, (cx, cy - 8), int(w.w * 0.30), (74, 96, 132), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy - 8), int(w.w * 0.30), (40, 50, 66), 1, cv2.LINE_AA)


def default_zones():
    def poly(x1, x2, y1=92, y2=500):
        return [[x1 / W, y1 / H], [x2 / W, y1 / H], [x2 / W, y2 / H], [x1 / W, y2 / H]]
    names = ["Aisle 1 - Produce", "Aisle 2 - Grocery", "Aisle 3 - Dairy", "Aisle 4 - Household"]
    colors = [[255, 176, 59], [94, 197, 116], [86, 122, 240], [216, 130, 240]]
    return {"zones": [
        {"id": f"aisle_{i+1}", "name": names[i], "points": poly(*AISLE_SPAN[i]),
         "color": colors[i]}
        for i in range(len(AISLE_SPAN))
    ]}


def main() -> None:
    rng = random.Random(7)
    # Seed the lanes round-robin so every aisle has traffic in the demo.
    walkers = [Walker(rng, lane=i % len(AISLE_X)) for i in range(10)]

    out_path = os.path.join(HERE, "demo_store.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open an mp4 writer on this machine.")

    frames_boxes = []
    frame = np.zeros((H, W, 3), np.uint8)
    for _ in range(FRAMES):
        draw_store(frame)
        boxes = []
        for wk in walkers:
            wk.step()
            draw_person(frame, wk)
            x1, y1, x2, y2 = wk.bbox()
            boxes.append([round(x1 / W, 4), round(y1 / H, 4),
                          round(x2 / W, 4), round(y2 / H, 4)])
        frames_boxes.append(boxes)
        writer.write(frame)
    writer.release()

    with open(os.path.join(HERE, "demo_detections.json"), "w", encoding="utf-8") as fh:
        json.dump({"frames": frames_boxes}, fh)

    zones_path = os.path.join(HERE, "zones.json")
    if not os.path.exists(zones_path):
        with open(zones_path, "w", encoding="utf-8") as fh:
            json.dump(default_zones(), fh, indent=2)
        print(f"wrote {zones_path} (4 demo aisles)")

    print(f"wrote {out_path}  ({FRAMES} frames, {SECONDS}s)")
    print("next: python app.py --source demo_store.mp4 --detector demo")


if __name__ == "__main__":
    main()
