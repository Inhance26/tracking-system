#!/usr/bin/env python3
"""Self-test: is YOLO actually working on this machine and this footage?

    python check_setup.py

Checks every dependency, loads the YOLO model, runs it over a sample of frames
from cctv_footage.mp4, reports how many people it found, and writes
yolo_check.jpg so you can see the boxes for yourself.

Run this once after installing. If it passes, `python app.py` will work.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "cctv_footage.mp4")
SAMPLE_FRAMES = 12
# Kept in step with app.Config.weights / .imgsz by hand - importing app here
# would pull in the whole pipeline just to read two constants.
WEIGHTS = "yolov8s.pt"
IMGSZ = 1280
OUT_IMAGE = os.path.join(HERE, "yolo_check.jpg")

OK, BAD, INFO = "  [ok] ", "  [!!] ", "       "


def main() -> int:
    print("\n  Setup check")
    print("  " + "-" * 46)
    print(f"{INFO}Python {sys.version.split()[0]}")

    # ---- core dependencies ------------------------------------------------
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(f"{BAD}opencv-python / numpy are missing.")
        print(f"{INFO}Fix:  pip install -r requirements.txt")
        return 1
    print(f"{OK}opencv-python {cv2.__version__}")
    print(f"{OK}numpy {np.__version__}")

    # ---- the video --------------------------------------------------------
    if not os.path.exists(VIDEO):
        print(f"{BAD}cctv_footage.mp4 is not in this folder.")
        return 1
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"{BAD}OpenCV cannot open cctv_footage.mp4 (corrupt or missing codec).")
        return 1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{OK}cctv_footage.mp4  {w}x{h}, {fps:.1f} fps, {n} frames")

    # ---- YOLO -------------------------------------------------------------
    try:
        import torch
        print(f"{OK}torch {torch.__version__}"
              f"  ({'CUDA GPU available' if torch.cuda.is_available() else 'CPU only'})")
    except ImportError:
        torch = None
        print(f"{INFO}torch not importable on its own (ultralytics may still bundle it)")

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"{OK}ultralytics {ultralytics.__version__}")
    except ImportError:
        cap.release()
        print(f"{BAD}ultralytics is NOT installed - YOLO cannot run.")
        print(f"{INFO}Fix:  pip install ultralytics")
        print(f"{INFO}Meanwhile the app still runs with:  python app.py --detector hog")
        return 1

    # Check the weights app.py actually defaults to, not a different model -
    # a setup that passes here should be a setup the app runs on.
    print(f"{INFO}loading {WEIGHTS} (downloads on first run)...")
    try:
        model = YOLO(WEIGHTS)
    except Exception as exc:  # noqa: BLE001 - any failure here is worth showing plainly
        cap.release()
        print(f"{BAD}Could not load the model: {exc}")
        print(f"{INFO}Usually no internet on the first run. Connect and retry, or")
        print(f"{INFO}download {WEIGHTS} manually into this folder.")
        return 1
    print(f"{OK}model loaded")

    # ---- run it on real frames -------------------------------------------
    print(f"{INFO}running detection on {SAMPLE_FRAMES} frames...")
    step = max(1, n // SAMPLE_FRAMES)
    total_people = 0
    frames_with_people = 0
    heights = []
    best_frame, best_count = None, -1
    t0 = time.time()

    for k in range(SAMPLE_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k * step)
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[0], conf=0.25, imgsz=IMGSZ, verbose=False)
        boxes = res[0].boxes
        count = 0 if boxes is None else len(boxes)
        total_people += count
        frames_with_people += 1 if count else 0
        if boxes is not None and count:
            for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
                heights.append(int(y2 - y1))
        if count > best_count:
            best_count = count
            best_frame = (frame.copy(), boxes)
    elapsed = time.time() - t0
    cap.release()

    # ---- save a picture of the best frame ---------------------------------
    if best_frame is not None:
        img, boxes = best_frame
        if boxes is not None:
            for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, (120, 235, 120), 2)
        cv2.putText(img, f"YOLO found {best_count}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(OUT_IMAGE, img)

    # ---- verdict ----------------------------------------------------------
    print("  " + "-" * 46)
    per_frame = total_people / max(1, SAMPLE_FRAMES)
    speed = SAMPLE_FRAMES / elapsed if elapsed else 0
    print(f"{INFO}people found: {total_people} across {SAMPLE_FRAMES} frames"
          f"  (avg {per_frame:.1f}/frame, best frame {best_count})")
    if heights:
        heights.sort()
        print(f"{INFO}person heights: smallest {heights[0]}px, "
              f"median {heights[len(heights)//2]}px, tallest {heights[-1]}px")
    print(f"{INFO}speed: {speed:.1f} frames/sec on this machine")

    if total_people == 0:
        print(f"{BAD}YOLO ran but found nobody. Try:  python app.py --conf 0.15")
        return 1

    print(f"{OK}YOLO is working. Annotated frame saved to yolo_check.jpg")
    print(f"\n  Next:  python app.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
