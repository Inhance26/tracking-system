"""RunPod Serverless handler - runs YOLO person detection on the GPU worker.

Deployed as a RunPod Serverless endpoint. Each invocation receives one video
frame (base64 JPEG) and returns person boxes in NORMALISED coordinates, the
same contract detector.py's local backends use. No tracking/IDs here - a
serverless endpoint may run on a different worker (or a fresh one) on every
call, so track continuity is handled locally by tracker.py instead.

Input:  {"input": {"image": "<base64 jpeg>", "conf": 0.25, "imgsz": 960}}
Output: {"boxes": [[x1, y1, x2, y2, conf], ...]}   # coords in 0..1
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import runpod
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
model = YOLO(MODEL_PATH)


def _decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode 'image' as JPEG/PNG bytes")
    return frame


def handler(event):
    job_input = event.get("input") or {}
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "missing 'input.image' (base64-encoded JPEG)"}

    conf = float(job_input.get("conf", 0.25))
    imgsz = int(job_input.get("imgsz", 960))

    frame = _decode_image(image_b64)
    h, w = frame.shape[:2]

    results = model.predict(
        frame, classes=[0], conf=conf, imgsz=imgsz, device=0, verbose=False,
    )

    boxes = []
    if results:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                boxes.append([x1 / w, y1 / h, x2 / w, y2 / h, float(c)])

    return {"boxes": boxes}


runpod.serverless.start({"handler": handler})
