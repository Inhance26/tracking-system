"""Person detectors.

Four backends, all returning boxes in NORMALISED frame coordinates:

  yolo   - Ultralytics YOLO + ByteTrack. What you want in production.
           Supplies its own stable track IDs.
  runpod - Same YOLO model, but run on a RunPod Serverless GPU endpoint
           instead of the local machine. Each frame is sent over HTTP; no
           local torch/ultralytics install needed. See runpod_serverless/.
  hog    - OpenCV's built-in HOG pedestrian detector. No model download, no
           torch. Weak on overhead CCTV angles, but useful to prove the
           plumbing works before you install anything heavy.
  demo   - Replays pre-recorded boxes from a JSON file (see make_demo_video.py).
           Lets you exercise the dashboard with zero models.

Each detector's detect(frame) returns (boxes, track_ids) where boxes is a list
of ((x1, y1, x2, y2), confidence) and track_ids is either None (let the local
tracker assign IDs) or a list of ints aligned with boxes.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[Tuple[float, float, float, float], float]


class YoloDetector:
    """Ultralytics YOLO restricted to the COCO 'person' class (id 0)."""

    name = "yolo"

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 640, device: str | None = None,
                 tracker_cfg: str = "bytetrack.yaml") -> None:
        try:
            from ultralytics import YOLO  # noqa: WPS433 (import here so other backends work without it)
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "The 'yolo' detector needs Ultralytics.\n"
                "  pip install ultralytics\n"
                "Or run with --detector hog to try the app without it.\n"
                "Run 'python check_setup.py' to test your install."
            ) from exc
        try:
            self.model = YOLO(weights)
        except Exception as exc:  # noqa: BLE001 - weights download / load failure
            raise SystemExit(
                f"Could not load YOLO weights '{weights}': {exc}\n"
                "The first run downloads yolov8n.pt (~6 MB), so this usually means\n"
                "no internet connection. Connect and retry, or place the .pt file\n"
                "next to app.py."
            ) from exc
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker_cfg = tracker_cfg

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        results = self.model.track(
            frame,
            persist=True,            # keep track state between calls
            classes=[0],             # person only
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker_cfg,
            verbose=False,
        )
        boxes: List[Box] = []
        ids: List[int] = []
        if not results:
            return boxes, None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return boxes, []

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        raw_ids = r.boxes.id
        raw_ids = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            boxes.append(((x1 / w, y1 / h, x2 / w, y2 / h), float(confs[i])))
            if raw_ids is not None:
                ids.append(int(raw_ids[i]))
        return boxes, (ids if raw_ids is not None else None)


class YoloxDetector:
    """YOLOX (a YOLO-family CNN) run through OpenCV's DNN module.

    Same job as the YOLO backend, but with no PyTorch and no Ultralytics - just
    the opencv-python you already have plus a ~20 MB .onnx file. Installs in
    seconds instead of downloading 3 GB.

    It has no built-in tracker, so IDs come from tracker.py the same way the
    HOG backend's do.
    """

    name = "yolox"
    STRIDES = (8, 16, 32)

    def __init__(self, model_path: str, conf: float = 0.25, imgsz: int = 640,
                 nms: float = 0.45) -> None:
        if not os.path.exists(model_path):
            raise SystemExit(
                f"YOLOX model not found: {model_path}\n"
                "Fetch one with:  python get_models.py"
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf = conf
        self.nms = nms
        # The network is fully convolutional but the grid maths below needs a
        # square input that divides by 32.
        self.size = max(320, int(round(imgsz / 32)) * 32)
        self._grids = self._build_grids(self.size)

    def _build_grids(self, size: int):
        grids, strides = [], []
        for s in self.STRIDES:
            g = size // s
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), s))
        return np.concatenate(grids, 0), np.concatenate(strides, 0)

    def _preprocess(self, frame: np.ndarray):
        """Letterbox into a square canvas, padded with grey, top-left aligned."""
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # YOLOX takes raw BGR 0-255 in CHW order - no mean/std normalisation.
        return canvas.transpose(2, 0, 1)[None].astype(np.float32), r

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob, r = self._preprocess(frame)
        self.net.setInput(blob)
        pred = self.net.forward()[0]          # (anchors, 85)

        # Raw output is grid-relative: decode centres and exponential sizes.
        grids, strides = self._grids
        cxcy = (pred[:, :2] + grids) * strides
        wh = np.exp(pred[:, 2:4]) * strides

        obj = pred[:, 4]
        cls = pred[:, 5:]
        cid = cls.argmax(1)
        score = obj * cls[np.arange(len(cls)), cid]

        keep = (cid == 0) & (score > self.conf)   # COCO class 0 is 'person'
        if not keep.any():
            return [], None

        cxcy, wh, score = cxcy[keep], wh[keep], score[keep]
        # xywh in original-frame pixels (undo the letterbox scale)
        xywh = np.stack([cxcy[:, 0] - wh[:, 0] / 2,
                         cxcy[:, 1] - wh[:, 1] / 2,
                         wh[:, 0], wh[:, 1]], 1) / r

        idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), float(self.conf), self.nms)
        if len(idx) == 0:
            return [], None
        idx = np.array(idx).flatten()

        boxes: List[Box] = []
        for i in idx:
            x, y, bw, bh = xywh[i]
            boxes.append((
                (max(0.0, x / w), max(0.0, y / h),
                 min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)),
                float(score[i]),
            ))
        return boxes, None


class RunPodDetector:
    """Sends each frame to a RunPod Serverless GPU endpoint for YOLO detection.

    Needs no local torch/ultralytics install - only `requests`. The endpoint
    runs runpod_serverless/handler.py; deploy it first (see that folder's
    README), then point this at it with the endpoint ID and API key.

    Has no local model, so it has no tracker either - IDs come from
    tracker.py, same as the hog/yolox backends.
    """

    name = "runpod"

    def __init__(self, endpoint_id: str, api_key: str, conf: float = 0.25,
                 imgsz: int = 960, timeout: float = 10.0,
                 jpeg_quality: int = 85) -> None:
        if not endpoint_id:
            raise SystemExit(
                "The 'runpod' detector needs an endpoint ID.\n"
                "  --runpod-endpoint <id>   or set RUNPOD_ENDPOINT_ID\n"
                "Deploy runpod_serverless/ first - see its README."
            )
        if not api_key:
            raise SystemExit(
                "The 'runpod' detector needs a RunPod API key.\n"
                "  --runpod-api-key <key>   or set RUNPOD_API_KEY"
            )
        try:
            import requests  # noqa: WPS433 - only this backend needs it
        except ImportError as exc:
            raise SystemExit(
                "The 'runpod' detector needs the requests package.\n"
                "  pip install requests"
            ) from exc
        self._requests = requests
        self.url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.conf = conf
        self.imgsz = imgsz
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality
        self._last_error_log = 0.0

    def detect(self, frame: np.ndarray):
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return [], None
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        payload = {"input": {"image": image_b64, "conf": self.conf, "imgsz": self.imgsz}}
        try:
            resp = self._requests.post(
                self.url, headers=self.headers, json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - network hiccups shouldn't kill the pipeline
            now = time.time()
            if now - self._last_error_log > 5.0:  # don't spam the console every frame
                print(f"  [runpod] request failed: {exc}")
                self._last_error_log = now
            return [], None

        output = body.get("output") or {}
        if "error" in output:
            now = time.time()
            if now - self._last_error_log > 5.0:
                print(f"  [runpod] endpoint error: {output['error']}")
                self._last_error_log = now
            return [], None

        boxes: List[Box] = []
        for x1, y1, x2, y2, conf in output.get("boxes", []):
            boxes.append(((x1, y1, x2, y2), float(conf)))
        return boxes, None


class HogDetector:
    """OpenCV HOG + SVM pedestrian detector. Ships with opencv-python."""

    name = "hog"

    def __init__(self, conf: float = 0.4, width: int = 640) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.conf = conf
        self.width = width

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = self.width / float(w) if w > self.width else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
        rects, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        boxes: List[Box] = []
        sh, sw = small.shape[:2]
        for (x, y, bw, bh), score in zip(rects, weights.ravel() if len(rects) else []):
            if float(score) < self.conf:
                continue
            boxes.append((
                (x / sw, y / sh, (x + bw) / sw, (y + bh) / sh),
                float(score),
            ))
        return boxes, None


class DemoDetector:
    """Replays boxes recorded alongside a synthetic video.

    The JSON is {"frames": [[[x1,y1,x2,y2], ...], ...]} in normalised coords,
    one entry per video frame.
    """

    name = "demo"

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise SystemExit(
                f"Demo detections not found at {path}.\n"
                "Run: python make_demo_video.py"
            )
        with open(path, "r", encoding="utf-8") as fh:
            self.frames: Sequence[Sequence[Sequence[float]]] = json.load(fh)["frames"]
        self.i = 0

    def detect(self, frame: np.ndarray):
        if not self.frames:
            return [], None
        boxes_raw = self.frames[self.i % len(self.frames)]
        self.i += 1
        return [((b[0], b[1], b[2], b[3]), 0.9) for b in boxes_raw], None


def build_detector(cfg) -> object:
    kind = (cfg.detector or "yolo").lower()
    if kind == "yolo":
        return YoloDetector(weights=cfg.weights, conf=cfg.conf, imgsz=cfg.imgsz,
                            device=cfg.device)
    if kind == "yolox":
        return YoloxDetector(model_path=cfg.yolox_model, conf=cfg.conf,
                             imgsz=cfg.imgsz)
    if kind == "runpod":
        return RunPodDetector(endpoint_id=cfg.runpod_endpoint, api_key=cfg.runpod_api_key,
                              conf=cfg.conf, imgsz=cfg.imgsz, timeout=cfg.runpod_timeout)
    if kind == "hog":
        return HogDetector(conf=cfg.conf)
    if kind == "demo":
        return DemoDetector(cfg.demo_detections)
    raise SystemExit(
        f"Unknown detector '{cfg.detector}'. Use yolo, runpod, yolox, hog or demo."
    )
