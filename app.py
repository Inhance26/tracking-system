#!/usr/bin/env python3
"""Zone people-tracker - counts people per floor zone from a CCTV feed.

Configured out of the box for the bundled cctv_footage.mp4, so this is enough:

  python app.py

Other sources:

  python app.py --source other_clip.mp4
  python app.py --source "rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102"
  python app.py --source 0                       # USB webcam
  python app.py --detector hog                   # no YOLO install needed

Then open http://127.0.0.1:8000 (dashboard) and http://127.0.0.1:8000/editor
(draw your zones).
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pipeline import Pipeline, SharedState  # noqa: E402
from server import serve  # noqa: E402


# Defaults are tuned for the bundled 848x478 workshop clip:
#   width 848  - its native size, so nothing is up- or down-scaled
#   imgsz 960  - give YOLO more pixels than the frame has; helps on the small,
#                distant figures at the back of the shop floor
#   conf  0.25 - lower than stock, for the same reason
DEFAULT_SOURCE = os.path.join(HERE, "cctv_footage.mp4")


@dataclass
class Config:
    source: str
    detector: str = "yolo"
    weights: str = "yolov8n.pt"
    device: str | None = None
    conf: float = 0.25
    imgsz: int = 960
    width: int = 848
    frame_skip: int = 1
    max_age: int = 30
    min_hits: int = 3
    jpeg_quality: int = 80
    loop: bool = True
    zones_path: str = os.path.join(HERE, "zones.json")
    demo_detections: str = os.path.join(HERE, "demo_detections.json")
    yolox_model: str = os.path.join(HERE, "models", "yolox_s.onnx")
    runpod_endpoint: str | None = None   # RunPod Serverless endpoint ID
    runpod_api_key: str | None = None    # RunPod API key
    runpod_timeout: float = 10.0         # seconds per-frame HTTP call may take
    long_dwell: float = 120.0          # seconds before a stay is flagged amber
    dwell_csv: str | None = None       # optional log of completed zone visits
    host: str = "127.0.0.1"
    port: int = 8000
    verbose: bool = False
    open_browser: bool = False


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Zone people-tracker for CCTV footage")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="video file path, rtsp:// URL, or webcam index (e.g. 0). "
                        "Defaults to the bundled cctv_footage.mp4")
    p.add_argument("--detector", default="yolo",
                   choices=["yolo", "runpod", "yolox", "hog", "demo"],
                   help="yolo = Ultralytics, local (best, needs torch); "
                        "runpod = same model on a RunPod Serverless GPU endpoint "
                        "(no local torch); yolox = ONNX via OpenCV (no torch); "
                        "hog / demo = no model")
    p.add_argument("--yolox-model", default=None,
                   help="path to a YOLOX .onnx (default: models/yolox_s.onnx, "
                        "falling back to models/yolox_tiny.onnx)")
    p.add_argument("--runpod-endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                   help="RunPod Serverless endpoint ID (--detector runpod). "
                        "Defaults to $RUNPOD_ENDPOINT_ID")
    p.add_argument("--runpod-api-key", default=os.environ.get("RUNPOD_API_KEY"),
                   help="RunPod API key (--detector runpod). "
                        "Defaults to $RUNPOD_API_KEY")
    p.add_argument("--runpod-timeout", type=float, default=10.0,
                   help="seconds to wait for each RunPod detection call")
    p.add_argument("--weights", default="yolov8n.pt",
                   help="YOLO weights (yolov8n.pt is fine on CPU; yolov8s.pt is more accurate)")
    p.add_argument("--device", default=None, help="cpu, 0, mps ... (YOLO only)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    p.add_argument("--width", type=int, default=848,
                   help="processing width in px (smaller = faster; 0 keeps the native size)")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="run detection every Nth frame (2-3 helps a lot on CPU)")
    p.add_argument("--long-dwell", type=float, default=120,
                   help="seconds in one zone before a person is highlighted "
                        "amber on the video (0 disables)")
    p.add_argument("--dwell-csv", default=None, metavar="FILE",
                   help="append every completed zone visit to a CSV "
                        "(e.g. --dwell-csv zone_visits.csv)")
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--no-loop", action="store_true", help="stop at the end of a video file")
    p.add_argument("--zones", default=os.path.join(HERE, "zones.json"))
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; 0.0.0.0 exposes the dashboard to your whole network")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="open the dashboard in your browser")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    return Config(
        source=a.source, detector=a.detector, weights=a.weights, device=a.device,
        conf=a.conf, imgsz=a.imgsz, width=a.width, frame_skip=a.frame_skip,
        jpeg_quality=a.jpeg_quality, loop=not a.no_loop, zones_path=a.zones,
        host=a.host, port=a.port, verbose=a.verbose, open_browser=a.open,
        yolox_model=a.yolox_model or _default_yolox_model(),
        runpod_endpoint=a.runpod_endpoint, runpod_api_key=a.runpod_api_key,
        runpod_timeout=a.runpod_timeout,
        long_dwell=a.long_dwell, dwell_csv=a.dwell_csv,
    )


def _default_yolox_model() -> str:
    """Prefer the more accurate model, fall back to the small one we ship."""
    for name in ("yolox_s.onnx", "yolox_m.onnx", "yolox_tiny.onnx"):
        path = os.path.join(HERE, "models", name)
        if os.path.exists(path):
            return path
    return os.path.join(HERE, "models", "yolox_s.onnx")


def main(argv=None) -> int:
    cfg = parse_args(argv)

    # A missing video file is by far the most common first-run mistake, and the
    # error OpenCV gives for it is unhelpful. Catch it here instead.
    looks_like_a_file = not str(cfg.source).isdigit() and "://" not in str(cfg.source)
    if looks_like_a_file and not os.path.exists(cfg.source):
        print(f"\n  Video file not found: {cfg.source}")
        print("  Check the file is in this folder, or pass --source <path>.\n")
        return 1

    state = SharedState()
    pipeline = Pipeline(cfg, state)
    pipeline.start()

    httpd = serve(state, pipeline, cfg)
    shown_host = "localhost" if cfg.host in ("0.0.0.0", "") else cfg.host
    url = f"http://{shown_host}:{cfg.port}"
    print(f"\n  Tracker running")
    print(f"  Dashboard   {url}")
    print(f"  Zone editor {url}/editor")
    print(f"  Source      {os.path.basename(str(cfg.source))}")
    print(f"  Detector    {cfg.detector}      Zones: {os.path.basename(cfg.zones_path)}")
    print(f"  Stop with Ctrl+C\n")
    if cfg.open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        pipeline.stop()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
