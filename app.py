#!/usr/bin/env python3
"""Zone people-tracker - counts people per floor zone from a CCTV feed.

Configured out of the box for the bundled my_clip.mp4, so this is enough:

  python app.py

Other sources:

  python app.py --source cctv_footage.mp4          # the older workshop clip
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


# Defaults are tuned for the bundled 1270x720 clip, my_clip.mp4:
#   width 0     - keep the native size, so nothing is up- or down-scaled. The
#                 old 848 default was the NATIVE width of the previous clip
#                 (cctv_footage.mp4, 848x478); on this one it would downscale
#                 for nothing and shrink the already-small distant people.
#   imgsz 1280  - give YOLO at least as many pixels as the frame has, so small,
#                 distant people survive the resize into the model.
#   conf  0.25  - lower than stock, for the same reason
#   yolov8s     - yolov8n is the smallest model in the family and misses the
#                 distant people entirely
#
# The imgsz/weights numbers above were measured on the OLDER cctv_footage.mp4
# (960 -> 1280 lifted the headcount 2.6 -> 3.7 against a true 4-5; yolov8s
# counted 4.1 vs yolov8n's 3.7). They have NOT been re-measured on my_clip.mp4,
# which is a different scene at a different resolution - rerun calibrate.py
# against it before trusting the count.
DEFAULT_SOURCE = os.path.join(HERE, "my_clip.mp4")


@dataclass
class Config:
    source: str
    detector: str = "yolo"
    weights: str = "yolov8s.pt"
    device: str | None = None
    conf: float = 0.25
    imgsz: int = 1280
    width: int = 0
    frame_skip: int = 1
    max_age: int = 30
    min_hits: int = 3          # detections before a track joins the headcount
    count_coast: int = 5       # frames a lost track stays counted
    dedupe_ios: float = 0.6    # overlap above which two boxes are one person
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
    db: str | None = None              # optional SQLite store, read by export_db.py
    camera_id: str = "cam1"            # stamped on every stored visit
    # -- pose (phase 1) -----------------------------------------------------
    # Off by default: --pose swaps the detection model for a pose model, which
    # changes the headcount this app reports. Nothing about the existing
    # behaviour moves unless you ask for it.
    pose: bool = False
    pose_weights: str = "yolo11s-pose.pt"
    pose_min_height: float = 80.0      # px of bbox height to trust a skeleton
    pose_min_kp_conf: float = 0.5      # per-keypoint confidence to count as seen
    pose_min_core: int = 8             # of 12 torso/limb joints, to pass the gate
    host: str = "127.0.0.1"
    port: int = 8000
    verbose: bool = False
    open_browser: bool = False


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Zone people-tracker for CCTV footage")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="video file path, rtsp:// URL, or webcam index (e.g. 0). "
                        "Defaults to the bundled my_clip.mp4")
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
    p.add_argument("--weights", default="yolov8s.pt",
                   help="YOLO weights. yolov8s finds noticeably more of the small, "
                        "distant people than yolov8n at ~2x the cost; drop to "
                        "yolov8n.pt on a slow CPU, raise to yolov8x.pt on a GPU")
    p.add_argument("--device", default=None, help="cpu, 0, mps ... (YOLO only)")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="detector input size; the big lever on whether small, "
                        "distant people are found at all (was 960)")
    p.add_argument("--width", type=int, default=0,
                   help="processing width in px (smaller = faster; 0 keeps the "
                        "native size, which is the default)")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="run detection every Nth frame (2-3 helps a lot on CPU)")
    p.add_argument("--min-hits", type=int, default=3,
                   help="frames a person must be detected on before they are "
                        "counted (raise to reject flickering false positives)")
    p.add_argument("--max-age", type=int, default=30,
                   help="frames a lost track is kept before it is discarded")
    p.add_argument("--count-coast", type=int, default=5,
                   help="frames a briefly-lost person stays in the headcount, so "
                        "the number doesn't jitter when a detection drops (0 = off)")
    p.add_argument("--dedupe-ios", type=float, default=0.6,
                   help="overlap (as a fraction of the smaller box) above which two "
                        "boxes are treated as the same person (0 disables)")
    p.add_argument("--long-dwell", type=float, default=120,
                   help="seconds in one zone before a person is highlighted "
                        "amber on the video (0 disables)")
    p.add_argument("--dwell-csv", default=None, metavar="FILE",
                   help="append every completed zone visit to a CSV "
                        "(e.g. --dwell-csv zone_visits.csv)")
    p.add_argument("--db", default=None, metavar="FILE",
                   help="record every completed zone visit to a SQLite database "
                        "(e.g. --db footfall.db). This is the file export_db.py "
                        "reads; without it, nothing writes one and the export "
                        "has nothing to export")
    p.add_argument("--camera-id", default="cam1",
                   help="stamped on every row written by --db, so a second "
                        "camera's visits stay distinguishable from this one's")
    p.add_argument("--pose", action="store_true",
                   help="extract 2D skeletons as well as boxes. Replaces the "
                        "detection model with a pose model (one pass, so "
                        "keypoints arrive already attached to their track ID), "
                        "which means --weights no longer applies and the "
                        "headcount is not the one calibrate.py measured for "
                        "yolov8s - re-run it before trusting counts")
    p.add_argument("--pose-weights", default="yolo11s-pose.pt",
                   help="pose model to use with --pose. yolo11s-pose is the "
                        "sensible default; yolo11m/x-pose find more of the "
                        "small, distant people and are close to free on a GPU")
    p.add_argument("--pose-min-height", type=float, default=80.0,
                   help="bbox height in px below which a skeleton is marked "
                        "unusable (drawn grey, excluded from behaviour data). "
                        "Measure yours with pose_audit.py rather than guessing")
    p.add_argument("--pose-min-kp-conf", type=float, default=0.5,
                   help="per-keypoint confidence below which a joint is treated "
                        "as not found")
    p.add_argument("--pose-min-core", type=int, default=8,
                   help="how many of the 12 torso/limb joints must be found for "
                        "a pose to pass the gate (the 5 face joints never count)")
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
        min_hits=a.min_hits, max_age=a.max_age, count_coast=a.count_coast,
        dedupe_ios=a.dedupe_ios,
        jpeg_quality=a.jpeg_quality, loop=not a.no_loop, zones_path=a.zones,
        host=a.host, port=a.port, verbose=a.verbose, open_browser=a.open,
        yolox_model=a.yolox_model or _default_yolox_model(),
        runpod_endpoint=a.runpod_endpoint, runpod_api_key=a.runpod_api_key,
        runpod_timeout=a.runpod_timeout,
        long_dwell=a.long_dwell, dwell_csv=a.dwell_csv,
        db=a.db, camera_id=a.camera_id,
        pose=a.pose, pose_weights=a.pose_weights,
        pose_min_height=a.pose_min_height, pose_min_kp_conf=a.pose_min_kp_conf,
        pose_min_core=a.pose_min_core,
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
    if cfg.pose:
        print(f"  Pose        {cfg.pose_weights}  gate: >={cfg.pose_min_height:.0f}px, "
              f">={cfg.pose_min_core}/12 joints")
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
