# Serverless worker image, built by RunPod's GitHub integration.
#
# Both this file and handler.py sit at the repository root because RunPod's
# pre-deploy check scans root-level files for runpod.serverless.start() and
# reports the handler as missing when it lives in a subdirectory.
#
# Note the requirements file below is the SERVERLESS one, not the root
# requirements.txt - the root file is for running app.py (video capture,
# tracking, dashboard) and pulls in openpyxl and requests, but not the runpod
# SDK this worker starts from. Both files use headless opencv.

FROM runpod/base:0.6.2-cuda12.1.0

COPY runpod_serverless/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Which YOLO weights the worker runs. This is a GPU worker, so the extra
# latency of a large model is close to free and buys back the small, distant
# people that yolov8n - the smallest model in the family - misses entirely.
# Measured on cctv_footage.mp4 at imgsz 1280 (see calibrate.py), mean headcount
# against a true 4-5: yolov8n 3.7, yolov8s 4.1, yolov8x 4.3.
# Build a smaller worker with:
#   docker build --build-arg YOLO_WEIGHTS=yolov8s.pt .
ARG YOLO_WEIGHTS=yolov8x.pt

# Bake the weights in so a cold-started worker doesn't hit the network first.
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_WEIGHTS}')" \
    && mv ${YOLO_WEIGHTS} /weights.pt

COPY handler.py /handler.py
WORKDIR /
ENV MODEL_PATH=/weights.pt

CMD ["python", "-u", "handler.py"]
