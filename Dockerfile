# Serverless worker image, built by RunPod's GitHub integration.
#
# Both this file and handler.py sit at the repository root because RunPod's
# pre-deploy check scans root-level files for runpod.serverless.start() and
# reports the handler as missing when it lives in a subdirectory.
#
# Note the requirements file below is the SERVERLESS one, not the root
# requirements.txt - the root file is for running app.py locally and has
# no runpod SDK and a non-headless opencv, neither of which works here.

FROM runpod/base:0.6.2-cuda12.1.0

COPY runpod_serverless/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Bake the weights in so a cold-started worker doesn't hit the network first.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv yolov8n.pt /yolov8n.pt

COPY handler.py /handler.py
WORKDIR /
ENV MODEL_PATH=/yolov8n.pt

CMD ["python", "-u", "handler.py"]
