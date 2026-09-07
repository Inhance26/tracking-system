# RunPod Serverless detector

Runs the YOLO person detector on a RunPod GPU instead of your machine.
`app.py` keeps running locally (video capture, tracking, dwell timing, the
dashboard) - only per-frame detection is sent over HTTP to a RunPod
Serverless endpoint. Nothing here needs torch/ultralytics installed locally.

```
                    (your machine, e.g. inside VS Code)
   video source --> pipeline.py --> RunPodDetector --HTTP--> RunPod GPU worker
                         |                                    (handler.py, YOLO)
                         v
                  tracker + dashboard
```

Because each frame makes a network round trip, this suits a few frames per
second, not raw throughput - use `--frame-skip 2` or `3` and keep frames
small (`--imgsz 640`) unless you provision an always-on worker. It's the
right fit when your dev machine has no/weak GPU, since detection happens on
RunPod's hardware instead.

The worker itself lives at the repository root - `handler.py` and the
`Dockerfile` that builds it. They sit there rather than in this folder
because RunPod's pre-deploy check scans root-level files for
`runpod.serverless.start()` and reports the handler as missing when it is in
a subdirectory. This folder keeps the deploy notes and the worker's own
`requirements.txt` (headless opencv plus the runpod SDK, which the root
`requirements.txt` deliberately does not carry).

## 1. Get the image built

Either let RunPod build it from GitHub - **Serverless** → **New Endpoint** →
choose the GitHub source and pick this repo and branch, no Docker needed
locally - or build and push it yourself, from the repository root:

```powershell
docker build -t <your-dockerhub-username>/retail-tracker-detector:latest .
docker push <your-dockerhub-username>/retail-tracker-detector:latest
```

## 2. Create the Serverless endpoint

1. Sign in at [runpod.io](https://www.runpod.io) → **Serverless** → **New Endpoint**.
2. **Source**: the GitHub repo, or the image you pushed above.
3. **GPU**: any CUDA GPU works for `yolov8n` (e.g. 16 GB tier is plenty).
4. **Active Workers**: `0` is cheapest (pay only per request, but the first
   request after idle time pays a cold-start delay of several seconds while
   the container boots and loads the model). Set to `1` to keep a worker warm
   and avoid that, at the cost of paying for idle GPU time.
5. Create it, then copy the **Endpoint ID** shown on its page.
6. Get an **API key** from **Settings → API Keys** (create one if you don't
   have one already).

## 3. Run the tracker against it

Back in the main project folder, in VS Code's terminal:

```powershell
$env:RUNPOD_ENDPOINT_ID = "your-endpoint-id"
$env:RUNPOD_API_KEY = "your-api-key"
python app.py --detector runpod --open
```

Or pass them as flags instead of env vars: `--runpod-endpoint`, `--runpod-api-key`.

Use the **"Run tracker (RunPod cloud GPU)"** launch configuration in the Run
and Debug panel (`F5`) to do the same with breakpoints - it reads the same
two environment variables, so set them in your shell (or a `.env` file your
shell loads) before pressing F5.

## Notes

- **No track IDs from the endpoint.** Each serverless call may land on a
  different (or restarted) worker, so `handler.py` returns boxes only, and
  the local `tracker.py` (the same one used by the `yolox`/`hog` backends)
  assigns and persists IDs across frames.
- **Costs are per-second of GPU time used** (plus idle time if you keep a
  worker warm). Stop sending traffic and workers scale back to zero on their
  own.
- **Never commit your API key.** Keep it in an environment variable or a
  local `.env` file that's gitignored, not in `launch.json` or source.
- To change the model, edit `MODEL_PATH`/the weights baked into `Dockerfile`
  and rebuild/push the image; `handler.py`'s `conf`/`imgsz` are already
  passed through from `app.py`'s `--conf`/`--imgsz` flags per request.
