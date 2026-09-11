# Zone People Tracker — CCTV build

Counts how many people are on the floor and which zone each one is standing in,
from a CCTV feed. Live numbers and the annotated video render in a web browser.

This copy is **pre-configured for the bundled `cctv_footage.mp4`** (the workshop
clip, 848×478, 11.4 fps, 3 minutes) with five zones already marked in
`zones.json`. No arguments needed:

```
python app.py
```

---

## Fastest path: two double-clicks

1. **`setup.bat`** — creates the virtual environment, installs everything
   including YOLO, then runs the self-test. Ultralytics pulls in PyTorch, so
   this downloads 2–3 GB and can take ten minutes or more. Leave it running.
2. **`run.bat`** — starts the tracker and opens the dashboard.

That's it. The rest of this file is for doing the same thing inside VS Code, and
for changing what it runs on.

---

## Running it in VS Code

1. **File → Open Folder** → pick this folder (the one containing `app.py`).
2. Install the **Python** extension if VS Code offers it (`.vscode/extensions.json`
   recommends it automatically).
3. Open the terminal with `` Ctrl+` `` and set up once:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   If PowerShell refuses the activate line, run
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again.

4. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside
   `.venv`.
5. Press **F5**.

F5 uses the launch configurations in `.vscode/launch.json`. Pick one from the
dropdown in the Run and Debug panel:

| Configuration | What it does |
|---|---|
| **Check setup (YOLO self-test)** | run this first — proves YOLO works before anything else |
| **Run tracker (CCTV footage + YOLO)** | the normal one — real detection |
| **Run tracker (YOLOX - no PyTorch needed)** | real CNN detection with no ultralytics install |
| **Run tracker (HOG - weak, plumbing test only)** | last-resort fallback; finds very little |
| **Run tracker (faster - skip frames)** | `--frame-skip 3 --width 640` for slow machines |
| **Run tracker (pick a video...)** | prompts for a file path, RTSP URL, or `0` for a webcam |

Each tracker configuration opens your browser on the dashboard automatically.
Breakpoints work — try one in `pipeline.py` inside the `for t in tracks:` loop
to step through the zone assignment.

Prefer the terminal? `python app.py` does the same thing.

---

## Detector options

| Backend | Install | Speed here | Quality |
|---|---|---|---|
| `--detector yolo` *(default)* | `pip install ultralytics` — 2–3 GB with PyTorch | best on a local GPU | best; Ultralytics also supplies the person IDs via ByteTrack |
| `--detector runpod` | nothing extra locally — needs `requests` only | limited by network round-trip, not your CPU | same YOLO model, run on a RunPod cloud GPU per frame |
| `--detector yolox` | nothing extra — a 36 MB `.onnx` run through OpenCV | ~2–7 fps on CPU | very good; a real YOLO-family CNN |

The `yolox` backend exists because installing PyTorch is a big commitment for a
first look. The model isn't bundled (36 MB), so fetch it once — it takes
seconds, and needs no pip install at all:

```powershell
python get_models.py                       # downloads models/yolox_s.onnx
python app.py --detector yolox --imgsz 640 --conf 0.15 --frame-skip 2
```

That command is what produced the counts in the screenshots — five people found
and tracked across zones on this clip, where the old HOG fallback found one.
Use `--detector yolo` once ultralytics is installed; it's better still.

### Running detection on a RunPod cloud GPU

If your machine has no GPU (or a weak one), offload detection to RunPod
instead of installing PyTorch locally. `app.py` still runs here — video
capture, tracking, dwell timing, the dashboard — only the YOLO call for each
frame goes over HTTP to a RunPod Serverless GPU endpoint.

```powershell
$env:RUNPOD_ENDPOINT_ID = "your-endpoint-id"
$env:RUNPOD_API_KEY = "your-api-key"
python app.py --detector runpod --frame-skip 2 --open
```

Deploying the endpoint (build the Docker image, push it, create the
Serverless endpoint on runpod.io) is a one-time setup covered in
[`runpod_serverless/README.md`](runpod_serverless/README.md). A ready-made
**"Run tracker (RunPod cloud GPU)"** launch configuration is already in
`.vscode/launch.json` for `F5` once you've set the two environment variables.
Because each frame is a network round trip, use `--frame-skip 2` or higher
unless you keep an always-on worker.

---

## Is YOLO actually working?

```powershell
python check_setup.py
```

It checks every dependency, loads `yolov8s.pt`, runs detection on 12 frames of
the bundled footage, and tells you how many people it found, how big they were
in pixels, and how fast your machine is. It also writes **`yolo_check.jpg`** —
the frame with the most detections, boxes drawn — so you can see for yourself
rather than trusting a number.

A healthy result on this clip looks roughly like:

```
  [ok] ultralytics 8.3.x
  [ok] model loaded
       people found: 21 across 12 frames  (avg 1.8/frame, best frame 3)
       person heights: smallest 31px, median 88px, tallest 140px
       speed: 6.4 frames/sec on this machine
  [ok] YOLO is working.
```

If it says ultralytics is missing, `pip install ultralytics`. If it runs but
finds nobody, try `python app.py --conf 0.15`.

---

## Calibrating the headcount

"People in view" is the end of a chain, and each link can be the one that is
wrong:

```
model -> --conf -> --dedupe-ios -> --min-hits -> --count-coast -> the number
```

`calibrate.py` measures that number over a clip so you can compare settings
instead of eyeballing the video:

```powershell
# what your current settings actually report
python calibrate.py --frames 400

# does a bigger model find the people at the back?
python calibrate.py --frames 60 --stride 12 --sweep-weights yolov8n.pt,yolov8x.pt --imgsz 1280

# you counted 4 people in that stretch - how often does it agree?
python calibrate.py --frames 400 --expect 4
```

It reports the mean count, its range, how often the number changes between
frames (the jitter you see on the dashboard), and — with `--expect` — how often
it is exactly right.

**Under-counting** (the usual problem) means the detector never saw someone.
Only the model and its input size fix that; no amount of tracker tuning
invents a missed person. On this clip, at 848×478 with workers at the back
about 12×40 px:

| setting | mean count |
|---|---|
| `yolov8n --imgsz 960` *(the old defaults)* | 2.6 |
| `yolov8n --imgsz 1280` | 3.7 |
| `yolov8s --imgsz 1280` *(the defaults now)* | 4.1 |
| `yolov8x --imgsz 1280` *(the GPU worker)* | 4.3 |

Raising `--imgsz` costs nothing but GPU time and is the first thing to try.
Lowering `--conf` also surfaces faint figures, at the price of false positives.

**Over-counting** has two causes, both handled by default now:

- *Two boxes on one person.* Ultralytics' NMS compares IoU, so a small box
  nested inside a larger one scores ~0.65 and survives the 0.7 cutoff — one
  person, counted twice. `--dedupe-ios 0.6` measures overlap against the
  smaller box instead, where a nested duplicate scores 1.0. Set `0` to disable.
- *Flickering false positives.* `--min-hits 3` requires a person to be detected
  on three frames before they join the count, so one frame of a track id on a
  stack of pipes doesn't register.

**Jitter** — the number bouncing 4-3-4 while nobody moved — is the detector
dropping someone for a frame. `--count-coast 5` keeps a lost person counted for
five more frames. On this clip that cut the frame-to-frame changes from 8.1% to
2.0%.

If you run detection on a GPU (`--detector runpod`), the worker now builds with
`yolov8x` — on a GPU the extra latency is close to free. Build a smaller worker
with:

```powershell
docker build --build-arg YOLO_WEIGHTS=yolov8s.pt .
```

Note the local default (`yolov8s`) and the worker default (`yolov8x`) differ, so
the same clip will not give identical counts through `--detector yolo` and
`--detector runpod`. The `x`-over-`s` margin above came from a 60-frame sample
and is small enough to be worth re-measuring on your own footage.

---

## Time spent in each zone

Every person's stay is timed. A **visit** starts when someone's feet enter a
zone and ends when they leave it — or when the tracker loses them for more than
two seconds. That grace period matters: detectors drop people for a frame or
two behind a pillar constantly, and without it one stay would be chopped into a
dozen fragments.

The dashboard shows three things:

- **On the video** — each person's box is labelled with how long they've been in
  their current zone. Anyone past `--long-dwell` (default 120s) turns amber.
- **Per zone** — how long the longest current stay is, the average completed
  visit, and how many visits there have been.
- **Time in zone right now** — a live list of everyone being tracked, which zone
  they're in, and for how long.

Hover a zone row for its total occupancy (person-seconds — two people for a
minute is two minutes) and its longest-ever visit.

```powershell
python app.py --long-dwell 300                    # flag anyone over 5 minutes
python app.py --long-dwell 0                      # never flag anyone
python app.py --dwell-csv zone_visits.csv         # log every completed visit
```

The CSV is one row per completed visit and is appended to, so it survives
restarts — this is what you'd analyse later in Excel:

```
entered_at,left_at,person_id,zone_id,seconds
2026-08-29 07:09:14,2026-08-29 07:09:20,5,workshop_table,5.9
2026-08-29 07:09:17,2026-08-29 07:09:32,7,far_bay,15.5
```

`/api/stats` carries the same numbers per zone — `visits`, `avg_visit_s`,
`max_visit_s`, `occupancy_s`, `longest_now_s` — plus `zone_s` for each tracked
person.

**Read the times as occupancy, not identity.** A person who leaves and comes
back is a new visit with a new ID, and a long occlusion splits one stay into
two. The totals are reliable; individual journeys are not.

---

## The zones

`zones.json` already contains five zones traced onto the workshop clip:

| Zone | Covers |
|---|---|
| Far bay | the open concrete floor across the background |
| Assembly area | the casting and blue frame in the middle band |
| Workshop table | the large machined slab in the foreground |
| Machine station | the blue machine and its operator platform, right |
| Left storage bay | the stacked castings, left |

To change them, open **http://127.0.0.1:8000/editor** while the app is running.
Click the corners of a zone's **floor area**, press **Enter** to close it, click
the name to rename it, then **Save zones**. Counting switches over immediately —
no restart.

Coordinates are stored as fractions of the frame (0–1), so the same zones keep
working if you change `--width` or move to a higher-resolution stream from the
same camera. Change the camera *angle*, though, and you must redraw.

**A person is counted into the zone their feet are in** — the bottom-centre of
their box, not the middle of their body.

---

## Switching to your own footage or a live camera

```powershell
python app.py --source "C:\path\to\your_clip.mp4"
python app.py --source "rtsp://user:pass@192.168.1.50:554/Streaming/Channels/102"
python app.py --source 0
```

Always quote an RTSP URL in PowerShell — `&` and `?` mean something else to the
shell otherwise. Prefer the camera's sub-stream (`102`, `subtype=1`, `stream2`);
it's lower resolution, which is all the detector needs and a fraction of the CPU.

New camera angle means new zones. Keep a file per camera:

```powershell
python app.py --source shop_floor.mp4 --zones zones-shop.json
```

The editor saves to whichever file you passed.

---

## Defaults in this build

Tuned for the bundled clip; all still overridable on the command line.

| Flag | Here | Stock | Why |
|---|---|---|---|
| `--source` | `cctv_footage.mp4` | *required* | so `python app.py` just runs |
| `--width` | `848` | `960` | the clip's native width — no rescaling |
| `--imgsz` | `1280` | `640` | far more pixels than the frame has; the biggest single lever on whether the small figures at the back are found at all |
| `--conf` | `0.25` | `0.35` | same reason — don't discard faint distant people |
| `--host` | `127.0.0.1` | `0.0.0.0` | serves only to this machine; no firewall prompt |

Counting flags: `--min-hits`, `--count-coast`, `--dedupe-ios` — see
[Calibrating the headcount](#calibrating-the-headcount).

Other flags: `--long-dwell`, `--dwell-csv`,
`--detector` (`yolo`/`runpod`/`yolox`/`hog`/`demo`), `--yolox-model`, `--weights`
(`yolov8n.pt` is ~2× faster and less accurate; `yolov8x.pt` is better still on a
GPU), `--device` (`cpu`, `0`, `mps`), `--frame-skip`,
`--port`, `--zones`, `--no-loop`, `--open`, `--verbose`,
`--runpod-endpoint`, `--runpod-api-key`, `--runpod-timeout` (`--detector runpod` only;
see [`runpod_serverless/README.md`](runpod_serverless/README.md)).

---

## What to expect on this clip

The camera angle is good — roughly 35° down from height, which detects far
better than a straight overhead view. Foreground workers are about 90 px tall
and should be found reliably. Three things will limit accuracy:

- **Machinery hides feet.** Zone assignment uses the bottom of the person's box.
  A worker standing behind the big casting has their box end at the casting
  edge, so they can be placed in the wrong zone. This is the main source of
  error on this footage.
- **The background is small.** People at the far end are 20–40 px tall. Some
  will be missed even at `--imgsz 1280`, and a bigger model finds more of them
  than a bigger input size does — see
  [Calibrating the headcount](#calibrating-the-headcount).
- **Few people.** With one to three on screen, a single miss is a large
  percentage error.

The first YOLO run downloads `yolov8s.pt` (~22 MB) automatically.

---

## Files

```
setup.bat            one-time: virtual environment + dependencies + self-test
run.bat              start the tracker and open the dashboard
check_setup.py       proves YOLO works on this machine and this footage
get_models.py        downloads YOLOX .onnx models for the no-PyTorch backend
models/              those .onnx files
app.py               entry point and command-line flags
pipeline.py          capture -> detect -> track -> assign zone -> annotate loop
detector.py          the three detector backends (YOLO lives here)
tracker.py           IoU + centroid tracker (used when the detector has no IDs)
zones.py             polygon storage and point-in-polygon assignment
dwell.py             per-zone visit timing, averages and the CSV log
server.py            stdlib HTTP server: MJPEG, stats, zone save/load
web/dashboard.html   the live dashboard
web/editor.html      the zone editor
make_demo_video.py   generates synthetic test footage, if you ever want it
cctv_footage.mp4     the workshop clip
zones.json           the five zones traced onto it
.vscode/             launch configurations and interpreter settings
handler.py           the --detector runpod GPU worker (root, so RunPod finds it)
Dockerfile           builds that worker image
runpod_serverless/   deploy notes and the worker's own requirements.txt
```

Detection lives in `detector.py`:

- `YoloDetector` — about 50 lines wrapping `ultralytics`. Calls
  `model.track(..., classes=[0], tracker="bytetrack.yaml")`, so detection and
  person IDs both come from Ultralytics.
- `YoloxDetector` — runs a YOLOX `.onnx` through `cv2.dnn`. Letterboxes the
  frame to a square, decodes the raw grid output (centres are grid-relative,
  sizes are exponential), filters to COCO class 0, then NMS. No PyTorch.
- `RunPodDetector` — JPEG-encodes each frame, POSTs it to a RunPod Serverless
  endpoint (`handler.py`, running the same Ultralytics model on a cloud GPU),
  and parses the boxes back. No local model at all.
- `HogDetector` / `DemoDetector` — no model at all.

`tracker.py` supplies IDs for every backend except `yolo`, which brings its own.

Only two files are not Python: the two HTML pages, which are plain HTML, CSS and
vanilla JavaScript with no build step. The web server is Python's standard
library — no Flask, no FastAPI.

---

## HTTP endpoints

| Route | Returns |
|---|---|
| `/` | dashboard |
| `/editor` | zone editor |
| `/video` | MJPEG stream of the annotated frames |
| `/snapshot.jpg` | single un-annotated frame (what the editor draws on) |
| `/api/stats` | JSON — counts per zone, per-person dwell |
| `/api/zones` | `GET` current zones, `POST` to replace them |

`/api/stats` is the integration point for pushing these numbers elsewhere — a
screen in the office, a database, Grafana.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `Video file not found` | `cctv_footage.mp4` isn't beside `app.py`, or you passed a bad `--source` |
| `The 'yolo' detector needs Ultralytics` | `pip install ultralytics`, or run the HOG launch configuration |
| VS Code can't find `cv2` | wrong interpreter — `Ctrl+Shift+P` → Python: Select Interpreter → the `.venv` one |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Page won't load | check the terminal still says "Tracker running"; if port 8000 is taken, add `--port 8080` |
| Very low frame rate | `--frame-skip 3 --width 640` |
| Counts flicker | raise `--conf` if it's finding things that aren't people; lower it if real people vanish |

---

## Privacy and security

- `--host 127.0.0.1` (the default here) serves only to this machine. Passing
  `0.0.0.0` exposes an unauthenticated dashboard to your whole network. Never
  port-forward it to the internet as-is.
- The app records no video and stores no images. Only counts and numeric track
  IDs, which reset on restart. Nothing is written to disk except `zones.json`.
- This footage is of staff at work. Monitoring employees carries heavier legal
  and ethical obligations than counting customers in most places, India
  included — notice, purpose limits, and stricter rules for anything that scores
  individuals. Counting people per zone is a lighter-touch use than measuring
  individual output. Worth checking your local rules before deploying; I'm not a
  lawyer and this isn't legal advice.
