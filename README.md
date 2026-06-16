---
title: Sports Multi-Object Tracker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Sports Multi-Object Detection & Persistent ID Tracking

Detect every person (or ball, car, etc.) in a video and give each one a stable ID that
follows them across frames. Built on **YOLOv8 + ByteTrack + Supervision + OpenCV**, with
both a web UI and a command line.

> The block at the very top of this file is configuration for Hugging Face Spaces. It is
> ignored by GitHub readers — just keep it there if you deploy to Spaces.

---

## Two ways to run it

**Web app (Streamlit) — easiest:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL it prints, upload a video or paste a public link, tune the sidebar
settings, and click **Run tracking**. Results (annotated video, heatmap, count chart,
summary) appear right in the page.

**Command line:**
```bash
python main.py --video match.mp4
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

The YOLOv8 weights (~6 MB) download automatically on first run.

---

## What it produces

| Deliverable | File |
|---|---|
| Annotated video (boxes, IDs, trails, HUD) | `output/annotated_output.mp4` |
| Movement heatmap | `output/heatmap.jpg` |
| Active-subjects-over-time chart | `output/count_chart.jpg` |
| Quick stats card | `output/summary_card.jpg` |
| Per-frame counts (for Excel) | `output/count_log.csv` |
| Full run stats | `output/summary.json` |
| Sample screenshots | `output/screenshots/` |

---

## Project structure

```
vscode_project/
├── app.py                  ← Streamlit web UI
├── main.py                 ← command-line interface
├── requirements.txt
├── packages.txt            ← system packages for Hugging Face Spaces
├── .streamlit/
│   └── config.toml         ← Streamlit server settings
├── src/
│   ├── tracker.py          ← YOLOv8 + ByteTrack core pipeline
│   ├── downloader.py       ← yt-dlp video downloader
│   └── visualizer.py       ← heatmap, count chart, summary card
├── reports/
│   ├── project_overview.md ← plain-English explainer (start here)
│   ├── technical_report.md ← deeper write-up of one benchmark run
│   └── demo_script.md      ← spoken walkthrough script
└── output/                 ← created automatically on first run
```

New here? Read `reports/project_overview.md` first — it explains the whole workflow and
why each tool was chosen.

---

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt
```

Tested with Python 3.10–3.13. Works on CPU; a GPU only speeds up inference.

---

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | — | YouTube / public video URL (downloaded automatically) |
| `--video FILE` | — | Path to a local video file |
| `--model MODEL` | `yolov8n.pt` | Model size: n (fast) / s / m / l (accurate) |
| `--conf FLOAT` | `0.35` | Detection confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--frame-skip N` | `1` | Process every Nth frame (use 2 on slow machines) |
| `--max-det N` | `50` | Max detections per frame (raise for crowds) |
| `--imgsz N` | `640` | Inference resolution. Raise to 1280/1920 for small/distant subjects in HD/4K video |
| `--classes IDS` | `0` | COCO class IDs (0=person, 2=car, 32=sports ball) |
| `--no-traj` | off | Disable trajectory trail lines |
| `--output-dir DIR` | `output/` | Folder for all output files |

```bash
# Faster on CPU
python main.py --video match.mp4 --frame-skip 2

# More accurate
python main.py --video match.mp4 --model yolov8m.pt --conf 0.40

# Track the ball too (class 32)
python main.py --video match.mp4 --classes 0 32

# Crowd / 4K — recover small distant people
python main.py --video crowd.mp4 --imgsz 1280 --conf 0.30 --max-det 150
```

---

## Deploying it live (Hugging Face Spaces)

Spaces is the recommended host — it's free and built for exactly this kind of
Streamlit + ML app.

Deploy from any laptop with git installed. Run these from **inside this folder**
(the one that contains `app.py`):

1. Create a new Space at https://huggingface.co/new-space, pick the **Streamlit** SDK,
   and copy the Space's git URL.
2. Make sure every file is committed (the UI, config and system-package files are new):
   ```bash
   git add -A
   git commit -m "Streamlit app, production config"
   ```
3. Point this repo at your Space and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main          # use "git push space master" if your branch is master
   ```
   When prompted, the password is a Hugging Face **access token** (Settings → Access
   Tokens → New token, role *write*), not your account password.
4. The Space reads `requirements.txt` and `packages.txt`, installs everything, and runs
   `app.py` automatically. First boot takes a few minutes while it installs PyTorch — that
   is normal, not a failure.

No git? You can also drag-and-drop every file into the Space's **Files** tab in the browser.

### Will it just work?

Yes, the common deploy blockers are already handled:
- `opencv-python-headless` plus `libgl1` in `packages.txt` prevents the usual `libGL.so.1`
  import crash that breaks most CV deployments.
- `numpy` is pinned below 2.0 to avoid version-conflict build failures.
- `app.py` catches download/processing errors and shows a message instead of crashing.

The one thing outside our control: YouTube often blocks downloads from cloud IPs. On the
live Space, upload a file or use a direct `.mp4` link rather than a YouTube URL.

That's it — your live URL is `https://huggingface.co/spaces/<your-username>/<space-name>`.

### A note on Vercel

This app **cannot run on Vercel**, and that's expected — it's the wrong type of host, not a
bug. Vercel is serverless: functions are capped at ~250 MB (PyTorch alone is far larger),
time out after 10–60 seconds, and have no always-on server, while Streamlit needs a
persistent process. If you specifically want something on Vercel, deploy a small static
landing page there that links out to the Hugging Face Space running the actual app. The
heavy lifting stays on Spaces (or Render / Railway, which also work).

---

## Model & tracker

| Component | Choice | Why |
|-----------|--------|-----|
| Detector | YOLOv8n | Fast single-stage detector, pre-trained on COCO, CPU-friendly |
| Tracker | ByteTrack | Uses low-confidence detections to recover briefly-occluded tracks |
| Detections / drawing | Supervision | Clean API for parsing model output and drawing |

---

## Limitations

- No appearance re-identification, so a person hidden for a while can return with a new ID
  (you'll see more total IDs than real people on a dense crowd).
- IDs can swap when two similar-looking subjects cross very closely.
- CPU processing of HD/4K video is slow — use `--frame-skip 2` or a smaller model.

Full failure analysis and improvement ideas are in `reports/technical_report.md`.

---

## Dependencies

```
ultralytics            — YOLOv8 detection + ByteTrack tracking
lap                    — linear assignment solver used by ByteTrack
supervision            — detection parsing + annotation utilities
opencv-python-headless — video read/write/drawing (no GUI deps; server-safe)
numpy                  — array operations
yt-dlp                 — download video from a URL
streamlit              — the web UI
imageio-ffmpeg         — bundled ffmpeg for browser-playable H.264 output
```
