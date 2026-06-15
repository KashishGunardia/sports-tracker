# 🏃 Multi-Object Detection & Persistent ID Tracking
### Sports / Event Video — End-to-End CV Pipeline

**Stack:** YOLOv8 · ByteTrack · Supervision · OpenCV

---

## 🎯 What this submission produced

Run on a public pedestrian-crowd clip (overhead view of a busy market square):

- **Source video:** `https://media.roboflow.com/supervision/video-examples/market-square.mp4`
  (Roboflow Supervision public sample asset — directly downloadable, no login)
- **Category:** public event / crowd footage with many moving people
- **Result:** 474 frames (4K @ 60 fps) tracked, **peak 79 / average 67 people per frame**,
  **224 unique IDs** assigned over the 8 s clip.

| Deliverable | File |
|---|---|
| Annotated video (1080p H.264, 7.4 MB) | `output/annotated_output.mp4` |
| Movement heatmap | `output/heatmap.jpg` |
| Count-over-time chart | `output/count_chart.jpg` |
| Sample screenshots | `output/screenshots/` |
| Per-frame counts | `output/count_log.csv` |
| Run stats | `output/summary.json` |
| Technical report | `reports/technical_report.md` |

Reproduce with:
```bash
python main.py --video market-square.mp4 --model yolov8n.pt --conf 0.30 --max-det 150 --imgsz 1280
```

---

## 📁 Project Structure

```
sports_tracker/
├── main.py                 ← RUN THIS FILE
├── requirements.txt        ← install dependencies
├── src/
│   ├── tracker.py          ← YOLOv8 + ByteTrack core pipeline
│   ├── downloader.py       ← yt-dlp video downloader
│   └── visualizer.py       ← heatmap, count chart, summary card
├── output/                 ← created automatically on first run
│   ├── annotated_output.mp4
│   ├── heatmap.jpg
│   ├── count_chart.jpg
│   ├── summary_card.jpg
│   ├── count_log.csv
│   ├── summary.json
│   └── screenshots/
└── reports/
    └── technical_report.md
```

---

## ⚙️ Setup (one time)

### Step 1 — Open project in VS Code
```
File → Open Folder → select the sports_tracker folder
```

### Step 2 — Open the VS Code terminal
```
Terminal → New Terminal   (or Ctrl + ` )
```

### Step 3 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

> The YOLOv8 model weights (~6 MB) are downloaded automatically on first run.

---

## 🚀 How to Run

### Option A — Process a YouTube video (auto-downloads it)
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Option B — Process a local video file
```bash
# Windows
python main.py --video "C:\Users\You\Downloads\match.mp4"

# macOS / Linux
python main.py --video "/home/user/videos/match.mp4"
```

---

## 🎛️ All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | — | YouTube / public video URL |
| `--video FILE` | — | Path to local video file |
| `--model MODEL` | `yolov8n.pt` | Model size: n (fast) / s / m / l (accurate) |
| `--conf FLOAT` | `0.35` | Detection confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--frame-skip N` | `1` | Process every Nth frame (use 2 on slow machines) |
| `--max-det N` | `50` | Max detections per frame (raise for crowds) |
| `--imgsz N` | `640` | Inference resolution. **Raise to 1280/1920 for small/distant subjects in HD/4K video** |
| `--classes IDS` | `0` | COCO class IDs (0=person, 2=car, 32=sports ball) |
| `--no-traj` | off | Disable trajectory trail lines |
| `--output-dir DIR` | `output/` | Folder for all output files |

### Speed tips
```bash
# Faster on CPU — skip every other frame + smallest model
python main.py --video match.mp4 --model yolov8n.pt --frame-skip 2

# More accurate — larger model, every frame
python main.py --video match.mp4 --model yolov8m.pt --conf 0.40

# Track the ball as well (class 32 = sports ball)
python main.py --video match.mp4 --classes 0 32

# Crowd / 4K footage — recover small distant people
python main.py --video crowd.mp4 --imgsz 1280 --conf 0.30 --max-det 150
```

### Compressing the annotated video (optional)
OpenCV writes large `mp4v` files (a 4K output can exceed GitHub's 100 MB limit). To produce
a small, widely-compatible H.264 file, transcode with ffmpeg:
```bash
ffmpeg -i output/annotated_output.mp4 -vf "scale=1080:-2" \
       -c:v libx264 -preset medium -crf 26 -pix_fmt yuv420p output/annotated_1080p.mp4
```
No system ffmpeg? `pip install imageio-ffmpeg` ships a static binary; get its path with
`python -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())"`.

---

## 📤 Outputs

| File | Description |
|------|-------------|
| `output/annotated_output.mp4` | Video with boxes, IDs, trajectory trails, HUD |
| `output/heatmap.jpg` | Movement density map overlaid on first frame |
| `output/count_chart.jpg` | Active subjects vs. time line chart |
| `output/summary_card.jpg` | Quick stats card |
| `output/count_log.csv` | Per-frame subject count (for Excel / analysis) |
| `output/summary.json` | Full run stats in JSON |
| `output/screenshots/` | Auto-captured frames at 10%, 35%, 60%, 85% |

---

## 🧠 Model & Tracker

| Component | Choice | Why |
|-----------|--------|-----|
| Detector | YOLOv8n | Fast single-stage detector, pre-trained on COCO |
| Tracker | ByteTrack | Uses low-confidence detections to recover lost tracks |
| Annotation | Supervision | Clean API for drawing boxes, labels |

---

## ⚠️ Limitations

- ID switches can occur when two players cross very closely
- Players leaving and re-entering the frame may get new IDs
- CPU processing of 1080p video is slow — use `--frame-skip 2` or `--model yolov8n`
- Similar-looking subjects (same jersey) may occasionally swap IDs

---

## 📦 Dependencies

```
ultralytics  — YOLOv8 detection + ByteTrack tracking
lap          — linear assignment solver used by ByteTrack
supervision  — annotation utilities
opencv-python— video read/write/drawing
numpy        — array operations
yt-dlp       — video download from YouTube etc.
```

> Tested with Python 3.13 on Windows (CPU). GPU is optional — it only speeds up inference.
