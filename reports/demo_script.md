# Demo Video Script (3–5 minutes)

A shot-by-shot script for the mandatory recorded walkthrough. Screen-record yourself
(e.g. OBS, or Windows `Win+G`) talking over these steps. Target ~4 minutes.

---

### 0:00 – 0:30 — Intro & problem (talking head or slide)
- "This is a computer-vision pipeline that detects every person in a video and gives each
  one a unique ID that stays stable over time — multi-object detection + tracking."
- State the video used and that it's a **public** asset:
  `https://media.roboflow.com/supervision/video-examples/market-square.mp4`
- Why this clip: an overhead market square with 50–80 people at once — a hard
  multi-object / occlusion / similar-appearance test, exactly what the task targets.

### 0:30 – 1:15 — Architecture (show README "Model & Tracker" section)
- **Detector:** YOLOv8n (COCO `person` class), single-stage, fast, auto-downloads.
- **Tracker:** ByteTrack — associates *every* detection (high + low confidence) in two
  stages, with a Kalman filter predicting motion and a ~30-frame buffer to survive occlusion.
- One line on why this pair: best speed-vs-occlusion trade-off without a heavy ReID network.

### 1:15 – 2:00 — Walk the code (open files in VS Code)
- `main.py` — CLI entry: resolves the video, runs the tracker, then builds the extra
  visualizations. Point out `--imgsz`, `--conf`, `--max-det`.
- `src/tracker.py` — the core loop: `model.track(persist=True, tracker="bytetrack.yaml", ...)`
  is where detection + ID assignment happens; `annotate_frame()` draws boxes, IDs, trails, HUD.
- `src/visualizer.py` — heatmap, count chart, summary card.

### 2:00 – 2:45 — Run it live (terminal)
```bash
python main.py --video market-square.mp4 --imgsz 1280 --conf 0.30 --max-det 150
```
- Let the progress log scroll; explain it processes every frame and writes the annotated MP4.
- (If too slow to show fully, cut to the finished output — that's fine.)

### 2:45 – 3:45 — Show the results (this is the key part)
- Play `output/annotated_output.mp4`: point out colour-coded boxes, the **ID label on each
  person**, and the **fading trajectory trail** showing each ID staying with its person.
- Open `output/heatmap.jpg` — movement density / walking paths around the fountain.
- Open `output/count_chart.jpg` — people-count over time.
- Open `output/summary.json` — peak 79, avg 67, 224 unique IDs.

### 3:45 – 4:30 — Honesty: limitations & improvements
- Be upfront: 224 IDs for ~67 average people means **ID inflation** — people fully hidden
  behind others come back with a new ID, because there's no appearance ReID.
- Mention the fix: BoT-SORT / StrongSORT + OSNet embeddings would cut crossing swaps and
  re-entry inflation; a larger model or frame tiling recovers the most distant people.
- Close: "Correct, modular, reproducible pipeline; clear about its trade-offs."

---

**Recording tips**
- 1080p screen capture, system audio + mic.
- Have the `output/` folder already populated so you can show results even if the live run
  is slow on your machine.
- Keep it under 5 minutes — the rubric asks for 3–5.
</content>
