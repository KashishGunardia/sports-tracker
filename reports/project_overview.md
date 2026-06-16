# Project Overview — Sports Multi-Object Tracker

A written walkthrough of what this project does, how the pieces fit together, and why
each piece of the tech stack was chosen. If you only read one document to understand the
project, read this one.

---

## 1. What the project does, in one paragraph

You give it a video — a sports clip, a crowd, any footage with moving people. It finds
every person in every frame, draws a box around each one, and gives each person a number
(an ID) that stays with them as they move around. The output is an annotated video where
you can literally watch ID `7` follow the same player across the whole clip, plus a few
analytics: a heatmap of where movement was concentrated, a chart of how many people were
on screen over time, and a JSON/CSV summary of the run. There are two ways to run it: a
command line (`main.py`) and a web app (`app.py`, built with Streamlit).

This is a classic **detection + tracking** problem. Detection answers "what is in this
frame and where," and tracking answers "is this the same thing I saw a moment ago." Doing
both well is what makes the IDs stable instead of flickering.

---

## 2. The workflow, end to end

```
  Video source                  Core pipeline                    Outputs
 ┌─────────────┐   ┌───────────────────────────────────┐   ┌──────────────────┐
 │ Upload file │   │ 1. Read frame                     │   │ annotated_output │
 │     or      │──▶│ 2. YOLOv8 detects people (boxes)  │──▶│ heatmap.jpg      │
 │  paste URL  │   │ 3. ByteTrack assigns/keeps IDs    │   │ count_chart.jpg  │
 │ (yt-dlp)    │   │ 4. Draw boxes + IDs + trails + HUD │   │ summary.json/csv │
 └─────────────┘   │ 5. Write frame, repeat            │   │ screenshots/     │
                   └───────────────────────────────────┘   └──────────────────┘
```

Step by step:

1. **Get the video.** Either the user uploads a local file, or pastes a public URL
   (YouTube, Vimeo, a direct `.mp4`). For URLs, `yt-dlp` downloads the clip first.
2. **Detect.** For each frame, YOLOv8 returns bounding boxes for the classes we asked for
   (person by default; ball, car, etc. are options).
3. **Track.** ByteTrack takes those per-frame boxes and links them across time, so the
   same person keeps the same ID even through brief occlusion. The key flag is
   `persist=True`, which carries tracker state from one frame to the next.
4. **Annotate.** We draw the box, the ID label with confidence, a fading trail of where
   that ID has been, and a small heads-up display (frame number, time, active count,
   total unique count).
5. **Summarise.** After the loop we build the heatmap, the count-over-time chart, a stats
   card, and write the machine-readable `summary.json` and `count_log.csv`.

The CLI and the Streamlit app run the *exact same* core (`src/tracker.py` and
`src/visualizer.py`). The UI is just a friendlier front door — nothing about the
detection or tracking logic is duplicated.

---

## 3. The tech stack, and why each piece is there

| Layer | Tool | Why this one |
|---|---|---|
| Detection | **YOLOv8 (Ultralytics)** | Single-stage, fast, accurate, runs on CPU. Pre-trained on COCO so "person" works out of the box. It ships ByteTrack built in, so detection and tracking share one API call. The Nano weights are ~6 MB and download themselves on first run. |
| Tracking | **ByteTrack** | Most trackers throw away low-confidence detections. ByteTrack keeps them and uses them in a second matching pass to rescue tracks that occlusion would otherwise drop. Best balance of speed and ID stability without a heavy re-identification network. |
| Drawing / detections | **Supervision (Roboflow)** | Clean, well-tested helpers for turning raw model output into a tidy `Detections` object (boxes, IDs, confidences). Saves writing fiddly array-handling code. |
| Video & image I/O | **OpenCV (headless)** | Reads and writes video frames, and does all the drawing (boxes, text, the chart, the heatmap blend). We use the **headless** build specifically so it runs on servers with no display — the regular build needs GUI libraries and crashes in a container. |
| Numbers | **NumPy** | The heatmap and chart are just arrays of pixels; NumPy makes that fast and simple. |
| Downloading | **yt-dlp** | The most reliable way to pull a public video from a URL without an API key. |
| Web UI | **Streamlit** | Turns a Python script into a web app with almost no front-end code. Perfect for an internal/demo tool — file upload, sliders, and result display in a few lines. |
| Browser video | **imageio-ffmpeg** | OpenCV writes a codec browsers won't play inline. This bundles a static ffmpeg so the app can re-encode the result to H.264 for the in-page player. |

The thread running through these choices is **"do the most with the least."** Every tool
is either standard, CPU-friendly, or removes a category of glue code. Nothing here needs
a GPU, a paid API, or a database.

---

## 4. How the code is organised

```
vscode_project/
├── app.py              ← Streamlit web UI (run: streamlit run app.py)
├── main.py             ← Command-line interface  (run: python main.py ...)
├── requirements.txt
├── packages.txt        ← system packages for Hugging Face Spaces
├── .streamlit/
│   └── config.toml     ← Streamlit server settings
├── src/
│   ├── tracker.py      ← the core: YOLOv8 + ByteTrack loop and frame annotation
│   ├── downloader.py   ← yt-dlp wrapper for URL inputs
│   └── visualizer.py   ← heatmap, count chart, summary card
├── reports/
│   ├── project_overview.md   ← this file
│   ├── technical_report.md   ← deeper write-up of one benchmark run
│   └── demo_script.md        ← spoken walkthrough script
└── output/             ← created on first run; all results land here
```

`tracker.py` is the heart of it. `app.py` and `main.py` are two thin wrappers that gather
inputs, call `SportsTracker.process(...)`, then call the `visualizer` helpers. If you want
to understand the project, read `tracker.py`'s `process()` method — everything else serves it.

---

## 5. Running it

**Web app (easiest):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL it prints, upload a video or paste a link, adjust the sliders, and
click *Run tracking*.

**Command line:**
```bash
python main.py --video match.mp4 --imgsz 1280 --conf 0.30 --max-det 150
```

The first run downloads the YOLOv8 weights (~6 MB) automatically.

---

## 6. Honest limitations

This pipeline does not use appearance-based re-identification, so a person who is fully
hidden behind someone else for a while can come back with a *new* ID. On a dense crowd you
will see more total IDs than there were actual people. Two similarly-dressed people who
cross closely can also occasionally swap IDs. These are known trade-offs of choosing a fast,
ReID-free tracker; the fix (BoT-SORT / StrongSORT with appearance embeddings) is slower and
was deliberately out of scope. The deeper benchmark and a full failure analysis are in
`reports/technical_report.md`.
