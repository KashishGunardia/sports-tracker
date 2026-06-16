# Spoken Walkthrough Script (4–6 minutes)

A narrated script for a recorded walkthrough of the project. Screen-record yourself
(OBS, or `Win+G` on Windows) and read these lines over the matching screen. Times are a
guide, not a rule. Square brackets are stage directions, not lines to read.

---

### 0:00 – 0:35 — What this is

> "This is a computer-vision project that takes any video of people moving around — a
> sports clip, a crowd, anything — and does two things at once. First it *detects* every
> person in every frame. Second it *tracks* them, meaning it gives each person an ID
> number that stays with them as they move. So you can follow player number seven across
> the entire clip. That second part, keeping the ID stable, is the hard and interesting
> bit, and it's what most of this project is about."

[Show the finished annotated video playing for a few seconds so the boxes and IDs are
visible while you say this.]

---

### 0:35 – 1:20 — The tech stack and why

> "The stack is deliberately lightweight — everything runs on a normal CPU, no GPU, no
> paid APIs. Detection is YOLOv8 from Ultralytics: it's fast, it's pre-trained on the COCO
> dataset so 'person' works straight away, and crucially it ships with the tracker built
> in. Tracking is ByteTrack. The clever thing about ByteTrack is that, unlike older
> trackers, it doesn't throw away low-confidence detections — it uses them in a second
> matching pass to rescue people who got briefly hidden behind someone else. OpenCV does
> all the video reading, writing, and drawing, and I'm using its headless build so it runs
> on a server. Supervision tidies up the model's raw output, and yt-dlp handles videos
> from a URL."

[Show the README "tech stack" table, or `reports/project_overview.md` section 3.]

---

### 1:20 – 2:10 — Walk the code

> "The whole project funnels into one file: `src/tracker.py`. Inside the `process` method
> there's a loop over every frame. The key line is `model.track` with `persist=True` — that
> `persist` flag is what carries the tracker's memory from one frame to the next, so IDs
> aren't reset each frame. Right after that, `annotate_frame` draws the box, the ID label,
> a fading trail showing where each person has been, and a small heads-up display in the
> corner. `src/visualizer.py` builds the extras — the heatmap, the count chart, the stats
> card. And `main.py` and `app.py` are just two front doors into that same core."

[Open `src/tracker.py`, scroll to `process()`, point at the `model.track(...)` call and
then `annotate_frame(...)`. Then briefly show `src/visualizer.py`.]

---

### 2:10 – 3:10 — Run it in the web app

> "There are two ways to run it. The friendly one is the Streamlit web app — I just run
> `streamlit run app.py`. I upload a video here, or paste a URL. On the left I can pick the
> model, choose what to track, and tune confidence and inference size. For a crowded or 4K
> clip I'd raise the inference size and the max detections. I click Run, it detects and
> tracks frame by frame, and then it shows me the annotated video right in the browser,
> plus the heatmap and the count chart."

[Run `streamlit run app.py`. Upload a short clip, set the sliders, click Run. If the live
run is slow, cut to a pre-recorded result — that's fine, just say so.]

---

### 3:10 – 3:50 — Run it from the command line

> "If I'd rather script it, the command line does exactly the same thing. `python main.py`,
> point it at a video, and I can pass the same options as flags — inference size, confidence,
> max detections. It writes everything into the `output` folder: the annotated video, the
> heatmap, the chart, the screenshots, and a JSON and CSV summary I could load into Excel."

[Show the terminal command and the `output/` folder with the result files. Open
`output/summary.json`.]

---

### 3:50 – 4:40 — The results, honestly

> "Here's what the output actually shows. Each person has a colour-coded box, an ID, and a
> trail proving the ID sticks to them. The heatmap shows where movement concentrated. The
> chart shows how many people were on screen over time. But I want to be honest about the
> limitation. This tracker doesn't use appearance — it doesn't 'recognise faces' or clothing.
> So if someone is completely hidden behind another person for too long, they come back with
> a *new* ID. On a dense crowd that means more total IDs than there were real people. That's
> a known trade-off of choosing a fast tracker over a heavy one."

[Show the heatmap and count chart. If you have a crowd example, point out the inflated ID
count and explain why.]

---

### 4:40 – 5:20 — What I'd improve, and deployment

> "The clean fix for that ID inflation is a tracker with appearance re-identification, like
> BoT-SORT or StrongSORT with OSNet embeddings — slower, but it would stop people getting a
> new ID when they reappear. A bigger detector or tiling the frame would catch the most
> distant people. And for deployment: this runs as a live web app on Hugging Face Spaces,
> which is built for exactly this kind of Streamlit-plus-ML app. It's worth noting it can't
> run on Vercel — Vercel is serverless and can't host a heavy, always-on ML app — so Spaces
> is the right home for it."

> "That's the project: a correct, modular, reproducible detection-and-tracking pipeline,
> with both a UI and a CLI, and clear about its trade-offs. Thanks for watching."

---

## Recording tips

- Record at 1080p with both system audio and your mic.
- Have the `output/` folder already populated, so you can show results even if the live run
  is slow on your machine.
- Keep a short clip handy (a few seconds) so a live run finishes on camera.
- Aim for under six minutes. Tighter is better than complete.
