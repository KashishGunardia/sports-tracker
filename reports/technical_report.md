# Technical Report
## Multi-Object Detection and Persistent ID Tracking in Public Event Footage

**Candidate:** _(your name)_
**Source video:** Roboflow Supervision public sample — "Market Square" pedestrian crowd
`https://media.roboflow.com/supervision/video-examples/market-square.mp4`
**Category:** Public event / pedestrian crowd (multiple moving people)

---

### 1. Overview

This report describes an end-to-end computer-vision pipeline that detects and persistently
tracks every person in a publicly available crowd video. Each subject is assigned a unique
numeric ID that is held as stable as possible across the clip, despite frequent occlusion,
overlapping pedestrians, varying scale (people near vs. far from the camera) and very
similar appearance (many people in dark coats).

The "Market Square" clip was chosen deliberately: it is an overhead view of a busy public
square with **50–80 people visible at once**, which stresses the *multi-object* and
*ID-persistence* aspects of the task far more than a sparse, few-player sports clip would.

**Run results (this submission):**

| Metric | Value |
|---|---|
| Resolution / FPS | 2160 × 3840 (vertical 4K) @ 60 fps |
| Duration / frames | 7.9 s / 474 frames (every frame processed) |
| Detector | YOLOv8n (COCO), inference size 1280 px |
| Tracker | ByteTrack |
| **Peak people tracked in a single frame** | **79** |
| **Average people per frame** | **66.6** |
| **Total unique IDs assigned over the clip** | **224** |
| Processing speed (CPU) | ~1.6 fps |

> The 224 total IDs vs. ~67 average concurrent subjects reflects ID fragmentation
> (re-identification of people after occlusion / re-entry) — discussed in §6.

---

### 2. Detection Model — YOLOv8n

**Model:** `yolov8n.pt` — YOLOv8 Nano, pre-trained on COCO (80 classes), filtered to class 0 (`person`).

YOLOv8 (Ultralytics) is an anchor-free, single-stage detector that produces bounding boxes
and class scores in one forward pass. It was selected because:

- It has **native ByteTrack integration** via `model.track()` — no extra glue code.
- COCO class `person` is exactly the target subject.
- The Nano variant is small (~6 MB) and runs on CPU; it auto-downloads on first run.
- The inference resolution is configurable (`--imgsz`), which is critical here.

**Why inference resolution mattered.** At the default 640 px, the 4K frame is downscaled so
aggressively that distant pedestrians shrink below the detector's effective receptive range —
only ~15–20 of ~40+ people were found. Raising inference size to **1280 px** roughly tripled
coverage (peak 79 detections) at the cost of ~2–3× slower inference. This is the single most
impactful tuning decision in the pipeline.

| Parameter | Value | Reason |
|---|---|---|
| Confidence | 0.30 | Crowd has many small/partly-occluded people; lower threshold recovers them |
| IoU (NMS) | 0.45 | Suppresses duplicate boxes on overlapping pedestrians |
| Max detections | 150 | The square holds well over 50 people at once |
| Inference size | 1280 | Recovers small/distant subjects in the 4K frame |
| Classes | [0] person | Human subjects only |

---

### 3. Tracking Algorithm — ByteTrack

**Tracker:** ByteTrack (Zhang et al., ECCV 2022), via Ultralytics' `bytetrack.yaml`.

Classical trackers (SORT, DeepSORT) discard low-confidence detections. ByteTrack instead
associates **every** detection box in a two-stage match:

```
Detections in frame t
   ├─ High confidence ── Stage 1: Hungarian matching against
   │                     Kalman-predicted positions of active tracks
   └─ Low confidence  ── Stage 2: matched only against tracks left
                         UNMATCHED after stage 1  (rescues occluded tracks)
```

A **Kalman filter** predicts each track's next position from its velocity, so an ID survives
short detection gaps (occlusion, motion blur). Unmatched tracks are kept alive in a buffer
for ~30 frames before deletion.

**Why ByteTrack over alternatives:**

| Tracker | Speed | Occlusion handling | Appearance (ReID) |
|---|---|---|---|
| SORT | Very fast | Poor | ✗ |
| **ByteTrack** | **Fast** | **Good** | **✗** |
| DeepSORT | Medium | Better | ✓ (slow) |
| BoT-SORT / StrongSORT | Slow | Best | ✓ (very slow) |

ByteTrack gives the best speed/occlusion trade-off for a dense crowd without needing a
separate, slow appearance-embedding (ReID) network.

---

### 4. How ID Consistency Is Maintained

1. **`persist=True`** in `model.track()` — tracker state (Kalman filters + track buffer) is
   carried between consecutive frames, so IDs are not reset each frame.
2. **Two-stage association** — high-confidence boxes are matched first; low-confidence boxes
   then re-link tracks that occlusion would otherwise drop.
3. **Track buffer (~30 frames)** — momentarily undetected people keep their ID instead of
   immediately getting a new one.
4. **Kalman motion prediction** — velocity-based prediction keeps the expected position
   accurate during brief misses.
5. **Per-ID colour + fading trajectory trail** (last 60 centre points) — makes ID stability
   *visually verifiable* in the annotated video and is the basis of the movement heatmap.

---

### 5. Challenges Faced

| Challenge | Effect | How it was handled |
|---|---|---|
| 4K + small distant people | Default 640 px missed most of the crowd | Raised inference size to 1280 px |
| Heavy occlusion (people cross constantly) | Tracks briefly disappear | ByteTrack low-conf stage + Kalman + track buffer |
| Near-identical appearance (dark coats) | IoU matcher cannot use colour | Accepted; ReID would be the proper fix (§7) |
| Scale variation (near vs. far) | Far people low-confidence | Lower conf threshold (0.30) |
| CPU-only, 4K, 60 fps | Slow (~1.6 fps) | Short 8 s clip; `--frame-skip` and `--imgsz` are tunable |
| Large 4K output file (180 MB) | Exceeds GitHub's 100 MB limit | Transcoded to a 7.4 MB 1080p H.264 file |

---

### 6. Failure Cases Observed

1. **ID inflation from re-entry / occlusion.** 224 unique IDs were assigned although only
   ~67 people are present on average. When a pedestrian is fully hidden behind another for
   more than the track-buffer window, they return with a **new** ID. Without appearance ReID,
   the tracker cannot know it is the same person.
2. **ID switches at close crossings.** When two similarly-dressed people cross directly,
   their boxes overlap heavily and IDs occasionally swap at the crossing point.
3. **Missed very-distant subjects.** A few people at the far top of the frame are still too
   small even at 1280 px and are intermittently undetected.
4. **Static-object false positives.** Seated café patrons and a couple of street fixtures are
   occasionally detected as `person`, adding spurious short-lived IDs.

---

### 7. Possible Improvements

**Near-term**
- **BoT-SORT / StrongSORT + OSNet ReID** — appearance embeddings would sharply cut the
  re-entry ID inflation and crossing swaps that dominate the failure cases here.
- **Larger detector** (`yolov8s/m`) or **tiling** the 4K frame — would recover the remaining
  distant pedestrians.
- **A region-of-interest mask** to exclude café seating, removing static false positives.

**Medium-term**
- **MOTA / IDF1 evaluation** against a few hand-labelled frames to quantify ID stability.
- **Homography / bird's-eye projection** of the square to analyse density and flow in metric
  coordinates, plus per-ID speed estimation.

**Long-term**
- **Transformer trackers** (MOTR, TrackFormer) for end-to-end detection+association.

---

### 8. Reproducing This Run

```bash
python main.py --video market-square.mp4 --model yolov8n.pt \
    --conf 0.30 --max-det 150 --imgsz 1280
```

The raw 4K annotated output was transcoded to the shipped 1080p H.264 file with the
ffmpeg command documented in the README.

---

### 9. References

- Ultralytics YOLOv8 — https://docs.ultralytics.com
- ByteTrack — Zhang, Y. et al. (2022), *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- Supervision — https://supervision.roboflow.com
- COCO dataset — https://cocodataset.org

---
*Prepared for the Multi-Object Detection and Persistent ID Tracking assignment.*
</content>
</invoke>
