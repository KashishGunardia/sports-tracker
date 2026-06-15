"""
tracker.py
==========
Core multi-object detection and persistent ID tracking pipeline.
Uses YOLOv8 for detection and ByteTrack for multi-object tracking.
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import time
import csv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Colour palette — one consistent colour per track ID
# ─────────────────────────────────────────────────────────────
PALETTE = [
    (255, 56,  56),  (0,  200, 100), (0,  150, 255), (255, 200,   0),
    (200,  0, 200),  (0,  220, 220), (255, 128,   0), (128, 255,   0),
    (255,  0, 128),  (100, 100, 255), (50, 205,  50), (255, 165,   0),
    (220,  20,  60), (0,  191, 255), (154,  50, 205), (255, 215,   0),
    (64,  224, 208), (255, 105, 180), (0,  128, 128), (210, 105,  30),
]


def get_color(track_id: int) -> tuple:
    """Return a consistent BGR colour for a given track ID."""
    return PALETTE[int(track_id) % len(PALETTE)]


# ─────────────────────────────────────────────────────────────
#  Trajectory store  (keeps last N centre-points per track ID)
# ─────────────────────────────────────────────────────────────
class TrajectoryStore:
    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.points: dict[int, list] = defaultdict(list)

    def update(self, track_id: int, cx: int, cy: int):
        self.points[track_id].append((cx, cy))
        if len(self.points[track_id]) > self.max_len:
            self.points[track_id].pop(0)

    def get(self, track_id: int) -> list:
        return self.points.get(track_id, [])


# ─────────────────────────────────────────────────────────────
#  Per-frame annotator
# ─────────────────────────────────────────────────────────────
def annotate_frame(
    frame: np.ndarray,
    detections,                  # supervision Detections object
    trajectory: TrajectoryStore,
    frame_idx: int,
    fps: float,
    total_subjects: set,
    draw_trajectory: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, persistent IDs, confidence scores,
    trajectory trails and a HUD overlay onto the frame.
    """
    annotated = frame.copy()

    # ── trajectory trails (drawn first so boxes render on top) ──
    if draw_trajectory and detections.tracker_id is not None:
        for tid in detections.tracker_id:
            pts = trajectory.get(int(tid))
            if len(pts) >= 2:
                color = get_color(int(tid))
                for i in range(1, len(pts)):
                    alpha = i / len(pts)          # fade older points
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(annotated, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)

    # ── bounding boxes + ID labels ───────────────────────────────
    if detections.tracker_id is not None:
        for box, tid, conf in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.confidence,
        ):
            x1, y1, x2, y2 = map(int, box)
            tid   = int(tid)
            conf  = float(conf)
            color = get_color(tid)

            # bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # label with coloured background
            label = f"ID:{tid}  {conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            lx, ly = x1, max(y1 - 6, th + 4)
            cv2.rectangle(
                annotated,
                (lx, ly - th - 4), (lx + tw + 6, ly + baseline),
                color, -1,
            )
            cv2.putText(
                annotated, label, (lx + 3, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            # centre dot
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 3, color, -1)

    # ── HUD (top-left) ──────────────────────────────────────────
    active = len(detections) if detections.tracker_id is not None else 0
    ts     = frame_idx / max(fps, 1)
    hud    = [
        f"Frame : {frame_idx}",
        f"Time  : {ts:.1f}s",
        f"Active: {active}",
        f"Total : {len(total_subjects)}",
    ]
    for j, line in enumerate(hud):
        y_pos = 26 + j * 22
        # dark shadow for readability on any background
        cv2.putText(annotated, line, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, line, (12, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── watermark ───────────────────────────────────────────────
    h, w = annotated.shape[:2]
    cv2.putText(
        annotated, "YOLOv8 + ByteTrack",
        (w - 210, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return annotated


# ─────────────────────────────────────────────────────────────
#  Main pipeline class
# ─────────────────────────────────────────────────────────────
class SportsTracker:
    """
    End-to-end multi-object detection + persistent ID tracking pipeline.

    Parameters
    ----------
    model_path  : YOLOv8 weights file (auto-downloaded on first run)
    conf        : detection confidence threshold  (0.0–1.0)
    iou         : NMS IoU threshold               (0.0–1.0)
    frame_skip  : process every Nth frame (1 = every frame)
    max_det     : maximum detections per frame
    draw_traj   : whether to draw trajectory trails
    """

    def __init__(
        self,
        model_path : str   = "yolov8n.pt",
        conf       : float = 0.35,
        iou        : float = 0.45,
        frame_skip : int   = 1,
        max_det    : int   = 50,
        draw_traj  : bool  = True,
        imgsz      : int   = 640,
    ):
        logger.info(f"Loading model: {model_path}  (auto-downloads if missing)")
        self.model      = YOLO(model_path)
        self.conf       = conf
        self.iou        = iou
        self.frame_skip = frame_skip
        self.max_det    = max_det
        self.draw_traj  = draw_traj
        self.imgsz      = imgsz
        self.trajectory = TrajectoryStore(max_len=60)

    def process(
        self,
        video_path      : str,
        output_path     : str,
        classes         : list = [0],        # 0 = person in COCO
        save_screenshots: bool = True,
        screenshot_dir  : str  = "output/screenshots",
        count_log_path  : str  = "output/count_log.csv",
    ) -> tuple:
        """
        Run detection + tracking on a video file and write the annotated output.

        Returns
        -------
        (summary dict, count_over_time list of (frame, count) tuples)
        """
        video_path  = str(video_path)
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # ── open source video ────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f}fps  |  {total_frames} frames")

        # ── output video writer ──────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── state ────────────────────────────────────────────────
        total_subjects : set  = set()
        count_over_time: list = []
        ss_frames = {
            int(total_frames * 0.10),
            int(total_frames * 0.35),
            int(total_frames * 0.60),
            int(total_frames * 0.85),
        }

        frame_idx    = 0
        last_ann     = None
        t_start      = time.time()

        logger.info("Starting tracking loop ...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:

                # ── YOLOv8 + ByteTrack ──────────────────────────
                results = self.model.track(
                    frame,
                    persist = True,               # keeps IDs between frames
                    tracker = "bytetrack.yaml",
                    conf    = self.conf,
                    iou     = self.iou,
                    classes = classes,
                    max_det = self.max_det,
                    imgsz   = self.imgsz,         # inference resolution (higher = better on small/distant subjects)
                    verbose = False,
                )

                detections = sv.Detections.from_ultralytics(results[0])

                # update trajectory & unique ID set
                if detections.tracker_id is not None:
                    for tid, box in zip(detections.tracker_id, detections.xyxy):
                        total_subjects.add(int(tid))
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        self.trajectory.update(int(tid), cx, cy)

                count_over_time.append((frame_idx, len(detections)))

                last_ann = annotate_frame(
                    frame, detections, self.trajectory,
                    frame_idx, fps, total_subjects,
                    draw_trajectory=self.draw_traj,
                )

            # reuse last annotated frame for skipped frames
            ann = last_ann if last_ann is not None else frame
            writer.write(ann)

            # save screenshot
            if save_screenshots and frame_idx in ss_frames:
                ss_dir = Path(screenshot_dir)
                ss_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ss_dir / f"frame_{frame_idx:06d}.jpg"), ann)
                logger.info(f"  Screenshot saved → {ss_dir}/frame_{frame_idx:06d}.jpg")

            # progress every 100 frames
            if frame_idx % 100 == 0 and frame_idx > 0:
                pct = frame_idx / max(total_frames, 1) * 100
                elapsed = time.time() - t_start
                logger.info(f"  Progress: {pct:.1f}%  ({frame_idx}/{total_frames})  {elapsed:.0f}s elapsed")

            frame_idx += 1

        cap.release()
        writer.release()

        # ── save count log ───────────────────────────────────────
        if count_log_path:
            Path(count_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(count_log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "active_subjects"])
                w.writerows(count_over_time)
            logger.info(f"Count log saved → {count_log_path}")

        elapsed_total = time.time() - t_start
        summary = {
            "total_frames_processed" : frame_idx,
            "total_unique_subjects"  : len(total_subjects),
            "unique_ids"             : sorted(total_subjects),
            "video_duration_sec"     : round(frame_idx / fps, 2),
            "processing_time_sec"    : round(elapsed_total, 2),
            "fps_processing"         : round(frame_idx / elapsed_total, 1),
            "output_video"           : output_path,
            "model"                  : self.model.model_name if hasattr(self.model, "model_name") else "YOLOv8",
            "tracker"                : "ByteTrack",
        }

        logger.info("─" * 55)
        logger.info(f"  Total unique subjects : {len(total_subjects)}")
        logger.info(f"  Processing speed      : {summary['fps_processing']} fps")
        logger.info(f"  Output video          : {output_path}")
        logger.info("─" * 55)

        return summary, count_over_time
