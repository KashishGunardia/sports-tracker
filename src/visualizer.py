"""
visualizer.py
=============
Optional enhancement visualizations (all pure OpenCV / NumPy — no matplotlib needed):

  - Movement heatmap overlaid on first video frame
  - Active subject count over time chart
  - Summary stats card image
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Movement heatmap
# ─────────────────────────────────────────────────────────────
class HeatmapAccumulator:
    """
    Accumulates Gaussian density blobs at each detected subject's
    centre point across all processed frames.
    """

    def __init__(self, width: int, height: int, blur_radius: int = 30):
        self.canvas      = np.zeros((height, width), dtype=np.float32)
        self.blur_radius = blur_radius

    def update(self, detections):
        """Add a Gaussian blob for every detected bounding box."""
        if detections is None:
            return
        # support both supervision Detections and plain list of (x1,y1,x2,y2)
        if hasattr(detections, "xyxy"):
            boxes = detections.xyxy
        else:
            boxes = detections

        r = self.blur_radius
        for box in boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            h, w = self.canvas.shape
            x1 = max(cx - r, 0); x2 = min(cx + r, w)
            y1 = max(cy - r, 0); y2 = min(cy + r, h)
            self.canvas[y1:y2, x1:x2] += 1.0

    def render(self, background: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Overlay the heatmap on a background frame."""
        norm    = cv2.normalize(self.canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask    = norm > 10
        overlay = background.copy()
        blended = cv2.addWeighted(background, 1 - alpha, colored, alpha, 0)
        overlay[mask] = blended[mask]
        return overlay

    def save(self, output_path: str, background: np.ndarray):
        """Save the heatmap overlay as a JPEG image."""
        result = self.render(background)
        cv2.imwrite(str(output_path), result)
        logger.info(f"Heatmap saved → {output_path}")


# ─────────────────────────────────────────────────────────────
#  Active subject count chart  (pure OpenCV, no matplotlib)
# ─────────────────────────────────────────────────────────────
def save_count_chart(
    count_over_time : list,
    output_path     : str,
    fps             : float = 25.0,
):
    """
    Draw a line chart of active subject count vs. time and save as JPEG.

    Parameters
    ----------
    count_over_time : list of (frame_index, subject_count) tuples
    output_path     : where to save the image
    fps             : frames-per-second of the source video
    """
    if not count_over_time:
        logger.warning("No count data to chart.")
        return

    W, H   = 900, 400
    MARGIN = 65
    img    = np.ones((H, W, 3), dtype=np.uint8) * 240   # light grey

    frames = [f for f, _ in count_over_time]
    counts = [c for _, c in count_over_time]
    max_c  = max(counts) if counts else 1
    max_f  = max(frames) if frames else 1

    def to_px(frame, count):
        x = MARGIN + int((frame / max_f) * (W - 2 * MARGIN))
        y = H - MARGIN - int((count / max(max_c, 1)) * (H - 2 * MARGIN))
        return (x, y)

    # horizontal grid lines
    step = max(1, max_c // 5)
    for g in range(0, max_c + step + 1, step):
        _, py = to_px(0, g)
        cv2.line(img, (MARGIN, py), (W - MARGIN, py), (200, 200, 200), 1)
        cv2.putText(img, str(g), (8, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # vertical time labels
    duration_s = int(max_f / fps)
    time_step  = max(1, duration_s // 6)
    for t in range(0, duration_s + time_step, time_step):
        px, _ = to_px(int(t * fps), 0)
        cv2.line(img, (px, MARGIN), (px, H - MARGIN), (200, 200, 200), 1)
        cv2.putText(img, f"{t}s", (px - 12, H - MARGIN + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # filled area under the line
    pts = [to_px(f, c) for f, c in count_over_time]
    poly_pts = (
        [(MARGIN, H - MARGIN)]
        + pts
        + [(W - MARGIN, H - MARGIN)]
    )
    poly = np.array(poly_pts, dtype=np.int32)
    fill = img.copy()
    cv2.fillPoly(fill, [poly], (173, 216, 230))       # light-blue fill
    cv2.addWeighted(fill, 0.4, img, 0.6, 0, img)

    # main line
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], (30, 100, 220), 2, cv2.LINE_AA)

    # axes border
    cv2.rectangle(img, (MARGIN, MARGIN), (W - MARGIN, H - MARGIN), (80, 80, 80), 2)

    # title & axis labels
    cv2.putText(img, "Active Subjects Over Time",
                (MARGIN, MARGIN - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2)
    cv2.putText(img, "Time (seconds)",
                (W // 2 - 65, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "Count",
                (6, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imwrite(str(output_path), img)
    logger.info(f"Count chart saved → {output_path}")


# ─────────────────────────────────────────────────────────────
#  Summary card image
# ─────────────────────────────────────────────────────────────
def save_summary_card(summary: dict, output_path: str):
    """Save a dark-theme summary card as a JPEG image."""
    W, H = 640, 360
    img  = np.ones((H, W, 3), dtype=np.uint8) * 22    # very dark background

    # accent bar
    cv2.rectangle(img, (0, 0), (W, 5), (30, 144, 255), -1)

    lines = [
        ("TRACKING SUMMARY",                                    (255, 200,   0), 0.80, 2),
        ("",                                                     None,           0,    0),
        (f"Frames Processed   :  {summary.get('total_frames_processed', '?')}",
                                                                 (210, 210, 210), 0.55, 1),
        (f"Video Duration     :  {summary.get('video_duration_sec', 0):.1f} s",
                                                                 (210, 210, 210), 0.55, 1),
        (f"Unique Subjects    :  {summary.get('total_unique_subjects', '?')}",
                                                                 ( 80, 255, 120), 0.65, 2),
        (f"Unique IDs         :  {summary.get('unique_ids', [])}",
                                                                 (160, 190, 255), 0.42, 1),
        ("",                                                     None,           0,    0),
        (f"Processing Speed   :  {summary.get('fps_processing', 0):.1f} fps",
                                                                 (210, 210, 210), 0.52, 1),
        (f"Total Process Time :  {summary.get('processing_time_sec', 0):.1f} s",
                                                                 (210, 210, 210), 0.52, 1),
        ("",                                                     None,           0,    0),
        ("Model   :  YOLOv8n          Tracker :  ByteTrack",    (130, 180, 255), 0.48, 1),
    ]

    y = 50
    for text, color, scale, thickness in lines:
        if not text:
            y += 16
            continue
        cv2.putText(img, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += int(scale * 58) + 4

    # bottom accent bar
    cv2.rectangle(img, (0, H - 4), (W, H), (30, 144, 255), -1)

    cv2.imwrite(str(output_path), img)
    logger.info(f"Summary card saved → {output_path}")
