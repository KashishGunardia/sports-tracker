"""
Extra visual outputs, all drawn with plain OpenCV and NumPy (no matplotlib):

  - a movement heatmap laid over the first frame
  - a line chart of how many subjects were active over time
  - a small summary card with the run stats
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HeatmapAccumulator:
    """
    Adds a soft blob at each subject's centre point on every frame. After enough
    frames the busy areas of the scene light up.
    """

    def __init__(self, width: int, height: int, blur_radius: int = 30):
        self.canvas = np.zeros((height, width), dtype=np.float32)
        self.blur_radius = blur_radius

    def update(self, detections):
        """Add a blob for every box. Accepts a supervision Detections or a list of boxes."""
        if detections is None:
            return

        boxes = detections.xyxy if hasattr(detections, "xyxy") else detections
        r = self.blur_radius
        h, w = self.canvas.shape

        for box in boxes:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            x1, x2 = max(cx - r, 0), min(cx + r, w)
            y1, y2 = max(cy - r, 0), min(cy + r, h)
            self.canvas[y1:y2, x1:x2] += 1.0

    def render(self, background: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Blend the heatmap over a background frame and return the result."""
        norm = cv2.normalize(self.canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = norm > 10  # only tint areas that actually saw activity

        overlay = background.copy()
        blended = cv2.addWeighted(background, 1 - alpha, colored, alpha, 0)
        overlay[mask] = blended[mask]
        return overlay

    def save(self, output_path: str, background: np.ndarray):
        cv2.imwrite(str(output_path), self.render(background))
        logger.info(f"Heatmap saved -> {output_path}")


def save_count_chart(count_over_time: list, output_path: str, fps: float = 25.0):
    """
    Draw a line chart of active-subject count versus time and save it as a JPEG.

    count_over_time is a list of (frame_index, count) tuples.
    """
    if not count_over_time:
        logger.warning("No count data to chart.")
        return

    W, H = 900, 400
    MARGIN = 65
    img = np.ones((H, W, 3), dtype=np.uint8) * 240  # light grey background

    frames = [f for f, _ in count_over_time]
    counts = [c for _, c in count_over_time]
    max_count = max(counts) if counts else 1
    max_frame = max(frames) if frames else 1

    def to_px(frame, count):
        x = MARGIN + int((frame / max_frame) * (W - 2 * MARGIN))
        y = H - MARGIN - int((count / max(max_count, 1)) * (H - 2 * MARGIN))
        return (x, y)

    # Horizontal grid lines with count labels.
    step = max(1, max_count // 5)
    for value in range(0, max_count + step + 1, step):
        _, py = to_px(0, value)
        cv2.line(img, (MARGIN, py), (W - MARGIN, py), (200, 200, 200), 1)
        cv2.putText(img, str(value), (8, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # Vertical grid lines with time labels.
    duration_s = int(max_frame / fps)
    time_step = max(1, duration_s // 6)
    for t in range(0, duration_s + time_step, time_step):
        px, _ = to_px(int(t * fps), 0)
        cv2.line(img, (px, MARGIN), (px, H - MARGIN), (200, 200, 200), 1)
        cv2.putText(img, f"{t}s", (px - 12, H - MARGIN + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # Shade the area under the line.
    pts = [to_px(f, c) for f, c in count_over_time]
    poly = np.array(
        [(MARGIN, H - MARGIN)] + pts + [(W - MARGIN, H - MARGIN)], dtype=np.int32
    )
    fill = img.copy()
    cv2.fillPoly(fill, [poly], (173, 216, 230))
    cv2.addWeighted(fill, 0.4, img, 0.6, 0, img)

    # The line itself.
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], (30, 100, 220), 2, cv2.LINE_AA)

    cv2.rectangle(img, (MARGIN, MARGIN), (W - MARGIN, H - MARGIN), (80, 80, 80), 2)
    cv2.putText(img, "Active Subjects Over Time", (MARGIN, MARGIN - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2)
    cv2.putText(img, "Time (seconds)", (W // 2 - 65, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "Count", (6, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    cv2.imwrite(str(output_path), img)
    logger.info(f"Count chart saved -> {output_path}")


def save_summary_card(summary: dict, output_path: str):
    """Render a small dark stats card as a JPEG."""
    W, H = 640, 360
    img = np.ones((H, W, 3), dtype=np.uint8) * 22  # near-black background

    cv2.rectangle(img, (0, 0), (W, 5), (30, 144, 255), -1)  # top accent bar

    lines = [
        ("TRACKING SUMMARY", (255, 200, 0), 0.80, 2),
        ("", None, 0, 0),
        (f"Frames Processed   :  {summary.get('total_frames_processed', '?')}",
         (210, 210, 210), 0.55, 1),
        (f"Video Duration     :  {summary.get('video_duration_sec', 0):.1f} s",
         (210, 210, 210), 0.55, 1),
        (f"Unique Subjects    :  {summary.get('total_unique_subjects', '?')}",
         (80, 255, 120), 0.65, 2),
        (f"Unique IDs         :  {summary.get('unique_ids', [])}",
         (160, 190, 255), 0.42, 1),
        ("", None, 0, 0),
        (f"Processing Speed   :  {summary.get('fps_processing', 0):.1f} fps",
         (210, 210, 210), 0.52, 1),
        (f"Total Process Time :  {summary.get('processing_time_sec', 0):.1f} s",
         (210, 210, 210), 0.52, 1),
        ("", None, 0, 0),
        ("Model   :  YOLOv8n          Tracker :  ByteTrack", (130, 180, 255), 0.48, 1),
    ]

    y = 50
    for text, color, scale, thickness in lines:
        if not text:
            y += 16
            continue
        cv2.putText(img, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += int(scale * 58) + 4

    cv2.rectangle(img, (0, H - 4), (W, H), (30, 144, 255), -1)  # bottom accent bar

    cv2.imwrite(str(output_path), img)
    logger.info(f"Summary card saved -> {output_path}")
