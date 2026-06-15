"""
main.py
=======
Entry point for the Sports Multi-Object Tracking Pipeline.

HOW TO RUN FROM VS CODE TERMINAL
─────────────────────────────────
# Option A — YouTube / public URL (auto-downloads the video):
    python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Option B — local video file already on your machine:
    python main.py --video "C:/Users/You/Downloads/match.mp4"

# All options:
    python main.py --help
"""

import argparse
import logging
import sys
import json
import cv2
from pathlib import Path

# ── make src/ importable without installing as a package ──────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from downloader import download_video, get_video_info
from tracker    import SportsTracker
from visualizer import HeatmapAccumulator, save_count_chart, save_summary_card

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers= [logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  CLI arguments
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description = "Multi-Object Detection & Persistent ID Tracking in Sports Videos",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        type=str,
        help="Public video URL (YouTube, Vimeo, etc.) — auto-downloaded",
    )
    source.add_argument(
        "--video",
        type=str,
        help="Path to a local video file (.mp4, .avi, .mkv, ...)",
    )

    p.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help=(
            "YOLOv8 weights to use.\n"
            "  yolov8n.pt  — fastest  (default)\n"
            "  yolov8s.pt  — balanced\n"
            "  yolov8m.pt  — more accurate\n"
            "  yolov8l.pt  — best accuracy, slowest\n"
            "(auto-downloaded on first use)"
        ),
    )
    p.add_argument("--conf",       type=float, default=0.35,
                   help="Detection confidence threshold  [default: 0.35]")
    p.add_argument("--iou",        type=float, default=0.45,
                   help="NMS IoU threshold               [default: 0.45]")
    p.add_argument("--frame-skip", type=int,   default=1,
                   help="Process every Nth frame (1=every frame, 2=every other frame)  [default: 1]")
    p.add_argument("--max-det",    type=int,   default=50,
                   help="Max detections per frame        [default: 50]")
    p.add_argument("--imgsz",      type=int,   default=640,
                   help="Inference resolution (640 fast; 1280/1920 better for small/distant subjects)  [default: 640]")
    p.add_argument("--classes",    type=int,   nargs="+", default=[0],
                   help="COCO class IDs to track (0=person, 2=car, ...)  [default: 0]")
    p.add_argument("--no-traj",    action="store_true",
                   help="Disable trajectory trail drawing")
    p.add_argument("--output-dir", type=str,   default="output",
                   help="Folder for all output files      [default: output/]")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss_dir  = out_dir / "screenshots"

    # ── 1. Resolve source video ───────────────────────────────
    if args.url:
        info = get_video_info(args.url)
        logger.info(f"Title    : {info.get('title', 'Unknown')}")
        logger.info(f"Duration : {info.get('duration', '?')} s")
        video_path = str(out_dir / "input_video.mp4")
        download_video(args.url, video_path)
    else:
        video_path = args.video
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
        logger.info(f"Using local video: {video_path}")

    output_video = str(out_dir / "annotated_output.mp4")
    count_log    = str(out_dir / "count_log.csv")

    # ── 2. Run detection + tracking ───────────────────────────
    tracker = SportsTracker(
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        frame_skip = args.frame_skip,
        max_det    = args.max_det,
        draw_traj  = not args.no_traj,
        imgsz      = args.imgsz,
    )

    summary, count_over_time = tracker.process(
        video_path       = video_path,
        output_path      = output_video,
        classes          = args.classes,
        save_screenshots = True,
        screenshot_dir   = str(ss_dir),
        count_log_path   = count_log,
    )

    # ── 3. Extra visualizations ───────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, first_frame = cap.read()
    cap.release()

    # count chart
    save_count_chart(
        count_over_time,
        output_path = str(out_dir / "count_chart.jpg"),
        fps         = fps,
    )

    # movement heatmap (built from first 300 detected frames)
    if ret and first_frame is not None:
        logger.info("Generating movement heatmap (scanning first 300 frames) ...")
        hm   = HeatmapAccumulator(first_frame.shape[1], first_frame.shape[0])
        cap2 = cv2.VideoCapture(video_path)
        fi   = 0
        import supervision as sv
        while fi < 300:
            ok, frm = cap2.read()
            if not ok:
                break
            if fi % max(args.frame_skip, 1) == 0:
                results = tracker.model.track(
                    frm,
                    persist  = False,
                    conf     = args.conf,
                    iou      = args.iou,
                    classes  = args.classes,
                    max_det  = args.max_det,
                    imgsz    = args.imgsz,
                    verbose  = False,
                )
                det = sv.Detections.from_ultralytics(results[0])
                hm.update(det)
            fi += 1
        cap2.release()
        hm.save(str(out_dir / "heatmap.jpg"), first_frame)

    # summary card + JSON
    save_summary_card(summary, str(out_dir / "summary_card.jpg"))

    summary_clean = {
        k: (sorted(v) if isinstance(v, set) else v)
        for k, v in summary.items()
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_clean, f, indent=2)

    # ── 4. Final output table ─────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE — output files:")
    logger.info("=" * 60)
    logger.info(f"  Annotated video  →  {output_video}")
    logger.info(f"  Heatmap image    →  {out_dir/'heatmap.jpg'}")
    logger.info(f"  Count chart      →  {out_dir/'count_chart.jpg'}")
    logger.info(f"  Summary card     →  {out_dir/'summary_card.jpg'}")
    logger.info(f"  Screenshots      →  {ss_dir}/")
    logger.info(f"  Count log (CSV)  →  {count_log}")
    logger.info(f"  Summary (JSON)   →  {out_dir/'summary.json'}")
    logger.info("=" * 60)
    logger.info(f"  Unique subjects tracked : {summary['total_unique_subjects']}")
    logger.info(f"  IDs assigned            : {summary['unique_ids']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
