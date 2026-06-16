"""
Command-line entry point for the sports tracking pipeline.

Examples:
    # Download and process a public/YouTube video:
    python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

    # Process a local file:
    python main.py --video "path/to/match.mp4"

    # See everything:
    python main.py --help

Prefer a UI? Run `streamlit run app.py` instead.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

# Let us import the modules in src/ without installing the project as a package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from downloader import download_video, get_video_info
from tracker import SportsTracker
from visualizer import HeatmapAccumulator, save_count_chart, save_summary_card

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-object detection and persistent ID tracking in sports videos",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--url", type=str,
                        help="Public video URL (YouTube, Vimeo, ...) — downloaded automatically")
    source.add_argument("--video", type=str,
                        help="Path to a local video file (.mp4, .avi, .mkv, ...)")

    p.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help=(
            "YOLOv8 weights to use.\n"
            "  yolov8n.pt  fastest (default)\n"
            "  yolov8s.pt  balanced\n"
            "  yolov8m.pt  more accurate\n"
            "  yolov8l.pt  best accuracy, slowest\n"
            "(downloaded automatically on first use)"
        ),
    )
    p.add_argument("--conf", type=float, default=0.35,
                   help="Detection confidence threshold [default: 0.35]")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold [default: 0.45]")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Process every Nth frame (1 = every frame) [default: 1]")
    p.add_argument("--max-det", type=int, default=50,
                   help="Max detections per frame [default: 50]")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference resolution (640 fast; 1280/1920 better for distant subjects) [default: 640]")
    p.add_argument("--classes", type=int, nargs="+", default=[0],
                   help="COCO class IDs to track (0=person, 2=car, 32=sports ball) [default: 0]")
    p.add_argument("--no-traj", action="store_true",
                   help="Turn off trajectory trails")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Folder for all output files [default: output/]")

    return p.parse_args()


def prompt_for_source(args):
    """If the user ran the script with no source, ask them interactively."""
    print("\nChoose input type:")
    print("1. Local video")
    print("2. YouTube / public URL")
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        path = input("Enter video path (example: basketball.mp4): ").strip()
        if not Path(path).exists():
            print(f"\nVideo not found: {path}")
            sys.exit(1)
        args.video = path
    elif choice == "2":
        url = input("Enter YouTube / public URL: ").strip()
        if not url:
            print("\nInvalid URL")
            sys.exit(1)
        args.url = url
    else:
        print("\nInvalid choice")
        sys.exit(1)


def build_heatmap(tracker, video_path, first_frame, frame_skip, args, out_dir):
    """Make a second, lightweight pass over the first 300 frames to build the heatmap."""
    import supervision as sv

    logger.info("Generating movement heatmap (scanning first 300 frames) ...")
    heatmap = HeatmapAccumulator(first_frame.shape[1], first_frame.shape[0])

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while frame_idx < 300:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(frame_skip, 1) == 0:
            results = tracker.model.track(
                frame, persist=False, conf=args.conf, iou=args.iou,
                classes=args.classes, max_det=args.max_det, imgsz=args.imgsz,
                verbose=False,
            )
            heatmap.update(sv.Detections.from_ultralytics(results[0]))
        frame_idx += 1
    cap.release()

    heatmap.save(str(out_dir / "heatmap.jpg"), first_frame)


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss_dir = out_dir / "screenshots"

    if not args.video and not args.url:
        prompt_for_source(args)

    # Resolve the source video (download it if we were given a URL).
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
    count_log = str(out_dir / "count_log.csv")

    # Detection + tracking.
    tracker = SportsTracker(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        frame_skip=args.frame_skip,
        max_det=args.max_det,
        draw_traj=not args.no_traj,
        imgsz=args.imgsz,
    )
    summary, count_over_time = tracker.process(
        video_path=video_path,
        output_path=output_video,
        classes=args.classes,
        save_screenshots=True,
        screenshot_dir=str(ss_dir),
        count_log_path=count_log,
    )

    # Read the first frame once, for both fps and the heatmap background.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, first_frame = cap.read()
    cap.release()

    save_count_chart(count_over_time, str(out_dir / "count_chart.jpg"), fps=fps)

    if ret and first_frame is not None:
        build_heatmap(tracker, video_path, first_frame, args.frame_skip, args, out_dir)

    save_summary_card(summary, str(out_dir / "summary_card.jpg"))

    # Sets can't be written to JSON, so turn them into sorted lists first.
    summary_clean = {
        k: (sorted(v) if isinstance(v, set) else v)
        for k, v in summary.items()
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_clean, f, indent=2)
    logger.info(f"Summary saved -> {summary_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — output files:")
    logger.info("=" * 60)
    logger.info(f"Annotated video -> {output_video}")
    logger.info(f"Heatmap image   -> {out_dir / 'heatmap.jpg'}")
    logger.info(f"Count chart     -> {out_dir / 'count_chart.jpg'}")
    logger.info(f"Summary card    -> {out_dir / 'summary_card.jpg'}")
    logger.info(f"Screenshots     -> {ss_dir}/")
    logger.info(f"Count log (CSV) -> {count_log}")
    logger.info(f"Summary (JSON)  -> {summary_path}")
    logger.info("=" * 60)
    logger.info(f"Unique subjects tracked : {summary['total_unique_subjects']}")
    logger.info(f"IDs assigned            : {summary['unique_ids']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
