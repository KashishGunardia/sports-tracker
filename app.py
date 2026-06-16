"""
Streamlit UI for the sports multi-object tracking pipeline.

Run locally with:
    streamlit run app.py

This is a thin front-end over the same code the CLI (main.py) uses. You
upload a video (or paste a public URL), pick a few settings, hit "Run
tracking", and the app shows the annotated video plus the heatmap, the
count-over-time chart and the run summary.
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

# Make the src/ modules importable whether we run from the repo root or elsewhere.
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from downloader import download_video, get_video_info
from tracker import SportsTracker
from visualizer import HeatmapAccumulator, save_count_chart, save_summary_card


# COCO classes we expose in the UI. People is the common case for sports.
CLASS_OPTIONS = {
    "Person (0)": 0,
    "Sports ball (32)": 32,
    "Car (2)": 2,
    "Bicycle (1)": 1,
    "Motorcycle (3)": 3,
}

MODEL_OPTIONS = {
    "YOLOv8n — fastest": "yolov8n.pt",
    "YOLOv8s — balanced": "yolov8s.pt",
    "YOLOv8m — more accurate": "yolov8m.pt",
}


def get_work_dir() -> Path:
    """One temp directory per browser session, instead of a new one every rerun."""
    if "work_dir" not in st.session_state:
        st.session_state.work_dir = tempfile.mkdtemp(prefix="tracker_")
    path = Path(st.session_state.work_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_browser_friendly_mp4(src_path: str) -> str:
    """
    OpenCV writes mp4v video, which most browsers refuse to play inline.
    Re-encode to H.264 so st.video() can show it. If ffmpeg isn't available
    we just return the original path and fall back to a download button.
    """
    try:
        import subprocess

        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        out_path = str(Path(src_path).with_name("annotated_h264.mp4"))
        subprocess.run(
            [
                ffmpeg, "-y", "-i", src_path,
                "-c:v", "libx264", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_path
    except Exception:
        return src_path


def run_pipeline(video_path, out_dir, settings):
    """Run detection + tracking and build every output file. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss_dir = out_dir / "screenshots"

    output_video = str(out_dir / "annotated_output.mp4")
    count_log = str(out_dir / "count_log.csv")

    tracker = SportsTracker(
        model_path=settings["model"],
        conf=settings["conf"],
        iou=settings["iou"],
        frame_skip=settings["frame_skip"],
        max_det=settings["max_det"],
        draw_traj=settings["draw_traj"],
        imgsz=settings["imgsz"],
    )

    summary, count_over_time = tracker.process(
        video_path=video_path,
        output_path=output_video,
        classes=settings["classes"],
        save_screenshots=True,
        screenshot_dir=str(ss_dir),
        count_log_path=count_log,
    )

    # Pull the first frame so we have a background for the heatmap and a fps value.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first_frame = cap.read()
    cap.release()

    save_count_chart(count_over_time, str(out_dir / "count_chart.jpg"), fps=fps)

    if ok and first_frame is not None:
        import supervision as sv

        heatmap = HeatmapAccumulator(first_frame.shape[1], first_frame.shape[0])
        cap2 = cv2.VideoCapture(video_path)
        frame_idx = 0
        while frame_idx < 300:
            grabbed, frame = cap2.read()
            if not grabbed:
                break
            if frame_idx % max(settings["frame_skip"], 1) == 0:
                results = tracker.model.track(
                    frame,
                    persist=False,
                    conf=settings["conf"],
                    iou=settings["iou"],
                    classes=settings["classes"],
                    max_det=settings["max_det"],
                    imgsz=settings["imgsz"],
                    verbose=False,
                )
                heatmap.update(sv.Detections.from_ultralytics(results[0]))
            frame_idx += 1
        cap2.release()
        heatmap.save(str(out_dir / "heatmap.jpg"), first_frame)

    save_summary_card(summary, str(out_dir / "summary_card.jpg"))

    # Sets aren't JSON-serialisable, so flatten them to sorted lists.
    summary_clean = {
        k: (sorted(v) if isinstance(v, set) else v)
        for k, v in summary.items()
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_clean, f, indent=2)

    return summary_clean


def main():
    st.set_page_config(page_title="Sports Object Tracker", page_icon="🎯", layout="wide")

    st.title("Sports Multi-Object Tracker")
    st.write(
        "Detect every person (or ball, car, etc.) in a video and give each one a "
        "stable ID that follows them across frames. Built on YOLOv8 + ByteTrack."
    )

    # ---- Sidebar settings -------------------------------------------------
    with st.sidebar:
        st.header("Settings")

        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
        model = MODEL_OPTIONS[model_label]

        class_labels = st.multiselect(
            "What to track",
            list(CLASS_OPTIONS.keys()),
            default=["Person (0)"],
        )
        classes = [CLASS_OPTIONS[c] for c in class_labels] or [0]

        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)
        imgsz = st.select_slider("Inference size", [640, 960, 1280, 1920], value=640)
        max_det = st.slider("Max detections / frame", 10, 300, 50, 10)
        frame_skip = st.slider("Process every Nth frame", 1, 5, 1)
        draw_traj = st.checkbox("Draw trajectory trails", value=True)

        st.caption(
            "Tip: raise inference size to 1280+ and max detections for crowded "
            "or 4K footage. Raise frame skip to speed things up on a slow CPU."
        )

    settings = {
        "model": model,
        "classes": classes,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "frame_skip": frame_skip,
        "draw_traj": draw_traj,
    }

    work_dir = get_work_dir()

    # ---- Choose a source --------------------------------------------------
    st.subheader("1. Pick a video")
    tab_upload, tab_url = st.tabs(["Upload a file", "Paste a URL"])

    video_path = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Video file", type=["mp4", "avi", "mkv", "mov"], label_visibility="collapsed"
        )
        if uploaded is not None:
            video_path = str(work_dir / uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.video(video_path)

    with tab_url:
        url = st.text_input("Public video URL (YouTube, Vimeo, direct mp4, ...)")
        st.caption(
            "Heads up: on a cloud host (like Hugging Face Spaces) YouTube often "
            "blocks downloads. A direct .mp4 link or an uploaded file is most reliable."
        )
        if url:
            try:
                info = get_video_info(url)
                if info.get("title"):
                    st.caption(f"{info.get('title')} — {info.get('duration', '?')}s")
            except Exception:
                pass  # metadata is optional; don't block on it

    # ---- Run --------------------------------------------------------------
    st.subheader("2. Run tracking")
    run = st.button("Run tracking", type="primary", use_container_width=True)

    if not run:
        return

    if video_path is None and not url:
        st.warning("Upload a video or paste a URL first.")
        return

    out_dir = work_dir / "output"

    try:
        with st.status("Working...", expanded=True) as status:
            if video_path is None:
                status.update(label="Downloading video...")
                video_path = str(work_dir / "input_video.mp4")
                download_video(url, video_path)

            if not Path(video_path).exists():
                raise FileNotFoundError("The video could not be found or downloaded.")

            status.update(label="Detecting and tracking (this can take a while on CPU)...")
            summary = run_pipeline(video_path, out_dir, settings)
            status.update(label="Done", state="complete")
    except Exception as e:
        st.error(
            f"Something went wrong while processing the video: {e}\n\n"
            "If you used a YouTube link on a cloud host, try uploading the file "
            "directly or using a direct .mp4 URL instead."
        )
        return

    # ---- Results ----------------------------------------------------------
    st.subheader("3. Results")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Unique subjects", summary.get("total_unique_subjects", "?"))
    col_b.metric("Frames processed", summary.get("total_frames_processed", "?"))
    col_c.metric("Speed", f"{summary.get('fps_processing', 0)} fps")

    annotated = out_dir / "annotated_output.mp4"
    if annotated.exists():
        playable = to_browser_friendly_mp4(str(annotated))
        st.video(playable)
        with open(annotated, "rb") as f:
            st.download_button(
                "Download annotated video", f, file_name="annotated_output.mp4",
                mime="video/mp4",
            )

    img_cols = st.columns(2)
    heatmap = out_dir / "heatmap.jpg"
    chart = out_dir / "count_chart.jpg"
    if heatmap.exists():
        img_cols[0].image(str(heatmap), caption="Movement heatmap")
    if chart.exists():
        img_cols[1].image(str(chart), caption="Active subjects over time")

    with st.expander("Full run summary (JSON)"):
        st.json(summary)


if __name__ == "__main__":
    main()

