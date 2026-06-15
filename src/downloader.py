"""
downloader.py
=============
Download any public video (YouTube, Vimeo, etc.) using yt-dlp.
"""

import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_video(url: str, output_path: str = "input_video.mp4") -> str:
    """
    Download a public video to output_path using yt-dlp.

    Parameters
    ----------
    url         : public video URL
    output_path : file path where the video will be saved

    Returns
    -------
    str : path to the saved video file
    """
    output_path = str(output_path)

    # auto-install yt-dlp if missing
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        logger.info("yt-dlp not found — installing ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "yt-dlp", "-q"
        ])

    logger.info(f"Downloading: {url}")
    logger.info(f"Saving to  : {output_path}")

    ydl_opts = {
        # prefer 720p mp4 so processing stays fast
        "format"               : "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl"              : output_path,
        "merge_output_format"  : "mp4",
        "quiet"                : False,
        "no_warnings"          : True,
    }

    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    logger.info("Download complete.")
    return output_path


def get_video_info(url: str) -> dict:
    """Return metadata (title, duration, uploader) without downloading."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title"     : info.get("title", "Unknown"),
                "duration"  : info.get("duration", 0),
                "uploader"  : info.get("uploader", "Unknown"),
                "view_count": info.get("view_count", 0),
                "url"       : url,
            }
    except Exception as e:
        logger.warning(f"Could not fetch video info: {e}")
        return {"url": url}
