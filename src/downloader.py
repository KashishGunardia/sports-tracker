"""
Download public videos (YouTube, Vimeo, direct links, ...) with yt-dlp.

yt-dlp is listed in requirements.txt, so it should already be installed. We
don't try to pip-install it at runtime — that doesn't work on most hosted
environments and hides real setup problems.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def _require_yt_dlp():
    if yt_dlp is None:
        raise ImportError(
            "yt-dlp is not installed. Run `pip install -r requirements.txt` "
            "(or `pip install yt-dlp`) and try again."
        )


def download_video(url: str, output_path: str = "input_video.mp4") -> str:
    """Download a video to output_path and return that path."""
    _require_yt_dlp()
    output_path = str(output_path)

    logger.info(f"Downloading: {url}")
    logger.info(f"Saving to:  {output_path}")

    options = {
        # Grab a single progressive mp4 so we don't need ffmpeg to merge streams.
        "format": "best[ext=mp4][height<=720]/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    logger.info("Download complete.")
    return output_path


def get_video_info(url: str) -> dict:
    """Return title/duration/uploader metadata without downloading the video."""
    if yt_dlp is None:
        return {"url": url}

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
            "view_count": info.get("view_count", 0),
            "url": url,
        }
    except Exception as e:
        logger.warning(f"Could not fetch video info: {e}")
        return {"url": url}
