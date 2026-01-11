from pathlib import Path


def latest_video_path(directory: Path):
    mp4s = list(directory.glob("*.mp4"))
    return max(mp4s, key=lambda p: p.stat().st_mtime) if mp4s else None
