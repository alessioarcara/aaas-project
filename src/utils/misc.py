import random
from pathlib import Path
from typing import Optional

import numpy as np


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def latest_video_path(directory: Path) -> Optional[Path]:
    mp4s = list(directory.glob("*.mp4"))
    return max(mp4s, key=lambda p: p.stat().st_mtime) if mp4s else None
