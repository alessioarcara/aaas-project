from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.typings import PathOrStr


def read_yaml(path: PathOrStr) -> Dict[str, Any]:
    """
    Read a YAML file from a path or string and return its content as a dictionary.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
