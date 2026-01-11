# Taken from my repository:
# https://github.com/alessioarcara/ml4cv_assignment
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import FilePath, ValidationError, validate_call

from src.config.config import Config
from src.utils.io import read_yaml


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge dict `b` into dict `a`
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@validate_call
def _get_experiment_config(config_paths: List[FilePath]) -> Dict[str, Any]:
    """
    Loads and merges multiple YAML configuration files into a single configuration dictionary.
    Later files in the list override keys from earlier ones.
    """

    merged_config: Dict[str, Any] = {}

    logger.info(f"📄 Building config from {len(config_paths)} files:")

    for path in config_paths:
        logger.info(f"    -> Loading: {path.name}")
        config = read_yaml(path)
        merged_config = _deep_merge(merged_config, config)

    return merged_config


def build_config(
    config_paths: List[str], overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Config, Dict[str, Any]]:
    """
    Orchestrates the loading, merging, and validation of configuration files.

    Args:
        config_paths: A list of file paths (strings or Path objects).
        overrides: An optional dictionary for the final override.

    Returns:
        A tuple containing:
        1. The validated Config object.
        2. The raw merged dictionary.
    """
    if not config_paths:
        logger.error("❌ No configuration paths provided.")
        sys.exit(1)

    paths = [Path(p) for p in config_paths]

    try:
        # Load and merge configs
        merged_config_dict = _get_experiment_config(paths)

        # Merge overrides if provided
        if overrides:
            merged_config_dict = _deep_merge(merged_config_dict, overrides)

        # Validate merged config
        config = Config.model_validate(merged_config_dict)

        return config, merged_config_dict

    except ValidationError as e:
        logger.error(f"❌ Configuration validation failed:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error building config: {e}")
        sys.exit(1)
