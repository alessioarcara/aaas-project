import sys
from pathlib import Path
from typing import Any, Optional, Self

from gymnasium.vector import VectorEnv
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from pydantic.types import PositiveInt

from src.config.utils import _deep_merge
from src.env_factory import create_vector_env
from src.utils.io import read_yaml
from src.utils.typings import LayoutName


class TrainingConfig(BaseModel):
    num_steps: PositiveInt = Field(
        ...,
        description="Number of steps collected in the fixed-length trajectory segments",
    )
    gae_lambda: float = Field(
        0.95,
        description="Controls the bias-variance trade-off for advantage estimation",
    )
    gae_gamma: float = Field(0.99, description="Discount factor for future rewards")


class EnvironmentConfig(BaseModel):
    num_envs: PositiveInt = Field(..., description="Number of parallel environments")
    layouts: list[LayoutName] = Field(
        default=["cramped_room"],
        min_length=1,
        description="List of Overcooked layouts to use for training (must contain at least one).",
    )
    info_level: int = Field(
        1, description="Information level for the environment logging/observation."
    )
    horizon: PositiveInt = Field(
        400, description="Time horizon (max steps) for each episode."
    )


class Config(BaseModel):
    seed: int = Field(default=42, description="Random seed for reproducibility")
    exp_name: str = Field(..., description="Name of the experiment")
    wandb_project_name: str = Field(..., description="W&B project name for logging")
    wandb_entity: str = Field(..., description="W&B entity (user or team) for logging")
    env_config: EnvironmentConfig = Field(..., description="Environment config")
    training_config: TrainingConfig = Field(..., description="Training config")

    @classmethod
    def from_files(
        cls: Self,
        config_paths: list[str | Path],
        overrides: Optional[dict[str, Any]] = None,
    ) -> "Config":
        """
        Factory method to create a Config instance from multiple YAML files and optional overrides.

        Args:
            config_paths: List of file paths (strings or Path objects) to YAML configuration files.
            overrides: An optional dictionary containing configuration overrides.
        Returns:
            An instance of Config with the merged and validated configuration.
        """
        if not config_paths:
            logger.error("❌ No configuration paths provided.")
            sys.exit(1)

        merged_config: dict[str, Any] = {}
        paths = [Path(p) for p in config_paths]

        # Loading and merging files
        logger.info(f"📄 Building config from {len(paths)} files:")
        for path in paths:
            logger.info(f"    -> Loading: {path.name}")
            try:
                config_data = read_yaml(path)
                merged_config = _deep_merge(merged_config, config_data)
            except Exception as e:
                logger.error(f"❌ Error reading {path}: {e}")
                sys.exit(1)

        # Merge overrides if provided
        if overrides:
            merged_config = _deep_merge(merged_config, overrides)

        # Validation and instantiation
        try:
            instance = cls.model_validate(merged_config)
            return instance
        except ValidationError as e:
            logger.error(f"❌ Configuration validation failed:\n{e}")
            sys.exit(1)

    @property
    def envs(self) -> VectorEnv:
        return create_vector_env(
            num_envs=self.env_config.num_envs,
            layouts=self.env_config.layouts,
            info_level=self.env_config.info_level,
            horizon=self.env_config.horizon,
        )
