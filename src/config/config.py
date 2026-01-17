import sys
from pathlib import Path
from typing import Any, Optional, Self

from gymnasium.vector import VectorEnv
from loguru import logger
from pydantic import BaseModel, DirectoryPath, Field, ValidationError
from pydantic.types import PositiveInt

from src.config.utils import _deep_merge
from src.env_factory import create_eval_env, create_vector_env
from src.utils.io import read_yaml
from src.utils.typings import LayoutName, ShapingMode


class TrainingConfig(BaseModel):
    total_updates: PositiveInt = Field(..., description="Total number of training updates")
    num_steps: PositiveInt = Field(
        ...,
        description="Length of each rollout segment collected per environment before an update.",
    )
    gae_lambda: float = Field(
        0.95,
        description="Controls the bias-variance trade-off for advantage estimation",
    )
    gae_gamma: float = Field(0.99, description="Discount factor for future rewards")
    ppo_epsilon: float = Field(0.2, description="Clip parameter (epsilon) to constrain policy updates.")
    minibatch_size: PositiveInt = Field(...)
    update_epochs: PositiveInt = Field(
        ..., description="Number of times to iterate through the entire collected rollout segment for update."
    )
    learning_rate: float = Field(...)
    adam_epsilon: float = Field(1e-5, description="Epsilon parameter for the Adam optimizer")
    value_coef: float = Field(
        0.5, description="Coefficient scaling the value function loss in the total loss objective."
    )
    entropy_coef: float = Field(0.01, description="Coefficient scaling the entropy bonus to encourage exploration.")


class EnvironmentConfig(BaseModel):
    num_envs: PositiveInt = Field(..., description="Number of parallel environments")
    layouts: list[LayoutName] = Field(
        default=["cramped_room"],
        min_length=1,
        description="List of Overcooked layouts to use for training (must contain at least one).",
    )
    info_level: int = Field(1, description="Information level for the environment logging/observation.")
    horizon: PositiveInt = Field(400, description="Time horizon (max steps) for each episode.")
    shaping_mode: ShapingMode = Field(
        default=ShapingMode.NONE, description="Mode of reward shaping to use in the environment."
    )


class Config(BaseModel):
    seed: int = Field(default=42, description="Random seed for reproducibility")
    exp_name: str = Field(..., description="Name of the experiment")
    wandb_project_name: str = Field(..., description="W&B project name for logging")
    wandb_entity: str = Field(..., description="W&B entity (user or team) for logging")
    env_config: EnvironmentConfig = Field(..., description="Environment config")
    training_config: TrainingConfig = Field(..., description="Training config")
    video_dir: DirectoryPath = Field(default=Path("./videos"), description="Directory to save training videos")
    eval_interval: PositiveInt = Field(default=1, description="Perform an evaluation run every N updates.")

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

        # 1. Loading and merging files
        logger.info(f"📄 Building config from {len(paths)} files:")
        for path in paths:
            logger.info(f"    -> Loading: {path.name}")
            try:
                config_data = read_yaml(path)
                merged_config = _deep_merge(merged_config, config_data)
            except Exception as e:
                logger.error(f"❌ Error reading {path}: {e}")
                sys.exit(1)

        # 2. Merge overrides if provided
        if overrides:
            merged_config = _deep_merge(merged_config, overrides)

        # 3. Validation and instantiation
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
            shaping_mode=self.env_config.shaping_mode,
        )

    @property
    def eval_env(self) -> VectorEnv:
        return create_eval_env(
            layouts=self.env_config.layouts,
            info_level=self.env_config.info_level,
            horizon=self.env_config.horizon,
            video_folder=str(self.video_dir),
        )
