import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Self, Union

from gymnasium.vector import VectorEnv
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic.types import PositiveInt

from src.config.utils import _deep_merge
from src.env_factory import create_eval_envs, create_train_envs
from src.utils.io import read_yaml
from src.utils.typings import EncodingType, LayoutName, ShapingMode


class BaseNetworkConfig(BaseModel):
    shared_backbone: bool = Field(
        True, description="Whether to share the backbone between the policy and value networks."
    )
    embedding_dim: PositiveInt = Field(...)


class MLPConfig(BaseNetworkConfig):
    type: Literal["mlp"] = Field("mlp", description="Network architecture identifier.")
    hidden_dim: PositiveInt = Field(...)


class CNNConfig(BaseNetworkConfig):
    type: Literal["cnn"] = Field("cnn", description="Network architecture identifier.")
    num_filters: list[PositiveInt] = Field(...)
    kernel_sizes: list[PositiveInt] = Field(...)
    strides: list[PositiveInt] = Field(...)
    paddings: list[Literal["SAME", "VALID"]] = Field(...)

    @model_validator(mode="after")
    def check_lengths_match(self) -> Self:
        if not (len(self.num_filters) == len(self.kernel_sizes) == len(self.strides) == len(self.paddings)):
            raise ValueError("num_filters, kernel_sizes, strides, and paddings must have the same length.")
        return self


NetworkConfig = Annotated[Union[MLPConfig, CNNConfig], Field(discriminator="type")]


class TrainingConfig(BaseModel):
    # ! --- Rollout ---
    total_updates: PositiveInt = Field(..., description="Total number of training updates")
    num_steps: PositiveInt = Field(
        ...,
        description="Length of each rollout segment collected per environment before an update.",
    )
    # ! --- PPO ---
    ppo_epsilon: float = Field(0.2, description="Clip parameter (epsilon) to constrain policy updates.")
    entropy_coef: float = Field(0.01, description="Coefficient scaling the entropy bonus to encourage exploration.")
    value_coef: float = Field(
        0.5, description="Coefficient scaling the value function loss in the total loss objective."
    )
    target_kl: Optional[float] = Field(
        None,
        description="Target KL divergence for early stopping the current update. If None, no early stopping is used.",
    )
    # ! --- GAE ---
    gae_lambda: float = Field(
        0.95,
        description="Controls the bias-variance trade-off for advantage estimation",
    )
    gae_gamma: float = Field(0.99, description="Discount factor for future rewards")
    use_advantage_normalization: bool = Field(True, description="Whether to normalize the computed advantages.")
    # ! --- Optimization ---
    learning_rate: float = Field(...)
    minibatch_size: PositiveInt = Field(...)
    update_epochs: PositiveInt = Field(
        ..., description="Number of times to iterate through the entire collected rollout segment for update."
    )
    adam_epsilon: float = Field(1e-5, description="Epsilon parameter for the Adam optimizer")
    adam_momentum: float = Field(0.9, description="Momentum parameter for the Adam optimizer")
    use_learning_rate_annealing: bool = Field(
        True, description="Whether to linearly anneal the learning rate during training."
    )
    use_gradient_clipping: bool = Field(
        False, description="Whether to clip gradients by global norm during the optimization step."
    )
    # ! --- Architecture ---
    network: NetworkConfig = Field(default=..., description="Configuration for the neural network architecture.")
    # ! --- Multi-Agent ---
    use_parameter_sharing: bool = Field(
        True, description="Whether to share parameters between the two agents in the environment."
    )


class EnvironmentConfig(BaseModel):
    num_envs: PositiveInt = Field(..., description="Number of parallel environments")
    layouts: list[LayoutName] = Field(
        default=["cramped_room"],
        min_length=1,
        description="List of Overcooked layouts to use for training (must contain at least one).",
    )
    encoding: EncodingType = Field(
        default=EncodingType.FEATURIZED,
        description="The observation format. Use 'LOSSLESS' for spatial grid tensors (suitable for CNNs) or 'FEATURIZED' for 1D vectors of hand-crafted features (suitable for MLPs).",
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
    wandb_group: Optional[str] = Field(default=None, description="W&B group name to group related runs together")
    env_config: EnvironmentConfig = Field(..., description="Environment config")
    training_config: TrainingConfig = Field(..., description="Training config")
    # ! --- Evaluation ---
    video_dir: Optional[Path] = Field(
        default=None, description="Directory to save training videos. Set to None to disable recording."
    )
    eval_interval: PositiveInt = Field(default=1, description="Perform an evaluation run every N updates.")
    num_eval_episodes: PositiveInt = Field(
        default=5, description="Number of evaluation episodes to run per layout during each evaluation."
    )
    checkpoint_dir: Optional[Path] = Field(default=None, description="Directory to save model checkpoints.")

    @classmethod
    def from_files(
        cls: Self,
        config_paths: list[str | Path],
        overrides: Optional[dict[str, Any]] = None,
    ) -> Self:
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

    def setup_directories(self):
        if self.video_dir is not None:
            if self.video_dir.exists():
                logger.info(f"Clearing existing video directory at {self.video_dir}")
                shutil.rmtree(self.video_dir)
            self.video_dir.mkdir(parents=True, exist_ok=True)

    @property
    def train_envs(self) -> VectorEnv:
        return create_train_envs(
            num_envs=self.env_config.num_envs,
            layouts=self.env_config.layouts,
            encoding=self.env_config.encoding,
            info_level=self.env_config.info_level,
            horizon=self.env_config.horizon,
            shaping_mode=self.env_config.shaping_mode,
        )

    @property
    def eval_envs(self) -> VectorEnv:
        return create_eval_envs(
            layouts=self.env_config.layouts,
            encoding=self.env_config.encoding,
            info_level=self.env_config.info_level,
            horizon=self.env_config.horizon,
            video_dir=self.video_dir,
        )
