from gymnasium.vector import VectorEnv
from pydantic import BaseModel, Field
from pydantic.types import PositiveInt

from src.env_factory import create_vector_env
from src.utils.typings import LayoutName


class TrainingConfig(BaseModel):
    num_steps: PositiveInt = Field(
        ...,
        description="Number of steps collected in the fixed-length trajectory segments",
    )
    wandb_project_name: str = Field(..., description="W&B project name for logging")
    wandb_entity: str = Field(..., description="W&B entity (user or team) for logging")


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
    env_config: EnvironmentConfig = Field(
        ..., description="Environment configuration parameters."
    )
    training: TrainingConfig = Field(
        ..., description="Training algorithm configuration parameters."
    )

    def envs(self) -> VectorEnv:
        return create_vector_env(
            num_envs=self.env_config.num_envs,
            layouts=self.env_config.layouts,
            info_level=self.env_config.info_level,
            horizon=self.env_config.horizon,
        )
