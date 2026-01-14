from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.typings import LayoutName


@validate_call
def create_vector_env(
    num_envs: int,
    layouts: list[LayoutName],
    info_level: int,
    horizon: int,
    video_dir: Path,
    video_interval: int,
) -> VectorEnv:
    def make_env(idx: int):
        def _thunk():
            render_mode = "rgb_array" if idx == 0 else None
            env = OvercookedGym(
                layouts=layouts,
                info_level=info_level,
                horizon=horizon,
                render_mode=render_mode,
            )

            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=str(video_dir),
                    step_trigger=lambda s: s % video_interval == 0,
                )
            return env

        return _thunk

    envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    return envs
