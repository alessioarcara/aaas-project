from pathlib import Path

import gymnasium as gym
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.constants import STATS_KEY
from src.utils.typings import LayoutName, ShapingMode
from src.wrappers import OvercookedVectorRewardShapingWrapper


@validate_call
def create_train_envs(
    num_envs: int, layouts: list[LayoutName], info_level: int, horizon: int, shaping_mode: ShapingMode
) -> gym.vector.VectorEnv:
    """ """

    def make_env():
        def _thunk():
            env = OvercookedGym(layouts=layouts, info_level=info_level, horizon=horizon, render_mode=None)
            return env

        return _thunk

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver")
    envs = OvercookedVectorRewardShapingWrapper(envs, shaping_mode=shaping_mode)
    return envs


@validate_call
def create_eval_envs(layouts: list[LayoutName], info_level: int, horizon: int, video_dir: Path) -> gym.vector.VectorEnv:
    """ """

    def make_env(layout: LayoutName):
        def _thunk():
            env = OvercookedGym(layouts=[layout], info_level=info_level, horizon=horizon, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir / layout, episode_trigger=lambda e: True)
            return env

        return _thunk

    envs = gym.vector.AsyncVectorEnv([make_env(layout) for layout in layouts], context="forkserver")
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs, stats_key=STATS_KEY)
    return envs
