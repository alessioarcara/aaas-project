from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.constants import STATS_KEY
from src.utils.typings import LayoutName, ShapingMode
from src.wrappers import OvercookedVectorRewardShapingWrapper


@validate_call
def create_train_envs(
    num_envs: int, layouts: list[LayoutName], info_level: int, horizon: int, shaping_mode: ShapingMode
) -> VectorEnv:
    def make_env():
        def _thunk():
            # ! fixed env for now
            env = OvercookedGym(layouts=["cramped_room"], info_level=info_level, horizon=horizon, render_mode=None)
            return env

        return _thunk

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver")
    envs = OvercookedVectorRewardShapingWrapper(envs, shaping_mode=shaping_mode)
    return envs


@validate_call
def create_eval_envs(
    layouts: list[LayoutName], info_level: int, horizon: int, video_dir: Path
) -> dict[LayoutName, gym.Env]:
    # ? is this the right way to do it?
    envs_dict = {}
    for layout in layouts:
        env = OvercookedGym(layouts=[layout], info_level=info_level, horizon=horizon, render_mode="rgb_array")
        env = RecordEpisodeStatistics(env, stats_key=STATS_KEY)
        env = RecordVideo(env, video_folder=video_dir / layout, episode_trigger=lambda x: True)
        envs_dict[layout] = env

    return envs_dict
