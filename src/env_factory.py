from pathlib import Path

import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.constants import STATS_KEY
from src.utils.typings import EncodingType, LayoutName, ShapingMode
from src.wrappers import OvercookedVectorRewardShapingWrapper, StackAgentObservationWrapper


@validate_call
def create_train_envs(
    num_envs: int,
    layouts: list[LayoutName],
    encoding: EncodingType,
    info_level: int,
    horizon: int,
    shaping_mode: ShapingMode,
) -> gym.vector.VectorEnv:
    """ """

    def make_env():
        def _thunk():
            env = OvercookedGym(
                layouts=layouts, encoding=encoding, info_level=info_level, horizon=horizon, render_mode=None
            )
            env = StackAgentObservationWrapper(env)
            return env

        return _thunk

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver")
    envs = OvercookedVectorRewardShapingWrapper(envs, shaping_mode=shaping_mode)
    return envs


@validate_call
def create_eval_envs(
    layouts: list[LayoutName], encoding: EncodingType, info_level: int, horizon: int, video_dir: Path
) -> gym.vector.VectorEnv:
    """ """
    max_h, max_w = 0, 0
    for layout in layouts:
        mdp = OvercookedGridworld.from_layout_name(layout)
        h, w = np.array(mdp.terrain_mtx).shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    common_shape = (max_h, max_w)

    def make_env(layout: LayoutName):
        def _thunk():
            env = OvercookedGym(
                layouts=[layout],
                encoding=encoding,
                info_level=info_level,
                horizon=horizon,
                render_mode="rgb_array",
                grid_shape=common_shape,
            )
            env = StackAgentObservationWrapper(env)
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir / layout, episode_trigger=lambda e: True)
            return env

        return _thunk

    envs = gym.vector.AsyncVectorEnv([make_env(layout) for layout in layouts], context="forkserver")
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs, stats_key=STATS_KEY)
    return envs
