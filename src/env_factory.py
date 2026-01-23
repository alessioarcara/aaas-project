import functools
from pathlib import Path

import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from pydantic import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.constants import STATS_KEY
from src.utils.typings import EncodingType, LayoutName, ShapingMode
from src.wrappers import OvercookedVectorRewardShapingWrapper, StackAgentObservationWrapper


def _make_train_env(layouts: list[LayoutName], encoding: EncodingType, info_level: int, horizon: int) -> gym.Env:
    env = OvercookedGym(
        layouts=layouts,
        encoding=encoding,
        info_level=info_level,
        horizon=horizon,
        render_mode=None,
    )
    env = StackAgentObservationWrapper(env)
    return env


def _make_eval_env(
    layout: LayoutName,
    encoding: EncodingType,
    info_level: int,
    horizon: int,
    grid_shape: tuple[int, int],
    video_path: Path,
) -> gym.Env:
    env = OvercookedGym(
        layouts=[layout],
        encoding=encoding,
        info_level=info_level,
        horizon=horizon,
        render_mode="rgb_array",
        grid_shape=grid_shape,
    )
    env = StackAgentObservationWrapper(env)
    env = gym.wrappers.RecordVideo(env, video_folder=str(video_path), episode_trigger=lambda e: True)
    return env


@validate_call
def create_train_envs(
    num_envs: int,
    layouts: list[LayoutName],
    encoding: EncodingType,
    info_level: int,
    horizon: int,
    shaping_mode: ShapingMode,
) -> gym.vector.VectorEnv:
    make_env_fn = functools.partial(
        _make_train_env, layouts=layouts, encoding=encoding, info_level=info_level, horizon=horizon
    )

    env_fns = [make_env_fn for _ in range(num_envs)]

    envs = gym.vector.AsyncVectorEnv(env_fns, context="forkserver")
    envs = OvercookedVectorRewardShapingWrapper(envs, shaping_mode=shaping_mode)
    return envs


@validate_call
def create_eval_envs(
    layouts: list[LayoutName], encoding: EncodingType, info_level: int, horizon: int, video_dir: Path
) -> gym.vector.VectorEnv:
    # Determine common grid shape for all layouts
    # to use as padding size
    max_h, max_w = 0, 0
    for layout in layouts:
        mdp = OvercookedGridworld.from_layout_name(layout)
        h, w = np.array(mdp.terrain_mtx).shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    common_shape = (max_h, max_w)

    env_fns = []
    for layout in layouts:
        video_sub_dir = video_dir / layout

        make_env_fn = functools.partial(
            _make_eval_env,
            layout=layout,
            encoding=encoding,
            info_level=info_level,
            horizon=horizon,
            grid_shape=common_shape,
            video_path=video_sub_dir,
        )
        env_fns.append(make_env_fn)

    envs = gym.vector.AsyncVectorEnv(env_fns, context="forkserver")
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs, stats_key=STATS_KEY)
    return envs
