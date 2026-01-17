from gymnasium.vector import AsyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.typings import LayoutName, ShapingMode
from src.wrappers import OvercookedVectorRewardShapingWrapper


@validate_call
def create_train_envs(
    num_envs: int, layouts: list[LayoutName], info_level: int, horizon: int, shaping_mode: ShapingMode
) -> VectorEnv:
    def make_env():
        def _thunk():
            env = OvercookedGym(layouts=layouts, info_level=info_level, horizon=horizon, render_mode=None)
            return env

        return _thunk

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver")
    envs = OvercookedVectorRewardShapingWrapper(envs, shaping_mode=shaping_mode)
    return envs


@validate_call
def create_eval_env(layouts: list[LayoutName], info_level: int, horizon: int, video_folder: str) -> VectorEnv:
    env = OvercookedGym(layouts=layouts, info_level=info_level, horizon=horizon, render_mode="rgb_array")
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
    )
    return env
