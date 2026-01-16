from gymnasium.vector import AsyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.typings import LayoutName


@validate_call
def create_vector_env(
    num_envs: int,
    layouts: list[LayoutName],
    info_level: int,
    horizon: int,
) -> VectorEnv:
    def make_env():
        def _thunk():
            return OvercookedGym(
                layouts=layouts,
                info_level=info_level,
                horizon=horizon,
                render_mode=None,
            )

        return _thunk

    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])

    return envs


def create_eval_env(layouts: list[LayoutName], info_level: int, horizon: int, video_folder: str) -> VectorEnv:
    env = OvercookedGym(
        layouts=layouts,
        info_level=info_level,
        horizon=horizon,
    )
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
    )
    return env
