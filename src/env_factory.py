from gymnasium.vector import AsyncVectorEnv
from pydantic.validate_call_decorator import validate_call

from src.overcooked_env import OvercookedGym
from src.utils.typings import LayoutName


@validate_call
def create_vector_env(
    num_envs: int,
    layouts: list[LayoutName] = ["cramped_room"],
    info_level: int = 1,
    horizon: int = 400,
):
    def make_env():
        env = OvercookedGym(layouts=layouts, info_level=info_level, horizon=horizon)
        return env

    return AsyncVectorEnv([make_env for _ in range(num_envs)])
