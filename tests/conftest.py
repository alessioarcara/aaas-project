import pytest

from src.env_factory import create_vector_env
from tests.constants import NUM_ENVS


@pytest.fixture
def fake_policy():
    def policy(obs):
        batch_size = obs["agent_0_obs"].shape[0]
        action = [(0, 0)] * batch_size  # stay
        return action

    return policy


@pytest.fixture
def vector_env():
    envs = create_vector_env(
        num_envs=NUM_ENVS,
        layouts=["cramped_room"],
        info_level=1,
        horizon=400,
        video_dir="./videos",
        video_interval=200,
        should_record_video=False,
    )
    yield envs
    envs.close()
