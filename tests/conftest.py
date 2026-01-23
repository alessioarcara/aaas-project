import pytest

from src.env_factory import create_train_envs
from src.utils.typings import EncodingType, ShapingMode
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
    envs = create_train_envs(
        num_envs=NUM_ENVS,
        layouts=["cramped_room"],
        encoding=EncodingType.FEATURIZED,
        info_level=1,
        horizon=400,
        shaping_mode=ShapingMode.NONE,
    )
    yield envs
    envs.close()
