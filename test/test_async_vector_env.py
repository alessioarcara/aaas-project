from typing import Final

import numpy as np
import pytest
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from src.utils import make_env

NUM_ENVS: Final[int] = 4


@pytest.fixture
def vector_env():
    env_fns = [make_env(["cramped_room"]) for _ in range(NUM_ENVS)]
    envs = AsyncVectorEnv(env_fns)
    yield envs
    envs.close()


def fake_policy(obs):
    batch_size = obs["agent_0_obs"].shape[0]
    actions = [(0, 0)] * batch_size  # stay
    return actions


def test_async_vector_env(vector_env):
    obs, info = vector_env.reset()

    assert obs["agent_0_obs"].shape[0] == NUM_ENVS
    assert obs["agent_1_obs"].shape[0] == NUM_ENVS

    actions = fake_policy(obs)

    obs, rewards, terminated, truncated, info = vector_env.step(actions)

    assert obs["agent_0_obs"].shape[0] == NUM_ENVS
    assert obs["agent_1_obs"].shape[0] == NUM_ENVS
    assert rewards.shape[0] == NUM_ENVS
    assert terminated.shape[0] == NUM_ENVS
    assert truncated.shape[0] == NUM_ENVS


@pytest.mark.parametrize(
    "steps, expect_done",
    [
        (400, True),  # Horizon limit -> all envs done
        (
            410,
            False,
        ),  # 10 steps into new episode due to auto-reset -> no envs done
    ],
)
def test_full_episode(vector_env, steps, expect_done):
    next_obs, info = vector_env.reset()

    max_steps = steps

    for _ in range(max_steps):
        obs = next_obs

        action = fake_policy(obs)

        next_obs, rewards, terminated, truncated, info = vector_env.step(action)

        has_terminated_or_truncated = np.logical_or(terminated, truncated)

    if expect_done:
        assert np.all(has_terminated_or_truncated), (
            f"At step {steps}, all environments should be done (horizon reached)."
        )
    else:
        assert not np.any(has_terminated_or_truncated), (
            f"At step {steps}, environments should be running (inside a new episode)."
        )
