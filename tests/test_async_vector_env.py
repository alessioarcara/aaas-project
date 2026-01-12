import numpy as np
import pytest

from tests.constants import NUM_ENVS


def test_async_vector_env(vector_env, fake_policy):
    obs, info = vector_env.reset()

    assert obs["agent_0_obs"].shape[0] == NUM_ENVS
    assert obs["agent_1_obs"].shape[0] == NUM_ENVS

    action = fake_policy(obs)
    print(action)

    obs, rewards, terminated, truncated, info = vector_env.step(action)

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
def test_full_episode(vector_env, fake_policy, steps, expect_done):
    next_obs, info = vector_env.reset()

    max_steps = steps

    for _ in range(max_steps):
        action = fake_policy(next_obs)

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
