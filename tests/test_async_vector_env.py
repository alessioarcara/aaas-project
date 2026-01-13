import numpy as np
import pytest

from tests.constants import NUM_ENVS


def test_async_vector_env(vector_env):
    obs, info = vector_env.reset()

    assert obs.shape[0] == NUM_ENVS
    assert obs.shape[0] == NUM_ENVS

    obs, rewards, terminated, truncated, info = vector_env.step(
        np.zeros((NUM_ENVS, 2), dtype=np.int32)
    )

    assert obs.shape[0] == NUM_ENVS
    assert obs.shape[0] == NUM_ENVS
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
        next_obs, rewards, terminated, truncated, info = vector_env.step(
            np.zeros((NUM_ENVS, 2), dtype=np.int32)
        )

        has_terminated_or_truncated = np.logical_or(terminated, truncated)

    if expect_done:
        assert np.all(has_terminated_or_truncated), (
            f"At step {steps}, all environments should be done (horizon reached)."
        )
    else:
        assert not np.any(has_terminated_or_truncated), (
            f"At step {steps}, environments should be running (inside a new episode)."
        )
