import jax.numpy as jnp
import numpy as np
import pytest

from src.train import Carry, collect_rollouts
from tests.constants import NUM_ENVS


@pytest.fixture
def mock_rollout(mocker):
    """
    Factory fixture that creates mocked envs and agent for rollout collection tests.
    """

    def _create(env_outcomes, agent_values, obs_dim=3):
        envs = mocker.Mock()
        env_iter = iter(env_outcomes)

        def env_step(action):
            obs, rew, term, trunc = next(env_iter)
            return (
                np.full((NUM_ENVS, obs_dim), obs, dtype=np.float32),
                np.full(NUM_ENVS, rew, dtype=np.float32),
                np.full(NUM_ENVS, term, dtype=bool),
                np.full(NUM_ENVS, trunc, dtype=bool),
                {},
            )

        envs.step.side_effect = env_step

        agent = mocker.Mock()
        agent_iter = iter(agent_values)

        def agent_act(obs, key=None):
            val = next(agent_iter)
            return (
                jnp.zeros(NUM_ENVS, dtype=jnp.int32),
                jnp.full(NUM_ENVS, val, dtype=jnp.float32),
            )

        agent.get_action_and_value.side_effect = agent_act

        return envs, agent

    return _create


def test_collect_rollout(mock_rollout):
    OBS_DIM = 3
    NUM_STEPS = 4

    env_outcomes = [
        (1.0, 1.0, False, False),  # Step 1: Normal step
        (99.0, 1.0, False, True),  # Step 2: RESET!
        (2.0, 1.0, False, False),  # Step 3: Normal step
        (3.0, 1.0, False, False),  # Step 4: Normal step
    ]

    agent_values = [10.0, 20.0, 30.0, 40.0]

    envs, agent = mock_rollout(env_outcomes, agent_values, obs_dim=3)

    # Step 0: initial observation
    init_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
    init_done = np.zeros(NUM_ENVS, dtype=bool)

    segment, carry = collect_rollouts(
        envs,
        agent,
        NUM_STEPS,
        Carry(
            obs=init_obs,
            done=init_done,
        ),
    )

    # assert shapes
    # num_steps because we collect t0 ... tN-1 observations
    assert segment.obs.shape == (NUM_STEPS, NUM_ENVS, OBS_DIM)
    assert segment.values.shape == (NUM_STEPS, NUM_ENVS)
    assert segment.dones.shape == (NUM_STEPS, NUM_ENVS)
    assert segment.rewards.shape == (NUM_STEPS, NUM_ENVS)

    assert np.all(segment.obs[0] == 0.0)  # init_obs
    assert np.all(segment.obs[1] == 1.0)  # obs after step 0
    assert np.all(segment.obs[2] == 99.0)  # obs after step 1 (reset)

    assert np.all(segment.dones[1])  # it should be done (reset)
    assert not np.any(segment.dones[0])  # it should not be done

    # t4 as last obs
    assert np.all(carry.obs == 3.0)
    assert not np.any(carry.done)
