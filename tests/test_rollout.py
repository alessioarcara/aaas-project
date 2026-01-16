import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.agent import AgentOutput
from src.rollout import Carry, TrajectorySegment, collect_rollouts, compute_gae
from tests.constants import NUM_ENVS


@pytest.fixture
def mock_rollout(mocker):
    """
    Factory fixture that creates mocked envs and agent for rollout collection tests.
    """

    def _create(env_outcomes, agent_values, num_agents=1, obs_dim=3):
        envs = mocker.Mock()
        env_iter = iter(env_outcomes)

        def env_step(action):
            obs, rew, term, trunc = next(env_iter)
            return (
                np.full((NUM_ENVS, num_agents, obs_dim), obs, dtype=np.float32),
                np.full((NUM_ENVS, num_agents), rew, dtype=np.float32),
                np.full((NUM_ENVS,), term, dtype=bool),
                np.full((NUM_ENVS,), trunc, dtype=bool),
                {},
            )

        envs.step.side_effect = env_step

        agent = mocker.Mock()
        agent_iter = iter(agent_values)

        def agent_act(obs, key=None):
            val = next(agent_iter)
            return AgentOutput(
                action=jnp.zeros((NUM_ENVS, num_agents), dtype=jnp.int32),
                action_log_prob=jnp.zeros((NUM_ENVS, num_agents), dtype=jnp.float32),
                entropy=jnp.zeros((NUM_ENVS, num_agents), dtype=jnp.float32),
                value=jnp.full((NUM_ENVS, num_agents), val, dtype=jnp.float32),
            )

        agent.get_action_and_value.side_effect = agent_act

        return envs, agent

    return _create


def test_collect_rollout(mock_rollout):
    OBS_DIM = 3
    NUM_STEPS = 4
    NUM_AGENTS = 2

    # Format: (obs, reward, terminated, truncated)
    env_outcomes = [
        (1.0, 1.0, False, False),  # Step 1: Normal step
        (99.0, 1.0, False, True),  # Step 2: RESET!
        (2.0, 1.0, False, False),  # Step 3: Normal step
        (3.0, 1.0, False, False),  # Step 4: Normal step
    ]

    agent_values = [10.0, 20.0, 30.0, 40.0, 50.0]

    envs, agent = mock_rollout(env_outcomes, agent_values, num_agents=NUM_AGENTS, obs_dim=3)

    # Step 0: initial observation
    init_obs = np.zeros((NUM_ENVS, NUM_AGENTS, OBS_DIM), dtype=np.float32)
    init_done = np.zeros((NUM_ENVS, NUM_AGENTS), dtype=bool)

    key = jax.random.key(0)

    segment, carry = collect_rollouts(
        envs,
        agent,
        NUM_STEPS,
        Carry(obs=init_obs, done=init_done, key=key),
    )

    # assert shapes
    # num_steps because we collect t0 ... tN-1 observations
    assert segment.obs.shape == (NUM_STEPS, NUM_ENVS, NUM_AGENTS, OBS_DIM)
    assert segment.values.shape == (NUM_STEPS, NUM_ENVS, NUM_AGENTS)
    assert segment.dones.shape == (NUM_STEPS, NUM_ENVS, NUM_AGENTS)
    assert segment.rewards.shape == (NUM_STEPS, NUM_ENVS, NUM_AGENTS)

    assert np.all(segment.obs[0] == 0.0)  # init_obs
    assert np.all(segment.obs[1] == 1.0)  # obs after step 0
    assert np.all(segment.obs[2] == 99.0)  # obs after step 1 (reset)

    assert np.all(segment.dones[1])  # it should be done (reset)
    assert not np.any(segment.dones[0])  # it should not be done

    # t4 as last obs
    assert np.all(carry.obs == 3.0)
    assert not np.any(carry.done)


def test_compute_gae():
    agent_values = jnp.array([10.0, 20.0, 30.0, 40.0])

    segment = TrajectorySegment(
        obs=jnp.zeros((4, NUM_ENVS, 3)),  # Dummy obs
        rewards=jnp.full((4, NUM_ENVS), 1.0),  # Rewards = 1.0 at each step
        dones=jnp.array(
            [
                [False] * NUM_ENVS,
                [True] * NUM_ENVS,  # Reset after step 1
                [False] * NUM_ENVS,
                [False] * NUM_ENVS,
            ]
        ),
        values=jnp.tile(agent_values[:, None], (1, NUM_ENVS)),
        last_value=jnp.zeros(NUM_ENVS),  # Dummy last value
        last_done=jnp.zeros(NUM_ENVS, dtype=bool),  # Dummy last done
    )

    # Formulas:
    # A_t = delta_t + gamma * lambda * A_t+1
    # where delta_t = r_t + gamma * V(s_t+1) - V(s_t)
    #
    # Step 3:
    # R3 = 1.0
    # V3 = 40.0
    # V4 = 0.0
    # delta3 = 1.0 + 0.9 * 0.0 - 40.0 = -39.0
    # A3 = delta3 = -39.0
    #
    # Step 2:
    # R2 = 1.0
    # V2 = 30.0
    # V3 = 40.0
    # delta2 = 1.0 + 0.9 * 40.0 - 30.0 = 10.6 = 7.0
    # A2 = 7.0 + + 0.9 * 0.95 * -39.0 = -26.345
    #
    # Step 1 (RESET):
    # R1 = 1.0
    # V1 = 20.0
    # V2 = 0.0
    # delta1 = 1.0 + 0.9 * 0.0 - 20.0 = -19.0
    # A2 = -19.0
    #
    # Step 0:
    # R0 = 1.0
    # V0 = 10.0
    # V1 = 20.0
    # delta0 = 1.0 + 0.9 * 20.0 - 10.0 = 9.0
    # A0 = 9.0 + 0.9 * 0.95 * -19.0 = -7.245
    expected_advantages = np.tile([-7.245, -19.0, -26.345, -39.0], (NUM_ENVS, 1)).T

    # Returns = Advantages + Values
    expected_returns = expected_advantages + np.array(agent_values)[:, None]

    advantages, returns = compute_gae(segment, 0.9, 0.95)

    np.testing.assert_allclose(advantages, expected_advantages, atol=1e-5, err_msg="Wrong advantages computed")

    np.testing.assert_allclose(returns, expected_returns, atol=1e-5, err_msg="Wrong returns computed")


def test_flatten_trajectory_segment():
    segment = TrajectorySegment(
        obs=jnp.zeros((400, 15, 10, 10, 3)),
        rewards=jnp.zeros((400, 15)),
        dones=jnp.zeros((400, 15), dtype=bool),
        values=jnp.zeros((400, 15)),
        last_value=jnp.zeros((15,)),
        last_done=jnp.zeros((15,), dtype=bool),
    )

    flat_segment = segment.flatten()

    assert flat_segment.obs.shape == (400 * 15, 10, 10, 3)
    assert flat_segment.rewards.shape == (400 * 15,)
    assert flat_segment.dones.shape == (400 * 15,)
    assert flat_segment.values.shape == (400 * 15,)
    assert flat_segment.last_value.shape == (15,)
    assert flat_segment.last_done.shape == (15,)
