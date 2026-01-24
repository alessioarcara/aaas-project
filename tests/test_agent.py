from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src.agent_pair import AgentPair
from src.rollout import TrajectorySegment
from src.utils.checkpoint import load_model_weights, save_model_weights


@pytest.fixture
def mock_cfg():
    return SimpleNamespace(
        total_steps=1000,
        use_learning_rate_annealing=False,
        learning_rate=3e-4,
        use_gradient_clipping=False,
        adam_epsilon=1e-8,
        adam_momentum=0.9,
        minibatch_size=4,
        update_epochs=1,
        ppo_epsilon=0.2,
        use_advantage_normalization=True,
        entropy_coef=0.01,
        value_coef=0.5,
        target_kl=None,
        use_parameter_sharing=False,
        network=SimpleNamespace(
            type="cnn",
            embedding_dim=512,
            shared_backbone=False,
            num_filters=[32, 64, 64],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            paddings=["SAME", "SAME", "SAME"],
        ),
    )


@pytest.fixture
def mock_env(mocker):
    env = mocker.MagicMock()
    env.num_agents = 2
    env.single_observation_space.shape = (2, 5, 4, 26)
    env.single_action_space.n = 6
    return env


@pytest.fixture
def mock_rngs():
    return nnx.Rngs(0)


@pytest.fixture
def dummy_obs():
    return jnp.zeros((4, 2, 5, 4, 26), dtype=jnp.float32)  # (B=4, A=2, obs_dim)


@pytest.mark.parametrize("use_parameter_sharing", [True, False])
def test_agent_pair_deterministic_action(mock_cfg, mock_env, mock_rngs, dummy_obs, use_parameter_sharing):
    mock_cfg.use_parameter_sharing = use_parameter_sharing
    agent_pair = AgentPair(mock_cfg, mock_env, mock_rngs)
    actions = agent_pair.get_deterministic_action(dummy_obs)
    assert actions.shape == (4, 2)  # (B=4, A=2)


@pytest.mark.parametrize("use_parameter_sharing", [True, False])
def test_agent_pair_get_value(mock_cfg, mock_env, mock_rngs, dummy_obs, use_parameter_sharing):
    mock_cfg.use_parameter_sharing = use_parameter_sharing
    agent_pair = AgentPair(mock_cfg, mock_env, mock_rngs)
    values = agent_pair.get_value(dummy_obs)
    assert values.shape == (4, 2)  # (B=4, A=2)


@pytest.mark.parametrize("use_parameter_sharing", [True, False])
def test_agent_pair_get_action_and_value(mock_cfg, mock_env, mock_rngs, dummy_obs, use_parameter_sharing):
    mock_cfg.use_parameter_sharing = use_parameter_sharing
    agent_pair = AgentPair(mock_cfg, mock_env, mock_rngs)
    key = jax.random.key(0)
    out = agent_pair.get_action_and_value(dummy_obs, action=None, key=key)
    assert out.action.shape == (4, 2)  # (B=4, A=2)
    assert out.action_log_prob.shape == (4, 2)
    assert out.entropy.shape == (4, 2)
    assert out.value.shape == (4, 2)


@pytest.mark.parametrize("use_parameter_sharing", [True, False])
def test_agent_pair_learn_from(mock_cfg, mock_env, mock_rngs, use_parameter_sharing):
    mock_cfg.use_parameter_sharing = use_parameter_sharing
    agent_pair = AgentPair(mock_cfg, mock_env, mock_rngs)

    segment = TrajectorySegment(
        jnp.zeros((8, 4, 2, 5, 4, 26), dtype=jnp.float32),  # (M=8, N=4, A=2, obs_dim)
        jnp.zeros((8, 4, 2), dtype=jnp.int32),  # (M=8, N=4, A=2)
        jnp.zeros((8, 4, 2), dtype=jnp.float32),
        jnp.zeros((8, 4, 2), dtype=jnp.float32),
        jnp.zeros((8, 4, 2), dtype=bool),
        jnp.zeros((8, 4, 2), dtype=jnp.float32),
        jnp.zeros((4, 2), dtype=jnp.float32),
        jnp.zeros((4, 2), dtype=bool),
    )

    advantages = jnp.zeros((8, 4, 2), dtype=jnp.float32)
    returns = jnp.zeros((8, 4, 2), dtype=jnp.float32)
    key = jax.random.key(0)
    metrics = agent_pair.learn_from(segment, advantages, returns, key)
    assert isinstance(metrics, dict)


def test_orbax_save_load_nnx(tmp_path: Path, mock_cfg, mock_env):
    ckpt_dir = tmp_path / "ckpt"

    model = AgentPair(mock_cfg, mock_env, nnx.Rngs(0))
    _, state = nnx.split(model)

    save_model_weights(model, ckpt_dir / "state")

    state_restored = load_model_weights(lambda: AgentPair(mock_cfg, mock_env, nnx.Rngs(0)), ckpt_dir / "state")

    jax.tree.map(np.testing.assert_array_equal, state, state_restored)


def test_orbax_overwrite(tmp_path: Path, mock_cfg, mock_env):
    # Test that when I save the model weights to the same path twice,
    # it overwrites the previous weights.

    ckpt_path = tmp_path / "ckpt" / "state"

    model = AgentPair(mock_cfg, mock_env, nnx.Rngs(0))

    save_model_weights(model, ckpt_path)

    _, state_v1 = nnx.split(model)
    state_v1 = jax.tree.map(jnp.array, state_v1)

    _, state = nnx.split(model)
    state = jax.tree.map(lambda x: x + 1.0, state)
    nnx.update(model, state)

    save_model_weights(model, ckpt_path)

    state_restored = load_model_weights(lambda: AgentPair(mock_cfg, mock_env, nnx.Rngs(0)), ckpt_path)

    _, state_v2 = nnx.split(model)

    with pytest.raises(AssertionError):
        jax.tree.map(np.testing.assert_array_equal, state_v1, state_restored)

    jax.tree.map(np.testing.assert_array_equal, state_v2, state_restored)
