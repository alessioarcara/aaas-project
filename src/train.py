import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium.vector.vector_env import VectorEnv

from src.agent import Agent


@struct.dataclass
class TrajectorySegment:
    obs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray


@struct.dataclass
class Carry:
    obs: jnp.ndarray
    done: jnp.ndarray


# we are not collecting entire episode rollouts here, just segments of fixed length
# if we reset envs it's like cutting episodes and we don't want that,
# thus we do not reset envs
def collect_rollouts(
    envs: VectorEnv, agent: Agent, num_steps: int, carry: Carry
) -> tuple[TrajectorySegment, Carry]:
    next_obs = np.array(carry.obs)
    next_done = np.array(carry.done)

    n_envs = next_obs.shape[0]
    obs_shape = next_obs.shape[1:]

    obs = np.zeros((num_steps, n_envs, *obs_shape), dtype=next_obs.dtype)
    rewards = np.zeros((num_steps, n_envs), dtype=np.float32)
    dones = np.zeros((num_steps, n_envs), dtype=bool)
    values = np.zeros((num_steps, n_envs), dtype=np.float32)

    for step in range(num_steps):
        obs[step] = next_obs

        obs_jnp = jax.device_put(next_obs)

        action_jnp, value_jnp = agent.get_action_and_value(obs_jnp)

        values[step] = np.array(value_jnp)

        action = np.array(action_jnp)

        next_obs, reward, terminated, truncated, _ = envs.step(action)

        done = np.logical_or(terminated, truncated)

        rewards[step] = reward
        dones[step] = done

        next_done = done

    segment = TrajectorySegment(
        obs=jnp.array(obs),
        rewards=jnp.array(rewards),
        dones=jnp.array(dones),
        values=jnp.array(values),
    )

    carry = Carry(
        obs=jnp.array(next_obs),
        done=jnp.array(next_done),
    )

    return segment, carry
