import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium.vector.vector_env import VectorEnv

from src.agent import Agent


@struct.dataclass
class TrajectorySegment:
    obs: jax.Array  # (M, N, A, obs_dim)
    actions: jax.Array  # (M, N, A)
    log_probs: jax.Array  # (M, N, A)
    rewards: jax.Array  # (M, N, A)
    dones: jax.Array  # (M, N, A)
    values: jax.Array  # (M, N, A)
    last_value: jax.Array  # (N, A)
    last_done: jax.Array  # (N, A)

    @jax.jit
    def flatten(self):
        ref_m, ref_n = self.obs.shape[:2]

        def _maybe_flatten(x: jax.Array) -> jax.Array:
            if x.shape[0] != ref_m:
                return x

            return x.reshape((ref_m * ref_n, *x.shape[2:]))

        return jax.tree.map(_maybe_flatten, self)


@struct.dataclass
class Carry:
    obs: jax.Array  # (N, A, obs_dim)
    done: jax.Array  # (N, A)
    key: jax.Array  # RNG key


def collect_rollouts(envs: VectorEnv, agent: Agent, num_steps: int, carry: Carry) -> tuple[TrajectorySegment, Carry]:
    """
    Collect rollout segments from vectorized environments

    Args:
        envs: Vectorized environments
        agent: The agent used to interact with the environments
        num_steps: Number of steps to collect
        carry: Carry object containing the last observations and dones

    Returns:
        segment: TrajectorySegment containing collected data
        carry: Updated Carry object with the last observations and dones
    """
    next_obs = np.array(carry.obs)
    next_done = np.array(carry.done)
    key = carry.key

    n_envs = next_obs.shape[0]
    n_agents = next_obs.shape[1]
    obs_shape = next_obs.shape[2:]

    obs = np.zeros((num_steps, n_envs, n_agents, *obs_shape), dtype=next_obs.dtype)
    actions = np.zeros((num_steps, n_envs, n_agents), dtype=np.int32)
    log_probs = np.zeros((num_steps, n_envs, n_agents), dtype=np.float32)
    rewards = np.zeros((num_steps, n_envs, n_agents), dtype=np.float32)
    dones = np.zeros((num_steps, n_envs, n_agents), dtype=bool)
    values = np.zeros((num_steps, n_envs, n_agents), dtype=np.float32)

    for step in range(num_steps):
        obs[step] = next_obs

        key, subkey = jax.random.split(key)

        obs_jnp = jax.device_put(next_obs)
        out = agent.get_action_and_value(obs=obs_jnp, key=subkey)

        values[step] = np.array(out.value)
        log_probs[step] = np.array(out.action_log_prob)
        actions[step] = np.array(out.action)

        next_obs, reward, terminated, truncated, _ = envs.step(actions[step])

        # ! HACK: if reward is (n_envs,), make it (n_envs, n_agents)
        if reward.ndim == 1:
            reward = np.repeat(reward[:, np.newaxis], n_agents, axis=1)

        done = np.logical_or(terminated, truncated)

        # ! HACK: (n_envs) -> (n_envs, n_agents)
        done_broadcast = np.repeat(done[:, np.newaxis], n_agents, axis=1)

        rewards[step] = reward
        dones[step] = done_broadcast
        next_done = done_broadcast

    last_obs_jnp = jax.device_put(next_obs)
    last_value_jnp = agent.get_value(last_obs_jnp)

    segment = TrajectorySegment(
        obs=jnp.array(obs),
        actions=jnp.array(actions),
        log_probs=jnp.array(log_probs),
        rewards=jnp.array(rewards),
        dones=jnp.array(dones),
        values=jnp.array(values),
        last_value=last_value_jnp,
        last_done=jnp.array(next_done),
    )

    carry = Carry(
        obs=jnp.array(next_obs),
        done=jnp.array(next_done),
        key=key,
    )

    return segment, carry


@jax.jit
def compute_gae(segment: TrajectorySegment, gamma: float, lam: float) -> tuple[jax.Array, jax.Array]:
    def gae_step(
        carry: tuple[jax.Array, jax.Array],
        step_data: tuple[jax.Array, jax.Array, jax.Array],
    ):
        next_val, next_adv = carry
        reward, done, value = step_data

        non_terminal = 1.0 - done

        delta = reward + gamma * next_val * non_terminal - value
        adv = delta + gamma * lam * non_terminal * next_adv

        new_carry = (value, adv)
        return new_carry, adv

    # using lax.scan inside @jax.jit it avoids unrolling the loop
    # just like a for loop would do, but it's more efficient
    scan_inputs = (
        segment.rewards[::-1],
        segment.dones[::-1],
        segment.values[::-1],
    )
    init_carry = (segment.last_value, jnp.zeros_like(segment.last_value))

    _, advantages_rev = jax.lax.scan(gae_step, init_carry, scan_inputs)

    advantages = advantages_rev[::-1]
    returns = advantages + segment.values

    return advantages, returns
