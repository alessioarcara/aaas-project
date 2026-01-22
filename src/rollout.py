import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnasium.vector.vector_env import VectorEnv

from src.agent import Agent


@struct.dataclass
class TrajectorySegment:
    """
    Data structure holding a trajectory segment collected from vectorized environments.

    M: number of steps
    N: number of environments
    A: number of agents
    """

    obs: jax.Array  # Shared: (M, N, A, ...), Independent: (M, N, ...)
    actions: jax.Array  # Shared: (M, N, A),      Independent: (M, N)
    log_probs: jax.Array  # Shared: (M, N, A),      Independent: (M, N)
    rewards: jax.Array  # Shared: (M, N, A),      Independent: (M, N)
    dones: jax.Array  # Shared: (M, N, A),      Independent: (M, N)
    values: jax.Array  # Shared: (M, N, A),      Independent: (M, N)
    last_value: jax.Array  # Shared: (N, A),         Independent: (N,)
    last_done: jax.Array  # Shared: (N, A),         Independent: (N,)

    def select_agent(self, agent_idx: int) -> "TrajectorySegment":
        """
        Extracts a slice of data for a specific agent.

        A dimension is removed in the returned TrajectorySegment.
        (M, N, A, ...) -> (M, N, ...)
        (N, A)      -> (N,)
        """
        ref_shape = self.actions.shape
        ref_ndim = len(ref_shape)

        def _slice(x: jax.Array) -> jax.Array:
            if x.ndim >= ref_ndim and x.shape[:ref_ndim] == ref_shape:
                return x[:, :, agent_idx]

            last_shape = ref_shape[1:]
            last_ndim = ref_ndim - 1

            if x.ndim >= last_ndim and x.shape[:last_ndim] == last_shape:
                return x[:, agent_idx]

            return x

        return jax.tree.map(_slice, self)

    @jax.jit
    def flatten(self):
        """
        Flattens the first two dimensions (M, N) -> (M * N).
        """
        # We use actions as reference
        batch_shape = self.actions.shape
        batch_rank = len(batch_shape)

        def _maybe_flatten(x: jax.Array) -> jax.Array:
            if x.ndim >= batch_rank and x.shape[:batch_rank] == batch_shape:
                return x.reshape((-1, *x.shape[batch_rank:]))

            return x

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

        next_obs, reward, terminated, truncated, info = envs.step(actions[step])

        done = np.logical_or(terminated, truncated)
        done_broadcast = done[:, np.newaxis]  # (N, 1) to (N, A)

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
    """ """

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
        segment.rewards,
        segment.dones,
        segment.values,
    )
    init_carry = (segment.last_value, jnp.zeros_like(segment.last_value))

    _, advantages = jax.lax.scan(gae_step, init_carry, scan_inputs, reverse=True)

    returns = advantages + segment.values

    return advantages, returns
