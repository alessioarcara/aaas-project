import jax
import jax.numpy as jnp


class Agent:
    def __init__(self):
        pass

    def get_action_and_value(self, obs_jnp: jax.Array) -> tuple[jax.Array, jax.Array]:
        # dummy implementation
        batch_size = obs_jnp.shape[0]
        actions = jnp.zeros((batch_size, 2), dtype=jnp.int32)
        values = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        return actions, values
