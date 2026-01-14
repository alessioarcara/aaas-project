from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from gymnasium.vector.vector_env import VectorEnv
from flax import struct


class MLP(nnx.Module):
    def __init__(
        self,
        din: int,
        dhid: int,
        dout: int,
        out_scale: float,
        rngs: nnx.Rngs,
    ):
        hidden_init = nnx.initializers.orthogonal(jnp.sqrt(2))
        bias_init = nnx.initializers.zeros
        out_init = nnx.initializers.orthogonal(out_scale)

        self.lin1 = nnx.Linear(
            din, dhid, rngs=rngs, kernel_init=hidden_init, bias_init=bias_init
        )
        self.lin2 = nnx.Linear(
            dhid, dhid, rngs=rngs, kernel_init=hidden_init, bias_init=bias_init
        )
        self.lin3 = nnx.Linear(
            dhid, dout, rngs=rngs, kernel_init=out_init, bias_init=bias_init
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.lin1(x)
        x = nnx.tanh(x)

        x = self.lin2(x)
        x = nnx.tanh(x)

        x = self.lin3(x)
        return x


@struct.dataclass
class AgentOutput:
    action: jax.Array
    action_log_prob: jax.Array
    entropy: jax.Array
    value: jax.Array


class Agent(nnx.Module):
    def __init__(self, envs: VectorEnv, rngs: nnx.Rngs):
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.n

        # din = int(jnp.prod(jnp.array(obs_shape)))
        din = obs_shape[-1]

        self.critic = MLP(
            din=din,
            dhid=64,
            dout=1,
            out_scale=1.0,
            rngs=rngs,
        )
        self.policy = MLP(
            din=din,
            dhid=64,
            dout=act_shape,
            out_scale=0.01,
            rngs=rngs,
        )

    def get_value(self, obs_jnp: jax.Array) -> jax.Array:
        return self.critic(obs_jnp).squeeze(-1)

    def get_action_and_value(
        self,
        obs_jnp: jax.Array,
        key: Optional[jax.Array] = None,
        action: Optional[jax.Array] = None,
    ) -> AgentOutput:
        logits = self.policy(obs_jnp)
        value = self.critic(obs_jnp).squeeze(-1)

        if action is None:
            if key is None:
                raise ValueError("key must be provided if action is None")
            action = jax.random.categorical(key, logits)

        log_probs_all = nnx.log_softmax(logits)
        action_log_prob = jnp.take_along_axis(
            log_probs_all, action[..., None], axis=-1
        ).squeeze(-1)

        # # H(x) = - sum(p(x) * log(p(x)))
        probs = nnx.softmax(logits)
        entropy = -jnp.sum(probs * log_probs_all, axis=-1)

        return AgentOutput(
            action=action,
            action_log_prob=action_log_prob,
            entropy=entropy,
            value=value,
        )
