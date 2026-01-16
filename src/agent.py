from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from gymnasium.vector.vector_env import VectorEnv

from src.nets import MLP

if TYPE_CHECKING:
    from src.config import TrainingConfig
    from src.rollout import TrajectorySegment


@struct.dataclass
class AgentOutput:
    action: jax.Array
    action_log_prob: jax.Array
    entropy: jax.Array
    value: jax.Array


@struct.dataclass
class Minibatch:
    obs: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array


class Agent(nnx.Module):
    def __init__(self, cfg: "TrainingConfig", envs: VectorEnv, rngs: nnx.Rngs):
        self.cfg = cfg
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.n

        # din = int(jnp.prod(jnp.array(obs_shape)))
        din = obs_shape[-1]

        self.critic = MLP(din=din, dhid=64, dout=1, out_scale=1.0, rngs=rngs)
        self.policy = MLP(din=din, dhid=64, dout=act_shape, out_scale=0.01, rngs=rngs)

        self.optim = nnx.optimizer.Optimizer(self, optax.adam(cfg.learning_rate), wrt=nnx.Param)

    @nnx.jit
    def get_deterministic_action(self, obs: jax.Array) -> jax.Array:
        logits = self.policy(obs)
        return jnp.argmax(logits, axis=-1)

    @nnx.jit
    def get_value(self, obs: jax.Array) -> jax.Array:
        return self.critic(obs).squeeze(-1)

    @nnx.jit
    def get_action_and_value(
        self,
        obs: jax.Array,
        action: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None,
    ) -> AgentOutput:
        logits = self.policy(obs)
        value = self.critic(obs).squeeze(-1)

        if action is None:
            if key is None:
                raise ValueError("key must be provided if action is None")
            action = jax.random.categorical(key, logits)

        log_probs_all = nnx.log_softmax(logits)
        action_log_prob = jnp.take_along_axis(log_probs_all, action[..., None], axis=-1).squeeze(-1)

        # # H(x) = - sum(p(x) * log(p(x)))
        probs = nnx.softmax(logits)
        entropy = -jnp.sum(probs * log_probs_all, axis=-1)

        return AgentOutput(
            action=action,
            action_log_prob=action_log_prob,
            entropy=entropy,
            value=value,
        )

    @nnx.jit
    def train_step(self, mb: Minibatch):
        def loss_fn(model: "Agent"):
            out = model.get_action_and_value(mb.obs, action=mb.actions)

            # * Policy loss
            logratio = out.action_log_prob - mb.log_probs
            ratio = jnp.exp(logratio)

            pg_loss1 = -mb.advantages * ratio
            pg_loss2 = -mb.advantages * jnp.clip(ratio, 1 - self.cfg.epsilon, 1 + self.cfg.epsilon)
            pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

            # * Value loss
            v_loss = jnp.mean((out.value - mb.returns) ** 2)

            loss = pg_loss + v_loss

            return loss

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(self)
        self.optim.update(grads)

    def learn_from(
        self,
        segment: "TrajectorySegment",
        advantages: jax.Array,
        returns: jax.Array,
        key: jax.Array,
    ):
        """
        TODO: add docstring
        """
        flat_segment = segment.flatten()
        b_obs = flat_segment.obs  # (M * N, A, obs_dim)
        b_advantages = advantages.reshape(b_obs.shape[0], -1)  # (M * N,)
        b_returns = returns.reshape(b_obs.shape[0], -1)  # (M * N,)

        dataset_size = b_obs.shape[0]
        minibatch_size = self.cfg.minibatch_size

        indices = jnp.arange(dataset_size)

        for _ in range(self.cfg.update_epochs):
            key, subkey = jax.random.split(key)
            perm_indices = jax.random.permutation(subkey, indices)

            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = perm_indices[start:end]

                mb = Minibatch(
                    obs=b_obs[mb_indices],
                    actions=flat_segment.actions[mb_indices],
                    log_probs=flat_segment.log_probs[mb_indices],
                    advantages=b_advantages[mb_indices],
                    returns=b_returns[mb_indices],
                )

                self.train_step(mb)
