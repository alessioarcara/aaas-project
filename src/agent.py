from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from gymnasium.vector.vector_env import VectorEnv
from loguru import logger

from src.nets import CNN, MLP

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
    def __init__(
        self,
        cfg: "TrainingConfig",
        envs: VectorEnv,
        rngs: nnx.Rngs,
        total_steps: int,
    ):
        self.cfg = cfg
        obs_shape = envs.single_observation_space.shape
        act_shape = envs.single_action_space.n

        # din = int(jnp.prod(jnp.array(obs_shape)))
        # din = obs_shape[-1]

        self.backbone = CNN(obs_shape=obs_shape, num_filters=64, dout=512, rngs=rngs)
        self.critic = MLP(din=512, dhid=64, dout=1, out_scale=1.0, rngs=rngs)
        self.policy = MLP(din=512, dhid=64, dout=act_shape, out_scale=0.01, rngs=rngs)

        if cfg.use_learning_rate_annealing:
            self._lr_schedule = optax.linear_schedule(
                init_value=cfg.learning_rate, end_value=0.0, transition_steps=total_steps
            )
        else:
            self._lr_schedule = optax.constant_schedule(cfg.learning_rate)

        self.optim = nnx.optimizer.Optimizer(
            self,
            optax.chain(
                optax.clip_by_global_norm(0.5) if cfg.use_gradient_clipping else optax.identity(),
                optax.adam(learning_rate=self._lr_schedule, eps=cfg.adam_epsilon, b1=cfg.adam_momentum),
            ),
            wrt=nnx.Param,
        )

    def get_learning_rate(self, step: int) -> jax.Array:
        return self._lr_schedule(step)

    @nnx.jit
    def get_deterministic_action(self, obs: jax.Array) -> jax.Array:
        emb = self.backbone(obs)
        logits = self.policy(emb)
        return jnp.argmax(logits, axis=-1)

    @nnx.jit
    def get_value(self, obs: jax.Array) -> jax.Array:
        emb = self.backbone(obs)
        return self.critic(emb).squeeze(-1)

    @nnx.jit
    def get_action_and_value(
        self,
        obs: jax.Array,
        action: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None,
    ) -> AgentOutput:
        emb = self.backbone(obs)
        logits = self.policy(emb)
        value = self.critic(emb).squeeze(-1)

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
    def train_step(self, mb: Minibatch) -> dict[str, jax.Array]:
        def loss_fn(model: "Agent") -> tuple[jax.Array, dict[str, jax.Array]]:
            out = model.get_action_and_value(mb.obs, action=mb.actions)
            mb_advantages = mb.advantages

            logratio = out.action_log_prob - mb.log_probs
            ratio = jnp.exp(logratio)
            # approximate KL divergence to monitor the size of the policy update
            # if the policy changes too drastically we may want to stop the update early
            approx_kl = jnp.mean((ratio - 1) - logratio)
            # how many examples were clipped?
            clip_frac = jnp.mean(jnp.abs(ratio - 1) > self.cfg.ppo_epsilon)

            if self.cfg.use_advantage_normalization:
                mb_advantages = (mb_advantages - jnp.mean(mb_advantages)) / (jnp.std(mb_advantages) + 1e-8)

            # * Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.cfg.ppo_epsilon, 1 + self.cfg.ppo_epsilon)
            pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

            # * Value loss
            v_loss = jnp.mean((out.value - mb.returns) ** 2)

            # * Entropy
            entropy = out.entropy.mean()
            loss = pg_loss - (self.cfg.entropy_coef * entropy) + (self.cfg.value_coef * v_loss)

            return loss, {
                "policy_loss": pg_loss,
                "value_loss": v_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_frac": clip_frac,
            }

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(self)
        self.optim.update(grads)

        return aux

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
        batch_size = flat_segment.obs.shape[0]  # M * N

        b_obs = flat_segment.obs  # (M * N, A, obs_dim)
        b_advantages = advantages.reshape(batch_size, -1)  # (M * N,)
        b_returns = returns.reshape(batch_size, -1)  # (M * N,)
        b_values = flat_segment.values.reshape(batch_size, -1)  # (M * N,)

        minibatch_size = self.cfg.minibatch_size

        indices = jnp.arange(batch_size)
        b_metrics = None
        update_steps = 0  # to count number of update steps done

        # TODO: use jax.lax.scan here for better performance
        # for now, keep it simple and for loop let me to use early stopping
        for _ in range(self.cfg.update_epochs):
            key, subkey = jax.random.split(key)
            perm_indices = jax.random.permutation(subkey, indices)

            early_stop_triggered = False

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = perm_indices[start:end]

                mb = Minibatch(
                    obs=b_obs[mb_indices],
                    actions=flat_segment.actions[mb_indices],
                    log_probs=flat_segment.log_probs[mb_indices],
                    advantages=b_advantages[mb_indices],
                    returns=b_returns[mb_indices],
                )

                mb_metrics = self.train_step(mb)

                if self.cfg.target_kl is not None and mb_metrics["approx_kl"] > 1.5 * self.cfg.target_kl:
                    early_stop_triggered = True
                    break

                update_steps += 1
                if b_metrics is None:
                    b_metrics = mb_metrics
                else:
                    b_metrics = jax.tree_util.tree_map(jnp.add, b_metrics, mb_metrics)

            if early_stop_triggered:
                logger.warning("⚠️ Early stopping triggered due to large KL divergence.")
                break

        avg_metrics = jax.tree_util.tree_map(lambda x: x / update_steps, b_metrics)

        # * R2: Explained variance for value function
        # it tells us how well our value function is predicting the returns
        y_pred, y_true = b_values, b_returns
        var_y = jnp.var(y_true)
        explained_var = jnp.where(var_y == 0, jnp.nan, 1 - jnp.var(y_true - y_pred) / var_y)
        avg_metrics["explained_variance"] = explained_var

        return avg_metrics
