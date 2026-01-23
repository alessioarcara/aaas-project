from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from gymnasium.vector.vector_env import VectorEnv

from src.agent import PPOAgent
from src.base_agent import Agent, AgentOutput

if TYPE_CHECKING:
    from src.config import TrainingConfig
    from src.rollout import TrajectorySegment


class AgentPair(nnx.Module, Agent):
    """
    Multi-agent wrapper class.
    - share_parameters=True -> 1 agent for both players
    - share_parameters=False -> 2 agents, one for each player
    """

    def __init__(
        self,
        cfg: "TrainingConfig",
        envs: VectorEnv,
        rngs: nnx.Rngs,
    ):
        self.share_parameters = cfg.use_parameter_sharing
        self.num_agents = envs.num_agents

        if self.share_parameters:
            self.agents = [PPOAgent(cfg, envs, rngs)]
        else:
            self.agents = [
                PPOAgent(cfg, envs, rngs),
                PPOAgent(cfg, envs, rngs),
            ]

    # ! Assuming both agents have the same learning rate schedule
    def get_learning_rate(self, step: int) -> jax.Array:
        return self.agents[0].get_learning_rate(step)

    @nnx.jit
    def get_deterministic_action(
        self,
        obs: jax.Array,  # (B, A, obs_dim)
    ) -> jax.Array:  # (B, A)
        B, A = obs.shape[:2]

        if self.share_parameters:
            flat_obs = obs.reshape((B * A, *obs.shape[2:]))
            flat_action = self.agents[0].get_deterministic_action(flat_obs)
            return flat_action.reshape((B, A))
        else:
            actions = [self.agents[i].get_deterministic_action(obs[:, i]) for i in range(len(self.agents))]
            return jnp.stack(actions, axis=1)

    @nnx.jit
    def get_value(
        self,
        obs: jax.Array,  # (B, A, obs_dim)
    ) -> jax.Array:  # (B, A)
        B, A = obs.shape[:2]

        if self.share_parameters:
            flat_obs = obs.reshape((B * A, *obs.shape[2:]))
            flat_value = self.agents[0].get_value(flat_obs)
            return flat_value.reshape((B, A))
        else:
            values = [self.agents[i].get_value(obs[:, i]) for i in range(len(self.agents))]
            return jnp.stack(values, axis=1)

    @nnx.jit
    def get_action_and_value(
        self,
        obs: jax.Array,  # (B, A, obs_dim)
        action: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None,
    ) -> AgentOutput:
        B, A = obs.shape[:2]

        if self.share_parameters:
            flat_obs = obs.reshape((B * A, *obs.shape[2:]))
            flat_action = action.reshape((B * A,)) if action is not None else None
            out = self.agents[0].get_action_and_value(flat_obs, flat_action, key)

            return AgentOutput(
                action=out.action.reshape((B, A)),
                action_log_prob=out.action_log_prob.reshape((B, A)),
                entropy=out.entropy.reshape((B, A)),
                value=out.value.reshape((B, A)),
            )
        else:
            keys = jax.random.split(key, num=len(self.agents)) if key is not None else [None] * len(self.agents)

            outputs = []
            for i in range(len(self.agents)):
                agent_obs = obs[:, i]  # (B, ...)
                agent_action = action[:, i] if action is not None else None
                out = self.agents[i].get_action_and_value(agent_obs, agent_action, keys[i])
                outputs.append(out)

            return AgentOutput(
                action=jnp.stack([out.action for out in outputs], axis=1),
                action_log_prob=jnp.stack([out.action_log_prob for out in outputs], axis=1),
                entropy=jnp.stack([out.entropy for out in outputs], axis=1),
                value=jnp.stack([out.value for out in outputs], axis=1),
            )

    def learn_from(
        self,
        segment: "TrajectorySegment",
        advantages: jax.Array,  # (M, N, A)
        returns: jax.Array,  # (M, N, A)
        key: jax.Array,
    ) -> dict:
        if self.share_parameters:
            # ! if the parameters are shared
            # ! and even if PPO is on-policy,
            # ! we don't need to split the segment for each agent
            # ! because both agents are the same and they can learn from
            # ! rollouts of each other
            metrics = self.agents[0].learn_from(segment, advantages, returns, key)
            return {f"agent_0/{k}": v for k, v in metrics.items()}
        else:
            agents_metrics = {}
            keys = jax.random.split(key, num=len(self.agents))

            for i in range(len(self.agents)):
                agent_segment = segment.select_agent(i)

                metrics = self.agents[i].learn_from(
                    agent_segment,
                    advantages[:, :, i],
                    returns[:, :, i],
                    keys[i],
                )

                for k, v in metrics.items():
                    agents_metrics[f"agent_{i}/{k}"] = v

            return agents_metrics
