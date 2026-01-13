import jax.numpy as jnp

from src.agent import Agent
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae
from loguru import logger


def train(cfg: Config):
    envs = cfg.envs
    agent = Agent()

    obs, _ = envs.reset()

    carry = Carry(obs, jnp.zeros(envs.num_envs, dtype=bool))
    segment, carry = collect_rollouts(envs, agent, cfg.training_config.num_steps, carry)
    advantages, returns = compute_gae(
        segment, cfg.training_config.gae_lambda, cfg.training_config.gae_gamma
    )

    logger.info("Finish")
