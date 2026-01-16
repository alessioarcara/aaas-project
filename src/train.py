import shutil

import jax
import jax.numpy as jnp
from flax import nnx
from loguru import logger
from tqdm import tqdm

import wandb
from src.agent import Agent
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae


def train(cfg: Config):
    video_dir = cfg.video_dir
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(cfg.seed)

    envs = cfg.envs

    agent = Agent(cfg.training_config, envs, nnx.Rngs(cfg.seed))

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        name=cfg.exp_name,
        config=cfg.model_dump(),
    )

    try:
        obs, _ = envs.reset()

        key, rollout_key = jax.random.split(key)

        carry = Carry(
            jnp.array(obs),
            jnp.zeros(envs.num_envs, dtype=bool),
            rollout_key,
        )

        total_timesteps = cfg.training_config.total_updates
        steps_per_update = cfg.training_config.num_steps * cfg.env_config.num_envs
        num_updates = total_timesteps // steps_per_update

        for update in tqdm(range(num_updates), desc="Training", unit="update", colour="blue"):
            segment, carry = collect_rollouts(
                envs,
                agent,
                cfg.training_config.num_steps,
                carry,
            )

            advantages, returns = compute_gae(
                segment,
                cfg.training_config.gae_lambda,
                cfg.training_config.gae_gamma,
            )

            key, learn_key = jax.random.split(key)

            _ = agent.learn_from(segment, advantages, returns, learn_key)

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
    except Exception as e:
        logger.exception("❌ Unhandled exception during training: {}", e)
        raise e

    finally:
        envs.close()
        wandb.finish()
