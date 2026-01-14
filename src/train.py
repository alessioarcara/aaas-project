import shutil
import jax.numpy as jnp
from loguru import logger

import wandb
from src.agent import Agent
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae

# from src.utils.misc import latest_video_path
import flax.nnx as nnx
import jax
from tqdm import tqdm


def train(cfg: Config):
    video_dir = cfg.video_dir
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(cfg.seed)

    envs = cfg.envs
    agent = Agent(envs, nnx.Rngs(cfg.seed))

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        name=cfg.exp_name,
    )

    try:
        obs, _ = envs.reset()

        key, subkey = jax.random.split(key)

        carry = Carry(
            jnp.array(obs),
            jnp.zeros(envs.num_envs, dtype=bool),
            subkey,
        )

        num_updates = cfg.training_config.total_updates // (
            cfg.training_config.num_steps * cfg.env_config.num_envs
        )

        for update in tqdm(
            range(num_updates), desc="Training", unit="update", colour="green"
        ):
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

            # video = latest_video_path(cfg.video_dir)

            # if video:
            #    wandb.log({"media/gameplay": wandb.Video(str(video), format="mp4")})

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

    finally:
        envs.close()
        wandb.finish()
