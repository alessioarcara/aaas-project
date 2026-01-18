import shutil

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from loguru import logger
from tqdm import tqdm

import wandb
from src.agent import Agent
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae
from src.utils.constants import STATS_KEY
from src.utils.misc import latest_video_path


def eval_agent(agent: Agent, env: gym.Env) -> float:
    """
    Using the deterministic policy, evaluate the agent in the eval_env
    """
    obs, _ = env.reset()

    done = False
    while not done:
        obs_jnp = jax.device_put(obs)
        action = agent.get_deterministic_action(obs_jnp)
        action = np.array(action)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return info.get(STATS_KEY, {}).get("r", 0.0)


def train(cfg: Config):
    """
    1. Collect rollouts
    2. Compute GAE advantages and returns
    3. Update agent using collected data
    """
    video_dir = cfg.video_dir
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(cfg.seed)

    train_envs = cfg.train_envs
    eval_envs = cfg.eval_envs

    total_updates = cfg.training_config.total_updates
    batch_size = cfg.training_config.num_steps * cfg.env_config.num_envs
    num_updates = total_updates // batch_size

    agent = Agent(
        cfg=cfg.training_config,
        envs=train_envs,
        rngs=nnx.Rngs(cfg.seed),
        total_steps=total_updates,
    )

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        name=cfg.exp_name,
        config=cfg.model_dump(),
    )

    try:
        obs, _ = train_envs.reset()
        key, rollout_key = jax.random.split(key)

        carry = Carry(
            jnp.array(obs),
            jnp.zeros(train_envs.num_envs, dtype=bool),
            rollout_key,
        )

        for update in tqdm(range(1, num_updates + 1), desc="Training", unit="update", colour="blue"):
            segment, carry = collect_rollouts(
                train_envs,
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
            metrics = agent.learn_from(segment, advantages, returns, learn_key)

            global_step = update * batch_size

            log_data = {f"train/{k}": v.item() for k, v in metrics.items()}
            log_data["train/lr"] = agent.get_learning_rate(global_step).item()

            if cfg.eval_interval > 0 and update % cfg.eval_interval == 0:
                for layout, env in eval_envs.items():
                    log_data[f"eval/{layout}_return"] = eval_agent(agent, env)

                    if vid_path := latest_video_path(cfg.video_dir / layout):
                        log_data[f"eval/{layout}_video"] = wandb.Video(str(vid_path), format="mp4")

            wandb.log(log_data, step=global_step)

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
    except Exception as e:
        logger.exception("❌ Unhandled exception during training: {}", e)
        raise e

    finally:
        train_envs.close()
        for env in eval_envs.values():
            env.close()
        wandb.finish()
