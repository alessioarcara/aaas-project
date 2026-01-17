import shutil

import gymnasium
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
from src.utils.misc import latest_video_path


def eval_agent(agent: Agent, eval_env: gymnasium.Env):
    """
    Using the deterministic policy, evaluate the agent in the eval_env
    """
    obs, _ = eval_env.reset()

    done = False
    while not done:
        obs_jnp = jax.device_put(obs)
        action = agent.get_deterministic_action(obs_jnp)
        action = np.array(action)

        obs, _, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    return info.get("episode", {}).get("r", 0.0)


def train(cfg: Config):
    video_dir = cfg.video_dir
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(cfg.seed)

    envs = cfg.envs
    eval_env = cfg.eval_env

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

        total_updates = cfg.training_config.total_updates
        steps_per_update = cfg.training_config.num_steps * cfg.env_config.num_envs
        num_updates = total_updates // steps_per_update

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

            metrics = agent.learn_from(segment, advantages, returns, learn_key)

            log_data = {f"train/{k}": v.item() for k, v in metrics.items()}

            if cfg.eval_interval > 0 and (update + 1) % cfg.eval_interval == 0:
                log_data["eval/episode_reward"] = eval_agent(agent, eval_env)

                if vid_path := latest_video_path(cfg.video_dir):
                    log_data["eval/video"] = wandb.Video(str(vid_path), format="mp4")

            wandb.log(log_data, step=(update + 1) * steps_per_update)

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
    except Exception as e:
        logger.exception("❌ Unhandled exception during training: {}", e)
        raise e

    finally:
        envs.close()
        eval_env.close()
        wandb.finish()
