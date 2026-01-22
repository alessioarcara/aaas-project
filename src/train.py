import shutil

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from loguru import logger
from tqdm import tqdm

import wandb
from src.agent_pair import AgentPair
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae
from src.utils.constants import STATS_KEY
from src.utils.misc import latest_video_path


def eval_agent(agent: AgentPair, eval_envs: gym.vector.VectorEnv) -> list[float]:
    """
    Evaluate the agent on all Overcooked layouts.
    Returns an episode reward for each layout.
    """
    obs, _ = eval_envs.reset()
    num_envs = eval_envs.num_envs

    done = np.zeros(num_envs, dtype=bool)

    while not np.all(done):
        obs_jnp = jax.device_put(obs)
        actions = agent.get_deterministic_action(obs_jnp)

        actions = np.array(actions)
        obs, _, terminated, truncated, info = eval_envs.step(actions)

        done = np.logical_or(done, np.logical_or(terminated, truncated))

    reward_per_layout = info.get(STATS_KEY, {}).get("r", 0.0)
    return reward_per_layout


def train(cfg: Config):
    """
    1. Collect rollouts
    2. Compute GAE advantages and returns
    3. Update agents using collected data
    4. Evaluate agents periodically
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

    agent = AgentPair(
        cfg=cfg.training_config,
        envs=train_envs,
        rngs=nnx.Rngs(cfg.seed),
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
            global_step = update * batch_size

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

            log_data = {f"train/{k}": v.item() for k, v in metrics.items()}
            log_data["train/lr"] = agent.get_learning_rate(global_step).item()

            if cfg.eval_interval > 0 and update % cfg.eval_interval == 0:
                rewards_per_layout = eval_agent(agent, eval_envs)

                for i, layout in enumerate(cfg.env_config.layouts):
                    log_data[f"eval/{layout}_reward"] = rewards_per_layout[i]

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
        eval_envs.close()
        wandb.finish()
