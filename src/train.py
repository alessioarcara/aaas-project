from typing import Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optuna
from flax import nnx
from loguru import logger
from tqdm import tqdm

import wandb
from src.agent_pair import AgentPair
from src.base_agent import Agent
from src.config import Config
from src.rollout import Carry, collect_rollouts, compute_gae
from src.utils.constants import STATS_KEY
from src.utils.misc import latest_video_path, set_global_seeds


def eval_agent(
    agent: Agent,
    eval_envs: gym.vector.VectorEnv,
    seed: int,
    num_episodes: int,
) -> tuple[float, list[float]]:
    """
    Evaluate the agent across all configured layouts over a specified number of episodes.

    Args:
        agent: The policy to evaluate.
        eval_envs: Vectorized environments containing the layouts.
        seed: Base seed for reproducibility.
        num_episodes: How many full episodes to run per layout.

    Returns:
        total_score: Sum of mean rewards across all layouts (scalar).
        layout_scores: List of mean rewards for each specific layout.
    """
    num_envs = eval_envs.num_envs
    all_episode_rewards = np.zeros((num_episodes, num_envs))

    eval_seeds = [seed + i for i in range(num_episodes)]

    for i, eval_seed in enumerate(eval_seeds):
        obs, _ = eval_envs.reset(seed=eval_seed)
        done = np.zeros(num_envs, dtype=bool)

        while not np.all(done):
            obs_jnp = jax.device_put(obs)
            actions = agent.get_deterministic_action(obs_jnp)
            actions = np.array(actions)
            obs, _, terminated, truncated, info = eval_envs.step(actions)
            done = np.logical_or(done, np.logical_or(terminated, truncated))

        per_layout_reward = info.get(STATS_KEY, {}).get("r", 0.0)
        all_episode_rewards[i] = per_layout_reward

    reward_per_layout = list(np.mean(all_episode_rewards, axis=0))
    total_reward = sum(reward_per_layout)

    return total_reward, reward_per_layout


def train(cfg: Config, trial: Optional[optuna.Trial] = None) -> float:
    """
    Main training loop for PPO agents.

    Steps:
    1. Collect rollouts
    2. Compute GAE advantages and returns
    3. Update agents using collected data
    4. Evaluate agents periodically
    """
    cfg.setup_directories()
    video_dir = cfg.video_dir

    set_global_seeds(cfg.seed)
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
        group=cfg.wandb_group,
        name=cfg.exp_name,
        config=cfg.model_dump(),
    )

    best_eval_reward = -float("inf")

    try:
        obs, _ = train_envs.reset(seed=cfg.seed)
        key, rollout_key = jax.random.split(key)

        carry = Carry(
            jnp.array(obs),
            jnp.zeros(train_envs.num_envs, dtype=bool),
            rollout_key,
        )

        for update in tqdm(range(1, num_updates + 1), desc="Training", unit="update", colour="blue", leave=False):
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
                total_reward, rewards_per_layout = eval_agent(
                    agent,
                    eval_envs,
                    seed=cfg.seed,
                    num_episodes=cfg.num_eval_episodes,
                )

                if total_reward > best_eval_reward:
                    best_eval_reward = total_reward

                log_data["eval/total_reward"] = total_reward
                for i, layout in enumerate(cfg.env_config.layouts):
                    log_data[f"eval/{layout}_reward"] = rewards_per_layout[i]

                    if video_dir is not None:
                        if vid_path := latest_video_path(cfg.video_dir / layout):
                            log_data[f"eval/{layout}_video"] = wandb.Video(str(vid_path), format="mp4")

                if trial:
                    trial.report(total_reward, step=update)
                    if trial.should_prune():
                        logger.info("✂️ Trial {} pruned at step {}", trial.number, update)
                        raise optuna.exceptions.TrialPruned()

            wandb.log(log_data, step=global_step)

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
        return best_eval_reward
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.exception("❌ Unhandled exception during training: {}", e)
        return -float("inf")
    finally:
        train_envs.close()
        eval_envs.close()
        wandb.finish()

    return best_eval_reward
