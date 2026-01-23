import argparse
import gc
from pathlib import Path

import optuna
from loguru import logger
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.config import Config
from src.train import train

STORAGE_URL = "sqlite:///optuna_study.db"
SEED = 42


def objective(trial: optuna.Trial, base_config_path: Path, group_name: str) -> float:
    trial_num = trial.number

    # ! --- Rollout ---
    num_steps = trial.suggest_categorical("num_steps", [400, 800])

    # ! --- PPO ---
    ent_coef = trial.suggest_float("entropy_coef", 0.01, 0.3, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    update_epochs = trial.suggest_int("update_epochs", 4, 10)

    # ! --- Optimization ---
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # ! --- Network ---
    net_type = trial.suggest_categorical("network_type", ["mlp", "cnn"])
    param_sharing = trial.suggest_categorical("use_parameter_sharing", [True, False])

    # ! --- MLP ---
    if net_type == "mlp":
        env_encoding = "featurized"
        hidden_dim = trial.suggest_categorical("mlp_hidden_dim", [64, 128, 256])
        shared_backbone = trial.suggest_categorical("mlp_shared_backbone", [True, False])
        network_config = {
            "type": "mlp",
            "hidden_dim": hidden_dim,
            "embedding_dim": 64,
            "shared_backbone": shared_backbone,
        }

    # ! --- CNN ---
    else:
        env_encoding = "lossless"
        cnn_preset = trial.suggest_categorical("cnn_preset", ["small", "medium"])
        shared_backbone = trial.suggest_categorical("cnn_shared_backbone", [True, False])

        if cnn_preset == "small":
            network_config = {
                "type": "cnn",
                "num_filters": [32, 32],
                "kernel_sizes": [3, 3],
                "strides": [1, 1],
                "paddings": ["SAME", "SAME"],
                "embedding_dim": 128,
            }
        else:
            network_config = {
                "type": "cnn",
                "num_filters": [32, 64, 64],
                "kernel_sizes": [3, 3, 3],
                "strides": [1, 1, 1],
                "paddings": ["SAME", "SAME", "SAME"],
                "embedding_dim": 256,
            }

        network_config["shared_backbone"] = shared_backbone

    overrides = {
        "seed": SEED,
        "exp_name": f"optuna_trial_{trial_num}",
        "wandb_group": group_name,
        "env_config": {
            "encoding": env_encoding,
        },
        "training_config": {
            "num_steps": num_steps,
            "learning_rate": lr,
            "entropy_coef": ent_coef,
            "gae_lambda": gae_lambda,
            "use_parameter_sharing": param_sharing,
            "update_epochs": update_epochs,
            "network": network_config,
        },
    }

    try:
        cfg = Config.from_files(config_paths=[base_config_path], overrides=overrides)
        best_score = train(cfg, trial=trial)
        gc.collect()
        return best_score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial_num} failed with error: {e}")
        return -float("inf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Overcooked PPO")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Path to the configuration YAML")
    parser.add_argument("--study-name", type=str, default="overcooked_ppo_v1", help="Name of the study")
    parser.add_argument("--trials", type=int, default=50, help="Total number of trials to run")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Config file {config_path} does not exist.")
        exit(1)

    sampler = TPESampler(seed=SEED)
    pruner = HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)

    logger.info(f"Initializing Optuna Study '{args.study_name}' using local DB: {STORAGE_URL}...")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    logger.info("🚀 Starting Hyperparameter Tuning...")
    try:
        study.optimize(
            lambda trial: objective(trial, config_path, group_name=args.study_name),
            n_trials=args.trials,
            n_jobs=1,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        logger.warning("⚠️ Tuning interrupted by user.")
    except Exception as e:
        logger.exception("❌ Critical error during tuning: {}", e)

    if len(study.trials) > 0:
        logger.success(f"Tuning complete. Best Value: {study.best_value:.6f}")
    else:
        logger.warning("No trials completed.")
