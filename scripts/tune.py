import argparse
import gc
import os
from pathlib import Path

import optuna
from loguru import logger
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.config import Config
from src.train import train

os.environ["SDL_AUDIODRIVER"] = "dummy"
STORAGE_URL = "sqlite:///optuna_study.db"
SEED = 42


def get_network_config(trial: optuna.Trial) -> dict[str, any]:
    net_type = trial.suggest_categorical("network_type", ["mlp", "cnn"])
    shared_backbone = trial.suggest_categorical(f"{net_type}_shared_backbone", [True, False])

    if net_type == "mlp":
        return "featurized", {
            "type": "mlp",
            "hidden_dim": trial.suggest_categorical("mlp_hidden_dim", [64, 128, 256]),
            "embedding_dim": 64,
            "shared_backbone": shared_backbone,
        }

    cnn_preset = trial.suggest_categorical("cnn_preset", ["small", "medium"])
    presets = {
        "small": {"num_filters": [32, 32], "kernel_sizes": [3, 3], "embedding_dim": 256},
        "medium": {"num_filters": [32, 64, 64], "kernel_sizes": [3, 3, 3], "embedding_dim": 512},
    }

    net_config = {
        "type": "cnn",
        "strides": [1] * len(presets[cnn_preset]["num_filters"]),
        "paddings": ["SAME"] * len(presets[cnn_preset]["num_filters"]),
        "shared_backbone": shared_backbone,
        **presets[cnn_preset],
    }
    return "lossless", net_config


def objective(trial: optuna.Trial, base_config_path: Path, group_name: str) -> float:
    env_encoding, network_config = get_network_config(trial)

    overrides = {
        "seed": SEED,
        "exp_name": f"trial_{trial.number}",
        "wandb_group": group_name,
        "env_config": {"encoding": env_encoding},
        "training_config": {
            "num_steps": trial.suggest_categorical("num_steps", [400, 800]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "entropy_coef": trial.suggest_float("entropy_coef", 0.01, 0.3, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "update_epochs": trial.suggest_int("update_epochs", 4, 10),
            "use_parameter_sharing": trial.suggest_categorical("use_parameter_sharing", [True, False]),
            "network": network_config,
        },
    }

    try:
        cfg = Config.from_files(config_paths=[base_config_path], overrides=overrides)
        return train(cfg, trial=trial)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error("Trial {} failed: {}", trial.number, e)
        return -float("inf")
    finally:
        gc.collect()


def main(args: argparse.Namespace):
    name = args.study_name
    n_trials = args.trials
    config_path = args.config
    study = optuna.create_study(
        study_name=name,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=SEED, multivariate=True, group=True),
        pruner=HyperbandPruner(min_resource=5, reduction_factor=3),
    )

    logger.info(f"Optimization '{name}' started (Trials: {n_trials})")
    study.optimize(lambda trial: objective(trial, Path(config_path), name), n_trials=n_trials, gc_after_trial=True)

    if study.trials:
        logger.success(f"Best Trial: {study.best_trial.number} | Value: {study.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Overcooked PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the study")
    parser.add_argument("--trials", type=int, default=50, help="Total number of trials to run")
    args = parser.parse_args()

    main(args)
