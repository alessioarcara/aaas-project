from typing import Any, Final

NUM_ENVS: Final[int] = 4

BASE_CONFIG: Final[dict[str, Any]] = {
    "exp_name": "test_exp",
    "wandb_project_name": "test_proj",
    "wandb_entity": "test_entity",
    "env_config": {"num_envs": 4, "horizon": 400},
    "training_config": {
        "num_steps": 100,
        "total_updates": 10,
    },
}
