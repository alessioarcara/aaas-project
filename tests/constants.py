from typing import Any, Final

NUM_ENVS: Final[int] = 4

BASE_CONFIG: Final[dict[str, Any]] = {
    "exp_name": "test_exp",
    "wandb_project_name": "test_proj",
    "wandb_entity": "test_entity",
    "env_config": {"num_envs": 4, "layouts": ["cramped_room"], "horizon": 400},
    "training_config": {
        "num_steps": 100,
        "total_updates": 10,
        "learning_rate": 3e-4,
        "minibatch_size": 4,
        "update_epochs": 1,
        "network": {
            "type": "cnn",
            "embedding_dim": 256,
            "shared_backbone": True,
            "num_filters": [32],
            "kernel_sizes": [3],
            "strides": [1],
            "paddings": ["SAME"],
        },
    },
}
