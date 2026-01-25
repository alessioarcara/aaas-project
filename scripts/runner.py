import os

from src.config import Config
from src.train import train
from src.utils.constants import TRAIN_LAYOUTS

os.environ["SDL_AUDIODRIVER"] = "dummy"


def main():
    for layout in TRAIN_LAYOUTS:
        config = Config.from_files(
            config_paths=["configs/optimized.yaml"],
            overrides={
                "exp_name": f"{layout}",
                "wandb_group": "specialized",
                "training_config": {
                    "total_updates": 5_000_000,
                },
                "env_config": {"layouts": [layout]},
            },
        )
        train(config)


if __name__ == "__main__":
    main()
