import os
from typing import Final

from src.config import Config
from src.train import train

os.environ["SDL_AUDIODRIVER"] = "dummy"

CONFIG_PATH: Final[str] = "configs/config.yaml"


def main():
    cfg = Config.from_files([CONFIG_PATH])
    train(cfg)


if __name__ == "__main__":
    main()
