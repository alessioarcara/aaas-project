import argparse
import os

from src.config import Config
from src.train import train
from src.utils.misc import set_global_seeds

os.environ["SDL_AUDIODRIVER"] = "dummy"


def main(config_paths: list[str]) -> None:
    cfg = Config.from_files(config_paths)
    set_global_seeds(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the YAML config files",
    )
    args = parser.parse_args()
    main(args.configs)
