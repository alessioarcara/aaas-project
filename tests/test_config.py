import pytest
import yaml

from src.config import Config

BASE_CONFIG = {
    "exp_name": "test_exp",
    "env_config": {
        "num_envs": 4,
    },
    "training_config": {
        "num_steps": 100,
        "wandb_project_name": "test_proj",
        "wandb_entity": "test_entity",
    },
}


@pytest.fixture
def create_yaml(tmp_path):
    """ """

    def _create(content, filename="config.yaml"):
        p = tmp_path / filename
        with open(p, "w") as f:
            yaml.dump(content, f)
        return str(p)

    return _create


def test_load_valid_config(create_yaml):
    path = create_yaml(BASE_CONFIG)

    config = Config.from_files([path])

    assert config.exp_name == "test_exp"
    assert config.env_config.num_envs == 4


def test_overrides(create_yaml):
    path = create_yaml(BASE_CONFIG)

    config = Config.from_files([path], overrides={"env_config": {"num_envs": 99}})

    assert config.env_config.num_envs == 99


def test_validation_error(create_yaml):
    bad_data = BASE_CONFIG.copy()
    bad_data["training_config"] = {"num_steps": "not_an_int"}

    path = create_yaml(bad_data)

    with pytest.raises(SystemExit):
        Config.from_files([path])


def test_path_to_non_existent_file(tmp_path):
    bad_path = str(tmp_path / "non_existent.yaml")

    with pytest.raises(SystemExit):
        Config.from_files([bad_path])
