from pydantic import BeforeValidator

from src.config.registry import Registry, build_from_config


def registry_validator(*registries: Registry) -> BeforeValidator:
    """
    Returns a Pydantic validator that instantiates objects from configuration using the provided registries.
    """
    return BeforeValidator(lambda v: build_from_config(v, *registries))
