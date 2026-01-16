from typing import Any, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str, cls: Type[T]) -> None:
        if name in self._registry:
            raise KeyError(f"{name} already registered")
        self._registry[name] = cls

    def get(self, name: str) -> Type[T] | None:
        return self._registry.get(name)


def build_from_config(value: dict | list[dict] | Any, *registries: Registry):
    """
    Instantiates objects from configuration searching for 'type' in the provided registries.
    """
    # List -> recursive call for each item
    if isinstance(value, list):
        return [build_from_config(v, *registries) for v in value]

    # Dict -> can be a class to instantiate or a simple parameter dict
    if isinstance(value, dict):
        if "type" in value:
            cls_name = value["type"]
            params = value.get("params", {})

            # Resolve parameters recursively BEFORE passing to the constructor
            # This ensures nested objects are built first
            resolved_params = {k: build_from_config(v, *registries) for k, v in params.items()}

            # Find the class in the provided registries
            cls_to_init = None
            for reg in registries:
                cls_to_init = reg.get(cls_name)
                if cls_to_init is not None:
                    break

            if cls_to_init is None:
                raise KeyError(f"Class {cls_name} not found in any of the provided registries")

            try:
                # Instantiate the class with resolved parameters
                return cls_to_init(**resolved_params)
            except Exception as e:
                raise ValueError(f"Failed to instantiate {cls_name} with params {params}: {e}")

        # if doesn't have 'type', it is a standard dict -> recursive call for each value
        return {k: build_from_config(val, *registries) for k, val in value.items()}

    # Primitives (int, float, str, etc.) -> return value as is
    return value
