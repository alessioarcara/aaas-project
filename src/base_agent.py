from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import jax
from flax import struct

if TYPE_CHECKING:
    from src.rollout import TrainingSegment


@struct.dataclass
class AgentOutput:
    action: jax.Array
    action_log_prob: jax.Array
    entropy: jax.Array
    value: jax.Array


class Agent(ABC):
    """Base class for RL agents."""

    @abstractmethod
    def get_learning_rate(self, step: int) -> jax.Array: ...

    @abstractmethod
    def get_deterministic_action(self, obs: jax.Array) -> jax.Array: ...

    @abstractmethod
    def get_value(self, obs: jax.Array) -> jax.Array: ...

    @abstractmethod
    def get_action_and_value(
        self,
        obs: jax.Array,
        action: Optional[jax.Array] = None,
        key: Optional[jax.Array] = None,
    ) -> AgentOutput: ...

    @abstractmethod
    def learn_from(
        self,
        segment: "TrainingSegment",
        advantages: jax.Array,
        returns: jax.Array,
        key: jax.Array,
    ) -> dict[str, Any]: ...
