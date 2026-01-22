from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

PathOrStr: TypeAlias = Path | str

LayoutName: TypeAlias = Literal[
    "cramped_room",
    "asymmetric_advantages",
    "counter_circuit",
    "forced_coordination",
    "coordination_ring",
]

OvercookedObs: TypeAlias = tuple[np.ndarray, np.ndarray]  # shape: (n_agents, obs_dim)
OvercookedAction: TypeAlias = tuple[int, int]  # (action_agent_0, action_agent_1)


class ShapingMode(StrEnum):
    INDIVIDUAL = "individual"
    SHARED = "shared"
    NONE = "none"


class EncodingType(StrEnum):
    LOSSLESS = "lossless"
    FEATURIZED = "featurized"
