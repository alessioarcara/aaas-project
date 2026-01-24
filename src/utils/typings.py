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
    "cramped_room_tomato",
    "five_by_five",
]

OvercookedObs: TypeAlias = tuple[np.ndarray, np.ndarray]  # pre_wrapper: (n_agents, obs_dim)
OvercookedAction: TypeAlias = tuple[int, int]  # (action_agent_0, action_agent_1)


class ShapingMode(StrEnum):
    INDIVIDUAL = "individual"
    SHARED = "shared"
    NONE = "none"


class EncodingType(StrEnum):
    LOSSLESS = "lossless"
    FEATURIZED = "featurized"
