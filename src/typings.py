from typing import Literal, TypeAlias

import numpy as np

LayoutName: TypeAlias = Literal[
    "cramped_room",
    "asymmetric_advantages",
    "counter_circuit",
    "forced_coordination",
    "coordination_ring",
]

OvercookedObs: TypeAlias = dict[str, np.ndarray]
OvercookedAction: TypeAlias = tuple[int, int]  # (action_agent_0, action_agent_1)
