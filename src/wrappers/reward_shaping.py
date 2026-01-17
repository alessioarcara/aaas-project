from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorWrapper

from src.utils.typings import ShapingMode


class OvercookedVectorRewardShapingWrapper(VectorWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        shaping_mode: ShapingMode,
        num_agents: int = 2,
    ) -> None:
        """
        A VectorWrapper that handles reward shaping and dimension broadcasting
        for the entire batch of environments at once.

        Args:
            env: The vectorized Gym environment.
            shaping_mode: The reward shaping mode (INDIVIDUAL, SHARED, NONE).
                - INDIVIDUAL: Each agent receives its own shaped reward.
                - SHARED: All agents receive the same sum of shaped rewards.
                - NONE: No reward shaping is applied.
            num_agents: The number of agents in the environment (default 2 for Overcooked).
        """
        super().__init__(env)
        self.shaping_mode = shaping_mode
        self.num_agents = num_agents

    def step(self, actions: Any) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)

        # ! broadcasting: (num_envs, ) -> (num_envs, num_agents)
        if isinstance(reward, np.ndarray) and reward.ndim == 1:
            reward = np.repeat(reward[:, np.newaxis], self.num_agents, axis=1)

        if self.shaping_mode != ShapingMode.NONE:
            shaping = info.get("shaped_r_by_agent", None)

            if shaping is not None:
                # ! Fix: 'shaping' is an object array of objects
                # ! NumPy tries to cast the whole list to a float, crashing.
                # ! .tolist() unwraps it into a pure Python list-of-lists, allowing NumPy to infer 2D structure.
                shaping_arr = np.array(shaping.tolist(), dtype=np.float32)

                if self.shaping_mode == ShapingMode.INDIVIDUAL:
                    # Logic: [R, R] + [S1, S2]
                    reward += shaping_arr

                elif self.shaping_mode == ShapingMode.SHARED:
                    # Logic: [R, R] + [S1+S2, S1+S2]
                    env_shaping_sum = np.sum(shaping_arr, axis=1, keepdims=True)
                    reward += env_shaping_sum

        return obs, reward, terminated, truncated, info
