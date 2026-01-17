from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from loguru import logger

from src.utils.typings import OvercookedAction, OvercookedObs, ShapingMode


class OvercookedRewardShapingWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env[OvercookedObs, OvercookedAction],
        shaping_mode: ShapingMode,
    ) -> None:
        """
        A Gym wrapper that adds reward shaping to the Overcooked environment.

        Args:
            env: The Overcooked Gym environment to wrap.
            shaping_mode: The mode of reward shaping to apply. Can be INDIVIDUAL, SHARED, or NONE.
                - INDIVIDUAL: Each agent receives its own shaped reward.
                - SHARED: All agents receive the same sum of shaped rewards.
                - NONE: No reward shaping is applied.
        """
        super().__init__(env)
        self.shaping_mode = shaping_mode

    def step(self, action: Any) -> Tuple[OvercookedObs, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if np.isscalar(reward):
            # ! In Overcooked, the number of agents is always 2
            reward = np.full((2,), reward, dtype=np.float32)
        else:
            logger.warning("⚠️ Expected scalar reward, but got an array!")

        if self.shaping_mode != ShapingMode.NONE:
            shaping = info.get("shaped_r_by_agent", None)

            if shaping is not None:
                shaping_arr = np.array(shaping, dtype=np.float32)

                if self.shaping_mode == ShapingMode.INDIVIDUAL:
                    # [R, R] + [S1, S2]
                    reward += shaping_arr

                elif self.shaping_mode == ShapingMode.SHARED:
                    # [R, R] + [S1+S2, S1+S2]
                    reward += np.sum(shaping_arr)

        return obs, reward, terminated, truncated, info
