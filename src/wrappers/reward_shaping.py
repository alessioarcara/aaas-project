from typing import Any, Tuple

import gymnasium as gym

from src.utils.typings import OvercookedObs


class OvercookedRewardShapingWrapper(gym.Wrapper):
    def step(self, action: Any) -> Tuple[OvercookedObs, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping = info.get("shaped_r_by_agent", 0)
        shaping_val = sum(shaping) if isinstance(shaping, list) else shaping

        return obs, reward + shaping_val, terminated, truncated, info
