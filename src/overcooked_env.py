from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from overcooked_ai_py.mdp.overcooked_env import (
    Overcooked,
    OvercookedEnv,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from src.utils.typings import LayoutName, OvercookedAction, OvercookedObs


class OvercookedGym(gym.Env[OvercookedObs, OvercookedAction]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layouts: list[LayoutName],
        info_level: int = 0,
        horizon: int = 400,
        render_mode: str = "rgb_array",
    ) -> None:
        """
        Args:
            layouts: A list of layout names (strings) to choose from.
            info_level: The verbosity level of the Overcooked environment.
            horizon: The max number of steps per episode.
            render_mode: The render mode, currently only "rgb_array" is supported.
        """
        self.render_mode = render_mode
        self.layouts = layouts
        self._envs = []

        for layout in layouts:
            mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self._envs.append(env)

        self._cur = self._envs[0]
        dummy = self._cur.reset()

        ref_obs = dummy["both_agent_obs"]
        self.num_agents = len(ref_obs)
        obs_shape = ref_obs[0].shape

        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_shape,
                    dtype=np.float32,
                )
                for _ in range(self.num_agents)
            )
        )
        self.action_space = self._envs[0].action_space

    @property
    def current_env(self) -> Overcooked:
        return self._cur

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[OvercookedObs, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            obs: the initial observation.
            info: Dictionary containing auxiliary information.
        """
        super().reset(seed=seed)

        idx = self.np_random.integers(0, len(self._envs))
        self._cur = self._envs[idx]

        raw_obs = self._cur.reset()

        obs = tuple(raw_obs["both_agent_obs"])

        info = {"layout_name": self.layouts[idx]}
        return obs, info

    def step(self, action: OvercookedAction) -> Tuple[OvercookedObs, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step within the environment.

        Args:
            action: A joint action tuple (action_agent_0, action_agent_1).

        Returns:
            A tuple containing (obs, reward, terminated, truncated, info).
        """
        raw_obs, reward, done, info = self._cur.step(action)
        obs = tuple(raw_obs["both_agent_obs"])

        terminated = False
        truncated = bool(done)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns:
            A NumPy array of shape (height, width, 3) representing the RGB image of the current state.
        """
        if self.render_mode == "rgb_array":
            return self._cur.render()
        return None

    def close(self) -> None:
        """
        Close all underlying Overcooked environments
        """
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass
