from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from overcooked_ai_py.mdp.overcooked_env import (
    Overcooked,
    OvercookedEnv,
)
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from src.typings import LayoutName, OvercookedAction, OvercookedObs


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
            base_env = OvercookedEnv.from_mdp(
                mdp, info_level=info_level, horizon=horizon
            )
            env = Overcooked(
                base_env=base_env, featurize_fn=base_env.featurize_state_mdp
            )
            self._envs.append(env)

        self._cur = self._envs[0]
        dummy_raw = self._cur.reset()

        obs_agent_0 = np.array(dummy_raw["both_agent_obs"][0])
        obs_agent_1 = np.array(dummy_raw["both_agent_obs"][1])

        # TODO:: with more layouts we cannot take shape from first obs
        # since in the reset we randomly pick a new layout and obs shape may differ
        # we will take the max shape
        self.observation_space = spaces.Dict(
            {
                "agent_0_obs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_agent_0.shape, dtype=np.float32
                ),
                "agent_1_obs": spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_agent_1.shape, dtype=np.float32
                ),
            }
        )

        self.action_space = self._envs[0].action_space

    @property
    def current_env(self) -> Overcooked:
        return self._cur

    def _process_obs(self, raw_obs: Dict[str, Any]) -> OvercookedObs:
        """
        Process the raw observation from the Overcooked environment into the desired format.
        """
        obs_agent_0 = np.array(raw_obs["both_agent_obs"][0])
        obs_agent_1 = np.array(raw_obs["both_agent_obs"][1])

        obs_dict = {
            "agent_0_obs": obs_agent_0,
            "agent_1_obs": obs_agent_1,
        }
        return obs_dict

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

        obs = self._process_obs(raw_obs)
        info: Dict[str, Any] = {"layout_name": self.layouts[idx]}

        return obs, info

    def step(
        self, action: OvercookedAction
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step within the environment.

        Args:
            action: A joint action tuple (action_agent_0, action_agent_1).

        Returns:
            A tuple containing (obs, reward, terminated, truncated, info).
        """
        raw_obs, reward, done, info = self._cur.step(action)
        obs = self._process_obs(raw_obs)

        terminated = False
        truncated = done

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns:
            A NumPy arrayof shape (height, width, 3) representing the RGB image of the current state.
        """
        if self.render_mode == "rgb_array":
            return self._cur.render(model="rgb_array")
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
