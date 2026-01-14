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
            base_env = OvercookedEnv.from_mdp(
                mdp, info_level=info_level, horizon=horizon
            )
            env = Overcooked(
                base_env=base_env, featurize_fn=base_env.featurize_state_mdp
            )
            self._envs.append(env)

        self._cur = self._envs[0]
        dummy_raw = self._cur.reset()

        # TODO:: with more layouts we cannot take shape from first obs
        # since in the reset we randomly pick a new layout and obs shape may differ
        # we will take the max shape
        dummy_obs_stack = np.stack(dummy_raw["both_agent_obs"])

        self.n_agents = dummy_obs_stack.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs_stack.shape,
            dtype=np.float32,
        )

        self.action_space = self._envs[0].action_space

    @property
    def current_env(self) -> Overcooked:
        return self._cur

    def _process_obs(self, raw_obs: Dict[str, Any]) -> OvercookedObs:
        """
        Process the raw observation from the Overcooked environment into the desired format.
        """
        return np.stack(raw_obs["both_agent_obs"]).astype(np.float32)

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

        info = {"layout_name": self.layouts[idx]}
        return obs, info

    def step(
        self, action: OvercookedAction
    ) -> Tuple[OvercookedObs, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Execute one step within the environment.

        Args:
            action: A joint action tuple (action_agent_0, action_agent_1).

        Returns:
            A tuple containing (obs, reward, terminated, truncated, info).
        """
        raw_obs, reward, done, info = self._cur.step(action)
        obs = self._process_obs(raw_obs)

        # HACK: Gymnasium `RecordEpisodeStatistics` wrapper attempts to write
        # to the 'episode' key in 'info' when an episode ends.
        # Overcooked env uses this key for its own purposes, so we rename it here
        # to avoid conflicts when using the wrapper
        if "episode" in info:
            info["overcooked_episode"] = info.pop("episode")

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
