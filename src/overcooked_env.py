import copy
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from overcooked_ai_py.mdp.overcooked_env import Action, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from src.utils.typings import EncodingType, LayoutName, OvercookedAction, OvercookedObs


class OvercookedGym(gym.Env[OvercookedObs, OvercookedAction]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layouts: list[LayoutName],
        encoding: EncodingType,
        info_level: int = 0,
        horizon: int = 400,
        render_mode: str = "rgb_array",
    ) -> None:
        """
        Args:
            layouts: A list of layout names (strings) to choose from.
            encoding: The type of state encoding to use.
            info_level: The verbosity level of the Overcooked environment.
            horizon: The max number of steps per episode.
            render_mode: The render mode, currently only "rgb_array" is supported.
        """
        self.layouts = layouts
        self.encoding = encoding
        self.render_mode = render_mode
        self._envs: list[OvercookedEnv] = []

        for layout in layouts:
            mdp = OvercookedGridworld.from_layout_name(layout)
            env = OvercookedEnv.from_mdp(mdp, info_level=info_level, horizon=horizon)
            self._envs.append(env)

        self._active_env = self._envs[0]
        self.agent_idx = 0

        ############ Definition of action space ###################
        # North, South, East, West, Stay, Interact -> 6 actions
        self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))
        ###########################################################

        ############ Definition of observation space ##############
        dummy_mdp: OvercookedGridworld = self._active_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()

        # Motion Level Action Manager used to featurize the state
        dummy_mlam = self._active_env.mlam

        if self.encoding == EncodingType.FEATURIZED:
            obs = dummy_mdp.featurize_state(dummy_state, dummy_mlam)[0]
            low = -np.inf
            high = np.inf

        elif self.encoding == EncodingType.LOSSLESS:
            obs = dummy_mdp.lossless_state_encoding(dummy_state)[0]
            low = 0.0
            high = np.inf

        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding}")

        # The observation space is a tuple of two agent observations
        self.observation_space = spaces.Tuple(
            tuple(spaces.Box(low=low, high=high, shape=obs.shape, dtype=np.float32) for _ in range(2))
        )
        ############################################################
        self.visualizer = StateVisualizer()

    def _encode_obs_and_swap(self, state: OvercookedState, env: OvercookedEnv) -> OvercookedObs:
        mdp = env.mdp

        if self.encoding == EncodingType.FEATURIZED:
            obs_p1, obs_p2 = mdp.featurize_state(state, env.mlam)
        else:
            obs_p1, obs_p2 = mdp.lossless_state_encoding(state)

        if self.agent_idx == 0:
            return (obs_p1, obs_p2)
        else:
            return (obs_p2, obs_p1)

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
        self._active_env = self._envs[idx]

        self.agent_idx = self.np_random.integers(0, 2)

        self._active_env.reset()
        obs = self._encode_obs_and_swap(self._active_env.state, self._active_env)

        info = {"layout_name": self.layouts[idx], "agent_idx": self.agent_idx}
        return obs, info

    def step(self, action: OvercookedAction) -> Tuple[OvercookedObs, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step within the environment.

        Args:
            action: A joint action tuple (action_agent_0, action_agent_1).

        Returns:
            A tuple containing (obs, reward, terminated, truncated, info).
        """
        action_p1, action_p2 = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (action_p1, action_p2)
        else:
            joint_action = (action_p2, action_p1)

        state, reward, done, info = self._active_env.step(joint_action)

        obs = self._encode_obs_and_swap(state, self._active_env)

        terminated = False
        truncated = bool(done)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Returns:
            A NumPy array of shape (height, width, 3) representing the RGB image of the current state.
        """
        if self.render_mode == "rgb_array":
            rewards_dict = {}  # dictionary of details you want rendered in the UI
            for key, value in self._active_env.game_stats.items():
                if key in [
                    "cumulative_shaped_rewards_by_agent",
                    "cumulative_sparse_rewards_by_agent",
                ]:
                    rewards_dict[key] = value

            image = self.visualizer.render_state(
                state=self._active_env.state,
                grid=self._active_env.mdp.terrain_mtx,
                hud_data=StateVisualizer.default_hud_data(self._active_env.state, **rewards_dict),
            )

            buffer = pygame.surfarray.array3d(image)
            image = copy.deepcopy(buffer)
            image = np.flip(np.rot90(image, 3), 1)
            image = cv2.resize(image, (2 * 528, 2 * 464))

            return image

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
