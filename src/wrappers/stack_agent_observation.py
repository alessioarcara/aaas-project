import gymnasium as gym
import numpy as np


class StackAgentObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper that stacks the observations of all agents into a single observation array.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        agent_space = env.observation_space[0]
        num_agents = len(env.observation_space.spaces)

        self.observation_space = gym.spaces.Box(
            low=np.stack([agent_space.low] * num_agents),
            high=np.stack([agent_space.high] * num_agents),
            dtype=agent_space.dtype,
        )

    def observation(self, obs: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.stack(obs)
