import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment import BlockHistoryEnv


class LilacRouteEnv(gym.ObservationWrapper):

    def __init__(self, data, worker_seed=1001, fixed=None):
        super().__init__(
            BlockHistoryEnv(data, history=0, worker_seed=worker_seed, fixed=fixed)
        )
        self.observation_space = spaces.Box(0, data.N - 1, shape=(2,), dtype=np.int64)

    def observation(self, observation):
        return observation["route"][[0, 2]].copy()
