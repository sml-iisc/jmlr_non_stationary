import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment import BlockHistoryEnv


class RoutingObservation(gym.ObservationWrapper):
    """Only current and goal one-hots. Connectivity mask is static graph data."""

    def __init__(self, data, worker_seed=1001, fixed=None):
        super().__init__(BlockHistoryEnv(data, 0, worker_seed=worker_seed, fixed=fixed))
        self.n = data.N
        self.observation_space = spaces.Box(0, 1, (2 * self.n,), dtype=np.float32)

    def observation(self, observation):
        result = np.zeros(2 * self.n, dtype=np.float32)
        current, _, goal = observation["route"]
        result[current] = 1
        result[self.n + goal] = 1

        return result
