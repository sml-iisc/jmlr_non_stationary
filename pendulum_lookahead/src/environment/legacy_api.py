"""Adapt Gymnasium to the four-return API used by archived training scripts."""
import gymnasium as gym

class LegacyAPI(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info

def make(*args, **kwargs):
    return LegacyAPI(gym.make(*args, **kwargs))
