import gymnasium as gym
import numpy as np

from gymnasium.wrappers import FrameStackObservation




class CustomCarRacingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        off_track = True

        if hasattr(self.env.unwrapped, 'car') and self.env.unwrapped.car is not None:
            for w in self.env.unwrapped.car.wheels:
                if len(w.tiles)>0:
                    off_track = False

        if not terminated and off_track:
            # print(self.env.unwrapped.prev_reward, self.env.unwrapped.reward, reward)
            reward -= 1
            self.env.unwrapped.prev_reward -= 1
            self.env.unwrapped.reward -= 1

            # terminated = True

        info['off_track'] = True

        return next_observation, reward, terminated, truncated, info


def make_env(render_mode=None, stack_size=4, max_episode_steps=800):
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=render_mode, domain_randomize=False, continuous=False, max_episode_steps=max_episode_steps)
        env = FrameStackObservation(env, stack_size=stack_size)
        env = CustomCarRacingWrapper(env)

        return env
    return _thunk
