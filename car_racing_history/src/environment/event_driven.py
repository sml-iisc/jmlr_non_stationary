import gymnasium as gym
import numpy as np

from gymnasium.wrappers import FrameStackObservation



from src.environment.events import DiscreteHawkesProcess


class CustomCarRacingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.hawkes_process = DiscreteHawkesProcess()
        self.steps_taken = 0
        self.alpha = 0.1
        self.decay = 0.05

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.hawkes_process = DiscreteHawkesProcess()  # Reset the Hawkes process
        self.steps_taken = 0
        info['events'] = self.hawkes_process.events  # Include events in info

        return observation, info

    def step(self, action):
        friction = 1.0
        current_time = self.hawkes_process.delta * self.steps_taken

        for event in self.hawkes_process.get_events():
            friction -= self.alpha * np.exp(-self.decay * (current_time - event))

        # print(f"Current friction: {max(friction,0.5)}")
        if hasattr(self.env.unwrapped, 'car') and self.env.unwrapped.car is not None:
            for w in self.env.unwrapped.car.wheels:
                for tile in w.tiles:
                    # print(tile.road_friction)
                    tile.road_friction = friction

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.hawkes_process.step()
        self.steps_taken += 1
        info['events'] = self.hawkes_process.events  # Include events in info
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
