import gymnasium_robotics

from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.maps import U_MAZE

from src.environment import events as processes

import numpy as np
from typing import Dict, List, Optional, Union

class StationaryMazeEnv(PointMazeEnv):
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        goal=-1,
        randomize_goal=False,
        **kwargs,
    ):
        self.map_rows, self.map_cols = len(maze_map), len(maze_map[0])

        if not randomize_goal:
            assert goal != -1

            for i in range(self.map_rows):
                for j in range(self.map_cols):
                    if maze_map[i][j]==0:
                        maze_map[i][j]='r'

            maze_map[goal[0]][goal[1]]='c'
        else:
            for i in range(self.map_rows):
                for j in range(self.map_cols):
                    if maze_map[i][j]==0:
                        maze_map[i][j]='c'

        super().__init__(maze_map=maze_map,
                         render_mode=render_mode,
                         reward_type=reward_type,
                         continuing_task=continuing_task,
                         reset_target=reset_target,
                         **kwargs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        obs_dict, info = super().reset(seed=seed, **kwargs)

        return obs_dict, info

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        obs_dict = self._get_obs(obs)

        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        # Update the goal position if necessary
        self.update_goal(obs_dict["achieved_goal"])

        return obs_dict, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if distance <= 0.45:
            return 0.0

        return -0.01*np.exp(distance/10)

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    #     if distance <= 0.45:
    #         return 0
    #     return -0.01

class NonStationaryMazeEnv(PointMazeEnv):
    def __init__(
        self,
        maze_map: List[List[Union[str, int]]] = U_MAZE,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        process_params={},
        event_warmup=0,
        gravity_centers=None,
        goal=-1,
        randomize_goal=False,
        **kwargs,
    ):
        self.gravity_centers = gravity_centers
        self.map_rows, self.map_cols = len(maze_map), len(maze_map[0])

        if not randomize_goal:
            assert goal != -1

            for i in range(self.map_rows):
                for j in range(self.map_cols):
                    if maze_map[i][j]==0:
                        maze_map[i][j]='r'

            maze_map[goal[0]][goal[1]]='c'
        else:
            for i in range(self.map_rows):
                for j in range(self.map_cols):
                    if maze_map[i][j]==0:
                        maze_map[i][j]='c'

        super().__init__(maze_map=maze_map,
                         render_mode=render_mode,
                         reward_type=reward_type,
                         continuing_task=continuing_task,
                         reset_target=reset_target,
                         **kwargs)

        # self.gravity_centers = np.random.uniform([1.5-len(maze_map[0])/2, 1.5-len(maze_map)/2], [len(maze_map[0])/2-1.5, len(maze_map)/2-1.5], size=(process_params['gravity_centers'],2))
        # self.max_episode_steps = kwargs['max_episode_steps']


        self.event_process = processes.EventProcess(params=process_params)
        self.event_warmup = event_warmup

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        obs_dict, info = super().reset(seed=seed, **kwargs)

        self.event_process.reset()

        for _ in range(self.event_warmup):
            self.event_process.step()

        info['event_history'] = self.event_process.event_history

        return obs_dict, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        perturbed_action = action.copy()
        K = 1.0*5
        eps = 1e-6
        cur_obs = self.point_env._get_obs()

        # print("#"*50)
        for i in range(self.gravity_centers.shape[0]):
            r = np.linalg.norm(cur_obs[0][:2]-self.gravity_centers[i])
            force_mag = K * self.event_process.event_history[-1, i] / (r+eps)**2
            # force_mag = K * 0.1 / (r+eps)**2
            # print(force_mag)
            force = self.gravity_centers[i] - cur_obs[0][:2]
            force = force * np.abs(force_mag) / np.linalg.norm(force)
            # print(force)
            perturbed_action += force

            # perturbed_action += np.clip(force, -1, 1)
        # perturbed_action =
        # print(perturbed_action-action)
        # print(self.gravity_centers[0])
        # print(self.point_env._get_obs())
        # print("#"*50)
        obs, _, _, _, info = self.point_env.step(perturbed_action)
        obs_dict = self._get_obs(obs)

        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)
        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        # Update the goal position if necessary
        self.update_goal(obs_dict["achieved_goal"])
        self.event_process.step()
        info['event_history'] = self.event_process.event_history

        return obs_dict, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if distance <= 0.45:
            return 0.0

        return -0.01*np.exp(distance/10)
    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    #     if distance <= 0.45:
    #         return 100
    #     return -1.0
        # return super().compute_reward(achieved_goal, desired_goal, info)
