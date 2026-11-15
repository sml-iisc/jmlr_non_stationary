"""Evaluate the supplied or newly trained models and save per-episode returns."""
from pathlib import Path
import argparse
import csv
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from src.models.networks import FeedforwardNeuralNetModel, FeedforwardHybridNeuralNetModel
from src.environment import maps
from src.models import lookahead as policy
from src.utils import features as utils
from src.environment import events as processes

from src.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--actor', type=Path, default=ROOT / 'data/checkpoints/actor.pth')
    p.add_argument('--naive', type=Path, default=ROOT / 'data/checkpoints/naive_transition.pt')
    p.add_argument('--intelligent', type=Path, default=ROOT / 'data/checkpoints/intelligent_transition.pt')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/evaluation_returns.csv')
    p.add_argument('--depth', type=int, default=5)
    p.add_argument('--trajectories', type=int, default=10)
    args = p.parse_args()

    if args.episodes < 1:
        p.error('--episodes must be positive')

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    params = dict(event_dim=1, time_horizon=1000, gravity_centers=5, base_intensity=0.1,
                  alpha_coef=1, beta_coef=1, lamda_intensity=0.02, lamda_means=0.2)
    maze, centers = maps.get_map(10, 10, 5)
    register(id='ReproductionMaze-v0', entry_point='src.environment.maze:NonStationaryMazeEnv', max_episode_steps=300)
    env = gym.make('ReproductionMaze-v0', maze_map=maze, gravity_centers=centers,
                   randomize_goal=True, goal=-1, reward_type='dense', continuing_task=False,
                   process_params=params, render_mode=None)
    obs_dim, event_horizon, event_dim = 6, 5, 5
    state_dim = obs_dim + event_dim * event_horizon
    action_dim = env.action_space.shape[0]
    actor = FeedforwardNeuralNetModel(obs_dim, 256, action_dim)
    actor.load_state_dict(torch.load(args.actor, map_location='cpu', weights_only=True))
    naive_model = FeedforwardHybridNeuralNetModel(state_dim, action_dim, 256, state_dim)
    intelligent_model = FeedforwardHybridNeuralNetModel(state_dim, action_dim, 256, obs_dim)
    naive_model.load_state_dict(torch.load(args.naive, map_location='cpu', weights_only=True))
    intelligent_model.load_state_dict(torch.load(args.intelligent, map_location='cpu', weights_only=True))
    naive_agent = policy.StationaryNaiveLookaheadPolicy(env, obs_dim, event_dim, event_horizon,
        action_dim, actor, naive_model, utils.reward_fn, perturbation=0.2, root_degree=5,
        depth=args.depth, num_trajectories=args.trajectories)
    intelligent_agent = policy.StationaryIntelligentLookaheadPolicy(env, obs_dim, event_dim,
        event_horizon, action_dim, actor, processes.EventProcess(params=params), intelligent_model,
        utils.reward_fn, 0.2, 5, args.depth, args.trajectories, torch.device('cpu'))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open('w', newline='') as f, torch.no_grad():
        writer = csv.writer(f)
        writer.writerow(['episode', 'baseline', 'intelligent', 'naive'])

        for episode in range(args.episodes):
            returns = []

            for kind in ['baseline', 'intelligent', 'naive']:
                obs_dict, _ = env.reset(seed=args.seed + episode)
                total = 0.0

                for step in range(300):
                    obs = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])
                    history = env.unwrapped.event_process.event_history

                    if kind == 'baseline':
                        action = actor(torch.tensor(obs, dtype=torch.float32))
                    elif kind == 'intelligent':
                        action = intelligent_agent(torch.tensor(obs, dtype=torch.float32), history)
                    else:
                        history = utils.get_fixed_length_vector(history, event_horizon).flatten()
                        action = naive_agent(torch.tensor(np.concatenate([obs, history]), dtype=torch.float32))

                    obs_dict, reward, terminated, truncated, _ = env.step(action.numpy())
                    total += reward

                    if terminated or truncated:
                        break

                returns.append(total)

            writer.writerow([episode, *returns])
            f.flush()
            print(episode, *returns, flush=True)

    env.close()
