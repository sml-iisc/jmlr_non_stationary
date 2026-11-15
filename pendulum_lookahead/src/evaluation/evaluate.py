"""Evaluate the stationary actor and optional trained lookahead models."""
import argparse
import csv
import json
from pathlib import Path
import random

from src.paths import CONFIG_DIR, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "intelligent", "naive"], default=["baseline", "intelligent", "naive"])
    parser.add_argument("--actor", type=Path, default=PROJECT_ROOT / "data/checkpoints/actor.pt")
    parser.add_argument("--intelligent", type=Path, default=PROJECT_ROOT / "runs/intelligent/transition_model.pt")
    parser.add_argument("--naive", type=Path, default=PROJECT_ROOT / "runs/naive/transition_model.pt")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--trajectories", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/evaluation_returns.csv")
    args = parser.parse_args()

    if min(args.episodes, args.depth, args.trajectories) < 1 or args.warmup < 0:
        parser.error("Episode/lookahead counts must be positive and warmup nonnegative")

    paths = [args.actor]
    paths.extend(getattr(args, method) for method in args.methods if method != "baseline")

    for path in paths:
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}. Train it first or select --methods baseline.")

    import gymnasium as gym
    from gymnasium.envs.registration import register
    import numpy as np
    import torch
    from src.models.networks import FeedforwardHybridNeuralNetModel
    from src.models import lookahead
    from src.environment.forecast import EventProcess
    from src.environment.features import reward_fn
    from src.utils.features import get_fixed_length_vector
    from src.utils.checkpoints import load_actor

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    params = json.loads((CONFIG_DIR / "environment.json").read_text())
    register(id="LookaheadPendulum-v0", entry_point="src.environment.pendulum:NonstationaryPendulumPolyFeaturesEnv", max_episode_steps=200)
    env = gym.make("LookaheadPendulum-v0", degree=1, event_params=params)
    actor = load_actor(args.actor)
    actor.eval()
    observation, _ = env.reset(seed=args.seed)
    obs_dim = len(observation) - 1
    horizon = 5
    state_dim = len(observation) + horizon
    action_dim = env.action_space.shape[0]
    agents = {}

    for method in args.methods:
        if method == "baseline":
            continue

        output_dim = obs_dim + 1 if method == "intelligent" else state_dim
        model = FeedforwardHybridNeuralNetModel(state_dim, action_dim, 256, output_dim)
        model.load_state_dict(torch.load(getattr(args, method), map_location="cpu", weights_only=True))
        model.eval()

        if method == "intelligent":
            agents[method] = lookahead.StationaryIntelligentLookaheadPolicy(
                env, obs_dim, 1, horizon, action_dim, actor, EventProcess(params=params),
                model, reward_fn, 0.2, 5, args.depth, args.trajectories, torch.device("cpu"))
        else:
            agents[method] = lookahead.StationaryNaiveLookaheadPolicy(
                env, obs_dim, 1, horizon, action_dim, actor, model, reward_fn, 0.2, 5, args.depth)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.output.open("w", newline="") as stream, torch.no_grad():
            writer = csv.writer(stream)
            writer.writerow(["episode", *args.methods])

            for episode in range(args.episodes):
                returns = []

                for method in args.methods:
                    random.seed(args.seed + episode)
                    np.random.seed(args.seed + episode)
                    torch.manual_seed(args.seed + episode)
                    observation, _ = env.reset(seed=args.seed + episode)
                    total = 0.0

                    for step in range(200):
                        tensor = torch.as_tensor(observation, dtype=torch.float32)
                        history = env.unwrapped.process.event_history

                        if method == "baseline":
                            action = actor(tensor[:obs_dim])
                        elif method == "intelligent":
                            action = agents[method](tensor, history)
                        else:
                            events = get_fixed_length_vector(history, horizon)
                            state = np.concatenate((observation, events))
                            action = agents[method](torch.as_tensor(state, dtype=torch.float32))

                        observation, reward, terminated, truncated, _ = env.step(action.numpy().copy())

                        if step + 1 > args.warmup:
                            total += reward

                        if terminated or truncated:
                            break

                    returns.append(total)

                writer.writerow([episode, *returns])
                stream.flush()
                print(episode, *returns, flush=True)
    finally:
        env.close()
