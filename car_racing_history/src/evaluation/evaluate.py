"""Evaluate a trained CarRacing policy using the training trajectory rules."""
import argparse
import json
from pathlib import Path
import tempfile

from src.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=["hist10", "hist15", "hist25", "stationary", "non_stationary"], required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--value", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/evaluation.json")
    parser.add_argument("--max-episode-steps", type=int, default=800)
    args = parser.parse_args()

    if min(args.episodes, args.num_envs, args.max_episode_steps) < 1:
        parser.error("Episode and environment counts must be positive")

    if not args.policy.is_file():
        parser.error(f"Policy checkpoint not found: {args.policy}")

    import gymnasium as gym
    import torch

    history = args.setting.startswith("hist")

    if history:
        from src.environment.event_driven import make_env
        from src.training.history_ppo import PolicyGradientAgent
    elif args.setting == "stationary":
        from src.environment.stationary import make_env
        from src.training.ppo import PolicyGradientAgent
    else:
        from src.environment.event_driven import make_env
        from src.training.ppo import PolicyGradientAgent

    torch.set_num_threads(2)
    env = gym.vector.AsyncVectorEnv([make_env("rgb_array", 4, args.max_episode_steps) for _ in range(args.num_envs)])

    try:
        with tempfile.TemporaryDirectory(prefix="car_racing_eval_") as directory:
            options = dict(nf=32, output_dim=env.action_space.nvec[0], device=args.device, model_dir=directory)

            if history:
                options["history_length"] = int(args.setting[4:])

            agent = PolicyGradientAgent(env.observation_space.shape[-1] * 4, **options)
            policy_weights = torch.load(args.policy, map_location=agent.device, weights_only=True)
            agent.policy_net.load_state_dict(policy_weights)

            if args.value:
                value_weights = torch.load(args.value, map_location=agent.device, weights_only=True)
                agent.value_net.load_state_dict(value_weights)

            mean, std = agent.evaluate(env, episodes=args.episodes, argmax=not args.stochastic)

        result = dict(setting=args.setting, episodes=args.episodes * args.num_envs,
                      mean_reward=float(mean), std_reward=float(std), greedy=not args.stochastic)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()
