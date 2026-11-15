"""Run the archived training code with explicit experiment settings."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import traceback
import json

from src.paths import PROJECT_ROOT, CONFIG_DIR


SETTINGS = ('hist10', 'hist15', 'hist25', 'non_stationary', 'stationary')


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(args, setting):
    import gymnasium as gym
    import torch

    torch.set_num_threads(args.threads)
    is_history = setting.startswith('hist')

    if is_history:
        from src.environment.event_driven import make_env
        from src.training.history_ppo import PolicyGradientAgent
    elif setting == 'stationary':
        from src.environment.stationary import make_env
        from src.training.ppo import PolicyGradientAgent
    else:
        from src.environment.event_driven import make_env
        from src.training.ppo import PolicyGradientAgent

    eval_count = args.eval_envs or (4 if is_history else 10)
    log_name = f'log_hist_{setting[4:]}.txt' if is_history else f'log_{setting}.txt'
    model_name = f'models_with_hist_{setting[4:]}' if is_history else f'models_{setting}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / model_name
    log_path = args.output_dir / log_name

    # Fresh runs never silently replace previous logs or checkpoints.
    if not args.overwrite and (log_path.exists() or model_dir.exists()):
        raise FileExistsError(f'Existing output for {setting}; choose a new --output-dir or use --overwrite')

    with log_path.open('w', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            envs = eval_envs = None

            try:
                print(f'Setting: {setting}; iterations: {args.iterations}; device: {args.device}', flush=True)
                envs = gym.vector.AsyncVectorEnv([make_env('rgb_array', 4, args.max_episode_steps) for _ in range(args.num_envs)])
                eval_envs = gym.vector.AsyncVectorEnv([make_env('rgb_array', 4, args.max_episode_steps) for _ in range(eval_count)])
                kwargs = dict(
                    gamma=0.99, nf=32, output_dim=envs.action_space.nvec[0],
                    model_dir=str(model_dir), device=args.device,
                )

                if is_history:
                    kwargs['history_length'] = int(setting[4:])

                agent = PolicyGradientAgent(envs.observation_space.shape[-1] * 4, **kwargs)
                print(f'Actual device: {agent.device}; training envs: {args.num_envs}; evaluation envs: {eval_count}', flush=True)
                agent.train(
                    envs, eval_envs, start_episode=0, num_episodes=args.iterations,
                    rollout_steps=args.rollout_steps, batch_size=args.batch_size,
                    epochs=args.epochs, eval_interval=args.eval_interval,
                )
                torch.save(agent.policy_net.state_dict(), model_dir / "policy_final.pth")
                torch.save(agent.value_net.state_dict(), model_dir / "value_final.pth")
                print('Training completed.', flush=True)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                if eval_envs is not None:
                    eval_envs.close()

                if envs is not None:
                    envs.close()


def main():
    settings = json.loads((CONFIG_DIR / "training.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=(*SETTINGS, 'all'), required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--iterations', type=int, default=settings['iterations'])
    parser.add_argument('--num-envs', type=int, default=settings['num_envs'])
    parser.add_argument('--eval-envs', type=int, default=settings['eval_envs'] or None)
    parser.add_argument('--device', default=settings['device'])
    parser.add_argument('--threads', type=int, default=settings['threads'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--rollout-steps', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=800)
    args = parser.parse_args()

    if min(args.iterations, args.num_envs, args.threads, args.rollout_steps, args.batch_size, args.epochs, args.eval_interval, args.max_episode_steps) < 1 or (args.eval_envs is not None and args.eval_envs < 1):
        parser.error('Iterations, environment counts, and threads must be positive')

    for setting in SETTINGS if args.setting == 'all' else (args.setting,):
        run(args, setting)


if __name__ == '__main__':
    main()
