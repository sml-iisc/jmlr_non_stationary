import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

from src import environment
from src.environment import BlockData, BlockHistoryEnv
from src.evaluation.episodes import evaluate
from src.models.history_ppo import MaskedSB3Policy, policy_kwargs
from src.training.callbacks import SaveCheckpoints
from src.training.scheduled_ppo import ScheduledPPO
from src.training.schedules import schedule_value
from src.utils.io import write_json
from src.paths import CONFIG_DIR


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Candidate event parameters; default is the frozen benchmark configuration",
    )
    parser.add_argument(
        "--ppo-config", type=Path, default=CONFIG_DIR / "ppo_config.json"
    )
    parser.add_argument("--histories", type=int, nargs="+", default=[64])
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[1001])
    parser.add_argument("--steps", type=int, default=4000000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(903000, 903016))
    )
    parser.add_argument("--train-seed-start", type=int, default=900000)
    parser.add_argument("--train-seed-count", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=250000)
    parser.add_argument("--eval-start", type=int, default=1000000)
    args = parser.parse_args()

    return parser, args


def run_training(parser, args):
    if (
        args.steps <= 0
        or args.n_envs <= 0
        or args.train_seed_count <= 0
        or (args.checkpoint_interval < 0)
    ):
        parser.error(
            "Steps, n-envs and train-seed-count must be positive; checkpoint-interval must be nonnegative"
        )

    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable; select --device cpu")

    environment.TRAIN_SEEDS = list(
        range(args.train_seed_start, args.train_seed_start + args.train_seed_count)
    )

    if len(set(args.histories)) != len(args.histories) or any(
        h < 0 or h > 128 for h in args.histories
    ):
        parser.error("History lengths must be unique integers between 0 and 128")

    if args.smoke:
        args.steps, args.n_envs, args.eval_seeds = (512, 1, [902000])
        environment.TRAIN_SEEDS = [900000, 900001]

    if set(environment.TRAIN_SEEDS) & set(args.eval_seeds):
        parser.error("Training and evaluation process seeds must be disjoint")

    torch.set_num_threads(1)
    config = json.loads(args.config.read_text()) if args.config else None
    ppo_config = json.loads(args.ppo_config.read_text())

    if args.smoke:
        ppo_config.update(n_steps=64, batch_size=64, n_epochs=1)

    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out / "cache"
    cache.mkdir()
    data = BlockData(cache, config=config)
    write_json(
        args.out / "experiment.json",
        dict(
            behavior="wait",
            smoke=args.smoke,
            steps=args.steps,
            n_envs=args.n_envs,
            device=args.device,
            histories=args.histories,
            policy_seeds=args.policy_seeds,
            train_process_seeds=environment.TRAIN_SEEDS,
            evaluation_process_seeds=args.eval_seeds,
            environment=data.config,
            ppo=ppo_config,
        ),
    )
    summaries = []

    with threadpool_limits(limits=1):
        for seed in args.policy_seeds:
            hashes = set()

            for history in args.histories:
                hashes.add(
                    train_policy(args, data, ppo_config, seed, history, summaries)
                )

            assert (
                len(hashes) == 1
            ), "Histories must share initial model weights for a matched policy seed"


def build_policy(args, data, ppo_config, seed, history, run):
    env = DummyVecEnv(
        [
            lambda i=i: Monitor(
                BlockHistoryEnv(data, history, worker_seed=seed * 1000 + i)
            )
            for i in range(args.n_envs)
        ]
    )
    kwargs = {
        k: v
        for k, v in ppo_config.items()
        if k not in ["id", "rationale", "lr_schedule", "entropy_schedule"]
    }
    kwargs["learning_rate"] = lambda remaining: schedule_value(
        ppo_config["learning_rate"], ppo_config["lr_schedule"], remaining, 0.1
    )
    model = ScheduledPPO(
        MaskedSB3Policy,
        env,
        seed=seed,
        device=args.device,
        policy_kwargs=policy_kwargs(data),
        verbose=0,
        **kwargs
    )
    model.entropy_start = ppo_config["ent_coef"]
    model.entropy_schedule = ppo_config["entropy_schedule"]
    model.set_logger(configure(str(run / "sb3"), ["csv"]))
    initial = hashlib.sha256(
        b"".join(
            (w.detach().cpu().numpy().tobytes() for w in model.policy.parameters())
        )
    ).hexdigest()

    return env, model, initial


def train_policy(args, data, ppo_config, seed, history, summaries):
    run = args.out / ("seed%s" % seed) / ("H%s" % history)
    run.mkdir(parents=True)
    env, model, initial = build_policy(args, data, ppo_config, seed, history, run)
    print("Training seed=%s H=%s steps=%s" % (seed, history, args.steps), flush=True)
    started = time.time()
    checkpoints = []
    callback = SaveCheckpoints(args, run, checkpoints, started)

    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        callback.save_due()
        model.save(run / "ppo_final.zip")
        rows = evaluate(model, data, history, args.eval_seeds, BlockHistoryEnv)

        with (run / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        summary = dict(
            seed=seed,
            history=history,
            mean_penalized_time_s=float(np.mean([r["penalized_time_s"] for r in rows])),
            success_rate=float(np.mean([r["success"] for r in rows])),
            actual_timesteps=model.num_timesteps,
            initial_policy_sha256=initial,
            training_and_evaluation_seconds=time.time() - started,
        )
        write_json(run / "result.json", summary)
        summaries.append(summary)
        curve = []

        for checkpoint in checkpoints:
            if checkpoint["target_timesteps"] < args.eval_start:
                continue

            saved = ScheduledPPO.load(checkpoint["path"], device=args.device)
            episodes = evaluate(saved, data, history, args.eval_seeds, BlockHistoryEnv)

            with (run / ("episodes_%09d.csv" % checkpoint["target_timesteps"])).open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
                writer.writeheader()
                writer.writerows(episodes)

            curve.append(
                dict(
                    target_timesteps=checkpoint["target_timesteps"],
                    actual_timesteps=checkpoint["actual_timesteps"],
                    mean_penalized_time_s=float(
                        np.mean([r["penalized_time_s"] for r in episodes])
                    ),
                    success_rate=float(np.mean([r["success"] for r in episodes])),
                )
            )
            del saved

        write_json(run / "curve.json", curve)
        write_json(args.out / "summary.json", summaries)
        print(summary, flush=True)
    finally:
        model.logger.close()
        env.close()

    return initial


def main():
    parser, args = parse_arguments()
    run_training(parser, args)


if __name__ == "__main__":
    main()
