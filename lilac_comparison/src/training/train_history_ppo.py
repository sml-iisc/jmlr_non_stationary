"""Train one PPO history window to an exact transition budget."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from threadpoolctl import threadpool_limits

from src import environment
from src.environment import BlockData, BlockHistoryEnv
from src.evaluation.episodes import evaluate
from src.evaluation.metrics import summarize
from src.models.history_ppo import MaskedSB3Policy, policy_kwargs
from src.paths import PROJECT_ROOT, CONFIG_DIR
from src.training.callbacks import SaveCheckpoints
from src.training.exact_ppo import ExactBudgetPPO
from src.training.schedules import schedule_value
from src.utils.io import write_csv, write_json
from src.utils.provenance import experiment_sources


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--history", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=250_000)
    parser.add_argument("--eval-start", type=int, default=1_000_000)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(903000, 903016))
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.steps < 4 or args.steps % 4:
        parser.error("steps must be a positive multiple of four")

    if not 0 <= args.history <= environment.MAX_H:
        parser.error("history must be between 0 and 128")

    if min(args.checkpoint_interval, args.eval_start) < 0:
        parser.error("checkpoint-interval and eval-start must be nonnegative")

    if set(args.eval_seeds) & set(environment.TRAIN_SEEDS):
        parser.error("training and evaluation seeds overlap")

    return args


def prepare_output(args, config):
    args.out.mkdir(parents=True, exist_ok=False)
    sources = experiment_sources(PROJECT_ROOT)
    metadata = dict(
        algorithm=f"PPO H{args.history}",
        steps=args.steps,
        policy_seed=args.seed,
        history=args.history,
        n_envs=4,
        device=args.device,
        ppo=config,
        train_seeds=environment.TRAIN_SEEDS,
        eval_seeds=args.eval_seeds,
        exact_final_rollout=True,
        smoke=args.smoke,
        source_sha256={
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
    )
    write_json(args.out / "experiment.json", metadata)

    for name in sources:
        destination = args.out / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / name, destination)

    return metadata


def create_model(args, data, config):
    workers = [
        lambda worker=worker: Monitor(
            BlockHistoryEnv(data, args.history, worker_seed=args.seed * 1000 + worker)
        )
        for worker in range(4)
    ]
    vector_env = DummyVecEnv(workers)
    options = {
        key: value
        for key, value in config.items()
        if key not in ("id", "rationale", "lr_schedule", "entropy_schedule")
    }
    options["learning_rate"] = lambda remaining: schedule_value(
        config["learning_rate"], config["lr_schedule"], remaining, 0.1
    )
    model = ExactBudgetPPO(
        MaskedSB3Policy,
        vector_env,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs(data),
        verbose=0,
        **options,
    )
    model.entropy_start = config["ent_coef"]
    model.entropy_schedule = config["entropy_schedule"]

    return model, vector_env


def evaluate_checkpoints(args, data, checkpoints):
    curve = []

    for checkpoint in checkpoints:
        if checkpoint["target_timesteps"] < args.eval_start:
            continue

        if checkpoint["actual_timesteps"] == args.steps:
            continue

        saved = ExactBudgetPPO.load(checkpoint["path"], device=args.device)
        episodes = evaluate(saved, data, args.history, args.eval_seeds, BlockHistoryEnv)
        write_csv(
            args.out / f"episodes_{checkpoint['target_timesteps']:09d}.csv", episodes
        )
        curve.append(
            dict(timesteps=checkpoint["actual_timesteps"], **summarize(episodes))
        )
        write_json(args.out / "curve.json", curve)
        del saved

    return curve


def run_training(args):
    torch.set_num_threads(1)
    config = json.loads((CONFIG_DIR / "ppo_config.json").read_text())

    if args.smoke:
        config.update(n_steps=64, batch_size=64, n_epochs=1)

    metadata = prepare_output(args, config)
    data = BlockData(args.out / "cache")

    for seed in environment.TRAIN_SEEDS + args.eval_seeds:
        data.schedule(seed)

    model, vector_env = create_model(args, data, config)
    initial_hash = hashlib.sha256(
        b"".join(
            parameter.detach().cpu().numpy().tobytes()
            for parameter in model.policy.parameters()
        )
    ).hexdigest()
    metadata["initial_policy_sha256"] = initial_hash
    write_json(args.out / "experiment.json", metadata)
    model.set_logger(configure(str(args.out / "sb3"), ["csv"]))
    started = time.monotonic()
    checkpoints = []
    callback = SaveCheckpoints(args, args.out, checkpoints, started)

    with threadpool_limits(limits=1):
        try:
            model.learn(total_timesteps=args.steps, callback=callback)
            callback.save_due()

            assert model.num_timesteps == args.steps
            model.save(args.out / "ppo_final.zip")
            write_json(
                args.out / "status.json",
                dict(status="evaluating", timesteps=model.num_timesteps),
            )
            episodes = evaluate(
                model, data, args.history, args.eval_seeds, BlockHistoryEnv
            )
            write_csv(args.out / "episodes.csv", episodes)
            result = dict(
                seed=args.seed,
                history=args.history,
                initial_policy_sha256=initial_hash,
                actual_timesteps=model.num_timesteps,
                **summarize(episodes),
                seconds=time.monotonic() - started,
                smoke=args.smoke,
            )
            write_json(args.out / "result.json", result)
            curve = evaluate_checkpoints(args, data, checkpoints)
            curve.append(dict(timesteps=args.steps, **summarize(episodes)))
            write_json(args.out / "curve.json", curve)
            write_json(
                args.out / "status.json",
                dict(status="complete", timesteps=model.num_timesteps),
            )
            print(json.dumps(result), flush=True)
        finally:
            model.logger.close()
            vector_env.close()


def main():
    run_training(parse_arguments())


if __name__ == "__main__":
    main()
