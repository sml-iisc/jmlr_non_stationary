import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src import environment
from src.environment import BlockData
from src.environment.observation import LilacRouteEnv
from src.evaluation.metrics import summarize
from src.evaluation.sequential import evaluate_sequential
from src.models.config import PaperConfig
from src.models.lilac_agent import PaperLilacAgent
from src.paths import PROJECT_ROOT
from src.training.checkpoints import restore_resume, save_resume
from src.utils.io import write_csv, write_json
from src.utils.provenance import experiment_sources

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class TrainingState:
    agent: PaperLilacAgent
    env: LilacRouteEnv
    start_step: int
    trajectory: list
    observation: np.ndarray
    context: tuple
    latent: np.ndarray
    curve: list


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4000000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=250000)
    parser.add_argument("--eval-start", type=int, default=1000000)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(903000, 903016))
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--config-json",
        type=Path,
        help="JSON PaperConfig overrides; defaults preserve the original experiment",
    )
    args = parser.parse_args()

    return parser, args


def run_training(parser, args):
    if args.steps < 1 or min(args.checkpoint_interval, args.eval_start) < 0:
        parser.error("invalid training/checkpoint budget")

    if set(args.eval_seeds) & set(environment.TRAIN_SEEDS):
        parser.error("training and evaluation seeds overlap")

    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config, warmup = configure_agent(parser, args)
    prepare_output(parser, args, config)
    data = BlockData(args.out / "cache")

    for seed in environment.TRAIN_SEEDS + args.eval_seeds:
        data.schedule(seed)

    state = initialize_training(args, data, config)
    agent, env = state.agent, state.env
    started = time.monotonic()
    latest = {}
    fields = [
        "timesteps",
        "updates",
        "episodes",
        "seconds",
        "q_loss",
        "actor_loss",
        "state_loss",
        "reward_loss",
        "kl_loss",
        "prior_loss",
        "alpha",
        "entropy",
        "beta",
    ]
    mode = "a" if args.resume else "w"
    stream = (args.out / "training.csv").open(mode, newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)

    if not args.resume:
        writer.writeheader()

    try:
        for step in range(state.start_step + 1, args.steps + 1):
            if step <= warmup:
                action = int(
                    agent.rng.choice(
                        np.flatnonzero(data.successors[state.observation[0]] >= 0)
                    )
                )
            else:
                action = agent.act(state.observation, state.latent)

            next_observation, reward, done, truncated, _ = env.step(action)
            state.trajectory.append(
                (
                    state.observation.copy(),
                    action,
                    reward,
                    next_observation.copy(),
                    bool(done or truncated),
                )
            )
            state.observation = next_observation

            if done or truncated:
                agent.finish_episode(state.trajectory)
                update_count = min(len(state.trajectory), max(0, step - warmup + 1))

                for _ in range(update_count):
                    latest = agent.update()

                embedding = agent.episode_embedding(state.trajectory)
                state.context = agent.advance_prior(embedding, state.context[1])
                state.latent = agent.sample_prior(state.context[0]).cpu().numpy()
                state.trajectory = []
                state.observation, _ = env.reset()

            if step % 1000 == 0:
                status = dict(
                    timesteps=step,
                    updates=agent.updates,
                    episodes=len(agent.cache),
                    seconds=time.monotonic() - started,
                    **latest,
                )
                writer.writerow(status)
                stream.flush()
                write_json(
                    args.out / "status.json",
                    dict(status="training", requested_timesteps=args.steps, **status),
                )
                print(json.dumps(status), flush=True)

            if (
                args.checkpoint_interval
                and step % args.checkpoint_interval == 0
                and (step < args.steps)
            ):
                agent.save(args.out / f"checkpoint_{step:09d}.pt", step)

                if step >= args.eval_start:
                    rows = evaluate_sequential(agent, data, args.eval_seeds)
                    write_csv(args.out / f"episodes_{step:09d}.csv", rows)
                    state.curve.append(dict(timesteps=step, **summarize(rows)))
                    write_json(args.out / "curve.json", state.curve)

                save_resume(
                    args.out / "latest_resume.pt",
                    agent,
                    step,
                    env,
                    state.trajectory,
                    state.observation,
                    state.context,
                    state.latent,
                    state.curve,
                )

        agent.save(args.out / "lilac_final.pt", args.steps)
        save_resume(
            args.out / "latest_resume.pt",
            agent,
            args.steps,
            env,
            state.trajectory,
            state.observation,
            state.context,
            state.latent,
            state.curve,
        )
        write_json(
            args.out / "status.json", dict(status="evaluating", timesteps=args.steps)
        )
        rows = evaluate_sequential(agent, data, args.eval_seeds)
        write_csv(args.out / "episodes.csv", rows)
        result = dict(
            seed=args.seed,
            actual_timesteps=args.steps,
            updates=agent.updates,
            **summarize(rows),
            seconds=time.monotonic() - started,
            smoke=args.smoke,
        )
        write_json(args.out / "result.json", result)
        state.curve.append(dict(timesteps=args.steps, **summarize(rows)))
        write_json(args.out / "curve.json", state.curve)
        write_json(
            args.out / "status.json", dict(status="complete", timesteps=args.steps)
        )
        print(json.dumps(result), flush=True)
    finally:
        stream.close()
        env.close()


def configure_agent(parser, args):
    config = PaperConfig()
    warmup = 1000

    if args.smoke:
        config = PaperConfig(
            hidden=32, task_batch=4, pretrain_updates=8, prior_sequence=8
        )
        warmup = 64

    if args.config_json:
        overrides = json.loads(args.config_json.read_text())
        unknown = set(overrides) - set(asdict(config))

        if unknown:
            parser.error(f"unknown PaperConfig fields: {sorted(unknown)}")

        settings = asdict(config)
        settings.update(overrides)
        config = PaperConfig(**settings)

        if not (
            0 < config.gamma <= 1
            and 0 < config.tau <= 1
            and (0 <= config.entropy_fraction <= 1)
        ):
            parser.error("invalid discount, target update rate or entropy fraction")

        if (
            min(
                config.lr,
                config.encoder_lr,
                config.decoder_lr,
                config.prior_lr,
                config.prior_sigma,
            )
            <= 0
        ):
            parser.error("learning rates and prior sigma must be positive")

    return config, warmup


def prepare_output(parser, args, config):
    root = PROJECT_ROOT
    sources = experiment_sources(PROJECT_ROOT)
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }

    if args.resume:
        metadata = json.loads((args.out / "experiment.json").read_text())

        if (
            metadata["source_sha256"] != hashes
            or metadata["steps"] != args.steps
            or metadata["eval_seeds"] != args.eval_seeds
            or (metadata["policy_seed"] != args.seed)
            or (metadata["device"] != args.device)
            or (metadata["config"] != asdict(config))
        ):
            parser.error("resume source hashes/budget/evaluation seeds differ")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        metadata = dict(
            algorithm="LILAC paper objective, categorical SAC",
            steps=args.steps,
            policy_seed=args.seed,
            config=asdict(config),
            device=args.device,
            train_seeds=environment.TRAIN_SEEDS,
            eval_seeds=args.eval_seeds,
            observation=["current_node", "goal_node"],
            history=0,
            evaluation="sequential posterior-to-prior inference, no weight updates",
            environment_reset="original event-seed and departure resampling at each trip",
            smoke=args.smoke,
            source_sha256=hashes,
        )
        write_json(args.out / "experiment.json", metadata)

        for name in sources:
            dest = args.out / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)


def initialize_training(args, data, config):
    if args.resume:
        agent, env, saved = restore_resume(
            args.out / "latest_resume.pt", data, args.device, args.seed
        )
        return TrainingState(
            agent=agent,
            env=env,
            start_step=saved["step"],
            trajectory=saved["trajectory"],
            observation=saved["observation"],
            context=saved["context"],
            latent=saved["latent"],
            curve=saved["curve"],
        )

    agent = PaperLilacAgent(data.successors, config, args.device, args.seed)
    env = LilacRouteEnv(data, worker_seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    context = agent.initial_context()

    return TrainingState(
        agent=agent,
        env=env,
        start_step=0,
        trajectory=[],
        observation=observation,
        context=context,
        latent=context[0].copy(),
        curve=[],
    )


def main():
    parser, args = parse_arguments()
    run_training(parser, args)


if __name__ == "__main__":
    main()
