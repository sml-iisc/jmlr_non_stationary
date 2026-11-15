import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.environment import BlockData
from src.environment.observation import RoutingObservation
from src.evaluation.dynamite import evaluate
from src.evaluation.metrics import summarize
from src.models.dynamite_agent import PaperAgent
from src.models.config import PAPER_CONFIG
from src.paths import PROJECT_ROOT
from src.training.objectives import update
from src.training.replay import EpisodeReplay
from src.training.rollouts import collect
from src.utils.io import write_csv as save_rows
from src.utils.io import write_json
from src.utils.provenance import experiment_sources


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(904000, 904016))
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    return parser, args


def run_training(parser, args):
    if args.steps <= 0 or args.batch_episodes <= 0 or args.eval_interval <= 0:
        parser.error("budgets must be positive")

    if set(args.eval_seeds) & set(range(900000, 900032)):
        parser.error("event seed overlap")

    cfg = dict(PAPER_CONFIG)

    if args.smoke:
        args.steps = 512
        args.batch_episodes = 4
        args.eval_interval = 512
        args.eval_seeds = [904000]
        cfg["n_epochs"] = 1

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    data = BlockData(args.out / "cache")
    agent = PaperAgent(data, args.device)
    optp = torch.optim.Adam(
        agent.policy.parameters(), lr=cfg["learning_rate"], eps=cfg["adam_eps"]
    )
    optv = torch.optim.Adam(
        agent.belief.parameters(), lr=cfg["vae_learning_rate"], eps=cfg["adam_eps"]
    )
    replay = EpisodeReplay(cfg["replay_capacity"], args.seed)
    run = args.out / f"seed{args.seed}" / "dynamite"
    run.mkdir(parents=True)
    write_experiment_metadata(args, data, cfg)
    envs = [
        RoutingObservation(data, worker_seed=args.seed * 1000 + i)
        for i in range(args.batch_episodes)
    ]
    rows = evaluate(agent, data, args.eval_seeds)
    save_rows(run / "episodes_initial.csv", rows)
    curve = [dict(actual_timesteps=0, **summarize(rows))]
    rows = evaluate(agent, data, args.eval_seeds, shortest_path=True)
    save_rows(args.out / "shortest_path_episodes.csv", rows)
    write_json(args.out / "shortest_path.json", summarize(rows))
    steps = 0
    iteration = 0
    next_eval = args.eval_interval
    baseline = 0.0
    started = time.time()
    print("Training paper-based DynaMITE", args.seed, args.steps, flush=True)

    with (run / "training.jsonl").open("w") as log:
        while steps < args.steps:
            batch = collect(agent, envs)
            replay.add(batch)
            steps += int(batch["valid"].sum())
            iteration += 1
            sample = replay.sample(cfg["replay_batch"], agent.device)
            baseline, metrics = update(agent, batch, sample, optp, optv, baseline, cfg)
            metrics.update(
                steps=steps,
                iteration=iteration,
                seconds=time.time() - started,
                score_baseline=baseline,
            )
            log.write(json.dumps(metrics) + "\n")
            log.flush()

            if iteration % 20 == 0:
                print(
                    "paper_dynamite",
                    steps,
                    "fps",
                    round(steps / (time.time() - started)),
                    flush=True,
                )

            if steps >= next_eval or steps >= args.steps:
                rows = evaluate(agent, data, args.eval_seeds)
                save_rows(run / f"episodes_{steps:09d}.csv", rows)
                result = dict(actual_timesteps=steps, **summarize(rows))
                curve.append(result)
                torch.save(
                    dict(
                        agent=agent.state_dict(),
                        policy_optimizer=optp.state_dict(),
                        vae_optimizer=optv.state_dict(),
                        steps=steps,
                        seed=args.seed,
                        method="dynamite_paper",
                        config=cfg,
                        score_baseline=baseline,
                        rng_torch=torch.get_rng_state(),
                        note="Inference checkpoint; replay/env states not saved for exact resume.",
                    ),
                    run / f"checkpoint_{steps:09d}.pt",
                )
                write_json(run / "curve.json", curve)
                print(result, flush=True)
                next_eval = (steps // args.eval_interval + 1) * args.eval_interval

    rows = evaluate(agent, data, args.eval_seeds, stochastic=True)
    save_rows(run / "episodes_stochastic.csv", rows)
    write_json(run / "stochastic_result.json", summarize(rows))
    result = dict(
        method="dynamite_paper",
        seed=args.seed,
        requested_timesteps=args.steps,
        training_seconds=time.time() - started,
        **curve[-1],
    )
    write_json(run / "result.json", result)
    write_json(args.out / "summary.json", [result])

    for env in envs:
        env.close()

    print(result, flush=True)


def write_experiment_metadata(args, data, cfg):
    files = experiment_sources(PROJECT_ROOT)
    write_json(
        args.out / "experiment.json",
        dict(
            algorithm="paper_based_hard_sessions",
            arguments=dict(
                out=str(args.out),
                steps=args.steps,
                policy_seeds=[args.seed],
                methods=["dynamite"],
                device=args.device,
                batch_episodes=args.batch_episodes,
                eval_seeds=args.eval_seeds,
                smoke=args.smoke,
            ),
            paper_config=cfg,
            ppo={
                k: cfg[k]
                for k in [
                    "learning_rate",
                    "gamma",
                    "gae_lambda",
                    "clip_range",
                    "vf_coef",
                    "ent_coef",
                    "max_grad_norm",
                    "adam_eps",
                    "n_epochs",
                    "batch_size",
                ]
            },
            training_event_seeds=list(range(900000, 900032)),
            environment=data.config,
            evaluation="Plot: deterministic MAP boundaries, posterior means, greedy actions. Also report Algorithm-2 sampled inference/actions separately.",
            implementation_status="Paper-based implementation with explicit numerical choices; not verified against author code.",
            versions=dict(torch=torch.__version__, numpy=np.__version__),
            source_hashes={
                f: hashlib.sha256((PROJECT_ROOT / f).read_bytes()).hexdigest()
                for f in files
            },
        ),
    )


def main():
    parser, args = parse_arguments()
    run_training(parser, args)


if __name__ == "__main__":
    main()
