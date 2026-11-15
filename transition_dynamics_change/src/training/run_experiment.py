import argparse
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
import torch

from src.environment.split import initialize_data
from src.paths import PROJECT_ROOT, CONFIG_DIR, subprocess_environment
from src.plotting.summary import plot_summary
from src.training.trial import run_trial
from src.utils.io import dump, write_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "ppo_config.json")
    parser.add_argument(
        "--histories", nargs="+", type=int, default=[0, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(901, 921)))
    parser.add_argument("--steps", type=int, default=4000000)
    parser.add_argument("--checkpoint-interval", type=int, default=256000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--wandb-project", default="town03-history-ablation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    return parser, args


def run_training(parser, args):
    if (
        len(set(args.histories)) != len(args.histories)
        or any(h < 0 or h > 128 for h in args.histories)
        or len(set(args.seeds)) != len(args.seeds)
        or (args.steps < 1)
    ):
        parser.error(
            "Require unique histories in 0..128, unique seeds, and positive steps"
        )

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    args.out, args.config = args.out.resolve(), args.config.resolve()
    params, plan = prepare_plan(parser, args)

    if args.check_only:
        print(json.dumps(plan, indent=2))
        return

    if args.out.exists() and (not args.resume):
        raise FileExistsError("Output exists; use --resume to skip completed runs")

    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "ablation_plan.json"

    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Resume plan differs from saved experiment")

    dump(plan_path, plan)

    if args.wandb_mode == "online":
        import wandb

        wandb.login()

    data, dev_pairs = initialize_data(args.out, args)
    data.history_slots = 128
    dump(
        args.out / "data_split.json",
        dict(
            train_pairs=data.train_pairs,
            development_pairs=dev_pairs,
            test_pairs=data.test_pairs,
        ),
    )
    runs = train_histories(args, data, dev_pairs, params)
    summary = summarize_runs(args, runs)
    write_rows(args.out / "summary.csv", summary)
    dump(args.out / "summary.json", summary)
    plot_summary(summary, args)
    dump(
        args.out / "status.json",
        dict(stage="checkpoint_test_evaluation", completed=len(runs), total=len(runs)),
    )
    from src.evaluation.evaluate import run_evaluation

    run_evaluation(argparse.Namespace(out=args.out, device=args.device))
    dump(
        args.out / "status.json",
        dict(stage="complete", completed=len(runs), total=len(runs)),
    )
    (args.out / "COMPLETE").write_text(
        "All history ablation training and evaluations completed.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


def summarize_runs(args, runs):
    summary = []

    for h in args.histories:
        rows = [r for r in runs if r["history"] == h]
        scores = [r["test_penalized_mean_s"] for r in rows]
        summary.append(
            dict(
                history=h,
                training_seeds=len(rows),
                mean_penalized_time_s=float(np.mean(scores)),
                std_across_training_seeds_s=(
                    float(np.std(scores, ddof=1)) if len(rows) > 1 else 0.0
                ),
                success_rate=float(np.mean([r["test_success_rate"] for r in rows])),
            )
        )

    return summary


def prepare_plan(parser, args):
    settings = json.loads(args.config.read_text())
    args.split_seed = settings["split_seed"]
    args.dev_pairs = settings["development_pair_count"]
    args.eval_interval = settings["eval_interval"]
    params = settings["parameters"]

    if args.checkpoint_interval <= 0 or args.checkpoint_interval % (
        4 * params["n_steps"]
    ):
        parser.error(
            "Checkpoint interval must be a positive multiple of n_envs * n_steps"
        )

    plan = dict(
        histories=args.histories,
        seeds=args.seeds,
        requested_steps=args.steps,
        configuration=params,
        config_source=str(args.config),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        split_seed=args.split_seed,
        development_pair_count=args.dev_pairs,
        eval_interval=args.eval_interval,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        training_event_seeds=list(range(20000, 20032)),
        development_event_seeds=[910, 911, 912],
        test_event_seeds=[902, 903, 904, 905, 906],
        n_envs=4,
        distance_shaping=False,
        history_tensor_shape=[128, 278],
        checkpoint_selection="Minimum development penalized time, independently for each H and training seed",
        initialization="Every run independently initialized; no checkpoint loading",
        caveat="Hyperparameters selected at H64. Test set reused in earlier experiments; exploratory comparison, no untouched-test claim.",
    )

    return params, plan


def train_histories(args, data, dev_pairs, params):
    runs = []

    for seed in args.seeds:
        for history in args.histories:
            args.history = history
            name = f"H{history}_config_{params['id'].split('_')[-1]}_seed{seed}"
            dump(
                args.out / "status.json",
                dict(
                    stage="training",
                    history=history,
                    seed=seed,
                    completed=len(runs),
                    total=len(args.histories) * len(args.seeds),
                ),
            )
            result = run_trial(
                data,
                dev_pairs,
                params,
                seed,
                args.steps,
                args.out / name,
                args,
                final_test=True,
            )
            runs.append(dict(history=history, **result))
            write_rows(args.out / "runs.csv", runs)

        import subprocess
        import sys

        dump(
            args.out / "status.json",
            dict(
                stage="checkpoint_test_evaluation",
                seed=seed,
                completed=len(runs),
                total=len(args.histories) * len(args.seeds),
            ),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.evaluate",
                "--out",
                str(args.out),
                "--device",
                args.device,
            ],
            check=True,
            cwd=PROJECT_ROOT,
            env=subprocess_environment(),
        )

    return runs


def main():
    parser, args = parse_arguments()
    run_training(parser, args)


if __name__ == "__main__":
    main()
