"""Launch the LILAC and PPO runs used in the comparison figure."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.paths import PROJECT_ROOT, subprocess_environment

METHODS = ("ppo_h0", "ppo_h4", "ppo_h16", "ppo_h128", "lilac")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--ppo-device", default="cpu")
    parser.add_argument(
        "--lilac-device",
        default="cpu",
        help="Use cuda to match the original LILAC runs",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a short execution check"
    )

    return parser.parse_args(argv)


def training_command(args, method, seed):
    is_lilac = method == "lilac"
    module = "src.training.train_lilac" if is_lilac else "src.training.train_history_ppo"
    steps = 256 if args.smoke else args.steps
    command = [
        sys.executable,
        "-m",
        module,
        "--out",
        str(args.out / f"{method}_seed{seed}"),
        "--seed",
        str(seed),
        "--steps",
        str(steps),
    ]

    if is_lilac:
        command.extend(
            [
                "--device",
                args.lilac_device,
                "--config-json",
                str(PROJECT_ROOT / "configs/tuned_lilac.json"),
            ]
        )
    else:
        command.extend(
            ["--device", args.ppo_device, "--history", method[len("ppo_h") :]]
        )

    if args.smoke:
        command.extend(
            ["--smoke", "--checkpoint-interval", "0", "--eval-seeds", "902000"]
        )

    return command


def run_training(args):
    methods = METHODS if args.method == "all" else [args.method]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    process_env = subprocess_environment()
    process_env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )

    for method in methods:
        for seed in args.seeds:
            output = args.out / f"{method}_seed{seed}"

            if output.exists():
                raise FileExistsError(
                    f"{output} exists; choose another --out to avoid overwriting"
                )

            command = training_command(args, method, seed)
            print(" ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, env=process_env, check=True)


def main():
    run_training(parse_arguments())


if __name__ == "__main__":
    main()
