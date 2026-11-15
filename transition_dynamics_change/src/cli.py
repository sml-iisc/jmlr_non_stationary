import argparse
import subprocess
import sys
from pathlib import Path

from src.paths import PROJECT_ROOT, CONFIG_DIR, subprocess_environment

CONFIG = CONFIG_DIR / "ppo_config.json"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "plot"])
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "runs/history_ablation"
    )
    parser.add_argument("--device", default="cuda")
    args, extra = parser.parse_known_args(argv)

    return (args, extra)


def command_for(args):
    base = [sys.executable]

    if args.action == "train":
        cmd = base + [
            "-m",
            "src.training.run_experiment",
            "--out",
            str(args.out),
            "--config",
            str(CONFIG),
            "--histories",
            "0",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
            "--wandb-mode",
            "disabled",
            "--device",
            args.device,
        ]
    elif args.action == "evaluate":
        cmd = base + [
            "-m",
            "src.evaluation.evaluate",
            "--out",
            str(args.out),
            "--device",
            args.device,
        ]
    elif args.action == "plot":
        cmd = base + ["-m", "src.plotting.learning_curves"]

    return cmd


def main(argv=None):
    args, extra = parse_arguments(argv)
    subprocess.run(command_for(args) + extra, check=True, cwd=PROJECT_ROOT, env=subprocess_environment())


if __name__ == "__main__":
    main()
