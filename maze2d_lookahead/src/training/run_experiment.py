"""Train a stationary actor or a lookahead transition model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from src.paths import CONFIG_DIR, PROJECT_ROOT, subprocess_environment

ACTOR_FILENAME = "actor_final.pth"

METHODS = ("actor", "intelligent", "naive")


def main():
    settings = json.loads((CONFIG_DIR / "training.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--episodes", type=int, default=settings["episodes"])
    parser.add_argument("--eval-episodes", type=int, default=settings["eval_episodes"])
    parser.add_argument("--eval-interval", type=int, default=settings["eval_interval"])
    parser.add_argument("--seed", type=int, default=settings["seed"])
    parser.add_argument("--threads", type=int, default=settings["threads"])
    parser.add_argument("--depth", type=int, default=settings["depth"])
    parser.add_argument("--trajectories", type=int, default=settings["trajectories"])
    parser.add_argument("--actor", type=Path, help="Optional stationary actor checkpoint")
    args = parser.parse_args()
    counts = [args.episodes, args.eval_episodes, args.eval_interval, args.threads, args.depth, args.trajectories]

    if min(counts) < 1:
        parser.error("Episode counts, intervals, threads, and lookahead sizes must be positive")

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    actor = args.actor.resolve() if args.actor else None
    methods = METHODS if args.method == "all" else [args.method]

    for method in methods:
        method_dir = args.out / method
        log_path = method_dir / "training.log"

        if log_path.exists():
            raise FileExistsError(f"{log_path} exists; choose a fresh --out directory")

        method_dir.mkdir(parents=True, exist_ok=True)
        env = subprocess_environment()
        env.update(
            RUN_DIR=str(args.out),
            TRAIN_EPISODES=str(args.episodes),
            EVAL_EPISODES=str(args.eval_episodes),
            EVAL_INTERVAL=str(args.eval_interval),
            SEED=str(args.seed),
            TORCH_NUM_THREADS=str(args.threads),
            LOOKAHEAD_DEPTH=str(args.depth),
            LOOKAHEAD_TRAJECTORIES=str(args.trajectories),
            MPLBACKEND="Agg",
        )

        if actor is not None:
            env["ACTOR_CHECKPOINT"] = str(actor)

        command = [sys.executable, "-u", "-m", "src.training.train_" + method]

        with log_path.open("w", buffering=1) as log:
            with subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)

                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)

        if method == "actor" and args.method == "all" and actor is None:
            actor = method_dir / ACTOR_FILENAME
