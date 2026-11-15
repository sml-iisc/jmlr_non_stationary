import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.paths import PROJECT_ROOT, subprocess_environment


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1021)))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["dynamite", "H0", "H4", "H16", "H128"],
        default=["dynamite", "H0", "H4", "H16", "H128"],
    )
    parser.add_argument("--steps", type=int, default=4000000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.steps < 250000 or args.steps % 250000 or args.workers < 1:
        parser.error(
            "steps must be a positive multiple of 250000; workers must be positive"
        )

    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(
        args.methods
    ):
        parser.error("Duplicate seeds/methods are not allowed")

    return args


def run_training(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    (out / "runs").mkdir()
    env = dict(
        subprocess_environment(),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    jobs = create_jobs(args, out)
    write_manifest(out, args, jobs)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute_job, j, out, env): j for j in jobs}

        for future in as_completed(futures):
            job = futures[future]
            job["returncode"] = future.result()
            job["state"] = "complete" if job["returncode"] == 0 else "failed"
            write_manifest(out, args, jobs)
            print(job["seed"], job["method"], job["state"], flush=True)

    if any((j["state"] == "failed" for j in jobs)):
        raise SystemExit(
            "Some training jobs failed; inspect logs. Outputs have been preserved."
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate",
            "aggregate",
            "--runs",
            str(out / "runs"),
            "--out",
            str(out / "curve_summary.csv"),
            "--steps",
            str(args.steps),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def create_jobs(args, out):
    jobs = []

    for seed in args.seeds:
        for method in args.methods:
            output = out / "runs" / f"seed{seed}" / method
            command = [sys.executable, "-u"]

            if method == "dynamite":
                command += [
                    "-m",
                    "src.training.train_dynamite",
                    "--out",
                    str(output),
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--batch-episodes",
                    "16",
                    "--eval-interval",
                    "250000",
                ]
            else:
                command += [
                    "-m",
                    "src.training.train_history_ppo",
                    "--out",
                    str(output),
                    "--histories",
                    method[1:],
                    "--policy-seeds",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--n-envs",
                    "4",
                    "--checkpoint-interval",
                    "250000",
                    "--eval-start",
                    "250000",
                ]

            command += [
                "--device",
                "cpu",
                "--eval-seeds",
                *map(str, range(904000, 904016)),
            ]
            jobs.append(dict(seed=seed, method=method, command=command, state="queued"))

    return jobs


def write_manifest(out, args, jobs):
    temp = out / "training_manifest.tmp"
    temp.write_text(
        json.dumps(
            dict(steps=args.steps, seeds=args.seeds, methods=args.methods, jobs=jobs),
            indent=2,
        )
        + "\n"
    )
    temp.replace(out / "training_manifest.json")


def execute_job(job, out, env):
    with (out / "logs" / f"seed{job['seed']}_{job['method']}.log").open("w") as log:
        return subprocess.run(
            job["command"],
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def main():
    run_training(parse_arguments())


if __name__ == "__main__":
    main()
