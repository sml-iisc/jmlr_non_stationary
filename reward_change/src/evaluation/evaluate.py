import argparse
import json
import os
import shutil
import statistics
from pathlib import Path

from src.evaluation.episodes import evaluate
from src.plotting.learning_curves import HERE, HISTORIES, render
from src.utils.io import write_csv


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runs", type=Path, help="Output directory from train.py")
    mode.add_argument(
        "--reference",
        action="store_true",
        help="Redraw using supplied original aggregate results",
    )
    parser.add_argument("--out", type=Path, default=HERE / "plots")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-start", type=int, default=1000000)
    args = parser.parse_args()

    return args


def run_evaluation(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/reward_change_matplotlib")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.reference:
        result_files = (
            "mean_evaluation_curves.csv",
            "mean_evaluation_success_curves.csv",
        )

        for name in result_files:
            destination = args.out / name

            if destination.resolve() != (HERE / "results" / name).resolve():
                shutil.copyfile(HERE / "results" / name, destination)
    else:
        import torch
        from threadpoolctl import threadpool_limits

        from src.environment import BlockData, BlockHistoryEnv
        from src.training.scheduled_ppo import ScheduledPPO

        torch.set_num_threads(1)
        experiment = json.loads((args.runs / "experiment.json").read_text())

        if experiment["histories"] != HISTORIES:
            print("Using requested history subset:", experiment["histories"])

        seeds = experiment["policy_seeds"]
        histories = experiment["histories"]
        curves = {}
        data = BlockData(args.runs / "cache", config=experiment["environment"])

        with threadpool_limits(limits=1):
            for seed in seeds:
                for history in histories:
                    run = args.runs / f"seed{seed}" / f"H{history}"
                    index = json.loads((run / "checkpoint_index.json").read_text())
                    curve = []

                    for checkpoint in index:
                        target = checkpoint["target_timesteps"]

                        if target < args.eval_start:
                            continue

                        print(
                            f"Evaluating seed={seed} H={history} step={target}",
                            flush=True,
                        )
                        model = ScheduledPPO.load(
                            run / Path(checkpoint["path"]).name, device=args.device
                        )
                        episodes = evaluate(
                            model,
                            data,
                            history,
                            experiment["evaluation_process_seeds"],
                            BlockHistoryEnv,
                        )
                        write_csv(run / f"episodes_{target:09d}.csv", episodes)
                        curve.append(
                            dict(
                                target_timesteps=target,
                                actual_timesteps=checkpoint["actual_timesteps"],
                                mean_penalized_time_s=statistics.mean(
                                    (r["penalized_time_s"] for r in episodes)
                                ),
                                success_rate=statistics.mean(
                                    (r["success"] for r in episodes)
                                ),
                            )
                        )
                        del model

                    if not curve:
                        raise ValueError(
                            f"No checkpoints at or above --eval-start for {run}"
                        )

                    write_csv(run / "evaluation_curve.csv", curve)
                    curves[seed, history] = {r["target_timesteps"]: r for r in curve}

        times, successes = aggregate_curves(histories, seeds, curves)
        write_csv(args.out / "mean_evaluation_curves.csv", times)
        write_csv(args.out / "mean_evaluation_success_curves.csv", successes)

    render(args)


def aggregate_curves(histories, seeds, curves):
    times, successes = ([], [])

    for history in histories:
        targets = sorted(curves[seeds[0], history])

        if any(sorted(curves[s, history]) != targets for s in seeds):
            raise ValueError("All seeds must have matching checkpoint targets")

        for target in targets:
            values = [
                curves[s, history][target]["mean_penalized_time_s"] for s in seeds
            ]
            common = dict(
                history=history, target_timesteps=target, completed_seeds=len(seeds)
            )
            times.append(
                dict(
                    common,
                    mean_travel_time_s=statistics.mean(values),
                    standard_error_s=(
                        statistics.stdev(values) / len(values) ** 0.5
                        if len(values) > 1
                        else 0.0
                    ),
                )
            )
            successes.append(
                dict(
                    common,
                    mean_success_percentage=statistics.mean(
                        (
                            100 * curves[s, history][target]["success_rate"]
                            for s in seeds
                        )
                    ),
                )
            )

    return times, successes


def main():
    args = parse_arguments()
    run_evaluation(args)


if __name__ == "__main__":
    main()
