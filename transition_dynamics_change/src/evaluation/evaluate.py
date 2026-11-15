import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.environment.split import initialize_data
from src.evaluation.episodes import evaluate
from src.plotting.checkpoints import plot_checkpoint_summary
from src.training.scheduled_ppo import ScheduledPPO
from src.utils.io import write_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    return args


def run_evaluation(args):
    torch.set_num_threads(1)
    plan = json.loads((args.out / "ablation_plan.json").read_text())
    args.split_seed, args.dev_pairs = (
        plan["split_seed"],
        plan["development_pair_count"],
    )
    data, _ = initialize_data(args.out, args)
    data.history_slots = plan["history_tensor_shape"][0]
    complete_seeds = [
        seed
        for seed in plan["seeds"]
        if all(
            (
                (
                    args.out
                    / f"H{h}_{plan['configuration']['id']}_seed{seed}"
                    / "complete.json"
                ).exists()
                for h in plan["histories"]
            )
        )
    ]
    rows = []

    for h in plan["histories"]:
        for seed in complete_seeds:
            run = args.out / f"H{h}_{plan['configuration']['id']}_seed{seed}"
            manifest = run / "checkpoints.csv"

            if not manifest.exists():
                continue

            for checkpoint in list(csv.DictReader(manifest.open())):
                step = int(checkpoint["timesteps"])
                result_path = run / "checkpoints" / f"test_step_{step:010d}.csv"

                if result_path.exists():
                    episodes = list(csv.DictReader(result_path.open()))
                else:
                    model = ScheduledPPO.load(
                        run / checkpoint["path"], device=args.device
                    )
                    episodes = evaluate(
                        model, data, data.test_pairs, plan["test_event_seeds"], h
                    )
                    write_rows(result_path, episodes)
                    del model

                rows.append(
                    dict(
                        history=h,
                        seed=seed,
                        timesteps=step,
                        test_mean_penalized_time_s=float(
                            np.mean([float(r["penalized_time_s"]) for r in episodes])
                        ),
                        test_success_percent=100
                        * float(
                            np.mean(
                                [
                                    str(r["success"]).lower() in ("true", "1", "1.0")
                                    for r in episodes
                                ]
                            )
                        ),
                    )
                )
                write_rows(args.out / "checkpoint_test_runs.csv", rows)
                print(f"EVALUATED H={h} seed={seed} step={step}", flush=True)

    if not rows:
        raise RuntimeError("No checkpoints found")

    summary = []

    for h, step in sorted({(r["history"], r["timesteps"]) for r in rows}):
        group = [r for r in rows if r["history"] == h and r["timesteps"] == step]
        item = dict(history=h, timesteps=step, training_seeds=len(group))

        for metric in ["test_mean_penalized_time_s", "test_success_percent"]:
            values = [r[metric] for r in group]
            item[metric] = float(np.mean(values))
            item[metric + "_seed_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )

        summary.append(item)

    write_rows(args.out / "checkpoint_test_summary.csv", summary)
    plot_checkpoint_summary(summary, plan, complete_seeds, args)


def main():
    args = parse_arguments()
    run_evaluation(args)


if __name__ == "__main__":
    main()
