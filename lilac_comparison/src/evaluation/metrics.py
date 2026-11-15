import numpy as np


def summarize(rows):
    seeds = sorted({row["process_seed"] for row in rows})
    seed_metrics = []

    for seed in seeds:
        seed_rows = [row for row in rows if row["process_seed"] == seed]
        success_rates = [row["success"] for row in seed_rows]
        penalized_times = [row["penalized_time_s"] for row in seed_rows]
        metrics = dict(
            process_seed=seed,
            success_rate=float(np.mean(success_rates)),
            mean_penalized_time_s=float(np.mean(penalized_times)),
        )
        seed_metrics.append(metrics)

    success_rates = [row["success"] for row in rows]
    penalized_times = [row["penalized_time_s"] for row in rows]

    return dict(
        episodes=len(rows),
        event_seeds=seeds,
        success_rate=float(np.mean(success_rates)),
        mean_penalized_time_s=float(np.mean(penalized_times)),
        per_event_seed=seed_metrics,
    )
