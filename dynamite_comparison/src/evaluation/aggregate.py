import csv
from pathlib import Path

import numpy as np

METHODS = ["dynamite", "H0", "H4", "H16", "H128"]


def aggregate_curves(records, output, steps=4000000):
    grid = list(range(250000, steps + 1, 250000))
    rows = []

    for method in METHODS:
        subset = [r for r in records if r["method"] == method]

        if not subset:
            continue

        assert len({r["seed"] for r in subset}) == len(subset), "Duplicate policy seed"
        curves = []

        for r in sorted(subset, key=lambda r: r["seed"]):
            points = {}

            for point in r["curve"]:
                if point["actual_timesteps"] == 0:
                    continue

                target = point.get(
                    "target_timesteps", point["actual_timesteps"] // 250000 * 250000
                )

                assert target not in points, "Duplicate checkpoint target"
                points[target] = point

            assert set(points) == set(grid), "Missing fixed-budget checkpoints"
            curves.append(
                [
                    [points[t]["success_rate"], points[t]["mean_penalized_time_s"]]
                    for t in grid
                ]
            )

        data = np.asarray(curves)
        mean = data.mean(0)
        sd = data.std(0, ddof=1) if len(subset) > 1 else np.zeros_like(mean)

        for t, m, s in zip(grid, mean, sd):
            rows.append(
                dict(
                    method=method,
                    requested_checkpoint=t,
                    success_mean=float(m[0]),
                    success_std=float(s[0]),
                    time_mean=float(m[1]),
                    time_std=float(s[1]),
                )
            )

    assert rows, "No completed curves found"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows
