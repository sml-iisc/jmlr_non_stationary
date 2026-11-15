import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.paths import PROJECT_ROOT

matplotlib.use("Agg")
ROOT = PROJECT_ROOT
MAIN = ROOT / "results/history_ablation"
H4 = ROOT / "results/history_4"


def completed_rows(directory):
    with (directory / "checkpoint_test_runs.csv").open() as f:
        rows = list(csv.DictReader(f))

    plan_path = directory / "ablation_plan.json"

    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        seeds = {
            seed
            for seed in plan["seeds"]
            if all(
                (
                    (
                        directory
                        / f"H{h}_{plan['configuration']['id']}_seed{seed}"
                        / "complete.json"
                    ).exists()
                    for h in plan["histories"]
                )
            )
        }
        rows = [r for r in rows if int(r["seed"]) in seeds]
    else:
        seeds = {int(r["seed"]) for r in rows}

    return seeds, rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=MAIN)
    parser.add_argument("--h4", type=Path, default=H4)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--moving-average",
        type=int,
        default=1,
        help="Trailing checkpoint window; values >1 save a separate PNG with faint raw curves",
    )
    parser.add_argument(
        "--legend-position",
        choices=["inside", "bottom"],
        default="bottom",
        help="Bottom placement saves a separate PNG",
    )
    args = parser.parse_args()

    return parser, args


def render_figure(parser, args):
    if args.moving_average < 1:
        parser.error("--moving-average must be positive")

    import seaborn as sns

    sns.set_theme()
    matplotlib.rc("font", family="DejaVu Sans", size=16)
    plt.rcParams.update(
        {
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 18,
        }
    )
    seeds, rows = completed_rows(args.main)
    h4_seeds, h4_rows = completed_rows(args.h4)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharex=True)

    for h in [0, 4, 8, 16, 32, 64, 128]:
        selected, expected = (h4_rows, h4_seeds) if h == 4 else (rows, seeds)
        points = summarize_history(selected, expected, h)

        if not points:
            continue

        color = {
            0: "#2ca02c",
            4: "#ff7f0e",
            8: "#1f77b4",
            16: "#9467bd",
            32: "#d62728",
            64: "#8c564b",
            128: "#e377c2",
        }[h]

        for index, ax in enumerate(axes, start=1):
            steps = np.array([point[0] for point in points])
            raw = np.array([point[index] for point in points])
            visible = steps >= 1000000
            values = raw

            if args.moving_average > 1:
                values = np.array(
                    [
                        raw[max(0, i - args.moving_average + 1) : i + 1].mean()
                        for i in range(len(raw))
                    ]
                )
                ax.plot(
                    steps[visible], raw[visible], color=color, alpha=0.2, linewidth=1
                )

            ax.plot(
                steps[visible],
                values[visible],
                color=color,
                label=f"Non Stationary w/ T={h}",
            )

        print(
            f"H={h}: {len(expected)} seeds, final step {points[-1][0]:,}, mean {points[-1][1]:.3f} s"
        )

    axes[0].set_ylabel("Test mean penalized time (s)")
    axes[1].set_ylabel("Test success (%)")

    for ax in axes:
        ax.set(xlabel="Training steps", xlim=(1000000, 4150000))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x / 1000000.0:g}M")
        )

    title = "Greedy evaluation"

    if args.moving_average > 1:
        title += (
            f" — {args.moving_average}-checkpoint trailing moving average (faint: raw)"
        )

    fig.suptitle(title, fontsize=22, y=0.99)

    if args.legend_position == "bottom":
        fig.set_figheight(8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=18,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.tight_layout(pad=1.0, rect=(0, 0.16, 1, 1))
    else:
        axes[0].legend(loc="upper right", ncol=1, fontsize=18)
        fig.tight_layout(pad=1.0)

    target = MAIN / "test_mean_penalized_time_s_vs_training_steps_from_1M_mean_only.png"

    if args.moving_average > 1:
        target = target.with_name(
            target.stem + f"_moving_average_{args.moving_average}.png"
        )

    if args.legend_position == "bottom":
        target = target.with_name(target.stem + "_legend_bottom.png")

    target = args.output or ROOT / "plots" / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {target}")


def summarize_history(selected, expected, h):
    group = [r for r in selected if int(r["history"]) == h]
    points = []

    for step in sorted({int(r["timesteps"]) for r in group}):
        sample = [r for r in group if int(r["timesteps"]) == step]

        assert (
            len(sample) == len(expected)
            and {int(r["seed"]) for r in sample} == expected
        ), (h, step, "Incomplete evaluation")
        points.append(
            (
                step,
                float(
                    np.mean([float(r["test_mean_penalized_time_s"]) for r in sample])
                ),
                float(np.mean([float(r["test_success_percent"]) for r in sample])),
            )
        )

    return points


def main():
    parser, args = parse_arguments()
    render_figure(parser, args)


if __name__ == "__main__":
    main()
