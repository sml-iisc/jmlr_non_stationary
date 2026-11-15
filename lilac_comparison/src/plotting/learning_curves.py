import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.paths import PROJECT_ROOT

matplotlib.use("Agg")


def parse_arguments():
    project = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=project / "results")
    parser.add_argument("--out", type=Path, default=project / "plots")
    args = parser.parse_args()

    return (args, project)


def render_figure(args, project):
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.out
    sns.set_theme()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    histories = [0, 4, 16, 128]
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#e377c2"]
    series = [
        (
            f"Non Stationary w/ T={h}",
            color,
            [args.curves / f"ppo_h{h}_seed{s}.json" for s in [1001, 1002, 1003]],
        )
        for h, color in zip(histories, colors)
    ]
    series.append(
        (
            "LILAC",
            "#1f77b4",
            [args.curves / f"lilac_seed{s}.json" for s in [1001, 1002, 1003]],
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    exported = []

    for label, color, paths in series:
        steps, times, success = summarize_method(paths)
        axes[0].plot(steps, times, color=color, label=label)
        axes[1].plot(steps, success, color=color, label=label)
        exported.extend(
            (
                dict(
                    method=label,
                    timesteps=step,
                    mean_penalized_time_s=float(value),
                    success_percentage=float(percentage),
                    policy_seeds=3,
                )
                for step, value, percentage in zip(steps, times, success)
            )
        )

    axes[0].set(ylabel="Test mean penalized time (s)")
    axes[1].set(ylabel="Test success (%)")

    for ax in axes:
        ax.set(xlabel="Training steps", xlim=(1000000, 4150000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000000.0:g}M"))

    figure.suptitle("Greedy evaluation", fontsize=22, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=18,
    )
    figure.tight_layout(pad=1.0, rect=(0, 0.16, 1, 1))

    assert figure._suptitle.get_fontsize() == 22
    assert all((text.get_fontsize() == 18 for text in legend.get_texts()))

    for ax in axes:
        assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 22
        assert all(
            (
                text.get_fontsize() == 18
                for text in ax.get_xticklabels() + ax.get_yticklabels()
            )
        )

    stem = "tuned_lilac_vs_ppo_bottom_legend"

    for extension in ["png", "pdf", "svg"]:
        figure.savefig(root / f"{stem}.{extension}", dpi=200, bbox_inches="tight")

    plt.close(figure)

    with (root / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)

    print(root / f"{stem}.png")


def summarize_method(paths):
    curves = [json.loads(path.read_text()) for path in paths]
    steps = [row["timesteps"] for row in curves[0]]

    assert len(steps) == 13 and len(set(steps)) == 13
    assert all([row["timesteps"] for row in curve] == steps for curve in curves)
    assert all(row["episodes"] == 480 for curve in curves for row in curve)
    times = np.mean(
        [[row["mean_penalized_time_s"] for row in curve] for curve in curves], axis=0
    )
    success = 100 * np.mean(
        [[row["success_rate"] for row in curve] for curve in curves], axis=0
    )

    assert np.all((success >= 0) & (success <= 100))

    return steps, times, success


def main():
    args, project = parse_arguments()
    render_figure(args, project)


if __name__ == "__main__":
    main()
