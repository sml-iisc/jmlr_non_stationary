import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottom-legend", action="store_true", default=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "results/curve_summary.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "plots")
    args = parser.parse_args()

    return args


def render_figure(args):
    stem = "learning_curves"
    base = args.out
    base.mkdir(parents=True, exist_ok=True)
    sns.set_theme()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 22,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    with args.data.open() as stream:
        rows = list(csv.DictReader(stream))

    methods = [
        ("dynamite", "DynaMITE (paper-based)", "#1f77b4"),
        ("H0", "Non Stationary w/ T=0", "#2ca02c"),
        ("H4", "Non Stationary w/ T=4", "#ff7f0e"),
        ("H16", "Non Stationary w/ T=16", "#9467bd"),
        ("H128", "Non Stationary w/ T=128", "#e377c2"),
    ]
    figure_size = (16, 8) if args.bottom_legend else (16, 6.6)
    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True)
    success_values = []

    for method, label, color in methods:
        data = sorted(
            (
                r
                for r in rows
                if r["method"] == method and int(r["requested_checkpoint"]) >= 1000000
            ),
            key=lambda r: int(r["requested_checkpoint"]),
        )

        assert len(data) == 13, (method, len(data))
        steps = [int(r["requested_checkpoint"]) for r in data]
        axes[0].plot(
            steps, [float(r["time_mean"]) for r in data], color=color, label=label
        )
        success = [100 * float(r["success_mean"]) for r in data]
        success_values.extend(success)
        axes[1].plot(steps, success, color=color, label=label)

    axes[0].set(ylabel="Test mean penalized time (s)")

    if not args.bottom_legend:
        axes[0].set_ylim(top=axes[0].get_ylim()[1] + 100)

    axes[1].set(
        ylabel="Test success (%)", ylim=(max(0, min(success_values) - 1), 100.6)
    )

    for ax in axes:
        ax.set(xlabel="Training steps", xlim=(1000000, 4150000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000000.0:g}M"))

    if args.bottom_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=18,
        )
    else:
        axes[0].legend(loc="upper right")

    fig.suptitle("Greedy evaluation", fontsize=22, y=0.99)
    fig.tight_layout(
        pad=1.0, rect=(0, 0.16, 1, 1) if args.bottom_legend else (0, 0, 1, 1)
    )

    for suffix in ["png", "pdf"]:
        fig.savefig(base / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")

    plt.close(fig)
    settings = (
        "styled_plot_bottom_legend_settings.json"
        if args.bottom_legend
        else "styled_plot_settings.json"
    )
    (base / settings).write_text(
        json.dumps(
            dict(
                source="curve_summary.csv",
                metrics=["time_mean", "100 * success_mean"],
                policy_seeds=20,
                start=1000000,
                end=4150000,
                x_coordinates="requested_checkpoint",
                show_std=False,
                static_shortest_path=False,
                font="DejaVu Sans",
                theme="seaborn default",
                figure_inches=list(figure_size),
                sharex=True,
                legend_location=(
                    "bottom shared"
                    if args.bottom_legend
                    else "left subplot upper right"
                ),
                legend_fontsize=18,
                font_size=22,
                axes_labelsize=22,
                tick_labelsize=18,
                title_fontsize=22,
                subplot_order=["penalized_time", "success_percentage"],
                success_axis_min=max(0, min(success_values) - 1),
                dpi=200,
                methods=[m[0] for m in methods],
            ),
            indent=2,
        )
        + "\n"
    )
    print(base / f"{stem}.png")


def main():
    args = parse_arguments()
    render_figure(args)


if __name__ == "__main__":
    main()
