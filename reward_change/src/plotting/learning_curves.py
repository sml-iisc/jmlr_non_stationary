import csv
import os

from src.paths import PROJECT_ROOT

HERE = PROJECT_ROOT
HISTORIES = [0, 4, 8, 16, 32, 64, 128]
COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#d62728", "#8c564b", "#e377c2"]


def render(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/reward_change_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    sns.set_theme()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

    with (args.out / "mean_evaluation_curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    with (args.out / "mean_evaluation_success_curves.csv").open(newline="") as stream:
        success = {
            (int(r["history"]), int(r["target_timesteps"])): float(
                r["mean_success_percentage"]
            )
            for r in csv.DictReader(stream)
        }

    fig, (ax, success_ax) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    for h, color in zip(HISTORIES, COLORS):
        curve = sorted(
            (r for r in rows if int(r["history"]) == h),
            key=lambda r: int(r["target_timesteps"]),
        )

        if not curve:
            continue

        percentages = [success[h, int(r["target_timesteps"])] for r in curve]
        ax.plot(
            [int(r["target_timesteps"]) for r in curve],
            [float(r["mean_travel_time_s"]) for r in curve],
            color=color,
            label=f"Non Stationary w/ T={h}",
        )
        success_ax.plot(
            [int(r["target_timesteps"]) for r in curve],
            percentages,
            color=color,
            label=f"Non Stationary w/ T={h}",
        )

    ax.set(
        xlabel="Training steps",
        ylabel="Test mean penalized time (s)",
        xlim=(
            min(int(r["target_timesteps"]) for r in rows),
            max(4150000, max(int(r["target_timesteps"]) for r in rows)),
        ),
    )
    success_ax.set(xlabel="Training steps", ylabel="Test success (%)")
    success_ax.set_ylim(top=100.05)

    for panel in (ax, success_ax):
        panel.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x / 1000000.0:g}M")
        )

    success_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        fontsize=18,
    )
    fig.suptitle("Greedy evaluation", fontsize=22, y=0.99)
    fig.tight_layout(pad=1.0, rect=(0, 0.16, 1, 1))

    for extension in ["png", "pdf"]:
        destination = args.out / f"mean_evaluation_curves.{extension}"
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        print(destination)

    plt.close(fig)
