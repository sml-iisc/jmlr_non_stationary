import numpy as np


def plot_checkpoint_summary(summary, plan, complete_seeds, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_metrics = [
        ("test_mean_penalized_time_s", "Test mean penalized time (s)"),
        ("test_success_percent", "Test success (%)"),
    ]

    for metric, label in plot_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        for h in plan["histories"]:
            group = [
                r
                for r in summary
                if r["history"] == h and r["training_seeds"] == len(complete_seeds)
            ]

            if not group:
                continue

            x = np.array([r["timesteps"] for r in group])
            y = np.array([r[metric] for r in group])
            sd = np.array([r[metric + "_seed_sd"] for r in group])
            (line,) = ax.plot(x, y, label=f"H={h}")
            ax.fill_between(x, y - sd, y + sd, alpha=0.15, color=line.get_color())

        ax.set(
            xlabel="Training timesteps",
            ylabel=label,
            title=f"Mean across {len(complete_seeds)} completed seeds; shading: ±1 seed SD",
        )
        ax.grid(alpha=0.25)

        if ax.lines:
            ax.legend()

        fig.tight_layout()
        fig.savefig(args.out / f"{metric}_vs_training_steps.png", dpi=180)
        plt.close(fig)
