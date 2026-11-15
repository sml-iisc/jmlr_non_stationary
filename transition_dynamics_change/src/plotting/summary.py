def plot_summary(summary, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [r["history"] for r in summary]
    axes[0].errorbar(
        x,
        [r["mean_penalized_time_s"] for r in summary],
        yerr=[r["std_across_training_seeds_s"] for r in summary],
        fmt="o-",
        capsize=4,
    )
    axes[0].set_ylabel("Test mean penalized time (s); error bars: seed SD")
    axes[1].plot(x, [100 * r["success_rate"] for r in summary], "o-")
    axes[1].set_ylabel("Test success (%)")

    for ax in axes:
        ax.set_xlabel("History H (seconds)")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out / "history_ablation.png", dpi=180)
    plt.close(fig)
