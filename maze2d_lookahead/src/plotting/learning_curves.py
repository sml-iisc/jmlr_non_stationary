from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
COLORS = ['#4c72b0', '#dd8452', '#55a868']
LABELS = ['Baseline', 'With Intelligent Lookahead', 'With Naive Lookahead']

def save_plot(x, ys, xlabel, output):
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for y, color, label in zip(ys, COLORS, LABELS):
        ax.plot(x, y, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Average Evaluation Reward')
    ax.legend(frameon=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=100)
    csv_output = ROOT / 'results/learning_curves.csv' if output == ROOT / 'plots/learning_curves.png' else output.with_suffix('.csv')
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    with csv_output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xlabel, *LABELS])
        writer.writerows(zip(x, *ys))

    plt.close(fig)
    print(output.resolve())

def main():
    parser = argparse.ArgumentParser(description='Reproduce the episode plot from its original reward logs.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--returns-csv', type=Path, help='Use fresh evaluate.py output instead of historical logs.')
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT / 'plots/learning_curves.png')
    args = parser.parse_args()

    if args.returns_csv:
        with args.returns_csv.open() as f:
            rows = list(csv.DictReader(f))

        rewards = [np.array([float(row[key]) for row in rows]) for key in ['baseline', 'intelligent', 'naive']]
    else:
        def read(name):
            with (args.data_dir / name).open() as f:
                return np.array([float(line.split(']')[-1]) for line in f if line.strip()])
        rewards = [read(name) for name in ['baseline_rewards.txt', 'intelligent_rewards.txt', 'naive_rewards.txt']]

    n = min(map(len, rewards))

    if not 0 <= args.start < n:
        parser.error('--start must be nonnegative and smaller than the number of episodes')

    ys = [(np.cumsum(y[:n]) / np.arange(1, n + 1))[args.start:] for y in rewards]
    # Notebook cell 10 plots [100:] at 0..399; screenshot labels them 100..499.
    save_plot(np.arange(args.start, n), ys, 'Episodes', args.output)

if __name__ == '__main__':
    main()
