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

def read_log(path):
    updates, baseline, lookahead = [], [], []

    for line in path.read_text().splitlines():
        if 'Updates:' in line:
            updates.append(int(line.split(' ')[1].split(';')[0]))
        elif 'Without lookahead' in line:
            baseline.append(float(line.split(' ')[-1]))
        elif 'With lookahead' in line:
            lookahead.append(float(line.split(' ')[-1]))

    n = min(len(updates), len(baseline), len(lookahead))

    if not n:
        raise ValueError(f'No complete evaluations in {path}')

    return [np.array(v[:n]) for v in [updates, baseline, lookahead]]

def main():
    parser = argparse.ArgumentParser(description='Reproduce the updates plot from runs 008 and 010.')
    parser.add_argument('--intelligent-log', type=Path, default=ROOT / 'results/intelligent_training.txt')
    parser.add_argument('--naive-log', type=Path, default=ROOT / 'results/naive_training.txt')
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT / 'plots/learning_curves.png')
    args = parser.parse_args()
    intelligent, naive = read_log(args.intelligent_log), read_log(args.naive_log)
    n = min(len(intelligent[0]), len(naive[0]))

    if not 1 <= args.window <= n:
        parser.error('--window must be between 1 and the number of complete evaluations')

    if not np.array_equal(intelligent[0][:n], naive[0][:n]):
        parser.error('The two runs must have matching update coordinates')

    baseline = (intelligent[1][:n] + naive[1][:n]) / 2
    ys = [np.convolve(y, np.ones(args.window) / args.window, 'valid')
          for y in [baseline, intelligent[2][:n], naive[2][:n]]]
    save_plot(intelligent[0][args.window-1:n], ys, 'Updates', args.output)

if __name__ == '__main__':
    main()
