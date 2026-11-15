"""Plot greedy evaluation rewards with the supplied figure's moving average."""
import argparse
from pathlib import Path
import re
import sys

from src.paths import PROJECT_ROOT


SERIES = (
    ('log_hist_10.txt', 'tab:blue'),
    ('log_non_stationary.txt', 'tab:orange'),
    ('log_stationary.txt', 'tab:green'),
    ('log_hist_25.txt', 'tab:red'),
    ('log_hist_15.txt', 'tab:purple'),
)
PATTERN = re.compile(r'^Greedy Eval:\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)', re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, default=PROJECT_ROOT / 'results')
    parser.add_argument('--window', type=int, default=25)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'plots/learning_curves.png')
    parser.add_argument('--require-all', action='store_true', help='Fail unless all five logs are present')
    args = parser.parse_args()

    if args.window < 1:
        parser.error('--window must be positive')

    data = []
    missing = []

    for name, color in SERIES:
        path = args.log_dir / name

        if not path.is_file():
            missing.append(name)
            continue

        values = [float(x) for x in PATTERN.findall(path.read_text(errors='replace'))]

        if len(values) < args.window:
            parser.error(f'{path}: {len(values)} evaluations; at least {args.window} required')

        data.append((name, color, values))

    if missing:
        message = 'Missing logs: ' + ', '.join(missing)

        if args.require_all:
            parser.error(message)

        print('WARNING: ' + message + '; plotting available curves only.', file=sys.stderr)

    if not data:
        parser.error('No usable logs found')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for name, color, values in data:
        averages = np.convolve(values, np.ones(args.window) / args.window, mode='valid')
        ax.plot(range(len(averages)), averages, color=color, label=f'{name} (MA)')
        print(f'{name}: {len(values)} evaluations, {len(averages)} plotted points; first={averages[0]:.2f}, last={averages[-1]:.2f}')

    ax.set(title='Greedy Evaluation (Moving Average)', xlabel='Evaluation Step', ylabel='Mean Reward')
    ax.legend(loc='lower right')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
