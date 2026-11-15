# DynaMITE comparison

Compares DynaMITE against PPO with truncated event-history windows of 0, 4, 16, and 128 seconds. External events change the environment's transition dynamics. DynaMITE infers latent context and session boundaries from transitions.

## Setup

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Plot the supplied results

```bash
python plot.py
```

Reads `results/curve_summary.csv` and saves figures in `plots/`.

## Train and evaluate

```bash
python train.py --out runs/full --workers 4
```

Runs all five methods for 20 policy seeds, with four million steps per run. Training also evaluates checkpoints and creates `runs/full/curve_summary.csv`.

To plot the new results:

```bash
python plot.py --data runs/full/curve_summary.csv --out plots/full
```

To aggregate the saved evaluation curves again:

```bash
python evaluate.py aggregate --runs runs/full/runs --out runs/full/curve_summary.csv
```

## Folders

- `configs/`: training settings.
- `data/`: environment and routing inputs.
- `src/`: implementation.
- `results/`: supplied experiment results.
- `plots/`: generated figures.

Run commands from this experiment's folder. Use `--help` to see available options. Choose a new output directory for fresh training. Traffic schedules are cached by process seed; use a fresh run folder after changing traffic settings or simulation code.

