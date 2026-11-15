# Reward change

Tests PPO with event-history windows of 0, 4, 8, 16, 32, 64, and 128 seconds. External events cause road-entry delays, changing travel costs and the resulting rewards.

## Setup

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a separate Python environment for this experiment because its dependency versions differ from the other experiments.

## Plot the supplied results

```bash
python plot.py
```

Reads the aggregate CSVs in `results/` and saves figures in `plots/`.

## Train and evaluate

```bash
python train.py --out runs/full --device cpu
python evaluate.py --runs runs/full --out plots/full --device cpu
```

Training uses all seven histories and 20 policy seeds, with four million requested steps per run. Evaluation loads the saved checkpoints and creates aggregate CSVs and figures.

For a short execution check:

```bash
python train.py --out runs/smoke --histories 0 4 --policy-seeds 1001 --smoke --device cpu
python evaluate.py --runs runs/smoke --out plots/smoke --eval-start 0 --device cpu
```

## Folders

- `configs/`: training settings.
- `data/`: environment and routing inputs.
- `src/`: implementation.
- `results/`: supplied experiment results.
- `plots/`: generated figures.

Run commands from this experiment's folder. Use `--help` to see available options. Choose a new output directory for fresh training. Traffic schedules are cached by process seed; use a fresh run folder after changing traffic settings or simulation code.
