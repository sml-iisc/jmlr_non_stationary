# LILAC comparison

Compares PPO with truncated event-history windows against LILAC. External events change the environment's transition dynamics. PPO uses history windows of 0, 4, 16, and 128 seconds; LILAC infers latent context from completed trips.

## Setup

Use Python 3.10 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Plot the supplied results

```bash
python plot.py
```

Reads `results/` and saves figures in `plots/`.

## Train and evaluate

```bash
python train.py --out runs/full
python evaluate.py --run runs/full/lilac_seed1001 --out runs/full/evaluation --device cpu
```

Training runs all five methods for three seeds, with four million steps per run. The evaluation command evaluates one trained run; change `--run` to evaluate another.

For a short execution check:

```bash
python train.py --method ppo_h4 --seeds 1001 --smoke --out runs/smoke
```

## Folders

- `configs/`: training settings.
- `data/`: environment and routing inputs.
- `src/`: implementation.
- `results/`: supplied experiment results.
- `plots/`: generated figures.

Run commands from this experiment's folder. Use `--help` to see available options. Choose a new output directory for fresh training. Traffic schedules are cached by process seed; use a fresh run folder after changing traffic settings or simulation code.

