# Transition dynamics change

Tests PPO with event-history windows of 0, 4, 8, 16, 32, 64, and 128 seconds. External events reduce road speeds, which recover over time, changing the environment's transition dynamics.

## Setup

Use Python 3.10 and install PyTorch and the remaining dependencies:

```bash
python -m pip install torch==2.8.0
python -m pip install -r requirements.txt
```

## Plot the supplied results

```bash
python plot.py --output plots/learning_curves.png
```

Reads `results/history_ablation/` and `results/history_4/`.

## Train and evaluate

```bash
python train.py --out runs/full --device cpu
python evaluate.py --out runs/full --device cpu
```

Training uses all seven histories and 20 policy seeds, with four million requested steps per run. It selects models using development cases and also evaluates test checkpoints. The separate evaluation command can rerun checkpoint evaluation.

To plot the new results:

```bash
python plot.py --main runs/full --h4 runs/full --output plots/full.png
```

The launcher disables optional Weights & Biases tracking. Use `--device cuda` if a suitable GPU is available.

## Folders

- `configs/`: training settings.
- `data/`: environment and routing inputs.
- `src/`: implementation.
- `results/`: supplied experiment results.
- `plots/`: generated figures.

Run commands from this experiment's folder. Use `--help` to see available options. Choose a new output directory for fresh training. Traffic schedules are cached by process seed; use a fresh run folder after changing traffic settings or simulation code.

The `data/event_profile/` and `data/speed_response/` folders hold the traffic-process parameters.

