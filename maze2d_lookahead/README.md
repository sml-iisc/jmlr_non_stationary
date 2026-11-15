# Maze2D lookahead

Compares a stationary actor with naive and intelligent lookahead in a maze affected by external events. The lookahead methods learn transition models.

## Setup

Use Python 3.10 or 3.11 in a separate environment:

```bash
python -m pip install -r requirements.txt
```

Training and checkpoint evaluation require MuJoCo. Plotting the supplied results only requires NumPy and Matplotlib.

## Plot the supplied results

```bash
python plot.py
```

Reads the reward logs in `results/` and writes `plots/learning_curves.png`. Curves are cumulative mean rewards, shown from episode 100 onward.

## Train and evaluate

```bash
python train.py --method intelligent --out runs/full
python train.py --method naive --out runs/full
python evaluate.py --episodes 500
```

The trainers use the bundled actor by default. Use `--method actor` to train an actor, or `--method all` to train an actor followed by both transition models. Defaults are 10,000 episodes per method. Outputs go under the chosen run folder, including `intelligent/transition_model.pt` and `naive/transition_model.pt`.

`evaluate.py` uses bundled checkpoints by default. To evaluate fresh transition models:

```bash
python evaluate.py --intelligent runs/full/intelligent/transition_model.pt --naive runs/full/naive/transition_model.pt
python plot.py --returns-csv runs/evaluation_returns.csv --output plots/new_evaluation.png
```

`configs/` contains launcher settings, `data/checkpoints/` contains supplied models, and `src/` contains the implementation. Run commands from this folder; use `--help` for options. Fresh training is stochastic; the archived logs are not tied to a verified exact generating configuration.
