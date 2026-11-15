# Pendulum lookahead

Compares a stationary actor with naive and intelligent lookahead when external events perturb Pendulum's applied torque.

## Setup

Use Python 3.10 or 3.11 in a separate environment:

```bash
python -m pip install -r requirements.txt
```

## Plot the supplied results

```bash
python plot.py
```

Reads `results/intelligent_training.txt` and `results/naive_training.txt`, applies a 200-evaluation moving average, and writes `plots/learning_curves.png`.

## Train and evaluate

```bash
python train.py --method intelligent --out runs/full
python train.py --method naive --out runs/full
python evaluate.py --intelligent runs/full/intelligent/transition_model.pt --naive runs/full/naive/transition_model.pt
```

Defaults are 10,000 episodes per method. The trainers use the bundled actor. Use `--method actor` to train an actor, or `--method all` to train an actor followed by both transition models. Pass `--actor` to evaluate a newly trained actor.

Only the stationary actor is supplied. To evaluate it without training transition models:

```bash
python evaluate.py --methods baseline --episodes 10
```

To plot new training logs, use `--window 1` for short runs:

```bash
python plot.py --intelligent-log runs/full/intelligent/training.log --naive-log runs/full/naive/training.log --output plots/full.png
```

`configs/` contains settings, `data/checkpoints/` contains the archived actor, and `src/` contains the implementation. The small image under `src/environment/assets/` is used by the environment renderer. Run commands from this folder; use `--help` for options. Fresh runs are stochastic and do not reproduce identical historical samples.
