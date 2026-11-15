# CarRacing event history

Compares PPO with event-history windows of 10, 15, and 25 steps against stationary and nonstationary baselines. Events change road friction.

## Setup

Use Python 3.11 in a separate environment:

```bash
python -m pip install swig==4.3.1
python -m pip install -r requirements.txt
```

## Plot the supplied results

```bash
python plot.py
```

Reads `results/` and writes `plots/learning_curves.png`. Only four curves are available: the archive does not contain the stationary log. `--require-all` requires all five logs.

## Train and evaluate

```bash
python train.py --setting all --output-dir runs/full --device cpu
python evaluate.py --setting hist10 --policy runs/full/models_with_hist_10/policy_final.pth --episodes 5 --device cpu
```

Use `--setting hist10`, `hist15`, `hist25`, `non_stationary`, or `stationary` for one setting. Defaults are 1,000 training iterations and 10 environments. Evaluation uses trained policy weights; none are supplied in the archive.

To plot fresh logs:

```bash
python plot.py --log-dir runs/full --require-all --output plots/full.png
```

`configs/` contains launcher defaults, `src/` contains the implementation, `results/` contains supplied logs, and `plots/` contains figures. Run commands from this folder; use `--help` for options. Fresh runs are stochastic and need not match the supplied curves.
