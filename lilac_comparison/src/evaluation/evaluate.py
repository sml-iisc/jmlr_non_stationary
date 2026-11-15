import argparse
import json
import os
from pathlib import Path

import torch

from src.environment import TRAIN_SEEDS, BlockData, BlockHistoryEnv
from src.evaluation.episodes import evaluate as evaluate_ppo
from src.evaluation.metrics import summarize
from src.evaluation.sequential import evaluate_sequential
from src.models.lilac_agent import PaperLilacAgent
from src.paths import PROJECT_ROOT
from src.training.exact_ppo import ExactBudgetPPO
from src.utils.io import write_csv, write_json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parse_arguments():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="One runs/METHOD_seedSEED directory"
    )
    parser.add_argument("--out", type=Path, default=root / "evaluated_curves")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--event-seeds", type=int, nargs="+", default=list(range(903000, 903016))
    )
    parser.add_argument("--minimum-step", type=int, default=1000000)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    return parser, args


def run_evaluation(parser, args):
    if set(args.event_seeds) & set(TRAIN_SEEDS):
        parser.error("evaluation event seeds overlap training")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.run / "experiment.json").read_text())
    is_lilac = meta["algorithm"].startswith("LILAC")
    method = "lilac" if is_lilac else f"ppo_h{meta['history']}"
    name = f"{method}_seed{meta['policy_seed']}"
    details = args.out / name
    details.mkdir(exist_ok=True)
    data = BlockData(args.out / "cache")
    final = args.run / ("lilac_final.pt" if is_lilac else "ppo_final.zip")
    extension = "pt" if is_lilac else "zip"
    paths = (
        [] if args.final_only else sorted(args.run.glob(f"checkpoint_*.{extension}"))
    )
    paths += [final]
    curve = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        if is_lilac:
            step = int(
                torch.load(path, map_location="cpu", weights_only=False)["steps"]
            )

            if step < args.minimum_step:
                continue

            agent = PaperLilacAgent.load(path, data.successors, args.device)
            rows = evaluate_sequential(agent, data, args.event_seeds)
        else:
            agent = ExactBudgetPPO.load(path, device=args.device)
            step = agent.num_timesteps

            if step < args.minimum_step:
                del agent
                continue

            rows = evaluate_ppo(
                agent, data, meta["history"], args.event_seeds, BlockHistoryEnv
            )

        write_csv(details / f"episodes_{step:09d}.csv", rows)
        curve[step] = dict(timesteps=step, **summarize(rows))
        write_json(args.out / f"{name}.json", [curve[s] for s in sorted(curve)])
        del agent
        print(f"{name}: evaluated {step:,} transitions, {len(rows)} cases", flush=True)

    if not curve:
        raise ValueError("No checkpoints met --minimum-step; use 0 for smoke runs")

    write_json(
        details / "evaluation.json",
        dict(
            run=str(args.run.resolve()),
            method=method,
            policy_seed=meta["policy_seed"],
            event_seeds=args.event_seeds,
            greedy=True,
            weight_updates=False,
            steps=sorted(curve),
        ),
    )


def main():
    parser, args = parse_arguments()
    run_evaluation(parser, args)


if __name__ == "__main__":
    main()
