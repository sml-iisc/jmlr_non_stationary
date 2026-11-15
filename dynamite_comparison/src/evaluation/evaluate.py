import argparse
import json
from pathlib import Path

import torch

from src.evaluation.aggregate import METHODS, aggregate_curves


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    args = sub.add_parser("aggregate")
    args.add_argument("--runs", type=Path, required=True)
    args.add_argument("--out", type=Path, required=True)
    args.add_argument("--steps", type=int, default=4000000)
    e = sub.add_parser("checkpoint")
    e.add_argument("--method", choices=METHODS, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(904000, 904016))
    )
    args = parser.parse_args()

    return parser, args


def run_evaluation(parser, args):
    torch.set_num_threads(1)

    if args.action == "aggregate":
        records = []
        expected_seeds = None

        for method in METHODS:
            method_seeds = set()

            for seed_dir in sorted(args.runs.glob("seed*")):
                run = seed_dir / method / seed_dir.name / method

                if not run.exists():
                    continue

                result = json.loads((run / "result.json").read_text())

                assert result["actual_timesteps"] >= args.steps, "Incomplete budget"
                curve = json.loads((run / "curve.json").read_text())
                seed = int(seed_dir.name[4:])

                assert result["seed"] == seed
                records.append(dict(seed=seed, method=method, curve=curve))
                method_seeds.add(seed)

            if method_seeds:
                if expected_seeds is None:
                    expected_seeds = method_seeds

                assert (
                    method_seeds == expected_seeds
                ), "Methods have mismatched policy seeds"

        aggregate_curves(records, args.out, args.steps)
        print(args.out)

        return

    if set(args.eval_seeds) & set(range(900000, 900032)):
        parser.error("Evaluation seeds overlap training")

    from src.environment import BlockData, BlockHistoryEnv
    from src.evaluation.metrics import summarize
    from src.utils.io import write_csv as save_rows
    from src.utils.io import write_json

    args.out.mkdir(parents=True, exist_ok=False)
    data = BlockData(args.out / "cache")

    if args.method == "dynamite":
        from src.evaluation.dynamite import evaluate
        from src.models.dynamite_agent import PaperAgent

        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        assert (
            saved["method"] == "dynamite_paper"
        ), "Expected corrected paper-based DynaMITE checkpoint"
        agent = PaperAgent(data)
        agent.load_state_dict(saved["agent"])
        agent.eval()
        rows = evaluate(agent, data, args.eval_seeds)
    else:
        from src.evaluation.episodes import evaluate
        from src.training.scheduled_ppo import ScheduledPPO

        agent = ScheduledPPO.load(args.checkpoint, device="cpu")
        rows = evaluate(
            agent, data, int(args.method[1:]), args.eval_seeds, BlockHistoryEnv
        )

    save_rows(args.out / "episodes.csv", rows)
    write_json(args.out / "summary.json", summarize(rows))
    print(summarize(rows))


def main():
    parser, args = parse_arguments()
    run_evaluation(parser, args)


if __name__ == "__main__":
    main()
