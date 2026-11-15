import numpy as np

from src.environment.traffic_dynamics import WinnerRoutingData


def initialize_data(out, args):
    data = WinnerRoutingData(out)
    rng = np.random.default_rng(args.split_seed)
    chosen = rng.choice(len(data.train_pairs), size=args.dev_pairs, replace=False)
    dev_pairs = [data.train_pairs[int(i)] for i in chosen]
    excluded = set(dev_pairs)
    data.train_pairs = [p for p in data.train_pairs if p not in excluded]

    assert not set(data.train_pairs) & (set(dev_pairs) | set(data.test_pairs))
    assert not set(dev_pairs) & set(data.test_pairs)

    return data, dev_pairs
