import copy

import numpy as np

from src.environment.routing_env import HistoryRoutingEnv


def evaluate_model(model, data, history, seeds):
    specs = [(i, j, seed) for seed in seeds for i, j in data.test_pairs]
    envs = [HistoryRoutingEnv(data, history, fixed=spec) for spec in specs]
    obs = [e.reset()[0] for e in envs]
    active = set(range(len(envs)))
    results = {}

    for _ in range(100):
        batch = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        actions, _ = model.predict(batch, deterministic=True)

        for idx in list(active):
            ob, _, done, _, info = envs[idx].step(actions[idx])
            obs[idx] = ob

            if done:
                results[idx] = dict(
                    pair=idx % len(data.test_pairs) + 1,
                    process_seed=specs[idx][2],
                    history=history,
                    success=info["success"],
                    travel_time_s=info["travel_time_s"],
                    steps=info["steps"],
                    penalized_time_s=(
                        info["travel_time_s"] if info["success"] else 2400.0
                    ),
                )
                active.remove(idx)

        if not active:
            break

    assert not active

    return [results[i] for i in range(len(envs))]


def evaluate(model, data, pairs, seeds, history=64):
    view = copy.copy(data)
    view.test_pairs = pairs

    return evaluate_model(model, view, history, seeds)


def score(rows):
    return float(np.mean([r["penalized_time_s"] for r in rows]))
