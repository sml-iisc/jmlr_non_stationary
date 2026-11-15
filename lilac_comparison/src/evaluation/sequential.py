import numpy as np

from src.environment import BUDGET
from src.environment.observation import LilacRouteEnv


def evaluate_sequential(agent, data, seeds):
    """Evaluate causal trip streams in pairs.json order with frozen weights."""
    specs = data.specs(seeds)
    streams = len(seeds)
    envs = [LilacRouteEnv(data, fixed=specs[30 * i]) for i in range(streams)]
    observations = [env.reset()[0] for env in envs]
    contexts = [agent.initial_context() for _ in seeds]
    trajectories = [[] for _ in seeds]
    positions = [0] * streams
    active, rows = (set(range(streams)), {})

    try:
        while active:
            indices = sorted(active)
            batch = np.stack([observations[i] for i in indices])
            means = np.stack([contexts[i][0] for i in indices])
            actions = agent.greedy_batch(batch, means)

            for i, action in zip(indices, actions):
                old = observations[i]
                observation, reward, done, truncated, info = envs[i].step(action)
                trajectories[i].append(
                    (
                        old.copy(),
                        int(action),
                        reward,
                        observation.copy(),
                        bool(done or truncated),
                    )
                )
                observations[i] = observation

                if done or truncated:
                    index = 30 * i + positions[i]
                    spec = specs[index]
                    rows[index] = dict(
                        sequence_index=positions[i],
                        pair=spec[4],
                        process_seed=spec[2],
                        departure_s=spec[3],
                        **info,
                        penalized_time_s=(
                            info["travel_time_s"] if info["success"] else BUDGET
                        )
                    )
                    embedding = agent.episode_embedding(trajectories[i])
                    contexts[i] = agent.advance_prior(embedding, contexts[i][1])
                    trajectories[i] = []
                    envs[i].close()
                    positions[i] += 1

                    if positions[i] == 30:
                        active.remove(i)
                    else:
                        envs[i] = LilacRouteEnv(
                            data, fixed=specs[30 * i + positions[i]]
                        )
                        observations[i], _ = envs[i].reset()
    finally:
        for env in envs:
            env.close()

    return [rows[i] for i in range(len(specs))]
