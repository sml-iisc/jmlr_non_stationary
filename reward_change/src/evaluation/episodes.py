import numpy as np


def evaluate(model, data, history, seeds, env_class):
    specs = data.specs(seeds)
    envs = [env_class(data, history, fixed=spec) for spec in specs]
    observations = [env.reset()[0] for env in envs]
    active, rows = (set(range(len(envs))), {})

    try:
        while active:
            indices = sorted(active)
            batch = {
                k: np.stack([observations[i][k] for i in indices])
                for k in observations[0]
            }
            actions, _ = model.predict(batch, deterministic=True)

            for i, action in zip(indices, actions):
                observations[i], _, done, truncated, info = envs[i].step(action)

                if done or truncated:
                    rows[i] = dict(
                        history=history,
                        pair=specs[i][4],
                        process_seed=specs[i][2],
                        departure_s=specs[i][3],
                        **info,
                        penalized_time_s=(
                            info["travel_time_s"] if info["success"] else 2400
                        )
                    )
                    active.remove(i)
    finally:
        for env in envs:
            env.close()

    return [rows[i] for i in range(len(specs))]
