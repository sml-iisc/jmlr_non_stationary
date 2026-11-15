import numpy as np


def summarize(rows):
    success_rates = [row["success"] for row in rows]
    penalized_times = [row["penalized_time_s"] for row in rows]
    episode_returns = []

    for row in rows:
        terminal_reward = 10 if row["success"] else -10
        episode_return = -row["travel_time_s"] / 100 + terminal_reward
        episode_returns.append(episode_return)

    return dict(
        success_rate=float(np.mean(success_rates)),
        mean_penalized_time_s=float(np.mean(penalized_times)),
        mean_return=float(np.mean(episode_returns)),
        episodes=len(rows),
    )
