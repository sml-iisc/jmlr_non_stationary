import numpy as np
import torch
from torch.nn import functional as F

from src.environment.observation import RoutingObservation


@torch.no_grad()
def evaluate(
    agent, data, seeds, stochastic=False, rng_seed=741001, shortest_path=False
):
    devices = [] if agent.device.type == "cpu" else [agent.device.index or 0]

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(rng_seed)
        specs = data.specs(seeds)
        envs = [RoutingObservation(data, fixed=s) for s in specs]
        obs = np.stack([e.reset()[0] for e in envs])
        active = set(range(len(envs)))
        rows = {}
        hidden, memory = agent.initial(len(envs))
        pa = torch.zeros(len(envs), agent.action_dim, device=agent.device)
        pr = torch.zeros(len(envs), 1, device=agent.device)

        while active:
            state = torch.as_tensor(obs, device=agent.device)
            z, hidden, memory = agent.infer(
                state, pa, pr, hidden, memory, deterministic=not stochastic
            )
            dist, _ = agent.distribution(state, z)
            actions = dist.sample() if stochastic else dist.logits.argmax(-1)
            rewards = np.zeros(len(envs), dtype=np.float32)

            for i in sorted(active):
                if shortest_path:
                    current, goal = (
                        obs[i, : data.N].argmax(),
                        obs[i, data.N :].argmax(),
                    )
                    succ = data.successors[current]
                    mask = succ >= 0
                    cost = np.full(data.max_actions, np.inf)
                    cost[mask] = (
                        data.edge_times[current, mask]
                        + data.distances[succ[mask], goal]
                    )
                    actions[i] = int(cost.argmin())

                obs[i], rewards[i], done, trunc, info = envs[i].step(actions[i].item())

                if done or trunc:
                    rows[i] = dict(
                        pair=specs[i][4],
                        process_seed=specs[i][2],
                        departure_s=specs[i][3],
                        **info,
                        penalized_time_s=(
                            info["travel_time_s"] if info["success"] else 2400
                        )
                    )
                    active.remove(i)

            pa = F.one_hot(actions, agent.action_dim).float()
            pr = torch.as_tensor(rewards, device=agent.device)[:, None]

        for env in envs:
            env.close()

        return [rows[i] for i in range(len(specs))]
