import numpy as np
import torch
from torch.nn import functional as F


def collect(agent, envs):
    batch_size = len(envs)
    observations = np.stack([e.reset()[0] for e in envs])
    active = np.ones(batch_size, dtype=bool)
    hidden, memory = agent.initial(batch_size)
    previous_action = torch.zeros(batch_size, agent.action_dim, device=agent.device)
    previous_reward = torch.zeros(batch_size, 1, device=agent.device)
    fields = [
        "states",
        "next_states",
        "actions",
        "action_onehots",
        "rewards",
        "latents",
        "log_probs",
        "values",
        "valid",
    ]
    rows = {field: [] for field in fields}

    while active.any():
        state = torch.as_tensor(observations, device=agent.device).clone()

        with torch.no_grad():
            latent, hidden, memory = agent.infer(
                state, previous_action, previous_reward, hidden, memory
            )
            distribution, value = agent.distribution(state, latent)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        reward = np.zeros(batch_size, dtype=np.float32)
        valid = active.copy()

        for i in np.flatnonzero(active):
            observations[i], reward[i], done, truncated, _ = envs[i].step(
                action[i].item()
            )
            active[i] = not (done or truncated)

        previous_action = F.one_hot(action, agent.action_dim).float()
        previous_reward = torch.as_tensor(reward, device=agent.device)[:, None]
        values = [
            state,
            torch.as_tensor(observations, device=agent.device).clone(),
            action,
            previous_action,
            previous_reward[:, 0],
            latent,
            log_probability,
            value,
            torch.as_tensor(valid, device=agent.device),
        ]

        for field, field_value in zip(fields, values):
            rows[field].append(field_value.detach().clone())

    return {field: torch.stack(field_value, 1) for field, field_value in rows.items()}
