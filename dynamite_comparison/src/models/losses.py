import torch


def gaussian_kl(mean, log_variance, prior_mean, prior_log_variance):
    return 0.5 * (
        prior_log_variance
        - log_variance
        + (log_variance.exp() + (mean - prior_mean).square()) / prior_log_variance.exp()
        - 1
    ).sum(-1)


def advantages(batch, gamma=0.995, gae_lambda=0.9):
    rewards, values, valid = (batch[k] for k in ["rewards", "values", "valid"])
    estimated_advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])

    for step in reversed(range(rewards.shape[1])):
        alive = (
            valid[:, step + 1]
            if step + 1 < rewards.shape[1]
            else torch.zeros_like(valid[:, step])
        )
        next_value = values[:, step + 1] if step + 1 < rewards.shape[1] else 0
        delta = rewards[:, step] + gamma * next_value * alive - values[:, step]
        running = (delta + gamma * gae_lambda * alive * running) * valid[:, step]
        estimated_advantages[:, step] = running

    returns = estimated_advantages + values
    valid_advantages = estimated_advantages[valid]
    estimated_advantages[valid] = (valid_advantages - valid_advantages.mean()) / (
        valid_advantages.std(unbiased=False) + 1e-08
    )

    return estimated_advantages, returns
