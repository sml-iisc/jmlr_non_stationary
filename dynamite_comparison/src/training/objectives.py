import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.config import PAPER_CONFIG
from src.models.losses import advantages, gaussian_kl


def hard_sessions(boundaries, valid):
    ids = torch.cat(
        [torch.zeros_like(boundaries[:, :1]), torch.cumsum(boundaries[:, :-1], 1)], 1
    ).long()
    same = (ids[:, :, None] == ids[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    end = torch.zeros_like(ids)
    latest = valid.sum(1).long() - 1

    for t in reversed(range(valid.shape[1])):
        latest = torch.where(valid[:, t] & boundaries[:, t].bool(), t, latest)
        end[:, t] = latest.clamp_min(0)

    return ids, same, end


def session_cost(agent, batch, encoded, config=PAPER_CONFIG):
    """Negative session ELBO and Eq. 2 consistency, given sampled boundaries."""
    valid = batch["valid"]
    batch_size, sequence_length = valid.shape
    ids, same, end = hard_sessions(encoded["boundaries"], valid)
    batch_indices, latent_steps, transition_steps = same.nonzero(as_tuple=True)
    x = torch.cat(
        [
            batch["states"][batch_indices, transition_steps],
            batch["action_onehots"][batch_indices, transition_steps],
            encoded["z"][batch_indices, latent_steps],
        ],
        -1,
    )
    reward_error = (
        agent.belief.reward_decoder(x).squeeze(-1)
        - batch["rewards"][batch_indices, transition_steps]
    ).square()
    state_error = (
        (
            agent.belief.state_decoder(x)
            - batch["next_states"][batch_indices, transition_steps]
        )
        .square()
        .mean(-1)
    )
    reconstruction = torch.zeros(batch_size, device=valid.device).index_add(
        0, batch_indices, reward_error + state_error
    )
    latent_kl = gaussian_kl(
        encoded["mu"], encoded["lv"], encoded["prior_mu"], encoded["prior_lv"]
    )
    p = torch.sigmoid(encoded["logits"])
    hazard = config["hazard"]
    boundary_kl = p * (F.logsigmoid(encoded["logits"]) - np.log(hazard)) + (1 - p) * (
        F.logsigmoid(-encoded["logits"]) - np.log(1 - hazard)
    )
    real_boundary = valid & torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], 1)
    kl = (latent_kl * valid).sum(1) + (boundary_kl * real_boundary).sum(1)
    index = end[:, :, None].expand(-1, -1, encoded["mu"].shape[-1])
    theta = gaussian_kl(
        encoded["mu"],
        encoded["lv"],
        encoded["mu"].gather(1, index),
        encoded["lv"].gather(1, index),
    )
    adjacent = valid[:, 1:] & valid[:, :-1] & (ids[:, 1:] == ids[:, :-1])
    consistency = (F.relu(theta[:, 1:] - theta[:, :-1]) * adjacent).sum(1)
    cost = (
        config["vae_weight"] * (reconstruction + kl)
        + config["consistency_weight"] * consistency
    )
    log_probability = (encoded["log_boundary_probability"] * real_boundary).sum(1)
    metrics = dict(
        reconstruction=reconstruction.mean().item(),
        latent_and_boundary_kl=kl.mean().item(),
        consistency=consistency.mean().item(),
        boundary_probability=(
            p[real_boundary].mean().item() if real_boundary.any() else 0.0
        ),
        variational_cost=cost.mean().item(),
    )

    return cost, log_probability, metrics


def score_function_objective(cost, log_probability, baseline):
    """Unbiased score-function correction for genuinely discrete session samples."""
    score = ((cost.detach() - baseline) * log_probability).mean()

    return cost.mean() + score - score.detach()


def update(
    agent, rollout, replay_batch, opt_policy, opt_vae, baseline, config=PAPER_CONFIG
):
    encoded = agent.belief.sequence(replay_batch)
    cost, logp, metrics = session_cost(agent, replay_batch, encoded, config)
    loss = score_function_objective(cost, logp, baseline)

    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite variational objective")

    opt_vae.zero_grad()
    loss.backward()
    belief_gradient_norm = nn.utils.clip_grad_norm_(
        agent.belief.parameters(), config["max_grad_norm"]
    )
    opt_vae.step()
    baseline = 0.95 * baseline + 0.05 * cost.detach().mean().item()
    estimated_advantages, returns = advantages(
        rollout, config["gamma"], config["gae_lambda"]
    )
    valid = rollout["valid"]
    state, z, action, old_log_probabilities, old_values = [
        rollout[k][valid]
        for k in ["states", "latents", "actions", "log_probs", "values"]
    ]
    estimated_advantages = estimated_advantages[valid]
    returns = returns[valid]

    with torch.no_grad():
        dist, _ = agent.distribution(state, z)
        error = (dist.log_prob(action) - old_log_probabilities).abs().max().item()

    if error > 1e-05:
        raise AssertionError("Collected policy inputs changed during VAE update")

    for _ in range(config["n_epochs"]):
        shuffled_indices = torch.randperm(len(state), device=agent.device)
        minibatches = shuffled_indices.split(config["batch_size"])

        for idx in minibatches:
            dist, value = agent.distribution(state[idx], z[idx])
            ratio = (dist.log_prob(action[idx]) - old_log_probabilities[idx]).exp()
            policy_loss = -torch.minimum(
                ratio * estimated_advantages[idx],
                ratio.clamp(1 - config["clip_range"], 1 + config["clip_range"])
                * estimated_advantages[idx],
            ).mean()
            clipped = old_values[idx] + (value - old_values[idx]).clamp(
                -config["clip_range"], config["clip_range"]
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (value - returns[idx]).square(), (clipped - returns[idx]).square()
                ).mean()
            )
            actor_loss = (
                policy_loss
                + config["vf_coef"] * value_loss
                - config["ent_coef"] * dist.entropy().mean()
            )
            opt_policy.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), config["max_grad_norm"])
            opt_policy.step()

    if not all(torch.isfinite(w).all() for w in agent.parameters()):
        raise FloatingPointError("Nonfinite parameters")

    metrics.update(
        initial_logprob_error=error,
        vae_gradient_norm=float(belief_gradient_norm),
        policy_loss=float(actor_loss.detach()),
    )

    return baseline, metrics
