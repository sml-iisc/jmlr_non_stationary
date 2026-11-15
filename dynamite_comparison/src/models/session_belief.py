import torch
from torch import nn
from torch.distributions import Bernoulli
from torch.nn import functional as F

from src.models.vendor.models import DynaMITE_VAE


class PaperBelief(DynaMITE_VAE):

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim, 5, 8, 64)
        self.latent_dim = 5
        self.session_condition = nn.Linear(7, 64)
        self.session_prior = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 10)
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim),
        )

    def initial(self, batch, device):
        zero = torch.zeros(batch, 5, device=device)
        return (
            torch.zeros(1, batch, 64, device=device),
            dict(
                previous_session=zero,
                prior_mu=zero.clone(),
                prior_lv=zero.clone(),
                previous_boundary=zero[:, 0].clone(),
                has_previous=zero[:, 0].clone(),
            ),
        )

    def posterior(self, state, action, reward, hidden, memory):
        embeddings = torch.cat(
            [
                F.relu(self.embed_state(state)),
                F.relu(self.embed_action(action)),
                F.relu(self.embed_reward(reward)),
            ],
            -1,
        )
        condition = torch.cat(
            [
                memory["previous_session"],
                memory["previous_boundary"][:, None],
                memory["has_previous"][:, None],
            ],
            -1,
        )
        x = self.encoder_mlp(embeddings) + self.session_condition(condition)
        out, hidden = self.gru(x[:, None], hidden)
        out = out[:, 0]

        return (
            self.fc_mu(out),
            self.fc_logvar(out).clamp(-10, 5),
            self.fc_termination(out).squeeze(-1),
            hidden,
        )

    def advance(
        self,
        state,
        action,
        reward,
        hidden,
        memory,
        deterministic=False,
        boundary=None,
        final=None,
        valid=None,
    ):
        mu, lv, logit, hidden = self.posterior(state, action, reward, hidden, memory)
        z = mu if deterministic else self.reparameterize(mu, lv)
        dist = Bernoulli(logits=logit)

        if boundary is None:
            boundary = (logit >= 0).float() if deterministic else dist.sample()

        if final is not None:
            boundary = torch.where(final, 1.0, boundary)

        if valid is not None:
            boundary = boundary * valid

        boundary = boundary.detach().float()
        change = boundary.bool()
        previous = torch.where(change[:, None], z, memory["previous_session"])
        next_mu, next_lv = self.session_prior(previous).chunk(2, -1)
        next_lv = next_lv.clamp(-10, 5)
        prior_mu = torch.where(change[:, None], next_mu, memory["prior_mu"])
        prior_lv = torch.where(change[:, None], next_lv, memory["prior_lv"])
        next_memory = dict(
            previous_session=previous,
            prior_mu=prior_mu,
            prior_lv=prior_lv,
            previous_boundary=boundary,
            has_previous=torch.maximum(memory["has_previous"], boundary),
        )
        hidden = hidden * (~change)[None, :, None]
        next_z = prior_mu if deterministic else self.reparameterize(prior_mu, prior_lv)
        actor_latent = torch.where(change[:, None], next_z, z)
        details = dict(
            mu=mu,
            lv=lv,
            z=z,
            logits=logit,
            boundaries=boundary,
            log_boundary_probability=dist.log_prob(boundary),
            prior_mu=memory["prior_mu"],
            prior_lv=memory["prior_lv"],
        )

        return actor_latent, hidden, next_memory, details

    def sequence(self, batch, boundaries=None):
        states = batch["next_states"]
        valid = batch["valid"]
        B, T = valid.shape
        hidden, memory = self.initial(B, states.device)
        output = {
            k: []
            for k in [
                "mu",
                "lv",
                "z",
                "logits",
                "boundaries",
                "log_boundary_probability",
                "prior_mu",
                "prior_lv",
            ]
        }

        for t in range(T):
            last = valid[:, t] & (
                ~valid[:, t + 1] if t + 1 < T else torch.ones_like(valid[:, t])
            )
            chosen = None if boundaries is None else boundaries[:, t]
            _, hidden, memory, details = self.advance(
                states[:, t],
                batch["action_onehots"][:, t],
                batch["rewards"][:, t, None],
                hidden,
                memory,
                boundary=chosen,
                final=last,
                valid=valid[:, t],
            )

            for k in output:
                output[k].append(details[k])

        return {k: torch.stack(v, 1) for k, v in output.items()}
