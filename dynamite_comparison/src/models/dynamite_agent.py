import torch
from torch import nn
from torch.distributions import Categorical

from src.models.session_belief import PaperBelief
from src.models.vendor.models import ActorCritic


class PaperAgent(nn.Module):

    def __init__(self, data, device="cpu"):
        super().__init__()
        self.obs_dim = 2 * data.N
        self.action_dim = data.max_actions
        self.latent_dim = 5
        self.belief = PaperBelief(self.obs_dim, self.action_dim)
        self.policy = ActorCritic(self.obs_dim, self.action_dim, 5, 128)

        for network in [self.policy.actor, self.policy.critic]:
            for i, layer in enumerate(network):
                if isinstance(layer, nn.Tanh):
                    network[i] = nn.ReLU()

        self.register_buffer("connectivity", torch.tensor(data.successors >= 0))
        self.to(device)

    @property
    def device(self):
        return self.connectivity.device

    def initial(self, batch):
        _, memory = self.belief.initial(batch, self.device)
        return None, memory

    def infer(self, state, pa, pr, hidden, memory, deterministic=False):
        if hidden is None:
            hidden, _ = self.belief.initial(state.shape[0], self.device)
            z = (
                memory["prior_mu"]
                if deterministic
                else self.belief.reparameterize(memory["prior_mu"], memory["prior_lv"])
            )

            return z, hidden, memory

        z, hidden, memory, _ = self.belief.advance(
            state, pa, pr, hidden, memory, deterministic
        )

        return z, hidden, memory

    def distribution(self, state, latent):
        x = torch.cat([state, latent], -1)
        mask = self.connectivity[state[:, : self.obs_dim // 2].argmax(-1)]

        return (
            Categorical(logits=self.policy.actor(x).masked_fill(~mask, -1000000000.0)),
            self.policy.critic(x).squeeze(-1),
        )
