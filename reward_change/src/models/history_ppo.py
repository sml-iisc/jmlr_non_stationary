import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from torch import nn

from src.environment.constants import MAX_H


class HistoryExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        n_nodes,
        n_groups,
        successors,
        groups,
        edge_times,
        distances,
    ):
        self.n_actions = len(successors[0])
        route_features = 96 + 3
        history_features = 32 + 16 * self.n_actions
        action_features = 3 * self.n_actions + self.n_actions
        features_dim = route_features + history_features + action_features
        super().__init__(observation_space, features_dim=features_dim)
        self.node_embedding = nn.Embedding(n_nodes, 32)
        self.group_embedding = nn.Embedding(n_groups, 16)
        self.age_filters = nn.Linear(64, 16, bias=False)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.query = nn.Linear(99, 32)

        graph_buffers = [
            ("successors", successors, torch.long),
            ("groups", groups, torch.long),
            ("edge_times", edge_times, torch.float32),
            ("distances", distances, torch.float32),
        ]

        for name, value, dtype in graph_buffers:
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))

    def forward(self, obs):
        route = obs["route"].long()
        current, start, destination = route.unbind(-1)
        node = self.node_embedding(route).flatten(1)
        context = obs["context"].float()

        event_history = obs["event_history"].float().transpose(1, 2)
        temporal_features = self.age_filters(event_history)
        temporal = torch.tanh(temporal_features)

        identities = self.group_embedding.weight.unsqueeze(0).expand(len(route), -1, -1)
        tokens = torch.cat([temporal, identities], dim=-1)
        query = self.query(torch.cat([node, context], -1))
        keys = self.key(tokens)
        attention_scores = (keys * query[:, None]).sum(-1)
        attention = torch.softmax(attention_scores / 32**0.5, -1)
        global_history = (attention[:, :, None] * self.value(tokens)).sum(1)

        candidate_groups = self.groups[current].clamp_min(0)
        batch_indices = torch.arange(len(route), device=route.device)[:, None]
        local = temporal[batch_indices, candidate_groups]
        valid = obs["valid_actions"].float()
        local = (local * valid[:, :, None]).flatten(1)

        successors = self.successors[current].clamp_min(0)
        remaining = self.distances[successors, destination[:, None]]
        edge = self.edge_times[current]
        slack = edge + remaining - self.distances[current, destination][:, None]
        static_features = torch.stack([edge, remaining, slack], -1) / 100.0
        static = (static_features * valid[:, :, None]).flatten(1)

        return torch.cat([node, context, global_history, local, static, valid], -1)


class MaskMlp(nn.Module):

    def __init__(self, features, actions, net_arch, activation_fn, device):
        super().__init__()
        self.actions = actions
        self.base = MlpExtractor(features - actions, net_arch, activation_fn, device)
        self.latent_dim_pi = self.base.latent_dim_pi + actions
        self.latent_dim_vf = self.base.latent_dim_vf

    def forward_actor(self, x):
        actor_features = self.base.forward_actor(x[:, : -self.actions])
        action_mask = x[:, -self.actions :]

        return torch.cat([actor_features, action_mask], -1)

    def forward_critic(self, x):
        return self.base.forward_critic(x[:, : -self.actions])

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class MaskedSB3Policy(MultiInputActorCriticPolicy):

    def _build_mlp_extractor(self):
        self.mlp_extractor = MaskMlp(
            self.features_dim,
            self.action_space.n,
            self.net_arch,
            self.activation_fn,
            self.device,
        )

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        invalid_actions = latent_pi[:, -self.action_space.n :] < 0.5
        logits = logits.masked_fill(invalid_actions, -100000000.0)

        return self.action_dist.proba_distribution(action_logits=logits)


class History128Extractor(HistoryExtractor):

    def __init__(self, observation_space, **kwargs):
        super().__init__(observation_space, **kwargs)
        self.age_filters = nn.Linear(MAX_H, 16, bias=False)


def policy_kwargs(data):
    return dict(
        features_extractor_class=History128Extractor,
        features_extractor_kwargs=dict(
            n_nodes=data.N,
            n_groups=data.G,
            successors=data.successors.tolist(),
            groups=data.groups.tolist(),
            edge_times=data.edge_times.tolist(),
            distances=data.distances.tolist(),
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
