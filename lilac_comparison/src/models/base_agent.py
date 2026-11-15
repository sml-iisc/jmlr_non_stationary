from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.config import LilacConfig
from src.models.networks import cache_average, mlp
from src.models.replay import EpisodeReplay


class LilacAgent(nn.Module):

    def __init__(self, successors, config=None, device="cpu", seed=1001):
        super().__init__()
        self.config = config or LilacConfig()
        settings = self.config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.n_nodes, self.n_actions = np.shape(successors)
        self.register_buffer("legal", torch.as_tensor(np.asarray(successors) >= 0))
        state_dim = 2 * self.n_nodes
        joint_dim = state_dim + settings.latent_dim
        self.actor = mlp(joint_dim, (settings.hidden,) * 3, self.n_actions)
        self.q1 = mlp(joint_dim, (settings.hidden,) * 3, self.n_actions)
        self.q2 = mlp(joint_dim, (settings.hidden,) * 3, self.n_actions)
        self.target1, self.target2 = (deepcopy(self.q1), deepcopy(self.q2))
        self.target1.requires_grad_(False)
        self.target2.requires_grad_(False)
        self.encoder = mlp(
            2 * state_dim + self.n_actions + 1, (64, 64), settings.latent_dim
        )
        self.decoder = mlp(joint_dim + self.n_actions, (64, 64), 1)
        self.prior = nn.LSTM(settings.latent_dim, settings.latent_dim, batch_first=True)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.to(self.device)
        self.optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=settings.lr),
            "q": torch.optim.Adam(
                list(self.q1.parameters()) + list(self.q2.parameters()), lr=settings.lr
            ),
            "encoder": torch.optim.Adam(
                self.encoder.parameters(), lr=settings.encoder_lr
            ),
            "decoder": torch.optim.Adam(
                self.decoder.parameters(), lr=settings.decoder_lr
            ),
            "prior": torch.optim.Adam(self.prior.parameters(), lr=settings.prior_lr),
            "alpha": torch.optim.Adam([self.log_alpha], lr=settings.lr),
        }
        self.replay = EpisodeReplay(settings.replay_capacity, self.rng)
        self.cache = []
        self.cache_counts = []
        self.updates = 0

    def tensor(self, array, dtype=torch.float32):
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def state_features(self, observation):
        observation = self.tensor(observation, torch.long)
        return F.one_hot(observation, self.n_nodes).float().flatten(-2)

    def encode(self, state, action, reward, next_state):
        return self.encoder(
            torch.cat(
                (
                    self.state_features(state),
                    self.state_features(next_state),
                    F.one_hot(action.long(), self.n_actions).float(),
                    reward.unsqueeze(-1),
                ),
                -1,
            )
        )

    def distribution(self, state, latent):
        state = self.tensor(state, torch.long)
        inputs = torch.cat((self.state_features(state), latent), -1)
        mask = self.legal[state[..., 0]]
        logits = self.actor(inputs).masked_fill(~mask, -1000000000.0)
        logs = logits.log_softmax(-1)

        return logs.exp(), logs, inputs

    @torch.no_grad()
    def act(self, observation, latent, deterministic=False):
        probabilities, _, _ = self.distribution(
            np.asarray(observation)[None], self.tensor(latent)[None]
        )

        if deterministic:
            return int(probabilities.argmax(-1).item())

        return int(torch.multinomial(probabilities, 1).item())

    @torch.no_grad()
    def greedy_batch(self, observations, latent):
        latents = self.tensor(latent).expand(len(observations), -1)
        probabilities, _, _ = self.distribution(observations, latents)

        return probabilities.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def episode_embedding(self, transitions):
        state, action, reward, next_state, _ = (
            np.asarray(x) for x in zip(*transitions)
        )
        return (
            self.encode(
                state, self.tensor(action, torch.long), self.tensor(reward), next_state
            )
            .mean(0)
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def predict_prior(self, previous_embeddings):
        """Predict before acting; no transitions from the upcoming episode are used."""
        if len(previous_embeddings) == 0:
            return np.zeros(self.config.latent_dim, np.float32)

        sequence = np.asarray(
            previous_embeddings[-self.config.prior_sequence :], dtype=np.float32
        )
        output, _ = self.prior(self.tensor(sequence)[None])

        return output[0, -1].cpu().numpy()

    def finish_episode(self, transitions):
        episode = len(self.cache)
        self.replay.add(episode, transitions)
        self.cache.append(self.episode_embedding(transitions))
        self.cache_counts.append(len(transitions))

    @torch.no_grad()
    def batch_prior(self, episodes):
        length = self.config.prior_sequence
        ids, inverse = np.unique(np.asarray(episodes) // length, return_inverse=True)
        chunks, _ = self.prior_chunks(ids)
        output, _ = self.prior(chunks)
        shifted = torch.cat((torch.zeros_like(output[:, :1]), output[:, :-1]), dim=1)

        return shifted[
            self.tensor(inverse, torch.long),
            self.tensor(np.asarray(episodes) % length, torch.long),
        ]

    def prior_chunks(self, chunk_ids):
        length, dim = (self.config.prior_sequence, self.config.latent_dim)
        chunks = np.zeros((len(chunk_ids), length, dim), dtype=np.float32)
        valid = np.zeros((len(chunk_ids), length - 1), dtype=bool)

        for i, chunk_id in enumerate(chunk_ids):
            chunk = self.cache[int(chunk_id) * length : (int(chunk_id) + 1) * length]
            chunks[i, : len(chunk)] = chunk
            valid[i, : max(0, len(chunk) - 1)] = True

        return self.tensor(chunks), self.tensor(valid, torch.bool)

    def train_prior(self, episodes):
        length = self.config.prior_sequence
        chunk_ids = np.unique(np.asarray(episodes) // length)
        chunks, valid = self.prior_chunks(chunk_ids)

        if not valid.any():
            return 0.0

        predicted, _ = self.prior(chunks[:, :-1])
        loss = (predicted - chunks[:, 1:]).square().mean(-1)[valid].mean()
        self.optimizers["prior"].zero_grad()
        loss.backward()
        self.optimizers["prior"].step()

        return float(loss.detach())

    def update(self):
        settings = self.config
        episodes, batch = self.replay.sample(
            settings.task_batch, settings.context_batch
        )
        state, action, reward, next_state, done = (self.tensor(x) for x in batch)
        state, next_state, action = (state.long(), next_state.long(), action.long())
        mus = self.encode(state, action, reward, next_state)

        with torch.no_grad():
            entries = mus.cpu().numpy()

            for episode in np.unique(episodes):
                idx = int(episode)
                values = entries[episodes == episode].reshape(-1, settings.latent_dim)
                self.cache[idx], self.cache_counts[idx] = cache_average(
                    self.cache[idx], self.cache_counts[idx], values
                )

        priors = self.batch_prior(episodes)

        beta = min(1.0, 1e-06 * max(0, self.updates - settings.pretrain_updates))
        regularizer = (mus - priors[:, None]).square().sum(-1).mean()
        latent = mus.mean(1, keepdim=True).expand_as(mus)
        s_features = self.state_features(state)
        joint = torch.cat((s_features, latent), -1)
        predicted_reward = self.decoder(
            torch.cat((joint, F.one_hot(action, self.n_actions).float()), -1)
        ).squeeze(-1)
        decoder_loss = F.mse_loss(predicted_reward, reward)
        q_loss = torch.zeros((), device=self.device)
        alpha = self.log_alpha.exp().detach()

        if self.updates >= settings.pretrain_updates:
            with torch.no_grad():
                probabilities, logs, next_joint = self.distribution(next_state, latent)
                next_q = torch.minimum(
                    self.target1(next_joint), self.target2(next_joint)
                )
                next_value = (probabilities * (next_q - alpha * logs)).sum(-1)
                target = reward + settings.gamma * (1 - done) * next_value

            q1 = self.q1(joint).gather(-1, action[..., None]).squeeze(-1)
            q2 = self.q2(joint).gather(-1, action[..., None]).squeeze(-1)
            q_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))

        representation_loss = decoder_loss + beta * regularizer + q_loss

        if not torch.isfinite(representation_loss):
            raise FloatingPointError("Nonfinite representation/critic loss")

        for key in ("encoder", "decoder", "q"):
            self.optimizers[key].zero_grad()

        representation_loss.backward()

        for key in ("encoder", "decoder", "q"):
            self.optimizers[key].step()

        prior_loss = self.train_prior(episodes)
        actor_loss, entropy = (
            torch.zeros((), device=self.device),
            torch.zeros((), device=self.device),
        )

        if self.updates >= settings.pretrain_updates:
            probabilities, logs, actor_inputs = self.distribution(
                state, latent.detach()
            )

            with torch.no_grad():
                minimum_q = torch.minimum(self.q1(actor_inputs), self.q2(actor_inputs))

            actor_loss = (probabilities * (alpha * logs - minimum_q)).sum(-1).mean()
            self.optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.optimizers["actor"].step()
            entropy_per_state = -(probabilities.detach() * logs.detach()).sum(-1)
            target_entropy = (
                settings.entropy_fraction
                * self.legal[state[..., 0]].sum(-1).float().log()
            )
            alpha_loss = (self.log_alpha * (entropy_per_state - target_entropy)).mean()
            self.optimizers["alpha"].zero_grad()
            alpha_loss.backward()
            self.optimizers["alpha"].step()
            entropy = entropy_per_state.mean()

            with torch.no_grad():
                for q, target_q in ((self.q1, self.target1), (self.q2, self.target2)):
                    for p, target_p in zip(q.parameters(), target_q.parameters()):
                        target_p.lerp_(p, settings.tau)

        self.updates += 1

        metrics = dict(
            q_loss=float(q_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            decoder_loss=float(decoder_loss.detach()),
            kl_loss=float(regularizer.detach()),
            prior_loss=prior_loss,
            alpha=float(alpha),
            entropy=float(entropy),
            beta=beta,
        )

        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite training metric")

        return metrics

    def save(self, path, steps=0):
        torch.save(
            dict(
                config=asdict(self.config),
                state_dict=self.state_dict(),
                optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
                cache=np.asarray(self.cache),
                cache_counts=self.cache_counts,
                updates=self.updates,
                steps=steps,
                successors_shape=(self.n_nodes, self.n_actions),
            ),
            path,
        )

    @classmethod
    def load(cls, path, successors, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=False)

        if tuple(np.shape(successors)) != tuple(saved["successors_shape"]):
            raise ValueError("Checkpoint graph dimensions do not match")

        expected_legal = torch.as_tensor(np.asarray(successors) >= 0)

        if not torch.equal(saved["state_dict"]["legal"].cpu(), expected_legal):
            raise ValueError("Checkpoint action topology does not match")

        agent = cls(successors, LilacConfig(**saved["config"]), device)
        agent.load_state_dict(saved["state_dict"])

        for key, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])

        agent.cache = list(saved["cache"])
        agent.cache_counts = list(saved["cache_counts"])
        agent.updates = saved["updates"]

        return agent
