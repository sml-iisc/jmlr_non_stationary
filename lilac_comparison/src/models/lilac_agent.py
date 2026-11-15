from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.base_agent import LilacAgent
from src.models.config import PaperConfig
from src.models.losses import gaussian_kl
from src.models.networks import cache_average, mlp
from src.models.replay import ContiguousEpisodeReplay


class PaperLilacAgent(LilacAgent):

    def __init__(self, successors, config=None, device="cpu", seed=1001):
        config = config or PaperConfig()
        super().__init__(successors, config, device, seed)
        config, state_dim = (config, 2 * self.n_nodes)
        self.encoder = mlp(
            2 * state_dim + self.n_actions + 1, (64, 64), 2 * config.latent_dim
        ).to(self.device)
        self.decoder = mlp(
            state_dim + config.latent_dim + self.n_actions, (64, 64), state_dim + 1
        ).to(self.device)

        for module in (self.actor, self.q1, self.q2, self.encoder, self.decoder):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        self.target1.load_state_dict(self.q1.state_dict())
        self.target2.load_state_dict(self.q2.state_dict())

        with torch.no_grad():
            self.encoder[-1].bias[config.latent_dim :].fill_(
                2 * np.log(config.prior_sigma)
            )
            self.prior.bias_ih_l0.zero_()
            self.prior.bias_hh_l0.zero_()
            self.prior.bias_ih_l0[config.latent_dim : 2 * config.latent_dim].fill_(1.0)

        self.optimizers["encoder"] = torch.optim.Adam(
            self.encoder.parameters(), lr=config.encoder_lr
        )
        self.optimizers["decoder"] = torch.optim.Adam(
            self.decoder.parameters(), lr=config.decoder_lr
        )
        self.replay = ContiguousEpisodeReplay(config.replay_capacity, self.rng)

    def posterior(self, state, action, reward, next_state):
        features = torch.cat(
            (
                self.state_features(state),
                self.state_features(next_state),
                F.one_hot(action.long(), self.n_actions).float(),
                reward.unsqueeze(-1),
            ),
            -1,
        )
        params = self.encoder(features)
        mu, log_variance = params.chunk(2, -1)

        return (
            mu,
            log_variance.clamp(
                self.config.log_variance_min, self.config.log_variance_max
            ),
        )

    def encode(self, state, action, reward, next_state):
        return self.posterior(state, action, reward, next_state)[0]

    def source_prior(self, episodes):
        """Uploaded chronological indexing, including its global one-step shift.

        For episode e>0, unroll cached posteriors from the chunk containing e-1,
        then take its output at e-1. Episode zero has a zero prior. No current or
        future episode enters the selected prediction. Gradients train the prior.
        """
        episodes = np.asarray(episodes, dtype=np.int64)
        length, dim = (self.config.prior_sequence, self.config.latent_dim)
        previous = np.maximum(0, episodes - 1)
        chunks, inverse = np.unique(previous // length, return_inverse=True)
        sequences = np.zeros((len(chunks), length, dim), np.float32)

        for i, chunk in enumerate(chunks):
            values = self.cache[int(chunk) * length : (int(chunk) + 1) * length]

            if values:
                sequences[i, : len(values)] = values

        predictions, _ = self.prior(self.tensor(sequences))
        selected = predictions[
            self.tensor(inverse, torch.long), self.tensor(previous % length, torch.long)
        ]

        return selected * self.tensor(episodes > 0)[:, None]

    @torch.no_grad()
    def advance_prior(self, embedding, hidden=None):
        """Paper execution: advance the recurrent prior after a completed trajectory."""
        if hidden is not None:
            hidden = tuple(h.to(self.device) for h in hidden)

        output, hidden = self.prior(self.tensor(embedding)[None, None], hidden)

        return (output[0, 0].cpu().numpy(), tuple(h.detach() for h in hidden))

    @torch.no_grad()
    def initial_context(self):
        return np.zeros(self.config.latent_dim, np.float32), None

    @torch.no_grad()
    def sample_prior(self, mean):
        return self.tensor(mean) + self.config.prior_sigma * torch.randn(
            self.config.latent_dim, device=self.device
        )

    def _refresh_embedding_cache(self, episodes, mu):
        config = self.config

        with torch.no_grad():
            entries = mu.cpu().numpy()

            for episode in np.unique(episodes):
                episode_index = int(episode)
                values = entries[episodes == episode].reshape(-1, config.latent_dim)
                self.cache[episode_index], self.cache_counts[episode_index] = (
                    cache_average(
                        self.cache[episode_index],
                        self.cache_counts[episode_index],
                        values,
                    )
                )

    def _critic_loss(self, state, action, reward, next_state, done, latent, joint):
        config = self.config
        q_loss = torch.zeros((), device=self.device)
        alpha = self.log_alpha.exp().detach()

        if self.updates >= config.pretrain_updates:
            with torch.no_grad():
                probabilities, logs, next_joint = self.distribution(next_state, latent)
                minimum = torch.minimum(
                    self.target1(next_joint), self.target2(next_joint)
                )
                value = (probabilities * (minimum - alpha * logs)).sum(-1)
                target = reward + config.gamma * (1 - done) * value

            q1 = self.q1(joint).gather(-1, action[..., None]).squeeze(-1)
            q2 = self.q2(joint).gather(-1, action[..., None]).squeeze(-1)
            q_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))

        return q_loss, alpha

    def _update_actor(self, state, latent, alpha):
        config = self.config
        actor_loss, entropy = (
            torch.zeros((), device=self.device),
            torch.zeros((), device=self.device),
        )

        if self.updates >= config.pretrain_updates:
            probabilities, logs, inputs = self.distribution(state, latent.detach())

            with torch.no_grad():
                minimum_q = torch.minimum(self.q1(inputs), self.q2(inputs))

            actor_loss = (probabilities * (alpha * logs - minimum_q)).sum(-1).mean()
            self.optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.optimizers["actor"].step()
            entropy_each = -(probabilities.detach() * logs.detach()).sum(-1)
            target_entropy = (
                config.entropy_fraction
                * self.legal[state[..., 0]].sum(-1).float().log()
            )
            alpha_loss = (self.log_alpha * (entropy_each - target_entropy)).mean()
            self.optimizers["alpha"].zero_grad()
            alpha_loss.backward()
            self.optimizers["alpha"].step()
            entropy = entropy_each.mean()

            with torch.no_grad():
                critic_pairs = (
                    (self.q1, self.target1),
                    (self.q2, self.target2),
                )

                for source, target in critic_pairs:
                    for p, target_p in zip(source.parameters(), target.parameters()):
                        target_p.lerp_(p, config.tau)

        return actor_loss, entropy

    def update(self):
        config = self.config
        episodes, batch = self.replay.sample(config.task_batch, config.context_batch)
        state, action, reward, next_state, done = (self.tensor(x) for x in batch)
        state, next_state, action = (state.long(), next_state.long(), action.long())

        mu, log_variance = self.posterior(state, action, reward, next_state)
        self._refresh_embedding_cache(episodes, mu)
        prior = self.source_prior(episodes)
        kl_encoder = gaussian_kl(
            mu, log_variance, prior.detach()[:, None], config.prior_sigma
        ).mean()
        kl_prior = gaussian_kl(
            mu.detach(), log_variance.detach(), prior[:, None], config.prior_sigma
        ).mean()

        sample = mu + torch.randn_like(mu) * (0.5 * log_variance).exp()
        latent = sample.mean(1, keepdim=True).expand_as(sample)
        joint = torch.cat((self.state_features(state), latent), -1)

        prediction = self.decoder(
            torch.cat((joint, F.one_hot(action, self.n_actions).float()), -1)
        )
        delta = self.state_features(next_state) - self.state_features(state)
        state_loss = 0.5 * (prediction[..., :-1] - delta).square().sum(-1).mean()
        reward_loss = 0.5 * (prediction[..., -1] - reward).square().mean()
        decoder_loss = state_loss + reward_loss

        q_loss, alpha = self._critic_loss(
            state, action, reward, next_state, done, latent, joint
        )

        beta = min(1.0, 1e-06 * max(0, self.updates - config.pretrain_updates))
        loss = decoder_loss + beta * kl_encoder + q_loss

        if not torch.isfinite(loss) or not torch.isfinite(kl_prior):
            raise FloatingPointError("Nonfinite LILAC objective")

        for key in ("encoder", "decoder", "q", "prior"):
            self.optimizers[key].zero_grad()

        loss.backward()
        kl_prior.backward()

        for key in ("encoder", "decoder", "q", "prior"):
            self.optimizers[key].step()

        actor_loss, entropy = self._update_actor(state, latent, alpha)
        self.updates += 1

        metrics = dict(
            q_loss=float(q_loss.detach()),
            actor_loss=float(actor_loss.detach()),
            state_loss=float(state_loss.detach()),
            reward_loss=float(reward_loss.detach()),
            kl_loss=float(kl_encoder.detach()),
            prior_loss=float(kl_prior.detach()),
            alpha=float(alpha),
            entropy=float(entropy),
            beta=beta,
        )

        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite LILAC diagnostic")

        return metrics

    def save(self, path, steps=0):
        torch.save(
            dict(
                profile="paper_objective_discrete",
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

        if saved.get("profile") != "paper_objective_discrete":
            raise ValueError(
                "This checkpoint is not from the paper-objective implementation"
            )

        agent = cls(successors, PaperConfig(**saved["config"]), device)

        if not torch.equal(
            saved["state_dict"]["legal"].cpu(),
            torch.as_tensor(np.asarray(successors) >= 0),
        ):
            raise ValueError("Checkpoint topology mismatch")

        agent.load_state_dict(saved["state_dict"])

        for key, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])

        agent.cache, agent.cache_counts = (
            list(saved["cache"]),
            list(saved["cache_counts"]),
        )
        agent.updates = saved["updates"]

        return agent
