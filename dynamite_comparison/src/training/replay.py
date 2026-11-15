from collections import deque

import numpy as np
from torch import nn


class EpisodeReplay:

    def __init__(self, capacity=1000, seed=1001):
        self.episodes = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, batch):
        for b in range(batch["valid"].shape[0]):
            length = int(batch["valid"][b].sum())
            self.episodes.append(
                {
                    k: batch[k][b, :length].detach().cpu().clone()
                    for k in [
                        "states",
                        "next_states",
                        "actions",
                        "action_onehots",
                        "rewards",
                        "valid",
                    ]
                }
            )

    def sample(self, count, device):
        indices = self.rng.choice(
            len(self.episodes), min(count, len(self.episodes)), replace=False
        )
        return {
            k: nn.utils.rnn.pad_sequence(
                [self.episodes[i][k] for i in indices], batch_first=True
            ).to(device)
            for k in self.episodes[0]
        }
