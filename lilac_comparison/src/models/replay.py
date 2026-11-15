from collections import OrderedDict

import numpy as np


class EpisodeReplay:
    """Only complete episodes are eligible, as in MultitaskReplayPool."""

    def __init__(self, capacity, rng):
        self.capacity, self.rng = (capacity, rng)
        self.episodes = OrderedDict()
        self.size = 0

    def add(self, episode, transitions):
        path = tuple(np.asarray(x) for x in zip(*transitions))
        self.episodes[episode] = path
        self.size += len(transitions)

        while self.size > self.capacity and len(self.episodes) > 1:
            _, removed = self.episodes.popitem(last=False)
            self.size -= len(removed[0])

    def sample(self, tasks, context):
        keys = np.asarray(list(self.episodes))
        selected = self.rng.choice(keys, size=tasks)
        batches = []

        for key in selected:
            path = self.episodes[int(key)]
            indices = self.rng.integers(len(path[0]), size=context)
            batches.append(tuple(x[indices] for x in path))

        return selected, tuple(np.stack([p[i] for p in batches]) for i in range(5))


class ContiguousEpisodeReplay(EpisodeReplay):
    """Same uniform episode sampler without copying every episode key per update."""

    def add(self, episode, transitions):
        if self.episodes and episode != next(reversed(self.episodes)) + 1:
            raise ValueError("Episode IDs must be contiguous")

        super().add(episode, transitions)

    def sample(self, tasks, context):
        first, last = (next(iter(self.episodes)), next(reversed(self.episodes)))
        selected = self.rng.integers(first, last + 1, size=tasks)
        batches = []

        for episode in selected:
            path = self.episodes[int(episode)]
            indices = self.rng.integers(len(path[0]), size=context)
            batches.append(tuple(x[indices] for x in path))

        return selected, tuple(np.stack([p[i] for p in batches]) for i in range(5))
