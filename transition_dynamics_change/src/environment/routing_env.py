import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HistoryRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data,
        history,
        worker_seed=901,
        train_seeds=None,
        budget=1800.0,
        step_cap=100,
        fixed=None,
        distance_shaping=True,
    ):
        self.data = data
        self.H = int(history)
        self.worker_seed = worker_seed

        if self.H < 0 or self.H > 128:
            raise ValueError("History must be in 0..128")

        self.history_slots = max(64, self.H, getattr(data, "history_slots", 64))
        self.distance_shaping = bool(distance_shaping)
        self.train_seeds = list(train_seeds or range(20000, 20032))
        self.budget = float(budget)
        self.step_cap = int(step_cap)
        self.fixed = fixed

        self.action_space = spaces.Discrete(data.max_actions)
        self.observation_space = spaces.Dict(
            dict(
                route=spaces.Box(0, data.N - 1, shape=(3,), dtype=np.int64),
                context=spaces.Box(0, np.inf, shape=(3,), dtype=np.float32),
                event_history=spaces.Box(
                    0, 1, shape=(self.history_slots, data.G), dtype=np.float32
                ),
                valid_actions=spaces.Box(
                    0, 1, shape=(data.max_actions,), dtype=np.float32
                ),
            )
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is None and (not hasattr(self, "rng")):
            self.rng = np.random.default_rng(self.worker_seed)
        elif seed is not None:
            self.rng = np.random.default_rng(seed)

        spec = (options or {}).get("spec", self.fixed)

        if spec is None:
            self.start, self.destination = self.data.train_pairs[
                int(self.rng.integers(len(self.data.train_pairs)))
            ]
            self.process_seed = int(self.rng.choice(self.train_seeds))
        else:
            self.start, self.destination, self.process_seed = map(int, spec)

        self.data.schedule(self.process_seed)
        self.current = self.start
        self.time = 0.0
        self.steps = 0
        self.done = False
        self.path = [self.data.nodes[self.start]]

        return self._obs(), {}

    def _obs(self):
        history = np.zeros((self.history_slots, self.data.G), np.float32)

        if self.H:
            last = int(np.floor(self.time))
            first = max(0, last - self.H + 1)
            events = self.data.schedule(self.process_seed)[0]
            rows = events[first : last + 1]
            history[-len(rows) :] = rows

        route = np.array([self.current, self.start, self.destination], np.int64)
        context_values = [
            self.time / self.budget,
            max(0.0, 1 - self.time / self.budget),
            max(0.0, 1 - self.steps / self.step_cap),
        ]
        context = np.array(context_values, np.float32)
        valid_actions = (self.data.successors[self.current] >= 0).astype(np.float32)

        return dict(
            route=route,
            context=context,
            event_history=history,
            valid_actions=valid_actions,
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset required after terminal step")

        action = int(action)

        if (
            not 0 <= action < self.data.max_actions
            or self.data.successors[self.current, action] < 0
        ):
            raise ValueError("Invalid structural action; policy mask failed")

        before = self.data.distances[self.current, self.destination] / 100.0
        nxt = int(self.data.successors[self.current, action])
        depart = self.time
        self.time = self.data.arrival(
            self.process_seed,
            self.data.nodes[self.current],
            self.data.nodes[nxt],
            depart,
        )
        self.current = nxt
        self.steps += 1
        self.path.append(self.data.nodes[nxt])

        success = self.current == self.destination and self.time <= self.budget
        time_limit_reached = self.time > self.budget
        action_limit_reached = self.steps >= self.step_cap
        self.done = bool(success or time_limit_reached or action_limit_reached)

        if self.done:
            after = 0.0
        else:
            after = self.data.distances[self.current, self.destination] / 100.0

        reward = -(self.time - depart) / 100.0

        if self.distance_shaping:
            reward += before - 0.999 * after

        if success:
            reward += 10.0
        elif self.done:
            reward -= 10.0

        info = dict(success=bool(success), travel_time_s=self.time, steps=self.steps)

        if self.done:
            info.update(
                process_seed=self.process_seed,
                start_index=self.start,
                destination_index=self.destination,
            )

        return self._obs(), float(reward), self.done, False, info
