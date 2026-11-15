import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import environment
from src.environment.constants import BUDGET, MAX_H, STEP_CAP


class BlockHistoryEnv(gym.Env):
    """A road-entry action waits for the road to open before traversing it."""

    metadata = {"render_modes": []}

    def __init__(self, data, history, worker_seed=901, fixed=None, train_seeds=None):
        if train_seeds is None:
            self.train_seeds = environment.TRAIN_SEEDS
        else:
            self.train_seeds = train_seeds
        self.data, self.H, self.fixed = (data, int(history), fixed)

        if not 0 <= self.H <= MAX_H:
            raise ValueError("history must be between 0 and 128")

        self.rng = np.random.default_rng(worker_seed)

        self.action_space = spaces.Discrete(data.max_actions)
        self.observation_space = spaces.Dict(
            dict(
                route=spaces.Box(0, data.N - 1, shape=(3,), dtype=np.int64),
                context=spaces.Box(0, np.inf, shape=(3,), dtype=np.float32),
                event_history=spaces.Box(0, 1, shape=(MAX_H, data.G), dtype=np.float32),
                valid_actions=spaces.Box(
                    0, 1, shape=(data.max_actions,), dtype=np.float32
                ),
            )
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.fixed is None:
            pair_index = int(self.rng.integers(len(self.data.train_pairs)))
            self.start, self.destination = self.data.train_pairs[pair_index]
            self.process_seed = int(self.rng.choice(self.train_seeds))
            self.departure = int(self.rng.choice(self.data.departures))
        else:
            start, goal, process_seed, departure, _ = self.fixed
            self.start, self.destination = start, goal
            self.process_seed, self.departure = process_seed, departure

        self.schedule = self.data.schedule(self.process_seed)
        self.current, self.elapsed, self.steps, self.wait_s = (self.start, 0, 0, 0)
        self.done = False
        self.records = []

        return self._obs(), {}

    def _obs(self):
        history = np.zeros((MAX_H, self.data.G), np.float32)

        if self.H:
            # Events at the current second are visible before choosing a road.
            stop = MAX_H + self.departure + self.elapsed + 1
            history[-self.H :] = self.schedule["history"][stop - self.H : stop]

        route = np.array([self.current, self.start, self.destination], np.int64)
        context_values = [
            self.elapsed / BUDGET,
            max(0, 1 - self.elapsed / BUDGET),
            max(0, 1 - self.steps / STEP_CAP),
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
            raise RuntimeError("Reset required")

        action = int(action)

        if (self.current, action) not in self.data.edges:
            raise ValueError("Illegal action")

        u, v, key, edge = self.data.edges[self.current, action]
        r = self.data.groups[self.current, action]

        now = self.departure + self.elapsed
        enter = int(self.schedule["next_open"][now, r])
        end = enter + int(self.data.params["travel"][r])
        old = self.elapsed
        self.steps += 1
        deadline = self.departure + BUDGET
        self.wait_s += max(0, min(enter, deadline) - now)

        if end <= deadline:
            self.records.append(
                dict(
                    source=u,
                    target=v,
                    edge=key,
                    macro=edge["macro"],
                    road=edge["road"],
                    direction=edge["direction"],
                    segment=edge["segment"],
                    last=edge["last"],
                    arrive_at_source_s=old,
                    depart_s=enter - self.departure,
                    arrive_s=end - self.departure,
                    wait_s=enter - now,
                )
            )
            self.elapsed = end - self.departure
            self.current = int(self.data.successors[self.current, action])
        else:
            self.elapsed = BUDGET

        success = self.current == self.destination and end <= deadline
        self.done = bool(success or self.elapsed >= BUDGET or self.steps >= STEP_CAP)

        reward = -(self.elapsed - old) / 100

        if success:
            reward += 10
        elif self.done:
            reward -= 10

        if success:
            terminal_reason = "goal"
        elif self.elapsed >= BUDGET:
            terminal_reason = "time_limit"
        elif self.done:
            terminal_reason = "action_limit"
        else:
            terminal_reason = None

        info = dict(
            success=bool(success),
            travel_time_s=int(self.elapsed),
            steps=self.steps,
            wait_s=int(self.wait_s),
            terminal_reason=terminal_reason,
        )

        return self._obs(), float(reward), self.done, False, info
