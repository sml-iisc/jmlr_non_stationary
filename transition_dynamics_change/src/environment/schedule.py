import numpy as np


class Schedule:

    def __init__(self, edges, speeds, lengths, events, tau, reductions=0.5):
        self.edges = edges
        self.lookup = {e: i for i, e in enumerate(edges)}
        self.lengths = lengths
        reductions = np.broadcast_to(np.asarray(reductions, dtype=float), (len(edges),))
        tau = np.broadcast_to(np.asarray(tau, dtype=float), (len(edges),))

        if (
            np.any(~np.isfinite(tau))
            or np.any(tau <= 0)
            or np.any((reductions < 0) | (reductions >= 1))
        ):
            raise ValueError("Positive recovery time and reductions in [0,1) required")

        self.speeds = np.empty(events.shape, dtype=np.float64)
        self.speeds[0] = speeds

        for t in range(1, len(events)):
            recovered = speeds + (self.speeds[t - 1] - speeds) * np.exp(-1 / tau)
            self.speeds[t] = recovered * np.where(events[t], 1.0 - reductions, 1.0)

        initial_distance = np.zeros(len(edges))
        travelled_distance = np.cumsum(self.speeds / 3.6, axis=0)
        self.distance = np.vstack([initial_distance, travelled_distance])

    def arrival(self, u, v, departure):
        e = self.lookup[u, v]

        if not 0 <= departure < len(self.speeds):
            raise RuntimeError("Departure exceeds saved schedule; increase --horizon")

        if self.lengths[e] == 0:
            return float(departure)

        t = int(np.floor(departure))
        fractional_distance = (departure - t) * self.speeds[t, e] / 3.6
        departure_distance = self.distance[t, e] + fractional_distance
        target = departure_distance + self.lengths[e]

        if target > self.distance[-1, e]:
            raise RuntimeError("Traversal exceeds saved schedule; increase --horizon")

        end = int(np.searchsorted(self.distance[:, e], target, side="left"))
        segment = max(0, end - 1)

        remaining_distance = target - self.distance[segment, e]
        speed_m_per_s = self.speeds[segment, e] / 3.6
        arrival_time = segment + remaining_distance / speed_m_per_s

        return float(arrival_time)
