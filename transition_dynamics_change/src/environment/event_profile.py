import json
from pathlib import Path

import numpy as np

from src.environment.data import ROOT, RoutingData
from src.environment.events import sample_coupled
from src.environment.schedule import Schedule

PROFILE = ROOT / "data/event_profile"


class EntropyRoutingData(RoutingData):
    """Drop-in data adapter for the existing HistoryRoutingEnv in a NEW run."""

    def __init__(self, output, profile=PROFILE):
        super().__init__(output)
        self.profile = Path(profile)
        self.config = json.loads((self.profile / "config.json").read_text())
        self.model = np.load(self.profile / "segment_model.npz")
        self.cache = {}

    def schedule(self, seed):
        seed = int(seed)

        if seed not in self.cache:
            path = self.output / "schedule_cache" / f"seed_{seed}.npz"

            if path.exists():
                with np.load(path) as z:
                    events, speeds, distance = (z["events"], z["speeds"], z["distance"])
            else:
                rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
                uniform = rng.random((6000, self.G))
                events, _ = sample_coupled(
                    uniform,
                    self.model["probabilities"],
                    self.model["spatial_weights"],
                    self.config["temporal_tau_s"],
                    self.config["excitation_mass"],
                )
                schedule = Schedule(
                    [(g, g) for g in range(self.G)],
                    self.model["base_speeds_kmh"],
                    self.model["representative_lengths_m"],
                    events,
                    self.model["recovery_tau_s"],
                    self.model["reductions"],
                )
                speeds, distance = (schedule.speeds, schedule.distance)
                np.savez_compressed(
                    path,
                    events=events,
                    speeds=speeds,
                    distance=distance,
                )

            self.cache[seed] = (events, speeds, distance)

        return self.cache[seed]
