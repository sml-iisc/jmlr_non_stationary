import json

import numpy as np

from src.environment.data import ROOT
from src.environment.events import sample_coupled
from src.environment.event_profile import PROFILE, EntropyRoutingData
from src.environment.schedule import Schedule

RESPONSE = ROOT / "data/speed_response"


class WinnerRoutingData(EntropyRoutingData):

    def __init__(self, output):
        super().__init__(output)
        self.response = json.loads((RESPONSE / "config.json").read_text())

        with np.load(RESPONSE / "segment_model.npz") as z:
            self.model = {k: z[k].copy() for k in z.files}

        with np.load(PROFILE / "segment_model.npz") as z:
            for k in z.files:
                if k not in ("reductions", "recovery_tau_s"):
                    np.testing.assert_array_equal(self.model[k], z[k])

            np.testing.assert_allclose(
                self.model["reductions"],
                1 - (1 - z["reductions"]) ** self.response["drop_strength"],
            )
            np.testing.assert_allclose(
                self.model["recovery_tau_s"],
                z["recovery_tau_s"] * self.response["recovery_multiplier"],
            )

        self.horizon = 30000
        self.environment_metadata = dict(
            event_process=self.config,
            speed_response=self.response,
            schedule_horizon_s=self.horizon,
        )
        (self.output / "environment.json").write_text(
            json.dumps(self.environment_metadata, indent=2) + "\n"
        )

    def schedule(self, seed):
        seed = int(seed)

        if seed not in self.cache:
            path = self.output / "schedule_cache" / f"seed_{seed}.npz"

            if path.exists():
                with np.load(path) as z:
                    events, speeds, distance = (z["events"], z["speeds"], z["distance"])
            else:
                rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
                events, _ = sample_coupled(
                    rng.random((self.horizon, self.G)),
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
                speeds = schedule.speeds
                floor = self.response["minimum_speed_fraction"]

                if floor != 0:
                    raise ValueError(
                        "This adapter supports the selected zero-floor winner only"
                    )

                distance = schedule.distance
                np.savez_compressed(
                    path,
                    events=events,
                    speeds=speeds,
                    distance=distance,
                )

            self.cache[seed] = (events, speeds, distance)

        return self.cache[seed]
