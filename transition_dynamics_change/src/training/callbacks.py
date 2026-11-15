import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.evaluation.episodes import evaluate, score
from src.utils.io import dump, write_rows


class DevelopmentCallback(BaseCallback):

    def __init__(self, data, pairs, seeds, out, interval, tracking=None, history=64):
        super().__init__()
        self.data, self.pairs, self.seeds, self.out = (data, pairs, seeds, out)
        self.interval, self.next_eval = (interval, interval)
        self.rows, self.best_weights = ([], None)
        self.best_score, self.best_step = (float("inf"), 0)
        self.tracking = tracking
        self.history = history

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.assess()

    def _on_rollout_start(self):
        if self.num_timesteps >= self.next_eval:
            self.assess()
            self.next_eval = self.num_timesteps + self.interval

    def assess(self):
        rows = evaluate(self.model, self.data, self.pairs, self.seeds, self.history)
        row = dict(
            timesteps=self.num_timesteps,
            development_score_s=score(rows),
            success_rate=float(np.mean([r["success"] for r in rows])),
            entropy_coefficient=float(self.model.ent_coef),
        )

        for key, value in self.model.logger.name_to_value.items():
            if key.startswith("train/") and np.isscalar(value):
                row[key] = float(value) if np.isfinite(value) else None

        self.rows.append(row)
        write_rows(self.out / "development_curve.csv", self.rows)

        if self.tracking is not None:
            self.tracking.log(
                {
                    "global_step": self.num_timesteps,
                    "development/penalized_time_s": row["development_score_s"],
                    "development/success_rate": row["success_rate"],
                    "development/best_penalized_time_s": min(
                        self.best_score, row["development_score_s"]
                    ),
                    "development/entropy_coefficient": row["entropy_coefficient"],
                }
            )

        if row["development_score_s"] < self.best_score:
            self.best_score, self.best_step = (
                row["development_score_s"],
                self.num_timesteps,
            )
            self.best_weights = {
                k: v.detach().cpu().clone()
                for k, v in self.model.policy.state_dict().items()
            }
            self.model.save(self.out / "best_development")
            dump(self.out / "best.json", row)
            write_rows(self.out / "best_development_episodes.csv", rows)

        print("DEVELOPMENT " + json.dumps(row), flush=True)
