import numpy as np
import torch
from stable_baselines3 import PPO

from src.training.schedules import schedule_value


class ScheduledPPO(PPO):
    """SB3 accepts an LR schedule; entropy must be scheduled before each update."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["checkpoint_hook"]

    def train(self):
        self.ent_coef = schedule_value(
            self.initial_entropy,
            self.entropy_schedule,
            self._current_progress_remaining,
            0.05,
        )
        self.logger.record("train/entropy_coefficient", self.ent_coef)
        super().train()

        if getattr(self, "checkpoint_hook", None) is not None:
            self.checkpoint_hook()

        for parameter in self.policy.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("Nonfinite policy parameters")

        for key, value in self.logger.name_to_value.items():
            if key.startswith("train/") and isinstance(value, (float, np.floating)):
                if key != "train/explained_variance" and (not np.isfinite(value)):
                    raise FloatingPointError(f"Nonfinite {key}")
