import torch
from stable_baselines3 import PPO

from src.training.schedules import schedule_value


class ScheduledPPO(PPO):
    entropy_start = 0.02
    entropy_schedule = "linear"

    def train(self):
        self.ent_coef = schedule_value(
            self.entropy_start,
            self.entropy_schedule,
            self._current_progress_remaining,
            0.05,
        )
        super().train()

        for parameter in self.policy.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("Nonfinite policy weights")
