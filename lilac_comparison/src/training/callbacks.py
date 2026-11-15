import time

from stable_baselines3.common.callbacks import BaseCallback

from src.utils.io import write_json


class SaveCheckpoints(BaseCallback):

    def __init__(self, args, run, checkpoints, started):
        super().__init__()
        self.args = args
        self.run = run
        self.checkpoints = checkpoints
        self.started = started

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        self.save_due()
        write_json(
            self.args.out / "status.json",
            dict(
                status="training",
                timesteps=self.model.num_timesteps,
                requested_timesteps=self.args.steps,
                seconds=time.monotonic() - self.started,
            ),
        )

    def save_due(self):
        if not self.args.checkpoint_interval:
            return

        target = (len(self.checkpoints) + 1) * self.args.checkpoint_interval

        while target <= min(self.model.num_timesteps, self.args.steps):
            path = self.args.out / f"checkpoint_{target:09d}.zip"
            self.model.save(path)
            self.checkpoints.append(
                dict(
                    target_timesteps=target,
                    actual_timesteps=self.model.num_timesteps,
                    path=str(path),
                )
            )
            target += self.args.checkpoint_interval

        write_json(self.args.out / "checkpoint_index.json", self.checkpoints)
