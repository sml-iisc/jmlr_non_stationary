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

    def save_due(self):
        if not self.args.checkpoint_interval:
            return

        target = (len(self.checkpoints) + 1) * self.args.checkpoint_interval

        while target <= min(self.model.num_timesteps, self.args.steps):
            path = self.run / ("checkpoint_%09d.zip" % target)
            self.model.save(path)
            self.checkpoints.append(
                dict(
                    target_timesteps=target,
                    actual_timesteps=self.model.num_timesteps,
                    path=path.name,
                )
            )
            target += self.args.checkpoint_interval

        write_json(self.run / "checkpoint_index.json", self.checkpoints)
