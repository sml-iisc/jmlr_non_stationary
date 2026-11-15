from src.training.scheduled_ppo import ScheduledPPO


class ExactBudgetPPO(ScheduledPPO):

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        remaining = self._total_timesteps - self.num_timesteps

        if remaining % self.n_envs:
            raise ValueError("Transition budget must be divisible by number of workers")

        steps = min(n_rollout_steps, remaining // self.n_envs)

        if steps < 1:
            return False

        if self.rollout_buffer.buffer_size != steps:
            self.rollout_buffer = self.rollout_buffer_class(
                steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs
            )

        return super().collect_rollouts(env, callback, self.rollout_buffer, steps)
