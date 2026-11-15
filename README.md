# Event-driven reinforcement learning experiments

Seven experiments study reinforcement learning under external events:

- [LILAC comparison](lilac_comparison/README.md): compares LILAC with PPO using truncated event history when transition dynamics change.
- [DynaMITE comparison](dynamite_comparison/README.md): compares DynaMITE with PPO using truncated event history when transition dynamics change.
- [Reward change](reward_change/README.md): tests PPO with different event-history windows when events change travel costs and rewards.
- [Transition dynamics change](transition_dynamics_change/README.md): tests PPO with different event-history windows when events change road speeds.

- [CarRacing event history](car_racing_history/README.md): compares PPO event-history windows and baselines when events change road friction.
- [Maze2D lookahead](maze2d_lookahead/README.md): compares stationary, naive-lookahead, and intelligent-lookahead policies in a maze.
- [Pendulum lookahead](pendulum_lookahead/README.md): compares lookahead policies when events perturb the applied torque.

Each experiment contains its own training, evaluation, and plotting scripts. See its README for setup and usage.

Codex was used to help write and refactor the code in this repository.
