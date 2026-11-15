import random
from dataclasses import asdict

import numpy as np
import torch

from src.environment.observation import LilacRouteEnv
from src.models.config import PaperConfig
from src.models.lilac_agent import PaperLilacAgent

ENV_FIELDS = [
    "start",
    "destination",
    "process_seed",
    "departure",
    "current",
    "elapsed",
    "steps",
    "wait_s",
    "done",
    "records",
]


def save_resume(
    path, agent, step, env, trajectory, observation, context, latent, curve
):
    """One rolling checkpoint includes replay, RNGs and the partially collected trip."""
    base = env.unwrapped
    payload = dict(
        profile="paper_objective_training_resume",
        config=asdict(agent.config),
        state_dict=agent.state_dict(),
        optimizers={k: v.state_dict() for k, v in agent.optimizers.items()},
        cache=np.asarray(agent.cache),
        cache_counts=agent.cache_counts,
        updates=agent.updates,
        replay=agent.replay.episodes,
        replay_size=agent.replay.size,
        agent_rng=agent.rng.bit_generator.state,
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=(
            torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else None
        ),
        step=step,
        env={key: getattr(base, key) for key in ENV_FIELDS},
        env_rng=base.rng.bit_generator.state,
        trajectory=trajectory,
        observation=observation,
        context=context,
        latent=latent,
        curve=curve,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_resume(path, data, device, seed):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    agent = PaperLilacAgent(
        data.successors, PaperConfig(**saved["config"]), device, seed
    )
    agent.load_state_dict(saved["state_dict"])

    for key, optimizer in agent.optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][key])

    agent.cache, agent.cache_counts = (
        list(saved["cache"]),
        list(saved["cache_counts"]),
    )
    agent.updates = saved["updates"]
    agent.replay.episodes, agent.replay.size = (saved["replay"], saved["replay_size"])
    agent.rng.bit_generator.state = saved["agent_rng"]
    env = LilacRouteEnv(data, worker_seed=seed)
    base = env.unwrapped

    for key, value in saved["env"].items():
        setattr(base, key, value)

    base.rng.bit_generator.state = saved["env_rng"]
    base.schedule = data.schedule(base.process_seed)
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])

    if saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])

    return agent, env, saved
