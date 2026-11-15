import hashlib
import json
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.routing_env import HistoryRoutingEnv
from src.evaluation.episodes import evaluate, score
from src.models.history_ppo import HistoryExtractor, MaskedSB3Policy
from src.training.callbacks import DevelopmentCallback
from src.training.scheduled_ppo import ScheduledPPO
from src.training.schedules import schedule_value
from src.utils.io import dump, write_rows
from src.utils.tracking import start_tracking


def build_policy(args, data, params, seed, history, out):
    env = DummyVecEnv(
        [
            lambda i=i: Monitor(
                HistoryRoutingEnv(
                    data,
                    history,
                    worker_seed=seed * 1000 + i,
                    train_seeds=list(range(20000, 20032)),
                    distance_shaping=False,
                )
            )
            for i in range(4)
        ]
    )

    assert all(not e.unwrapped.distance_shaping for e in env.envs)
    features = dict(
        n_nodes=data.N,
        n_groups=data.G,
        successors=data.successors.tolist(),
        groups=data.groups.tolist(),
        edge_times=data.edge_times.tolist(),
        distances=data.distances.tolist(),
    )
    options = {
        k: v
        for k, v in params.items()
        if k not in ("id", "lr_schedule", "entropy_schedule")
    }
    lr = options.pop("learning_rate")
    model = ScheduledPPO(
        MaskedSB3Policy,
        env,
        seed=seed,
        device=args.device,
        learning_rate=lambda p: schedule_value(lr, params["lr_schedule"], p, 0.1),
        policy_kwargs=dict(
            features_extractor_class=HistoryExtractor,
            features_extractor_kwargs=features,
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
        ),
        verbose=0,
        **options,
    )
    model.initial_entropy, model.entropy_schedule = (
        params["ent_coef"],
        params["entropy_schedule"],
    )
    initial = hashlib.sha256(
        b"".join(
            (
                v.detach().cpu().numpy().tobytes()
                for v in model.policy.state_dict().values()
            )
        )
    ).hexdigest()
    model.set_logger(configure(str(out / "sb3"), ["csv", "tensorboard"]))

    return env, model, initial


def run_trial(data, dev_pairs, params, seed, steps, out, args, final_test=False):
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text())

    if out.exists():
        out.rename(out.with_name(out.name + f"_interrupted_{time.time_ns()}"))

    out.mkdir(parents=True)
    dump(
        out / "parameters.json",
        dict(
            config=params,
            seed=seed,
            requested_steps=steps,
            initialization="random; no checkpoint loaded",
        ),
    )
    history = getattr(args, "history", 64)
    started = time.monotonic()
    env, model, initial = build_policy(args, data, params, seed, history, out)
    tracking = start_tracking(
        args, data, params, seed, steps, history, out, model, dev_pairs
    )
    callback = DevelopmentCallback(
        data,
        dev_pairs,
        [910, 911, 912],
        out,
        max(1, min(args.eval_interval, steps // 4)),
        tracking,
        history=history,
    )

    if getattr(args, "checkpoint_interval", 0):
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_rows = []

        def save_checkpoint(force=False):
            step = model.num_timesteps

            if checkpoint_rows and checkpoint_rows[-1]["timesteps"] == step:
                return

            if not force and step % args.checkpoint_interval:
                return

            name = f"ppo_step_{step:010d}"
            model.save(checkpoint_dir / name)
            checkpoint_rows.append(dict(timesteps=step, path=f"checkpoints/{name}.zip"))
            write_rows(out / "checkpoints.csv", checkpoint_rows)
            print(f"CHECKPOINT {out.name} step={step}", flush=True)

        model.checkpoint_hook = save_checkpoint
        save_checkpoint(force=True)

    succeeded = False

    try:
        model.learn(total_timesteps=steps, callback=callback, log_interval=10)

        if getattr(args, "checkpoint_interval", 0):
            save_checkpoint(force=True)

        callback.assess()
        model.logger.dump(model.num_timesteps)
        model.save(out / "ppo_final")

        if final_test:
            write_rows(
                out / "final_test_episodes.csv",
                evaluate(
                    model, data, data.test_pairs, [902, 903, 904, 905, 906], history
                ),
            )
            model.policy.load_state_dict(callback.best_weights)
            test_rows = evaluate(
                model, data, data.test_pairs, [902, 903, 904, 905, 906], history
            )
            write_rows(out / "selected_test_episodes.csv", test_rows)

        result = dict(
            configuration=params["id"],
            seed=seed,
            requested_steps=steps,
            actual_steps=model.num_timesteps,
            best_development_score_s=callback.best_score,
            best_step=callback.best_step,
            final_development_score_s=callback.rows[-1]["development_score_s"],
            wall_seconds=time.monotonic() - started,
            initial_policy_sha256=initial,
            parameters=sum(p.numel() for p in model.policy.parameters()),
        )

        if final_test:
            result.update(
                test_penalized_mean_s=score(test_rows),
                test_success_rate=float(np.mean([r["success"] for r in test_rows])),
            )

        dump(out / "complete.json", result)

        if tracking is not None:
            import wandb

            tracking.summary.update(result)

            if final_test:
                tracking.log(
                    {
                        "global_step": model.num_timesteps,
                        "test/episodes": wandb.Table(
                            columns=list(test_rows[0]),
                            data=[list(r.values()) for r in test_rows],
                        ),
                    }
                )

        succeeded = True
        print("TRIAL_COMPLETE " + json.dumps(result), flush=True)

        return result
    finally:
        env.close()
        model.logger.close()

        if tracking is not None:
            import wandb

            tracking.finish(exit_code=0 if succeeded else 1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
