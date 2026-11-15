import numpy as np
from stable_baselines3.common.logger import KVWriter

from src.utils.io import dump


class WandbOutput(KVWriter):
    """Forward SB3 scalar dumps using environment transitions as the x axis."""

    def __init__(self, run):
        self.run = run

    def write(self, key_values, key_excluded, step=0):
        values = {
            k: float(v)
            for k, v in key_values.items()
            if isinstance(v, (int, float, np.integer, np.floating))
            and "wandb" not in key_excluded.get(k, ())
        }
        self.run.log({**values, "global_step": step})

    def close(self):
        pass


def start_tracking(args, data, params, seed, steps, history, out, model, dev_pairs):
    tracking = None

    if args.wandb_mode != "disabled":
        import wandb

        tracking = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group or args.out.name,
            name=out.name,
            job_type=out.name.split("_config_")[0],
            dir=str(out),
            mode=args.wandb_mode,
            config=dict(
                **params,
                training_seed=seed,
                requested_steps=steps,
                history=history,
                n_envs=4,
                distance_shaping=False,
                initialization="random; no checkpoint loaded",
                development_pairs=dev_pairs,
                development_event_seeds=[910, 911, 912],
                training_event_seeds=list(range(20000, 20032)),
            ),
        )
        tracking.define_metric("global_step")
        tracking.define_metric("*", step_metric="global_step")
        model.logger.output_formats.append(WandbOutput(tracking))
        dump(
            out / "wandb_run.json",
            dict(
                id=tracking.id,
                url=tracking.url,
                project=args.wandb_project,
                mode=args.wandb_mode,
            ),
        )
        print(f"WANDB_RUN {tracking.url}", flush=True)

    return tracking
