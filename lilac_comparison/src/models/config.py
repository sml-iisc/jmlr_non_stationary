from dataclasses import dataclass


@dataclass
class LilacConfig:
    latent_dim: int = 8
    hidden: int = 256
    lr: float = 0.0003
    encoder_lr: float = 0.0001
    decoder_lr: float = 0.001
    prior_lr: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    task_batch: int = 32
    context_batch: int = 8
    pretrain_updates: int = 10000
    prior_sequence: int = 200
    replay_capacity: int = 1000000
    entropy_fraction: float = 0.5


@dataclass
class PaperConfig(LilacConfig):
    prior_sigma: float = 0.1
    log_variance_min: float = -10.0
    log_variance_max: float = 2.0
