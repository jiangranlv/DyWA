#!/usr/bin/env python3

import typing
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass, field, InitVar, replace
from collections import Mapping, OrderedDict
from util.config import ConfigBase
import numpy as np

from gym import spaces

from env.env.iface import EnvIface

S = Union[int, Tuple[int, ...]]


@dataclass
class DomainConfig(ConfigBase):
    """ Domain related stuff """
    num_env: int = -1
    num_act: int = -1
    clip_action: Optional[Tuple[float, float]] = None
    # output = discrete actions?
    discrete: bool = False
    # Observation normalization
    normalize_obs: bool = True
    obs_eps: float = 1e-6

    # Reward normalization
    normalize_rew: bool = True
    # Necessary??
    center_rew: bool = False
    rew_eps: float = 1e-6
    # Maximum number of steps per episode.
    timeout: Optional[int] = None
    # obs_space: Union[Dict[str, S], S, None] = None

    @classmethod
    def from_env(cls, env: EnvIface, **kwds) -> 'DomainConfig':
        return domain_config_from_env(env,
                                      cls=cls,
                                      **kwds)


@dataclass
class LossConfig(ConfigBase):
    gamma: float = 0.99
    lmbda: float = 0.95
    clip: float = 0.3

    # Advantage normalization
    adv_eps: float = 1e-6
    normalize_adv: bool = True
    # Hmm...
    normalize_val: bool = False

    # PPO loss scaling coefficients
    # entropy coefficient
    k_ent: float = 0.0
    # critic(value) loss coefficient
    k_val: float = 0.5
    # policy loss coefficient
    k_pi: float = 0.5
    # bound violation loss
    k_bound: float = 0.0001
    k_aux: Optional[Dict[str, float]] = None

    # PPO clipping
    clip_type: int = 3

    # Value clipping
    clip_value: bool = False
    max_dv: Optional[float] = 0.2


@dataclass
class AdaptiveLearningRateConfig(ConfigBase):
    # Adaptive learning rate
    # according to Minchan, very very important
    use_alr: bool = True
    # KL target used for early stopping and
    # adaptive learning rates.
    kl_target: float = 0.01
    kl_bounds: Tuple[float, float] = (0.5, 2.0)
    factor: float = 1.5
    bounds: Tuple[float, float] = (1e-6, 1e-2)
    lr: float = 3e-4
    initial_scale: float = 1.0

    def __post_init__(self):
        if self.use_alr:
            assert (self.bounds[0] <= self.lr)
            assert (self.lr < self.bounds[1])

    @property
    def scale_bounds(self):
        return (
            self.bounds[0] / self.lr,
            self.bounds[1] / self.lr
        )


@dataclass
class ExplorationConfig(ConfigBase):
    use_sde: bool = False

    # NOTE: does not work yet
    use_psn: bool = False

    # Exploration via ornstein-uhlenbeck process.
    ou_process: bool = False
    ou_theta: float = 0.15
    ou_dt: float = 1e-2  # or cfg.dt

    # Log-std initialization value.
    log_std_init: float = -0.6
    # Linear annealing rate for log-std.
    # applied once per training iteration.
    log_std_anneal: float = 0.0
    log_std_min: float = -6.0


@dataclass
class TrainConfig(ConfigBase):
    # Bellman updates and training
    epoch: int = 5
    shuffle: bool = True
    lr: float = 3e-4

    # Data management
    rollout_size: int = 512
    # Number of "chunks"
    chunk_size: int = 32
    # `None` or some reasonable value (16?)
    eval_batch_size: Optional[int] = None
    loss: LossConfig = LossConfig()
    alr: AdaptiveLearningRateConfig = AdaptiveLearningRateConfig()
    use_early_stopping: bool = True
    burn_in: int = 0

    # Logging and saving
    log_period: int = 2048
    save_period: int = 16384
    train_steps: int = 8192
    use_tqdm: bool = True

    # Use mixed-precision training.
    use_amp: Optional[bool] = None
    # Initizlie env with "mixed reset"
    # strategy, while training.
    mixed_reset: bool = True

    version: int = 1

    def __post_init__(self):
        # Apply defaults based on version.
        if self.version == 0:
            self.shuffle = False
        if self.eval_batch_size is None:
            self.eval_batch_size = self.td_horizon
        self.alr = replace(self.alr, lr=self.lr)

    @property
    def td_horizon(self):
        """
        "horizon" or, in other words,
        the length of effective rollout
        """
        return self.rollout_size // self.chunk_size


def domain_config_from_env(env: EnvIface, **kwds) -> DomainConfig:
    cls = kwds.pop('cls', DomainConfig)
    assert isinstance(env.observation_space, (spaces.Box, spaces.Dict))
    assert isinstance(env.action_space, (spaces.Box, spaces.Discrete))

    discrete: bool = isinstance(env.action_space, spaces.Discrete)

    # Check if `env` already returns normalized inputs.
    # Based on the result, configure the defaults (which may
    # still be overridden).
    obs_normalized = (hasattr(env, 'normalizer')
                      and env.normalizer.normalize_obs)
    rew_normalized = (hasattr(env, 'normalizer')
                      and env.normalizer.normalize_rew)
    normalize_obs: bool = kwds.pop('normalize_obs',
                                   not obs_normalized)
    normalize_rew: bool = kwds.pop('normalize_rew',
                                   not rew_normalized)

    if (not discrete) and (env.action_space.is_bounded()):
        clip_action = (
            tuple(float(x) for x in env.action_space.low),
            tuple(float(x) for x in env.action_space.high))
    else:
        clip_action = None

    # if isinstance(env.observation_space, spaces.Dict):
    #     obs_space = {k: v.shape for (k, v) in
    #                  env.observation_space.items()}
    # else:
    #     obs_space = None

    if isinstance(env.action_space, spaces.Discrete):
        num_act = env.action_space.n
    elif isinstance(env.action_space, spaces.Box):
        if len(env.action_space.shape) == 1:
            num_act = env.action_space.shape[0]
        else:
            num_act = env.action_space.shape

    # NOTE: remaining items that are supplied from kwds:
    # - obs_eps, rew_eps, center_rew
    # obs_eps and rew_eps are probably fine to be left alone.
    # And center_rew should always be False IMO.
    return cls(
        num_env=env.num_env,
        num_act=num_act,
        discrete=discrete,
        clip_action=clip_action,
        normalize_obs=normalize_obs,
        normalize_rew=normalize_rew,
        timeout=env.timeout,
        # obs_space=obs_space,
        **kwds)


def main():
    cfg = TrainConfig(lr=1e-3)
    print(cfg)


if __name__ == '__main__':
    main()
