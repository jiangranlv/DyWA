#!/usr/bin/env python3

from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv
from util.config import ConfigBase
from util.torch_util import dcn


class GenericAdaptiveDomainTuner(WrapperEnv):
    """
    Adaptively tune parameters of a domain,
    based on the success rate.
    Only tries to make the domain "harder" over time.
    """

    @dataclass
    class Config(ConfigBase):
        # Number of episodes between
        # parameter tuning attempts.
        period_episodes: int = 8192

        # Number of steps between
        # paramter tuning attempts.
        period_steps: int = 1024

        # Target success rate
        # if x > target_upper: tune_upper()
        target_upper: float = 0.6
        target_lower: float = float('inf')  # if x < target_lower: tune_lower()

        metric: str = 'suc_rate'  # or 'return'

        # FIXME:
        # `gamma` is required for manually computing the
        # "true" cumulative return for tuning purposes...
        # better way to handle this would be to
        # explicitly track environmental metrics.
        gamma: float = 0.99

    def __init__(self,
                 cfg: Config,
                 env: EnvIface,
                 tune_cb: Callable[[int], None]):
        super().__init__(env)
        self.cfg = cfg
        self._tune_cb = tune_cb

        self._episode_count: int = 0

        if cfg.metric == 'suc_rate':
            self._success_count: int = 0
        elif cfg.metric == 'return':
            N: int = env.num_env
            # Return buffer
            self.__return_buf = th.zeros(N, dtype=th.float,
                                         device=env.device)
            # Bookkeeping step count for each environment
            self.__episode_step = th.zeros(N, dtype=th.int32,
                                           device=env.device)
            self.__avg_return_numer: float = 0.0
            self.__avg_return_denom: float = 0.0

        self._step_since_tune: int = 0
        self.__global_step: int = 0

    def step(self, *args, **kwds):
        cfg = self.cfg
        obs, rew, done, info = super().step(*args, **kwds)
        # NOTE:
        # `success` item is required in `info` :)
        suc = info['success']

        # Increment the counts.
        self._episode_count += dcn(done.sum())

        # Check adjustment conditions (intervals)
        cond_a = (self._episode_count > cfg.period_episodes)
        cond_b = (self._step_since_tune > cfg.period_steps)
        self._step_since_tune += 1
        self.__global_step += 1

        # Bookkeeping metrics for return.
        if cfg.metric == 'return':
            self.__return_buf.add_(
                rew.mul(th.pow(cfg.gamma, self.__episode_step))
            )
            self.__episode_step.add_(1)
            self.__avg_return_numer += (
                (self.__return_buf * done).sum()).item()
            self.__avg_return_denom += (done.sum()).item()
            self.__episode_step.masked_fill_(done, 0)

        if cond_a and cond_b:
            if cfg.metric == 'suc_rate':
                self._success_count += dcn(suc.sum())
                suc_rate = (self._success_count) / (self._episode_count)
                step_up = (suc_rate > cfg.target_upper)
                self._tune_cb(self.__global_step, +1 if step_up else 0)
                self._success_count = 0
            elif cfg.metric == 'return':
                # NOTE: this `if` isn't strictly needed
                # due to `cond_a`, but just to be sure...
                if self.__avg_return_denom > 0:
                    avg_return = (self.__avg_return_numer
                                  / self.__avg_return_denom)
                    if avg_return > cfg.target_upper:
                        self._tune_cb(self.__global_step, +1)
                    elif avg_return < cfg.target_lower:
                        self._tune_cb(self.__global_step, -1)
                    else:
                        self._tune_cb(self.__global_step, 0)
                self.__avg_return_numer = 0.0
                self.__avg_return_denom = 0.0

            # Reset counts.
            self._episode_count = 0
            self._step_since_tune = 0

        return (obs, rew, done, info)


class MultiplyScalarAdaptiveDomainTuner(GenericAdaptiveDomainTuner):

    @dataclass
    class Config(GenericAdaptiveDomainTuner.Config):
        # multiply by `step` to make it `harder`
        step: float = 0.95
        easy: Optional[float] = None
        hard: Optional[float] = None
        # multiply by `step_down` to make it `easier`
        step_down: Optional[float] = 1.05

    def __init__(self, cfg: Config, env: EnvIface,
                 get_fn: Callable[[None], float],
                 set_fn: Optional[Callable[[float], None]] = None,
                 key: Optional[str] = None):
        super().__init__(cfg, env, self._tune)
        self._min_value = min(
            cfg.easy,
            cfg.hard)
        self._max_value = max(
            cfg.easy,
            cfg.hard)
        self._get_fn = get_fn
        self._set_fn = set_fn
        self._key = key

        # Initialization...! Hopefully robust enough
        self._set_fn(cfg.easy)

    def _tune(self, step: int, sign: int):
        cfg = self.cfg
        value = self._get_fn()
        if sign == +1:
            value = np.clip(value * cfg.step,
                            self._min_value, self._max_value)
        elif sign == -1:
            value = np.clip(value * cfg.step_down,
                            self._min_value, self._max_value)
        if (self._key is not None) and (self.writer is not None):
            self.writer.add_scalar(self._key, value,
                                   global_step=step)
        self._set_fn(value)


def main():
    pass


if __name__ == '__main__':
    main()
