#!/usr/bin/env python3

from typing import Tuple, Dict, Iterable
from dataclasses import dataclass
from util.config import ConfigBase

import numpy as np
import torch as th

from env.task.base import TaskBase


class NullTask(TaskBase):

    @dataclass
    class Config(ConfigBase):
        pass

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.timeout: th.Tensor = None

    def create_assets(self, *args, **kwds):
        return {}

    def create_actors(self, *args, **kwds):
        return {}

    def create_sensors(self, *args, **kwds):
        return {}

    def setup(self, env: 'EnvBase'):
        self.timeout = th.empty(
            size=env.cfg.num_env,
            dtype=th.int32,
            device=env.cfg.th_device)

    def compute_feedback(self, env) -> Tuple[float, bool, Dict]:
        # rew, done, info
        rew = th.zeros(env.cfg.num_env, dtype=th.float,
                       device=env.cfg.th_device)
        # done = th.zeros(env.cfg.num_env, dtype=bool,
        #                 device=env.cfg.th_device)
        # done = th.randint(2, size=(env.cfg.num_env,),
        #                   dtype=th.bool)
        # done = env.buffers['step'] >= -1#self.timeout
        # done = env.buffers['step'] >= 50
        done = env.buffers['step'] >= self.timeout
        info = {}
        return (rew, done, info)

    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        if indices is None:
            self.timeout = th.as_tensor(
                np.random.binomial(128*8, 0.5, size=env.cfg.num_env),
                dtype=th.int32, device=env.cfg.th_device)
        else:
            self.timeout[indices] = th.as_tensor(
                np.random.binomial(128*8, 0.5, size=len(indices)),
                dtype=th.int32, device=env.cfg.th_device)
