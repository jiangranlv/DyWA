#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Tuple
from util.config import ConfigBase

from env.env.base import EnvIface

import torch as th

T = th.Tensor

from icecream import ic
from gym import spaces



class WrapDiscreteWrench(EnvIface):

    @dataclass
    class Config(ConfigBase):
        num_env: int = -1
        num_actions: int = 6
        action_scale: float = +2.0
        device: str = 'cuda:0'

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self._actions_buf = th.zeros(
            (cfg.num_env, cfg.num_actions),
            dtype=th.float32,
            device=cfg.device
        )
        self.action_space = spaces.Discrete(2 * cfg.num_actions)

    def setup(self, *args, **kwds):
        cfg = self.cfg
        self.env.setup(*args, **kwds)

    def reset(self):
        return self.env.reset()

    def reset_indexed(self, indices: Optional[th.Tensor]):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor) -> Tuple[T, T, T, T]:
        cfg = self.cfg
        if actions is None:
            out_actions = None
        else:
            if th.is_floating_point(actions):
                # given as logits
                i = th.argmax(actions, dim=-1, keepdim=True)
            else:
                # given as indices
                i = actions[..., None]
            out_actions = self._actions_buf
            out_actions.fill_(0)
            out_actions.scatter_(
                -1, i % cfg.num_actions,
                th.where(i >= cfg.num_actions, +cfg.action_scale, -cfg.action_scale))
        return self.env.step(out_actions)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


def wrap_discrete_wrench(
        num_env: int,
        num_actions: int,
        action_scale: float):
    def decorator(cls):
        class Wrapper(WrapDiscreteWrench):
            class Config(cls.Config):
                pass

            def __init__(self, *args, **kwds):
                super().__init__(WrapDiscreteWrench.Config(
                    num_env, num_actions, action_scale), cls(*args, **kwds))
        return Wrapper

    return decorator
