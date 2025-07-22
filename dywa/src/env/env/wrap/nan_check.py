#!/usr/bin/env python3

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv
from copy import deepcopy
import pickle

from typing import Optional, Mapping
from dataclasses import dataclass
from util.config import ConfigBase

import torch as th
from torch.utils.tensorboard import SummaryWriter


class NanCheck(WrapperEnv):
    def __init__(self, env: EnvIface,
                 histlen: int = 256):
        super().__init__(env)
        self.__history = []
        self.histlen = histlen

    def __check_obs(self, obs):
        assert (isinstance(obs, Mapping))
        for k, v in obs.items():
            if not isinstance(v, th.Tensor):
                pass
            not_finite = (~th.isfinite(v))
            if not_finite.any():
                with open('/tmp/nancheck.pkl', 'wb') as fp:
                    pickle.dump(self.__history, fp)
                raise ValueError(F'NaN found in obs[{k}]')

    def reset(self):
        obs = self.env.reset()
        self.__check_obs(obs)
        return obs

    def step(self, action: th.Tensor):
        if action is not None:
            not_finite = (~th.isfinite(action))
            if not_finite.any():
                with open('/tmp/nancheck.pkl', 'wb') as fp:
                    pickle.dump(self.__history, fp)
                raise ValueError(F'NaN found in action')
        obs, rew, done, info = self.env.step(action)

        self.__history.append({
            'obsn': deepcopy(obs),
            'rewd': deepcopy(rew),
            'done': deepcopy(done),
            'info': deepcopy(info)
        })
        if len(self.__history) > self.histlen:
            self.__history.pop(0)

        self.__check_obs(obs)
        return (obs, rew, done, info)
