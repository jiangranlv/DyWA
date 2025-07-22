#!/usr/bin/env python3

from env.env.wrap.base import ObservationWrapper

from typing import Iterable, Dict, Optional
import numpy as np
import torch as th
from gym import spaces


class MergeKeys(ObservationWrapper):
    def __init__(self, env, keys: Optional[Iterable[str]] = None):
        super().__init__(env, self._merge_keys)
        if keys is None:
            assert (isinstance(env.observation_space, spaces.Dict))
            keys = sorted(list(env.observation_space.keys()))
        self._keys = keys
        lo = np.concatenate([env.observation_space[k].low
                             for k in self._keys], axis=0)
        hi = np.concatenate([env.observation_space[k].high
                             for k in self._keys], axis=0)
        self._observation_space = spaces.Box(lo, hi)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _merge_keys(self, obs: Dict[str, th.Tensor]) ->  Dict[str, th.Tensor]:
        out = {}
        for k, v in self._keys.items():
            out[k] = th.cat([obs[sub_k] for sub_k in v], dim=-1)
        return out
