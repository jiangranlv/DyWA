#!/usr/bin/env python3

from typing import Optional
from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper

import numpy as np
import torch as th

from gym import spaces
from icecream import ic


class SliceObs(ObservationWrapper):

    class _Helper:
        """
        Helper class for supplying slice arg for
        SliceObs() creation.
        """

        def __init__(self, cls, env, *args, **kwds):
            self.cls = cls
            self.env = env
            self.args = args
            self.kwds = kwds

        def __getitem__(self, s):
            return self.cls(self.env, s, *self.args, **self.kwds)

    def __init__(self, env, zero_slice,
                 in_place: bool = False,
                 key: Optional[str] = None
                 ):
        super().__init__(env, self._slice_out)
        self._slice = zero_slice
        self._in_place = in_place
        self._key = key
        if key is None:
            assert (isinstance(env.observation_space, spaces.Box))
            lo = np.copy(env.observation_space.low)
            hi = np.copy(env.observation_space.high)
            lo = lo[self._slice]
            hi = hi[self._slice]
            self._observation_space = spaces.Box(lo, hi)
        else:
            assert (isinstance(env.observation_space, spaces.Dict))
            lo = np.copy(env.observation_space[key].low)
            hi = np.copy(env.observation_space[key].high)
            lo = lo[self._slice]
            hi = hi[self._slice]
            space_slice = spaces.Box(lo, hi)
            self._observation_space = spaces.Dict({
                k: (space_slice if (k == key) else v)
                for (k, v) in env.observation_space.items()
            })

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _slice_out(self, obs: th.Tensor) -> th.Tensor:
        if self._key is None:
            if not self._in_place:
                out = obs[self._slice]
                if out.storage().data_ptr() == obs.storage().data_ptr():
                    out = out.detach().clone()
            return out
        else:
            obs = dict(obs)
            obs[self._key] = obs[self._key][self._slice]
            if not self._in_place:
                obs[self._key] = obs[self._key].detach().clone()
            return obs

    @classmethod
    def create(cls, env: EnvIface, *args, **kwds):
        return cls._Helper(cls, env, *args, **kwds)
