#!/usr/bin/env python3

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper

import numpy as np
import torch as th

from gym import spaces


class ZeroOutObs(ObservationWrapper):

    class _Helper:
        """
        Helper class for supplying slice arg for
        ZeroOutObs() creation.
        """

        def __init__(self, cls, env, *args, **kwds):
            self.cls = cls
            self.env = env
            self.args = args
            self.kwds = kwds

        def __getitem__(self, s):
            return self.cls(self.env, s, *self.args, **self.kwds)

    def __init__(self, env, zero_slice,
                 in_place: bool = False):
        super().__init__(env, self._zero_out)
        self._slice = zero_slice
        self._in_place = in_place
        assert (isinstance(env.observation_space, spaces.Box))
        lo = np.copy(env.observation_space.low)
        hi = np.copy(env.observation_space.high)
        lo[self._slice] = 0.0
        hi[self._slice] = 0.0
        self._observation_space = spaces.Box(lo, hi)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _zero_out(self, obs: th.Tensor) -> th.Tensor:
        if not self._in_place:
            obs = obs.detach().clone()
        obs[self._slice] = 0
        return obs

    @classmethod
    def create(cls, env: EnvIface, *args, **kwds):
        return cls._Helper(cls, env, *args, **kwds)
