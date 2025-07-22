#!/usr/bin/env python3

from typing import Dict

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper

import torch as th
from gym import spaces


class RenameItem(ObservationWrapper):
    """
    Wrapper to only select subset of dict from obs.
    """

    def __init__(self, env: EnvIface, src: str, dst: str):
        assert isinstance(env.observation_space, spaces.Dict)
        super().__init__(env, self._wrap_obs)
        self._src = src
        self._dst = dst
        self._obs_space = spaces.Dict({
            (dst if (k == src) else k): v
            for (k, v) in env.observation_space.items()
        })

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        out = dict(obs)
        out[self._dst] = out.pop(self._src)
        return out
