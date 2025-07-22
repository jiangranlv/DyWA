#!/usr/bin/env python3

from typing import Iterable, Dict

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper

import torch as th
from gym import spaces


class PopDict(ObservationWrapper):
    """
    Wrapper to remove subset of dict from obs.
    """

    def __init__(self, env: EnvIface,
                 keys: Iterable[str]):
        assert isinstance(env.observation_space, spaces.Dict)
        super().__init__(env, self._wrap_obs)
        self.keys = list(keys)
        self._obs_space = spaces.Dict({
            k: v for (k, v) in env.observation_space.items()
            if (k not in self.keys)})

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return {k: obs[k] for (k, v) in obs.items() if (k not in self.keys)}
