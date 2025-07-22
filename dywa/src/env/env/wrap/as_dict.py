#!/usr/bin/env python3

from typing import Dict

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from gym import spaces

import torch as th


class AsDict(ObservationWrapper):
    """
    Wrapper class to convert Box observations to a dict.
    Acts as a no-op if env.observation_space is not a Box.
    """

    def __init__(self, env: EnvIface, key: str = 'obs'):
        super().__init__(env, self._wrap_obs)
        self.__key = key

        self._need_wrap = False
        self._obs_space = env.observation_space
        if isinstance(env.observation_space, spaces.Box):
            self._need_wrap = True
            self._obs_space = spaces.Dict({self.__key: env.observation_space})

    @property
    def observation_space(self) -> spaces.Space:
        return self._obs_space

    def _wrap_obs(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
        if self._need_wrap:
            return {self.__key: obs}
        return obs
