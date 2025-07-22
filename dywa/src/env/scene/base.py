#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Iterable


class SceneBase(ABC):

    @abstractmethod
    def setup(self, env: 'EnvBase'):
        pass

    @abstractmethod
    def reset(self, gym, sim, env, indices: Iterable[int]):
        """ reset scene, and potentially apply domain randomization. """
        pass

    @abstractmethod
    def create_actors(self, gym, sim, env):
        """ create scene-related actors. """
        pass

    @abstractmethod
    def create_assets(self, gym, sim):
        """ create assets (shared across envs). """
        pass
