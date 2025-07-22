#!/usr/bin/env python3

from typing import Optional

from abc import ABC, abstractmethod, abstractproperty

import torch as th


class EnvIface(ABC):
    """
    Basic (Isaac) Gym interface.
    Generally follows the interface from openai gym,
    but assumes that the environment will run in a vectorized manner.

    > CONSIDER ADDING:
    >> `device`
    >> `num_env`,
    >> `timeout`
    arguments, as additional specification.
    """

    @abstractproperty
    def action_space(self):
        pass

    @abstractproperty
    def observation_space(self):
        pass

    @abstractproperty
    def device(self) -> str:
        pass

    @abstractproperty
    def num_env(self) -> int:
        pass

    @abstractproperty
    def timeout(self) -> Optional[int]:
        pass

    # Should the discount factor be included
    # in the domain spec?
    # @abstractproperty
    # def gamma(self) -> Optional[int]:
    #     pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reset_indexed(self, indices: Optional[th.Tensor]):
        pass

    @abstractmethod
    def step(self, actions: th.Tensor):
        pass
