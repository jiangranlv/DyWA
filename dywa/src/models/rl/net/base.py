#!/usr/bin/env python3

from abc import ABC, abstractproperty
from typing import Tuple, Union
from dataclasses import dataclass
from util.config import ConfigBase


class FeatureBase(ABC):
    @dataclass
    class Config(ConfigBase):
        dim_in: Tuple[int, ...] = ()
        dim_out: int = -1


class AggregatorBase(ABC):
    @dataclass
    class Config(ConfigBase):
        dim_obs: Tuple[int, ...] = ()
        dim_act: int = -1
        dim_out: int = -1

    @abstractproperty
    def dim_out(self) -> int:
        # return self.cfg.dim_out
        pass


class FuserBase(ABC):
    @dataclass
    class Config(ConfigBase):
        # dim_in: Tuple[int, ...] = ()
        dim_out: int = -1
