#!/usr/bin/env python3


from typing import (Tuple, Union, Optional, Dict, Mapping)
from dataclasses import dataclass, fields, InitVar, replace

import numpy as np
import torch as th
import torch.nn as nn


from models.common import merge_shapes
from models.rl.net.base import (
    FeatureBase, AggregatorBase, FuserBase)
from util.config import ConfigBase

S = Union[int, Tuple[int, ...]]
T = th.Tensor


class NoOpAggNet(nn.Module, AggregatorBase):

    @dataclass(init=False)
    class Config(AggregatorBase.Config):
        dim_obs: Tuple[int, ...] = ()
        dim_out: int = -1

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            if self.dim_obs != self.dim_out:
                print(F'{self.dim_obs} != {self.dim_out}!! overwriting.')
                dim_obs = merge_shapes(self.dim_obs)
                if len(dim_obs) == 1:
                    self.dim_out = dim_obs[0]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        try:
            assert (merge_shapes(cfg.dim_obs)[0] == cfg.dim_out)
        except AssertionError:
            print(merge_shapes(cfg.dim_obs)[0], cfg.dim_out)
            raise

    @property
    def dim_out(self) -> S:
        return self.cfg.dim_out

    def init_state(self, batch_shape: S,
                   *args, **kwds):
        S = merge_shapes(batch_shape, self.cfg.dim_out)
        return th.zeros(S, *args, **kwds)

    def forward(self,
                h0: th.Tensor,
                a: th.Tensor,
                o: th.Tensor) -> th.Tensor:
        return o


class NoOpFeatNet(nn.Module, FeatureBase):

    @dataclass(init=False)
    class Config(FeatureBase.Config):
        dim_in: Tuple[int, ...] = ()
        dim_out: int = -1
        dim_hidden: Optional[Tuple[int, ...]] = None

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            if self.dim_in != self.dim_out:
                print(F'{self.dim_in} != {self.dim_out}!! overwriting.')
                dim_in = merge_shapes(self.dim_in)
                if len(dim_in) == 1:
                    self.dim_out = dim_in[0]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        assert (merge_shapes(cfg.dim_in)[0] == cfg.dim_out)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x


class CatFuser(nn.Module, FuserBase):
    @dataclass(init=False)
    class Config(FuserBase.Config):
        input_dims: InitVar[Union[int, Dict[str, int], None]] = None
        dim_out: int = -1

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__(**kwds)

        def __post_init__(self, input_dims=None, **kwds):
            if input_dims is not None:
                dim_in = input_dims
                if isinstance(dim_in, Mapping):
                    # NOTE: np.prod() here
                    # collapses inputs such as (3,) into `3`
                    dim_in = int(sum([np.prod(x) for x in dim_in.values()]))
                self.dim_out = int(merge_shapes(dim_in)[0])

    def __init__(self,
                 cfg: Config,
                 input_dims: Union[int, Dict[str, int], None]):
        super().__init__()
        cfg = replace(cfg, input_dims=input_dims)
        self.cfg = cfg

        # FIXME: cannot handle nested mappings
        if isinstance(input_dims, Mapping):
            self.keys = sorted(input_dims.keys())
        else:
            self.keys = None

    def forward(self, x: Union[th.Tensor, Dict[str, th.Tensor]]):
        # 1. concatenate inputs.
        if self.keys is None:
            x = x
        else:
            x = th.cat([x[k] for k in self.keys], dim=-1)
        return x
