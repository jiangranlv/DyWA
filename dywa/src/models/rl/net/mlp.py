#!/usr/bin/env python3


from typing import (Dict, Tuple, Union, Iterable, Optional)
from collections import Mapping
from dataclasses import dataclass, replace, InitVar
from util.config import ConfigBase

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from models.common import (merge_shapes, MLP, map_struct, get_activation_function)
from models.rl.net.base import (
    FeatureBase, AggregatorBase, FuserBase)

S = Union[int, Tuple[int, ...]]
T = th.Tensor


class MLPFeatNet(nn.Module, FeatureBase):

    @dataclass
    class Config(FeatureBase.Config):
        # mlp: MLP.Config = MLP.Config()
        dim_in: Tuple[int, ...] = ()
        dim_out: int = -1
        dim_hidden: Tuple[int, ...] = ()
        act_cls: str = 'tanh'
        use_bn: bool = False
        use_ln: bool = False
        bias: bool = True
        pre_ln_bias: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # NOTE:
        # MLPFeatNet is currently only designed for
        # uni-dimensional-vector inputs (not multi-dim tensor)!
        if isinstance(cfg.dim_in, Iterable) and len(cfg.dim_in) > 1:
            raise ValueError(
                'MLPFeatNet is currently only designed for' +
                ' uni-dimensional-vector inputs (not multi-dim tensor)!' +
                F' got {cfg.dim_in}; len = {len(cfg.dim_in)}')

        dims = merge_shapes(cfg.dim_in, cfg.dim_hidden, cfg.dim_out)
        act_cls = get_activation_function(cfg.act_cls)
        self.mlp = MLP(dims, act_cls, True, cfg.use_bn, use_ln=cfg.use_ln,
                       bias=cfg.bias,
                       pre_ln_bias=cfg.pre_ln_bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.mlp(x)


# class DictMLPFeatNet(nn.Module, FeatureBase):
#     @dataclass
#     class Config(FeatureBase.Config):
#         # mlp: MLP.Config = MLP.Config()
#         dim_in: Dict[str, S] = None
#         dim_hidden: Dict[str, S] = None
#         dim_outs: Dict[str, int] = None
#         dim_out: int = -1
#         act_cls: str = 'tanh'
#         use_bn: bool = False

#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.cfg = cfg

#         if cfg.dim_outs is None:
#             assert ((cfg.dim_out % len(cfg.dim_in)) == 0)

#         mlps = {}
#         num_field: int = len(cfg.dim_in)
#         for k, v in cfg.dim_in.items():
#             if not isinstance(cfg.dim_hidden, Mapping):
#                 hidden = cfg.dim_hidden
#             else:
#                 hidden = cfg.dim_hidden.get(k, ())

#             if cfg.dim_outs is None:
#                 dim_out = cfg.dim_out  # // len(cfg.dim_in)
#             else:
#                 dim_out = cfg.dim_outs.get(k, dim_out)

#             dims = merge_shapes(
#                 cfg.dim_in[k],
#                 hidden,
#                 dim_out
#             )
#             act_cls = get_activation_function(cfg.act_cls)
#             mlp = MLP(dims, act_cls, True, cfg.use_bn)
#             mlps[k] = mlp

#         self.keys = tuple(sorted(mlps.keys()))
#         self.mlps = nn.ModuleDict(mlps)
#         self.project = MLP((num_field * cfg.dim_out, cfg.dim_out),
#                            act_cls,
#                            True,
#                            cfg.use_bn)

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         outputs = [self.mlps[k](x[k]) for k in self.keys]
#         output = self.project(th.cat(outputs, dim=-1))
#         return output


class MLPFuser(nn.Module, FuserBase):
    @dataclass
    class Config(FuserBase.Config):
        mlp: MLPFeatNet.Config = MLPFeatNet.Config()
        input_dims: InitVar[Union[int, Dict[str, int], None]] = None
        keys: Optional[Tuple[str, ...]] = None

        def __post_init__(self, input_dims=None):
            self.mlp = replace(self.mlp,
                               dim_out=self.dim_out)
            if input_dims is not None:
                dim_in = input_dims
                if isinstance(dim_in, Mapping):
                    # NOTE: np.prod() here
                    # collapses inputs such as (3,) into `3`
                    if self.keys is None:
                        dim_in = int(sum([np.prod(x)
                                     for x in dim_in.values()]))
                    else:
                        dim_in = int(sum([np.prod(v) for k, v in dim_in.items()
                                          if k in self.keys]))
                dim_in = tuple(merge_shapes(dim_in))
                self.mlp = replace(self.mlp,
                                   dim_in=dim_in)

    def __init__(self,
                 cfg: Config,
                 input_dims: Union[int, Dict[str, int], None]):
        super().__init__()
        cfg = replace(cfg, input_dims=input_dims)
        self.cfg = cfg
        self.mlp = map_struct(cfg.mlp,
                              lambda src, _: MLPFeatNet(src),
                              base_cls=MLPFeatNet.Config)

        # FIXME: cannot handle nested mappings
        if isinstance(input_dims, Mapping):
            if cfg.keys is not None:
                # WARN: we do _not_ sort the input keys
                # in case the orderings have semantic meaning.
                self.keys = cfg.keys
            else:
                self.keys = sorted(input_dims.keys())
        else:
            self.keys = None

    def forward(self, x: Union[th.Tensor, Dict[str, th.Tensor]]):
        # 1. concatenate inputs.
        if self.keys is None:
            x = x
        else:
            try:
                x = th.cat([x[k] for k in self.keys], dim=-1)
            except TypeError:
                for k in self.keys:
                    print(k)
                    if isinstance(x[k], th.Tensor):
                        print(x[k].dtype, x[k].shape)
                    else:
                        print(x[k])
                raise

        # 2. run through MLP layers.
        # print(F'get {x}, {x.min()}, {x.max()}')
        x = self.mlp(x)
        # for k,v in self.mlp.named_parameters():
        #    print(k)
        #    print(v.min(), v.max())
        # print(F'set {x}, {x.min()}, {x.max()}')
        return x


class MLPAggNet(nn.Module, AggregatorBase):
    @dataclass
    class Config(AggregatorBase.Config):
        dim_obs: Tuple[int, ...] = ()
        dim_act: int = -1
        dim_out: int = -1

        # Extra args for MLP
        dim_hidden: Tuple[int, ...] = ()
        act_cls: str = 'tanh'
        use_bn: bool = False
        use_ln: bool = False
        bias: bool = True
        pre_ln_bias: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        dim_obs = merge_shapes(cfg.dim_obs)
        dim_act = merge_shapes(cfg.dim_act)
        assert (len(dim_obs) == 1)
        assert (len(dim_act) == 1)
        self.dim_obs = dim_obs
        self.dim_act = dim_act

        dim_in: int = (dim_obs[0] + dim_act[0])
        dims = merge_shapes(dim_in,
                            cfg.dim_hidden,
                            cfg.dim_out)
        self.dim_in = dim_in

        act_cls = get_activation_function(cfg.act_cls)
        self.mlp = MLP(dims, act_cls, True, cfg.use_bn,
                       use_ln=cfg.use_ln,
                       bias=cfg.bias,
                       pre_ln_bias=cfg.pre_ln_bias)

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
        cfg = self.cfg

        # Convert indices into one-hot representation.
        if (a is not None) and (self.dim_act[0] > 0):
            if not th.is_floating_point(a):
                a = F.one_hot(a.long(), self.dim_act[0])
            ao = th.cat([a, o], dim=-1)
        else:
            ao = o
        return self.mlp(ao)
