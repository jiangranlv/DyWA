#!/usr/bin/env python3


from typing import (Tuple, Union)
from dataclasses import dataclass, fields
from util.config import ConfigBase

import torch as th
import torch.nn as nn
import torch.nn.functional as F


from models.common import merge_shapes


from models.rl.net.base import AggregatorBase

S = Union[int, Tuple[int, ...]]
T = th.Tensor


class MultiLayerGRUAggNet(nn.Module):
    @dataclass
    class Config(ConfigBase):
        dim_obs: Tuple[int, ...] = ()
        dim_act: int = -1
        dim_out: int = -1
        num_layer: int = 2

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim_in: int = (cfg.dim_obs + cfg.dim_act)
        self.dim_in = dim_in
        self.gru = nn.GRU(dim_in, cfg.dim_out,
                          cfg.num_layer,
                          batch_first=False)
        # (x=T,N,Hi, h=L,N,Ho)

    def init_state(self, batch_shape: S,
                   *args, **kwds):
        cfg = self.cfg
        S = merge_shapes(cfg.num_layer, batch_shape, self.cfg.dim_out)
        return th.zeros(S, *args, **kwds)

    def forward(self,
                h_0: th.Tensor,
                a: th.Tensor,
                o: th.Tensor) -> th.Tensor:
        cfg = self.cfg

        # Convert indices into one-hot representation.
        T: int = a.shape[0]
        D: int = a.shape[-1]
        L: int = h_0.shape[0]  # == cfg.num_layes

        # Combine action and observation to one tensor.
        if not th.is_floating_point(a):
            a = F.one_hot(a.long(), cfg.dim_act)
        ao = th.cat([a, o], dim=-1)

        # Flatten and feed through GRU.
        # We assume broadcasting(batch) in the middle dimensions.
        aor = ao.reshape(T, -1, self.dim_in)
        h0r = h_0.reshape(L, -1, cfg.dim_out)
        out, h_n = self.gru(aor, h0r)

        # Reshape outputs and return.
        out = out.reshape(*a.shape[:-1], out.shape[-1])
        h_n = h_n.reshape(h_0.shape)
        return (out, h_n)


class GRUAggNet(nn.Module, AggregatorBase):

    @dataclass(init=False)
    class Config(AggregatorBase.Config):
        #TODO: dim act = to disalbe action concat
        dim_obs: Tuple[int, ...] = ()
        dim_act: int = -1
        dim_out: int = -1

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            pass

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # FIXME: currently can only handle
        # flattened obs,act !!!!
        dim_obs = merge_shapes(cfg.dim_obs)
        dim_act = merge_shapes(cfg.dim_act)
        assert (len(dim_obs) == 1)
        assert (len(dim_act) == 1)

        dim_in: int = (dim_obs[0] + dim_act[0])

        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.dim_in = dim_in
        self.gru = nn.GRUCell(dim_in, cfg.dim_out)
        # self.gru = LayerNormGRUCell(dim_in, cfg.dim_out)

        # with th.no_grad():
        #     nn.init.orthogonal_(self.gru.weight_ih, gain=1.0)
        #     nn.init.orthogonal_(self.gru.weight_hh, gain=1.0)
        #     self.gru.bias_ih.fill_(0.0)
        #     self.gru.bias_hh.fill_(0.0)

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
        aor = ao.reshape(-1, self.dim_in)
        # print('aor', aor)
        h0r = h0.reshape(-1, cfg.dim_out)
        h1 = self.gru(aor, h0r)
        h1r = h1.reshape(h0.shape)
        # print('h1r', h1r)
        return h1r


def test_gru():
    O: int = 13
    A: int = 11
    H: int = 7
    L: int = 5
    T: int = 3
    N: int = 2
    gru_2 = MultiLayerGRUAggNet(
        MultiLayerGRUAggNet.Config(O, A, H, L))

    o = th.zeros((T, N, O))
    a = th.zeros((T, N, A))
    h0 = th.zeros((L, N, H))
    out, hn = gru_2(h0, a, o)
    print('out', out.shape)
    print('hn', hn.shape)
