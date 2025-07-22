#!/usr/bin/env python3

from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import einops

try:
    from torch_geometric.nn import (
        PointConv, fps, radius, global_max_pool,
        knn_interpolate)
except ImportError:
    print('To run PointNet, please install torch_geometric.')

from models.rl.net.base import FeatureBase


class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = th.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(th.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = th.arange(x.size(0), device=batch.device)
        return x, pos, batch


def make_mlp(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = th.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetEncoder(nn.Module):
    @dataclass
    class Config(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        dim_out: int = 256
        sa_dims: Tuple[int, ...] = (128, 256, 1024)
        keys: Optional[Dict[str, int]] = None

        def __post_init__(self):
            self.dim_out = self.sa_dims[-1]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keys = (dict() if self.cfg.keys is None
                     else self.cfg.keys)
        sa_dims = cfg.sa_dims
        dim_0: int = cfg.dim_in[-1] + sum(self.keys.values(), 0)

        self.sa1_module = SAModule(
            0.2, 0.2, make_mlp([dim_0 + 3, 64, 64, sa_dims[0]]))
        self.sa2_module = SAModule(
            0.25, 0.4,
            make_mlp([sa_dims[0] + 3, 128, 128, sa_dims[1]])
        )
        self.sa3_module = GlobalSAModule(
            make_mlp([sa_dims[1] + 3, 256, 512, sa_dims[2]]))

        self.tokenizer = nn.ModuleDict(dict([(k, nn.Identity())
                                             for k, v in self.keys.items()]))

    def forward(self, x: th.Tensor,
                ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None):
        cfg = self.cfg

        # Concatenate global context reatures as point features.
        tokens = [self.tokenizer[k](ctx[k].reshape(*x.shape[:-2], -1))
                  for k in self.keys
                  if k in ctx]
        token = th.cat(tokens, dim=-1)
        token = einops.repeat(token, '... d -> ... p d',
                              p=x.shape[-2])
        x = th.cat([x, token], dim=-1)

        # Add batch arguments for the homogeneous input clouds
        s = x.shape
        if len(s) > 2:
            x = x.reshape(-1, *s[-2:])
            b = (th.arange(x.shape[0], device=x.device)[:, None].expand(
                *x.shape[:-1]))
            x = x.reshape(-1, x.shape[-1])
            b = b.reshape(*x.shape[:-1])
        else:
            b = th.zeros(*x.shape[:-1],
                         dtype=th.int64,
                         device=x.device)

        sa0_out = (x, x[..., :3], b)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        if aux is not None:
            aux['outputs'] = (
                sa0_out,
                sa1_out,
                sa2_out,
                sa3_out)

        x = sa3_out[0]
        x = x.reshape(*s[:-2], x.shape[-1])
        return x


def main():
    encoder = PointNetEncoder(
        PointNetEncoder.Config(
            keys={'hand_state': 9}
        )
    )
    x = th.randn((1, 2, 512, 3))
    ctx = {'hand_state': th.randn((1, 2, 9))}
    out = encoder(x, ctx)
    print(encoder)
    print(out.shape)


if __name__ == '__main__':
    main()
