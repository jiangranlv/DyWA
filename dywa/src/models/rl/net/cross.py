#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import einops

from dataclasses import dataclass, fields, replace
from typing import Tuple
from icecream import ic

from models.rl.net.base import FeatureBase
from models.common import (attention, MultiHeadAttentionV2)


class CrossFeatNet(nn.Module, FeatureBase):
    """
    Masked world-models with simsiam for cls tokens
    """

    @dataclass(init=False)
    class CrossFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (8, 256)
        dim_out: int = 128
        num_emb_token: int = 4

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self, *args, **kwds):
            pass

    Config = CrossFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_parameter(
            'query_token',
            nn.Parameter(
                th.zeros(cfg.num_emb_token, cfg.dim_in[-1]),
                requires_grad=True
            ))

    def forward(self, x: th.Tensor):
        # x should look like (..., NUM_KEY, EMBED_DIM)
        q = self.query_token
        q = q.broadcast_to(*x.shape[:-2], *q.shape[-2:])
        out = attention(q, x, x)
        return out.reshape(*x.shape[:-2], -1)


def main():
    x = th.randn((4, 8, 256))
    model = CrossFeatNet(CrossFeatNet.Config())
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
