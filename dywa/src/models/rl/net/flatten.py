#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import einops

from dataclasses import dataclass, fields, replace
from typing import Tuple
from icecream import ic

from models.rl.net.base import FeatureBase
from models.common import (attention, MultiHeadAttentionV2)


class FlattenFeatNet(nn.Module, FeatureBase):
    """
    Masked world-models with simsiam for cls tokens
    """

    @dataclass(init=False)
    class FlattenFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (8, 256)
        dim_out: int = 2048

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self, *args, **kwds):
            pass

    Config = FlattenFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: th.Tensor):
        return x.reshape(*x.shape[:-2], -1)


def main():
    x = th.randn((4, 8, 256))
    model = FlattenFeatNet(FlattenFeatNet.Config())
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
