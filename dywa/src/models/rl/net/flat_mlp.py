#!/usr/bin/env python3

from dataclasses import dataclass, replace, fields

import numpy as np
import torch as th

from models.rl.net.mlp import MLPFeatNet
from models.common import (merge_shapes)


class FlatMLPFeatNet(MLPFeatNet):

    @dataclass(init=False)
    class Config(MLPFeatNet.Config):
        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            # self.__post_init__()

    def __init__(self, cfg: MLPFeatNet.Config):
        dim_in = merge_shapes(cfg.dim_in)
        self._rank = len(cfg.dim_in)
        dim_in = int(np.prod(dim_in))
        inner_cfg = replace(cfg, dim_in=(dim_in,))
        super().__init__(inner_cfg)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x.reshape(*x.shape[:-self._rank], -1)
        out = super().forward(x)
        return out


def main():
    model = FlatMLPFeatNet(MLPFeatNet.Config(
        dim_in=(2, 48, 48),
        dim_out=64))
    x = th.zeros((4, 2, 48, 48))
    print(model)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
