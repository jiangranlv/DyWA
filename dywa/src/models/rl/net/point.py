#!/usr/bin/env python3

import torch as th
import torch.nn as nn

from dataclasses import dataclass, fields, replace
from typing import Tuple
from icecream import ic

from models.rl.net.base import FeatureBase
from models.sdf.encoder.point_tokens import PointTokenizer

class PointFeatNet(nn.Module, FeatureBase):
    """
    Masked world-models with simsiam for cls tokens
    """

    @dataclass(init=False)
    class PointFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (1024, 3)
        dim_out: int = 128
        model: PointTokenizer.Config = PointTokenizer.Config(
            use_viewpoint=False)

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            self.model = replace(self.model,
                                 output_dim=self.dim_out)

    Config = PointFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = PointTokenizer(cfg.model)

    def forward(self, x: th.Tensor):
        # if not th.isfinite(x).all():
        #     raise ValueError('x is not all finite')
        return self.model(x)


def main():
    from omegaconf import OmegaConf
    device: str = 'cpu'

    with open('/tmp/point.yaml', 'w') as fp:
        OmegaConf.save(OmegaConf.structured(PointFeatNet.Config()),
                       fp)
    cfg = PointFeatNet.Config(dim_out=128)
    cfg.model = replace(cfg.model, patch_size=64)
    model = PointFeatNet(cfg).to(device)
    cloud = th.zeros((1, 512, 3), dtype=th.float,
                     device=device)
    feats = model(cloud)
    print('feats', feats.shape)
    ic(model)


if __name__ == '__main__':
    main()
