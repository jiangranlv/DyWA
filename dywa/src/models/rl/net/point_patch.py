#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import einops
import cv2
import numpy as np

from dataclasses import dataclass, fields
from typing import Tuple, Dict, Optional

from models.rl.net.base import FeatureBase
from models.sdf.encoder.point_tokens import (
    SpatialSort, HilbertCode)
from models.cloud.point_mae import (
    KNNPatchEncoder,
    MLPPatchEncoder
)
# from models.cloud.point_mae import KNNPatchEncoder
from models.rl.net.pw_attn import PatchWiseAttentionFeatNet
from util.torch_util import dcn
from util.vis.attn import rollout, draw_patch_attentions

from icecream import ic


class PointPatchFeatNet(nn.Module, FeatureBase):
    """
    only patchifies point clouds, without learnable operations.
    """

    @dataclass(init=False)
    class PointPatchFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        # dim_out: Tuple[int, ...] = (16, 256)  # possible?
        dim_out: int = 256  # possible?
        knn: int = 8
        hidden: int = 128
        patch_size: int = 32
        token_size: int = 64
        patch_type: str = 'knn'
        pw_attn: PatchWiseAttentionFeatNet.Config = (
            PatchWiseAttentionFeatNet.Config(
                dim_in=(dim_in[0] / patch_size, token_size),
                dim_out=dim_out)
        )
        ckpt: Optional[str] = None

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)

    Config = PointPatchFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.code = th.jit.script(HilbertCode())
        self.sort = SpatialSort(self.code)
        # assert (cfg.dim_in[0] % cfg.dim_out[0] == 0)
        # patch_size: int = cfg.dim_in[0] // cfg.dim_out[0]
        if cfg.patch_type == 'knn':
            self.patch = KNNPatchEncoder(cfg.patch_size,
                                         cfg.token_size,
                                         d=cfg.dim_in[-1],
                                         k=cfg.knn,
                                         f=cfg.hidden)
        elif cfg.patch_type == 'mlp':
            self.patch = MLPPatchEncoder(MLPPatchEncoder.Config([cfg.hidden]),
                                         cfg.patch_size,
                                         cfg.token_size)
        else:
            raise ValueError(F'Unknown patch_type = {cfg.patch_type}')
        self.encoder = PatchWiseAttentionFeatNet(cfg.pw_attn)

        # self._patch_clouds = None
        self._patch_index = None
        # self._patch_colors = None
        self._patch_attn = None

    def forward(self, x: th.Tensor, ctx: Dict[str, th.Tensor]) -> th.Tensor:
        cfg = self.cfg
        s = x.shape

        # to patch
        aux = {}
        x = self.sort(x, aux=aux)
        x = einops.rearrange(x, '... (s p) d -> ... s p d',
                             p=cfg.patch_size)
        x0 = x

        # to token
        x = self.patch(x)

        # Encoder
        if True:
            aux['output_attn'] = True
            x = self.encoder(x, ctx, aux=aux)
            a = aux['attn']

            # self._patch_clouds = x0
            self._patch_index = aux['sort_index'].reshape(
                *s[:-2], -1, cfg.patch_size)
            # self._patch_colors = draw_patch_attentions(a)
            self._patch_attn = a[-1][..., 0, 1:].max(dim=-2).values
        else:
            x = self.encoder(x, ctx, aux=aux)

        return x.reshape(*s[:-2], self.cfg.dim_out)


def main():
    from omegaconf import OmegaConf
    batch_size: int = 7
    x = th.randn((batch_size, 512, 3),
                 dtype=th.float32,
                 device='cuda')
    ctx = {'a': th.randn((batch_size, 32),
                         device='cuda'), 'b': th.randn((batch_size, 16),
                                                       device='cuda')}
    num_query: int = 1
    cfg = PointPatchFeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        pw_attn=PatchWiseAttentionFeatNet.Config(
            dim_in=(16, 64),
            dim_out=256,
            keys=['a', 'b'],
            num_query=num_query,
            ctx_dim=32 + 16
        )
    )
    OmegaConf.save(OmegaConf.structured(cfg),
                   '/tmp/docker/point_patch.yaml')
    model = PointPatchFeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


if __name__ == '__main__':
    main()
