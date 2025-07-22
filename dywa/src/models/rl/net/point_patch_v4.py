#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import einops

from dataclasses import dataclass, fields
from typing import Tuple, Dict, Optional

from models.rl.net.base import FeatureBase
from models.sdf.encoder.point_tokens import (
    SpatialSort, HilbertCode)
from models.cloud.point_mae import (
    KNNPatchEncoder,
    MLPPatchEncoder,
    # GroupFPS
    get_group_module,
    get_pos_enc_module
)
# from models.cloud.point_mae import KNNPatchEncoder
from models.rl.net.pw_attn import PatchWiseAttentionFeatNet

from icecream import ic

from util.vis.attn import rollout, draw_patch_attentions


class PointPatchV4FeatNet(nn.Module, FeatureBase):
    """
    successor of PointPatchFeatNet
    can use fps grouping
    """

    @dataclass(init=False)
    class PointPatchV4FeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        # dim_out: Tuple[int, ...] = (16, 256)  # possible?
        dim_out: int = 256  # possible?
        knn: int = 8
        hidden: int = 128
        patch_size: int = 32
        token_size: int = 64
        group_type: str = 'fps'
        patch_type: str = 'knn'
        pos_embed_type: str = 'mlp'
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

    Config = PointPatchV4FeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        (self.sort, self.group,
         self.true_patch_size, self.patch_stride) = get_group_module(
            cfg.group_type, cfg.patch_size, 1.0
        )
        # self.code = th.jit.script(HilbertCode())
        # self.sort = SpatialSort(self.code)
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
        self.pos_embed = get_pos_enc_module(cfg.pos_embed_type,
                                            cfg.token_size)

    def _group(self, x: th.Tensor,
               center: Optional[th.Tensor] = None,
               aux: Optional[Dict[str, th.Tensor]] = None):
        cfg = self.cfg
        with th.no_grad():
            if cfg.group_type == 'hilbert':
                assert (center is None)
                # PATCH BY HILBERT MAPPING
                # First, sort by hilbert code
                x = self.sort(x, aux=aux)
                # Group into normalized patches and
                # patch centers.
                p, c = self.group(x, center=center, aux=aux)
            elif cfg.group_type == 'fps':
                # PATCH BY FPS
                p, c = self.group(x, center=center, aux=aux)
            else:
                raise ValueError(F'Unknown group_type={cfg.group_type}')
        return (p, c)

    def forward(self, x: th.Tensor, ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None) -> th.Tensor:
        cfg = self.cfg
        s = x.shape

        # to patch
        # if aux is None:
        #    aux = {'output_attn': True}
        p, c = self._group(x, aux=aux)

        # Embed each patch
        x0 = x
        z = self.patch(p)
        pe = self.pos_embed(c)
        z = z + pe

        # encoder
        if False:
            x = self.encoder(z, ctx, aux=aux)
            a = aux['attn']
            ic(list(aux.keys()))

            self._patch_clouds = (p + c[..., None, :]).reshape(p.shape)
            self._patch_index = aux['fps_nn_idx'].reshape(
                *s[:-2], -1, cfg.patch_size)
            self._patch_colors = draw_patch_attentions(a)
        else:
            x = self.encoder(z, ctx)

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
    cfg = PointPatchV4FeatNet.Config(
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
    model = PointPatchV4FeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


if __name__ == '__main__':
    main()
