#!/usr/bin/env python3

from typing import Dict, Optional

import torch as th
import torch.nn as nn
import einops
import numpy as np

from dataclasses import dataclass, fields, replace
from typing import Tuple
from icecream import ic

from models.rl.net.base import FeatureBase
from models.common import (attention, MultiHeadAttentionV2,
                               MultiHeadLinear)
from models.cloud.point_mae import (
    PointMAEEncoder,
)
from util.config import recursive_replace_map
from matplotlib import pyplot as plt
from util.torch_util import dcn
from util.vis.attn import rollout


class PatchWiseAttentionFeatNet(nn.Module, FeatureBase):
    """
    State-dependet cross attention, where we derive the query
    from a linear projection of the current task specification.
    """

    @dataclass(init=False)
    class PatchWiseAttentionFeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (8, 256)
        dim_out: int = 128
        keys: Optional[Tuple[str, ...]] = None
        num_query: int = 4
        ctx_dim: int = 0
        emb_dim: int = 256
        encoder: PointMAEEncoder.Config = recursive_replace_map(
            PointMAEEncoder.Config(), {'layer.hidden_size': 256})

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self, *args, **kwds):
            self.emb_dim = self.encoder.layer.hidden_size

    Config = PatchWiseAttentionFeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_dim: int = (cfg.dim_in[-1])
        self.tokenize = MultiHeadLinear(
            cfg.ctx_dim,
            cfg.num_query,
            cfg.emb_dim,
            unbind=False)
        self.encoder = PointMAEEncoder(cfg.encoder)
        self.in_proj = nn.Linear(in_dim, cfg.emb_dim)
        self.out_proj = nn.Linear(cfg.num_query * cfg.emb_dim, cfg.dim_out)

    def forward(self, x: th.Tensor, ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> th.Tensor:
        # x should look like (..., NUM_KEY, EMBED_DIM)
        cfg = self.cfg
        if cfg.keys is None:
            return None
        s = x.shape
        x = x.reshape(-1, *s[-2:])
        x = self.in_proj(x)

        # Build ctx->query mapping
        ctx = th.cat([ctx[k] for k in cfg.keys], dim=-1)
        ctx = ctx.reshape(-1, ctx.shape[-1])
        tok = self.tokenize(ctx)  # N, 1, D

        seq = th.cat([tok, x], dim=-2)

        output_attn: bool = False
        try:
            output_attn = aux['output_attn']
        except (TypeError, KeyError):
            pass
        z, _, a = self.encoder(seq,
                               output_attentions=output_attn
                               )

        if aux is not None:
            aux['attn'] = a

        if False:
            m = rollout(a, 0.5)
            m = dcn(m)
            ax = plt.gcf().subplot_mosaic(
                [[str(i)] for i in range(len(m))])
            for i, mm in enumerate(m):
                k = str(i)
                plt.clf()
                # plt.imshow(m, cmap='viridis')
                m = dcn(mm.ravel())
                print(m)
                ax[k].bar(np.arange(len(m)), m)
                # plt.colorbar()
            plt.pause(0.01)

        if False:
            # [L], 64, 4, 17, 17
            # Show first attention...? I guess??
            a0 = a[-1]
            # print('a0', a0.shape)
            attn = a0[0, :, 0, :]  # 64,4,17
            # print(attn[0])  # head 1
            # print(attn[1])  # head 2
            # print(attn[2])  # head 3
            # print(attn[3])  # head 4
            plt.clf()
            plt.imshow(dcn(attn), cmap='viridis')
            plt.colorbar()
            plt.pause(0.01)

        # z = layernorm(z) # needed
        emb = z[..., :tok.shape[-2], :]
        emb = emb.reshape(*x.shape[:-2], -1)
        out = self.out_proj(emb)
        out = out.reshape(*s[:-2], -1)
        return out


def main():
    from omegaconf import OmegaConf
    batch_size: int = 7
    num_query: int = 4
    # x = th.randn((batch_size, 8, 256),
    #              dtype=th.float32,
    #              device='cuda')
    # ctx = {'a': th.randn((batch_size, 32),
    #                      device='cuda'), 'b': th.randn((batch_size, 16),
    #                                                    device='cuda')}
    cfg = PatchWiseAttentionFeatNet.Config(
        dim_in=(8, 256),
        dim_out=256,
        keys=['a', 'b'],
        num_query=num_query,
        ctx_dim=32 + 16
    )
    OmegaConf.save(OmegaConf.structured(cfg), '/tmp/docker/pw_attn.yaml')
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


if __name__ == '__main__':
    main()
