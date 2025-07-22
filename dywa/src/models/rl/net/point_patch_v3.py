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
    MLPPatchEncoder
)
# from models.cloud.point_mae import KNNPatchEncoder
from models.rl.net.pw_attn import PatchWiseAttentionFeatNet
from models.common import (transfer, MultiHeadLinear,
                               attention)
from models.cloud.point_mae import (
    PointMAEEncoder,
    get_group_module,
    get_patch_module,
    get_pos_enc_module,
    _get_overlap_patch_params
)
from train.ckpt import load_ckpt
from util.config import recursive_replace_map

from icecream import ic


class PointPatchV3FeatNet(nn.Module):
    """
    Patchwise point features jointly processed with
    proprioceptive inputs and task configuration;
    compatible with point-mae pretraining.
    """
    @dataclass
    class PointPatchV3FeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        dim_out: int = 256

        ctx_dim: int = 0
        keys: Tuple[str, ...] = ()
        num_query: int = 1

        patch_size: int = 32
        encoder_channel: int = 128
        encoder: PointMAEEncoder.Config = PointMAEEncoder.Config()

        # Type of positional embedding
        pos_embed_type: str = 'mlp'
        # TODO: rename to `group-type`
        patch_type: str = 'fps'  # fps/hilbert
        # TODO: rename to `patch-type`
        patch_encoder_type: str = 'mlp'  # mlp/knn/cnn
        # some points might be included in multiple patches
        patch_overlap: float = 1.0
        # apply hilbert-sort for `mlp` patch encoder
        sort_mlp: bool = False
        # Dropout probability
        p_drop: float = 0.0

        ckpt: Optional[str] = None
        freeze_encoder: bool = False

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder = recursive_replace_map(self.encoder, {
                'layer.hidden_size': self.encoder_channel,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop,
            })

    Config = PointPatchV3FeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        (self.sort, self.group,
         self.true_patch_size, self.patch_stride) = get_group_module(
            cfg.patch_type, cfg.patch_size, cfg.patch_overlap
        )
        self.patch_encoder = get_patch_module(cfg.patch_encoder_type,
                                              cfg.encoder_channel,
                                              self.true_patch_size,
                                              sort_mlp=cfg.sort_mlp)
        self.pos_embed = get_pos_enc_module(cfg.pos_embed_type,
                                            cfg.encoder_channel)

        self.encoder = PointMAEEncoder(cfg.encoder)
        assert ((cfg.num_query == 1))
        self.ctx_to_q = MultiHeadLinear(cfg.ctx_dim,
                                        cfg.num_query,
                                        cfg.encoder_channel,
                                        unbind=False)
        self.ctx_to_v = MultiHeadLinear(cfg.ctx_dim,
                                        cfg.num_query,
                                        cfg.encoder_channel,
                                        unbind=False)
        self.layernorm = nn.LayerNorm(cfg.encoder.layer.hidden_size,
                                      eps=cfg.encoder.layer.layer_norm_eps)
        if cfg.ckpt is not None:
            load_ckpt(dict(model=self), cfg.ckpt,
                      strict=False)
        if cfg.freeze_encoder:
            for k, v in self.patch_encoder.named_parameters():
                v.requires_grad_(False)
            for k, v in self.pos_embed.named_parameters():
                v.requires_grad_(False)
            for k, v in self.encoder.named_parameters():
                v.requires_grad_(False)

    def _embed(self, p: th.Tensor, c: th.Tensor,
               noise: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`th.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`th.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        cfg = self.cfg
        z = self.patch_encoder(p)
        pe = self.pos_embed(c)
        z = z + pe
        return z, pe

    def _group(self, x: th.Tensor,
               center: Optional[th.Tensor] = None,
               aux: Optional[Dict[str, th.Tensor]] = None):
        # [1] Parameter-free grouping of points
        cfg = self.cfg
        with th.no_grad():
            if cfg.patch_type == 'hilbert':
                # PATCH BY HILBERT MAPPING
                # First, sort by hilbert code
                x = self.sort(x, aux=aux)
                # Group into normalized patches and
                # patch centers.
                p, c = self.group(x)
            elif cfg.patch_type == 'fps':
                # PATCH BY FPS
                p, c = self.group(x)
            else:
                raise ValueError(F'Unknown patch_type={cfg.patch_type}')
        return (p, c)

    def forward(self, x: th.Tensor,
                ctx: Dict[str, th.Tensor]) -> th.Tensor:
        cfg = self.cfg
        s = x.shape
        x = x.reshape(-1, *s[-2:])

        # Group point cloud into patch.
        p, c = self._group(x)

        # Embed each patch.
        z, pe = self._embed(p, c)

        # Encode patches.
        seq = z
        z, _, a = self.encoder(seq,
                               # output_attentions=True
                               output_attentions=False)
        # needed, or not?
        z = self.layernorm(z)

        # Compute context token.
        # Then project to query and value.
        ctx = th.cat([ctx[k] for k in cfg.keys], dim=-1)
        ctx = ctx.reshape(x.shape[0], -1)

        if True:
            query = self.ctx_to_q(ctx)
            value = self.ctx_to_v(ctx)
            vals = th.cat([value, z], dim=-2)
            emb = attention(query, vals, vals)  # cross attention with state
        else:
            # Cross attention with state
            query = self.ctx_to_q(ctx)
            emb = attention(query, z, z)
            emb = th.cat([emb.squeeze(dim=-2), ctx], dim=-1)

        emb = emb.reshape(*s[:-2], -1)
        out = emb
        return out


def test_forward():
    from omegaconf import OmegaConf
    batch_size: int = 7
    x = th.randn((batch_size, 512, 3),
                 dtype=th.float32,
                 device='cuda')
    ctx = {'a': th.randn((batch_size, 32),
                         device='cuda'), 'b': th.randn((batch_size, 16),
                                                       device='cuda')}
    num_query: int = 1
    cfg = PointPatchV3FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
    )
    OmegaConf.save(OmegaConf.structured(cfg),
                   '/tmp/docker/point_patch.yaml')
    model = PointPatchV3FeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


def test_load():
    from omegaconf import OmegaConf
    num_query = 1
    model_cfg = OmegaConf.load('/tmp/cfg.yaml').model
    cfg = PointPatchV3FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        ctx_dim=48,
        patch_size=model_cfg.patch_size,
        token_size=model_cfg.encoder_channel,
        pos_embed_type=model_cfg.pos_embed_type,
        patch_encoder_type=model_cfg.patch_encoder_type,
        patch_type=model_cfg.patch_type,  # very confusing
        # encoder_channel=model_cfg.encoder_channel,
        encoder_channel=256,  # ??
    )
    # OmegaConf.save(OmegaConf.structured(cfg),
    #                '/tmp/docker/point_patch.yaml')
    model = PointPatchV3FeatNet(cfg)  # .to('cuda')
    load_ckpt(dict(model=model),
              '/tmp/pmaets/run-045/ckpt/epoch-060.ckpt',
              strict=False)
    ic(model)


def test_load_and_forward():
    from omegaconf import OmegaConf

    batch_size: int = 7
    num_query = 1
    device: str = 'cuda:0'

    model_cfg = OmegaConf.load('/tmp/cfg.yaml').model
    cfg = PointPatchV3FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        ctx_dim=48,
        keys=('a', 'b'),
        patch_size=model_cfg.patch_size,
        token_size=model_cfg.encoder_channel,
        pos_embed_type=model_cfg.pos_embed_type,
        patch_encoder_type=model_cfg.patch_encoder_type,
        patch_type=model_cfg.patch_type,  # very confusing
        # encoder_channel=model_cfg.encoder_channel,
        encoder_channel=256,  # ??
    )
    model = PointPatchV3FeatNet(cfg).to(device)
    load_ckpt(dict(model=model),
              '/tmp/pmaets/run-045/ckpt/epoch-060.ckpt',
              strict=False)
    ic(model)

    x = th.randn((batch_size, 512, 3),
                 dtype=th.float32,
                 device=device)
    ctx = {'a': th.randn((batch_size, 32),
                         device=device), 'b': th.randn((batch_size, 16),
                                                       device=device)}

    y = model(x, ctx)
    print(y.shape)
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


def main():
    # test_load()
    test_load_and_forward()


if __name__ == '__main__':
    main()
