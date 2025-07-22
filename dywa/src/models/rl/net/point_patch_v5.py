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
    mask,
    PointMAEEncoder,
    KNNPatchEncoder,
    MLPPatchEncoder,
    # GroupFPS
    get_group_module,
    get_pos_enc_module
)
# from models.cloud.point_mae import KNNPatchEncoder
from models.rl.net.pw_attn import PatchWiseAttentionFeatNet
from models.rl.net.icp import ICPNet
from util.config import recursive_replace_map
from models.common import (attention, MultiHeadAttentionV2,
                               MultiHeadLinear)

from icecream import ic
from util.vis.attn import rollout, draw_patch_attentions
import nvtx


class PseudoStudent(nn.Module):
    @dataclass
    class Config:
        encoder: PointMAEEncoder.Config = PointMAEEncoder.Config()
        mask_ratio: float = 0.9
        ctx_dim: int = -1
        emb_dim: int = 128
        num_query: int = 1

        # Coefficient to try to normalize
        # the time indices by a bit
        time_coef: float = 0.1

        # Inspired by dreamer-v2
        alpha: float = 0.9

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # positional encoding across time+space
        self.pos_enc = get_pos_enc_module('linear',
                                          cfg.encoder.layer.hidden_size,
                                          1)
        ic(cfg.encoder)
        self.encoder = PointMAEEncoder(cfg.encoder)
        self.to_q = MultiHeadLinear(
            # context + time encoding
            cfg.ctx_dim + cfg.encoder.layer.hidden_size,
            cfg.num_query,
            cfg.emb_dim,
            unbind=False)
        self.to_kv = MultiHeadLinear(cfg.encoder.layer.hidden_size,
                                     2, cfg.emb_dim,
                                     unbind=True)
        self.loss = nn.MSELoss()

    def forward(self,
                x: th.Tensor,
                q: th.Tensor,
                target: th.Tensor,
                aux: Dict[str, th.Tensor]):
        cfg = self.cfg
        s = q.shape  # T B Q D

        # == ADD TIME-AWARE POSITIONAL ENCODING ==
        T: int = s[0]
        # Relative time indices, -t-1 ~ 0
        # Does this actually matter ??
        t = cfg.time_coef * th.arange(-T + 1, 1, dtype=th.float,
                                      device=x.device)
        pe_t = self.pos_enc(t[..., None])  # T 1 -> T D
        x = x + einops.repeat(pe_t, 't d -> (t b) s d',
                              b=s[1],
                              s=x.shape[-2])

        # == APPLY MASKING, AND ENCODE ==
        x = einops.rearrange(x,
                             '(t b) s d -> b (t s) d',
                             t=s[0])
        z_keep, m, ids_restore = mask(x, cfg.mask_ratio)
        enc, _, _ = self.encoder(z_keep)  # b s', d

        # Convert encoding to key/value paris for attention.
        k, v = self.to_kv(enc)

        # Create the query token as a product
        # of the query context and the time information.
        # TODO: concatenation of pe_t seems a bit expensive here...
        q = th.cat([q, (pe_t[..., None, :]).expand(-1, q.shape[-2], -1)],
                   dim=-1)
        q = self.to_q(q)

        # Decode the output state representation via cross attention.
        out = attention(q, k, v, aux=aux,
                        key_padding_mask=aux.get('key_padding_mask', None))
        out = out.reshape(*out.shape[:-2], -1)

        # Compute the (balanced) loss. Basically, for alpha=0.9
        # we train the student more than the teacher (9x)
        # and the teacher is weakly regularized toward the student (1x).
        loss = th.lerp(self.loss(out.detach(), target),
                       self.loss(out, target.detach()),
                       cfg.alpha)
        return loss


class PointPatchV5FeatNet(nn.Module, FeatureBase):
    """
    successor of PointPatchFeatNet
    can use fps grouping
    """

    @dataclass(init=False)
    class PointPatchV5FeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        # dim_out: Tuple[int, ...] = (16, 256)  # possible?
        dim_out: int = 256  # possible?
        knn: int = 8
        icp: ICPNet.Config = recursive_replace_map(
            ICPNet.Config(), {
                'encoder_channel': 256,
                'encoder.num_hidden_layers': 4,
                'patch_size': 32,
                'p_drop': 0.0,
                # 'patch_encoder_type': 'knn',
                'patch_encoder_type': 'mlp',
                'patch_overlap': 1.0,
                # 'patch_type': 'hilbert'
                'group_type': 'fps',
                'patch_type': 'mlp',

                'keys': {'hand_state': 7},
                'headers': []
            })
        query_keys: Optional[Tuple[str, ...]] = None
        ctx_dim: int = 0
        emb_dim: int = 128
        num_query: int = 4
        cat_query: bool = False
        cat_ctx: bool = False

        add_student: bool = False
        student: PseudoStudent.Config = PseudoStudent.Config()

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            self.student = recursive_replace_map(
                self.student, {
                    'encoder.layer.hidden_size': self.icp.encoder_channel,
                    'emb_dim': self.emb_dim,
                    'ctx_dim': self.ctx_dim,
                    'num_query': self.num_query
                })

    Config = PointPatchV5FeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ICPNet(cfg.icp)
        self.to_q = MultiHeadLinear(
            cfg.ctx_dim,
            cfg.num_query,
            cfg.emb_dim,
            unbind=False)
        self.to_kv = MultiHeadLinear(cfg.icp.encoder_channel,
                                     2, cfg.emb_dim, unbind=True)
        self._patch_attn = None
        self._patch_index = None

        # Hmm...
        self.student = None
        if cfg.add_student:
            self.student = PseudoStudent(cfg.student)
        if cfg.cat_ctx and cfg.cat_query:
            raise ValueError("both cat_ctx and cat_query cannot be true!")

    def hook_aux_loss(self, on_loss):
        if self.student is not None:
            self.student.register_forward_hook(on_loss)

        if self.encoder.vq is not None:
            self.encoder.output_vq_loss.register_forward_hook(on_loss)

    @nvtx.annotate("PointPatchV5FeatNet.forward")
    def forward(self, x: th.Tensor, ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None) -> th.Tensor:
        cfg = self.cfg
        _aux = {}
        _, emb = self.encoder(x, ctx, _aux)
        self._patch_index = _aux['fps_nn_idx']

        query_ctx = th.cat([ctx[k] for k in cfg.query_keys], -1)
        query = self.to_q(query_ctx)
        k, v = self.to_kv(emb)
        out = attention(query, k, v, aux=_aux,
                        key_padding_mask=_aux.get('key_padding_mask', None))
        self._patch_attn = _aux['attn']

        if aux is not None:
            _aux['cross'] = _aux.get('attn', None)
            aux.update(_aux)
        out = out.reshape(*x.shape[:-2], -1)

        if self.training and cfg.add_student:
            with nvtx.annotate("student"):
                _ = self.student(_aux['z'], query_ctx, out, _aux)

        if cfg.cat_query:
            return th.cat([out, query.reshape(*x.shape[:-2], -1)], -1)
        elif cfg.cat_ctx:
            return th.cat([out, query_ctx], -1)
        else:
            return out


def test_inference():
    from omegaconf import OmegaConf
    batch_size: int = 7
    x = th.randn((batch_size, 512, 3),
                 dtype=th.float32,
                 device='cuda')
    ctx = {'hand_state': th.randn((batch_size, 7),
                                  device='cuda'),
           'b': th.randn((batch_size, 16),
                         device='cuda'),
           'c': th.randn((batch_size, 5), device="cuda"),
           'd': th.randn((batch_size, 45), device="cuda")}
    num_query: int = 4
    keys = {'hand_state': 7, 'b': 16}

    cfg = PointPatchV5FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        encoder=ICPNet.Config(
            keys=keys,
            ckpt='/tmp/docker/icp.ckpt',
            encoder_channel=256
        ),
        ctx_dim=50,
        emb_dim=128,
        num_query=num_query,
        query_keys=['c', 'd']
    )
    OmegaConf.save(OmegaConf.structured(cfg),
                   '/tmp/docker/point_patch.yaml')
    model = PointPatchV5FeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)
    # model = PatchWiseAttentionFeatNet().cuda()
    # y = model(x, ctx=ctx)
    # print(y.shape)
    # ic(model)


def test_student():
    from torchview import draw_graph
    num_frames: int = 10
    batch_size: int = 7
    x = th.randn((num_frames, batch_size, 512, 3),
                 dtype=th.float32,
                 device='cuda')
    ctx = {'hand_state': th.randn((num_frames, batch_size, 7),
                                  device='cuda'),
           'b': th.randn((num_frames, batch_size, 16),
                         device='cuda'),
           'c': th.randn((num_frames, batch_size, 5), device="cuda"),
           'd': th.randn((num_frames, batch_size, 45), device="cuda")}
    num_query: int = 4
    keys = {'hand_state': 7, 'b': 16}

    cfg = PointPatchV5FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        encoder=ICPNet.Config(
            keys=keys,
            ckpt='/tmp/docker/icp.ckpt',
            encoder_channel=256
        ),
        ctx_dim=50,
        emb_dim=128,
        num_query=num_query,
        query_keys=['c', 'd'],
        add_student=True
    )
    model = PointPatchV5FeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)

    model_graph = draw_graph(model,
                             input_data=(x, ctx),
                             # input_size=x.shape,
                             device='cuda')
    model_graph.visual_graph.render('/tmp/ppv5-student.svg')


def main():
    test_student()


if __name__ == '__main__':
    main()
