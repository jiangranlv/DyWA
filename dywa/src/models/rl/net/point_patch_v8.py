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
from omegaconf import OmegaConf
import sys
import logging

try:
    sys.path.append("/tmp/point2vec/")
    from point2vec.models import Point2Vec
except ImportError:
    logging.warn('Skipping point2vec import.')

BN_MAX_NUM = 131070


class PointPatchV8FeatNet(nn.Module, FeatureBase):
    """
    successor of PointPatchFeatNet
    Use pretrained point2vector encoder
    """

    @dataclass(init=False)
    class PointPatchV8FeatNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        # dim_out: Tuple[int, ...] = (16, 256)  # possible?
        dim_out: int = 256  # possible?
        cfg_path: str = "/tmp/p2v/shapenet.yaml"
        ckpt_path: str = "/tmp/p2v/pre_point2vec-epoch.799-step.64800.ckpt"

        query_keys: Optional[Tuple[str, ...]] = None
        ctx_dim: int = 0
        emb_dim: int = 128
        num_query: int = 4
        cat_query: bool = False
        is_finetune: bool = False

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            pass

    Config = PointPatchV8FeatNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Load p2v config
        self.p2v = OmegaConf.load(cfg.cfg_path)
        self.encoder = Point2Vec(**OmegaConf.to_container(self.p2v.model))
        self.num_group = self.p2v.model.tokenizer_num_groups
        # Load pretrained point2vec encoder
        ckpt = th.load(cfg.ckpt_path)
        self.encoder.load_state_dict(ckpt['state_dict'], strict=False)
        self.encoder.to(device='cuda')
        # Frozen encoder
        self.encoder.eval()
        self.to_q = MultiHeadLinear(
            cfg.ctx_dim,
            cfg.num_query,
            cfg.emb_dim,
            unbind=False)

        self.to_kv = MultiHeadLinear(self.p2v.model.encoder_dim,
                                     2, cfg.emb_dim, unbind=True)
        self._patch_attn = None
        self._patch_index = None
        for name, param in self.encoder.named_parameters():
            if not cfg.is_finetune:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        if cfg.is_finetune:
            self.encoder.train()

    @nvtx.annotate("PointPatchV8FeatNet.forward")
    def forward(self, x: th.Tensor, ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None) -> th.Tensor:
        cfg = self.cfg
        _aux = {}

        # Merge the time and batch dimension
        shape = x.shape  # T B Q D
        x = x.reshape(-1, *shape[-2:])
        # Frozen encoder
        with th.no_grad():
            # x.unstack(x.shape[0]//131070)
            # T B Q D
            # (T*B*Q/32) 32 D
            # Q/32= 1024/32 = 32
            xs = th.split(x, BN_MAX_NUM // self.num_group)
            emb = th.empty(x.shape[0], self.num_group,
                           self.p2v.model.encoder_dim, device=x.device)
            i0 = i1 = 0
            for xx in xs:
                num_samples = xx.shape[0]
                i1 = i0 + num_samples
                embeddings, centers = self.encoder.tokenizer(xx)
                pos = self.encoder.positional_encoding(centers)
                # Get patch embeddings
                emb[i0:i1] = self.encoder.student(
                    embeddings, pos).last_hidden_state
                i0 = i1

        # Retrieve time and batch dimension
        # (B P) H => B P H
        emb = emb.reshape(*shape[:-2], -1, emb.shape[-1])

        query_ctx = th.cat([ctx[k] for k in cfg.query_keys], -1)
        query = self.to_q(query_ctx)
        k, v = self.to_kv(emb)
        out = attention(query, k, v, aux=_aux,
                        key_padding_mask=_aux.get('key_padding_mask', None))
        self._patch_attn = _aux['attn']

        if aux is not None:
            _aux['cross'] = _aux.get('attn', None)
            aux.update(_aux)
        out = out.reshape(*shape[:-2], -1)

        if cfg.cat_query:
            return th.cat([out, query.reshape(*shape[:-2], -1)], -1)
        else:
            return out


def test_point_patch_v8():
    # from torchview import draw_graph
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

    cfg = PointPatchV8FeatNet.Config(
        dim_in=(512, 3),
        dim_out=256,  # What value should I set?
        cfg_path='/tmp/p2v/shapenet.yaml',
        ckpt_path='/tmp/p2v/pre_point2vec-epoch.799-step.64800.ckpt',
        ctx_dim=50,
        emb_dim=128,
        num_query=4,
        query_keys=['c', 'd']
        # add_student=True
    )
    model = PointPatchV8FeatNet(cfg).to('cuda')
    ic(model)
    y = model(x, ctx)
    print(y.shape)

    # model_graph = draw_graph(model,
    #                          input_data=(x, ctx),
    #                          # input_size=x.shape,
    #                          device='cuda')
    # model_graph.visual_graph.render('/tmp/docker/ppv8-student.svg')


def main():
    test_point_patch_v8()


if __name__ == '__main__':
    main()
