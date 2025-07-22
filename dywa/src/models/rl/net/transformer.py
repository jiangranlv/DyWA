#!/usr/bin/env python3

from abc import ABC, abstractproperty

from typing import (Optional, Dict, Tuple, Union, Iterable, Any)
from collections import Mapping
from dataclasses import dataclass
from util.config import ConfigBase

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

from torchvision.models import efficientnet_b0

from models.common import (merge_shapes, MLP)
from models.rl.v2.policy import get_activation_function
from models.rl.v3.ln_gru import LayerNormGRUCell
from models.rl.v3.gtrxl import StableTransformerXL

from models.rl.v3.vision.depth_image_encoder import (
    DepthImageEncoder,
    Config as DIEConfig)

from icecream import ic

from models.rl.net.base import (
    FeatureBase, AggregatorBase)

S = Union[int, Tuple[int, ...]]
T = th.Tensor


class TransformerAggNet(nn.Module, AggregatorBase):
    """
    Temporal Aggregation with transformers.
    """
    @dataclass
    class Config(ConfigBase):
        dim_obs: int = -1
        dim_act: int = -1
        dim_out: int = -1
        # FIXME:
        # The current TPPO memory initialization
        # _ONLY_ works with n_layer==1.
        n_layer: int = 1
        n_head: int = 4
        d_head: int = 64
        d_ff: int = 128
        p_drop: float = 0.0
        same_length: bool = True
        mem_len: int = 8

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # NOTE:
        # if you don't like this, then
        # just pass it through nn.Linear()!
        if cfg.dim_out != cfg.dim_obs + cfg.dim_act:
            dim = cfg.dim_obs + cfg.dim_act
            self.fc_in = nn.Linear(dim, cfg.dim_out)

        self.model = StableTransformerXL(
            cfg.dim_out, cfg.n_layer, cfg.n_head, cfg.d_head, cfg.d_ff,
            dropout=cfg.p_drop,
            same_length=cfg.same_length,
            mem_len=cfg.mem_len)

    def process_inputs(self, act: th.Tensor, obs: th.Tensor):
        cfg = self.cfg
        ao = th.cat([act, obs], dim=-1)
        if cfg.dim_out != cfg.dim_obs + cfg.dim_act:
            f = self.fc_in(ao)
        else:
            f = ao
        return f

    def forward(self,
                act: th.Tensor,
                obs: th.Tensor,
                memory):
        cfg = self.cfg
        f = self.process_inputs(act, obs)
        return self.model(f, memory)

    def init_memory(self,
                    batch_shape: int,
                    *args, **kwds):
        cfg = self.cfg
        S = merge_shapes(cfg.mem_len, batch_shape, cfg.dim_out)
        memory = [th.zeros(S, *args, **kwds)
                  for _ in range(cfg.n_layer + 1)]
        return memory
