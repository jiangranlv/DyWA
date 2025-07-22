#!/usr/bin/env python3


from typing import (Tuple, Union)
from dataclasses import dataclass
from util.config import ConfigBase

import torch as th
import torch.nn as nn
import einops


# FIXME: AddCoord import only to avoid
# breaking imports :P
from models.common import merge_shapes, AddCoord


S = Union[int, Tuple[int, ...]]
T = th.Tensor


class MixTokens(nn.Module):

    @dataclass
    class Config(ConfigBase):
        dim_head: int = 64
        num_head: int = 8
        dim_hidden: int = 128
        p_drop: float = 0.0
        num_layers: int = 1

    def __init__(self, cfg: Config):
        super().__init__()
        self.mix = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.dim_head,
                cfg.num_head,
                cfg.dim_hidden,
                cfg.p_drop,
                batch_first=True),
            cfg.num_layers)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: Tensor(..., C, H, W)
        Returns:
            Tensor(..., C)
        """
        s = x.shape
        # Spatial Mix
        x = einops.rearrange(x,
                             '... c h w -> (...) (h w) c')
        x = self.mix(x)

        # "Average pooling"
        x = x.mean(dim=-2)

        return x.reshape(*s[:-3], -1)


class ShallowMAEBlock(nn.Module):
    """
    Randomly mask the input tokens
    and apply Transformer (N layers of self-attention)
    to recover the masked regions.
    """

    @dataclass
    class Config(ConfigBase):
        p_mask: float = 0.8
        block: int = 1

        dim_input: int = 64
        num_head: int = 8
        dim_hidden: int = 128
        num_layer: int = 2

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.rec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.dim_input,
                cfg.num_head,
                cfg.dim_hidden,
                0.0,
                batch_first=True),
            cfg.num_layer)

    def forward(self, x: th.Tensor):
        """
        Args:
            x: (..., C, H, W): feature map
        """
        cfg = self.cfg
        s = x.shape
        b = cfg.block

        mask_base_shape = merge_shapes(x.shape[:-3], 1,
                                       x.shape[-2] // b, x.shape[-1] // b)

        # Generate masks expanded as spatial blocks.
        mask_base = th.rand(mask_base_shape, device=x.device) < cfg.p_mask
        mask = einops.repeat(mask_base,
                             '... c h w -> ... c (h bh) (w bw)',
                             bh=b, bw=b)
        # apply zero-out mask.
        x = x * mask

        # move patches to time dimension and
        # run through self-attention layers.
        x = einops.rearrange(x, '... c h w -> ... (h w) c')
        x = self.rec(x)
        # restore shape
        x = einops.rearrange(x,
                             '... (h w) c -> ... c h w',
                             h=s[-2],
                             w=s[-1])
        return x
