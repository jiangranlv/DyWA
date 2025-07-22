#!/usr/bin/env python3

from typing import Iterable, Optional, Tuple, Union, Dict, List
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import nvtx


from flash_attn.flash_attention import FlashMHA, FlashAttention
from transformers.activations import ACT2FN
from models.sdf.encoder.point_tokens import SpatialSort, HilbertCode
from models.common import (
    PosEncodingMLP,
    PosEncodingNeRF,
    PosEncodingSine,
    PosEncodingLinear,
    MLP,
    get_activation_function
)

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from models.common import merge_shapes
from util.config import recursive_replace_map, ConfigBase
from util.torch_util import dcn

from icecream import ic
from einops import rearrange
# from torchsummary import summary

def _knn_gather(x: th.Tensor, i: th.Tensor):
    """
    x: [..., P, D] point cloud
    i: [..., P, K] K indices per each point, for each index along P
    """

    # [bpd], [bpk]
    # [b1pd], [bpk1]
    return th.take_along_dim(x[..., None, :, :],
                             i[..., None], dim=-2)


def _get_overlap_patch_params(
        num_points: int,
        patch_size: int,
        ratio_bound: Tuple[float, float] = (1.25, 1.75)
):
    num_points: int = 512
    patch_size: int = 32
    num_patch: int = num_points // patch_size
    # target_ratio: float = 1.5
    # ratio_bound: float = [1.25, 1.75]
    # target_ratio: float = 1.5
    num_patch = num_points // patch_size
    true_patch_size = np.arange(
        np.ceil(patch_size * ratio_bound[0]),
        np.floor(patch_size * ratio_bound[1]))
    stride = np.round((num_points - true_patch_size) / (num_patch - 1))
    true_num_point = (num_patch - 1) * stride + true_patch_size
    index = np.argwhere(num_points == true_num_point).ravel()
    if len(index) > 0:
        index = index[0]
    sol_tps = true_patch_size[index]
    # FIXME: hardcoded `3` as the cloud dimension
    sol_str = 3 * stride[index]
    return (int(sol_tps), int(sol_str))


def normalize(x: th.Tensor,
              in_place: bool = False,
              iso: bool = True,
              diag: bool = True,
              aux: Optional[Dict[str, th.Tensor]] = None,
              center: Optional[th.Tensor] = None,
              radius: Optional[th.Tensor] = None,
              scale: Optional[Tuple[float, float]] = None
              ):
    if (center is None) or (radius is None):
        bmin = x.min(dim=-2, keepdim=True).values
        bmax = x.max(dim=-2, keepdim=True).values

    if center is None:
        center = 0.5 * (bmin + bmax)

    if radius is None:
        if diag:
            radius = (
                th.linalg.norm(x - center, dim=-1, keepdim=True)
                .max(dim=-2, keepdim=True).values
            )
        else:
            radius = 0.5 * (bmax - bmin)
            # Isotropic scaling
            if iso:
                radius[...] = radius.max(dim=-1, keepdim=True).values

    if scale is not None:
        radius *= th.empty_like(radius).uniform_(scale[0], scale[1])

    if aux is not None:
        aux['center'] = center
        aux['radius'] = radius

    if in_place:
        return x.sub_(center).div_(radius)
    else:
        return x.sub(center).div_(radius)


def subsample(x: th.Tensor, n: int,
              y: Optional[th.Tensor] = None,
              aux: Optional[Dict[str, th.Tensor]] = None):
    index = th.randint(x.shape[-2], size=(*x.shape[:-2], n),
                       dtype=th.long, device=x.device)
    if aux is not None:
        aux['index'] = index
    return th.take_along_dim(x, index[..., None], -2, out=y)

class No_Linear_Proj_FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        
        !!!!!!use residual!!!!!
        
        """
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return rearrange(context, 'b s h d -> b s (h d)'), attn_weights


class PointMAESelfAttention(nn.Module):
    '''
    对Flash attention 库的封装
    FlashMHA是一个高效的transformer 库

    '''
    @dataclass
    class Config(ConfigBase):
        hidden_size: int = 128
        num_attention_heads: int = 4
        qkv_bias: bool = True
        attention_probs_dropout_prob: float = 0.0
        linear_proj: bool=True ## attention后过一层mlp

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0 and not hasattr(
                cfg, "embedding_size"):
            raise ValueError(
                f"The hidden size {cfg.hidden_size,} is not a multiple of the number of attention "
                f"heads {cfg.num_attention_heads}.")

        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = int(
            cfg.hidden_size / cfg.num_attention_heads)
        if cfg.linear_proj:
            self.attention = FlashMHA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                bias=cfg.qkv_bias,
                attention_dropout=cfg.attention_probs_dropout_prob)
        else:
            self.attention = No_Linear_Proj_FlashMHA(
                cfg.hidden_size,
                cfg.num_attention_heads,
                bias=cfg.qkv_bias,
                attention_dropout=cfg.attention_probs_dropout_prob)
        return

    def forward_slow(self, hidden_states,
                     head_mask: Optional[th.Tensor] = None,
                     key_padding_mask: Optional[th.Tensor] = None,
                     output_attentions: bool = False) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:
        """
        key_padding_mask:
            0 = keep
            1 = drop
        """
        # == from FlashMHA ==
        # qkv = self.Wqkv(x)
        # qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        # context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
        #                                         need_weights=need_weights, causal=self.causal)
        # return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')),
        # attn_weights

        # self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        # self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        qkv = self.attention.Wqkv(hidden_states)
        qkv = einops.rearrange(qkv,
                               '... s (three h d) -> ... three h s d',
                               three=3, h=self.num_attention_heads)
        query_layer, key_layer, value_layer = th.unbind(qkv, dim=-4)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = th.matmul(
            query_layer, key_layer.transpose(-1, -2))

        # ... h s s // [head, output, input]
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if key_padding_mask is not None:
            # ic(key_padding_mask)
            attention_scores.masked_fill_(key_padding_mask[..., None, :],
                                          float('-inf'))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = th.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[
        #     :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*context_layer.shape[:-2], -1)

        context_layer = self.attention.out_proj(context_layer)

        outputs = (
            context_layer,
            attention_probs) if output_attentions else (
            context_layer,
        )

        return outputs

    def forward(
        self,
        hidden_states,
        head_mask: Optional[th.Tensor] = None,
        key_padding_mask: Optional[th.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:
        """
        key_padding_mask interface:
            0 = keep
            1 = drop
        """
        assert (head_mask is None)

        if output_attentions:
            # On the other hand, forward_slow implements
            # same logic as common.attention(), which means
            # 0=keep, 1=drop. So we invert again
            # if we use forward_slow().
            x_slow, attn = self.forward_slow(
                hidden_states, head_mask,
                key_padding_mask=key_padding_mask,
                output_attentions=output_attentions)

            # == VALIDATION ==
            if False:
                with th.no_grad():
                    with th.cuda.amp.autocast(True, th.float16):
                        if key_padding_mask is not None:
                            key_padding_mask = ~key_padding_mask
                        x, _ = self.attention(
                            hidden_states,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)
                    ic(hidden_states.shape)  # 1,16,96
                    dx = (x - x_slow)[..., :4, :]
                    ic(dx.min(), dx.max(), dx.std(), dx.mean())
            return (x_slow, attn)

        with th.cuda.amp.autocast(True, th.float16):
            # According to FlashMHA, 1=keep, 0=drop.
            # Thus we invert key_padding_mask
            # to keep ourselves consistent
            # to FlashMHA convention.
            if key_padding_mask is not None:
                key_padding_mask = ~key_padding_mask
            x, attention_probs = self.attention(
                hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions)
        x = x.to(dtype=hidden_states.dtype)
        outputs = (x, attention_probs) if output_attentions else (x,)
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with
# ViT->ViTMAE
class PointMAESelfOutput(nn.Module):
    """
    The residual connection is defined in PointMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    @dataclass
    class Config(ConfigBase):
        hidden_size: int = 128
        hidden_dropout_prob: float = 0.0

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states: th.Tensor,
                input_tensor: th.Tensor) -> th.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with
# ViT->ViTMAE
class PointMAEAttention(nn.Module):
    '''Multihead attention
    目前的实现是Multihead attention 后加了两个linear 层(并不是feed-forward layer), 不知道有什么用处.
    '''

    @dataclass
    class Config(ConfigBase):
        linear_proj:bool=True ## attention 后是否使用linear 层
        self_attn: PointMAESelfAttention.Config = PointMAESelfAttention.Config()
        output: PointMAESelfOutput.Config = PointMAESelfOutput.Config()
        

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        cfg.self_attn.linear_proj = cfg.linear_proj ## 实例化的时候才能确定cfg参数, 进行传参
        self.attention = PointMAESelfAttention(cfg.self_attn)

        if cfg.linear_proj:
            self.output = PointMAESelfOutput(cfg.output)
        else:
            pass

    def forward(
        self,
        hidden_states: th.Tensor,
        head_mask: Optional[th.Tensor] = None,
        key_padding_mask: Optional[th.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:
        self_outputs = self.attention(
            hidden_states, head_mask, key_padding_mask,
            output_attentions)

        if self.cfg.linear_proj:
            attention_output = self.output(self_outputs[0], hidden_states) ### 只看第一个tensor, self_outputs[0]
        else:
            attention_output = self_outputs[0]

        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->ViTMAE
class PointMAEIntermediate(nn.Module):
    '''
    feed-forward layer 第一个MLP + activation
    '''
    @dataclass
    class Config(ConfigBase):
        hidden_size: int = 128
        intermediate_size: int = 128
        hidden_act: str = 'gelu'

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        if isinstance(cfg.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[cfg.hidden_act]
        else:
            self.intermediate_act_fn = cfg.hidden_act

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->ViTMAE
class PointMAEOutput(nn.Module):
    '''
    这个封装的是真的蛋疼
    feed-forward layer 第二个MLP + residual link
    '''
    @dataclass
    class Config(ConfigBase):
        intermediate_size: int = 128
        hidden_size: int = 128
        hidden_dropout_prob: float = 0.0

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, hidden_states: th.Tensor,
                input_tensor: th.Tensor) -> th.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTMAE
class PointMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    @dataclass
    class Config(ConfigBase):
        attention: PointMAEAttention.Config = PointMAEAttention.Config()
        intermediate: PointMAEIntermediate.Config = PointMAEIntermediate.Config()
        output: PointMAEOutput.Config = PointMAEOutput.Config()
        hidden_size: int = 128
        layer_norm_eps: float = 1e-6
        use_adapter: bool = False
        adapter_dim: int = 64

        def __post_init__(self):
            self.attention = recursive_replace_map(self.attention, {
                'self_attn.hidden_size': self.hidden_size,
                'output.hidden_size': self.hidden_size,
            })
            self.intermediate = recursive_replace_map(self.intermediate, {
                'hidden_size': self.hidden_size})
            self.output = recursive_replace_map(self.output, {
                'hidden_size': self.hidden_size})

    def __init__(self, cfg: Config, use_adapter: bool) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = PointMAEAttention(cfg.attention)
        self.intermediate = PointMAEIntermediate(cfg.intermediate)
        self.output = PointMAEOutput(cfg.output)
        self.layernorm_before = nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.use_adapter = use_adapter
        if use_adapter:
            self.use_adapter = True
            self.adapter = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.adapter_dim),
                nn.ELU(),
                nn.Linear(cfg.adapter_dim, cfg.hidden_size)
            )
            # FIXME: HARDCORDED std
            for k, v in self.adapter.named_parameters():
                nn.init.normal_(v, std=1e-3)

    def forward(
        self,
        hidden_states: th.Tensor,
        head_mask: Optional[th.Tensor] = None,
        key_padding_mask: Optional[th.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:
        self_attention_outputs = self.attention(
            # in ViTMAE, layernorm is applied before self-attention
            self.layernorm_before(hidden_states),
            head_mask,
            key_padding_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        # adapter
        if self.use_adapter:
            adapter_feature = self.adapter(layer_output)
            layer_output = layer_output + adapter_feature

        outputs = (layer_output,) + outputs
        return outputs


class PointMAEEncoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        layer: PointMAELayer.Config = PointMAELayer.Config()
        num_hidden_layers: int = 4
        use_adapter: bool = False

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.config = cfg
        # FIXME: HARDCORDED selection of adapter
        self.layer = nn.ModuleList([PointMAELayer(cfg.layer,
                                                  (l < cfg.num_hidden_layers - 1) & cfg.use_adapter)
                                    for l in range(cfg.num_hidden_layers)])

    def forward(
            self,
            hidden_states: th.Tensor,
            head_mask: Optional[th.Tensor] = None,
            key_padding_mask: Optional[th.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False) -> Tuple[th.Tensor, ...]:
        """
        key_padding_mask:
            0 = keep
            1 = drop
        """

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states, layer_head_mask, key_padding_mask,
                output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return (hidden_states, all_hidden_states, all_self_attentions)


class PointMAEDecoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        layer: PointMAELayer.Config = PointMAELayer.Config()
        num_hidden_layers: int = 4
        hidden_size: int = 128
        decoder_hidden_size: int = 128
        layer_norm_eps: float = 1e-6
        patch_size: int = 32
        num_channels: int = 3
        initializer_range: float = 0.02
        add_pos_embed: bool = True
        use_pred: bool = True

        pred_embed: bool = False
        embed_size: int = 384

        def __post_init__(self):
            self.layer = recursive_replace_map(self.layer, {
                'hidden_size': self.hidden_size
            })

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.decoder_embed = nn.Linear(
            cfg.hidden_size, cfg.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(th.zeros(1, 1, cfg.decoder_hidden_size))
        self.decoder_layers = nn.ModuleList(
            [PointMAELayer(cfg.layer, False)
             for _ in range(cfg.num_hidden_layers)])

        self.decoder_norm = nn.LayerNorm(
            cfg.decoder_hidden_size, eps=cfg.layer_norm_eps)

        if cfg.use_pred:
            self.decoder_pred = nn.Linear(
                cfg.decoder_hidden_size,
                cfg.patch_size * cfg.num_channels,
                bias=True)  # encoder to decoder

        if cfg.pred_embed:
            self.decoder_pred = nn.Linear(
                cfg.decoder_hidden_size,
                cfg.embed_size,
                bias=True)  # encoder to decoder

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as
        # cutoff is too big (2.)
        th.nn.init.normal_(self.mask_token, std=self.cfg.initializer_range)

        # try zero-init ??
        # th.nn.init.normal_(self.decoder_pred.weight, std=0.01)
        # th.nn.init.zeros_(self.decoder_pred.bias)

    def unshuffle(self, x, ids_restore, pos_embed):
        cfg = self.cfg
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = th.cat([x, mask_tokens], dim=1)

        # unshuffle
        # x = th.gather(x_, dim=1,
        # index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = th.take_along_dim(x_, ids_restore.unsqueeze(-1),
                              dim=1)

        # Add pos embed?
        if cfg.add_pos_embed:
            hidden_states = x + pos_embed
        else:
            hidden_states = x
        return hidden_states

    def forward(
        self,
        hidden_states,
        ids_restore=None,
        pos_embed=None,

        output_attentions=False,
        output_hidden_states=False,
    ):
        cfg = self.cfg
        if ((ids_restore is not None) and
                ((not cfg.add_pos_embed) or (pos_embed is not None))):
            hidden_states = self.unshuffle(hidden_states,
                                           ids_restore,
                                           pos_embed)

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, head_mask=None,
                output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # NOTE: not sure if this matters
        # but it feels more correct to output the
        # hidden state after LN...
        hidden_states = self.decoder_norm(hidden_states)
        # print('hs', hidden_states.shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if cfg.use_pred:
            # predictor projection
            logits = self.decoder_pred(hidden_states)
            # reshape to full cloud
            logits = logits.reshape(*logits.shape[:-2], -1, cfg.num_channels)
        elif cfg.pred_embed:
            logits = self.decoder_pred(hidden_states)
        else:
            logits = None
        return (logits, all_hidden_states, all_self_attentions)


class KNNPatchEncoder(nn.Module):
    """
    DGCNN-style patch encoder that builds
    neighborhood graph (once) and applies
    edge convolution
    """

    def __init__(self,
                 patch_size: int,
                 encoder_channel: int,
                 d: int = 3,
                 k: int = 8,
                 f: int = 128):
        super().__init__()
        self.d = d
        self.k = k
        self.ec1 = MLP((d * 2, f // 2), act_cls='elu',
                       use_bn=False,
                       use_ln=True)
        self.ec2 = MLP((f // 2 * 2, f), act_cls='elu',
                       use_bn=False,
                       use_ln=True)
        self.out_project = nn.Linear(f,
                                     encoder_channel)

    def extra_repr(self):
        return F'd={self.d}, k={self.k},'

    def _edge_conv(self,
                   x: th.Tensor,
                   i: th.Tensor,
                   L: th.Tensor,
                   m: nn.Module):
        # "source" nodes
        src = einops.repeat(x, '... n d -> ... n k d', k=self.k)
        # "target" nodes (neighbors)
        dst = _knn_gather(x, i)  # , L)
        edge = th.cat([src, dst], dim=-1)  # d -> d*2
        feat = m(edge).max(dim=-2).values
        return feat

    def forward(self, x: th.Tensor):
        s = x.shape
        # n = patch size
        # k = num neighbor
        # f = feature dim

        # Move num patch to batch dimension
        x = einops.rearrange(x,
                             '... g n d -> (... g) n d')

        # THIS IS ONLY POSSIBLE
        # BECAUSE I KNOW I"M ONLY
        # GOING TO USE THIS KNN OP
        # ON X without the gradients I think?
        L = th.full((x.shape[0],), x.shape[1],
                    dtype=th.int64, device=x.device)
        with th.no_grad():
            _, nn_idx, _ = knn_points(x, x, L, L, K=self.k,
                                      return_nn=False,
                                      return_sorted=False)

        # Apply two edge covolution iterations
        f = self._edge_conv(x, nn_idx, L, self.ec1)
        f = self._edge_conv(f, nn_idx, L, self.ec2)
        f = f.mean(dim=-2)  # aggregate across points
        f = self.out_project(f)  # project to output dimension
        f = f.reshape(*s[:-2], -1)
        return f


class MiniPNPatchEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        # patch size = (32, 3) = 96
        # it would be stupid to exceed 96 by a large margin
        self.mlp1 = MLP((3, 16), use_bn=False, use_ln=True,
                        activate_output=True)
        self.mlp2 = MLP((32, out_dim), use_bn=False, use_ln=True,
                        activate_output=True)

    def forward(self, x: th.Tensor):
        x = self.mlp1(x)
        x_l = x
        x_g = th.max(x, dim=-2, keepdim=True).values.expand_as(x_l)
        x = th.cat([x_l, x_g], dim=-1)
        x = self.mlp2(x).max(dim=-2).values
        return x


class MLPPatchEncoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        hidden: Tuple[int, ...] = (256, 256)
        sort: bool = False
        pre_ln_bias: bool = False

    def __init__(self, cfg: Config,
                 patch_size: int,
                 encoder_channel: int):
        super().__init__()
        self.cfg = cfg
        dims = merge_shapes(patch_size * 3,
                            cfg.hidden, encoder_channel)
        self.mlp = MLP(
            dims,
            get_activation_function('gelu'),
            activate_output=False,
            use_bn=False,
            bias=True,
            use_ln=True,
            pre_ln_bias=cfg.pre_ln_bias
        )
        if cfg.sort:
            self.sort = SpatialSort(th.jit.script(HilbertCode()))

    def forward(self, x: th.Tensor, target_pc=False):
        with th.no_grad():
            if self.cfg.sort:
                x = self.sort(x)
        x = einops.rearrange(x, '... g n three -> ... g (n three)', three=3)
        out = self.mlp(x)
        return out

class GroupAndMLPPatchEncoder(nn.Module):
    def __init__(self, cfg:MLPPatchEncoder.Config, patch_size, encoder_channel):
        super().__init__()

        self.group = get_group_module_v2('fps', patch_size)
        self.mlp = MLPPatchEncoder(cfg, patch_size, encoder_channel)
        
    def forward(self, x: th.Tensor, target_pc=False):
        p, c = self.group(x)
        z = self.mlp(p - c[..., None, :])
        return c, z

class ConvPatchEncoder(nn.Module):  # Embedding module
    '''
    两层local mlp, 两层global mlp
    '''
    def __init__(self, encoder_channel: int, latent_channel: int = 64, in_channel:int=3, norm='bn'):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.in_channel=in_channel
        self.norm=norm

        norm_type=None
        if norm == 'bn':
            norm_type = nn.BatchNorm1d
        elif norm == 'ln':
            norm_type = nn.LayerNorm
        else:
            raise NotImplementedError

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, latent_channel //2, 1),  # 3, 32
            norm_type(latent_channel //2),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_channel //2, latent_channel, 1) # (32, 64)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(latent_channel * 2, self.encoder_channel, 1), # (64, 128)
            norm_type(self.encoder_channel),
            nn.ReLU(inplace=True), 
            nn.Conv1d(self.encoder_channel, self.encoder_channel, 1) # (128, 128)
        )


    def forward(self, point_groups, target_pc=False):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
s        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.in_channel)

        # encoder
        # feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        x = point_groups.transpose(2, 1)
        x = self.first_conv[0](x)
        if self.norm == 'bn':
            x = self.first_conv[1](x)
        else:
            x = self.first_conv[1](x.transpose(2, 1)).transpose(2, 1)
        x = self.first_conv[2:](x)


        feature_global = th.max(x, dim=2, keepdim=True)[0]  # BG 256 1
        feature = th.cat(
            [feature_global.expand(-1, -1, n),
             x],
            dim=1)  # BG 512 n

        ###global conv
        # feature = self.second_conv(feature)  # BG 1024 n
        glob_x = self.second_conv[0](feature)
        if self.norm == 'bn':
            glob_x = self.second_conv[1](glob_x)
        else:
            glob_x  = self.second_conv[1](glob_x .transpose(2, 1)).transpose(2, 1)
        glob_x  = self.second_conv[2:](glob_x )

        feature_global = th.max(glob_x, dim=2, keepdim=False)[0]  # BG 1024

        return feature_global.reshape(bs, g, self.encoder_channel)


class DoubleConvPatchEncoder(nn.Module):  # Embedding module, 
    '''
    seperate init pc and target pc, use two same model but not share params
    '''
    def __init__(self, encoder_channel: int, latent_channel: int = 64):
        super().__init__()
        
        self.encoder_targetPC = ConvPatchEncoder(encoder_channel, latent_channel)
        self.encoder_curPC = ConvPatchEncoder(encoder_channel, latent_channel)

    def forward(self, point_groups, target_pc=False):
        
        if target_pc:
            return self.encoder_targetPC(point_groups)
        else:
            return self.encoder_curPC(point_groups)

class My_GroupFPS(nn.Module):
    def __init__(self,
                 patch_num: int,
                 K:int,
                 random_start_point:Optional[bool] = False
                 ):
        super().__init__()
        self.patch_num = patch_num
        self.K = K
        self.random_start_point=random_start_point

    def forward(self, x: th.Tensor,
                center: Optional[th.Tensor] = None,
                sub: bool = True,
                sort: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:

        s = x.shape
        x = x.reshape(-1, *x.shape[-2:])

        if center is None:
            c, _ = sample_farthest_points(
                x, K=self.patch_num, random_start_point=self.random_start_point)
        else:
            c = center
        _, nn_idx, p = knn_points(c, x, K=self.K,
                                  return_nn=True,
                                  return_sorted=sort
                                  )

        c = c.reshape(*s[:-2], *c.shape[1:])
        p = p.reshape(*s[:-2], *p.shape[1:])
        if aux is not None:
            # nn_idx = nn_idx.reshape(*s[:-2], nn_idx.shape[1:])
            aux['fps_nn_idx'] = nn_idx
        if sub:
            p -= c[..., None, :]
        return (p, c)


class FPS_PATCH_BASE(nn.Module):
    def __init__(self, nsample, npoints, FPS_random_start_point=False):
        super(FPS_PATCH_BASE, self).__init__()
        self.nsample = nsample

        self.group = My_GroupFPS(nsample, npoints, random_start_point=FPS_random_start_point)

    def get_sampled_points_and_xyz(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points: sample points feature data, [B, S, P, C+D]
        """
        aux = {}
        new_xyz, centers = self.group(xyz, aux=aux) #B * nsample * npoint * 3, B * nsample * 3, xyz has demean center
        new_points = self.index_points(points, aux["fps_nn_idx"]) # [B, nsample, npoint, D]
        new_points = th.cat([new_xyz, new_points], dim=-1) # [B, nsample, npoint, C+D]
        new_xyz = centers # B * nsample * 3

        return new_xyz, new_points

    @staticmethod
    def index_points(points:th.Tensor, idx:th.Tensor):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S, K]
        Return:
            new_points:, indexed points data, [B, S, K, C]
        """

        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = th.arange(B, dtype=th.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

class PointNet2SetAbstraction(FPS_PATCH_BASE):

    def __init__(self, nsample:int, npoints:int, FPS_random_start_point:bool, 
                 in_channel, mlp:Tuple[int, ...]=(),
                 concat_xyz:bool=True,
                 norm='bn'
                 ):
        super().__init__(nsample, npoints, FPS_random_start_point)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.concat_xyz = concat_xyz

        norm_type=None
        if norm == 'bn':
            norm_type = nn.BatchNorm2d
        elif norm == 'ln':
            raise NotImplementedError
            norm_type = nn.LayerNorm
        else:
            raise NotImplementedError

        last_channel = in_channel + 3 if self.concat_xyz and in_channel != 3 else in_channel 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(norm_type(out_channel))
            last_channel = out_channel

        
    def forward(self, xyz:th.Tensor, points:th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concat: sample points feature data, [B, S, D']
        """

        new_xyz, new_points = self.get_sampled_points_and_xyz(xyz, points) # B * S * K * (3 + D)

        ### 去掉points中的原始xyz
        if self.concat_xyz and points.shape[-1] != 3:
            pass
        else:
            new_points = new_points[..., 3:]

        ### mlps 
        new_points = new_points.permute(0, 3, 1, 2) # B * D * S * K, 对应COnv2d B * C * H * W
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points))) # B * D' * S * K
        
        ### max pooling
        new_points = th.max(new_points, 3)[0] # B * D' * S
        
        return new_xyz, new_points.permute(0, 2, 1)

class ConvPatchSetAbstraction(FPS_PATCH_BASE):
    def __init__(self, 
                 nsample:int, 
                 npoints:int, 
                 FPS_random_start_point:bool=False, 
                 encoder_channel:int=128, 
                 latent_channel:int=64, 
                 in_channel:int=3,
                 concat_xyz:bool=True,
                 norm='bn'):
        
        super().__init__(nsample, npoints, FPS_random_start_point)

        self.patch_encoder = ConvPatchEncoder(
            encoder_channel = encoder_channel, 
            latent_channel=latent_channel, 
            in_channel=in_channel + 3 if concat_xyz and in_channel != 3 else in_channel,
            norm=norm
        )

        self.concat_xyz=concat_xyz
        
    def forward(self, xyz:th.Tensor, points:th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concat: sample points feature data, [B, S, D']
        """

        new_xyz, new_points = self.get_sampled_points_and_xyz(xyz, points) ## B * S * K * (C+D)
        if self.concat_xyz and points.shape[-1] != 3:
            new_points = self.patch_encoder(new_points)
        else: ## points.shape[-1] == 3, 说明输入只有xyz, 没必要concat两遍
            new_points = self.patch_encoder(new_points[..., 3:])

        return new_xyz, new_points

class MultiLayer_BASE(nn.Module):
    
    def __init__(self, 
                 mlps:List[List[int]]=[[32, 128]], 
                 patch_nums:List[int] = [16],
                 K:int=32, 
                 FPS_random_start_point=False,
                 norm='bn'):
        
        super().__init__()

        assert len(mlps) == len(patch_nums)

        self.model = nn.ModuleList()

    def forward(self, x:th.Tensor)-> Tuple[th.Tensor, th.Tensor]:

        '''
        x.shape: B * N * 3 
        return : latent vector: B * S * D, S = patch_num[-1]
        '''

        xyz = x; points = x
        for model in self.model:
            xyz, points = model(xyz, points)

        return xyz, points


class MultiLayer_Pointnet2PatchEncoder(MultiLayer_BASE):
    '''
    pointNet ++ implementation
    mlps外层为pointnet layer, 内层为每个pointnet layer的mlp参数
    '''

    def __init__(self, 
                 mlps:List[List[int]]=[[32, 128]], 
                 patch_nums:List[int] = [16],
                 K:int=32, 
                 FPS_random_start_point=False,
                 norm='bn'):
        
        super().__init__(mlps, patch_nums, K, FPS_random_start_point, norm=norm)

        last_channel=3
        for mlp, ps in zip(mlps, patch_nums):
            self.model.append(
                PointNet2SetAbstraction(nsample=ps,
                                        npoints=K, 
                                        FPS_random_start_point=FPS_random_start_point,
                                        mlp=mlp,
                                        in_channel=last_channel,
                                        concat_xyz=True,
                                        norm=norm)
                )
            last_channel=mlp[-1]


class MultiLayer_ConvPatchEncoder(MultiLayer_BASE):
    '''
    group + pointnet,
    '''
    
    def __init__(self, 
                 mlps:List[List[int]]=[[128]], 
                 ### in this model only use first item in each sub-tuple, because the channel was fix in ConvPatchEncoder
                 patch_nums:List[int] = [16], 
                 K:int=32, 
                 FPS_random_start_point=False,
                 norm='bn'):
        
        super().__init__(mlps, patch_nums, K, FPS_random_start_point, norm=norm)

        last_channel=3
        for mlp, ps in zip(mlps, patch_nums):
            self.model.append(
                ConvPatchSetAbstraction(nsample=ps,
                                        npoints=K, 
                                        FPS_random_start_point=FPS_random_start_point,
                                        encoder_channel=mlp[0], 
                                        latent_channel=mlp[0] // 2, 
                                        in_channel=last_channel,
                                        concat_xyz=True,
                                        norm=norm)
            )
            last_channel=mlp[0]


class My_PointNetEncoder(nn.Module):
    """
    Simplified Feature Extractor without TNet and Feature Transform.
    """
    def __init__(self, global_feat=True, encode_dim=1024):
        super().__init__()
        self.global_feat = global_feat
        self.encode_dim=encode_dim

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, encode_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(encode_dim)

    def forward(self, x):
        n_pts = x.size(2)

        x = F.relu(self.bn1(self.conv1(x.transpose(2, 1))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global feature
        x = th.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.encode_dim)

        if self.global_feat:
            return x  # Return global feature only
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return th.cat([x], 1)  # Return concatenated feature

class PointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, c_dim=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers

        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.act = act_func

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, c_dim, kernel_size=1)

    def forward(self, x, ret_perpoint_feat=False):

        y = self.act(self.conv_in(x.transpose(2, 1)))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = th.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = th.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values

        if ret_perpoint_feat:
            ret = x.transpose(1, 2)
        else:
            ret = x_global

        return ret

def mask(seq: th.Tensor, mask_ratio: float = 0.75,
         noise: Optional[th.Tensor] = None,
         aux: Optional[Dict[str,th.Tensor]]=None):
    """
    mask:
        0 = keep
        1 = hide
    """
    batch_size, seq_length, dim = seq.shape
    len_keep = int(seq_length * (1 - mask_ratio))

    if noise is None:
        noise = th.rand(
            batch_size,
            seq_length,
            device=seq.device)  # noise in [0, 1]

    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = th.argsort(noise, dim=1)
    ids_restore = th.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    keep = th.take_along_dim(seq, ids_keep[..., None], dim=1)

    if aux is not None:
        aux['ids_hide'] = ids_shuffle[:, len_keep:]

    # keep = th.gather(
    #    seq, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = th.ones([batch_size, seq_length], device=seq.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = th.gather(mask, dim=1, index=ids_restore)

    return keep, mask, ids_restore


def combine(src: th.Tensor,
            rec: th.Tensor,
            sort_index: th.Tensor,
            hide_mask: th.Tensor,
            patch_size: int):
    # Sort `src` identically as `rec` input.
    src = th.take_along_dim(src, sort_index[..., None], -2)
    # Patchify both clouds, from (..., D, 3) -> (..., S, P, 3)
    rec_p = rec.reshape(*rec.shape[:-2], -1, patch_size, 3)
    src_p = src.reshape(*src.shape[:-2], -1, patch_size, 3)

    # Combine both clouds, according to hide_mask.
    out = th.where(hide_mask[..., None, None], src_p, rec_p)
    return out.reshape(src.shape)


class GroupHilbert(nn.Module):
    def __init__(self,
                 patch_size: int,
                 patch_overlap: float = 1.0,
                 true_patch_size: Optional[int] = None
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        if true_patch_size is None:
            assert (patch_overlap == 1.0)
            true_patch_size = patch_size
        self.true_patch_size = true_patch_size

    def _group(self, x: th.Tensor, sub: bool = True) -> th.Tensor:
        """
        Args:
            x: (..., N, D) point cloud (sorted)
        Returns:
            p: (..., S, P, D) patches
            c: (..., S, D) patch centers
        """
        s = x.shape  # ..., N D
        if self.patch_overlap == 1.0:
            x = x.reshape(*x.shape[:-2], -1, self.patch_size, 3)
            c = x.mean(dim=-2, keepdim=True)
            p = x
            if sub:
                p = p - c
            p = p.reshape(*s[:-2], -1, *p.shape[-2:])
            c = c.reshape(*s[:-2], -1, p.shape[-1])
        else:
            xx = x.reshape(*x.shape[:-2], -1, self.patch_size, 3)
            num_patch: int = x.shape[-2] // self.patch_size
            s1 = (*s[:-2], num_patch, self.true_patch_size, 3)
            s2 = (*s[1:-2], num_patch, self.patch_stride, 3, 1)
            strides = np.cumprod(s2[::-1])[::-1]
            strides = list([int(x) for x in strides])
            strides[:-3] = xx.stride()[:-3]
            strides = tuple(strides)
            x = x.as_strided(s1, strides)
            c = x.mean(dim=-2, keepdim=True)
            p = x
            if sub:
                p = p - c
            c = c.reshape(*s[:-2], -1, x.shape[-1])
        return p, c

    def forward(self, x: th.Tensor, center: Optional[th.Tensor] = None,
                sub: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:
        # x = self.sort(x)
        p, c = self._group(x, sub=sub)
        return (p, c)


class GroupFPS(nn.Module):
    def __init__(self,
                 patch_size: int,
                 true_patch_size: Optional[int] = None,
                 random_start_point:Optional[bool] = False
                 ):
        super().__init__()
        self.patch_size = patch_size
        if true_patch_size is None:
            true_patch_size = patch_size
        self.true_patch_size = true_patch_size
        self.random_start_point=random_start_point

    def forward(self, x: th.Tensor,
                center: Optional[th.Tensor] = None,
                sub: bool = True,
                sort: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:

        s = x.shape
        x = x.reshape(-1, *x.shape[-2:])

        if center is None:
            c, _ = sample_farthest_points(
                x, K=x.shape[-2] // self.patch_size, random_start_point=self.random_start_point)
        else:
            c = center

        _, nn_idx, p = knn_points(c, x, K=self.true_patch_size,
                                  return_nn=True,
                                  return_sorted=sort
                                  )

        c = c.reshape(*s[:-2], *c.shape[1:])
        p = p.reshape(*s[:-2], *p.shape[1:])
        if aux is not None:
            # nn_idx = nn_idx.reshape(*s[:-2], nn_idx.shape[1:])
            aux['fps_nn_idx'] = nn_idx
        if sub:
            p -= c[..., None, :]
        return (p, c)


class GroupHilbertV2(nn.Module):
    def __init__(self, patch_size: int,
                 recenter: bool = False):
        super().__init__()
        self.recenter = recenter
        out = get_group_module('hilbert',
                               patch_size,
                               1.0)
        (self.sort,
         self.group,
         self.true_patch_size,
         self.patch_stride) = out

    def forward(self, x: th.Tensor,
                sort: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None):
        assert (sort)
        _aux = {}
        x = self.sort(x, aux=_aux)
        if aux is not None:
            aux['patch_index'] = _aux.pop('sort_index').reshape(
                *x.shape[:-2], -1,
                self.true_patch_size)
        p, c = self.group(x, aux=aux, sub=False)
        if self.recenter:
            c = p.mean(dim=-2)
        return (p, c)


class GroupFPSV2(nn.Module):
    def __init__(self, patch_size: int,
                 recenter: bool = False,
                 random_start_point:bool=False):
        super().__init__()
        self.recenter = recenter
        out = get_group_module('fps',
                               patch_size,
                               1.0,
                               FPS_random_start_point=random_start_point)
        (_,
         self.group,
         self.true_patch_size,
         self.patch_stride) = out

    def forward(self, x: th.Tensor,
                sort: bool = True,
                aux: Optional[Dict[str, th.Tensor]] = None):
        _aux = {}
        p, c = self.group(x, aux=_aux, sort=sort, sub=False)
        if aux is not None:
            aux['patch_index'] = _aux.pop('fps_nn_idx').reshape(
                *x.shape[:-2], -1,
                self.true_patch_size)
        if self.recenter:
            c = p.mean(dim=-2)
        return (p, c)


def get_pos_enc_module(
        pos_embed_type: str,
        out_channels: int,
        in_channels: int = 3
):
    if pos_embed_type == 'nerf':
        # FIXME: hardcoded pos_embed dimensions
        # Probably will break for most cases ...
        return PosEncodingNeRF(in_channels,
                               num_frequencies=32,
                               cat_input=False)
    elif pos_embed_type == 'sine':
        return PosEncodingSine(in_channels, out_channels - in_channels)
    elif pos_embed_type == 'linear':
        return PosEncodingLinear(in_channels, out_channels - in_channels)
    elif pos_embed_type == 'mlp':
        return PosEncodingMLP(in_channels, out_channels - in_channels,
                              dim_hidden=[128])
    raise ValueError(F'Unknown pos_embed_type={pos_embed_type}')


def get_group_module_v2(
        patch_type: str,
        patch_size: int,
        recenter: bool = False,
        FPS_random_start_point:bool=False):
    """
    Return V2 modules which are slightly more ergonomic to use.
    """
    if patch_type == 'hilbert':
        return GroupHilbertV2(patch_size, recenter)
    elif patch_type == 'fps':
        return GroupFPSV2(patch_size, recenter, FPS_random_start_point)
    raise KeyError(F'Unknown patch_type={patch_type}')


def get_group_module(patch_type: str,
                     patch_size: int,
                     patch_overlap: float,
                     FPS_random_start_point:bool=False):
    if patch_type == 'hilbert':
        sort = SpatialSort(th.jit.script(HilbertCode()))
        if patch_overlap != 1.0:
            # FIXME: hardcoded `512`, `(1.25, 1.75)`
            params = _get_overlap_patch_params(512,
                                               patch_size,
                                               (1.25, 1.75))
            true_patch_size, patch_stride = params
        else:
            true_patch_size = patch_size
            patch_stride = patch_size
        group = GroupHilbert(patch_size,
                             patch_overlap,
                             true_patch_size)
    elif patch_type == 'fps':
        true_patch_size = int(patch_overlap * patch_size)
        sort = None
        patch_stride = None
        group = GroupFPS(patch_size, true_patch_size, random_start_point=FPS_random_start_point)
    return (sort, group, true_patch_size, patch_stride)


def get_patch_module(patch_type: str, embed_size: int,
                     patch_size: Optional[int] = None,
                     sort_mlp: bool = False,
                     pre_ln_bias: bool = False
                     ):
    if patch_type == 'cnn':
        return ConvPatchEncoder(embed_size)
    elif patch_type == 'mlp':
        return MLPPatchEncoder(MLPPatchEncoder.Config(
            sort=sort_mlp, pre_ln_bias=pre_ln_bias), patch_size, embed_size)
    elif patch_type == 'knn':
        return KNNPatchEncoder(patch_size, embed_size)
    elif patch_type == 'minipn':
        return MiniPNPatchEncoder(embed_size)
    raise ValueError(F'Unknown patch_type={patch_type}')


class PointMAE(nn.Module):
    '''
    not used currently
    '''
    @dataclass
    class Config(ConfigBase):
        mask_ratio: float = 0.0
        patch_size: int = 32
        encoder_channel: int = 128
        encoder: PointMAEEncoder.Config = PointMAEEncoder.Config()
        decoder: PointMAEDecoder.Config = PointMAEDecoder.Config()
        patch_type: str = 'fps'  # fps/hilbert
        patch_encoder_type: str = 'mlp'  # mlp/knn/cnn
        patch_overlap: float = 1.0  # only used for `fps`
        decode_offset: bool = True
        pos_embed_type: str = 'mlp'

        # Loss configs I guess
        patchwise_chamfer: bool = True
        # `scale_loss` reweights losses so that
        # "smaller" objects are not penalized
        # simply because distances are smaller for them.
        # this flag is probably required
        # in case normalization is disabled.
        scale_loss: bool = False
        # Only compute loss for "hidden" part
        # of MAE.
        hide_only: bool = True
        chamfer_norm: int = 1
        embed_type: str = 'post_encoder'
        # embed_type: str = 'pre_decoder'
        sort_embed: bool = True
        p_drop: float = 0.0

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder = recursive_replace_map(self.encoder, {
                'layer.hidden_size': self.encoder_channel,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop,
            })
            self.decoder = recursive_replace_map(self.decoder, {
                'patch_size': self.patch_size,
                'hidden_size': self.encoder_channel,
                'decoder_hidden_size': self.encoder_channel,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop,
            })

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        raise NotImplementedError

        (self.sort, self.group,
         self.true_patch_size, self.patch_stride) = get_group_module(
            cfg.patch_type, cfg.patch_size, cfg.patch_overlap
        )
        self.patch_encoder = get_patch_module(cfg.patch_encoder_type,
                                              cfg.encoder_channel,
                                              self.true_patch_size)
        self.pos_embed = get_pos_enc_module(cfg.pos_embed_type,
                                            cfg.encoder_channel)

        self.encoder = PointMAEEncoder(cfg.encoder)
        self.decoder = PointMAEDecoder(cfg.decoder)
        self.layernorm = nn.LayerNorm(cfg.encoder.layer.hidden_size,
                                      eps=cfg.encoder.layer.layer_norm_eps)

    def _group(self, x: th.Tensor,
               center: Optional[th.Tensor] = None,
               aux: Optional[Dict[str, th.Tensor]] = None):
        cfg = self.cfg
        with th.no_grad():
            if cfg.patch_type == 'hilbert':
                assert (center is None)
                # PATCH BY HILBERT MAPPING
                # First, sort by hilbert code
                x = self.sort(x, aux=aux)
                # Group into normalized patches and
                # patch centers.
                p, c = self.group(x, center=center, aux=aux)
            elif cfg.patch_type == 'fps':
                # PATCH BY FPS
                p, c = self.group(x, center=center, aux=aux)
            else:
                raise ValueError(F'Unknown patch_type={cfg.patch_type}')
        return (p, c)

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
        if isinstance(cfg.mask_ratio, Iterable):
            mask_ratio = np.random.choice(cfg.mask_ratio)
            z_keep, m, i = mask(z, mask_ratio=mask_ratio, noise=noise)
        else:
            z_keep, m, i = mask(z, mask_ratio=cfg.mask_ratio, noise=noise)
        return (z_keep, m, i, pe)

    def _encode(self, z: th.Tensor):
        # Encoder
        z, _, _ = self.encoder(z)
        z = self.layernorm(z)
        return z

    def _decode(self, z: th.Tensor, i: th.Tensor, pe: th.Tensor,
                **kwds):
        return self.decoder(z, i, pe, **kwds)

    def forward(self, x: th.Tensor,
                noise: Optional[th.Tensor] = None,
                aux: Optional[Dict[str, th.Tensor]] = None,
                loss: bool = False,
                rec: bool = False,
                combine: bool = True,
                cloud_gt: Optional[th.Tensor] = None,
                sort_embed: bool = False,
                center: Optional[th.Tensor] = None,
                zd_only: bool = False
                ) -> th.Tensor:
        cfg = self.cfg

        p, c = self._group(x,
                           center=center,
                           aux=aux)

        if aux is not None:
            aux['patch_center'] = c
            aux['centered_patches'] = p

        # Embed each patch
        z, m, i, pe = self._embed(p, c, noise=noise)

        # Encoder across patches
        z = self._encode(z)

        if aux is not None:
            if sort_embed:
                # assert mask_ratio==0
                if cfg.embed_type == 'post_encoder':
                    zz = th.gather(
                        z, dim=1, index=i.unsqueeze(-1).repeat(1, 1, z.shape[2]))
                elif cfg.embed_type == 'pre_decoder':
                    zz = self.decoder.unshuffle(z, i, pe)
                else:
                    raise ValueError(F'Unknown embed_type={cfg.embed_type}')
                aux['embed'] = zz
            else:
                aux['embed'] = z

        # At this point, we assume
        # y' = y+c
        y, zd, _ = self._decode(z, i,
                                pe,
                                # th.randn_like(pe)
                                # pe=self.pos_embed(c + 0.1 * th.randn_like(c)),
                                output_hidden_states=True
                                )

        if aux is not None:
            aux['mask'] = m

        if zd_only:
            # return zd
            return y

        if loss:
            hide = m.bool()
            if cfg.hide_only:
                loss_mask = hide
            else:
                # NOTE: not very efficient,
                # but robust and convenient
                loss_mask = th.ones_like(hide)

            # At the end of this operation, we get
            # `target` in the same form as `p`
            if cloud_gt is not None:
                if cfg.patch_type == 'hilbert':
                    target = th.take_along_dim(
                        cloud_gt, aux['sort_index'][..., None], -2)
                    target = self._group(target, sub=False)[0]
                else:
                    target = _knn_gather(cloud_gt, aux['fps_nn_idx'])
                target = target.reshape(p.shape)

                if cfg.decode_offset:
                    # Apply centering operation to gt patches.
                    # `c` statistics must be taken from `x`.
                    target = (target - c[..., None, :])
            else:
                if cfg.decode_offset:
                    # `p` is already subtracted
                    # if (cfg.patch_type == 'fps') and (cfg.patch_overlap != 0):
                    #     # Need to recompute `target` without overlaps i guess?
                    #     target = knn_points(c, x, K=cfg.patch_size,
                    #                         return_nn=False)
                    #     target = target - c[..., None, :]
                    # else:
                    target = p
                else:
                    # Otherwise we reshape `x` to match `p`
                    # This is OK since `x` is already sorted (above)
                    # to match the patch order.
                    target = x.reshape(p.shape)

            if cfg.patchwise_chamfer:
                target = einops.rearrange(target,
                                          '... s p d -> (... s) p d')
                pred = einops.rearrange(y,
                                        '... (s p) d -> (... s) p d',
                                        p=cfg.patch_size)
            else:
                target = einops.rearrange(target,
                                          '... s p d -> ... (s p) d')
                pred = y

            # weights = None
            scale = None
            if cfg.scale_loss:
                scale_src = (x if cloud_gt is None else cloud_gt)
                stats = {}
                with th.no_grad():
                    _ = normalize(scale_src, aux=stats, in_place=False)
                scale = th.reciprocal(stats['radius']).squeeze()  # b

                if cfg.patchwise_chamfer:
                    # => (b s) p
                    scale = einops.repeat(scale, '... -> (... s)',
                                          s=p.shape[-3],
                                          # p=p.shape[-2]
                                          )
                # else:
                #    # => (b)
                #    # scale = einops.repeat(scale, '... -> ... (s p)',
                #    #                       s=p.shape[-3],
                #    #                       p=p.shape[-2])
                #    pass

                # if weights is None:
                #    weights = scale
                # else:
                #    weights = weights * scale

            # Compute scaled (normalized) predictions and targets
            if cfg.scale_loss:
                pred_s = scale[:, None, None] * pred
                targ_s = scale[:, None, None] * target
            else:
                pred_s = pred
                targ_s = target

            if cfg.hide_only:
                s0 = merge_shapes(
                    # batch leading dimensions
                    x.shape[:-2],
                    # num patches
                    x.shape[-2] // self.cfg.patch_size,
                    self.cfg.patch_size,
                    x.shape[-1])

                s1 = merge_shapes(
                    # batch leading dimensions
                    x.shape[:-2],
                    # num patches
                    x.shape[-2] // self.cfg.patch_size,
                    self.true_patch_size,
                    x.shape[-1])
                # target_s = (4096,47,3) = (256,16,47,3)
                len_keep = int(s0[-3] * (1.0 - cfg.mask_ratio))
                len_hide = s0[-3] - len_keep
                # print(s0,s1)
                XX = pred_s.reshape(s0)[loss_mask].reshape(
                    -1, len_hide * self.cfg.patch_size, pred.shape[-1])
                YY = targ_s.reshape(s1)[loss_mask].reshape(
                    -1, len_hide * self.true_patch_size, target.shape[-1])
                cd = chamfer_distance(XX, YY,
                                      norm=cfg.chamfer_norm
                                      # weights=weights
                                      )[0]
            else:
                cd = chamfer_distance(pred_s, targ_s,
                                      # weights=weights
                                      norm=cfg.chamfer_norm
                                      )[0]
            # if cfg.scale_loss:
            #    cd = weights.sum() / p.shape[-3] * cd

            # cd = (cd * weights) / weights.sum()
            aux['loss'] = cd

        if rec:
            if combine:
                hide = m.bool()
            else:
                # pretend like everything was hidden
                hide = th.ones_like(m).bool()

            # rec = th.where(hide[..., None, None],
            # y.reshape(p.shape), p.add(c[...,None,:])).reshape(x.shape)
            if cfg.patch_overlap != 1.0:
                s0 = merge_shapes(
                    # batch leading dimensions
                    x.shape[:-2],
                    # num patches
                    x.shape[-2] // self.cfg.patch_size,
                    self.cfg.patch_size,
                    x.shape[-1])

                if cfg.decode_offset:
                    yy = y.reshape(s0).add(c[..., None, :])
                    pp = p.add(c[..., None, :])
                else:
                    yy = y.reshape(s0)
                    pp = p.add(c[..., None, :])

                args = [yy[hide].reshape(*x.shape[:-2], -1, x.shape[-1]),
                        pp[~hide].reshape(*x.shape[:-2], -1, x.shape[-1])]
                rec = th.cat(args, dim=-2)

                args = [th.ones_like(yy)[hide].reshape(*x.shape[:-2], -1, x.shape[-1]),
                        th.zeros_like(pp)[~hide].reshape(*x.shape[:-2], -1, x.shape[-1])]
                aux['label'] = th.cat(args, dim=-2)
            else:
                yy = y.reshape(p.shape)
                if cfg.decode_offset:
                    yy = yy + c[..., None, :]

                rec = th.where(hide[..., None, None],
                               yy,
                               p + c[..., None, :])
                rec = rec.reshape(*x.shape[:-2], -1, rec.shape[-1])
                aux['label'] = hide[..., None, None].expand_as(p)
            aux['rec'] = rec

        return y


def test_hilbert_patch_overlap():
    from matplotlib import pyplot as plt

    cloud = 0 * th.randn((1, 512, 3), dtype=th.float)
    num_patch: int = 16
    patch_size: int = 32
    overlap_ratio: float = 1.5
    true_patch_size, stride = _get_overlap_patch_params(cloud.shape[-2],
                                                        patch_size)
    print(true_patch_size)
    print(stride)

    x = cloud
    s = x.shape
    xx = x.reshape(*s[:-2], -1, patch_size, 3)
    lengths = (*s[:-2], num_patch, true_patch_size, 3)
    strides = xx.stride()[:-3] + (stride, 3, 1)
    xxx = xx.as_strided(lengths, strides)
    for i in range(xxx.shape[-3]):
        xxx[0, i, :] += 1
        plt.clf()
        plt.bar(np.arange(x.shape[-2]),
                dcn(x)[..., 0].ravel())
        plt.savefig(F'/tmp/{i:03d}.png')
        plt.show()
    print(xxx.shape)
    print(cloud.sum())


def test_knn_patch_encoder():
    enc = KNNPatchEncoder(64, 128)
    ic(enc)
    cloud = th.randn((8, 512, 3), dtype=th.float)
    out = enc(cloud)
    print(out.shape)

    index = th.randperm(cloud.shape[-2])
    cloud = th.take_along_dim(cloud, index[None, :, None],
                              dim=-2)
    out2 = enc(cloud)
    print((out - out2).max())
    print((out - out2).min())


def test_minipn():
    model = MiniPNPatchEncoder(128)
    patch = th.randn((4, 16, 32, 3), dtype=th.float)
    z1 = model(patch)
    patch2 = th.take_along_dim(patch, th.randperm(
        patch.shape[-2])[None, None, :, None], dim=-2)
    print(patch.shape, patch2.shape)
    z2 = model(patch2)
    print(z1.shape, z1.shape)
    dz = ((z1 - z2))
    print(dz.min(), dz.max())
    ic(model)


def test_mlp_sort():
    model = MLPPatchEncoder(MLPPatchEncoder.Config(sort=True), 32, 128)
    ic(model)

    patch = th.randn((4, 16, 32, 3), dtype=th.float)
    patch2 = th.take_along_dim(patch, th.randperm(
        patch.shape[-2])[None, None, :, None], dim=-2)
    patch3 = patch2 + 0.001 * th.randn_like(patch2)
    patch4 = patch + 0.001 * th.randn_like(patch)

    z1 = model(patch)
    z2 = model(patch2)
    z3 = model(patch3)
    z4 = model(patch4)
    zs = [z1, z2, z3, z4]
    label = ['plain', 'perm', 'noise+perm', 'noise']
    with th.no_grad():
        for i in range(4):
            for j in range(i + 1, 4):
                dz = zs[i] - zs[j]
                print(label[i], label[j])
                print(i, j, dz.min(), dz.max(), dz.std())


def test_knn_gather():
    for _ in range(16):
        with nvtx.annotate("x"):
            x = th.randn((8, 64, 3), dtype=th.float,
                         device='cuda:0')
        with nvtx.annotate("d"):
            d = th.linalg.norm(x[..., None, :, :] - x[..., :, None, :], dim=-1)
        with nvtx.annotate("i"):
            i = th.topk(d, 8, dim=-1).indices
        with nvtx.annotate("g0"):
            g0 = knn_gather(x, i)
        with nvtx.annotate("g1"):
            g1 = _knn_gather(x, i)
        with nvtx.annotate("compare"):
            print((g0 - g1).min(),
                  (g0 - g1).max())


def test_groups():
    # group = get_group_module_v2('hilbert', 32)
    # aux = {}
    # x = th.randn((1, 512, 3))
    # p, c = group(x, aux=aux)
    # print(p.shape)
    # print(c.shape)
    # print(aux.keys())

    # group = get_group_module_v2('fps', 32)
    # x = th.randn((1, 512, 3))
    # p, c = group(x, aux=aux)
    # print(p.shape)
    # print(c.shape)
    # print(aux.keys())
    group = My_GroupFPS(16, 64)
    data = th.randn((1, 512, 3))
    aux = {}
    p, c = group(data, aux=aux)
    idx = aux['fps_nn_idx']
    FPS_PATCH_BASE.index_points(data, idx)
    print(p.shape, c.shape)
    print(aux)


def test_output_attention():
    device: str = 'cuda:1'
    model = PointMAEEncoder(recursive_replace_map(PointMAEEncoder.Config(),
                                                  {'layer.hidden_size': 96}))
    x = th.randn((1, 512, 3))
    group = get_group_module_v2('fps', 32)
    p, c = group(x, aux={})
    print(p.shape, c.shape)
    z = p - c[..., :, None, :]
    z = einops.rearrange(z, '... s p d -> ... s (p d)')
    model = model.to(device)
    z = z.to(device)
    out1, _, _ = model(z)
    out2, _, _ = model(z, output_attentions=True)
    delta = (out1 - out2)
    ic(delta.min(), delta.max(),
       delta.mean(), delta.std())


def test_key_padding_mask():
    device: str = 'cuda:1'
    model = PointMAEEncoder(recursive_replace_map(PointMAEEncoder.Config(),
                                                  {'layer.hidden_size': 96}))
    x = th.randn((1, 512, 3))
    group = get_group_module_v2('fps', 32)
    p, c = group(x, aux={})
    z = p - c[..., :, None, :]
    z = einops.rearrange(z, '... s p d -> ... s (p d)')
    model = model.to(device)
    z = z.to(device)

    if True:
        print('A')
        key_padding_mask = th.zeros(z.shape[:-1],
                                    dtype=th.bool,
                                    device=device)
        print(key_padding_mask)
        print(z.shape, key_padding_mask.shape)
        # key_padding_mask[:] = 1
        out1, _, _ = model(z)
        out2, _, _ = model(z, key_padding_mask=key_padding_mask,
                           output_attentions=False)
        delta = (out1 - out2)
        ic(delta.min(), delta.max(),
           delta.mean(), delta.std())

    if True:
        print('B')
        key_padding_mask = th.zeros(z.shape[:-1],
                                    dtype=th.bool,
                                    device=device)
        key_padding_mask[..., 2:] = 1
        print(key_padding_mask)
        out1, _, _ = model(z[..., :2, :])
        out2, _, _ = model(z, key_padding_mask=key_padding_mask)
        print(out1.shape, out2.shape)
        delta = (out1 - out2[..., :2, :])
        ic(delta.min(), delta.max(),
           delta.mean(), delta.std())

    if True:
        print('C')
        key_padding_mask = th.zeros(z.shape[:-1],
                                    dtype=th.bool,
                                    device=device)
        key_padding_mask[..., 4:] = 1
        print(key_padding_mask)
        out1, _, _ = model(z[..., :4, :])
        out2, _, _ = model(z,
                           key_padding_mask=key_padding_mask,
                           output_attentions=True)
        print(out1.shape, out2.shape)
        delta = (out1 - out2[..., :4, :])
        ic(delta.min(), delta.max(),
           delta.mean(), delta.std())

def test_patch_encoder():
    encoder = MultiLayer_ConvPatchEncoder(
        mlps = [[32], [128]],
        patch_nums = [16, 16],
        K=32
    )
    print(encoder)
    data = th.randn((2, 512, 3))
    xyz, vec = encoder(data)
    print(vec.shape)


def main():
    # test_knn_gather()
    # test_mlp_sort()
    # cfg = PointMAE.Config()
    # ic(cfg)
    # test_knn_patch_encoder()
    # test_hilbert_patch_overlap()
    test_groups()
    # test_output_attention()
    # test_key_padding_mask()
    # test_patch_encoder()


if __name__ == '__main__':
    main()
