#!/usr/bin/env python3

import math
from collections import defaultdict
from typing import (Tuple, Optional, Iterable,
                    List, Union, Mapping, Dict,
                    Any, Callable)
from dataclasses import (
    dataclass, is_dataclass, fields, replace)

from functools import partial
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange
from einops.layers.torch import EinMix, Rearrange

from util.config import ConfigBase
from util.torch_util import merge_shapes

from icecream import ic
from flash_attn.flash_attention import FlashAttention

S = Union[int, Tuple[int, ...]]
T = th.Tensor

class SingleGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int):
        super().__init__()
        self.state_size = state_size
        self.cell = nn.GRUCell(input_size, state_size)

    def init_memory(self, batch_shape: Tuple[int, ...] = (),
                    **kwds):
        return th.zeros((*batch_shape, self.state_size), **kwds)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out = self.cell(x, s)
        return (out, out)


class SingleLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, state_size)

    def init_memory(self, batch_shape: Tuple[int, ...] = (),
                    **kwds):
        h_0 = th.zeros((*batch_shape, self.state_size), **kwds)
        c_0 = th.zeros((*batch_shape, self.state_size), **kwds)
        return (h_0, c_0)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out, hid = self.cell(x, s)
        return (out, hid)


class DeepGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 state_size: int,
                 num_layer: int = 2, batch_shape: Tuple[int, ...] = (),
                 **kwargs):
        super().__init__()
        self.num_layer = num_layer
        self.state_size = state_size
        gru = nn.GRU(
            input_size=input_size,
            hidden_size=state_size,
            num_layers=num_layer,
            bias=True,
            batch_first=False,
            bidirectional=False)
        self.gru = gru

        self.init_memory(batch_shape)

    def init_memory(self,
                    batch_shape: Tuple[int, ...] = (),
                    **kwds):
        
        self.register_buffer('memory', th.zeros((self.num_layer, *batch_shape, self.state_size), **kwds))
        return self.memory
        # device = next(self.gru.parameters()).device 
        # return th.zeros(
        #                 (self.num_layer, *batch_shape, self.state_size), device= device,
        #                 **kwds)

    def forward(self, x: th.Tensor, s: th.Tensor):
        out, hid = self.gru(x[None], s)
        out = out.squeeze(dim=0)
        return (out, hid)

class CountDormantNeurons:
    def __init__(self, on_count,
                 target_cls: Union[nn.Module, Iterable[nn.Module]] = nn.ELU,
                 # "0.025 for default setting, 0.1 otherwise"
                 tau: float = 0.1):
        self._on_count = on_count
        self._target_cls = target_cls
        self._stats = defaultdict(list)
        self._tau = tau
        self._keys = []

    def _on_fwd(self, model, input: th.Tensor,
                output: th.Tensor, name: str = None):
        h = output.detach()

        ah = th.abs(h)
        s_numer = ah.mean(dim=0)
        # s_denom = s_numer.sum(dim=-1, keepdim=True)
        s_denom = s_numer.mean(dim=-1, keepdim=True)
        self._stats[name].append(
            (s_numer / s_denom))

        # For now, just assume the same module won't be called twice
        if (name in self._stats) and (self._on_count is not None):
            self._on_count(self._stats)
            self._stats = defaultdict(list)

    def hook(self, model: nn.Module):
        # model.register_forward_hook(self._on_fwd)
        for name, layer in model.named_modules():
            if isinstance(layer, self._target_cls):
                layer.register_forward_hook(partial(self._on_fwd, name=name))
                self._keys.append(name)


def conv2d_dim_out(
        dim_in: Tuple[int, int, int],
        channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int):
    c, h, w = dim_in
    c_out = channels
    w_out = int(
        (w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    h_out = int(
        (h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )
    return (c_out, h_out, w_out)


class FlipFlopReLU6(nn.Module):
    """
    ReLU6, except half of the channels are clamped to (-6~0),
    the other half is clamped to (0~+6).
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.neg = nn.modules.activation.Hardtanh(-6.0, 0.0,
                                                  inplace=inplace)
        self.pos = nn.modules.activation.Hardtanh(0.0, 6.0,
                                                  inplace=inplace)

    def forward(self, x: th.Tensor):
        p, n = th.chunk(x, 2, dim=-1)
        p = self.pos(p)
        n = self.neg(n)
        if self.inplace:
            return x.mul_(2.0)
        else:
            return 2.0 * th.cat([p, n], dim=-1)


def _map_struct(
        src: Union[Iterable[Any], Mapping[str, Any], Any],
        op: Callable[[Any, Any], Any],
        dst: Optional[Union[Iterable[Any], Mapping[str, Any], Any]] = None,
        base_cls=None,
        base_fn=None,
        dict_cls=(Mapping, nn.ModuleDict)
):
    """
    Define a generic mapping from one structure to another.
    Generally supports iterables and mappings.
    """
    map_element = partial(map_struct,
                          base_cls=base_cls,
                          base_fn=base_fn,
                          dict_cls=dict_cls)
    # Since stuff like torch.Tensor is technically an iterable,
    # we can artificially set the "leaf" calls by explicitly
    # checking for membership in `base_cls`.
    if base_cls is not None and isinstance(src, base_cls):
        return op(src, dst)

    if base_fn is not None and base_fn(src, dst):
        return op(src, dst)

    # map_dataclass()
    if is_dataclass(src):
        if is_dataclass(dst):
            # Update `dst` in-place...
            for field in fields(src):
                setattr(dst, field.name,
                        map_element(
                            getattr(src, field.name),
                            op,
                            getattr(dst, field.name)))
            return dst
        else:
            # out-of-place version
            out_dict = {}
            for field in fields(src):
                out_dict[field.name] = map_element(
                    getattr(src, field.name), op,
                    None
                )
            out = replace(src, **out_dict)
            return out

    # NOTE: Mapping should _always_ come before
    # Iterable, since Mapping often also an iterable.
    # for instance, dict() is also an iterable.
    # map_mapping()
    # FIXME: nn.ModuleDict specialization is required
    # due to nn.ModuleDict not including Mapping in MRO
    if isinstance(src, dict_cls):
        out = {}
        dst_is_mapping = isinstance(dst, dict_cls)
        for k in src.keys():
            try:
                if dst_is_mapping and k in dst:
                    out[k] = map_element(src[k], op, dst[k])
                else:
                    out[k] = map_element(src[k], op, None)
            except BaseException:
                print(F'Processing failed at key = {k}')
                raise
        return out

    # map_iterable()
    if isinstance(src, Iterable):
        if isinstance(dst, Iterable) and len(src) == len(dst):
            out = [map_element(vs, op, vd)
                   for (vs, vd) in zip(src, dst)]
        else:
            out = [map_element(vs, op, None) for vs in src]
        return out

    return op(src, dst)


def map_struct(*args, **kwds):
    out = _map_struct(*args, **kwds)
    if is_dataclass(out) and hasattr(out, '__post_init__'):
        out.__post_init__()
    return out


map_tensor = partial(map_struct,
                     base_cls=th.Tensor)


def get_activation_function(act_cls: str) -> nn.Module:
    if not isinstance(act_cls, str):
        return act_cls
    act_cls = act_cls.lower()
    if act_cls == 'tanh':
        out = nn.Tanh
    elif act_cls == 'relu':
        out = nn.ReLU
    elif act_cls == 'lrelu':
        out = nn.LeakyReLU
    elif act_cls == 'elu':
        out = nn.ELU
    elif act_cls == 'relu6':
        out = nn.ReLU6
    elif act_cls == 'gelu':
        out = nn.GELU
    elif act_cls == 'selu':
        out = nn.SELU
    elif act_cls == 'frelu6':
        out = FlipFlopReLU6
    elif act_cls == 'none':
        out = nn.Identity
    else:
        raise KeyError(F'Unknown act_cls={act_cls}')
    return out


class DClamp(th.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
            ctx, input: th.Tensor, min: Union[float, th.Tensor],
            max: Union[float, th.Tensor]):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(
        input: th.Tensor, min: Union[float, th.Tensor],
        max: Union[float, th.Tensor]):
    return DClamp.apply(input, min, max)


def grad_step(
        loss: Optional[th.Tensor],
        optimizer: th.optim.Optimizer,
        scaler: Optional[th.cuda.amp.GradScaler] = None,
        parameters: Optional[nn.ParameterList] = None,
        max_grad_norm: Optional[float] = 1.0,
        step_grad: bool = True,
        skip_nan: bool = True,
        zero_grad: bool = True,
        ** bwd_args
):
    """
    Optimizer Step with optional AMP / grad clipping.

    Performs following operations:
        * loss.backward()
        * clip_grad_norm()
        * step() (optional)
        * zero_grad() (optional)

    Optionally applies scaler() related operations
    if AMP is enabled (triggered by scaler != None)
    """

    # Try to automatically fill out `parameters`
    if (max_grad_norm is not None) and (parameters is None):
        parameters = optimizer.param_groups[0]['params']

    if (scaler is not None):
        # With AMP + clipping
        if loss is not None:
            scaler.scale(loss).backward(**bwd_args)
        if step_grad:
            if (max_grad_norm is not None) and (parameters is not None):
                skip_step: bool = False
                try:
                    grad_norm = nn.utils.clip_grad_norm_(
                        parameters,
                        max_grad_norm,
                        error_if_nonfinite=skip_nan)
                except RuntimeError:
                    skip_step = True
                if not skip_step:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                if zero_grad:
                    optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                if zero_grad:
                    optimizer.zero_grad()
    else:
        # Without AMP
        if loss is not None:
            loss.backward(**bwd_args)
        if (max_grad_norm is not None) and (parameters is not None):
            skip_step: bool = False
            try:
                grad_norm = nn.utils.clip_grad_norm_(parameters,
                                                     max_grad_norm)
            except RuntimeError:
                skip_step = True
            if not skip_step:
                optimizer.step()
            if zero_grad:
                optimizer.zero_grad()  # set_to_none?
        if step_grad:
            optimizer.step()
            if zero_grad:
                optimizer.zero_grad()  # set_to_none?


def soft_update(
        src: nn.Module,
        dst: nn.Module,
        tau: float):
    """ soft update by `tau`: src -> dst"""
    with th.no_grad():
        for p0, p1 in zip(src.parameters(), dst.parameters()):
            p1.copy_(p1 * (1.0 - tau) + p0 * tau)


def hard_update(
        src: nn.Module,
        dst: nn.Module):
    """ hard update (copy): src -> dst"""
    with th.no_grad():
        for p0, p1 in zip(src.parameters(), dst.parameters()):
            p1.copy_(p0)


class LinearBn(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self, dim_in: int, dim_out: int,
                 use_bn: bool = True,
                 use_ln: bool = False, **kwds):
        super().__init__()

        if use_bn and use_ln:
            raise ValueError('use_bn and use_ln cannot both be true!')

        if use_bn or use_ln:
            if use_bn:
                kwds['bias'] = False
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            if use_ln:
                self.bn = nn.LayerNorm(dim_out)
            else:
                self.bn = nn.BatchNorm1d(dim_out)
        else:
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            self.bn = nn.Identity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.bn(x)
        x = x.reshape(s)
        return x


class MultiHeadLinear(nn.Module):
    """
    Linear layer with multiple "heads".

    1. Supports multi-head output.
    2. Supports arbitrary input sizes.
    """

    def __init__(self,
                 dim_in: int,
                 num_head: int,
                 dim_head: int,
                 unbind: bool = False,
                 *args, **kwds):
        super().__init__()
        self.linear = nn.Linear(dim_in, num_head * dim_head,
                                *args, **kwds)
        self.h = num_head
        self.d = dim_head
        self.unbind = unbind

    def extra_repr(self):
        return F'head: {self.h}'

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: (..., d_x)
        Returns:
            y: (..., h, d_h)
        """
        s = x.shape

        # Apply Linear() on last axis.
        x = x.reshape(-1, x.shape[-1])
        y = self.linear(x)
        y = y.reshape(*s[:-1], self.h, self.d)
        if self.unbind:
            y = y.unbind(dim=-2)
        return y


class MLP(nn.Module):
    """ Generic multilayer perceptron. """

    def __init__(self, dims: Tuple[int, ...],
                 act_cls: nn.Module = nn.LeakyReLU,
                 activate_output: bool = False,
                 use_bn: bool = True,
                 bias: bool = True,
                 use_ln: bool = False,
                 pre_ln_bias: bool = True):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(act_cls, str):
            act_cls = get_activation_function(act_cls)

        layers = []
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            # FIXME: incorrect `bias` logic
            if not use_ln:
                layer_bias = bias
            else:
                layer_bias = pre_ln_bias
            layers.extend(
                (LinearBn(
                    d0,
                    d1,
                    use_bn=use_bn,
                    bias=layer_bias,
                    use_ln=use_ln),
                    act_cls(),
                 ))
        if activate_output:
            layers.extend((
                LinearBn(
                    dims[-2],
                    dims[-1],
                    use_bn=use_bn, bias=bias, use_ln=use_ln),
                act_cls()))
        else:
            # FIXME: not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        return self.model(x)


class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in https://arxiv.org/abs/1504.00702.
    Concretely, the spatial softmax of each feature map is used to compute a weighted
    mean of the pixel locations, effectively performing a soft arg-max over the feature
    dimension.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()

        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: th.device,
    ) -> th.Tensor:
        if self.normalize:
            return th.stack(
                th.meshgrid(
                    th.linspace(-1, 1, w, device=device),
                    th.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        return th.stack(
            th.meshgrid(
                th.arange(0, w, device=device),
                th.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # Compute a spatial softmax over the input:
        # Given an input of shape (B, C, H, W), reshape it to (B*C, H*W) then apply the
        # softmax operator over the last dimension.
        _, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # Create a meshgrid of normalized pixel coordinates.
        xc, yc = self._coord_grid(h, w, x.device)

        # Element-wise multiply the x and y coordinates with the softmax, then sum over
        # the h*w dimension. This effectively computes the weighted mean x and y
        # locations.
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # Concatenate and reshape the result to (B, C*2) where for every feature we have
        # the expected x and y pixel locations.
        return th.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with two intermediate layers,
    connected with LeakyReLU activation.
    """

    def __init__(self, depth: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depth, depth, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        )

    def forward(self, x: th.Tensor):
        out = self.block(x)
        return out + x


class AddCoord(nn.Module):
    """
    Add (two) normalized spatial coordinates to the channel dimensions
    for `CoordConv` (for better spatial awareness).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # [1] create grid.
        h, w = x.shape[-2:]
        grid = th.cartesian_prod(
            th.linspace(-1.0, +1.0, h, device=x.device),
            th.linspace(-1.0, +1.0, w, device=x.device)
        ).reshape(h, w, 2)
        s = list(x.shape)
        s[-3] = -1
        grid = th.permute(grid, (2, 0, 1)).expand(s)

        # [2] concat grid and return.
        out = th.cat([x, grid], dim=-3)
        return out


class ConvBlock(nn.Module):

    @dataclass
    class Config(ConfigBase):
        # c_in:int
        channels: int = -1
        kernel_size: int = 3
        stride: int = 1
        padding: int = 1  # corresponds to "same" padding.
        # We omit dilation/groups
        # assuming we will not use them too much ...
        batch_norm: bool = True
        add_coord: bool = False
        bias: bool = True  # ignored if batch_norm :)
        act_cls: str = 'relu'

        def dim_out(
            self,
            dim_in: Tuple[int, int, int]
        ) -> Tuple[int, int, int]:
            return conv2d_dim_out(dim_in,
                                  self.channels,
                                  self.kernel_size,
                                  self.stride,
                                  self.padding,
                                  1)

    def __init__(self, cfg: Config, in_channels: int):
        super().__init__()
        self.cfg = cfg

        bias = (cfg.bias and not cfg.batch_norm)

        coord = (AddCoord() if cfg.add_coord else nn.Identity())
        bn = (nn.BatchNorm2d(cfg.channels) if cfg.batch_norm else nn.Identity())
        act_cls = get_activation_function(cfg.act_cls)
        c_in = (in_channels + 2 if cfg.add_coord else in_channels)

        self.conv = nn.Sequential(
            coord,
            nn.Conv2d(c_in, cfg.channels,
                      cfg.kernel_size,
                      cfg.stride,
                      cfg.padding,
                      bias=bias),
            bn,
            act_cls()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        s = x.shape
        x = x.reshape(-1, *x.shape[-3:])
        x = self.conv(x)
        x = x.reshape(*s[:-3], *x.shape[1:])
        return x


class SimpleCNN(nn.Module):

    def __init__(self,
                 blocks: Tuple[ConvBlock.Config, ...],
                 in_channels: int):
        super().__init__()

        layers = []
        c_in: int = in_channels
        for block in blocks:
            layers.append(ConvBlock(block, c_in))
            c_in = block.channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        s = x.shape
        x = x.reshape(-1, *x.shape[-3:])
        x = self.net(x)
        x = x.reshape(*s[:-3], *x.shape[1:])
        return x


class CNN(nn.Module):
    def __init__(self, in_channels: int,
                 blocks: Tuple[int, ...] = (16, 32, 32),
                 activate: bool = True
                 ):
        super().__init__()

        hid_in = in_channels

        layers = []
        for hid_out in blocks:
            layers.extend(
                [
                    nn.Conv2d(hid_in, hid_out, 3, padding=1,
                              bias=False),
                    nn.BatchNorm2d(hid_out),
                    ResidualBlock(hid_out)
                ]
            )
            hid_in = hid_out
        self.net = nn.Sequential(*layers)
        self.activate = (nn.ReLU(True) if activate
                         else nn.Identity())

    def forward(self, x: th.Tensor):
        out = self.net(x)
        out = self.activate(out)
        return out

# class ViT(nn.Module):
#     """
#     Wrapper arround huggingface vit
#     """
#     @dataclass
#     class Config(ConfigBase):
#         hidden_size: int = 256
#         num_hidden_layers: int = 4
#         num_attention_heads: int = 8
#         intermediate_size: int = 256
#         hidden_act: str = 'gelu'
#         hidden_dropout_prob: float = 0.0
#         attention_probs_dropout_prob: float = 0.0
#         initializer_range: float = 0.02
#         layer_norm_eps: float = 1e-12
#         image_size: int = 48
#         patch_size: int = 4
#         num_channels: int = 2
#         qkv_bias: bool = True
#         decoder_num_attention_heads: int = 8
#         decoder_hidden_size: int = 256
#         decoder_num_hidden_layers: int = 4
#         decoder_intermediate_size: int = 256
#         mask_ratio: float = 0.75
#         norm_pix_loss: bool = False


class CBA2D(nn.Module):
    """ CONV+BN+ACTIVATION """

    def __init__(self, **kwds):
        super().__init__()
        kwds['bias'] = False
        self.model = nn.Sequential(
            nn.Conv2d(**kwds),
            nn.BatchNorm2d(kwds['out_channels']),
            nn.LeakyReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


class CBAR2D(nn.Module):
    """ CONV+BN+ACTIVATION+RESIDUAL BLOCK"""

    def __init__(self, **kwds):
        super().__init__()
        self.model = nn.Sequential(
            CBA2D(**kwds),
            ResidualBlock(kwds['out_channels'])
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


def attention(q: th.Tensor, k: th.Tensor, v: th.Tensor,
              aux: Optional[Dict[str, th.Tensor]] = None,
              key_padding_mask: Optional[th.Tensor] = None
              ) -> th.Tensor:
    """
    Args:
        q: (b,#q,dk)
        k: (b,#k,dk)
        v: (b,#k,dv)
    """
    # attention(Q,K,V) = softmax(Q@K.T/sqrt(d_k))@V
    scale: float = 1.0 / math.sqrt(q.shape[-1])
    dots = th.einsum('...qd,...kd->...qk', q, k)
    if key_padding_mask is not None:
        dots.masked_fill_(key_padding_mask[..., None, :], float('-inf'))
    # (a-b) = a.a b.b a.b
    attn = th.softmax(dots * scale, dim=-1)
    if aux is not None:
        aux['attn'] = attn
    outs = th.einsum('...qk,...kv->...qv', attn, v)
    return outs


@th.jit.script
def s_attention(q: th.Tensor, k: th.Tensor, v: th.Tensor):
    return attention(q, k, v)


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention implementation with einops.
    """

    def __init__(self,
                 dim_c: int,
                 dim_x: int,

                 dim_k: int,
                 dim_v: int,

                 dim_o: int,
                 num_k: int,
                 num_h: int = 1):
        """
        Args:
            dim_c: dim of context vector.
            dim_x: dim of query inputs.
            dim_k: dim of key.
            dim_v: dim of value.
            dim_o: dim of output.
            num_k: number of keys.
            num_h: number of heads. Defaults to 1.
        """
        super().__init__()
        self.to_q = EinMix('b q d -> b h q k',
                           weight_shape='d h k',
                           d=dim_x, k=dim_k, h=num_h)
        self.to_k = EinMix('b d -> b h n k',
                           weight_shape='d h n k',
                           d=dim_c, k=dim_k, h=num_h,
                           n=num_k)
        self.to_v = EinMix('b d -> b h n v',
                           weight_shape='d h n v',
                           d=dim_c, v=dim_v, h=num_h,
                           n=num_k)

        # Option #1
        # self.mix = EinMix('b q h v -> b q o',
        #                   weight_shape='h v o',
        #                   h=num_h, v=dim_v,
        #                   o=dim_o)

        # Option #2: reduce & mix, similar to
        # depthwise convolution.
        dim_v0 = dim_v // num_h
        dim_h0 = dim_o // dim_v0
        assert (dim_h0 * dim_v0 == dim_o)
        self.mix = nn.Sequential(
            # Reduce dimensionality at each head.
            # Sort of like depthwise
            # 1x1 convolution.
            EinMix('b h q v -> b q h v0',
                   weight_shape='v v0',
                   v=dim_v, v0=dim_v0),

            # Mix between heads.
            # Sort of like pointwise convolution.
            EinMix('b q h v -> b q (h0 v)',
                   weight_shape='h h0',
                   h=num_h,
                   h0=dim_h0)
        )

    def forward(self,
                context: th.Tensor,
                inputs: th.Tensor) -> th.Tensor:
        """
        Args:
            context: (b, d_c)
            inputs:   (b, q, d_x)

        Returns:
            output:  (b, q, h, d_o)
        """
        k = self.to_k(context)
        v = self.to_v(context)
        q = self.to_q(inputs)
        hv = attention(q, k, v)
        o = self.mix(hv)
        return o


class MultiHeadAttentionV2(nn.Module):
    """
    Adapted from einops.
    """

    def __init__(self,
                 n_head: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 dropout=0.1):
        super().__init__()
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_q.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_k.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_v.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q: th.Tensor, k: th.Tensor, v: th.Tensor, mask=None):
        residual = q
        q = rearrange(self.w_q(q), 'b l (h k) -> b l h k', h=self.n_head)
        k = rearrange(self.w_k(k), 'b t (h k) -> b t h k', h=self.n_head)
        v = rearrange(self.w_v(v), 'b t (h v) -> b t h v', h=self.n_head)
        attn = th.einsum('blhk,bthk->blht', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = th.softmax(attn, dim=3)
        output = th.einsum('blht,bthv->blhv', [attn, v])
        output = rearrange(output, 'b l h v -> b l (h v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads = 4, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # 创建线性层用于生成Q, K, V
        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)

        # FlashAttention 实例
        self.flash_attention = FlashAttention(softmax_scale=softmax_scale, attention_dropout=attention_dropout)
        
        # 输出投影层
        self.Wo = nn.Linear(embed_size, embed_size)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, key_padding_mask=None, causal=False):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)

        # 计算 Q, K, V
        Q = self.Wq(query)  # (batch_size, seq_len_q, embed_size)
        K = self.Wk(key)    # (batch_size, seq_len_kv, embed_size)
        V = self.Wv(value)  # (batch_size, seq_len_kv, embed_size)

        # 使用 FlashAttention 计算 Cross-Attention
        output = attention(Q, K, V, key_padding_mask=key_padding_mask)

        # 重塑输出并进行输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_size)
        output = self.Wo(output)

        # 添加残差连接和 LayerNorm
        output = self.layernorm1(query + output)

        return output

class CrossAttentionLayer(nn.Module):
    @dataclass
    class Config(ConfigBase):
        embed_size: int = 128
        num_heads: int = 4
        softmax_scale: Optional[float] = None
        attention_dropout: float = 0.0
        ffn_dropout: float = 0.1

    def __init__(self, cfg: Config):
        super().__init__()

        self.embed_size = cfg.embed_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_size // cfg.num_heads
        self.hidden_size = 4 * cfg.embed_size

        # Cross Attention (基于FlashAttention)
        self.cross_attention = CrossAttention(cfg.embed_size, cfg.num_heads, cfg.softmax_scale, cfg.attention_dropout)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_size, self.hidden_size),
            nn.GELU(),  # 激活函数可以选择 GELU 或 ReLU
            nn.Dropout(cfg.ffn_dropout),
            nn.Linear(self.hidden_size, cfg.embed_size)
        )

        # LayerNorms
        self.layernorm1 = nn.LayerNorm(cfg.embed_size)
        self.layernorm2 = nn.LayerNorm(cfg.embed_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, key_padding_mask=None, causal=False):
        # Cross-Attention Layer
        # Step 1: Cross-Attention + Residual Connection + LayerNorm
        attention_output = self.cross_attention(query, key, value, key_padding_mask=key_padding_mask, causal=causal)
        attention_output = self.layernorm1(query + attention_output)  # 残差连接 + LayerNorm

        # Feed-Forward Network (FFN)
        # Step 2: FFN + Residual Connection + LayerNorm
        ffn_output = self.ffn(attention_output)
        ffn_output = self.layernorm2(attention_output + ffn_output)  # 残差连接 + LayerNorm

        return ffn_output

class CrossAttensionDecoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        num_hidden_layers: int = 6
        layer: CrossAttentionLayer.Config = CrossAttentionLayer.Config()
    
    def __init__(self, cfg: Config):
        super().__init__()

        # 初始化多个 Cross-Attention 层
        self.layers = nn.ModuleList([
            CrossAttentionLayer(cfg.layer)
            for _ in range(cfg.num_hidden_layers)
        ])

    def forward(self, query, key, value, key_padding_mask=None, causal=False):
        # 多层 Cross-Attention
        for layer in self.layers:
            query = layer(query, key, value, key_padding_mask=key_padding_mask, causal=causal)

        return query    
    
def get_num_frequencies_nyquist(num_samples: int) -> int:
    """
    Get the number of octaves under the nyquist frequency.

    How this function works:

    Let N = num_samples.
    * In a span of (-1.0, +1.0), period = 2.0
    * So, sample frequency (f_s) =  N / 2.0
    * So, nyquist frequency = 1/2 * f_s = N / 4.0
    * So max_freq = max_{k} (2^{k} <= N / 4.0)
    * --> k <= log2(N / 4.0)
    * --> k <= log2(N) - 2

    Args:
        resolution: number of samples in a span of (-1, +1).
    """
    return np.floor(np.log2(num_samples) - 2).astype(np.int32)

# class PosEncodingCross(nn.Module):
#     """ skew-symmetric matrix """
#     def __init__(self, dim_in:int):
#         self.out_dim = 3

#     def forward(self,


class PosEncodingSine(nn.Module):
    """
    \\hat{x} = [x;sin(s*Wx)]
    """

    def __init__(self, dim_in: int, dim_out: int,
                 scale: float = 30.0):
        super().__init__()
        self.linear = nn.Sequential(*[
            nn.Linear(dim_in, dim_out),
            # nn.BatchNorm1d(dim_out)
        ])
        self.out_dim = dim_out + dim_in
        self.scale = scale

        with th.no_grad():
            m = self.linear[0]
            num_input = m.weight.size(-1)
            assert (num_input == dim_in)
            m.weight.uniform_(-1 / num_input,
                              1 / num_input)

    def forward(self, x: th.Tensor):
        s = x.shape
        x_f = x.reshape(-1, x.shape[-1])
        out = th.sin(self.scale * self.linear(x_f))
        out = out.reshape(*s[:-1], out.shape[-1])
        out = th.cat([x, out], dim=-1)
        return out


class PosEncodingLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.out_dim = dim_in
        self.linear = nn.Linear(dim_in, dim_out)
        self.out_dim = dim_out + dim_in

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.linear(x)
        out = th.cat([x, y], dim=-1)
        return out


class PosEncodingMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int,
                 dim_hidden: Tuple[int, ...],
                 act_cls: str = 'gelu'
                 ):
        super().__init__()
        self.out_dim = dim_in
        # self.linear = nn.Linear(dim_in, dim_out)
        self.mlp = MLP(merge_shapes(dim_in, dim_hidden, dim_out),
                       act_cls=get_activation_function(act_cls),
                       activate_output=False,
                       use_bn=False,
                       bias=True,
                       use_ln=False)
        self.out_dim = dim_out + dim_in

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.mlp(x)
        out = th.cat([x, y], dim=-1)
        return out


class PosEncodingNull(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.out_dim = dim_in

    def forward(self, x: th.Tensor):
        return x


class PosEncodingNeRF(nn.Module):
    """
    NeRF-style positional encoding.
    [Mildenhall et al. 2020].

    Computes the positional encoding for the
    normalized coordinate input in range (-1.0, +1.0).
    """

    def __init__(self,
                 dim_in: int,
                 num_frequencies: Optional[int] = None,
                 num_samples: Optional[int] = None,
                 flatten: bool = True,
                 cat_input: bool = True
                 ):
        """
        Args:
            dim_in: dimensionality of coordinate input.
            num_frequencies: Number of higher-frequency elements.
            num_samples: Fallback computation of number of frequencies.
            flatten: If true, flatten positional encoding to one channel.
        """
        super().__init__()

        self.dim_in = dim_in

        # Compute or retrieve the number of frequency elements.
        if num_frequencies is None:
            assert (num_samples is not None)
            num_frequencies = get_num_frequencies_nyquist(num_samples)
        self.num_frequencies = num_frequencies
        self.cat_input = cat_input

        # Compute the output dimensions, which is the
        # sin(x),cos(x) for each frequency + x
        # (2*F+1)*D
        self.out_dim = (
            dim_in * ((1 if cat_input else 0) + 2 * self.num_frequencies)
        )

        # Precompute the coefficient multipliers.
        self.register_buffer(
            'coefs', th.as_tensor(
                np.pi * (2 ** th.arange(self.num_frequencies)),
                dtype=th.float))
        self.flatten = flatten

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """
        Args:
            coords: (..., D)
        Returns:
            pos_enc: (..., (2*F+1)*D) if flatten else (..., (2*F+1), D)
        """
        octaves = coords[..., None, :] * self.coefs[:, None]
        s = th.sin(octaves)
        c = th.cos(octaves)
        if self.cat_input:
            out = th.concat([coords[..., None, :], s, c], dim=-2)
        else:
            out = th.concat([s, c], dim=-2)

        # Optionally flatten the output.
        if self.flatten:
            out = out.view(coords.shape[:-1] + (self.out_dim,))
        return out


def ortho_init(
        m: nn.Module,
        gain: float,
        target_cls: Tuple[nn.Module] = None):
    """
    Apply orthogonal initialization.
    """
    if m is None:
        return

    if target_cls is None:
        target_cls = (nn.Linear, nn.Conv2d)
    with th.no_grad():
        if isinstance(m, target_cls):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                m.bias.fill_(0.0)


def transfer(
        model: nn.Module,
        state_dict: Union[str, Dict[str, th.Tensor]],
        prefix_map: Optional[Dict[str, str]] = None,
        substrs: Optional[Iterable[str]] = None,
        strict: bool = False,
        freeze: bool = False,
        verbose: bool = False
):
    """
    transfer weights to model from state_dict, optionally
    rewriting weight named according to `prefix_map`.
    Furthermore, it is possible to filter the entries in
    `state_dict` according to membership in`substrs`.
    """

    if not isinstance(state_dict, Mapping):
        # assume `state_dict` is a path
        state_dict = th.load(state_dict)
    print('keys', list(state_dict.keys()))

    if prefix_map is None:
        prefix_map = {}

    def _replace_prefix(s: str, p: Dict[str, str]):
        for src, dst in p.items():
            if not s.startswith(src):
                continue
            s = dst + s[len(src):]
        return s

    renamed_state_dict = {
        _replace_prefix(k, prefix_map): v
        for (k, v) in state_dict.items()
    }

    if substrs is not None:
        # filter by substrs
        renamed_state_dict = {k: v for (k, v) in renamed_state_dict.items()
                              if any(s in k for s in substrs)}

    if verbose:
        source_keys = list(renamed_state_dict.keys())
        target_keys = [k for (k, v) in model.named_parameters()]
        update_keys = set(source_keys).intersection(target_keys)
        print(F'source = {source_keys}')
        print(F'target = {target_keys}')
        print(F'update = {update_keys}')
    out = model.load_state_dict(renamed_state_dict, strict)
    if freeze:
        for k, v in model.named_parameters():
            if k in out.missing_keys:
                continue
            if k in out.unexpected_keys:
                continue
            v.requires_grad_(False)
            # v.eval()
    return out


def replace_layer(model: nn.Module, old_layer_cls,
                  new_layer_fn):
    def get_all_parent_layers(net, type):
        layers = []
        for name, l in net.named_modules():
            if isinstance(l, type):
                tokens = name.strip().split('.')

                layer = net
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                layers.append([layer, tokens[-1]])
        return layers

    for parent_layer, last_token in get_all_parent_layers(
            model, old_layer_cls):
        source = getattr(parent_layer, last_token)
        target = new_layer_fn(source)
        setattr(parent_layer, last_token, target)


def test_mha():
    num_b: int = 4
    num_q: int = 7
    dim_c: int = 32
    dim_x: int = 3
    dim_k: int = 8
    dim_v: int = 16
    num_k: int = 9
    num_h: int = 2
    mha = MultiHeadCrossAttention(dim_c, dim_x,
                                  dim_k, dim_v,
                                  dim_v, num_k=num_k,
                                  num_h=num_h)

    context = th.randn((num_b, dim_c))
    inputs = th.randn((num_b, num_q, dim_x))
    outputs = mha(context, inputs)
    print('out', outputs.shape)


def test_map_struct_with_dataclass():
    @dataclass
    class A:
        a: int = 1
        b: int = 2

        def __post_init__(self):
            self.b = self.a % 2

    src = A(a=3, b=4)
    print(src)  # 3, 3

    dst = map_struct(src, lambda src, dst: 4,
                     dst=None)
    print(src)
    print(dst)

    dst = A(a=2, b=6)
    dst2 = map_struct(src, lambda src, dst: src + 5,
                      dst=dst)
    print(src)
    print(dst)
    print(dst2)


def test_map_struct():
    # Copy
    if True:
        src = {'a': 1, 'b': 2}
        dst = map_struct(
            src=src,
            op=(lambda src, dst: src),
            dst=None,
        )
        print(dst)

        def validate(src, dst):
            assert (src == dst)
            return src
        map_struct(src=src, dst=dst, op=validate)

    # All elements are mapped to None
    if True:
        dst = map_struct(
            src={'a': 1, 'b': 2},
            op=(lambda src, dst: None),
            dst=None,
        )
        print(dst)

        def validate(src, dst):
            assert (src is None)
            return src
        map_struct(src=dst, op=validate)


def print_memory(dev: int):
    t = th.cuda.get_device_properties(dev).total_memory
    r = th.cuda.memory_reserved(dev)
    a = th.cuda.memory_allocated(dev)
    f = r - a  # free inside reserved
    print(t, r, a, f)


def test_attention():
    # num_b: int = 512
    num_b: int = 512
    num_q: int = 1
    dim_k: int = 256
    dim_v: int = 256
    num_k: int = 144
    device: str = 'cuda:1'
    th.cuda.set_device(device)

    # print_memory(1)
    k = th.randn((num_b, num_k, 1, dim_k), dtype=th.float32,
                 device=device)
    v = th.randn((num_b, num_k, 1, dim_v), dtype=th.float32,
                 device=device)
    q = th.randn((num_b, num_q, 1, dim_k), dtype=th.float32,
                 device=device)

    print('attention')
    print_memory(1)
    o1 = attention(
        q.squeeze(dim=-2),
        k.squeeze(dim=-2),
        v.squeeze(dim=-2))
    print_memory(1)

    # print('e_attention')
    # print_memory(1)
    # o2 = e_attention(q, k, v,
    #                  query_chunk_size=1,
    #                  key_chunk_size=12  # sqrt(144)
    #                  ).squeeze(dim=-2)
    # print_memory(1)

    # if True:
    #     print(o1.shape)
    #     print(o2.shape)
    #     print(o1 - o2)
    #     print((o1 - o2).max())
    #     print((o1 - o2).min())
    #     print((o1 - o2).mean())
    #     print((o1 - o2).std())


from torch.utils._pytree import (
    tree_map,
    tree_flatten,
    tree_unflatten)


def tree_map_n(f, *args):
    # y = tree_map(lambda x: (x + 1), {'a': th.zeros(3)})
    leafs, specs = zip(*[tree_flatten(x) for x in args])
    leafs = zip(*leafs)
    outputs = [f(*l) for l in leafs]
    return tree_unflatten(outputs, next(iter(specs)))


def comp_map_to_treemap():
    y = tree_map(lambda x: (x + 1), {'a': th.zeros(3)})
    # y=tree_map(lambda x,y : (x+y), {'a':th.zeros(3)})
    print(y)

    x = {'a': 2, 'b': 3}
    y = {'a': th.zeros(2), 'b': th.zeros(4)}

    xr, spec = tree_flatten(x)
    yr, spec = tree_flatten(y)
    print(spec)
    xy = tree_unflatten(list(zip(xr, yr)), spec)
    print(xy)

    def _add(x, y):
        print('x', x)
        print('y', y)
        return y + x
    out = tree_map_n(_add, x, y)
    print(out)


def main():
    # test_attention()
    comp_map_to_treemap()


if __name__ == '__main__':
    main()
