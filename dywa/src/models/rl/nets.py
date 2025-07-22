#!/usr/bin/env python3


from typing import (Optional, Dict, Tuple, Union)
from dataclasses import dataclass
from util.config import ConfigBase

import torch as th
import torch.nn as nn
import torch.nn.functional as F


from models.common import (
    merge_shapes,
    MLP,
    ortho_init,
    get_activation_function)


S = Union[int, Tuple[int, ...]]
T = th.Tensor


class PiNet(nn.Module):
    """ Gaussian Policy """
    @dataclass
    class Config(ConfigBase):
        dim_feat: int = -1
        dim_hidden: Tuple[int, ...] = ()
        dim_act: int = -1
        log_std_init: float = 0.0
        act_cls: str = 'tanh'
        use_bn: bool = False
        use_ln: bool = False
        ortho_init: bool = False
        use_sde: bool = False
        bias: bool = True
        pre_ln_bias: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = merge_shapes(cfg.dim_feat, cfg.dim_hidden, cfg.dim_act)
        print(
            F'PiNet got dims={dims} from '
            F'{cfg.dim_feat, cfg.dim_hidden, cfg.dim_act}'
        )
        act_cls = get_activation_function(cfg.act_cls)
        self.mu = MLP(dims, act_cls, False, cfg.use_bn,
                      use_ln=cfg.use_ln,
                      bias=cfg.bias,
                      pre_ln_bias=cfg.pre_ln_bias)
        ls_dim = merge_shapes(
            cfg.dim_feat,
            cfg.dim_act) if cfg.use_sde else merge_shapes(
            cfg.dim_act)
        self.register_parameter('log_std',
                                nn.Parameter(th.full(ls_dim,
                                                     cfg.log_std_init),
                                             requires_grad=True))
        if cfg.ortho_init:
            ortho_init(self.mu, 0.01)

    def forward(self, x: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:
        mu = self.mu(x)
        ls = self.log_std
        return (mu, ls)


class CategoricalPiNet(nn.Module):
    """ Categorical Policy """
    @dataclass
    class Config(ConfigBase):
        dim_feat: int = -1
        dim_hidden: S = ()
        dim_act: int = -1
        act_cls: str = 'tanh'
        use_bn: bool = False
        use_ln: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        dims = merge_shapes(cfg.dim_feat, cfg.dim_hidden, cfg.dim_act)
        act_cls = get_activation_function(cfg.act_cls)
        self.logits = MLP(dims, act_cls, False, cfg.use_bn,
                          use_ln=cfg.use_ln)

    def forward(self, x: th.Tensor,
                aux: Optional[Dict[str, th.Tensor]] = None
                ) -> Tuple[th.Tensor, th.Tensor]:
        logits = self.logits(x)
        return logits


class VNet(nn.Module):

    @dataclass
    class Config(ConfigBase):
        dim_feat: int = -1
        dim_hidden: Tuple[int, ...] = ()
        act_cls: str = 'tanh'
        use_bn: bool = False
        bias: bool = True
        use_ln: bool = False
        ortho_init: bool = False
        pre_ln_bias: bool = True

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        dims = merge_shapes(cfg.dim_feat,
                            cfg.dim_hidden,
                            1)
        act_cls = get_activation_function(cfg.act_cls)
        self.value = MLP(dims, act_cls, False, cfg.use_bn,
                         cfg.bias,
                         use_ln=cfg.use_ln,
                         pre_ln_bias=cfg.pre_ln_bias)
        if cfg.ortho_init:
            ortho_init(self.value, 1.0)

    def forward(self, x: th.Tensor):
        v = self.value(x).squeeze(dim=-1)
        return v


class MLPTransitionNet(nn.Module):

    @dataclass
    class Config(ConfigBase):
        dim_s: int = -1
        dim_a: int = -1
        dim_h: S = ()
        act_cls: str = 'tanh'
        use_bn: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dims = merge_shapes(
            cfg.dim_s + cfg.dim_a,
            cfg.dim_h,
            cfg.dim_s)
        act_cls = get_activation_function(cfg.act_cls)
        self.mlp = MLP(dims, act_cls, True, cfg.use_bn)

    def forward(self, s0: th.Tensor, a: th.Tensor):
        if not th.is_floating_point(a):
            a = F.one_hot(a.long(), self.cfg.dim_a)
        sa = th.cat([s0, a], dim=-1)
        s1 = self.mlp(sa)
        return s1


class MLPFwdBwdDynLossNet(nn.Module):

    @dataclass
    class Config(ConfigBase):
        dim_state: int = -1
        dim_act: int = -1
        dim_hidden: S = ()
        act_cls: str = 'tanh'
        use_bn: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        s = merge_shapes(cfg.dim_state)[0]
        a = merge_shapes(cfg.dim_act)[0]
        fwd_dims = merge_shapes(s + a,
                                cfg.dim_hidden,
                                s)
        bwd_dims = merge_shapes(s + s,
                                cfg.dim_hidden,
                                a)
        act_cls = get_activation_function(cfg.act_cls)
        self.fwd_mlp = MLP(fwd_dims, act_cls, False,
                           cfg.use_bn)
        self.bwd_mlp = MLP(bwd_dims, act_cls, False,
                           cfg.use_bn)
        self.fwd_loss = nn.MSELoss()
        self.bwd_loss = nn.MSELoss()

    def hook_aux_loss(self, on_loss):
        self.register_forward_hook(on_loss)

    def forward(self, states: th.Tensor, actions: th.Tensor):
        s0 = states[:, :-1]
        s1 = states[:, 1:]
        a = actions[:, :-1]
        s1_pred = self.fwd_mlp(th.cat([s0, a], dim=-1))
        a_pred = self.bwd_mlp(th.cat([s0, s1], dim=-1))
        fwd_loss = self.fwd_loss(s1_pred, s1.detach())
        bwd_loss = self.bwd_loss(a_pred, a.detach())
        loss = (fwd_loss + bwd_loss)
        return loss
