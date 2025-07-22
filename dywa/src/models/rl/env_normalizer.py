#!/usr/bin/env python3

from collections import Mapping
from typing import (
    Optional, Dict, Tuple, Union, List, Callable, Any)

from dataclasses import dataclass, replace
from util.config import ConfigBase

from models.rl.running_mean_std import (
    RunningMeanStd, ConstantMeanStd,
    RollingMeanStd)
from models.common import merge_shapes, map_struct
from util.torch_util import masked_mean

import torch as th
import torch.nn as nn
import numpy as np
import logging

from icecream import ic

T = th.Tensor
S = Union[int, Tuple[int, ...]]


class EnvNormalizer(nn.Module):

    @dataclass
    class Config(ConfigBase):
        normalize_obs: bool = True
        normalize_rew: bool = True
        center_rew: bool = False
        obs_eps: float = 1e-6
        rew_eps: float = 1e-6
        gamma: float = 0.99

        constlist: Tuple[str, ...] = ()
        stats: Optional[Dict[str, Optional[List[List[float]]]]] = None

        rms_type: str = 'running'
        rolling_alpha: float = 0.001
        fix_return: bool = False

    def __init__(self,
                 cfg: Config,
                 num_env: int = -1,
                 obs_space: Union[Dict[str, S], S] = -1,
                 device: str = 'cuda:0'):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.num_env = num_env
        self.obs_space = obs_space

        # self.rew_rms = RunningMeanStd(device=self.device)
        self.rew_rms = None
        if self.cfg.normalize_rew:
            if cfg.rms_type == 'running':
                self.ret_rms = RunningMeanStd(device=self.device)
            elif cfg.rms_type == 'rolling':
                self.ret_rms = RollingMeanStd(device=self.device,
                                              alpha=cfg.rolling_alpha)
            else:
                raise ValueError(F'Unknown rms_type={cfg.rms_type}')
        else:
            self.ret_rms = None

        if cfg.normalize_obs:
            if isinstance(self.obs_space, Mapping):
                stats = cfg.stats
                if stats is None:
                    stats = {}

                obs_rmss = {}
                for k, v in self.obs_space.items():
                    if k in cfg.constlist:
                        mean, std = stats.get(k, (0.0, 1.0))
                        obs_rmss[k] = ConstantMeanStd(self.device,
                                                      merge_shapes(v),
                                                      cfg.obs_eps,
                                                      mean=mean,
                                                      var=np.square(std))
                    else:
                        obs_rmss[k] = RunningMeanStd(self.device,
                                                     merge_shapes(v),
                                                     cfg.obs_eps)
                self.obs_rms = nn.ModuleDict(obs_rmss)
            else:
                # Box
                self.obs_rms = RunningMeanStd(
                    device=self.device, shape=merge_shapes(
                        self.obs_space))
        else:
            self.obs_rms = None

        # Some stats
        self.returns = th.zeros(
            self.num_env,
            dtype=th.float,
            device=self.device)
        self.init = th.zeros(
            self.num_env,
            dtype=th.bool,
            device=self.device)
        # episode lengths
        # self.eplen = th.zeros(
        #     self.num_env,
        #     dtype=th.int32,
        #     device=self.device)

    def normalize_obs(self, obs: th.Tensor,
                      in_place: bool = False) -> th.Tensor:
        cfg = self.cfg

        if isinstance(obs, Mapping):
            obs = dict(obs)
            for k in obs:
                # print(k, self.obs_rms[k].mean, self.obs_rms[k].var,
                #       obs[k].shape)
                scale = (
                    th.reciprocal(th.sqrt(self.obs_rms[k].var) +
                                  cfg.obs_eps))
                try:
                    obs[k] = (obs[k] - self.obs_rms[k].mean) * scale
                except RuntimeError:
                    logging.error(F'Obs normalization failed for {k}')
                    raise
        else:
            if not in_place:
                obs = obs.detach().clone()
            # obs = obs.sub_(self.obs_rms.mean).mul_(
            #     th.reciprocal(th.sqrt(self.obs_rms.var) + cfg.obs_eps)
            # )
            scale = (th.reciprocal(th.sqrt(self.obs_rms.var) + cfg.obs_eps))
            obs = (obs - self.obs_rms.mean) * scale
        return obs

    def unnormalize_obs(self,
                        n_obs: th.Tensor,
                        v: Optional[th.Tensor] = None,
                        c: Optional[th.Tensor] = None,
                        eps: Optional[float] = None) -> th.Tensor:
        if not self.cfg.normalize_obs:
            return n_obs

        if eps is None:
            eps = self.cfg.obs_eps

        def _unnormalize(src, dst):
            if v is None:
                vv = dst.var
            else:
                vv = v
            if c is None:
                cc = dst.mean
            else:
                cc = c
            return src * th.sqrt(vv + eps) + cc

        return map_struct(
            n_obs,
            _unnormalize,
            self.obs_rms,
            base_cls=th.Tensor)

    def normalize_rew(self, rew: th.Tensor,
                      in_place: bool = False) -> th.Tensor:
        """ R' = R / sqrt(Var(G)) """
        cfg = self.cfg
        if not in_place:
            rew = rew.detach().clone()

        if cfg.center_rew:
            raise ValueError('`center_rew` no longer supported...!!')
            rew.sub_(self.rew_rms.mean)

        if cfg.normalize_rew:
            # sigma^2 = var(reward) + gamma? * var(
            # denom = th.sqrt(self.ret_rms.var)
            rew.mul_(th.reciprocal(th.sqrt(self.ret_rms.var) + cfg.rew_eps))
        return rew

    # def update_obs(self, obs: th.Tensor, mask: th.Tensor):
    #     # FIXME: rank-1 obs. assumption (dangerous!)
    #     return self.obs_rms.update(
    #         obs.reshape(-1, obs.shape[-1]),
    #         mask.reshape(-1, 1))

    # def update(self, batch):
    #     cfg = self.cfg

    #     v0 = self.obs_rms.var.detach().clone()
    #     c0 = self.obs_rms.mean.detach().clone()
    #     rv0 = self.ret_rms.var.detach().clone()
    #     rc0 = self.ret_rms.mean.detach().clone()

    #     # for mini_batch in batches:
    #     # FIXME: very very bad
    #     # n_obs, action, logp0, mu0, ls0, adv, n_ret, mask = mini_batch

    #     # FIXME: awkward normalize-unnormalize loop
    #     obs = self.unnormalize_obs(batch['obsn'], v=v0, c=c0)
    #     self.update_obs(obs, mask)

    #     # ret = self.unnormalize_ret(n_obs, v=v0, c=c0)
    #     ret = n_ret * (th.sqrt(rv0) + cfg.rew_eps)
    #     # self.update_(obs, mask)
    #     self.ret_rms.update(ret, mask)

    def reset(self, env):
        cfg = self.cfg
        obs = env.reset()
        if cfg.normalize_obs:
            obs = self.normalize_obs(obs)
        return obs

    def reset_indexed(self, env, indices: th.Tensor):
        return env.reset_indexed(indices)

    def step(self,
             env, action: th.Tensor
             ) -> Tuple[T, T, T, Dict[str, th.Tensor]]:
        cfg = self.cfg
        obs, rew, done, info = env.step(action)

        if self.training:
            with th.no_grad():
                if cfg.normalize_rew:
                    if cfg.fix_return:
                        step = env.buffers['step']
                        # G_t += r_t * gamma ** t
                        self.returns.add_(rew * th.pow(cfg.gamma, step))
                    else:
                        # G_t = gamma * G_t + r_t ??
                        self.returns.mul_(cfg.gamma).add_(rew)
                    mask = (~self.init)
                    batch_stats = self.ret_rms.update(self.returns, mask)
                    if batch_stats is not None:
                        (vv, mm, cc) = batch_stats
                        info['batch_var'] = vv.detach_()
                        info['batch_mean'] = mm.detach_()
                        info['batch_count'] = cc.detach_()
                    self.returns *= ~done

                # NOTE: `obs` is always valid,
                # pretty much.
                # NOTE: for whatever reason, below code
                # results in memory error...??
                # obs_mask = mask[..., None].expand(obs.shape)
                # self.obs_rms.update(obs, obs_mask)
                if cfg.normalize_obs:
                    if isinstance(obs, Mapping):
                        for k in obs:
                            self.obs_rms[k].update(obs[k])
                    else:
                        self.obs_rms.update(obs)

        assert ('raw_obs' not in info)
        assert ('raw_rew' not in info)
        info['raw_obs'] = obs
        info['raw_rew'] = rew

        if cfg.normalize_obs:
            n_obs = self.normalize_obs(obs)
        else:
            n_obs = obs

        if cfg.normalize_rew:
            n_rew = self.normalize_rew(rew)
        else:
            n_rew = rew
        self.init[...] = done
        return (n_obs, n_rew, done, info)
