#!/usr/bin/env python3

from typing import Optional, Dict, Tuple, List
from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv
from pathlib import Path
from dataclasses import dataclass
from util.config import ConfigBase
from torch.utils.tensorboard import SummaryWriter
from models.rl.env_normalizer import EnvNormalizer
from util.path import ensure_directory
from train.ckpt import save_ckpt, load_ckpt
from util.path import RunPath

import torch as th

from gym import spaces
from icecream import ic
import nvtx


class NormalizeEnv(WrapperEnv):
    """
    I guess this class ended up being similar to
    SB3 VecNormalize.

    > normalize observations based on rolling stats.
    > normalize (and clip?) actions, in range (-1.0, +1.0)
    > normalize rewards.
    """

    @dataclass
    class Config(ConfigBase):
        norm: EnvNormalizer.Config = EnvNormalizer.Config()
        normalize_act: bool = True
        clip_action: bool = True
        stat_save_period: int = 1024
        log_period: int = 1024
        obs_shape: Optional[Dict[str, List[int]]] = None

    def __init__(self,
                 cfg: Config,
                 env: EnvIface,
                 path: Optional[RunPath] = None):
        super().__init__(env)

        self.cfg = cfg
        self.path = path

        # Try to automatically figure out normalization options
        # from `env`.
        if isinstance(env.observation_space, spaces.Box):
            obs_shape = env.observation_space.shape
        else:
            obs_shape = {
                k: v.shape for (
                    k, v) in env.observation_space.items()}
            if cfg.obs_shape is not None:
                obs_shape.update(cfg.obs_shape)
        self.normalizer = EnvNormalizer(cfg.norm,
                                        self.num_env,
                                        obs_shape, self.device)
        ic(self.normalizer)
        for k, v in self.normalizer.named_buffers():
            print(k)
            print(v.shape)

        # NOTE: not necessarily always correct
        if cfg.norm.normalize_obs:
            # print(self.env.observation_space)
            # assert (isinstance(self.env.observation_space, spaces.Box))
            if isinstance(env.observation_space, spaces.Box):
                self._obs_space = spaces.Box(-1.0, +1.0, obs_shape)
            else:
                self._obs_space = spaces.Dict({k: spaces.Box(-1.0, +1.0, v) for
                                               (k, v) in obs_shape.items()})
        else:
            self._obs_space = env.observation_space

        self.__run_step: int = 0

        if cfg.normalize_act:
            if isinstance(self.env.action_space, spaces.Box) and (
                    self.env.action_space.is_bounded()):
                # Continuous action-space + bounded
                self._act_space = spaces.Box(-1.0, +1.0,
                                             self.env.action_space.shape)
                lo = self.env.action_space.low
                hi = self.env.action_space.high
                self.lo = th.as_tensor(lo,
                                       dtype=th.float, device=self.env.device)
                self.hi = th.as_tensor(hi,
                                       dtype=th.float, device=self.env.device)
                with th.no_grad():
                    self.action_center = 0.5 * (self.lo + self.hi)
                    self.action_scale = 0.5 * (self.hi - self.lo)
            else:
                # discrete action-space = no normalization
                self._act_space = self.env.action_space
                self.action_center = None
                self.action_scale = None
        else:
            self._act_space = self.env.action_space
            lo = self.env.action_space.low
            hi = self.env.action_space.high
            self.lo = th.as_tensor(lo,
                                   dtype=th.float, device=self.env.device)
            self.hi = th.as_tensor(hi,
                                   dtype=th.float, device=self.env.device)

    def _wrap_act(self, action: th.Tensor,
                  in_place: bool = False):
        cfg = self.cfg
        if action is None:
            return action

        if not in_place:
            action = action.clone()

        # Wrap action (only for continuous actions).
        if isinstance(self._act_space,
                      spaces.Box) and self._act_space.is_bounded():
            # [1] unnormalize action
            if cfg.normalize_act:
                action.mul_(self.action_scale).add_(self.action_center)
            # [2] clip action
            if cfg.clip_action:
                action.clamp_(min=self.lo, max=self.hi)
        return action

    def update_action_scale(self):
        if not self.cfg.normalize_act:
            raise ValueError(
                'no action scale update if not normalizing actions')
        if isinstance(self.env.action_space, spaces.Box) and (
                self.env.action_space.is_bounded()):
            lo = self.env.action_space.low
            hi = self.env.action_space.high
            self.lo = th.as_tensor(lo,
                                   dtype=th.float, device=self.env.device)
            self.hi = th.as_tensor(hi,
                                   dtype=th.float, device=self.env.device)
            with th.no_grad():
                self.action_center = 0.5 * (self.lo + self.hi)
                self.action_scale = 0.5 * (self.hi - self.lo)
        else:
            raise ValueError(
                'no action scale update for discrete action space')

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def setup(self):
        return self.env.setup()

    def reset_indexed(self, *args, **kwds) -> th.Tensor:
        return self.normalizer.reset_indexed(self.env,
                                             *args, **kwds)

    def reset(self) -> th.Tensor:
        return self.normalizer.reset(self.env)

    def _add_scalar(self, k, v, step: Optional[int] = None):
        if step is None:
            step = self.__run_step
        if self.writer is not None:
            return self.writer.add_scalar(k, v, global_step=step)
        else:
            print(F'{k}={v}')

    def _log(self):
        cfg = self.cfg
        if cfg.norm.normalize_rew:
            self._add_scalar('log/ret_rms/mean',
                             self.normalizer.ret_rms.mean)
            self._add_scalar('log/ret_rms/var',
                             self.normalizer.ret_rms.var)

        # `obs` stats are (usually) not scalars; not sure how to add this.
        # if cfg.norm.normalize_obs:
        #     self._add_scalar('log/obs_rms/mean',
        #             self.normalizer.obs_rms.mean)
        #     self._add_scalar('log/obs_rms/var',
        #             self.normalizer.obs_rms.var)

    def compute_reward_from_trajectory(
            self, traj: th.Tensor, extra: th.Tensor,
            goal: th.Tensor, rewd: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Compute reward from trajectory.
        `traj` should be buffered by one?
        (Assumes TxNxD layout)
        """
        rewd = self.task.compute_reward_from_trajectory(
            traj, extra, goal, rewd)
        cfg = self.cfg
        if cfg.norm.normalize_rew:
            self.normalizer.normalize_rew(rewd, in_place=True)
        return rewd

    @nvtx.annotate("NormalizeEnv.step")
    def step(self, action: th.Tensor):
        cfg = self.cfg

        w_action = self._wrap_act(action)

        # [1] without obs,rew normalization
        # return self.env.step(w_action)
        # [2] _with__ obs,rew normalization
        out = self.normalizer.step(self.env, w_action)

        # DEBUGGING
        if False:
            _, _, _, info = out
            if 'batch_count' in info:
                self._add_scalar('log/ret_rms_batch_count',
                                 info['batch_count'])
                self._add_scalar('log/ret_rms_batch_mean',
                                 info['batch_mean'])
                self._add_scalar('log/ret_rms_batch_var',
                                 info['batch_var'])

        # Periodically save normalization stats.
        # NOTE: only enabled if `path` is not None
        if self.path is not None:
            if self.__run_step % cfg.stat_save_period == 0:
                ckpt_file = self.path.stat / F'env-{self.__run_step:05d}.ckpt'
                self.save(ckpt_file)

            if (self.__run_step % cfg.log_period == 0):
                self._log()
            self.__run_step += 1

        return out

    def state_dict(self):
        return self.normalizer.state_dict()

    def load(self, path: str, strict: bool = False):
        if Path(path).is_dir():
            # `path` is a directory
            path = str(Path(path) / 'norm.ckpt')
        ic(F'load from {path}')
        load_ckpt(dict(normalizer=self.normalizer),
                  ckpt_file=path,
                  strict=strict)

    def save(self, path: str):
        if Path(path).is_dir():
            # `path` is a directory
            path = str(Path(path) / 'norm.ckpt')
        ensure_directory(Path(path).parent)
        save_ckpt(dict(normalizer=self.normalizer),
                  ckpt_file=path)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)
