#!/usr/bin/env python3

import numpy as np
import torch as th
import einops

import mvp
import nvtx
from gym import spaces

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper


class MvpWrapper(ObservationWrapper):
    def __init__(self, env: EnvIface, use_amp: bool = True):
        super().__init__(env, self._wrap_obs)
        self.mvp = (
            mvp.load("vits-mae-hoi").to(env.device)
        )
        self.mvp.requires_grad_(False)
        self.mvp.eval()
        self.mvp = th.jit.script(self.mvp)
        # FIXME: hardcoded 384 latent dims
        self._obs_space = spaces.Box(-np.inf, +np.inf, (384 + 3,))
        self.use_amp = use_amp

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        # We're getting depth image.
        # might as well convert to something reasonable?
        with th.no_grad():
            with nvtx.annotate("mvp"):
                with th.cuda.amp.autocast(enabled=self.use_amp):
                    with nvtx.annotate("rep3"):
                        d3 = einops.repeat(obs['img'],
                                           'n h w -> n c h w', c=3)
                    with nvtx.annotate("pred"):
                        goal = obs['raw'][..., :3]
                        out = th.cat((self.mvp(d3), goal), dim=-1)
        return out

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        # hack
        info['img'] = obs['img']
        w_obs = self.obs_wrapper(obs)
        return (w_obs, rew, done, info)
