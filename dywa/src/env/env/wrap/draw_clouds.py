#!/usr/bin/env python3

from typing import Iterable
from isaacgym import gymapi, gymtorch

import torch as th
import numpy as np

from env.env.wrap.base import WrapperEnv
from env.util import (draw_one_inertia_box, from_vec3, from_mat3,
                          draw_cloud_with_sphere,
                          draw_cloud_with_ray)
from util.torch_util import dcn
from env.env.wrap.normalize_env import NormalizeEnv
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper


class DrawClouds(WrapperEnv):
    def __init__(self, env, check_viewer: bool = True,
                 cloud_key: str = 'cloud',
                 stride: int = 8,
                 radius: float = 0.005,
                 style: str = 'sphere'):
        super().__init__(env)
        self.__stride = stride
        self.__radius = radius
        self.__key = cloud_key
        self.__style = style

    def __draw(self, obs):
        if (self.check_viewer) and (self.viewer is None):
            return

        # clouds = {'init': dcn(obs['object_state']), 'goal': dcn(obs['goal'])}
        nenv = self.unwrap(target=NormalizeEnv)
        if isinstance(nenv, NormalizeEnv):
            nobs = nenv.normalizer.unnormalize_obs(obs)
        else:
            nobs = obs
        clouds = {'init': dcn(nobs[self.__key])}
        if 'goal' in nobs and nobs['goal'].shape[-1] == 3:
            clouds['goal'] = dcn(nobs['goal'])
        if 'partial' in nobs:
            clouds['partial'] = dcn(nobs['partial_cloud'])

        colors = {'init': (1.0, 0.0, 0.0),
                  'goal': (0.0, 0.0, 1.0),
                  'partial': (0.0, 1.0, 0.0),
                  }

        if self.__style == 'sphere':
            for k, v in clouds.items():
                for c, e in zip(v, self.envs):
                    draw_cloud_with_sphere(self.gym, self.viewer,
                                           c[::self.__stride], e,
                                           radius=self.__radius,
                                           color=np.asarray(colors[k])
                                           )
        elif self.__style == 'ray':
            cam_env = self.unwrap(target=NvdrCameraWrapper)
            for k, v in clouds.items():
                for c, e in zip(v, self.envs):
                    draw_cloud_with_ray(self.gym, self.viewer,
                                        c[::self.__stride], e,
                                        radius=self.__radius,
                                        color=np.asarray(colors[k]),
                                        eye=cam_env.cfg.eye)
        else:
            raise ValueError(F'Unknown style = {self.__style}')

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        obs, rew, done, info = out
        self.__draw(obs)
        return out
