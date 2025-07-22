#!/usr/bin/env python3

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from gym import spaces
import torch as th
import numpy as np
import einops


class FlattenImgGoal(ObservationWrapper):
    """
    Wrapper class to pass heterogenous observations
    (such as images and goals) without creating a dict.
    """

    def __init__(self, env: EnvIface, **kwds):
        super().__init__(env, self._wrap_obs)
        assert (isinstance(env.observation_space, spaces.Dict))
        img_dims = int(np.product(env.observation_space['img'].shape))
        self._obs_space = spaces.Box(-np.inf, +np.inf, (img_dims + 3,))
        # self._count = 0
        self.use_mask_encoder: bool = kwds.pop('use_mask_encoder', False)
        self.object_label: int = kwds.pop('object_label', 2)
        self.use_cube: bool = kwds.pop('use_cube', False)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        # We're getting depth image.
        # might as well convert to something reasonable?

        with th.no_grad():
            if self.use_mask_encoder == True:
                # Explict rounding due to floating point precision
                object_mask = th.round(obs['seg']) == self.object_label
                # Add temporary channel dimension
                img = th.unsqueeze(obs['img'], dim=-3)
                object_mask = th.unsqueeze(object_mask, dim=-3) 
                img = th.cat((img, object_mask), dim=-3)
                img_flat = img.reshape(img.shape[0], -1) 
            else:
                img_flat = obs['img'].reshape(obs['img'].shape[0], -1)
            goal = obs['raw'][..., :3]

            if self.use_cube: # Add cube state
                cube_state = obs['raw'][..., 22:29]
                # Hacky slicing for extracting cube state : must not be used
                Raise(ValueError)
                out = th.cat((img_flat, goal, cube_state), dim=-1)
            else:
                out = th.cat((img_flat, goal), dim=-1)
        return out
