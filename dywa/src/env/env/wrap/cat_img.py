#!/usr/bin/env python3

from typing import Dict, Iterable
from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from gym import spaces
import torch as th
import numpy as np
import einops
from util.torch_util import dcn


class MaskFromSegImg(ObservationWrapper):
    """
    'seg' -> 'mask'
    by filtering with label
    """

    def __init__(self, label: int, env: EnvIface):
        super().__init__(env, self._wrap_obs)
        self.label = label
        assert isinstance(env.observation_space, spaces.Dict)
        obs_space = dict(env.observation_space.spaces)

        # NOTE: `pop` here is necessary, otherwise
        # `seg` remains in obs_space.
        seg_space = obs_space.pop('label')
        mask_space = spaces.Box(0, 1, shape=seg_space.shape,
                                dtype=bool)
        obs_space['mask'] = mask_space
        self._obs_space = spaces.Dict(obs_space)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        obs = dict(obs)
        seg = obs['label']
        mask = (th.round(seg) == self.label)

        # import cv2
        # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        # print('mask', mask.shape)
        # cv2.imshow('mask', dcn(mask[0].squeeze(0)*255).astype(np.uint8))
        # cv2.waitKey(0)

        obs['mask'] = mask
        return obs


class CatImg(ObservationWrapper):
    """
    Concatenate images.
    """

    def __init__(self, keys: Iterable[str], env: EnvIface):
        super().__init__(env, self._wrap_obs)
        assert (isinstance(env.observation_space, spaces.Dict))
        self.keys = list(keys)
        obs_space = dict(env.observation_space.spaces)

        los = []
        his = []
        for key in self.keys:
            img_space = obs_space.pop(key)
            assert (isinstance(img_space, spaces.Box))
            los.append(img_space.low)
            his.append(img_space.high)
        lo = np.concatenate(los, axis=-3)
        hi = np.concatenate(his, axis=-3)
        new_img_space = spaces.Box(lo, hi)
        obs_space['img'] = new_img_space
        self._obs_space = spaces.Dict(obs_space)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        obs = dict(obs)
        img = th.cat([obs.pop(k) for k in self.keys], dim=-3)
        obs['img'] = img
        return obs
