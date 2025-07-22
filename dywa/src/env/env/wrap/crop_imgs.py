#!/usr/bin/env python3

from typing import Optional, Dict
import numpy as np
import torch as th
from gym import spaces

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper

from env.env.help.crop_bbox import CropBbox
from env.robot.fe_gripper import FEGripper
from env.scene.tabletop_with_object_scene import TableTopWithObjectScene

KEYS: str = ('hand_image', 'scene_image', 'object_image')


class CropImgs(ObservationWrapper):
    """
    Concatenate images.
    """

    def __init__(self,
                 env: EnvIface,
                 img_key: str,
                 out_keys: Optional[Dict[str, str]] = {k: k for k in KEYS},
                 cat: bool = True
                 ):

        super().__init__(env, self._wrap_obs)
        assert (isinstance(env.observation_space, spaces.Dict))
        assert (isinstance(env.robot, FEGripper))
        assert (isinstance(env.scene, TableTopWithObjectScene))

        self.img_key = img_key
        self.out_keys = out_keys
        self.cat = cat
        self.crop = CropBbox(overwrite=False)
        self.crop.reset(env)

        obs_space = dict(env.observation_space.spaces)
        img_space = obs_space.pop(img_key)
        assert (isinstance(img_space, spaces.Box))

        if cat:
            # fuse all outputs into one image.
            assert (len(out_keys) == 1)
            out_key = list(out_keys.values())[0]
            num_crop: int = len(out_keys)
            lo = np.concatenate([img_space.low] * num_crop, axis=-3)
            hi = np.concatenate([img_space.high] * num_crop, axis=-3)
            new_img_space = spaces.Box(lo, hi)
            obs_space[out_key] = new_img_space
        else:
            # independent output (but identical dims)
            # for each image
            for out_key in out_keys:
                obs_space[out_key] = img_space

        self._obs_space = spaces.Dict(obs_space)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        obs = dict(obs)
        img = obs.pop(self.img_key)

        # [1] Query object and robot properties
        robot_center = self.tensors['body'][self.robot.hand_ids, ..., :3]
        robot_radius = self.robot.robot_radii
        object_center = self.tensors['root'][
            self.scene.cur_ids.long(), ..., :3]
        object_radius = self.scene.cur_radii

        # [2] Crop.
        robot_crops = self.crop(img,
                                robot_center,
                                robot_radius)
        robot_image = robot_crops['crop']
        object_crops = self.crop(img, object_center, object_radius)
        object_image = object_crops['crop']

        if self.cat:
            out_key = list(self.out_keys.values())[0]
            obs[out_key] = th.cat([img, robot_image, object_image], dim=-3)
        else:
            obs[self.out_keys['scene_image']] = img
            obs[self.out_keys['hand_image']] = robot_image
            obs[self.out_keys['object_image']] = object_image
        return obs
