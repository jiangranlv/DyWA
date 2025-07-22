#!/usr/bin/env python3

from isaacgym import gymapi, gymtorch
from env.env.wrap.base import WrapperEnv
# from env.util import draw_inertia_box
from env.util import (draw_one_inertia_box, from_vec3, from_mat3)
from util.torch_util import dcn

import numpy as np
import cv2


class AddMassToInfo(WrapperEnv):
    """
    Add object mass to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.__masses = None

    def __get_masses(self):
        gym = self.gym
        masses = []
        actor_handles = dcn(self.scene.cur_handles)
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            mass = prop[0].mass
            masses.append(mass)
        return masses

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)
        # [1] masses
        if self.__masses is None:
            self.__masses = self.__get_masses()
        info['mass'] = self.__masses
        # [2] object states
        obj_ids = self.scene.cur_ids.long()
        state = self.tensors['root'][obj_ids, :]
        info['object_state'] = state
        # [3] robot states
        hand_state = self.tensors['root'][
            self.robot.actor_ids.long()]
        info['hand_state'] = hand_state
        print('appending mass')
        return (obs, rew, done, info)
