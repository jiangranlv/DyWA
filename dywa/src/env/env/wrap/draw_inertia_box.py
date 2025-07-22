#!/usr/bin/env python3

from typing import Iterable
from isaacgym import gymapi, gymtorch
from env.env.wrap.base import WrapperEnv
# from env.util import draw_inertia_box
from env.util import (draw_one_inertia_box, from_vec3, from_mat3)
from util.torch_util import dcn

import numpy as np
import cv2


class DrawInertiaBox(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, eps: float = 2e-7,
                 blocklist: Iterable[int] = None,
                 check_viewer: bool = True):
        super().__init__(env)
        self.__eps = eps
        if blocklist is None:
            blocklist = []
        self.blocklist = blocklist
        self.check_viewer = check_viewer

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return

        gym = self.gym
        viewer = self.viewer

        body_tensor = dcn(self.tensors['body'])
        for env in self.envs:
            num_actors: int = gym.get_actor_count(env)
            for actor_index in range(num_actors):
                if actor_index in self.blocklist:
                    continue
                actor_handle = gym.get_actor_handle(env, actor_index)

                body_count: int = gym.get_actor_rigid_body_count(
                    env, actor_handle)
                prop = gym.get_actor_rigid_body_properties(
                    env, actor_handle)
                for i in range(body_count):
                    body_index = gym.get_actor_rigid_body_index(
                        env, actor_handle, i,
                        gymapi.IndexDomain.DOMAIN_SIM)
                    state = body_tensor[body_index]
                    if prop[i].mass <= self.__eps:
                        continue

                    mass_index = np.reshape(
                        (prop[i].mass / 2.0) * 255,
                        (1,
                         1)).astype(
                        np.uint8)
                    color = (
                        cv2.applyColorMap(
                            mass_index,
                            cv2.COLORMAP_TURBO) /
                        255.0)[
                        0,
                        0].astype(
                        np.float32)
                    color = tuple(float(x) for x in color)
                    draw_one_inertia_box(gym, viewer, env,
                                         state,
                                         prop[i].mass,
                                         from_vec3(prop[i].com),
                                         from_mat3(prop[i].inertia),
                                         color=color)

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        self.__draw()
        return out
