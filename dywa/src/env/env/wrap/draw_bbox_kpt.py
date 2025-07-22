#!/usr/bin/env python3

from typing import Iterable, Callable
from isaacgym import gymapi, gymtorch
import torch as th
import numpy as np
import cv2

from env.env.wrap.base import WrapperEnv
from env.util import (draw_bbox, draw_keypoints, from_vec3, from_mat3)
from env.common import quat_rotate
from util.torch_util import dcn


class DrawBboxKeypoint(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, pose_fn: Callable[[None], th.Tensor]):
        super().__init__(env)
        self.pose_fn = pose_fn

    def __draw(self):
        if self.viewer is None:
            return
        gym = self.gym
        viewer = self.viewer
        obj_pose = self.pose_fn()
        bboxes = self.scene.cur_bboxes
        new_bboxes = dcn(obj_pose[..., None, 0:3] +
                         quat_rotate(obj_pose[..., None, 3:7], bboxes))
        # print('<draw_keypoints>')
        for index, env in enumerate(self.envs):
            draw_keypoints(gym, viewer, env, new_bboxes[index])
        # print('</draw_keypoints>')

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        self.__draw()
        return out


class DrawObjectBBoxKeypoint(DrawBboxKeypoint):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env):
        super().__init__(env, self.__get_pose)

    def __get_pose(self):
        obj_ids = self.scene.cur_ids.long()
        return self.tensors['root'][obj_ids, :7]


class DrawGoalBBoxKeypoint(DrawBboxKeypoint):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env):
        super().__init__(env, self.__get_pose)

    def __get_pose(self):
        return self.task.goal
