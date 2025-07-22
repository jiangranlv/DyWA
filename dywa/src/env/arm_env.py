#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from typing import Dict, Optional, Iterable
from dataclasses import dataclass
from gym import spaces
from functools import reduce

import torch as th
import numpy as np
import einops

from env.env.wrap.base import (
    WrapperEnv, ObservationWrapper, add_obs_field)
from env.push_env import PushEnv
from env.common import quat_rotate
from env.util import draw_sphere
from env.task.push_with_arm_task import PushWithArmTask
from util.config import ConfigBase
from util.torch_util import dcn
from util.math_util import apply_pose, apply_pose_tq, matrix_from_quaternion

from icecream import ic
import nvtx

Point = spaces.Box(-np.inf, +np.inf, (3,))
Pose = spaces.Box(-np.inf, +np.inf, (7,))
Pose6d = spaces.Box(-np.inf, +np.inf, (9,))
PoseVel = spaces.Box(-np.inf, +np.inf, (13,))
Pose6dVel = spaces.Box(-np.inf, +np.inf, (15,))
Wrench = spaces.Box(-np.inf, +np.inf, (6,))
Keypoint = spaces.Box(-np.inf, +np.inf, (24,))
Cloud512 = spaces.Box(-np.inf, +np.inf, (512, 3))
FrankaDofPosNoGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (7,))
FrankaDofPosVelNoGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (14,))
# in case we implement mimic joints, should be (9,) -> (8,)
FrankaDofPosWithGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (9,))
# in case we implement mimic joints, should be (16,) -> (15,)
FrankaDofPosVelWithGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (16,))
Mass = spaces.Box(-np.inf, +np.inf, (1,))
PhysParams = spaces.Box(-np.inf, +np.inf, (5,))

OBS_SPACE_MAP = {
    'point': Point,
    'pose': Pose,
    'pose6d': Pose6d,
    # FIXME: not entirely correct
    'relpose': Pose,
    # FIXME: not entirely correct
    'relpose6d': Pose6d,
    'pose_vel': PoseVel,
    'pose6d_vel': Pose6dVel,
    'keypoint': Keypoint,
    'cloud': Cloud512,
    'pos7': FrankaDofPosNoGripper,
    'pos9': FrankaDofPosWithGripper,
    'pos_vel7': FrankaDofPosVelNoGripper,
    'pos_vel9': FrankaDofPosVelWithGripper,
    'wrench': Wrench,
    'mass': Mass,
    'phys_params': PhysParams,
    'none': None
}

# NOTE: why do we independently track `ObsBoundMap`
# instead of integrating into spaces.Box?
# The main reason is that our normalization scheme,
# x' = (x-c)/s,
# is not necessarily "compatible" with the format of
# spaces.Box (offset/scaling vs. lower-upper bounds).


def _sanitize_bounds(b):
    if b is None:
        return None
    try:
        assert (len(b) == 2)
    except AssertionError:
        print('b')
        print(b)
        raise

    # Explicit ~scalar conversions
    try:
        b0 = float(b[0])
        b1 = float(b[1])
        return (b0, b1)
    except TypeError:
        pass

    try:
        b0 = [float(x) for x in b[0]]
        b1 = [float(x) for x in b[1]]
    except TypeError:
        print(b)

    if isinstance(b0, Iterable):
        b0 = tuple(b0)
    if isinstance(b1, Iterable):
        b1 = tuple(b1)

    return (b0, b1)


def _merge_bounds(*args):
    s = _sanitize_bounds
    args = [s(x) for x in args]
    return reduce(lambda a, b: (tuple(a[0]) + tuple(b[0]),
                                tuple(a[1]) + tuple(b[1])),
                  args)


def _identity_bound(shape):
    bound = np.zeros(shape), np.ones(shape)
    return _sanitize_bounds(bound)


def _get_obs_bound_map():
    point = ((0.0, 0.0, 0.55), (0.4, 0.4, 0.4))
    vector = ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    quat = ([0] * 4, [1] * 4)
    sixd = ([0] * 6, [1] * 6)
    rot_6d = ([0] * 6, [1] * 6)
    lin_vel = ([0] * 3, [1] * 3)
    ang_vel = ([0] * 3, [1] * 3)
    pose = _merge_bounds(point, quat)
    pose6d = _merge_bounds(point, sixd)
    relpose = _merge_bounds(vector, quat)
    relpose6d = _merge_bounds(vector, sixd)
    pose_vel = _merge_bounds(pose, lin_vel, ang_vel)
    pose6d_vel = _merge_bounds(pose6d, lin_vel, ang_vel)
    keypoint = (point[0] * 8, point[1] * 8)  # flattened for some reason...
    cloud = point  # exploits broadcasting
    pos7 = ([0] * 7, [1] * 7)
    pos9 = ([0] * 9, [1] * 9)
    vel7 = ([0] * 7, [1] * 7)
    vel9 = ([0] * 9, [1] * 9)
    pos_vel7 = _merge_bounds(pos7, vel7)
    pos_vel9 = _merge_bounds(pos9, vel9)
    force = ([0] * 3, [30] * 3)
    torque = ([0] * 3, [1] * 3)

    wrench = _merge_bounds(force, torque)
    # FIXME: the ranges here are set arbitrarily !!
    mass = ([0], [1])
    friction = ([0], [2])
    phys_params = ([0.5, 1.0, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 1.0, 0.5])

    out = dict(
        point=point,
        quat=quat,
        rot_6d=rot_6d,

        lin_vel=lin_vel,
        ang_vel=ang_vel,
        pose=pose,
        pose6d=pose6d,
        relpose=relpose,
        relpose6d=relpose6d,
        pose_vel=pose_vel,
        pose6d_vel=pose6d_vel,
        keypoint=keypoint,
        cloud=cloud,
        pos7=pos7,
        pos9=pos9,
        vel7=vel7,
        vel9=vel9,
        pos_vel7=pos_vel7,
        pos_vel9=pos_vel9,
        force=force,
        torque=torque,
        wrench=wrench,
        mass=mass,
        friction=friction,
        phys_params=phys_params
    )
    out = {k: _sanitize_bounds(v)
           for (k, v) in out.items()}
    return out


OBS_BOUND_MAP = _get_obs_bound_map()


class ArmEnvWrapper(ObservationWrapper):

    @dataclass
    class Config(ConfigBase):
        # options: (point,pose,cloud,keypoint,none)
        # add_goal: bool = True
        goal_type: str = 'pose'
        # options: (pose,pose_vel,cloud,keypoint,none)
        object_state_type: str = 'pose_vel'
        # options: (pose,pose_vel,none)
        hand_state_type: str = 'pose_vel'
        # options: (pos,pos_vel,none)
        robot_state_type: str = 'pos_vel7'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)

        new_obs = {
            'goal': cfg.goal_type,
            'object_state': cfg.object_state_type,
            'hand_state': cfg.hand_state_type,
            'robot_state': cfg.robot_state_type,
        }

        obs_space = env.observation_space
        update_fn = {}
        for k, t in new_obs.items():
            if isinstance(t, spaces.Space):
                s = t
            else:
                s = OBS_SPACE_MAP.get(t)
            if s is None:
                continue
            obs_space, update_fn[k] = add_obs_field(obs_space, k, s)
        self._obs_space = obs_space
        self._update_fn = update_fn

        self._get_fn = {
            'goal': self.__goal,
            'object_state': self.__object_state,
            'robot_state': self.__robot_state,
            'hand_state': self.__hand_state,
        }

    @property
    def observation_space(self):
        return self._obs_space

    def push_data(self, *args, **kwds):
        # FIXME:
        # this method was added only for
        # CRM-env drop-in replacement compatibility.
        pass

    def __goal(self):
        cfg = self.cfg
        # FIXME: padding `cur_cloud` is only valid
        # if using full-cloud inputs...!
        return self.__rigid_body_state(cfg.goal_type,
                                       self.task.goal,
                                       self.scene.cur_cloud,
                                       self.scene.cur_bboxes)

    def __object_state(self):
        cfg = self.cfg
        obj_ids = self.scene.cur_ids.long()
        obj_state = self.tensors['root'][obj_ids, :]
        # FIXME: padding `cur_cloud` is only valid
        # if using full-cloud inputs...!
        return self.__rigid_body_state(cfg.object_state_type,
                                       obj_state,
                                       self.scene.cur_cloud,
                                       self.scene.cur_bboxes)

    def __robot_state(self):
        cfg = self.cfg
        if cfg.robot_state_type.startswith('pos_vel'):
            return self.tensors['dof'].reshape(
                self.tensors['dof'].shape[0], -1)
        elif cfg.robot_state_type.startswith('pos'):
            return self.tensors['dof'][..., :, 0]
        else:
            raise ValueError(
                F'Unknown robot_state_type={cfg.robot_state_type}')

    def __hand_state(self):
        cfg = self.cfg
        body_tensors = self.tensors['body']
        body_indices = self.robot.tip_body_indices.long()
        eef_pose = body_tensors[body_indices, :]
        return self.__rigid_body_state(cfg.hand_state_type,
                                       eef_pose,
                                       None,
                                       None)

    def __rigid_body_state(self,
                           obs_type: str,
                           ref_pose_vel: Optional[th.Tensor] = None,
                           ref_cloud: Optional[th.Tensor] = None,
                           ref_bbox: Optional[th.Tensor] = None):
        if obs_type in ['pose', 'relpose']:
            return ref_pose_vel[..., 0:7]
        elif obs_type == 'pose_vel':
            return ref_pose_vel[..., 0:13]
        elif obs_type in ['pose6d', 'relpose6d']:
            rot_mat = matrix_from_quaternion(ref_pose_vel[..., 3:7])
            qd = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
            # print( th.cat([ref_pose_vel[..., :3], qd], dim=-1).shape)
            return th.cat([ref_pose_vel[..., :3], qd], dim=-1)
        elif obs_type == 'pose6d_vel':
            rot_mat = matrix_from_quaternion(ref_pose_vel[..., 3:7])
            qd = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
            return th.cat([ref_pose_vel[..., :3],
                           qd, ref_pose_vel[..., 7:13]], dim=-1)
        elif obs_type == 'cloud':
            assert (ref_cloud is not None)
            return apply_pose_tq(ref_pose_vel[..., None, 0:7], ref_cloud)
        elif obs_type == 'keypoint':
            assert (ref_bbox is not None)
            out = apply_pose_tq(ref_pose_vel[..., None, 0:7], ref_bbox)
            out = einops.rearrange(out, '... k d -> ... (k d)')
            return out
        else:
            raise ValueError(F'unknown type = {obs_type}')

    def _wrap_obs(self, obs: Dict[str, th.Tensor]):
        for k, u in self._update_fn.items():
            try:
                obs = u(obs, self._get_fn[k]())
            except Exception:
                print(F'Failed to wrap {k}')
                raise
        return obs


@dataclass
class ArmEnvConfig(PushEnv.Config, ArmEnvWrapper.Config):
    task: PushWithArmTask.Config = PushWithArmTask.Config()


def ArmEnv(cfg: ArmEnvConfig, **kwds):
    # FIXME: rename to get_arm_env when
    # migration as PushEnv wrapper is complete.
    task_cls = kwds.pop('task_cls', PushWithArmTask)
    env = PushEnv(cfg, task_cls=task_cls)
    env = ArmEnvWrapper(cfg, env)
    return env
