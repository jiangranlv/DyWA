#!/usr/bin/env python3

from typing import Tuple, Dict, Iterable, Any, Optional
from dataclasses import dataclass
from util.config import ConfigBase

from env.task.base import TaskBase
from env.env.base import EnvBase
from env.util import get_mass_properties
from util.math_util import (quat_multiply, quat_conjugate)

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from icecream import ic

import nvtx


def axis_angle_from_quat(quat):
    axis = F.normalize(quat[..., 0:3])
    half_angle = th.acos(quat[..., 3:].clamp(-1.0, +1.0))
    angle = (2.0 * half_angle + th.pi) % (2 * th.pi) - th.pi
    return axis * angle


def dot(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    return th.einsum('...i, ...i -> ...', a, b)


# @th.jit.script
def check_if_object_stopped(
        # data
        root_tensor: th.Tensor,
        contact_tensor: th.Tensor,
        step_tensor: th.Tensor,
        ref_pose: th.Tensor,
        counter: th.Tensor,

        # indices
        object_ids: th.Tensor,
        table_ids: th.Tensor,

        max_speed: float,
        max_ang_speed: float,
        contact_thresh: float,
        max_delta: float,
        max_ang_delta: float,

        # Fail (OOB,timeout) check
        bound_lo: th.Tensor,
        bound_hi: th.Tensor,
        max_steps: int,

        # Reward coefficients
        succ_coef: float,
        fail_coef: float,
        patience: int,
):
    oid = object_ids.long()
    obj_tensor = root_tensor[oid]
    pos_3d = obj_tensor[..., :3]

    # TODO: can we better calculate obj_on_table condition?
    # e.g. by incorporating the object's mass X gravity?
    table_contact_force = contact_tensor[table_ids.long()]
    # contact_mag = th.linalg.norm(table_contact_force, dim=-1)
    # should be approximately (mass X gravity)
    # ic(table_contact_force)
    contact_mag = th.abs(table_contact_force[..., 2])
    obj_on_table = (contact_mag >= contact_thresh)

    # FIXME :this coefficient was arbitrarily set...
    # should be removed and replaced with something better
    lin_vel = obj_tensor[..., 7:10]
    ang_vel = obj_tensor[..., 10:13]
    lin_speed = th.linalg.norm(lin_vel, dim=-1)
    ang_speed = th.linalg.norm(ang_vel, dim=-1)
    # print('lin_speed', lin_speed)
    # print('ang_speed', ang_speed)
    # print('ang_vel', ang_vel)
    # TODO: how should we
    # correctly incorporate ang.speed?
    lin_delta = th.linalg.norm(ref_pose[..., 0:3] - obj_tensor[..., 0:3],
                               dim=-1)
    ang_delta = th.linalg.norm(
        axis_angle_from_quat(
            quat_multiply(
                quat_conjugate(obj_tensor[..., 3:7]),
                ref_pose[..., 3:7])
        ), dim=-1)

    slow = th.logical_and(
        lin_speed <= max_speed,
        ang_speed <= max_ang_speed)

    stop = th.logical_or(
        lin_delta <= max_delta,
        ang_delta <= max_ang_delta
    )

    is_stable = th.logical_and(slow, stop)

    oob = th.logical_or((pos_3d < bound_lo).any(dim=-1),
                        (pos_3d >= bound_hi).any(dim=-1))

    # ic(obj_on_table,
    #    slow,
    #    stop,
    #    is_stable,
    #    oob,
    #    lin_speed, max_speed,
    #    ang_speed, max_ang_speed,
    #    step_tensor, max_steps)

    stable_now = th.logical_and(
        th.logical_and(obj_on_table, is_stable),
        th.logical_not(oob))
    counter.add_(1)  # increment counters by default
    counter.mul_(th.logical_not(stable_now))  # reset counters for failures

    # Also initialize reference pose
    start = th.logical_and(stable_now, counter == 1)
    ref_pose[start] = obj_tensor[..., :7][start]

    timeout = (step_tensor >= max_steps)
    success = (counter >= patience)
    done = th.logical_or(th.logical_or(success,
                                       oob), timeout)
    rew = (-fail_coef) * done.float() + (succ_coef + fail_coef) * success.float()
    return (rew, done, success, timeout)


def compute_workspace(pos, dim,
                      max_height: float = 0.3,
                      margin: float = 0.0):

    # max bound
    max_bound = th.add(
        pos,
        th.multiply(0.5, dim))
    max_bound[..., :2] += margin
    max_bound[..., 2] += max_height

    # min bound
    min_bound = th.subtract(
        pos,
        th.multiply(0.5, dim))
    min_bound[..., :2] -= margin
    min_bound[..., 2] = max_bound[..., 2] - max_height
    # min_bound[..., 2] = (
    #    max_bound[..., 2] - max_height
    #    - 0.5 * dim[..., 2]
    # )

    # workspace
    ws_lo = min_bound
    ws_hi = max_bound
    return (ws_lo, ws_hi)


class WaitTillStopTask(TaskBase):

    @dataclass
    class Config(ConfigBase):
        timeout: int = 1024
        rel_contact_thresh: float = 0.9

        # `fail_coef` is negated in practice :)
        fail_coef: float = 1.0
        succ_coef: float = 1.0

        # Check whether or not the object has
        # stopped at the goal, based on
        # max-velocity threshold.
        max_speed: float = 1e-2

        # Hmm... too strict?
        # max_ang_speed: float = float(np.deg2rad(0.1))
        # max_ang_speed: float = 1e-2
        max_ang_speed: float = float('inf')

        max_delta: float = 0.01
        max_ang_delta: float = float(np.deg2rad(3.0))

        # Workspace height
        ws_height: float = 0.3
        ws_margin: float = 0.0

        # Minimum separation distance
        # between the object CoM and the goal.
        # min_separation_scale: float = 1.0
        eps: float = 1e-6
        patience: int = 16

        # dummy arg
        check_stable: bool = True

    def __init__(self, cfg: Config, writer: Optional[SummaryWriter] = None):
        super().__init__()
        self.cfg = cfg
        self._writer = writer

    @property
    def timeout(self) -> int:
        return self.cfg.timeout

    def create_assets(self, *args, **kwds):
        return {}

    def create_actors(self, *args, **kwds):
        return {}

    def create_sensors(self, *args, **kwds):
        return {}

    def setup(self, env: 'EnvBase'):
        device: th.device = th.device(env.cfg.th_device)
        num_env: int = env.num_env

        p = np.asarray(env.scene_cfg.table_pos)
        d = np.asarray(env.scene_cfg.table_dims)

        min_bound = p - 0.5 * d
        max_bound = p + 0.5 * d
        # set min_bound also to tabletop.
        min_bound[..., 2] = max_bound[..., 2]
        self.ws_bound = th.zeros((num_env, 2, 3),
                                 dtype=th.float,
                                 device=device)
        self.ws_lo = self.ws_bound[:, 0]
        self.ws_hi = self.ws_bound[:, 1]

        self.table_body_ids = th.as_tensor(
            env.scene.table_body_ids,
            dtype=th.int32,
            device=device)
        mass_props, _ = get_mass_properties(env.gym, env.envs,
                                            handles=env.scene.obj_handles)
        self.masses = th.as_tensor(mass_props,
                                   dtype=th.float,
                                   device=device)
        self.counter = th.zeros((num_env,),
                                dtype=th.int32,
                                device=device)
        self.ref_pose = th.zeros((num_env, 7),
                                 dtype=th.float32,
                                 device=device)

    def compute_feedback(self,
                         env: 'EnvBase',
                         obs: th.Tensor,
                         action: th.Tensor
                         ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        cfg = self.cfg
        out = check_if_object_stopped(
            env.tensors['root'],
            env.tensors['net_contact'],
            env.buffers['step'],
            self.ref_pose,
            self.counter,

            env.scene.cur_ids,
            self.table_body_ids,

            cfg.max_speed,
            cfg.max_ang_speed,
            # cfg.rel_contact_thresh * self.masses * 9.81,
            cfg.rel_contact_thresh * self.masses * 9.81,
            cfg.max_delta,
            cfg.max_ang_delta,

            self.ws_lo,
            self.ws_hi,
            cfg.timeout,

            cfg.succ_coef,
            cfg.fail_coef,
            cfg.patience
        )
        rew, done, suc, timeout = out

        info: Dict[str, Any] = {}
        info['success'] = suc
        info['timeout'] = timeout
        return (rew, done, info)

    @nvtx.annotate('PushTask.reset()', color="blue")
    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        cfg = self.cfg
        if indices is None:
            indices = th.arange(env.cfg.num_env,
                                dtype=th.long,
                                device=env.cfg.th_device)
        else:
            indices = th.as_tensor(indices,
                                   dtype=th.long,
                                   device=env.cfg.th_device)

        # Revise workspace dimensions.
        table_pos = env.scene.table_pos[indices]
        table_dims = env.scene.table_dims[indices]
        ws_lo, ws_hi = compute_workspace(
            table_pos,
            table_dims,
            cfg.ws_height,
            cfg.ws_margin
        )
        self.ws_lo[indices] = ws_lo
        self.ws_hi[indices] = ws_hi
        self.counter[indices] = 0
        self.ref_pose[indices] = env.tensors['root'][
            env.scene.cur_ids.long()][..., :7][indices]
