#!/usr/bin/env python3

from isaacgym import gymtorch

from typing import Tuple, Dict, Iterable, Any, Optional
from dataclasses import dataclass
from util.config import ConfigBase
from isaacgym.torch_utils import (
    quat_mul,
    quat_conjugate, quat_from_euler_xyz)


from env.task.base import TaskBase
from env.env.base import EnvBase
# from env.util import rotate_keypoint

import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.task.util import (
    sample_goal_v2 as sample_goal,
    sample_box_xy,
    sample_yaw)
from util.math_util import (quat_diff_rad,
                                quat_rotate,
                                quat_multiply,
                                quat_conjugate
                                )

import nvtx
from icecream import ic

MAX_HEIGHT: float = 0.3


def sample_cuboid_orientation(n: int,
                              dtype=th.float,
                              device=None):
    """ sample orientations """
    h = th.empty((n,), dtype=dtype, device=device).uniform_(
        -th.pi / 2, +th.pi / 2)
    z = th.sin(h)
    w = th.cos(h)
    q_z = th.stack([0 * z, 0 * z, z, w], dim=-1)

    irt2 = math.sqrt(2.0) / 2.0

    sc = th.as_tensor([
        [irt2, irt2],
        [1.0, 0.0],
        [-irt2, irt2]],
        dtype=dtype,
        device=device)  # 3,2
    sel = th.randint(3, size=(n,), device=device, dtype=th.long)
    scs = sc[sel, :]
    q_c = th.zeros_like(q_z)
    q_c[th.arange(n, dtype=th.long, device=device),
        th.randint(3, size=(n,))] = scs[..., 0]
    q_c[th.arange(n, dtype=th.long, device=device), 3] = scs[..., 1]
    return quat_multiply(q_z, q_c)


def dot(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    return th.einsum('...i, ...i -> ...', a, b)


def reset_goals_legacy(
        goal: th.Tensor,
        indices: th.Tensor,
        mnb: th.Tensor,
        mxb: th.Tensor,
        z: float,
        randomize: bool):
    if randomize:
        N: int = indices.shape[0]
        goal[indices, :3] = th.rand(
            (N, 3), dtype=goal.dtype,
            device=goal.device) * (mxb - mnb) + mnb
        goal[indices, 2] = z
    else:
        goal[indices, 0] = 0
        goal[indices, 1] = 0.3
        goal[indices, 2] = z


@th.jit.script
def potential(goal_pos: th.Tensor, obj_pos: th.Tensor) -> th.Tensor:
    """
    negative distance between two positions.
    This means, high potential = good
    """
    return -th.linalg.norm(
        obj_pos[..., :3] - goal_pos[..., :3],
        dim=-1)


# @th.jit.script
def pose_error(src_pos: th.Tensor,
               dst_pos: th.Tensor,
               src_orn: Optional[th.Tensor] = None,
               dst_orn: Optional[th.Tensor] = None,
               planar: bool = True,
               skip_orn: bool = False):
    ndim = (2 if planar else 3)
    pos_err = th.linalg.norm(dst_pos[..., :ndim] - src_pos[..., :ndim], dim=-1)
    if skip_orn:
        orn_err = th.zeros_like(pos_err)
    else:
        orn_err = th.abs(quat_diff_rad(dst_orn, src_orn))
    return (pos_err, orn_err)


# @th.jit.script
def keypoint_error(kpt: th.Tensor,
                   src_pos: th.Tensor, dst_pos: th.Tensor,
                   src_orn: th.Tensor, dst_orn: th.Tensor,
                   planar: bool = True,
                   fix: bool = True):
    if planar:
        t_src = src_pos.clone()
        t_src[..., 2] = 0
        t_dst = dst_pos.clone()
        t_dst[..., 2] = 0
    else:
        t_src = src_pos
        t_dst = dst_pos
    q_src = src_orn[..., None, :]
    q_dst = dst_orn[..., None, :]
    src_kpt = quat_rotate(q_src, kpt) + t_src[..., None, :]
    dst_kpt = quat_rotate(q_dst, kpt) + t_dst[..., None, :]

    if fix:
        pos_err = th.linalg.norm(dst_kpt - src_kpt, dim=-1)
    else:
        pos_err = th.linalg.norm(dst_kpt - src_kpt, dim=-1).mean(dim=-1)
    return pos_err


def compute_reward_from_path(
        path: th.Tensor,
        goal: th.Tensor,
        suc: th.Tensor,
        done: th.Tensor,
        sparse_reward: bool,
        use_keypoint: bool,
        canonical_keypoints: th.Tensor,
        use_pose_goal: bool,
        use_potential: bool,
        pot_coef: float,
        fail_coef: float,
        time_coef: float,
        succ_coef: float,
        epsilon: float,

        use_log_reward: bool = False,
        use_exp_reward: bool = False,
        k_1: Optional[float] = None,
        k_2: Optional[float] = None,
        gamma: float = 0.999,
        planar: bool = True
):
    if sparse_reward:
        rew = suc.float()
    else:
        if use_keypoint:
            kpt_err = keypoint_error(canonical_keypoints[None],
                                     path[..., 0:3],
                                     goal[..., 0:3],
                                     path[..., 3:7],
                                     goal[..., 3:7],
                                     fix=True)
            err = kpt_err
        else:
            # ic(path.shape)
            # ic(goal.shape)
            pos_err, orn_err = pose_error(
                path[..., 0:3],
                goal[..., 0:3],
                (path[..., 3:7] if use_pose_goal else None),
                (goal[..., 3:7] if use_pose_goal else None),
                planar=planar,
                skip_orn=(not use_pose_goal),
            )
            # TODO: consider weighting
            # ang_err and pos_err terms differently.
            # ic(pos_err.shape)
            # ic(orn_err.shape)
            err = (pos_err + orn_err)
            # err = orn_err

        if use_potential:
            if use_log_reward:
                # <log>
                pot = -k_1 * th.log(k_2 * err + 1)
                if use_keypoint:
                    pot = pot.mean(dim=-1)
                rew = (gamma * pot[1:] - pot[:-1])
                # rew = (pot[1:] - pot[:-1])
            elif use_exp_reward:
                # <exp>
                pot = k_1 * th.pow(gamma, k_2 * err)
                if use_keypoint:
                    pot = pot.mean(dim=-1)
                rew = (gamma * pot[1:] - pot[:-1])
                # rew = (pot[1:] - pot[:-1])
            else:
                # <linear>
                pot = -err
                if use_keypoint:
                    pot = pot.mean(dim=-1)
                rew = pot_coef * (gamma * pot[1:] - pot[:-1])
                # rew = pot_coef * (pot[1:] - pot[:-1])
        else:
            # rew = pot_coef * (1 / (err[1:] + epsilon))
            pot_rew = th.mean(1 / (err[1:] + epsilon), dim=-1)
            rew = pot_coef * pot_rew
        rew -= time_coef

        # Replace final step rewards with terminal rewards.
        rew[done] = -fail_coef
        rew[suc] = succ_coef

    return rew


def compute_feedback_legacy(
        # data
        root_tensor: th.Tensor,
        goal_tensor: th.Tensor,
        prev_obj_state: th.Tensor,
        has_prev: th.Tensor,
        contact_tensor: th.Tensor,
        step_tensor: th.Tensor,
        keypoints_tensor: th.Tensor,

        # indices
        object_ids: th.Tensor,
        table_ids: th.Tensor,

        # Goal check
        goal_radius: th.Tensor,
        goal_angle: th.Tensor,
        use_pose_goal: bool,
        check_stable: bool,
        max_speed: float,
        contact_thresh: float,
        use_keypoint: bool,
        use_potential: bool,

        # Fail (OOB,timeout) check
        bound_lo: th.Tensor,
        bound_hi: th.Tensor,
        max_steps: int,

        # sparse reward flag
        sparse_reward: bool,

        # Reward coefficients
        succ_coef: float,
        fail_coef: float,
        pot_coef: float,
        time_coef: float,
        epsilon: float,

        use_log_reward: bool = False,
        use_exp_reward: bool = False,
        k_1: Optional[float] = None,
        k_2: Optional[float] = None,
        gamma: float = 0.999,
        planar: bool = True
):
    oid = object_ids.long()
    obj_tensor = root_tensor[oid]
    obj_on_table = None
    table_contact_force = contact_tensor[table_ids.long()]
    contact_mag = th.linalg.norm(table_contact_force, dim=-1)
    obj_on_table = (contact_mag >= contact_thresh)

    # Compute 2D distance to goal.
    pos_3d = obj_tensor[..., :3]
    # pos_2d = pos_3d[..., :2]
    # pos_error = goal_tensor[..., :2] - pos_2d
    # goal_distance = th.linalg.norm(pos_error, dim=-1)

    pos_err1, orn_err1 = pose_error(
        obj_tensor[..., 0:3],
        goal_tensor[..., 0:3],
        (obj_tensor[..., 3:7] if use_pose_goal else None),
        (goal_tensor[..., 3:7] if use_pose_goal else None),
        skip_orn=(not use_pose_goal)
    )

    # Use same success criteria whatever reward we use
    suc = th.logical_and(
        pos_err1 <= goal_radius,
        obj_on_table
    )
    # Conditionally update `suc` based on orn_err.
    if use_pose_goal:
        suc = th.logical_and(suc,
                             orn_err1 <= goal_angle)

    # Conditionally update `suc` based on stability.
    if check_stable:
        # FIXME :this coefficient was arbitrarily set...
        # should be removed and replaced with something better
        C: float = 0.06889 / 4.0
        lin_vel = obj_tensor[..., 7:10]
        ang_vel = obj_tensor[..., 10:13]
        lin_speed = th.linalg.norm(lin_vel, dim=-1)
        ang_speed = th.linalg.norm(ang_vel, dim=-1)
        # TODO: how should we
        # correctly incorporate ang.speed?
        is_stable = th.logical_and(
            lin_speed <= max_speed,
            ang_speed <= max_speed / C,
        )
        suc = th.logical_and(suc, is_stable)

    # == Check other termination conditions ==
    oob = th.logical_or((pos_3d < bound_lo).any(dim=-1),
                        (pos_3d >= bound_hi).any(dim=-1))
    timeout = (step_tensor >= max_steps)
    # ic(suc.shape, oob.shape,
    #    timeout.shape)
    done = th.logical_or(th.logical_or(suc, oob), timeout)

    rew = compute_reward_from_path(
        th.stack((prev_obj_state[..., :7], obj_tensor[..., :7]), dim=0),
        # these are single-timestep quantities,
        # but we pretend like they are not.
        goal_tensor[None],
        suc[None],
        done[None],

        sparse_reward,
        use_keypoint,
        keypoints_tensor,
        use_pose_goal,
        use_potential,
        pot_coef,
        fail_coef,
        time_coef,
        succ_coef,
        epsilon,
        use_log_reward,
        use_exp_reward,
        k_1,
        k_2,
        gamma,
        planar=planar
    ).squeeze(dim=0)

    return (rew, done, suc, timeout)


def compute_workspace(pos, dim,
                      max_height: float = 0.3,
                      margin: float = 0.1):
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
    min_bound[..., 2] = (
        max_bound[..., 2] - max_height
        - 0.5 * dim[..., 2]
    )

    # workspace
    ws_lo = min_bound
    ws_hi = max_bound
    return (ws_lo, ws_hi)


class PushTask(TaskBase):

    @dataclass
    class Config(ConfigBase):
        goal_radius: float = 1e-1
        # goal_angle: float = float(np.deg2rad(30))
        use_pose_goal: bool = False
        goal_angle: float = float('inf')
        timeout: int = 1024
        sparse_reward: bool = False
        contact_thresh: float = 1e-2
        randomize_goal: bool = False
        goal_type: str = 'stable'
        # Only used if goal_type == stable
        randomize_yaw: bool = False

        # Use keypoint based reward or not
        use_keypoint: bool = False
        planar: bool = True
        # Use potential based reward or inverse
        use_potential: bool = True
        epsilon: float = 0.02

        # "potential reward" coefficient
        # between the object and the goal.
        pot_coef: float = 1.0

        # `fail_coef` is negated in practice :)
        fail_coef: float = 1.0
        succ_coef: float = 1.0
        time_coef: float = 0.001

        # Check whether or not the object has
        # stopped at the goal, based on
        # max-velocity threshold.
        check_stable: bool = False
        max_speed: float = 1e-1

        # Workspace height
        ws_height: float = 0.5
        ws_margin: float = 0.3

        # Minimum separation distance
        # between the object CoM and the goal.
        min_separation_scale: float = 1.0
        eps: float = 1e-6

        use_cliff: bool = False

        sample_thresh: Tuple[float, float] = (1.0, 1.0)
        margin_scale: float = 0.95

        use_log_reward: bool = False
        use_exp_reward: bool = False
        k_1: float = 0.37
        k_2: float = 27.2
        gamma: float = 0.995
        mode: str = 'train'

        filter_stable_goal: bool = True

    def __init__(self, cfg: Config, writer: Optional[SummaryWriter] = None):
        super().__init__()
        self.cfg = cfg
        self.goal: th.Tensor = None

        # Cache previous object positions for calculating potential-based
        # rewards
        self.__prev_obj_pose: th.Tensor = None
        self.__has_prev: th.Tensor = None

        # Adaptive goal thresholds
        self.goal_radius: float = cfg.goal_radius
        self.goal_angle: float = cfg.goal_angle
        self.max_speed: float = cfg.max_speed
        self.gamma: float = cfg.gamma

        self.goal_radius_samples: th.Tensor = None
        self.goal_angle_samples: th.Tensor = None
        self.max_speed_samples: th.Tensor = None

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
        cfg = self.cfg
        if cfg.filter_stable_goal:
            assert (env.scene.cfg.load_stable_mask)
        device: th.device = th.device(env.cfg.th_device)
        num_env: int = env.num_env
        self.env = env

        if cfg.use_pose_goal:
            # pose goal
            self.goal = th.empty(
                (num_env, 7),
                dtype=th.float,
                device=device)
        else:
            # position-only goal
            self.goal = th.empty(
                (num_env, 3),
                dtype=th.float,
                device=device)
        self.__prev_obj_pose = th.zeros(
            (num_env, 7),
            dtype=th.float,
            device=device)
        self.__has_prev = th.zeros(
            (num_env,),
            dtype=th.bool,
            device=device)

        p = np.asarray(env.scene_cfg.table_pos)
        d = np.asarray(env.scene_cfg.table_dims)

        min_bound = p - 0.5 * cfg.margin_scale * d
        max_bound = p + 0.5 * cfg.margin_scale * d
        # set min_bound also to tabletop.
        min_bound[..., 2] = max_bound[..., 2]
        self.goal_bound = th.zeros((num_env, 2, 3),
                                   dtype=th.float,
                                   device=device)
        self.goal_lo = self.goal_bound[..., 0, :]
        self.goal_hi = self.goal_bound[..., 1, :]
        self.goal_lo[...] = th.as_tensor(min_bound,
                                         dtype=th.float,
                                         device=device)
        self.goal_hi[...] = th.as_tensor(max_bound,
                                         dtype=th.float,
                                         device=device)
        self.env_lo = th.as_tensor(env.cfg.env_bound_lower,
                                   dtype=th.float,
                                   device=device)
        self.env_hi = th.as_tensor(env.cfg.env_bound_upper,
                                   dtype=th.float,
                                   device=device)
        self.ws_bound = th.zeros((num_env, 2, 3),
                                 dtype=th.float,
                                 device=device)
        self.ws_lo = self.ws_bound[:, 0]
        self.ws_hi = self.ws_bound[:, 1]

        self.table_body_ids = th.as_tensor(
            env.scene.table_body_ids,
            dtype=th.int32,
            device=device)

        self.goal_radius_samples = th.full((num_env,),
                                           self.goal_radius,
                                           dtype=th.float,
                                           device=device)
        self.goal_angle_samples = th.full((num_env,),
                                          self.goal_angle,
                                          dtype=th.float,
                                          device=device)
        self.max_speed_samples = th.full((num_env,),
                                         self.max_speed,
                                         dtype=th.float,
                                         device=device)

    def compute_reward_from_trajectory(
            self, traj: th.Tensor, extra: th.Tensor, goal: th.Tensor,
            rewd: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Compute reward from trajectory.
        `traj` should be buffered by one?
        (Assumes TxNxD layout)
        """
        cfg = self.cfg

        succ = th.zeros_like(goal[..., 0])
        succ[-1] = True
        done = succ

        # FIXME: ad-hoc hack that
        # assumes each environment has a
        # single unique bounding box that can be used as
        # keypoints_tensor
        keypoints = None
        if cfg.use_keypoint:
            keypoints = self.env.scene.cur_bboxes
            assert ((keypoints is not None))

        if rewd is None:
            rewd = th.zeros_like(goal[..., 0])
        if cfg.sparse_reward:
            rewd[-1] = 1.0
        else:
            rewd[1:] = compute_reward_from_path(
                traj,
                goal,
                succ[1:].bool(),
                done[1:].bool(),
                cfg.sparse_reward,
                cfg.use_keypoint,
                keypoints,
                cfg.use_pose_goal,
                cfg.use_potential,
                cfg.pot_coef,
                cfg.fail_coef,
                cfg.time_coef,
                cfg.succ_coef,
                cfg.epsilon,
                cfg.use_log_reward,
                cfg.use_exp_reward,
                cfg.k_1,
                cfg.k_2,
                self.gamma
            )
        return rewd

    def compute_feedback(self,
                         env: 'EnvBase',
                         obs: th.Tensor,
                         action: th.Tensor
                         ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        cfg = self.cfg
        max_speed: float = self.max_speed

        out = compute_feedback_legacy(
            env.tensors['root'],
            self.goal,
            self.__prev_obj_pose,
            self.__has_prev,
            env.tensors['net_contact'],
            env.buffers['step'],
            env.scene.cur_bboxes,

            env.scene.cur_ids,
            self.table_body_ids,

            self.goal_radius_samples,
            self.goal_angle_samples,
            cfg.use_pose_goal,
            cfg.check_stable,
            self.max_speed_samples,
            cfg.contact_thresh,
            cfg.use_keypoint,
            cfg.use_potential,

            self.ws_lo,
            self.ws_hi,
            cfg.timeout,

            cfg.sparse_reward,

            cfg.succ_coef,
            cfg.fail_coef,
            cfg.pot_coef,
            cfg.time_coef,
            cfg.epsilon,

            cfg.use_log_reward,
            cfg.use_exp_reward,
            cfg.k_1,
            cfg.k_2,
            self.gamma,
            cfg.planar
        )
        rew, done, suc, timeout = out

        info: Dict[str, Any] = {}
        info['success'] = suc
        info['timeout'] = timeout

        self.__prev_obj_pose[...] = env.tensors['root'][
            env.scene.cur_ids.long(), :7]
        return (rew, done, info)

    @nvtx.annotate('PushTask.reset()', color="blue")
    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        cfg = self.cfg
        if indices is None:
            indices = th.arange(env.cfg.num_env,
                                dtype=th.long,
                                device=env.cfg.th_device)
            num_reset: int = env.cfg.num_env
        else:
            indices = th.as_tensor(indices,
                                   dtype=th.long,
                                   device=env.cfg.th_device)
            num_reset: int = len(indices)

        # Revise workspace dimensions.
        table_pos = env.scene.table_pos[indices]
        table_dims = env.scene.table_dims[indices]

        goal_lo, goal_hi = compute_workspace(
            table_pos,
            table_dims * cfg.margin_scale,
            0.0,
            0.0)
        self.goal_lo[indices] = goal_lo
        self.goal_hi[indices] = goal_hi

        ws_lo, ws_hi = compute_workspace(
            table_pos,
            table_dims,
            cfg.ws_height,
            cfg.ws_margin
        )
        self.ws_lo[indices] = ws_lo
        self.ws_hi[indices] = ws_hi

        tabletop_z = (
            env.scene.table_pos[indices, 2]
            + 0.5 * env.scene.table_dims[indices, 2]
            # FIXME: VERY VERY VERY DANGEROUS CODE
            # THAT ASSUMES TARGET OBJET IS A CUBE.
            + self.env.scene.cur_radii[indices] / math.sqrt(3)
        )
        if cfg.use_cliff:
            # Sample along the edge of the table.
            bound = th.stack([
                goal_lo[..., 0],
                # goal_hi here is not a typo;
                # we purposefully sampling along
                # one edge.
                goal_hi[..., 1],
                goal_hi[..., 0],
                goal_hi[..., 1]], dim=-1).reshape(-1, 2, 2)
            self.goal[indices, :3] = sample_box_xy(bound, tabletop_z)
        elif env.scene.cfg.override_cube:
            x = env.scene.xsampler.sample((num_reset,)).to(env.device)

            yg = env.scene.Gysampler.sample((num_reset,)).to(env.device)

            z = (env.scene.cfg.table_dims[2] + env.scene.box_dim[2] / 2) * th.ones(
                num_reset, dtype=th.float, device=env.device)
            self.goal[indices, 0] = x
            self.goal[indices, 1] = yg
            self.goal[indices, 2] = z
        else:
            # Sample (somewhat) uniformly across
            # the tabletop surface.
            if cfg.min_separation_scale > 0:
                # FIXME: only works for
                # TabletopWith[...]
                obj_pos = env.tensors['root'][env.scene.cur_ids.long(), :2]
                self.goal[indices, :3] = sample_goal(
                    self.goal_bound[indices, ..., :2],
                    obj_pos[indices, ..., : 2],
                    self.goal_radius_samples[indices] * cfg.min_separation_scale,
                    z=tabletop_z,
                    eps=cfg.eps,
                    num_samples=16)

            else:
                reset_goals_legacy(self.goal, indices,
                                   self.goal_lo, self.goal_hi,
                                   tabletop_z,
                                   cfg.randomize_goal)

        if True:
            r0, r1 = cfg.sample_thresh
            self.goal_radius_samples[indices] = self.goal_radius * (
                (r1 - r0) * th.rand(size=(num_reset, ), dtype=self.goal_radius_samples.dtype,
                                    device=self.goal_radius_samples.device) + r0)
            self.goal_angle_samples[indices] = self.goal_angle * (
                (r1 - r0) * th.rand(size=(num_reset, ), dtype=self.goal_angle_samples.dtype,
                                    device=self.goal_angle_samples.device) + r0)
            self.max_speed_samples[indices] = self.max_speed * (
                (r1 - r0) * th.rand(size=(num_reset, ), dtype=self.max_speed_samples.dtype,
                                    device=self.max_speed_samples.device) + r0)

        if cfg.use_pose_goal:
            if env.scene.cfg.override_cube:

                roll = (np.pi / 2) * th.randint(0, 4,
                                                (num_reset,), device=env.device)
                pitch = (np.pi / 2) * th.randint(0, 4,
                                                 (num_reset,), device=env.device)
                yaw = (2 * np.pi) * th.rand(num_reset,
                                            dtype=th.float, device=env.device)
                self.goal[indices, 3: 7] = quat_from_euler_xyz(
                    roll, pitch, yaw)

            elif cfg.goal_type == 'stable':
                is_yaw_only = env.scene.cur_yaw_only[indices]
                random_indices = indices[~is_yaw_only]

                if cfg.filter_stable_goal:
                    EPS: float = 1e-6
                    prob = env.scene.cur_stable_masks.float().add_(EPS)
                    prob = prob.div_(prob.sum(keepdim=True, dim=-1))
                    which_pose = th.multinomial(
                        prob, num_samples=1, replacement=True).squeeze(
                        dim=-1)[random_indices]
                else:
                    which_pose = th.randint(
                        env.scene.cur_stable_poses.shape[1],
                        size=(len(random_indices),),
                        dtype=th.long,
                        device=env.scene.cur_stable_poses.device,
                    )
                # Option 1. Independently set z and quaternion values
                # self.goal[indices, 2] = env.scene.cur_stable_poses[indices, which_pose, 2]
                # self.goal[indices, 3:7] = (
                #     env.scene.cur_stable_poses[indices, which_pose, 3:7]
                # )
                # Option 2. Set them together.
                # Expression from option 1 is more safe, but we can actually
                # just couple them because z and quaternion happens to be
                # right next to each other.
                src_pose = env.tensors['root'][
                    env.scene.cur_ids[indices].long(),
                    : 7]
                self.goal[indices, 3:7] = src_pose[..., 3:7]
                self.goal[random_indices, 2:7] = (
                    env.scene.cur_stable_poses[random_indices, which_pose, 2:7]
                )

                if cfg.randomize_yaw:
                    qz = sample_yaw(num_reset, device=self.goal.device)
                    self.goal[indices, 3:7] = quat_multiply(
                        qz, self.goal[indices, 3:7])
            elif cfg.goal_type == 'stable_nonplanar':
                # src_pose = env.scene.cur_stable_poses[indices]  # N, P
                src_pose = env.tensors['root'][
                    env.scene.cur_ids[indices].long(),
                    : 7]
                all_pose = env.scene.cur_stable_poses[indices]  # N, M, P
                err_quat = quat_multiply(
                    quat_conjugate(src_pose[..., 3:7])[..., None, :],
                    all_pose[..., 3:7])
                err_axis = F.normalize(err_quat[..., :3], dim=-1)
                sel_prob = (err_axis[..., 2] <= math.cos(
                    math.radians(15.0))).float()  # N, M
                which_pose = th.multinomial(
                    sel_prob, 1, replacement=False).squeeze(
                    dim=-1)
                # see `stable` for goal-setting logic
                self.goal[indices, 2:7] = (
                    env.scene.cur_stable_poses[indices, which_pose, 2:7]
                )
                if cfg.randomize_yaw:
                    qz = sample_yaw(num_reset, device=self.goal.device)
                    self.goal[indices, 3:7] = quat_multiply(
                        qz, self.goal[indices, 3:7])
            elif cfg.goal_type == 'cuboid':
                self.goal[indices, 3:7] = sample_cuboid_orientation(
                    num_reset,
                    device=self.goal.device)
            elif cfg.goal_type == 'yaw_only':
                self.goal[indices, 3:7] = sample_yaw(
                    num_reset,
                    device=self.goal.device)
            elif cfg.goal_type == 'sampled':
                # TODO What if num of goal in scenario > 1?
                self.goal[indices, :] = env.scene.cur_goals[indices, 0, :]
            else:
                # Sample random unit quaternion.
                self.goal[indices, 3: 7] = F.normalize(th.randn(
                    size=(num_reset, 4),
                    dtype=self.goal.dtype, device=self.goal.device),
                    p=2, dim=-1)

        self.__prev_obj_pose[indices] = env.tensors['root'][
            env.scene.cur_ids.long()[indices], :7]
        self.__has_prev[indices] = False


def main():
    print(th.unique(sample_cuboid_orientation(1024), dim=0))


if __name__ == '__main__':
    main()
