#!/usr/bin/env python3

import pkg_resources
from typing import Tuple, Iterable, List, Optional
import einops

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import (
    tf_combine, quat_unit, quat_mul,
    normalize
)

from dataclasses import dataclass
from util.config import ConfigBase
import numpy as np
import torch as th
from icecream import ic

from env.env.base import EnvBase
from env.robot.base import RobotBase
from gym import spaces

from env.robot.ur5_ik import ur5_ik, ur5_fk
from util.math_util import (
    matrix_from_quaternion,
    quat_from_axa
)

import nvtx

USE_CUSTOM_CONTROLLER: bool = False
KP: float = 500.0
KD: float = 0.0
VISCOUS_FRICTION: float = 10.0  # 0.0#100.0

BODY_NAMES = [
    'base_link',
    'base',
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
    'tool0',
    'panda_hand',
    'panda_leftfinger',
    'panda_rightfinger',
    'tool_tip']


def find_actor_indices(gym, envs: Iterable, name: str,
                       domain=gymapi.IndexDomain.DOMAIN_SIM) -> List[int]:
    return [gym.find_actor_index(envs[i], name, domain)
            for i in range(len(envs))]


def find_actor_handles(gym, envs: Iterable, name: str) -> List[int]:
    return [gym.find_actor_handle(envs[i], name)
            for i in range(len(envs))]

# def find_actor_handles(gym, envs:Iterable, name:str) -> List[int]:
#     return [gym.find_actor_rigid_body_index(envs[i], name)
#             for i in range(len(envs))]


def _pd_control(p, v, p_target, v_target, kp: float, kd: float):
    if p_target is None:
        dp = 0.0
    else:
        dp = (p_target - p)
    if v_target is None:
        dv = 0.0
    else:
        dv = (v_target - v)
    return kp * dp + kd * dv


@th.jit.script
def wrap_to_joint_limits(
        q: th.Tensor,
        q_ref: th.Tensor,
        q_lo: th.Tensor,
        q_hi: th.Tensor):
    """Wrap `q1` to the solution nearest to q0 within `q_lim`."""
    # Compute smallest possible deviation.
    dq = (q - q_ref + th.pi) % (2 * th.pi) - th.pi

    # This is the `q` value that is nearest to q0
    # among equivalent solutions.
    q = q_ref + dq

    # (2) wrap.
    # NOTE: reminder that `q_lim` has to be on the
    # RHS of the comparison, assuming that `q0`,`q1` are
    # arranged like (..., DoF).

    # FIXME: I'm going to override this just for now.
    # apparently it's really expensive, for some reason.
    # JUST BECAUES I know the joint limits are from -2pi to +2pi.
    # q[q < q_lo] += (2 * th.pi)
    # q[q >= q_hi] -= (2 * th.pi)
    q = (q + 2 * th.pi) % (4 * th.pi) - 2 * th.pi
    return q


class UR5FE(RobotBase):
    # action_space = spaces.Box(-np.inf, +np.inf, (8,))

    @dataclass
    class Config(ConfigBase):
        # cube_dims: Tuple[float, float, float] = (0.08, 0.08, 0.08)
        # cube_dims: Tuple[float, float, float] = (0.045, 0.045, 0.045)
        # apply_mask: bool = False

        # What are reasonably 'random' joint initializations?
        # I can think of four:
        # Option#0 - home position
        # Option#1 - uniformly sample joint cfgs
        # Option#2 - discretely sample from "valid" cfgs
        # Option#3 - kinematics-based euclidean-ish sampling
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        robot_file: str = 'ur5-fe/robot.urdf'
        # randomize_init_joints: bool = False
        # if `sample`, use one of the pre-sampled configurations
        # init_type: str = 'sample'
        init_type: str = 'home'

        # 1. joint position control
        # 2. joint velocity control
        # 3. cartesian position control; numerical IK
        # 4. cartesian position control; analytic IK (unsupported)
        # (jpos, jvel, cpos_n, cpos_a)
        ctrl_mode: str = 'jvel'
        target_type: str = 'rel'  # or 'abs'
        # Numerical IK damping factor.
        damping: float = 0.05
        # hold_pos: bool = False
        rot_type: str = 'axis_angle'

        keepout_radius: float = 0.2

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # if self.cfg.target_type != 'rel':
        #     raise ValueError('unsupported target type')

        self.n_bodies: int = None
        self.n_dofs: int = None
        self.dof_limits: Tuple[np.ndarray, np.ndarray] = None
        self.assets = {}
        self.valid_qs: Optional[th.Tensor] = None
        self.q_lo: th.Tensor = None
        self.q_hi: th.Tensor = None
        self._first = True
        self.q_home: th.Tensor = None

    def setup(self, env: 'EnvBase'):
        # FIXME: introspection!
        self.num_env = env.cfg.num_env
        self.device = th.device(env.cfg.th_device)

        self.indices = th.as_tensor(
            find_actor_indices(env.gym, env.envs, 'robot'),
            dtype=th.int32, device=self.device)
        self.handles = find_actor_handles(env.gym, env.envs, 'robot')
        self._control = th.zeros((self.num_env, self.n_dofs),
                                 dtype=th.float, device=self.device)
        self.q_lo = th.as_tensor(self.dof_limits[0],
                                 dtype=th.float, device=self.device)
        self.q_hi = th.as_tensor(self.dof_limits[1],
                                 dtype=th.float, device=self.device)
        if self.cfg.init_type == 'sample':
            # valid_qs = np.load('/tmp/qs.npy')
            valid_qs = np.load('/tmp/qs-000000-0001.npy')
            self.valid_qs = th.as_tensor(valid_qs,
                                         dtype=th.float32, device=self.device)
            self.valid_qs = (self.valid_qs + np.pi) % (2 * np.pi) - np.pi
            ic(self.valid_qs)
        if self.cfg.init_type == 'home':
            self.q_home = th.as_tensor(
                [0.0, -0.5 * np.pi, 0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi, 0.0],
                dtype=th.float32,
                device=self.device)

        # Acquire jacobian
        # FIXME: why here? well, it's because it's kinda
        # hard for env to know a prior what the name for the `robot`
        # should be...right? Hmmm....
        if self.cfg.ctrl_mode == 'cpos_n':
            self._jacobian = gymtorch.wrap_tensor(
                env.gym.acquire_jacobian_tensor(
                    env.sim, 'robot'))
            # FIXME: ACTUALLY I HAVE NO IDEA
            # WHAT proper EE_INDEX SHOULD BE !

            # I'm assuming that this is n-1 since
            # the very first body doesn't have a meaningfing jacobian :P
            EE_INDEX: int = self.n_bodies - 2
            self.j_eef = self._jacobian[:, EE_INDEX, :, :self.n_dofs]
            # presumably it looks like this:
            # num_env X (num_body-1) X 6(pose?) X 8(DoF)
            self.lmbda = th.eye(6, dtype=th.float,
                                device=self.device) * (self.cfg.damping**2)

        # Allocate homogeneous T. mat
        if self.cfg.ctrl_mode == 'cpos_a':
            self._T = th.zeros((self.num_env, 4, 4),
                               dtype=th.float32,
                               device=self.device)
            self._T[..., 3, 3] = 1

        ee_body_indices = []
        tip_body_indices = []
        for i in range(self.num_env):
            ee_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                # 'tool_tip',
                # 'wrist_3_link',
                'ee_link',
                gymapi.DOMAIN_SIM
            )
            ee_body_indices.append(ee_idx)

            tip_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                'tool_tip',
                gymapi.DOMAIN_SIM
            )
            tip_body_indices.append(tip_idx)

        self.ee_body_indices = th.as_tensor(
            ee_body_indices,
            dtype=th.int32,
            device=self.device)

        self.tip_body_indices = th.as_tensor(
            tip_body_indices,
            dtype=th.int32,
            device=self.device)

        self._hack_base_offset = th.as_tensor(
            [0.4, 0, -0.4],
            dtype=th.float32,
            device=self.device)

        assert (self.num_env > 0)
        self.link_body_indices = [env.gym.find_actor_rigid_body_index(
            env.envs[0],
            self.handles[0],
            BODY_NAMES[i],
            gymapi.DOMAIN_ENV)
            for i in range(len(BODY_NAMES))]

    def create_assets(self, gym, sim):
        cfg = self.cfg
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.enable_gyroscopic_forces = True

        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True
        # asset_options.override_inertia = True
        # asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.override_com = True

        robot_asset = gym.load_urdf(sim,
                                    cfg.asset_root,
                                    cfg.robot_file,
                                    asset_options)

        # Cache some properties.
        self.n_bodies = gym.get_asset_rigid_body_count(robot_asset)
        print(F'>>>> n_bodies = {self.n_bodies}')
        self.n_dofs = gym.get_asset_dof_count(robot_asset)
        dof_props = gym.get_asset_dof_properties(robot_asset)
        dof_lo = []
        dof_hi = []
        for i in range(self.n_dofs):
            dof_lo.append(dof_props['lower'][i])
            dof_hi.append(dof_props['upper'][i])
        self.dof_limits = (
            np.asarray(dof_lo), np.asarray(dof_hi)
        )

        if cfg.target_type == 'rel':
            # rel
            if cfg.ctrl_mode == 'jpos':
                # FIXME: configure appropriate
                # limits for relative joint commands
                # self.action_space = spaces.Box(dof_lo, dof_hi)
                raise ValueError(
                    'Unknown control combination:' +
                    F'{cfg.ctrl_mode}|{cfg.target_type}'
                )
            elif cfg.ctrl_mode == 'jvel':
                # FIXME: not *quite* accurate for the grippers!
                # self.action_space = spaces.Box(-np.pi / 2, +np.pi / 2, (8,))
                raise ValueError(
                    'Unknown control combination:' +
                    F'{cfg.ctrl_mode}|{cfg.target_type}'
                )
            elif cfg.ctrl_mode in ('cpos_n', 'cpos_a'):
                # FIXME: what about the grippers !?!?!
                # shouldn't this be (pose=7 + gripper=2) == 9?
                # OR should the gripper be symmetrical (1)??
                # FIXME: wrong action space !!
                if cfg.rot_type == 'axis_angle':
                    # ur5 EE about 1m/s; 1m/s * 0.02 = 0.02
                    self.action_space = spaces.Box(-0.02, +0.02, (6,))
                else:
                    self.action_space = spaces.Box(
                        np.asarray([-0.2, -0.2, -0.2, -1, -1, -1, -1]),
                        np.asarray([+0.2, +0.2, +0.2, +1, +1, +1, +1]),
                    )
        elif cfg.target_type == 'abs':
            # abs
            if cfg.ctrl_mode == 'jpos':
                self.action_space = spaces.Box(dof_lo, dof_hi)
            elif cfg.ctrl_mode == 'jvel':
                # FIXME: not *quite* accurate for the grippers!
                self.action_space = spaces.Box(-np.pi / 2, +np.pi / 2, (8,))
            if cfg.ctrl_mode in ('cpos_n', 'cpos_a'):
                # FIXME: what about the grippers !?!?!
                # shouldn't this be (pose=7 + gripper=2) == 9?
                # OR should the gripper be symmetrical (1)??
                # FIXME: wrong action space !!
                self.action_space = spaces.Box(
                    np.asarray([-0.4 - 1, -1, 0.4 - 1, -1, -1, -1, -1]),
                    np.asarray([-0.4 + 1, +1, 0.4 + 1, +1, +1, +1, +1]),
                )
        else:
            raise ValueError(F'unknown target = {cfg.target_type}')

        # ic(gym.get_asset_joint_names(robot_asset))
        # ic('body count', self.n_bodies)
        ic('body names', gym.get_asset_rigid_body_names(robot_asset))
        # for i in range(self.n_dofs):
        #     ic(gym.get_asset_actuator_name(robot_asset, i))

        self.assets = {'robot': robot_asset}
        return dict(self.assets)

    def create_actors(self, gym, sim, env, env_id: int):
        # collision group = env_id
        # indicates that it will collide with all other objects in the env.
        # collision filter = 0
        # means no collisions will be filtered out :)
        robot = gym.create_actor(env,
                                 self.assets['robot'],
                                 gymapi.Transform(),
                                 'robot',
                                 env_id,
                                 1)

        # Configure the controller.
        robot_dof_props = gym.get_asset_dof_properties(
            self.assets['robot'])

        CTRL_MODES = {
            'jpos': gymapi.DOF_MODE_POS,
            'jvel': gymapi.DOF_MODE_VEL,
            'cpos_n': gymapi.DOF_MODE_POS,
            'cpos_a': gymapi.DOF_MODE_POS,
        }
        ctrl_mode = CTRL_MODES[self.cfg.ctrl_mode]
        for i in range(self.n_dofs):
            # robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # Apparently this is what ACRONYM expects!
            if USE_CUSTOM_CONTROLLER:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i] = 0.0
            else:
                robot_dof_props['driveMode'][i] = ctrl_mode
                # 300.0 is apparently the default according to UR5 docs
                if ctrl_mode == gymapi.DOF_MODE_POS:
                    # <position control>
                    # ignore velocity error
                    robot_dof_props['stiffness'][i] = 300.0
                else:
                    # <velocity control>
                    # ignore position error
                    robot_dof_props['stiffness'][i] = 0.0
            # Modeling viscous friction
            robot_dof_props['damping'][i] = VISCOUS_FRICTION

        gym.set_actor_dof_properties(env,
                                     robot, robot_dof_props)

        return {'robot': robot}

    def reset(self, gym, sim, env, env_id) -> th.Tensor:
        """ Reset the _intrinsic states_ of the robot. """
        qpos = None
        qvel = None
        cfg = self.cfg
        if env_id is None:
            env_id = th.arange(self.num_env,
                               dtype=th.int32,
                               device=self.device)

        indices = self.indices[env_id.long()]
        # indices = env_id
        # I = indices.long()
        I = env_id.long()

        if self._first:
            iii = indices.long()
            root_tensor = env.tensors['root']
            # zero out first
            root_tensor[iii] = 0

            # Offset by keepout radius + table clearance
            root_tensor[iii, 0] = (
                env.scene.table_pos[..., 0]
                - 0.5 * env.scene.table_dims[..., 0]
                - cfg.keepout_radius
            )
            root_tensor[iii, 2] = env.scene.table_dims[..., 2]

            # unit quaternion
            root_tensor[iii, 6] = 1

        dof_tensor = env.tensors['dof']

        if True:
            # Initialize the joint positions.
            if cfg.init_type == 'zero':
                dof_tensor[I, ..., 0] = 0.0  # pos (zero?)
            elif cfg.init_type == 'sample':
                sample_indices = th.randint(self.valid_qs.shape[0],
                                            size=(len(I),))
                dof_tensor[I, ..., 0] = self.valid_qs[sample_indices.long()]
            elif cfg.init_type == 'home':
                dof_tensor[I, :6, 0] = self.q_home
                dof_tensor[I, 6:, 0] = 0.0

            # Currently, we always initialize velocity to zero.
            dof_tensor[I, ..., 1] = 0.0

        # This shouldn't matter right?
        # ppp = dof_tensor[..., 1].detach().clone()

        self._first = False
        # ___ SET ___
        # if True:  # self._first:
        #     self._first = False
        #     suc = gym.set_actor_root_state_tensor_indexed(
        #         sim,
        #         gymtorch.unwrap_tensor(root_tensor),
        #         gymtorch.unwrap_tensor(indices),
        #         len(indices)
        #     )
        # print(F'suc={suc}')
        # gym.set_dof_position_target_tensor_indexed(
        #     sim,
        #     gymtorch.unwrap_tensor(ppp),
        #     gymtorch.unwrap_tensor(indices),
        #     len(indices)
        # )
        # FIXME: should _NOT_ be called inside this class

        # NOTE SURE ABOUT THIS PART
        # FIXME: should _NOT_ be called inside this class
        # gym.set_dof_actuation_force_tensor_indexed(
        #     sim, gymtorch.unwrap_tensor(self._control),
        #     gymtorch.unwrap_tensor(indices),
        #     len(indices))

        # FIXME: should _NOT_ be called inside this class
        # BUT it's ok for now...
        # gym.set_dof_state_tensor_indexed(
        #     sim, gymtorch.unwrap_tensor(dof_tensor),
        #     gymtorch.unwrap_tensor(indices),
        #     len(indices))

        if self.cfg.ctrl_mode == 'jvel':
            # Currently, we always initialize control velocity to zero as well.
            self._control[I] = 0.0
            qvel = self._control
            # gym.set_dof_velocity_target_tensor_indexed(
            #     sim, gymtorch.unwrap_tensor(self._control),
            #     gymtorch.unwrap_tensor(indices),
            #     len(indices)
            # )
            # q = dof_tensor[I, ..., 0].detach().clone().contiguous()
            # gym.set_dof_position_target_tensor_indexed(
            #     sim, gymtorch.unwrap_tensor(q),
            #     gymtorch.unwrap_tensor(indices),
            #     len(indices)
            # )
        elif self.cfg.ctrl_mode == 'jpos':
            self._control[I] = dof_tensor[I, ..., 0]
            qpos = self._control
            # gym.set_dof_position_target_tensor_indexed(
            #     sim, gymtorch.unwrap_tensor(self._control),
            #     gymtorch.unwrap_tensor(indices),
            #     len(indices)
            # )
        elif self.cfg.ctrl_mode in ('cpos_n', 'cpos_a'):
            self._control[I] = dof_tensor[I, ..., 0]
            qpos = self._control
            # gym.set_dof_position_target_tensor_indexed(
            #     sim, gymtorch.unwrap_tensor(self._control),
            #     gymtorch.unwrap_tensor(indices),
            #     len(indices)
            # )
        else:
            raise KeyError('Unknown ctrl_mode')
        if USE_CUSTOM_CONTROLLER:
            return indices, None, None
        else:
            return indices, qpos, qvel

    @nvtx.annotate("UR5FE.apply_actions")
    def apply_actions(self, gym, sim, env, actions,
                      done=None):
        """ Set the actuation targets for the simulator. """
        if actions is None:
            return

        # TODO: think if this exactly what we want.
        # do we need _indexed() form in this case??
        indices = self.indices

        if self.cfg.ctrl_mode == 'cpos_n':
            # Solve damped least squares.
            j_eef = self.j_eef

            j_eef_T = th.transpose(j_eef, 1, 2)
            A = j_eef @ j_eef_T + self.lmbda
            # [1] Relative
            if self.cfg.target_type == 'rel':
                b = actions
            else:
                # [2] automatically compute the offset(s)
                # [abs] == automatically compute the offset(s)
                raise ValueError('Unsupported `abs` target type')
            rhs = th.linalg.solve(A, b)
            u = th.einsum('...jp,...p->...j', j_eef_T, rhs)
            # NOTE:
            # rather than "no action" for `done`,
            # we explicitly command to u==0, which seems
            # like a better idea.
            if done is not None:
                u *= (~done[..., None]).float()
            jpos0 = env.tensors['dof'][..., 0]
            self._control[...] = jpos0 + u
            gym.set_dof_position_target_tensor_indexed(
                sim, gymtorch.unwrap_tensor(self._control),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
        if self.cfg.ctrl_mode == 'cpos_a':
            with nvtx.annotate("cpos_a_ctrl"):
                with nvtx.annotate("QCVT"):
                    # [1] Convert actions into translations
                    # and (normalized) quaternion parameters.
                    if self.cfg.rot_type == 'axis_angle':
                        a_qxn = quat_from_axa(actions[..., 3:6])
                    elif self.cfg.rot_type == 'quaternion':
                        a_qxn = quat_unit(actions[..., 3:7])
                with nvtx.annotate("ATXN"):
                    a_txn = actions[..., :3]

                if self.cfg.target_type == 'rel':
                    with nvtx.annotate("REL"):
                        # Compute IK solution relative to current pose
                        # T' = T(action.translation) @ T @ T(action.rotation)
                        body_tensors = env.tensors['body']
                        body_indices = self.ee_body_indices.long()
                        Q0 = body_tensors[body_indices, ..., :7]

                        # txn = Q0[..., :3] + self._hack_base_offset
                        txn = (Q0[..., :3] - env.tensors['root']
                               [self.indices.long(), :3])

                        if True:
                            # T' = T(action.translation) @ T @
                            # T(action.rotation)
                            qxn = quat_mul(Q0[..., 3:7], a_qxn)
                            txn += a_txn
                        else:
                            # naive relative coords
                            qxn, txn = tf_combine(
                                Q0[..., 3:7], txn,
                                a_qxn, a_txn,
                            )

                        # qxn = Q0[..., 3:7]
                        # txn = Q0[..., :3]
                        matrix_from_quaternion(qxn, out=self._T[..., :3, :3])
                        self._T[..., :3, 3] = txn
                        # self._T[..., 0, 3] += 0.4
                        # self._T[..., 2, 3] -= 0.4
                        # ic(self._T)
                else:
                    with nvtx.annotate("ABS"):
                        # Compute IK solution in abs. frame
                        matrix_from_quaternion(qxn, out=self._T[..., :3, :3])
                        # [2] self._T gets transformed relative to the body frame
                        self._T[..., :3, 3] = txn
                        self._T[..., 0, 3] += 0.4
                        self._T[..., 2, 3] -= 0.4

                q_ref = env.tensors['dof'][..., :6, 0]
                with nvtx.annotate("IK"):
                    # ic(q_ref)
                    # ic('giving for IK', self._T)
                    q_target = ur5_ik(self._T, q_ref, self._control[..., :6])

                with nvtx.annotate("WRAPJ"):
                    # TODO: wrap to q_ref rather than
                    # blind normalization.
                    # q_target = (q_target + np.pi) % (2 * np.pi) - np.pi
                    q_target = wrap_to_joint_limits(
                        q_target, q_ref, self.q_lo[: 6],
                        self.q_hi[: 6])

                with nvtx.annotate("CTRL"):
                    self._control[..., :6] = q_target

                if False:
                    ic(actions)

                    ic('from', q_ref)
                    ic('got', self._control)
                    T2 = ur5_fk(self._control[..., :6])

                    ic('compare ur5 fk<->ik')
                    print(self._T)
                    print(T2)

                if actions.shape[-1] == 9:
                    # <enable_gripper>
                    self._control[..., 6:] = actions[..., -2:]
                else:
                    # <disable_gripper>
                    self._control[..., 6:] = 0.0

                # Reset "action" for the done ones.
                with nvtx.annotate("mask_done()"):
                    if done is not None:
                        # self._control[done] = env.tensors['dof'][done, ..., 0]
                        self._control[...] = th.where(
                            done[..., None], env.tensors['dof'][..., 0],
                            self._control)

                with nvtx.annotate("set_dof()"):
                    if USE_CUSTOM_CONTROLLER:
                        efforts = _pd_control(
                            env.tensors['dof'][..., :, 0],
                            env.tensors['dof'][..., :, 1],
                            self._control,
                            # None,
                            th.zeros_like(env.tensors['dof'][..., :, 1]),
                            KP, KD)
                        gym.set_dof_actuation_force_tensor_indexed(
                            sim, gymtorch.unwrap_tensor(efforts),
                            gymtorch.unwrap_tensor(indices),
                            len(indices)
                        )
                    else:
                        gym.set_dof_position_target_tensor_indexed(
                            sim, gymtorch.unwrap_tensor(self._control),
                            gymtorch.unwrap_tensor(indices),
                            len(indices)
                        )
        elif self.cfg.ctrl_mode == 'jvel':
            self._control[...] = actions
            # NOTE: assumes rank of action == 1
            if done is not None:
                self._control *= (~done[..., None]).float()
            gym.set_dof_velocity_target_tensor_indexed(
                sim, gymtorch.unwrap_tensor(self._control),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
        elif self.cfg.ctrl_mode == 'jpos':
            self._control[...] = actions
            gym.set_dof_position_target_tensor_indexed(
                sim, gymtorch.unwrap_tensor(self._control),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
