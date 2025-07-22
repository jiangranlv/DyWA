#!/usr/bin/env python3

import pkg_resources
from dataclasses import dataclass
from util.config import ConfigBase
from gym import spaces

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_mul
import numpy as np
import torch as th

from env.env.base import EnvBase
from env.robot.base import RobotBase
from env.env.help.pid import PID
from util.math_util import (orientation_error, axisaToquat)
from icecream import ic


def reset_pos(
        pos: th.Tensor,
        indices: th.Tensor,
        lo: th.Tensor,
        hi: th.Tensor) -> th.Tensor:
    N: int = indices.shape[0]
    pos[indices.long()] = th.rand(
        (N, 3), dtype=pos.dtype,
        device=pos.device) * (hi - lo) + lo
    return pos


class FEGripper(RobotBase):
    """
    Robot without a body that
    applies an arbitrary "poking" force to the
    target object at the specified position, direction, and magnitude.
    """

    @dataclass
    class Config(ConfigBase):
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        robot_file: str = 'fe-gripper/robot.urdf'
        randomize_init_pos: bool = True
        randomize_init_orn: bool = True
        # we can choose between wrench and position
        ctrl_mode: str = 'wrench'
        max_force: float = 20.0
        max_torque: float = 5.0 * 0.1
        # with poisition parametrized control mode
        max_pos: float = 0.1
        max_ori: float = 0.01
        KP_pos: float = 200.0
        KP_ori: float = 50.0
        lin_vel_damping: float = 1.0
        ang_vel_damping: float = 5.0
        max_lin_vel: float = 2.0
        max_ang_vel: float = 6.28

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.assets = {}
        self.actor_ids = {}
        self.n_dofs: int = 0
        self.dof_limits = None
        self.robot_radius: float = 0.12

    def create_assets(self, gym, sim):
        cfg = self.cfg

        # CONFIGURE URDF LOADING OPTIONS.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_inertia = True
        asset_options.override_com = True
        # ic(asset_options.linear_damping)
        # ic(asset_options.angular_damping)
        asset_options.linear_damping = cfg.lin_vel_damping
        asset_options.angular_damping = cfg.ang_vel_damping
        asset_options.max_linear_velocity = cfg.max_lin_vel
        asset_options.max_angular_velocity = cfg.max_ang_vel
        robot_asset = gym.load_urdf(sim,
                                    cfg.asset_root,
                                    cfg.robot_file,
                                    asset_options)

        # LOOKUP DOF PROPERTIES.
        self.n_dofs = gym.get_asset_dof_count(robot_asset)
        dof_props = gym.get_asset_dof_properties(robot_asset)
        dof_lo = []
        dof_hi = []
        for i in range(self.n_dofs):
            dof_lo.append(dof_props['lower'][i])
            dof_hi.append(dof_props['upper'][i])
        self.dof_limits = (
            np.asarray(dof_lo),
            np.asarray(dof_hi)
        )

        # CACHE ASSETS.
        self.assets = {'robot': robot_asset}
        return dict(self.assets)

    def create_actors(self, gym, sim, env, env_id: int):
        robot_actor = gym.create_actor(
            env,
            self.assets['robot'],
            gymapi.Transform(),
            F'robot',
            env_id,
            # 1,
            # -1,
            # -1
            0,
        )

        # Configure gripper properties.
        num_dof = gym.get_asset_dof_count(
            self.assets['robot'])
        dof_props = gym.get_asset_dof_properties(
            self.assets['robot'])

        if True:
            for i in range(num_dof):
                # Apparently this is what ACRONYM expects!
                # dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
                dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                # FIXME: these parameters may pose issues
                # when we want to control the gripper later.
                # dof_props['stiffness'][i] = 0.0  # 5000.0
                # dof_props['damping'][i] = 1e3  # 1e2
                # dof_props['friction'][i] = 1e3  # 1e2
                # dof_props['armature'][i] = 10  # 1e2
        else:
            for i in range(num_dof):
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        gym.set_actor_dof_properties(env,
                                     robot_actor, dof_props)

        return {'robot': robot_actor}

    def reset(self, gym, sim, env, env_id):
        # [0] Sanitize env_id arg.
        if env_id is None:
            env_id = th.arange(env.num_env,
                               dtype=th.int32,
                               device=env.device)
        if env_id.dtype in (bool, th.bool):
            index = env_id
            num_reset = env_id.sum()
        else:
            index = env_id.long()
            num_reset = len(env_id)

        # [1] reset joints - position, velocity
        # TODO: consider randomizing joint positions.
        qpos = th.zeros((env.num_env, self.n_dofs),
                        dtype=th.float32,
                        device=env.device)
        if False:
            # Set `qpos` randomly between lower/upper bounds.
            qpos[index] = (
                th.rand((num_reset, 1),
                        dtype=qpos.dtype, device=qpos.device) *
                (self.q_hi - self.q_lo) + self.q_lo
            )
        else:
            # Set `qpos` uniformly to a fixed "middle value".
            qpos[index] = 0.5 * (self.q_lo + self.q_hi)

        qvel = th.zeros((env.num_env, self.n_dofs),
                        dtype=th.float32,
                        device=env.device)
        # [2] reset poses.
        indices = self.actor_ids[index]
        root_tensor = env.tensors['root']
        # print('A')
        # print((~th.isfinite(root_tensor)).sum())

        if num_reset > 0:
            I = indices.long()
            # root_tensor.index_fill_(0, I, 0)
            root_tensor[I] = 0
            # root_tensor[I] = th.as_tensor(
            #     0, dtype=th.float, device=root_tensor.device)[
            #     None, None].expand_as(root_tensor[I])
            if self.cfg.randomize_init_pos:
                # Configure randomization and such
                p = env.scene.table_pos[index]
                d = env.scene.table_dims[index]
                min_bound = p - 0.5 * d
                max_bound = p + 0.5 * d
                # Set min_bound also to tabletop.
                min_bound[..., 2] = max_bound[..., 2] = (
                    max_bound[..., 2] + self.robot_radius)

                reset_pos(root_tensor[..., :3], I,
                          min_bound, max_bound)
            else:
                # Hardcode gripper pose at center of table.
                p = env.scene.table_pos[index]
                d = env.scene.table_dims[index]
                r = self.robot_radii[index]
                # NOTE: 0.2 = offset
                z = p[..., 2] + 0.5 * d[..., 2] + r
                root_tensor[I, :3] = th.stack([
                    p[..., 0] - 0.0,
                    p[..., 1] + 0.3,
                    z], dim=-1)

            # TODO: support orn randomization maybe
            if self.cfg.randomize_init_orn:
                root_tensor[I, 3:7] = th.nn.functional.normalize(
                    th.randn_like(root_tensor[I, 3:7]))
            else:
                # This is roughly "flat on table" configuration,
                # with the gripper finger facing away from the origin.
                root_tensor[I, 3:7] = th.as_tensor(
                    (0.5, 0.5, 0.5, -0.5),
                    dtype=root_tensor.dtype,
                    device=root_tensor.device)
            # print(self._prev_actions.shape)
            # print(env_id.min(), env_id.max())
            # print('B')
            # print((~th.isfinite(root_tensor)).sum())

            # self._prev_actions[env_id.long()] = 0
            # self._prev_actions.index_fill_(0, index, 0)
            # self._prev_actions.fill_(0)
            self._prev_actions[index] = th.as_tensor(
                0, dtype=th.float, device=self._prev_actions.device)[
                None, None].expand_as(self._prev_actions[index])

        if self.n_dofs <= 0:
            # "fixed finger" version
            return (indices, None, None)
        else:
            # "articulated finger" version
            return (indices, qpos, qvel)

    @property
    def action_space(self):
        cfg = self.cfg
        F: float = cfg.max_force if cfg.ctrl_mode == 'wrench' else cfg.max_pos
        T: float = cfg.max_torque if cfg.ctrl_mode == 'wrench' else cfg.max_ori

        return spaces.Box(
            np.asarray([-F, -F, -F, -T, -T, -T]),
            np.asarray([+F, +F, +F, +T, +T, +T]))

    def setup(self, env: 'EnvBase'):
        """ extra domain-specific setup """

        # Lookup actor ids and handles.
        actor_ids = []
        actor_handles = []
        for i in range(env.num_env):
            actor_id = env.gym.find_actor_index(
                env.envs[i],
                'robot',
                gymapi.IndexDomain.DOMAIN_SIM)
            actor_ids.append(actor_id)

            actor_handle = env.gym.find_actor_handle(
                env.envs[i],
                F'robot')
            actor_handles.append(actor_handle)
        self.actor_ids = th.as_tensor(actor_ids,
                                      dtype=th.int32,
                                      device=env.device)
        self.actor_handles = actor_handles

        self.q_lo = th.as_tensor(self.dof_limits[0],
                                 dtype=th.float, device=env.device)
        self.q_hi = th.as_tensor(self.dof_limits[1],
                                 dtype=th.float, device=env.device)

        # Bodies, forces, torques
        if env.num_env > 0:
            num_bodies = env.gym.get_env_rigid_body_count(
                env.envs[0])
        else:
            num_bodies = 0

        self.forces = th.zeros((env.num_env, num_bodies, 3),
                               device=env.device,
                               dtype=th.float32)
        self.torques = th.zeros((env.num_env, num_bodies, 3),
                                device=env.device,
                                dtype=th.float32)

        # Needed for wrench application
        # FIXME: hardcoded `panda_hand`
        self.rigid_body_ids = [
            env.gym.find_actor_rigid_body_index(
                env.envs[i],
                actor_handles[i],
                'panda_hand',
                gymapi.IndexDomain.DOMAIN_ENV
            ) for i in range(env.num_env)]
        self.rigid_body_ids = th.as_tensor(
            self.rigid_body_ids,
            dtype=th.long,
            device=env.device)

        # Needed for rendering
        # FIXME: hardcoded `panda_hand`
        self.hand_ids = [
            env.gym.find_actor_rigid_body_index(
                env.envs[i],
                actor_handles[i],
                'panda_hand',
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(env.num_env)
        ]
        self.hand_ids = th.as_tensor(
            self.hand_ids,
            dtype=th.long,
            device=env.device
        )

        self.robot_radii = th.full((env.num_env,),
                                   self.robot_radius,
                                   dtype=th.float,
                                   device=env.device)
        self._prev_actions = th.zeros((env.num_env, 6),
                                      dtype=th.float,
                                      device=env.device)
        target_dim = 6 if self.cfg.ctrl_mode == 'wrench' else 7
        self.target = th.zeros((env.num_env, target_dim),
                               dtype=th.float,
                               device=env.device)

    def step_controller(self, gym, sim, env):
        self.forces.fill_(0)
        self.torques.fill_(0)
        idx = self.rigid_body_ids

        if self.cfg.ctrl_mode == 'position':
            body_tensor = env.tensors['body']
            body_state = body_tensor[self.hand_ids]

            pos_err = self.target[:, :3] - body_state[:, :3]
            orn_err = orientation_error(
                self.target[:, 3: 7],
                body_state[:, 3: 7])
            hand_vel = body_state[:, 7:]
            force = (self.cfg.KP_pos * pos_err
                     - np.sqrt(self.cfg.KP_pos) * hand_vel[:, :3]
                     ).clamp_(
                min=-self.cfg.max_force, max=self.cfg.max_force)
            torque = (self.cfg.KP_ori * orn_err
                      - np.sqrt(self.cfg.KP_ori) * hand_vel[:, 3:]
                      ).clamp_(
                min=-self.cfg.max_torque, max=self.cfg.max_torque)
        else:
            force = self.target[:, :3]
            torque = self.target[:, 3:]

        self.forces[th.arange(len(idx)), idx] = force
        self.torques[th.arange(len(idx)), idx] = torque

        # Apply force/torque buffers to the actual simulation.
        out = gym.apply_rigid_body_force_tensors(
            sim,
            gymtorch.unwrap_tensor(self.forces),
            gymtorch.unwrap_tensor(self.torques),
            gymapi.ENV_SPACE
        )

        # Apply position target??
        if False:
            print('<set_dof_position_target>')
            qpos = th.zeros((env.num_env, self.n_dofs),
                            dtype=th.float32,
                            device=env.device)
            qpos[:] = 0.5 * (self.q_lo + self.q_hi)
            gym.set_dof_position_target_tensor(sim,
                                               gymtorch.unwrap_tensor(qpos))

    def apply_actions(self, gym, sim, env, actions: th.Tensor,
                      done=None):
        """ We assume here joint-space actions.
        if you want specific behaviors (e.g.
        object-centric pushing, or grasping),
        then write a high-level controller on top
        of the joint-space actions. """
        if actions is None:
            return

        # Clear actions for `done` environments.
        if done is not None:
            u = actions * (~done[..., None]).float()

        if self.cfg.ctrl_mode == 'position':
            idx = self.rigid_body_ids
            body_tensor = env.tensors['body']
            body_state = body_tensor[self.hand_ids]
            self.target[:, :3] = body_state[:, :3] + u[:, :3]
            self.target[:, 3:] = quat_mul(
                axisaToquat(u[:, 3:]), body_state[:, 3:7])

        else:
            self.target[:, :3] = u[..., :3]
            self.target[:, 3:] = u[..., 3:]

        self._prev_actions.copy_(u)

        # # self._prev_actions.copy_(actions)
        # self._prev_actions[:, :3] = force
        # self._prev_actions[:, 3:] = torque

        # # Apply actions to force/torque buffers.
        # self.forces.fill_(0)
        # self.torques.fill_(0)
        # idx = self.rigid_body_ids
        # self.forces[th.arange(len(idx)), idx] = force
        # self.torques[th.arange(len(idx)), idx] = torque

        # # print(self.forces[0])
        # # print(self.torques[0])

        # # Apply force/torque buffers to the actual simulation.
        # out = gym.apply_rigid_body_force_tensors(
        #     sim,
        #     gymtorch.unwrap_tensor(self.forces),
        #     gymtorch.unwrap_tensor(self.torques),
        #     gymapi.ENV_SPACE
        # )
