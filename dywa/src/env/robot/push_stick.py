#!/usr/bin/env python3

import pkg_resources
from dataclasses import dataclass
from util.config import ConfigBase
from gym import spaces
from typing import Iterable, List

from isaacgym import gymtorch
from isaacgym import gymapi
import numpy as np
import torch as th

from env.env.base import EnvBase
from env.robot.base import RobotBase


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


def find_actor_indices(gym, envs: Iterable, name: str,
                       domain=gymapi.IndexDomain.DOMAIN_SIM) -> List[int]:
    return [gym.find_actor_index(envs[i], name, domain)
            for i in range(len(envs))]


class PushStick(RobotBase):
    """
    toy robot connected by a single prismatic joint.
    """

    @dataclass
    class Config(ConfigBase):
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        robot_file: str = 'push_stick/robot.urdf'
        randomize_init_pos: bool = False
        randomize_init_orn: bool = False
        max_force: float = 50.0 * 0.1
        max_torque: float = 5.0 * 0.1

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
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.override_inertia = True
        asset_options.override_com = True
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
            -1)

        # Configure gripper properties.
        num_dof = gym.get_asset_dof_count(
            self.assets['robot'])
        dof_props = gym.get_asset_dof_properties(
            self.assets['robot'])

        for i in range(num_dof):
            # robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # Apparently this is what ACRONYM expects!
            # dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            # FIXME: ???
            # dof_props['stiffness'][i] = 0.0  # 5000.0
            # dof_props['damping'][i] = 1e3  # 1e2
            # dof_props['friction'][i] = 1e3  # 1e2
            # dof_props['armature'][i] = 10  # 1e2
        gym.set_actor_dof_properties(env,
                                     robot_actor, dof_props)

        # Does this matter??
        # self.actor_ids[env_id] = gym.find_actor_index(env,
        #                                               F'robot',
        #                                               # gymapi.IndexDomain.DOMAIN_ENV
        #                                               gymapi.IndexDomain.DOMAIN_SIM
        #                                               )
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
        qpos = None
        qvel = None
        # [2] reset poses.
        indices = self.actor_ids[index]
        root_tensor = env.tensors['root']

        I = indices.long()
        if True:
            root_tensor.index_fill_(0, I, 0)

            # Configure randomization and such
            p = env.scene.table_pos[index]
            d = env.scene.table_dims[index]
            if self.cfg.randomize_init_pos:
                min_bound = p - 0.5 * d
                max_bound = p + 0.5 * d
            else:
                dd = d.clone()
                dd[..., :2] = 0
                pp = p.clone()
                pp[..., 0] += 0.3
                min_bound = pp - 0.5 * dd
                max_bound = pp + 0.5 * dd

            # Set min_bound also to tabletop.
            min_bound[..., 2] = max_bound[..., 2] = (
                max_bound[..., 2] + 0.2)

            reset_pos(root_tensor[..., :3], I,
                      min_bound, max_bound)
        else:
            # Hardcode gripper pose at center of table>
            z = (env.scene.table_pos[2]
                 + 0.5 * env.cfg.scene.table_dims[2]
                 + 0.5 * self.cfg.cube_dims[2])
            root_tensor[I, :3] = th.as_tensor([
                env.cfg.scene.table_pos[0],
                env.cfg.scene.table_pos[1],
                z], dtype=root_tensor.dtype,
                device=root_tensor.device)

        # TODO: support orn randomization maybe
        if self.cfg.randomize_init_orn:
            root_tensor[I, 3:7] = th.nn.functional.normalize(
                root_tensor[I, 3:7].normal_())
        else:
            root_tensor[I, 3:7] = th.as_tensor(
                (0, 0, 0, 1),
                dtype=root_tensor.dtype,
                device=root_tensor.device)

        return (indices, qpos, qvel)

    @property
    def action_space(self):
        cfg = self.cfg
        F: float = cfg.max_force
        return spaces.Box(np.asarray([-F]), np.asarray([+F]))

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
        self._control = th.zeros((env.num_env, self.n_dofs),
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

        self.indices = th.as_tensor(
            find_actor_indices(env.gym, env.envs, 'robot'),
            dtype=th.int32, device=env.device)

    def apply_actions(self, gym, sim, env, actions: th.Tensor,
                      done=None):
        """ We assume here joint-space actions.
        if you want specific behaviors (e.g.
        object-centric pushing, or grasping),
        then write a high-level controller on top
        of the joint-space actions. """
        if actions is None:
            return
        self._control[...] = actions
        # NOTE: assumes rank of action == 1
        if done is not None:
            self._control *= (~done[..., None]).float()
        indices = self.indices

        if True:
            gym.set_dof_actuation_force_tensor_indexed(
                sim, gymtorch.unwrap_tensor(self._control),
                gymtorch.unwrap_tensor(indices),
                len(indices)
            )
