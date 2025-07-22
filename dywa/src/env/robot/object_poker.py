#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass
from util.config import ConfigBase

from isaacgym import gymtorch
from isaacgym import gymapi
import torch as th
import numpy as np

from env.env.base import EnvBase
from env.robot.base import RobotBase

import nvtx
from gym import spaces


# @th.jit.script
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


class ObjectPoker(RobotBase):
    """
    "cube" robot.

    Applies an arbitrary force to the CoM of the cube
    in order to change other objects in the scene.
    I guess...
    """

    @dataclass
    class Config(ConfigBase):
        apply_mask: bool = False
        randomize_init_pos: bool = True

    @property
    def action_space(self):
        F: float = 20.0
        T: float = 0.2
        return spaces.Box(
            np.asarray([-F, -F, -F, -T, -T, -T]),
            np.asarray([+F, +F, +F, +T, +T, +T]))

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.forces: th.Tensor = None
        self.force_pos: th.Tensor = None
        self.torques: th.Tensor = None
        self.init_lo: th.Tensor = None
        self.init_hi: th.Tensor = None

        # NOTE: needed? or can we
        # query directly from the `gym` instances?
        self.ee_object_keys = {}
        self.ee_object_radius: th.Tensor = None

    def create_assets(self, gym, sim):
        return {}

    def create_actors(self, gym, sim, env, env_id: int,
                      scene):
        cfg = self.cfg

        # Sample N objects from the pool.
        object_actors = []

        keys = scene.keys  # [env_id * cfg.num_obj_per_env:]
        key = np.random.choice(keys)
        self.ee_object_keys[env_id] = key
        obj_pose = gymapi.Transform()

        # Spawn objects.
        ee_object_actor = gym.create_actor(
            env,
            scene.assets['objects'][key],
            obj_pose,
            F'ee_object',
            env_id,
            0)

        # FIXME: hardcoded `100` offset
        # for the actor segmentation indices...
        # probably not the brightest idea.
        gym.set_rigid_body_segmentation_id(env,
                                           ee_object_actor,
                                           0, 100 + 0)

        # we consider the "object" to be representing
        # some sort of an end effector :)
        return {'ee_object': ee_object_actor}

    @nvtx.annotate('Robot.reset()', color="green")
    def reset(self, gym, sim, env, env_id):
        if env_id is None:
            env_id = th.arange(self.num_env,
                               dtype=th.int32,
                               device=self.device)
        indices = self.actor_ids[env_id.long()]
        root_tensor = env.tensors['root']

        I = indices.long()
        if self.cfg.randomize_init_pos:
            root_tensor.index_fill_(0, I, 0)
            reset_pos(root_tensor[..., :3], I,
                      self.init_lo, self.init_hi)
        else:
            # FIXME:
            # hardcoded cube pose at center of table
            # z = (env.cfg.scene.table_pos[2]
            #      + 0.5 * env.cfg.scene.table_dims[2]
            #      + 0.5 * self.cfg.cube_dims[2])
            z = env.cfg.scene.table_pos[2] + ee_object_radius
            root_tensor[I, :3] = th.as_tensor([
                env.cfg.scene.table_pos[0] - 0.0,
                env.cfg.scene.table_pos[1] - 0.5,
                z[I]],
                dtype=root_tensor.dtype,
                device=root_tensor.device)

        # TODO: support orn randomization maybe
        # if cfg.randomize_init_orn:
        root_tensor[I, 3:7] = th.as_tensor(
            (0, 0, 0, 1),
            dtype=root_tensor.dtype,
            device=root_tensor.device)

        # out = gym.set_actor_root_state_tensor_indexed(
        #     sim,
        #     gymtorch.unwrap_tensor(root_tensor),
        #     gymtorch.unwrap_tensor(indices),
        #     len(indices)
        # )
        # return out
        return indices, None, None

    def setup(self, env: 'EnvBase'):
        """ extra domain-specific setup """
        self.num_env = env.cfg.num_env
        self.device = env.cfg.th_device
        if self.num_env > 0:
            self.num_bodies = env.gym.get_env_rigid_body_count(
                env.envs[0])
        else:
            self.num_bodies = 0

        self.forces = th.zeros((self.num_env, self.num_bodies, 3),
                               device=self.device,
                               dtype=th.float32)
        self.torques = th.zeros((self.num_env, self.num_bodies, 3),
                                device=self.device,
                                dtype=th.float32)

        # Find indices
        actor_ids = []
        actor_handles = []
        for i in range(self.num_env):
            actor_id = env.gym.find_actor_index(
                env.envs[i],
                'ee_object',
                gymapi.IndexDomain.DOMAIN_SIM)
            actor_ids.append(actor_id)

            actor_handle = env.gym.find_actor_handle(
                env.envs[i],
                F'ee_object')
            actor_handles.append(actor_handle)
        self.actor_ids = th.as_tensor(actor_ids,
                                      dtype=th.int32,
                                      device=self.device)
        self.actor_handles = actor_handles

        # Find rigid-body indices
        self.rigid_body_ids = [
            env.gym.find_actor_rigid_body_index(
                env.envs[i],
                actor_handles[i],
                'base_link',
                gymapi.IndexDomain.DOMAIN_ENV
            ) for i in range(env.cfg.num_env)]
        self.rigid_body_ids = th.as_tensor(
            self.rigid_body_ids,
            dtype=th.long,
            device=self.device)

        # Precompute object radii for each env.
        ee_object_radius = [None for _ in range(env.cfg.num_env)]
        for (env_id, key) in self.ee_object_keys.items():
            hull = env.scene.hulls[key]
            vertices = hull.vertices
            # FIXME: figure this out
            radius = np.linalg.norm(vertices, axis=-1).max()
            print('radius', radius)
            ee_object_radius[env_id] = radius
            pass
        self.ee_object_radius = th.as_tensor(ee_object_radius,
                                             device=self.device,
                                             dtype=th.float)

        # Configure randomization and such
        p = np.asarray(env.cfg.scene.table_pos)
        d = np.asarray(env.cfg.scene.table_dims)
        min_bound = p - 0.5 * d
        max_bound = p + 0.5 * d
        # set min_bound also to tabletop.
        min_bound[..., 2] = max_bound[..., 2]
        self.init_lo = th.as_tensor(min_bound, device=self.device,
                                    dtype=th.float)
        self.init_hi = th.as_tensor(max_bound, device=self.device,
                                    dtype=th.float)

    def apply_actions(self, gym, sim, env,
                      actions: th.Tensor, done=None):
        """
        Actions is an implicitly structured tensor
        with the following elements:
        A = (N, (3+3)) where A[i \\in n_{env}]
          = (force_vector, force_point)
        """
        cfg = self.cfg
        if actions is None:
            return

        # TODO:
        # I don't really think introspection into
        # env.buffers() is the ideal solution...
        with nvtx.annotate("a"):
            idx = self.rigid_body_ids
            if env.buffers['done'] is not None:
                mask = ~env.buffers['done']
                if cfg.apply_mask:
                    actions = actions[mask]
                    idx = idx[mask]
                else:
                    actions = mask.float()[..., None] * actions

        if (cfg.apply_mask) and (len(actions) <= 0):
            return

        with nvtx.annotate("b"):
            # indirect wrench from force + point
            # print('actions', actions.shape, actions)
            force_action = actions[..., 0:3]
            torque_action = actions[..., 3:6]

        with nvtx.annotate("c"):
            forces = self.forces
            forces.fill_(0)

            torques = self.torques
            torques.fill_(0)

        with nvtx.annotate("d"):
            forces[th.arange(len(idx)), idx] = (
                force_action
            )
            torques[th.arange(len(idx)), idx] = (
                torque_action
            )

        with nvtx.annotate("e"):
            # directly apply wrench
            out = gym.apply_rigid_body_force_tensors(
                sim,
                gymtorch.unwrap_tensor(forces),
                gymtorch.unwrap_tensor(torques),
                gymapi.ENV_SPACE
            )
        return out
