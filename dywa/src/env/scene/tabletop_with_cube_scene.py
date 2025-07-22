#!/usr/bin/env python3

from typing import Tuple, Optional, Iterable
from dataclasses import dataclass
from util.config import ConfigBase
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from cho_util.math import transform as tx

import torch as th

from env.scene.base import SceneBase
from env.scene.tabletop_scene import TableTopScene
from env.common import create_camera


class TableTopWithCubeScene(TableTopScene):

    @dataclass
    class Config(TableTopScene.Config):
        cube_dims: Tuple[float, float, float] = (0.045, 0.045, 0.045)
        img_shape: Tuple[int, int] = (128, 128)

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self, env: 'EnvBase'):
        self.cube_ids = [env.gym.find_actor_index(
            env.envs[i],
            F'cube',
            # env.actors['scene'][i]['cube'],
            # 'box',
            gymapi.IndexDomain.DOMAIN_SIM)
            for i in range(env.cfg.num_env)
        ]
        self.cube_ids = th.as_tensor(self.cube_ids,
                                     dtype=th.int32,
                                     device=env.cfg.th_device)

    def _sample_cube_poses(self, n: int):
        cfg = self.cfg
        rxn = tx.rotation.matrix.random(size=n)  # Nx3x3
        # dx = 0.5 * cfg.cube_dims[0] * np.abs(rxn[..., 0]).sum(axis=-1)
        # dy = 0.5 * cfg.cube_dims[1] * np.abs(rxn[..., 1]).sum(axis=-1)
        dz = 0.5 * cfg.cube_dims[2] * np.abs(rxn[..., 2]).sum(axis=-1)

        x = cfg.table_pos[0] + 0.5 * \
            np.random.uniform(-cfg.table_dims[0], cfg.table_dims[0], size=n)
        y = cfg.table_pos[1] + 0.5 * \
            np.random.uniform(-cfg.table_dims[1], cfg.table_dims[1], size=n)
        z = cfg.table_pos[2] + 0.5 * cfg.table_dims[2] + dz

        txn = np.stack([x, y, z], axis=-1)
        qxn = tx.rotation.quaternion.from_matrix(rxn)
        # tx.rotation.quaternion.rotate(qxn[:,None],
        return txn, qxn

    def reset(self, gym, sim, env,
              indices: Optional[Iterable[int]] = None):
        # TODO: consider (is it worth it?)
        # domain-randomization on:
        # * table pose
        # * table dimensions
        # * table texture (color)
        # env.tensors['root']

        # print(env.tensors['root'].shape)
        # print(self.cube_ids)
        if indices is not None:
            target_ids = self.cube_ids[indices]
        else:
            target_ids = self.cube_ids
        txn, qxn = self._sample_cube_poses(len(target_ids))

        env.tensors['root'][target_ids.long(), 0:3] = th.as_tensor(
            txn,
            dtype=env.tensors['root'].dtype,
            device=env.tensors['root'].device)
        env.tensors['root'][target_ids.long(), 3:7] = th.as_tensor(
            qxn, dtype=env.tensors['root'].dtype,
            device=env.tensors['root'].device)

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(env.tensors['root']),
            gymtorch.unwrap_tensor(target_ids),
            len(target_ids)
        )

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg
        # Spawn table.
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*cfg.table_pos)
        table_pose.r = gymapi.Quat(*cfg.table_orn)
        table_actor = gym.create_actor(
            env, self.assets['table'],
            table_pose, F'table', env_id, 0)

        cube_pose = gymapi.Transform()
        z_cube = (cfg.table_pos[2]
                  + 0.5 * cfg.table_dims[2]
                  + 0.5 * cfg.cube_dims[2])
        cube_pose.p = gymapi.Vec3(
            cfg.table_pos[0],
            cfg.table_pos[1],
            z_cube)
        cube_pose.r = gymapi.Quat()
        cube_actor = gym.create_actor(
            env, self.assets['cube'],
            cube_pose, F'cube', env_id, 0)
        gym.set_rigid_body_color(env,
                                 cube_actor, 0, gymapi.MESH_VISUAL,
                                 gymapi.Vec3(0.7, 0.7, 0.3))

        return {'table': table_actor, 'cube': cube_actor}

    def create_assets(self, gym, sim):
        cfg = self.cfg
        # Create table asset.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = gym.create_box(sim,
                                     *cfg.table_dims,
                                     asset_options)

        asset_options = gymapi.AssetOptions()
        cube_asset = gym.create_box(sim,
                                    *cfg.cube_dims,
                                    asset_options)

        self.assets = {'table': table_asset, 'cube': cube_asset}
        return dict(self.assets)

    def create_sensors(self, gym, sim, env, env_id: int):
        cfg = self.cfg
        camera, img_tensors = create_camera(
            cfg.img_shape[0], cfg.img_shape[1],
            gym, sim, env)

        # Initialize at top-down
        gym.set_camera_location(
            camera, env, gymapi.Vec3(
                # FIXME:
                # looking straight down
                # is apparently not possible.
                cfg.table_pos[0] + 0.01,
                cfg.table_pos[1] + 0.01,
                cfg.table_pos[2] + 1.0),
            gymapi.Vec3(*cfg.table_pos))
        return {'camera': camera, 'tensors': img_tensors}
