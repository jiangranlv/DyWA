#!/usr/bin/env python3

from typing import Tuple, Dict, List, Any, Optional, Iterable
from dataclasses import dataclass
from util.config import ConfigBase
import pkg_resources
from pathlib import Path
import json
import os

import numpy as np
from tqdm.auto import tqdm
import trimesh
import itertools

from isaacgym import gymtorch
from isaacgym import gymapi

import torch as th

from cho_util.math import transform as tx

from env.env.base import EnvBase
from env.scene.base import SceneBase
from env.scene.tabletop_scene import TableTopScene
from env.common import create_camera, apply_domain_randomization
from data.transforms.io_xfm import load_mesh
from util.torch_util import dcn

from env.scene.common import load_objects


from isaacgym.gymutil import WireframeBoxGeometry, draw_lines

import nvtx

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')


class TableTopWithMultiObjectScene(TableTopScene):

    @dataclass
    class Config(TableTopScene.Config):
        data_root: str = F'{DATA_ROOT}/ACRONYM/urdf'
        # Convex hull for quickly computing initial placements.
        hull_root: str = F'{DATA_ROOT}/ACRONYM/hull'
        mesh_count: str = F'{DATA_ROOT}/ACRONYM/mesh_count.json'
        urdf_stats_file: str = F'{DATA_ROOT}/ACRONYM/urdf_stats.json'

        use_wall: bool = False
        # wall_table_urdf: str = '../../data/assets/table-with-wall/robot.urdf'
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        # table_file: str = 'table-with-wall/table-with-wall-open.urdf'
        table_file: str = 'table-with-wall/table-with-wall.urdf'

        num_obj_per_env: int = 1
        z_eps: float = 1e-2
        randomize_init_pos: bool = False
        randomize_init_orn: bool = False
        add_force_sensor_to_com: bool = False
        use_dr: bool = False
        use_dr_on_setup: bool = False

        # FIXME: `num_object_types` should usually match num_env.
        num_object_types: int = 512
        max_vertex_count: int = 2500
        max_chull_count: int = 128

        # Randomize table dim
        table_dim_range: Tuple[List[float], List[float]] = (
            (0.3, 0.5, 0.2), (0.4, 1.0, 0.6))

    def __init__(self, cfg: Config):
        self.cfg = cfg

        (self.object_files, self.hull_files) = load_objects(
            cfg.data_root,
            cfg.hull_root,
            cfg.urdf_stats_file,
            False,
            cfg.max_vertex_count,
            cfg.max_chull_count,
            cfg.num_object_types)

        self.keys: List[str] = []
        self.assets: Dict[str, Any] = {}
        self.hulls: Dict[str, Any] = None
        self.sensors = {}

        # WHAT WE NEED:
        # > ACTOR INDEX, for set_actor_root_state_tensor_index
        # > RIGID-BODY INDEX, for apply_rigid_body_force_at_pos_tensors
        self.obj_ids: th.Tensor = None
        self.obj_handles: th.Tensor = None
        self.obj_body_ids: th.Tensor = None

        # Currently "active" objects cache.
        self.cur_mask: th.Tensor = None  # which object is used per env
        self.cur_ids: th.Tensor = None  # which object is used per env
        self.cur_handles: th.Tensor = None
        self.cur_radii: th.Tensor = None
        self.body_ids: th.Tensor = None

        # Initialization scale parameter.
        self._pos_scale: float = 1.0

    def setup(self, env: 'EnvBase'):
        cfg = self.cfg

        # Lookup actor handles and indices.
        obj_ids = []
        obj_handles = []
        obj_body_ids = []
        obj_radii = []
        for i in range(env.num_env):
            for j in range(cfg.num_obj_per_env):
                obj_id = env.gym.find_actor_index(
                    env.envs[i],
                    F'object-{j:02d}',
                    gymapi.IndexDomain.DOMAIN_SIM)
                obj_ids.append(obj_id)

                obj_handle = env.gym.find_actor_handle(
                    env.envs[i],
                    F'object-{j:02d}')
                obj_handles.append(obj_handle)

                obj_body_id = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    obj_handle,
                    'base_link',
                    gymapi.IndexDomain.DOMAIN_ENV
                )
                obj_body_ids.append(obj_body_id)

        # Convert indices and handles to torch tensors.
        self.obj_ids = th.as_tensor(obj_ids,
                                    dtype=th.int32,
                                    device=env.device).reshape(
            env.num_env, cfg.num_obj_per_env)
        self.obj_handles = th.as_tensor(obj_handles,
                                        dtype=th.int32,
                                        device=env.device).reshape(
            env.num_env, cfg.num_obj_per_env)
        self.obj_body_ids = th.as_tensor(obj_body_ids,
                                         dtype=th.int32,
                                         device=env.device).reshape(
            env.num_env, cfg.num_obj_per_env)

        # Also lookup table-related stuff.
        # FIXME: it will be quite nontrivial
        # to figure out the "domains" of these ids
        # in the perspective of the external API...
        self.table_handles = [
            env.gym.find_actor_handle(env.envs[i], 'table')
            for i in range(env.num_env)]
        self.table_body_ids = [
            env.gym.get_actor_rigid_body_index(
                env.envs[i],
                self.table_handles[i],
                0,
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(env.num_env)
        ]

        self.mask = th.zeros(
            env.tensors['root'].shape[0],
            dtype=bool,
            device=env.device
        )
        self.cur_mask = th.zeros(
            (env.num_env, cfg.num_obj_per_env),
            dtype=bool,
            device=env.device
        )
        self.cur_ids = th.zeros(
            (env.num_env,),
            dtype=th.int32,
            device=env.device
        )
        self.cur_handles = th.zeros(
            (env.num_env,),
            dtype=th.int32,
            device=env.device
        )
        self.cur_radii = th.zeros(
            (env.num_env,),
            dtype=th.float,
            device=env.device
        )
        self.body_ids = th.zeros(
            (env.num_env,),
            dtype=th.int32,
            device=env.device
        )

        # Stencil for how many objects will be spawned per env.
        self.stencil = th.zeros(
            (env.num_env, cfg.num_obj_per_env),
            dtype=bool,
            device=env.device
        )
        for i in range(env.num_env):
            m = int(np.random.randint(1, cfg.num_obj_per_env + 1))
            self.stencil[i, :m] = 1

        # Optionally apply domain randomization.
        if cfg.use_dr_on_setup:
            for i in range(env.num_env):
                for j in range(cfg.num_obj_per_env):
                    apply_domain_randomization(
                        env.gym, env.envs[i],
                        self.obj_handles[i, j],
                        enable_mass=True)

    def _get_xy(self, env, device, dtype, n: int,
                env_ids_each):
        if not self.cfg.randomize_init_pos:
            out = th.zeros((n, 2), dtype=dtype, device=device)
            # out[..., 1] = -0.3
            out[..., 1] = 0.0
            return out

        min_bound = th.subtract(
            self.table_pos[env_ids_each, :2],
            th.multiply(0.5, self.table_dims[env_ids_each, :2]))
        max_bound = th.add(
            self.table_pos[env_ids_each, :2],
            th.multiply(0.5, self.table_dims[env_ids_each, :2]))
        mnb = th.as_tensor(min_bound, device=device,
                           dtype=dtype)
        mxb = th.as_tensor(max_bound, device=device,
                           dtype=dtype)
        ctr = 0.5 * (mnb + mxb)
        scale = 0.5 * (mxb - mnb) * min(self._pos_scale, 1.0)

        # ... sample goal ...
        return (th.rand((n, 2), dtype=dtype,
                        device=device) - 0.5) * scale + ctr

    def _select_objects(self, env, env_ids, reset_all: bool):
        """
        Select the set of objects to move to table.
        """
        # <==> SELECT OBJECT TO MOVE TO TABLE <==>
        cfg = self.cfg

        num_reset: int = len(env_ids)

        base_ids = cfg.num_obj_per_env * env_ids

        prv_ids = self.obj_ids[env_ids][self.cur_mask[env_ids]]

        # Select which objects will spawn on the new environment.
        sel_mask = th.zeros(
            (env.num_env, cfg.num_obj_per_env),
            dtype=bool,
            device=env.device
        )
        offsets = th.randint(
            cfg.num_obj_per_env,
            size=(num_reset, cfg.num_obj_per_env)
        )
        targets = offsets[..., 0]
        sel_mask[env_ids[..., None], offsets] = True

        nxt_ids = self.obj_ids[sel_mask]
        nxt_handles = self.obj_handles[sel_mask]

        hulls = [self.hulls[self.keys[i]] for i in dcn(th.argwhere(
            sel_mask.ravel()).ravel())]

        target_ids = (
            env_ids *
            cfg.num_obj_per_env +
            targets.to(
                env_ids.device))
        radii = [self.radii[self.keys[i]]
                 for i in dcn(target_ids.ravel())]

        # if not reset_all:
        #     prv_ids = self.obj_ids[env_ids][self.cur_mask[env_ids]]
        #     offsets = th.randint(cfg.num_obj_per_env,
        #                          size=(num_reset,))
        #     nxt_ids = self.obj_ids[env_ids, offsets]
        #     nxt_handles = self.obj_handles[env_ids, offsets]
        #     nxt_body_ids = self.obj_body_ids[env_ids, offsets]
        #     hulls = [self.hulls[self.keys[i]] for i in
        #              dcn(env_ids) * cfg.num_obj_per_env + dcn(offsets)]
        # else:
        #     if self.cur_mask is None:
        #         prv_ids = None
        #     else:
        #         prv_ids = self.obj_ids[self.cur_mask]
        #     offsets = th.randint(cfg.num_obj_per_env,
        #                          size=(num_reset,))
        #     nxt_ids = self.obj_ids[env_ids, offsets]
        #     nxt_handles = self.obj_handles[env_ids, offsets]
        #     nxt_body_ids = self.obj_body_ids[env_ids, offsets]
        #     hulls = [self.hulls[self.keys[i]] for i in np.arange(
        #         env.num_env) * cfg.num_obj_per_env + dcn(offsets)]
        return (prv_ids, nxt_ids, nxt_handles, hulls, radii, sel_mask,
                targets)

    def _randomize_R_and_z(self, hulls: Iterable[np.ndarray],
                           env_ids_each):
        cfg = self.cfg
        n: int = len(hulls)

        # Generate rotations (q->R)
        if cfg.randomize_init_orn:
            qs = tx.rotation.quaternion.random(
                size=n)
        else:
            qs = np.zeros((n, 4), dtype=np.float32)
            qs[..., 3] = 1
        Rs = tx.rotation.matrix.from_quaternion(qs)

        # [3] Reset nxt objects' poses so that
        # the convex hull rests immediately on the tabletop surface.
        zs = [None for _ in range(n)]
        for ii in range(n):
            p = hulls[ii].vertices
            z = Rs[ii] @ [0, 0, 1]
            dz = (p @ z).min()
            zs[ii] = self.table_dims[env_ids_each[ii], 2] - dz + cfg.z_eps
        zs = th.stack(zs, dim=0)

        return (qs, zs)

    def _randomize_pose(self,
                        env,
                        hulls: Iterable[np.ndarray],
                        indices: th.Tensor,
                        root_tensor: th.Tensor,
                        env_ids_each):
        qs, zs = self._randomize_R_and_z(hulls, env_ids_each)
        root_tensor[indices] = 0
        # TODO: also set (x,y) values !!
        xy = self._get_xy(
            env, root_tensor.device, root_tensor.dtype,
            len(indices),
            env_ids_each
        )
        root_tensor[indices, :2] = xy
        root_tensor[indices, 2] = zs
        root_tensor[indices, 3:7] = th.as_tensor(
            np.asarray(qs), dtype=root_tensor.dtype,
            device=root_tensor.device)

    def _update_indices(self,
                        env_ids: th.Tensor,
                        prv_ids: th.Tensor,
                        nxt_ids: th.Tensor,
                        reset_all: bool
                        ) -> th.Tensor:
        cfg = self.cfg
        with nvtx.annotate("g"):
            # merge pi, ni
            if (cfg.num_obj_per_env > 1):
                mask = self.mask
                mask.fill_(0)
                if (prv_ids is not None):
                    mask[prv_ids] = 1
                mask[nxt_ids] = 1
                set_ids = th.argwhere(mask).ravel().to(
                    dtype=th.int32)
            else:
                set_ids = nxt_ids.to(dtype=th.int32)
        return set_ids

    @nvtx.annotate('Scene.reset()', color="red")
    def reset(self, gym, sim, env,
              env_ids: Optional[Iterable[int]] = None) -> th.Tensor:
        """
        Current reset logic roughly based on:
            https://forums.developer.nvidia.com/t/
            would-it-be-possible-to-destroy-an-actor-
            and-add-a-new-one-during-simulation/169517/2

        What happens during this reset script?
        1. select the object to move to table.
        2. apply domain randomization on target object.
        3. apply camera-pose randomization.
        4. reset (prv/cur) object poses.
        5. Bookkeeping for indices...
        """
        if env_ids is not None and len(env_ids) == 0:
            return env_ids
        cfg = self.cfg
        # Reset object poses and potentially
        # apply domain randomization.
        # set_actor_rigid_body_properties()
        # set_actor_rigid_shape_properties()
        # set_actor_root_state_tensor_indexed()

        with nvtx.annotate("a"):
            reset_all: bool = False
            if env_ids is None:
                env_ids = th.arange(env.num_env)
                reset_all = True
            num_reset: int = len(env_ids)

        # print(F'(obj) resetting = {env_ids}')
        with nvtx.annotate("b"):
            (prv_ids, nxt_ids, nxt_handles, hulls, radii, sel_mask,
             targets) = self._select_objects(env, env_ids, reset_all)

        root_tensor = env.tensors['root']

        # [1] Reset prv objects' poses
        # to arbitrary positions in the environment.
        if (cfg.num_obj_per_env > 1) and (prv_ids is not None):
            pi = prv_ids.long()
            # pos, orn, lin.vel/ang.vel --> 0
            root_tensor[pi] = 0
            # NOTE: "somewhere sufficiently far away"
            root_tensor[pi, 0] = (prv_ids + 1).float() * 100.0
            root_tensor[pi, 2] = 1.0
            root_tensor[pi, 6] = 1

        env_ids_each = th.argwhere(sel_mask)[..., 0]
        self._randomize_pose(env, hulls, nxt_ids.long(),
                             root_tensor, env_ids_each)

        # [2] Commit nxt objects.
        # WARN: may not be correct, like
        # > clearing prv_ids
        # > localizing nxt_ids range to env_ids
        # self.cur_mask[nxt_ids.long()] = True # ???????

        self.cur_mask[env_ids] = 0
        self.cur_mask |= sel_mask
        self.cur_ids[env_ids] = self.obj_ids[env_ids, targets]
        self.cur_handles[env_ids] = self.obj_handles[env_ids, targets]
        self.body_ids[env_ids] = self.obj_body_ids[env_ids, targets]
        radii = th.as_tensor(
            radii,
            dtype=self.cur_radii.dtype,
            device=self.cur_radii.device
        )
        self.cur_radii[env_ids] = radii
        pi = (None if prv_ids is None else prv_ids.long())
        set_ids = self._update_indices(env_ids, pi, nxt_ids.long(),
                                       reset_all)

        return set_ids

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg

        # Sample N objects from the pool.

        # Spawn ground...?

        # Spawn table.
        table_pose = gymapi.Transform()
        # table_pose.p = gymapi.Vec3(*cfg.table_pos)
        table_pose.p = gymapi.Vec3(*self.table_pos[env_id])
        table_pose.r = gymapi.Quat(*cfg.table_orn)

        table_actor = gym.create_actor(
            env, self.assets['table'][env_id],
            table_pose, F'table', env_id,
            0)

        object_actors = []

        keys = self.keys[env_id * cfg.num_obj_per_env:]
        for i, key in enumerate(keys[:cfg.num_obj_per_env]):
            obj_pose = gymapi.Transform()

            # Spawn objects.
            object_actor = gym.create_actor(
                env,
                self.assets['objects'][key],
                obj_pose,
                F'object-{i:02d}',
                env_id,
                # NOTE: I believe `-1` disables self collision
                -1)
            gym.set_rigid_body_segmentation_id(env, object_actor,
                                               0, 1 + i)
            object_actors.append(object_actor)

        return {'table': table_actor,
                'object': object_actors}

    def create_assets(self, gym, sim, env: 'EnvBase'):
        cfg = self.cfg

        # (1) Create table.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        # table_asset = gym.create_box(sim,
        #                              *cfg.table_dims,
        #                              asset_options)
        num_table: int = env.num_env
        self.table_dims = np.random.uniform(low=cfg.table_dim_range[0],
                                            high=cfg.table_dim_range[1],
                                            size=(num_table, 3))
        self.table_pos = np.zeros((num_table, 3))
        self.table_pos[..., 2] = self.table_dims[..., 2] * 0.5

        self.table_dims = th.as_tensor(self.table_dims,
                                       dtype=th.float32,
                                       device=env.device)
        self.table_pos = th.as_tensor(self.table_pos,
                                      dtype=th.float32,
                                      device=env.device)

        table_assets = []
        if cfg.use_wall:
            filename = cfg.table_file
            # asset_options.disable_gravity = False
            asset_options.vhacd_enabled = False
            asset_options.thickness = 0.0001  # ????
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.override_com = True
            asset_options.override_inertia = True
            table_asset = gym.load_urdf(sim,
                                        str(cfg.asset_root),
                                        str(cfg.table_file),
                                        asset_options)
            table_assets = [table_asset] * num_table
        else:
            for i in range(num_table):
                table_asset = gym.create_box(sim,
                                             *self.table_dims[i],
                                             asset_options)
                table_assets.append(table_asset)

        # (3) Create objects.
        # TODO: do we need to load a bunch of
        # diverse-ish objects ??
        # TODO: probably need to make either a
        # dummy URDF file or auto-generate URDF file
        # based on inertial and mass properties (e.g.
        # assuming density = 0.1kg/m^3, domain randomization,
        # ...)
        obj_assets = {}
        force_sensors = {}

        num_load: int = cfg.num_obj_per_env * env.num_env
        object_files = np.random.choice(
            self.object_files,
            size=min(num_load, len(self.object_files)),
            replace=False
        )

        hulls = {}
        radii = {}
        for index, filename in enumerate(
            tqdm(
                object_files,
                desc='create_object_assets')):
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = False
            asset_options.thickness = 0.0001  # ????
            # NOTE: set to `True` since we're directly using
            # the convex decomposition result from CoACD.
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.override_com = False
            asset_options.override_inertia = False
            obj_asset = gym.load_urdf(sim,
                                      str(Path(filename).parent),
                                      str(Path(filename).name),
                                      asset_options)
            # key = F'{index}-{filename}'
            key = filename
            obj_assets[key] = obj_asset
            hull_file = (
                Path(cfg.hull_root) / Path(filename).with_suffix('.glb').name
            )
            hull = load_mesh(hull_file, as_mesh=True)
            hulls[key] = hull
            v = np.asarray(hull.vertices)
            radii[key] = np.linalg.norm(v, axis=-1).max(axis=0)

            # FIXME:
            # this might require the assumption that
            # the center of mass is located at the body origin.
            if cfg.add_force_sensor_to_com:
                props = gymapi.ForceSensorProperties()
                # FIXME: might be slightly unexpected
                props.use_world_frame = True

                # NOTE: should really be disabled !!!
                # props.enable_forward_dynamics_forces = False  # no gravity
                # props.enable_constraint_solver_forces = True  # contacts

                # FIXME: hardcoded `0` does not work
                # if the object is articulated
                force_sensor = gym.create_asset_force_sensor(
                    obj_asset, 0, gymapi.Transform(),
                    props)
                force_sensors[key] = force_sensor

        # (2) Create objects.
        self.assets = {
            'table': table_assets,
            'objects': obj_assets,
            'force_sensors': force_sensors
        }

        self.keys = list(itertools.islice(
            itertools.cycle(list(obj_assets.keys())),
            env.num_env * cfg.num_obj_per_env))

        self.hulls = hulls
        self.radii = radii
        return self.assets

    def create_sensors(self, gym, sim, env, env_id: int):
        return {}


def main():
    scene = TableTopWithMultiObjectScene(
        TableTopWithMultiObjectScene.Config()
    )


if __name__ == '__main__':
    main()
