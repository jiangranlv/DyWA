#!/usr/bin/env python3

from typing import Tuple, Dict, List, Any, Optional, Iterable
from dataclasses import dataclass, replace
from util.config import ConfigBase
import pkg_resources
from pathlib import Path
import json
import os
import pickle
from functools import partial
from tempfile import TemporaryDirectory

import numpy as np
import einops
from tqdm.auto import tqdm
import trimesh
import itertools
import re
import math
import random

from isaacgym import gymtorch
from isaacgym import gymapi

import torch as th

from cho_util.math import transform as tx

from env.env.base import EnvBase
from env.scene.base import SceneBase
from env.scene.tabletop_scene import TableTopScene
from env.robot.fe_gripper import FEGripper
from env.common import (
    create_camera, apply_domain_randomization, set_actor_friction,
    aggregate, set_actor_restitution)
from data.transforms.io_xfm import load_mesh
from util.torch_util import dcn
from env.task.util import (sample_goal_v2 as sample_goal,
                               sample_yaw)
from util.math_util import quat_multiply

from isaacgym.gymutil import WireframeBoxGeometry, draw_lines
from isaacgym.torch_utils import quat_from_euler_xyz

from env.scene.object_set import ObjectSet
from env.scene.acronym_object_set import AcronymObjectSet
from env.scene.cuboid_object_set import CuboidObjectSet
from env.scene.dgn_object_set import DGNObjectSet
from env.scene.cone_object_set import ConeObjectSet
from env.scene.cylinder_object_set import CylinderObjectSet
from env.scene.prism_object_set import PrismObjectSet
from env.scene.mesh_object_set import MeshObjectSet
from env.scene.filter_object_set import FilteredObjectSet, FilterDims
from env.scene.combine_object_set import CombinedObjectSet
from env.scene.scenario_object_set import ScenarioObjectSet
from env.scene.util import create_bin, _is_stable

from models.common import merge_shapes

import nvtx
from icecream import ic
from tempfile import mkdtemp

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')


def is_thin(extent, threshold: float = 2.5):
    size = np.sort(extent)
    return (size[1] >= threshold * size[0])


def sample_cuboid_poses(n: int, noise: float = np.deg2rad(5.0)):
    IRT2 = math.sqrt(1.0 / 2)
    canonicals = np.asarray([
        [0.000, 0.000, 0.000, 1.000],
        [1.000, 0.000, 0.000, 0.000],
        [-IRT2, 0.000, 0.000, +IRT2],
        [+IRT2, 0.000, 0.000, +IRT2],
        [0.000, -IRT2, 0.000, +IRT2],
        [0.000, +IRT2, 0.000, +IRT2],
    ], dtype=np.float32)
    indices = np.random.choice(
        len(canonicals),
        size=n)
    qs = canonicals[indices]

    # Add slight noise to break symmetry
    # in case of highly unstable configurations
    qzs = tx.rotation.axis_angle.random(size=n)
    # qzs[..., 3] *= noise
    qzs[..., 3] = np.random.uniform(-noise, +noise,
                                    size=qzs[..., 3].shape)
    qzs = tx.rotation.quaternion.from_axis_angle(qzs)
    qs = tx.rotation.quaternion.multiply(qzs, qs)

    return qs


def _pad_hulls(hulls: Dict[str, trimesh.Trimesh]) -> Dict[str, np.ndarray]:
    n: int = max([len(h.vertices) for h in hulls.values()])
    out = {}
    for k, h in hulls.items():
        p = np.empty((n, 3), dtype=np.float32)
        v = h.vertices
        c = np.mean(v, axis=0, keepdims=True)
        p[:len(v)] = v
        p[len(v):] = c
        out[k] = p
    return out


def _array_from_map(
        keys: List[str],
        maps: Dict[str, th.Tensor],
        **kwds):
    if not isinstance(next(iter(maps.values())), th.Tensor):
        arr = np.stack([maps[k] for k in keys])
        return th.as_tensor(arr, **kwds)
    return th.stack([maps[k] for k in keys], dim=0).to(**kwds)


class TableTopWithObjectScene(TableTopScene):

    @dataclass
    class Config(TableTopScene.Config):
        data_root: str = F'{DATA_ROOT}/ACRONYM/urdf'
        # Convex hull for quickly computing initial placements.
        hull_root: str = F'{DATA_ROOT}/ACRONYM/hull'
        mesh_count: str = F'{DATA_ROOT}/ACRONYM/mesh_count.json'
        urdf_stats_file: str = F'{DATA_ROOT}/ACRONYM/urdf_stats.json'
        stable_poses_file: str = F'{DATA_ROOT}/ACRONYM/stable_poses.pkl'
        # stable_poses_file: str = F'/input/ACRONYM/train_chair_poses.pkl'
        embeddings_file: str = F'{DATA_ROOT}/ACRONYM/embedding.pkl'
        patch_center_file: str = '/input/ACRONYM/patch-v12.pkl'
        bbox_file: str = F'{DATA_ROOT}/ACRONYM/bbox.pkl'
        cloud_file: str = F'{DATA_ROOT}/ACRONYM/cloud.pkl'
        cloud_normal_file: str = F'{DATA_ROOT}/ACRONYM/cloud_normal.pkl'
        volume_file: str = F'{DATA_ROOT}/ACRONYM/volume.pkl'

        stable_poses_url: Optional[str] = None
        embeddings_url: Optional[str] = None
        bbox_url: Optional[str] = None
        cloud_url: Optional[str] = None
        cloud_normal_url: Optional[str] = None
        volume_url: Optional[str] = None

        use_wall: bool = False
        use_bin: bool = False
        # wall_table_urdf: str = '../../data/assets/table-with-wall/robot.urdf'
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        # table_file: str = 'table-with-wall/table-with-wall-open.urdf'
        # table_file: str = 'table-with-wall/table-with-wall.urdf'
        table_file: str = 'table-with-wall/table-with-wall-small.urdf'

        table_friction: Optional[float] = None
        object_friction: Optional[float] = None

        num_obj_per_env: int = 1
        z_eps: float = 1e-2

        # Or "sample", "zero", ...
        init_type: str = 'sample'
        goal_type: str = 'random'
        randomize_yaw: bool = False

        randomize_init_pos: bool = False
        randomize_init_orn: bool = False
        # In order to increase the likelihood of sampling dynamically
        # unstable goal configurations, sample init pose of dropping
        # as identity with given probability
        canonical_pose_prob: float = 0.2  # need to be randomize_init_orn True

        add_force_sensor_to_com: bool = False
        avoid_overlap: bool = True

        use_dr: bool = False
        use_dr_on_setup: bool = False
        use_mass_set: bool = False
        min_mass: float = 0.01
        max_mass: float = 2.0
        mass_set: Tuple[float, ...] = (0.1, 1.0, 10.0)
        use_scale_dr: bool = False
        min_scale: float = 0.09
        max_scale: float = 0.15

        min_object_friction: float = 0.2
        max_object_friction: float = 1.0
        min_table_friction: float = 0.6
        max_table_friction: float = 1.0
        min_object_restitution: float = 0.0
        max_object_restitution: float = 0.2

        # Old setting, load one single object
        diverse_object: bool = False

        # `filter_index` is only used during stable_poses generation...
        filter_index: Optional[Tuple[int, ...]] = None
        # `filter_class` to select categories...
        filter_class: Optional[Tuple[str, ...]] = None
        # `filter_key` to select specific objects ...
        filter_key: Optional[Tuple[str, ...]] = None
        # `filter_file` to select specific objects from file...
        filter_file: Optional[str] = None
        filter_complex: bool = True
        filter_dims: Optional[Tuple[float, float, float]] = None
        filter_pose_count: Optional[Tuple[int, int]] = None
        truncate_pose_count: int = 64

        # keys or file that contains key for yaw only objects
        yaw_only_key: Optional[Tuple[str, ...]] = None
        yaw_only_file: Optional[str] = None
        use_yaw_only_logic: bool = True
        thin_threshold: float = 2.5

        load_embedding: bool = False
        load_bbox: bool = False
        load_obb: bool = False
        load_cloud: bool = False
        load_normal: bool = False
        load_patch_centers: bool = False
        load_predefined_goal: bool = False
        load_stable_mask: bool = True

        # default_mesh: str = 'Speaker_64058330533509d1d747b49524a1246e_0.003949258269301651.glb'
        default_mesh: str = 'RubiksCube_d7d3dc14748ec6d347cd142fcccd1cc2_8.634340549903529e-05.glb'
        # default_mesh: str = 'RubiksCube_cdda3ea70d829d5baa9ba8e71ae84fd3_0.02768212782072632.glb'
        # default_mesh: str = 'RubiksCube_d060362a42a3ef0af0a72b757e578a97_0.05059491278967159.glb'

        # FIXME: `num_object_types` should usually match num_env.
        num_object_types: int = 512
        # max_vertex_count: int = 2500
        max_vertex_count: int = 8192
        max_chull_count: int = 128
        margin_scale: float = 0.95
        prevent_fall: bool = True

        override_cube: bool = False
        load_convex: bool = False
        override_inertia: bool = False
        density: float = 200.0
        target_mass: Optional[float] = None

        base_set: Tuple[str, ...] = ('acronym',)
        acronym: AcronymObjectSet.Config = AcronymObjectSet.Config()
        cuboid: CuboidObjectSet.Config = CuboidObjectSet.Config()
        cone: ConeObjectSet.Config = ConeObjectSet.Config()
        cylinder: CylinderObjectSet.Config = CylinderObjectSet.Config()
        prism: PrismObjectSet.Config = PrismObjectSet.Config()
        dgn: DGNObjectSet.Config = DGNObjectSet.Config()
        scenario: ScenarioObjectSet.Config = ScenarioObjectSet.Config()
        mesh: MeshObjectSet.Config = MeshObjectSet.Config()
        need_attr: Optional[Tuple[str, ...]] = ('num_verts',)

        mode: str = 'train'
        num_valid_poses: int = 1

        restitution: Optional[float] = None

        def __post_init__(self):
            if self.filter_dims is not None:
                d_min, d_max, r_max = self.filter_dims
                self.cuboid = replace(self.cuboid,
                                      min_dim=d_min,
                                      max_dim=d_max,
                                      max_aspect=r_max)
                self.cone = replace(self.cone,
                                    min_dim=d_min,
                                    max_dim=d_max,
                                    max_aspect=r_max)
                self.cylinder = replace(self.cylinder,
                                        min_dim=d_min,
                                        max_dim=d_max,
                                        max_aspect=r_max)
                self.prism = replace(self.prism,
                                     min_dim=d_min,
                                     max_dim=d_max,
                                     max_aspect=r_max)

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # cfg.filter_index,
        self.meta = self.load_obj_set()

        if cfg.override_cube:
            assert (cfg.goal_type != 'stable')
            self.object_embeddings = None  # {'cube': None}
            from torch.distributions import Uniform
            # FIXME : Hardcorded init
            self.box_dim = (0.09, 0.09, 0.09)
            safety_margin = 0.01
            xmin = -0.2
            xmax = 0.2
            self.bboxes = {'cube': trimesh.bounds.corners(
                trimesh.creation.box(self.box_dim).bounds)}
            self.cloud = {'cube':
                          trimesh.creation.box(
                              self.box_dim).vertices}
            self.normal = {'cube':
                           trimesh.creation.box(
                               self.box_dim).vertex_normals}
            self.stable_poses = {'cube': None}
            allowlist = ['cube']
            self.object_files = ['cube']
            self.hull_files = ['cube']
            boxLength = np.sqrt(self.box_dim[0]**2 + self.box_dim[1]**2) / 2
            self.xsampler = Uniform((boxLength + xmin), (xmax - boxLength))
            self.ysampler = Uniform(
                low=(-cfg.table_dims[1] / 2 + 2 * boxLength + safety_margin),
                high=(cfg.table_dims[1] / 2 - 2 * boxLength - safety_margin))
            self.Gysampler = Uniform(
                low=(-cfg.table_dims[1] / 2 + 2 * boxLength),
                high=(cfg.table_dims[1] / 2 - 2 * boxLength))

        self.keys: List[str] = []
        self.assets: Dict[str, Any] = {}
        self.hulls: th.Tensor = None
        self.radii: Dict[str, float] = None
        self.sensors = {}

        # WHAT WE NEED:
        # > ACTOR INDEX, for set_actor_root_state_tensor_index
        # > RIGID-BODY INDEX, for apply_rigid_body_force_at_pos_tensors
        self.obj_ids: th.Tensor = None
        self.obj_handles: th.Tensor = None
        self.obj_body_ids: th.Tensor = None
        self.cur_ids: th.Tensor = None
        self.cur_handles: th.Tensor = None
        self.cur_radii: th.Tensor = None
        self.cur_stable_poses: th.Tensor = None
        self.cur_stable_masks: th.Tensor = None
        self.cur_embeddings: th.Tensor = None
        self.cur_patch_centers: th.Tensor = None
        self.cur_bboxes: th.Tensor = None
        self.cur_extent: th.Tensor = None
        self.cur_names: List[str] = None
        self.cur_cloud: th.Tensor = None
        self.cur_normal: th.Tensor = None
        self.cur_predefined_goals: th.Tensor = None
        self.cur_object_friction: th.Tensor = None
        self.cur_table_friction: th.Tensor = None
        self.body_ids: th.Tensor = None
        self._pos_scale: float = 1.0
        self.table_pos: th.Tensor = None
        self.table_dims: th.Tensor = None

        self.is_yaw_only: th.Tensor = None
        self.cur_yaw_only: th.Tensor = None

        self._per_env_offsets: th.Tensor = None

    def load_obj_set(self) -> ObjectSet:
        cfg = self.cfg
        allowlist = None

        assert (len(cfg.base_set) > 0)

        obj_sets = []
        for base_set in cfg.base_set:
            if base_set == 'acronym':
                meta = AcronymObjectSet(cfg.acronym)
            elif base_set == 'cuboid':
                meta = CuboidObjectSet(cfg.cuboid)
            elif base_set == 'cone':
                meta = ConeObjectSet(cfg.cone)
            elif base_set == 'cylinder':
                meta = CylinderObjectSet(cfg.cylinder)
            elif base_set == 'prism':
                meta = PrismObjectSet(cfg.prism)
            elif base_set == 'dgn':
                meta = DGNObjectSet(cfg.dgn)
            elif base_set == 'scenario':
                meta = ScenarioObjectSet(cfg.scenario)
            elif base_set == 'mesh':
                meta = MeshObjectSet(cfg.mesh)
            else:
                raise KeyError(F'Unknown base object set = {cfg.base_set}')
            keys = meta.keys()
            print(F'init : {len(keys)}')
            # if cfg.load_patch_centers:
            #    with open(cfg.patch_center_file, 'rb') as fp:
            #        patch_centers = pickle.load(fp)
            #    patch_centers = {k: 0.1 * v for k, v in patch_centers.items()}
            #    # FIXME: hardcoded omission of _sanitize
            #    # patch_centers = _sanitize(patch_centers)

            # First, filter by availability of all required fields.
            # Determine required attributes...
            need_attr = list(cfg.need_attr)
            if need_attr is None:
                need_attr = []
            if cfg.goal_type == 'stable':
                need_attr.append('pose')
            for attr in need_attr:
                query = getattr(meta, attr)
                fkeys = []
                for key in keys:
                    try:
                        if query(key) is None:
                            continue
                    except KeyError:
                        continue
                    fkeys.append(key)
                keys = fkeys
            print(F'after filter by "need" : {len(keys)}')

            # Filter by `filter_class`.
            if cfg.filter_class is not None:
                keys = [key for key in keys
                        if meta.label(key) in cfg.filter_class]
            print(F'after filter by "class" : {len(keys)}')

            if cfg.filter_key is not None:
                keys = [key for key in keys
                        if key in cfg.filter_key]
            print(F'after filter by "filter_key" : {len(keys)}')

            if cfg.filter_file is not None:
                with open(cfg.filter_file, 'r') as fp:
                    allowlist = [str(s) for s in json.load(fp)]
                keys = [key for key in allowlist if key in keys]
                # keys = [key for key in keys
                #         if key in allowlist]
            print(F'after filter by "filter_file" : {len(keys)}')
            print('keys:', keys)

            # Filter by size, and remove degenerate mesh
            if cfg.filter_complex:
                keys = [key for key in keys if
                        (meta.num_verts(key) < cfg.max_vertex_count and
                         meta.num_hulls(key) < cfg.max_chull_count and
                         key != '4Shelves_fd0fd7b2c19cce39e3783ec57dd5d298_0.001818238917007018')
                        ]
            print(F'after filter by "complex" : {len(keys)}')

            if cfg.filter_dims is not None:
                d_min, d_max, r_max = cfg.filter_dims
                f = FilterDims(d_min, d_max, r_max)
                keys = [key for key in keys if f(meta, key)]
            print(F'after filter by "dims" : {len(keys)}')

            # Filter by size, and remove degenerate mesh
            if cfg.filter_pose_count:
                pmin, pmax = cfg.filter_pose_count
                keys = [key for key in keys if
                        meta.pose(key) is not None
                        and pmin <= meta.pose(key).shape[0]
                        and meta.pose(key).shape[0] < pmax
                        ]
            print(F'after filter by "pose_count" : {len(keys)}')

            obj_sets.append(FilteredObjectSet(meta, keys=keys))
        if len(obj_sets) == 1:
            return obj_sets[0]
        return CombinedObjectSet(obj_sets)

    def setup(self, env: 'EnvBase'):
        cfg = self.cfg

        obj_ids = []
        obj_handles = []
        obj_body_ids = []
        self.tmpdir = None
        num_env: int = env.num_env

        for i in range(num_env):
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

        # actor indices
        self.obj_ids = th.as_tensor(obj_ids,
                                    dtype=th.int32,
                                    device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        # actor handles
        self.obj_handles = th.as_tensor(obj_handles,
                                        dtype=th.int32,
                                        device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)
        self.obj_body_ids = th.as_tensor(obj_body_ids,
                                         dtype=th.int32,
                                         device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        self.table_handles = [
            env.gym.find_actor_handle(env.envs[i], 'table')
            for i in range(env.cfg.num_env)]

        # FIXME: it will be quite nontrivial
        # to figure out the domains of these ids
        # in the perspective of the external API...
        self.table_body_ids = [
            env.gym.get_actor_rigid_body_index(
                env.envs[i],
                self.table_handles[i],
                0,
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(env.cfg.num_env)
        ]

        self.mask = th.zeros(
            env.tensors['root'].shape[0],
            dtype=bool,
            device=env.cfg.th_device
        )
        self.scales = []
        self.cur_object_friction = (
            th.empty(size=(num_env, cfg.num_obj_per_env),
                     device=env.device)
            .uniform_(cfg.min_object_friction, cfg.max_object_friction))
        self.cur_table_friction = (
            th.empty(size=(num_env,), device=env.device)
            .uniform_(cfg.min_table_friction, cfg.max_table_friction)
        )
        self.cur_object_restitution = (
            th.empty(size=(num_env, cfg.num_obj_per_env), device=env.device)
            .uniform_(cfg.min_object_restitution, cfg.max_object_restitution)
        )
        if cfg.use_dr_on_setup:
            # TODO: dcn() necessary??
            obj_fr = dcn(self.cur_object_friction)
            tbl_fr = dcn(self.cur_table_friction)
            obj_rs = dcn(self.cur_object_restitution)

            # Apply domain randomization.
            for i in range(env.cfg.num_env):
                apply_domain_randomization(
                    env.gym, env.envs[i],
                    self.table_handles[i],
                    enable_friction=True,
                    min_friction=tbl_fr[i],
                    max_friction=tbl_fr[i])
                for j in range(cfg.num_obj_per_env):
                    out = apply_domain_randomization(
                        env.gym, env.envs[i],
                        self.obj_handles[i, j],

                        enable_mass=True,
                        min_mass=cfg.min_mass,
                        max_mass=cfg.max_mass,
                        use_mass_set=cfg.use_mass_set,
                        mass_set=cfg.mass_set,

                        change_scale=cfg.use_scale_dr,
                        min_scale=cfg.min_scale,
                        max_scale=cfg.max_scale,
                        radius=self.radii[i],

                        enable_friction=True,
                        min_friction=obj_fr[i, j],
                        max_friction=obj_fr[i, j],

                        enable_restitution=True,
                        min_restitution=obj_rs[i, j],
                        max_restitution=obj_rs[i, j]
                    )
                    if 'scale' in out:
                        self.scales.append(out['scale'])
        if cfg.restitution is not None:
            for i in range(env.cfg.num_env):
                for j in range(cfg.num_obj_per_env):
                    set_actor_restitution(env.gym,
                                          env.envs[i],
                                          self.obj_handles[i, j],
                                          restitution=cfg.restitution)

        if len(self.scales) > 0:
            from xml.dom import minidom
            if self.tmpdir is None:
                self.tmpdir = mkdtemp()
            for idx, scale in enumerate(self.scales):
                key = self.keys[idx]
                urdf = self.meta.urdf(key)
                with open(urdf, 'r', encoding='utf-8') as f:
                    str_urdf = f.read()
                dom = minidom.parseString(str_urdf)
                meshes = dom.getElementsByTagName("mesh")
                for mesh in meshes:
                    mesh_scales = mesh.attributes['scale'].value.split(' ')
                    new_scale = [str(scale.item() * float(mesh_scale))
                                 for mesh_scale in mesh_scales]
                    mesh.attributes['scale'].value = \
                        ' '.join(new_scale)
                with open(f'{self.tmpdir}/{key}.urdf', "w") as f:
                    dom.writexml(f)
            self.scales = th.as_tensor(self.scales,
                                       device=env.device)
            self.radii = self.radii * self.scales
            self.hulls = self.hulls * self.scales[:, None, None]
            if cfg.load_bbox:
                self.bboxes[..., :] = (self.bboxes[..., :] *
                                       self.scales[:, None, None])
            if cfg.load_cloud:
                self.cloud[..., :] = (self.cloud[..., :] *
                                      self.scales[:, None, None])
            if cfg.goal_type == 'stable':
                poses = self.stable_poses.clone()
                table_h = cfg.table_dims[-1]
                poses[..., 2] -= table_h
                poses[..., 2] = poses[..., 2] * self.scales[:, None]
                self.stable_poses[..., 2] = poses[..., 2] + table_h
        # Validation.
        # masses = np.zeros(env.cfg.num_env)
        # for i in range(env.cfg.num_env):
        #     prop = env.gym.get_actor_rigid_body_properties(
        #         env.envs[i],
        #         self.obj_handles[i, 0].item()
        #     )
        #     masses[i] = prop[0].mass
        # print(F'masses = {masses}')
        self.table_pos = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_pos),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)
        self.table_dims = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_dims),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)
        self._per_env_offsets = th.zeros((env.num_env),
                                         dtype=th.long,
                                         device=env.device)

    def _get_xy(self, env, device, dtype, n: int,
                env_ids,
                prevent_fall: bool = False, avoid_overlap: bool = False):
        cfg = self.cfg
        shape = merge_shapes(n, 2)

        if not cfg.randomize_init_pos:
            if cfg.init_type == 'zero':
                out = th.zeros(shape, dtype=dtype, device=device)
            elif cfg.init_type == 'easy':
                out = th.zeros(shape, dtype=dtype, device=device)
                out[..., 1] = +0.35
            else:
                raise ValueError(F'Unknown init type = {cfg.init_type}')
            return out

        center = self.table_pos[env_ids, :2]
        min_bound = th.subtract(center, th.multiply(
            0.5 * cfg.margin_scale, self.table_dims[env_ids, :2]))
        max_bound = th.add(center, th.multiply(
            0.5 * cfg.margin_scale, self.table_dims[env_ids, :2]))
        # print(min_bound.shape)
        # print(max_bound.shape)

        if prevent_fall:
            # FIXME:
            # this code _may_ result in
            # min_bound >= max_bound !
            radius = self.cur_radii[env_ids]
            min_bound[..., :2] += radius[..., None]
            max_bound[..., :2] -= radius[..., None]
            mask = (min_bound >= max_bound).any(dim=-1, keepdim=False)
            min_bound[mask] = center[mask]
            max_bound[mask] = center[mask]

        if avoid_overlap:
            assert (env.cfg.reset_order == 'robot+scene')
            bound = th.stack([min_bound, max_bound], dim=-2)
            if isinstance(env.robot, FEGripper):
                robot_ids = env.robot.actor_ids.long()
                keepout_center = env.tensors['root'][
                    robot_ids[env_ids], ..., :2]
            else:
                # franka
                body_tensors = env.tensors['body']
                hand_ids = env.robot.ee_body_indices.long()
                eef_pose = body_tensors[hand_ids, :]
                keepout_center = eef_pose[env_ids, ..., :2]
            # FIXME: assumes `robot_radius` is scalar
            keepout_radius = env.robot.robot_radius
            return sample_goal(bound,
                               keepout_center,
                               keepout_radius,
                               z=None,
                               num_samples=16)
        else:
            center = 0.5 * (min_bound + max_bound)
            scale = 0.5 * (max_bound - min_bound) * min(self._pos_scale, 1.0)

            # ... sample goal ...
            return (th.rand(shape, dtype=dtype,
                            device=device) - 0.5) * scale + center

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
        cfg = self.cfg
        # Reset object poses and potentially
        # apply domain randomization.
        # set_actor_rigid_body_properties()
        # set_actor_rigid_shape_properties()
        # set_actor_root_state_tensor_indexed()

        with nvtx.annotate("a"):
            reset_all: bool = False
            if env_ids is None:
                env_ids = th.arange(env.num_env, device=env.device)
                reset_all = True
            num_reset: int = len(env_ids)

        # print(F'(obj) resetting = {env_ids}')

        with nvtx.annotate("b"):
            # <==> SELECT OBJECT TO MOVE TO TABLE <==>
            if reset_all:
                if self.cur_ids is not None:
                    prv_ids = self.cur_ids.clone()
                else:
                    prv_ids = None
            else:
                prv_ids = self.cur_ids[env_ids]

            if cfg.mode == 'valid':
                # Cycle through objects in deterministic order.
                self._per_env_offsets[env_ids] += 1
                # self._per_env_offsets[env_ids] %= cfg.num_obj_per_env
                offsets = self._per_env_offsets[env_ids] % cfg.num_obj_per_env
            else:
                offsets = th.randint(cfg.num_obj_per_env,
                                     size=(num_reset,),
                                     device=env.device)

            nxt_ids = self.obj_ids[env_ids, offsets]
            nxt_handles = self.obj_handles[env_ids, offsets]
            nxt_body_ids = self.obj_body_ids[env_ids, offsets]
            ni = nxt_ids.long()

            # lookup-table indices
            lut_indices = env_ids * cfg.num_obj_per_env + offsets
            lut_indices_np = dcn(lut_indices)

            nxt_names = [self.keys[i] for i in lut_indices_np]
            nxt_radii = self.radii[lut_indices]
            nxt_hulls = self.hulls[lut_indices]
            if cfg.goal_type == 'stable':
                nxt_stable_poses = self.stable_poses[lut_indices]
                if self.is_yaw_only is not None:
                    nxt_yaw_indicates = self.is_yaw_only[lut_indices]
                if cfg.load_stable_mask:
                    nxt_stable_masks = self.stable_masks[lut_indices]
            if cfg.load_embedding:
                nxt_embeddings = self.object_embeddings[lut_indices]
                # if self.patch_centers is not None:
                #    nxt_patch_centers = self.patch_centers[lut_indices]
            if cfg.load_bbox:
                nxt_bboxes = self.bboxes[lut_indices]
            if cfg.load_cloud:
                nxt_cloud = self.cloud[lut_indices]
                if cfg.load_normal:
                    nxt_normal = self.normal[lut_indices]
            if cfg.load_predefined_goal:
                nxt_predefined_goal = self.predefined_goal[lut_indices]

            if cfg.use_dr:
                nxt_object_friction = (
                    th.empty(
                        size=(num_reset, cfg.num_obj_per_env),
                        device=env.device)
                    .uniform_(cfg.min_object_friction,
                              cfg.max_object_friction))
                nxt_table_friction = (
                    th.empty(size=(num_reset,), device=env.device)
                    .uniform_(cfg.min_table_friction, cfg.max_table_friction)
                )
                nxt_object_restitution = (
                    th.empty(
                        size=(num_reset, cfg.num_obj_per_env),
                        device=env.device)
                    .uniform_(cfg.min_object_restitution,
                              cfg.max_object_restitution))

        # <==> DOMAIN RANDOMIZATION <==>
        if cfg.use_dr:
            # one per env
            tbl_fr = dcn(nxt_table_friction).ravel()
            # maybe many per env
            obj_frs = dcn(nxt_object_friction).ravel()
            obj_rss = dcn(nxt_object_restitution).ravel()
            with nvtx.annotate("c"):
                for i, (env_id, obj_handle, obj_fr, obj_rs) in enumerate(
                        zip(env_ids, nxt_handles, obj_frs, obj_rss)):
                    dr_params = apply_domain_randomization(
                        gym, env.envs[int(env_id)], obj_handle,
                        enable_friction=True,
                        min_friction=obj_fr,
                        max_friction=obj_fr,
                        enable_restitution=True,
                        min_restitution=obj_rs,
                        max_restitution=obj_rs,
                    )
                    dr_params = apply_domain_randomization(
                        gym, env.envs[int(env_id)],
                        self.table_handles[int(env_id)],
                        enable_friction=True,
                        min_friction=tbl_fr[i],
                        max_friction=tbl_fr[i]
                    )
                # env.refresh_tensors()

        root_tensor = env.tensors['root']

        # print('Q')
        # print((~th.isfinite(root_tensor)).sum())

        with nvtx.annotate("d"):
            # [1] Reset prv objects' poses
            # to arbitrary positions in the environment.
            if (cfg.num_obj_per_env > 1) and (prv_ids is not None):
                pi = prv_ids.long()
                if True:
                    # pos, orn, lin.vel/ang.vel --> 0
                    root_tensor[pi] = 0
                    # NOTE: "somewhere sufficiently far away"
                    root_tensor[pi, 0] = (prv_ids + 1).float() * 100.0
                    root_tensor[pi, 2] = 1.0
                    # (0,1,2), (3,4,5,6)
                    # Set orientation to unit quaternion
                    root_tensor[pi, 6] = 1

        with nvtx.annotate("e"):

            # [2] Commit nxt objects.
            if not reset_all:
                self.cur_ids[env_ids] = nxt_ids
            else:
                self.cur_ids = nxt_ids

            if not reset_all:
                self.cur_handles[env_ids] = nxt_handles
                self.cur_radii[env_ids] = nxt_radii
                if cfg.goal_type == 'stable' and nxt_stable_poses.shape[0] > 0:
                    self.cur_stable_poses[env_ids] = nxt_stable_poses
                    if self.is_yaw_only is not None:
                        self.cur_yaw_only[env_ids] = nxt_yaw_indicates
                    if cfg.load_stable_mask:
                        self.cur_stable_masks[env_ids] = nxt_stable_masks

                if cfg.load_embedding and nxt_embeddings.shape[0] > 0:
                    self.cur_embeddings[env_ids] = nxt_embeddings
                    # if self.patch_centers is not None:
                    #     self.cur_patch_centers[env_ids] = nxt_patch_centers

                if cfg.load_bbox and nxt_bboxes.shape[0] > 0:
                    self.cur_bboxes[env_ids] = nxt_bboxes

                if cfg.load_cloud and nxt_cloud.shape[0] > 0:
                    self.cur_cloud[env_ids] = nxt_cloud
                    if cfg.load_normal:
                        self.cur_normal[env_ids] = nxt_normal

                if (cfg.load_predefined_goal and
                        nxt_predefined_goal.shape[0] > 0):
                    self.cur_predefined_goals[env_ids] = \
                        nxt_predefined_goal

                if cfg.use_dr:
                    self.cur_object_friction[env_ids] = nxt_object_friction
                    self.cur_table_friction[env_ids] = nxt_table_friction
                    self.cur_object_restitution[env_ids] = nxt_object_restitution

                for i, j in enumerate(dcn(env_ids)):
                    self.cur_names[j] = nxt_names[i]

            else:
                self.cur_handles = nxt_handles
                self.cur_radii = nxt_radii
                if cfg.goal_type == 'stable':
                    self.cur_stable_poses = nxt_stable_poses
                    if self.is_yaw_only is not None:
                        self.cur_yaw_only = nxt_yaw_indicates
                    if cfg.load_stable_mask:
                        self.cur_stable_masks = nxt_stable_masks
                if cfg.load_embedding:
                    self.cur_embeddings = nxt_embeddings
                    # if self.patch_centers is not None:
                    #     self.cur_patch_centers = nxt_patch_centers
                if cfg.load_bbox:
                    self.cur_bboxes = nxt_bboxes
                if cfg.load_cloud:
                    self.cur_cloud = nxt_cloud
                    if cfg.load_normal:
                        self.cur_normal = nxt_normal

                if cfg.use_dr:
                    self.cur_object_friction = nxt_object_friction
                    self.cur_table_friction = nxt_table_friction
                    self.cur_object_restitution = nxt_object_restitution

                if cfg.load_predefined_goal:
                    self.cur_predefined_goals = nxt_predefined_goal

                self.cur_names = nxt_names

            # [3] Reset nxt objects' poses so that
            # the convex hull rests immediately on the tabletop surface.
            # NOTE: Only activated if init_type is not stable,
            # in which we need to use the z value from precomputed
            # stable poses.
            if cfg.init_type != 'stable':
                # Generate rotations.
                if cfg.randomize_init_orn:
                    qs_random = tx.rotation.quaternion.random(size=num_reset)
                    qs_cuboid = sample_cuboid_poses(num_reset,
                                                    noise=np.deg2rad(1.0))
                    selector = (np.random.uniform(size=num_reset)
                                < cfg.canonical_pose_prob)
                    qs = np.where(selector[..., None], qs_cuboid, qs_random)
                else:
                    qs = np.zeros((num_reset, 4), dtype=np.float32)
                    qs[..., 3] = 1
                Rs = tx.rotation.matrix.from_quaternion(qs)

                z_axis = th.as_tensor(Rs[..., 2],
                                      device=nxt_hulls.device,
                                      dtype=nxt_hulls.dtype)
                dz = th.einsum(
                    '...ni, ...i -> ...n',
                    nxt_hulls,
                    z_axis).min(dim=-1).values
                zs = cfg.table_dims[2] - dz + cfg.z_eps

        with nvtx.annotate("f"):
            if len(ni) > 0:
                with nvtx.annotate("1"):
                    root_tensor[ni] = 0
                    # root_tensor.index_fill_(0, ni, 0)
                    # root_tensor[ni] = th.as_tensor(
                    #     0, dtype=th.float, device=root_tensor.devic)[
                    #     None, None].expand_as(root_tensor[ni])
                if cfg.mode == 'valid':
                    # ic(self._valid_poses.shape)  # 16,1,7
                    # ic(ni.shape)
                    # ic(self._per_env_offsets.shape)
                    # ic(ni)
                    # ic((self._per_env_offsets[env_ids]
                    #    // cfg.num_obj_per_env)
                    #   % self._valid_poses.shape[-2])
                    pose_index = ((self._per_env_offsets[env_ids]
                                   // cfg.num_obj_per_env)
                                  % self._valid_poses.shape[-2])
                    # ic(pose_index)
                    root_tensor[ni, :7] = self._valid_poses[
                        # Object / environment index
                        env_ids,
                        # Pose index
                        pose_index
                    ]
                elif cfg.mode == 'train':
                    if cfg.override_cube:

                        size = len(ni)
                        x = self.xsampler.sample((size,)).to(env.device)
                        yo = self.ysampler.sample((size,)).to(env.device)

                        z = (
                            cfg.table_dims[2] + self.box_dim[2] / 2) * th.ones(
                            (len(ni)), dtype=th.float, device=env.device)

                        roll = (np.pi / 2) * th.randint(0, 4,
                                                        (size,), device=env.device)
                        pitch = (np.pi / 2) * th.randint(0, 4,
                                                         (size,), device=env.device)
                        yaw = (2 * np.pi) * th.rand(size,
                                                    dtype=th.float, device=env.device)
                        quat = quat_from_euler_xyz(roll, pitch, yaw)
                        root_tensor[ni, 0] = x
                        root_tensor[ni, 1] = yo
                        root_tensor[ni, 2] = z
                        root_tensor[ni, 3:7] = quat
                    else:
                        with nvtx.annotate("2"):
                            # TODO: also set (x,y) values !!
                            root_tensor[ni, :2] = self._get_xy(
                                env, root_tensor.device, root_tensor.dtype, len(ni),
                                env_ids,
                                prevent_fall=cfg.prevent_fall,
                                avoid_overlap=cfg.avoid_overlap,
                            )
                        if cfg.init_type != 'stable':
                            root_tensor[ni, 2] = th.as_tensor(
                                zs, dtype=root_tensor.dtype,
                                device=root_tensor.device)
                            root_tensor[ni, 3:7] = th.as_tensor(
                                np.asarray(qs), dtype=root_tensor.dtype,
                                device=root_tensor.device)

                            if cfg.randomize_yaw:
                                qz = sample_yaw(
                                    num_reset, device=root_tensor.device)
                                root_tensor[ni, 3:7] = quat_multiply(
                                    qz, root_tensor[ni, 3:7])
                        else:
                            which_pose = th.randint(
                                self.cur_stable_poses.shape[1],
                                size=(num_reset,),
                                dtype=th.long,
                                device=self.cur_stable_poses.device,
                            )
                            root_tensor[ni, 2:7] = (
                                self.cur_stable_poses[env_ids, which_pose, 2:7]
                            )
                            root_tensor[ni, 2] += cfg.z_eps
                            if cfg.randomize_yaw:
                                qz = sample_yaw(
                                    num_reset, device=root_tensor.device)
                                root_tensor[ni, 3:7] = quat_multiply(
                                    qz, root_tensor[ni, 3:7])

        with nvtx.annotate("g"):
            # merge pi, ni
            if (cfg.num_obj_per_env > 1):
                mask = self.mask
                mask.fill_(0)
                if (prv_ids is not None):
                    mask[pi] = 1
                mask[ni] = 1
                set_ids = th.argwhere(mask).ravel().to(
                    dtype=th.int32)
            else:
                set_ids = ni.to(dtype=th.int32)

        with nvtx.annotate("h"):
            if not reset_all:
                self.body_ids[env_ids] = nxt_body_ids
            else:
                self.body_ids = nxt_body_ids

        with nvtx.annotate("i"):
            return set_ids

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg

        # Sample N objects from the pool.

        # Spawn table.
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*cfg.table_pos)
        table_pose.r = gymapi.Quat(*cfg.table_orn)

        table_actor = gym.create_actor(
            env, self.assets['table'],
            table_pose, F'table', env_id,
            0b0001
        )

        if True:
            shape_props = gym.get_actor_rigid_shape_properties(
                env, table_actor)
            for p in shape_props:
                p.filter = 0b0001
            gym.set_actor_rigid_shape_properties(env, table_actor, shape_props)

        if cfg.table_friction is not None:
            set_actor_friction(gym, env, table_actor, cfg.table_friction)

        object_actors = []

        keys = self.keys[env_id * cfg.num_obj_per_env:]
        for i, key in enumerate(keys[:cfg.num_obj_per_env]):
            obj_pose = gymapi.Transform()

            # Spawn objects.
            # -1: load from asset; 0: enable self-collision; >0: disable self-collision
            obj_asset = self.assets['objects'][key]
            body_count = gym.get_asset_rigid_body_count(obj_asset)
            shape_count = gym.get_asset_rigid_shape_count(obj_asset)
            with aggregate(gym, env,
                           body_count,
                           shape_count,
                           False,
                           use=False):
                object_actor = gym.create_actor(
                    env,
                    obj_asset,
                    obj_pose,
                    F'object-{i:02d}',
                    env_id,
                    0b0010
                )

            if True:
                shape_props = gym.get_actor_rigid_shape_properties(
                    env, object_actor)
                for p in shape_props:
                    p.filter = 0b0010
                gym.set_actor_rigid_shape_properties(
                    env, object_actor, shape_props)

            if cfg.object_friction is not None:
                set_actor_friction(gym, env, object_actor,
                                   cfg.object_friction)
            gym.set_rigid_body_segmentation_id(env, object_actor,
                                               0, 1 + i)
            object_actors.append(object_actor)

        return {'table': table_actor,
                'object': object_actors}

    def create_assets(self, gym, sim, env: 'EnvBase',
                      counts: Optional[Dict[str, int]] = None
                      ):
        cfg = self.cfg

        # (1) Create table.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False

        total_body_count: int = 0
        total_shape_count: int = 0

        if cfg.use_wall:
            if cfg.use_bin:
                bin_urdf_text = create_bin(table_dims=cfg.table_dims,
                                           wall_width=cfg.wall_width,
                                           wall_height=cfg.wall_height)

                asset_options.vhacd_enabled = False
                asset_options.thickness = 0.001  # ????
                asset_options.convex_decomposition_from_submeshes = True
                asset_options.override_com = True
                asset_options.override_inertia = True

                with TemporaryDirectory() as tmpdir:
                    with open(tmpdir / 'bin.urdf', 'w') as fp:
                        fp.write(bin_urdf_text)
                    table_asset = gym.load_urdf(sim,
                                                tmpdir,
                                                'bin.urdf',
                                                asset_options)
            else:
                filename = cfg.table_file
                # asset_options.disable_gravity = False
                asset_options.vhacd_enabled = False
                asset_options.thickness = 0.001  # ????
                asset_options.convex_decomposition_from_submeshes = True
                asset_options.override_com = True
                asset_options.override_inertia = True
                # asset_options.override_com = False
                # asset_options.override_inertia = False
                table_asset = gym.load_urdf(sim,
                                            str(cfg.asset_root),
                                            str(cfg.table_file),
                                            asset_options)
        else:
            table_asset = gym.create_box(sim,
                                         *cfg.table_dims,
                                         asset_options)
        total_body_count += gym.get_asset_rigid_body_count(table_asset)
        total_shape_count += gym.get_asset_rigid_shape_count(table_asset)

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

        max_load: int = cfg.num_obj_per_env * env.cfg.num_env
        urdfs = [self.meta.urdf(k) for k in self.meta.keys()]

        num_obj = min(max_load, len(urdfs), cfg.num_object_types)
        if cfg.mode == 'train':
            object_files = np.random.choice(
                urdfs,
                size=num_obj,
                replace=False
            )
        elif cfg.mode == 'valid':
            # Deterministic and ordered list of object_files
            object_files = list(itertools.islice(
                itertools.cycle(urdfs),
                num_obj))
        else:
            raise KeyError(F'Unknown mode = {cfg.mode}')

        max_obj_body_count: int = 0
        max_obj_shape_count: int = 0

        for index, filename in enumerate(
                tqdm(object_files, desc='create_object_assets')):

            # FIXME: relies on string parsing
            # to identify the key for the specific
            # URDF file
            # key = filename
            key = Path(filename).stem

            if key in obj_assets:
                continue

            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = False
            # FIXME: hardcoded `thickness`
            asset_options.thickness = 0.001

            # asset_options.linear_damping = 1.0
            # asset_options.angular_damping = 1.0

            if filename == 'cube':
                # CRM cube
                asset_options.density = cfg.density
                obj_asset = gym.create_box(sim,
                                           0.09, 0.09, 0.09,
                                           asset_options)
            else:
                if cfg.override_inertia:
                    asset_options.override_com = True
                    asset_options.override_inertia = True
                    if cfg.target_mass is None:
                        asset_options.density = cfg.density
                    else:
                        idx = 1 if cfg.load_convex else 0
                        volume = self.volume[key][idx]
                        asset_options.density = cfg.target_mass / volume
                else:
                    asset_options.override_com = False
                    asset_options.override_inertia = False
                # NOTE: set to `True` since we're directly using
                # the convex decomposition result from CoACD.
                asset_options.vhacd_enabled = False
                if cfg.load_convex:
                    asset_options.convex_decomposition_from_submeshes = False
                else:
                    asset_options.convex_decomposition_from_submeshes = True
                obj_asset = gym.load_urdf(sim,
                                          str(Path(filename).parent),
                                          str(Path(filename).name),
                                          asset_options)

            obj_body_count = gym.get_asset_rigid_body_count(obj_asset)
            obj_shape_count = gym.get_asset_rigid_shape_count(obj_asset)

            max_obj_body_count = max(max_obj_body_count,
                                     obj_body_count)
            max_obj_shape_count = max(max_obj_shape_count,
                                      obj_shape_count)

            # key = F'{index}-{filename}'
            obj_assets[key] = obj_asset

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

        total_body_count += max_obj_body_count
        total_shape_count += max_obj_shape_count

        if counts is not None:
            counts['body'] = total_body_count
            counts['shape'] = total_shape_count

        # (2) Create objects.
        self.assets = {
            'table': table_asset,
            'objects': obj_assets,
            'force_sensors': force_sensors
        }

        self.keys = list(itertools.islice(
            itertools.cycle(list(obj_assets.keys())),
            env.cfg.num_env * cfg.num_obj_per_env))

        convert = partial(_array_from_map,
                          self.keys,
                          dtype=th.float,
                          device=env.device)

        self.radii = convert({k: self.meta.radius(k) for k in self.keys})
        self.hulls = convert(_pad_hulls(
            {k: self.meta.hull(k) for k in self.keys}))

        if True:
            self.table_pos = einops.repeat(
                th.as_tensor(np.asarray(cfg.table_pos),
                             dtype=th.float,
                             device=env.device),
                '... -> n ...', n=env.num_env)
            self.table_dims = einops.repeat(
                th.as_tensor(np.asarray(cfg.table_dims),
                             dtype=th.float,
                             device=env.device),
                '... -> n ...', n=env.num_env)

        if cfg.goal_type == 'stable':
            stable_poses = {k: self.meta.pose(k) for k in self.keys}
            min_len = min([len(v) for v in stable_poses.values()])
            max_len = min(
                max([len(v) for v in stable_poses.values()]),
                cfg.truncate_pose_count)
            print(F'\tmin_len = {min_len}, max_len = {max_len}')

            def _pad(x: np.ndarray, max_len: int):
                if len(x) < max_len:
                    extra = max_len - len(x)
                    x = np.concatenate(
                        [x, x[np.random.choice(len(x), size=extra, replace=True)]],
                        axis=0)
                else:
                    x = x[np.random.choice(
                        len(x), size=max_len, replace=False)]
                return x
            stable_poses = {k: _pad(v[..., :7], max_len)
                            for k, v in stable_poses.items()}
            self.stable_poses = convert(stable_poses)

            # == additionally we pregenerate a bunch of validation poses ==
            if cfg.mode == 'valid':
                self._valid_poses = (self.stable_poses[:, :cfg.num_valid_poses]
                                     .detach().clone())
                self._valid_poses[..., 0:2] = self._get_xy(
                    env,
                    self._valid_poses.device,
                    self._valid_poses.dtype,
                    # self._valid_poses.shape[:-1],
                    (cfg.num_valid_poses, self.stable_poses.shape[0]),
                    env_ids=th.arange(env.num_env, device=env.device),
                    prevent_fall=False
                ).swapaxes(0, 1)

            is_yaw_only = None

            if cfg.use_yaw_only_logic:
                is_yaw_only = {
                    k: is_thin(self.meta.obb(k)[1], threshold=cfg.thin_threshold)
                    for k in self.keys
                }

            if cfg.yaw_only_key is not None:
                is_yaw_only = {
                    k: 1. if k in cfg.yaw_only_key else 0.
                }
                print(cfg.yaw_only_key)
                print(is_yaw_only)

            if cfg.yaw_only_file is not None:
                with open(cfg.filter_file, 'r') as fp:
                    yawonly_list = [str(s) for s in json.load(fp)]
                if is_yaw_only is not None:
                    for k in yawonly_list:
                        is_yaw_only[k] = 1.
                else:
                    is_yaw_only = {
                        k: 1. if k in yawonly_list else 0.
                        for k in self.keys
                    }

            if is_yaw_only is not None:
                self.is_yaw_only = convert(is_yaw_only).bool()
            else:
                self.is_yaw_only = th.zeros(len(self.keys), dtype=th.bool,
                                            device=env.device)

        if cfg.load_embedding:
            self.object_embeddings = convert(
                {k: self.meta.code(k) for k in self.keys})
            # if self.patch_centers is not None:
            #     self.patch_centers = convert(self.patch_centers)
        if cfg.load_bbox:
            self.bboxes = convert({k: self.meta.bbox(k) for k in self.keys})
        if cfg.load_obb:
            self.obbs = convert({k: self.meta.obb(k)[1] for k in self.keys})
        if cfg.load_cloud:
            self.cloud = convert({k: self.meta.cloud(k) for k in self.keys})
            if cfg.load_normal:
                self.normal = convert({k: self.meta.normal(k)
                                       for k in self.keys})
        if cfg.load_predefined_goal:
            self.predefined_goal = convert(
                {k: self.meta.predefined_goal(k) for k in self.keys})

        if cfg.load_stable_mask:
            num_obj = len(self.cloud)

            stable_masks_np = np.zeros(
                (num_obj, self.stable_poses.shape[1]),
                dtype=bool)

            for i in range(num_obj):
                cloud = dcn(self.cloud[i])
                for j in range(len(self.stable_poses[i])):
                    pose = dcn(self.stable_poses[i, j])
                    cloud_at_pose = tx.rotation.quaternion.rotate(
                        pose[None, 3:7],
                        cloud) + pose[None, 0:3]
                    stable_masks_np[i, j] = _is_stable(cloud_at_pose)

            self.stable_masks = th.as_tensor(stable_masks_np,
                                             dtype=bool,
                                             device=env.device)

        return self.assets

    def create_sensors(self, gym, sim, env, env_id: int):
        return {}


def main():
    scene = TableTopWithObjectScene(
        TableTopWithObjectScene.Config())


if __name__ == '__main__':
    main()
