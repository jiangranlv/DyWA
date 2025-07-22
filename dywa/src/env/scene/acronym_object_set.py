#!/usr/bin/env python3

import json
import numpy as np
import shutil
from collections import defaultdict
from env.scene.object_set import ObjectSet
from util.config import ConfigBase
from util.path import ensure_directory
from functools import cached_property, partial
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from data.transforms.io_xfm import (
    load_mesh, scene_to_mesh)
import open3d as o3d

from yourdfpy import URDF
import trimesh
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import gdown
from cho_util.math import transform as tx
from env.scene.util import (
    sample_stable_poses, load_npy, load_glb, load_pkl,
    stat_from_urdf,
    lookup_normal
)

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')


class AcronymObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        data_path: str = F'{DATA_ROOT}/ACRONYM/meta-v0'
        meta_file: Optional[str] = None
        # Robot URDF files.
        urdf_path: Optional[str] = None
        # Convex hull for quickly computing initial placements.
        hull_path: Optional[str] = None
        cloud_path: Optional[str] = None
        normal_path: Optional[str] = None
        pose_path: Optional[str] = None
        code_path: Optional[str] = None

        # Allow "fallback" strategies for
        # computing desired attributes online.
        fallback: bool = False

        # Extra stuff
        # patch_center_file: str = '/input/ACRONYM/patch-v12.pkl'
        def __post_init__(self):
            if self.data_path is not None:
                data_path = self.data_path
                if self.meta_file is None:
                    self.meta_file = F'{data_path}/metadata.json'
                if self.urdf_path is None:
                    self.urdf_path = F'{data_path}/urdf'
                if self.hull_path is None:
                    self.hull_path = F'{data_path}/hull'
                if self.cloud_path is None:
                    self.cloud_path = F'{data_path}/cloud'
                if self.normal_path is None:
                    self.normal_path = F'{data_path}/normal'
                if self.pose_path is None:
                    self.pose_path = F'{data_path}/pose'
                if self.code_path is None:
                    self.code_path = F'{data_path}/code'

    def __init__(self, cfg: Config):
        self.cfg = cfg
        with open(cfg.meta_file, 'r') as fp:
            self.__metadata = json.load(fp)

    def keys(self):
        return self.__metadata.keys()

    @cached_property
    def poses(self):
        return {k: load_npy(F'{self.cfg.pose_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def codes(self):
        # FIXME: temporary exception
        if Path(self.cfg.code_path).is_file():
            return load_pkl(self.cfg.code_path)
        return {k: load_npy(F'{self.cfg.code_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def clouds(self):
        return {k: load_npy(F'{self.cfg.cloud_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def normals(self):
        out = {k: load_npy(F'{self.cfg.normal_path}/{k}.npy')
               for k in self.keys()}
        return out

    @cached_property
    def hulls(self) -> Dict[str, trimesh.Trimesh]:
        return {k: load_glb(F'{self.cfg.hull_path}/{k}.glb')
                for k in self.keys()}

    def label(self, key: str) -> str:
        """ Category of this object """
        # FIXME: string parsing for
        # determining class, usually not a
        # good idea...
        return str(key).split('_', 1)

    def urdf(self, key: str):
        cfg = self.cfg
        return F'{cfg.urdf_path}/{key}.urdf'

    def pose(self, key: str):
        # Optional fallback guard
        if (((key not in self.poses) or (self.poses[key] is None))
                and (self.cfg.fallback) and (key in self.hulls)):
            pose = sample_stable_poses(self.hulls[key],
                                       # FIXME: hardcoded table height
                                       0.4,
                                       16)
            self.poses[key] = pose
        if key in self.poses:
            return self.poses[key]
        return self.poses[key]

    def code(self, key: str):
        return self.codes[key]

    def cloud(self, key: str):
        return self.clouds[key]

    def normal(self, key: str):
        return self.normals[key]

    def bbox(self, key: str):
        return np.asarray(self.__metadata[key]['bbox'],
                          dtype=np.float32)

    def aabb(self, key: str):
        return np.asarray(self.__metadata[key]['aabb'],
                          dtype=np.float32)

    def obb(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        obb = self.__metadata[key]['obb']
        xfm = np.asarray(obb[0], dtype=np.float32)
        extent = np.asarray(obb[1], dtype=np.float32)
        return (xfm, extent)

    def hull(self, key: str) -> trimesh.Trimesh:
        return self.hulls[key]

    def radius(self, key: str) -> float:
        return self.__metadata[key]['radius']
        # if 'radius' in self.__metadata[key]:
        #    return self.__metadata[key]['radius']
        # else:
        #    radius = np.linalg.norm(self.hull(key).vertices, axis=-1).max()
        #    self.__metadata[key]['radius'] = radius
        #    return self.radius(key)

    def volume(self, key: str) -> float:
        return self.__metadata[key]['volume']

    def num_verts(self, key: str) -> float:
        return self.__metadata[key]['num_verts']

    def num_faces(self, key: str) -> float:
        return self.__metadata[key]['num_faces']

    def num_hulls(self, key: str) -> float:
        return self.__metadata[key]['num_chulls']


def _add_entry(urdf: str):
    urdf = Path(urdf)
    key = str(urdf.stem)
    # Get some mesh-related statistics.
    entry = stat_from_urdf(urdf)
    return (key, entry)


def _convert_from_previous_version():
    out_path: str = ensure_directory('/input/ACRONYM/meta-v0')
    urdf_path = ensure_directory(out_path / 'urdf')
    hull_path = ensure_directory(out_path / 'hull')
    cloud_path = ensure_directory(out_path / 'cloud')
    normal_path = ensure_directory(out_path / 'normal')
    pose_path = ensure_directory(out_path / 'pose')
    code_path = ensure_directory(out_path / 'code')
    meta_file = out_path / 'metadata.json'

    meta = defaultdict(dict)
    for k, v in load_pkl('/input/ACRONYM/bbox.pkl').items():
        key = str(Path(k).stem)
        meta[key]['bbox'] = v.tolist()

    with open('/input/ACRONYM/urdf_stats.json', 'r') as fp:
        d = json.load(fp)
        d = {str(Path(k).stem): (v.tolist() if isinstance(
            v, np.ndarray) else v) for k, v in d.items()}
        meta.update(d)

    ctx = mp.get_context('spawn')

    # COPY URDF file
    for urdf in tqdm(Path('/input/ACRONYM/urdf/').glob('*.urdf'), desc='urdf'):
        key = str(urdf.stem)
        shutil.copy(str(urdf), str(urdf_path / F'{key}.urdf'))

    # UPDATE METADATA
    if True:
        with ctx.Pool(16) as pool:
            urdf_files = list(Path('/input/ACRONYM/urdf/').glob('*.urdf'))
            for (key, entry) in tqdm(pool.imap_unordered(
                    _add_entry, urdf_files), desc='urdf'):
                meta[key].update(entry)

    for hull in Path('/input/ACRONYM/hull/').glob('*.glb'):
        key = str(hull.stem)
        shutil.copy(str(hull), str(hull_path / F'{key}.glb'))

    for k, v in load_pkl('/input/ACRONYM/cloud.pkl').items():
        key = str(k)
        np.save(str(cloud_path / F'{key}.npy'), v)

    for k, v in load_pkl('/input/ACRONYM/stable_poses_filtered.pkl').items():
        key = str(k)
        np.save(str(pose_path / F'{key}.npy'), v)

    for k, v in load_pkl('/tmp/embed-p2v-ours.pkl').items():
        key = str(k)
        np.save(str(code_path / F'{key}.npy'), v)

    with open(str(meta_file), 'w') as fp:
        json.dump(meta, fp)


def _export_stable_poses(key: str, hull_path: str, out_dir: str):
    out_file = F'{out_dir}/{key}.npy'
    if Path(out_file).exists():
        return
    hull = load_glb(F'{hull_path}/{key}.glb')
    if hull is None:
        return
    pose = sample_stable_poses(hull, 0.4, 32, 8)
    np.save(out_file, pose)


def sample_extra_stable_poses_with_trimesh():
    from tqdm.auto import tqdm
    import multiprocessing as mp
    dataset = AcronymObjectSet(
        AcronymObjectSet.Config()
    )
    out_dir = ensure_directory('/tmp/extra_poses')
    ctx = mp.get_context('spawn')
    export_fn = partial(_export_stable_poses,
                        hull_path=dataset.cfg.hull_path,
                        out_dir=out_dir)
    with ctx.Pool(16) as pool:
        keys = list(dataset.keys())
        for _ in tqdm(pool.imap_unordered(export_fn, keys)):
            pass


def sample_normals_with_trimesh():
    dataset = AcronymObjectSet(
        AcronymObjectSet.Config()
    )
    out_dir = ensure_directory('/tmp/normals')
    for key in tqdm(dataset.keys()):
        out_file = F'{out_dir}/{key}.npy'
        if Path(out_file).is_file():
            continue
        urdf = dataset.urdf(key)
        scene = URDF.load(urdf,
                          load_meshes=True,
                          build_collision_scene_graph=True,
                          force_collision_mesh=True,
                          load_collision_meshes=True
                          ).collision_scene
        mesh = scene_to_mesh(scene)
        cloud = dataset.cloud(key)
        if cloud is None:
            continue
        normals = lookup_normal(mesh, )

        # == visualize ==
        if False:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dataset.cloud(key))
            pcd.normals = o3d.utility.Vector3dVector(normals)
            o3d.visualization.draw_geometries([pcd],
                                              point_show_normal=True)
        # == export ==
        np.save(out_file, normals)


def print_all_attrs():
    dataset = AcronymObjectSet(
        AcronymObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    print(len(dataset.codes))


def recompute_metadata():
    pass


def show():
    from env.scene.util import _show_stable_poses
    from env.scene.filter_object_set import FilteredObjectSet
    from matplotlib import pyplot as plt
    dataset = AcronymObjectSet(
        AcronymObjectSet.Config(
            data_path='/input/ACRONYM/meta-v1',
            pose_path='/input/ACRONYM/meta-v1/pose2'
        ))
    with open('/tmp/hard.json', 'r') as fp:
        keys = json.load(fp)

    def _has_pose(key):
        try:
            pose = dataset.pose(key)
            return pose is not None
        except Exception:
            return False

    # dataset = FilteredObjectSet(dataset, keys=keys)
    # dataset = FilteredObjectSet(dataset, filter_fn=_has_pose)
    # _show_stable_poses(dataset)
    print(dataset.cloud(next(iter(dataset.keys()))).shape)  # 512... !

    def num_pose(key):
        try:
            pose = dataset.pose(key)
            if pose is None:
                return 0
            return len(pose)
        except KeyError:
            return 0
    num_poses = {k: num_pose(k) for k in
                 dataset.keys()}
    plt.hist(num_poses.values(),
             bins=64)
    plt.show()


def main():
    # sample_extra_stable_poses_with_trimesh()
    # sample_normals_with_trimesh()
    # recompute_metadata()
    show()
    pass


if __name__ == '__main__':
    main()
