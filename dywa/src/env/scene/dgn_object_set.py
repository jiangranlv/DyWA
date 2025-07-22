#!/usr/bin/env python3

import json
import math
import numpy as np
from typing import Dict, Optional
from env.scene.object_set import ObjectSet
from util.config import ConfigBase
import os
import shutil
from dataclasses import dataclass
from typing import Tuple
import trimesh
from tempfile import mkdtemp
from util.torch_util import dcn
from tqdm.auto import tqdm
from pathlib import Path
from util.math_util import quat_rotate
from util.path import ensure_directory
import torch as th
from env.scene.util import (
    # stat_from_urdf,
    mesh_from_urdf,
    stat_from_mesh,
    sample_stable_poses,
    sample_stable_poses_fast,
    load_npy, load_glb, load_pkl,
    stat_from_urdf
)
from yourdfpy import URDF
from functools import cached_property, partial
from icecream import ic
import multiprocessing as mp

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="{com}"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{filename}" scale="{sx} {sy} {sz}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{filename}" scale="{sx} {sy} {sz}"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


class DGNObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        # density: float = 300.0
        # table_dims: Tuple[float, float, float] = (0.4, 0.5, 0.4)
        # num_poses: int = 256
        # '/input/DGN/meta-v0'
        dgn_path: str = F'{DATA_ROOT}/DGN'
        # meta_path: Optional[str] = None

        data_path: Optional[str] = None  # '/input/DGN/meta-v0/'
        meta_file: Optional[str] = None
        # /input/DGN/meshdata/core-bottle-1071fa4cddb2da2fc8724d5673a063a6/coacd/
        # Convex hull for quickly computing initial placements.
        urdf_path: Optional[str] = None
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
        ic(cfg)
        print('loading metadata')
        with open(cfg.meta_file, 'r') as fp:
            self.__metadata = json.load(fp)
        print('loaded metadata')

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
        return key

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
        return None

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

    def volume(self, key: str) -> float:
        return self.__metadata[key]['volume']

    def num_verts(self, key: str) -> float:
        return self.__metadata[key]['num_verts']

    def num_faces(self, key: str) -> float:
        return self.__metadata[key]['num_faces']

    def num_hulls(self, key: str) -> float:
        # return self.__metadata[key]['num_chulls']
        return self.__metadata[key]['num_hulls']


def __convert_one(name: str, out_path: str):
    src_urdf = F'/input/DGN/meshdata/{name}/coacd/coacd.urdf'
    ann_file = F'/input/DGN/dexgraspnet/{name}.npy'
    out_path = Path(out_path)
    if not Path(ann_file).exists():
        return

    grasp_anns = np.load(ann_file, allow_pickle=True)
    scales = np.unique([x['scale'] for x in grasp_anns])

    hull_path = ensure_directory(out_path / 'hull')
    meta_path = ensure_directory(out_path / 'meta')
    cloud_path = ensure_directory(out_path / 'cloud')
    normal_path = ensure_directory(out_path / 'normal')
    pose_path = ensure_directory(out_path / 'pose')
    code_path = ensure_directory(out_path / 'code')
    urdf_path = ensure_directory(out_path / 'urdf')
    urdf = URDF.load(src_urdf)

    for scale in scales:
        key = F'{name}-{scale:.03f}'
        scene = urdf.scene.scaled(scale)
        stat = stat_from_mesh(scene)
        single = scene.dump(concatenate=True)
        hull = single.convex_hull
        # pose = sample_stable_poses(hull, num_samples=64)
        com = single.center_mass
        # pose = sample_stable_poses_fast(hull, com)
        pose = None
        cvx_obj = list(Path(F'/input/DGN/meshdata/{key}/coacd/').glob(
            F'coacd_convex_piece_*.obj'))
        stat['num_hulls'] = len(cvx_obj)

        cloud, face_index = trimesh.sample.sample_surface(single, 512)
        normal = single.face_normals[face_index]

        # == EXPORT ==
        hull.export(str(hull_path / F'{key}.glb'))
        np.save(str(cloud_path / F'{key}.npy'), cloud)
        np.save(str(normal_path / F'{key}.npy'), normal)
        if pose is not None:
            np.save(str(pose_path / F'{key}.npy'), pose)
        # FIXME: DO
        # np.save(str(code_path / F'{key}.npy'), stat)

        # Rewrite geometry with fixed
        for link in urdf.link_map.values():
            # <inertial, visual, collision>
            for v in link.visuals:
                mesh_file = str(
                    (Path(src_urdf).parent / v.geometry.mesh.filename)
                    .resolve())
                if not Path(mesh_file).exists():
                    raise FileNotFoundError(
                        F'{mesh_file} does not exist...')
                v.geometry.mesh.filename = mesh_file
                v.geometry.mesh.scale = scale
            for c in link.collisions:
                mesh_file = str(
                    (Path(src_urdf).parent / c.geometry.mesh.filename)
                    .resolve())
                if not Path(mesh_file).exists():
                    raise FileNotFoundError(
                        F'{mesh_file} does not exist...')
                c.geometry.mesh.filename = mesh_file
                c.geometry.mesh.scale = scale
        urdf.write_xml_file(str(urdf_path / F'{key}.urdf'))

        # meta
        KEYS = ['bbox', 'aabb', 'obb', 'radius', 'volume',
                'num_verts', 'num_faces', 'num_hulls']
        with open(F'{meta_path}/{key}.json', 'w') as fp:
            json.dump({k: stat[k] for k in KEYS}, fp)


def _convert_from_default():
    # need:
    # keys
    names = [d.name for d in Path('/input/DGN/meshdata/').glob('*')]
    # names = names[:100]
    # no real "label" available for now...
    labels = names
    # urdf (OK?)
    out_path = ensure_directory(F'/input/DGN/meta-v7')

    # for name in tqdm(names):
    #     __convert_one(name, out_path)
    convert_fn = partial(__convert_one, out_path=str(out_path))
    ctx = mp.get_context('spawn')
    with ctx.Pool(16) as pool:
        for _ in tqdm(pool.imap_unordered(convert_fn, names),
                      total=len(names), desc='convert'):
            pass

    # Dump meta into one file
    meta_path = ensure_directory(out_path / 'meta')
    meta = {}
    for filename in Path(meta_path).glob('*.json'):
        key = filename.stem
        with open(filename, 'r') as fp:
            data = json.load(fp)
        meta[key] = data

    with open(F'{out_path}/metadata.json', 'w') as fp:
        json.dump(meta, fp)


def _rewrite_urdf():
    cfg = DGNObjectSet.Config(
        data_path='/input/DGN/meta-v7/'
    )
    out_path = ensure_directory('/tmp/dgn-urdf/')
    dataset = DGNObjectSet(cfg)
    for key in tqdm(dataset.keys()):
        # /input/DGN/coacd/core-bottle-1071fa4cddb2da2fc8724d5673a063a6.obj
        # /input/DGN/meshdata/core-bottle-1071fa4cddb2da2fc8724d5673a063a6/coacd/coacd.urdf
        mesh_id, scale = key.rsplit('-', 1)
        scale = float(scale)

        new_obj_file = F'/input/DGN/coacd/{mesh_id}.obj'
        new_obj = trimesh.load(new_obj_file)
        new_obj.apply_scale(scale)
        src_urdf = dataset.urdf(key)

        volume = new_obj.volume
        # FIXME: fixed density
        density: float = 300.0
        mass = volume * density
        # trimesh moment of inertia is calculated
        # by taking the volume as the mass
        inertia = (mass / volume) * new_obj.moment_inertia
        urdf_text = URDF_TEMPLATE.format(
            mass=mass,
            com=''.join([F' {x:.04f}' for x in new_obj.center_mass]),
            ixx=inertia[0, 0],
            ixy=inertia[0, 1],
            ixz=inertia[0, 2],
            iyy=inertia[1, 1],
            iyz=inertia[1, 2],
            izz=inertia[2, 2],
            filename=new_obj_file,
            sx=scale,
            sy=scale,
            sz=scale,
        )
        urdf_file = F'{out_path}/{key}.urdf'
        with open(str(urdf_file), 'w') as fp:
            fp.write(urdf_text)
        # /input/DGN/coacd/core-bottle-1071fa4cddb2da2fc8724d5673a063a6.obj


def print_all_attrs():
    dataset = DGNObjectSet(
        DGNObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    print(len(dataset.codes))


def show():
    from env.scene.filter_object_set import FilteredObjectSet
    from env.scene.util import _show_stable_poses
    cfg = DGNObjectSet.Config(
        data_path='/input/DGN/meta-v8/',
        # pose_path='/tmp/dedup_poses/'
        # pose_path='/tmp/acr_poses/'
        # pose_path='/tmp/dedup_poses_2/'
        # pose_path='/tmp/dedup_poses_3/'
        pose_path='/tmp/acr_poses_4/'
    )
    print(cfg)
    dataset = DGNObjectSet(cfg)
    # keys = [k for k in dataset.keys() if 'ToyFig' in k]
    # keys = [k for k in dataset.keys() if
    #         (dataset.pose(k) is not None and
    #          len(dataset.pose(k)) <= 4)]

    # with open('/tmp/yes.json', 'r') as fp:
    #     keys = json.load(fp)
    keys = [
        "core-bottle-26a8d94392d86aa0940806ade53ef2f-0.100",
        "sem-Flashlight-70cfc22ffe0e9835fc3223d52c1f21a9-0.150",
        "sem-Pillow-359576c82de003cf96907985b34a7ba3-0.100",
        "sem-Thumbtack-58823e7be75befa9b2b6352649b85133-0.080",
        "sem-VideoGameController-1046b3d5a381a8902e5ae31c6c631a39-0.120",
        "core-can-669033f9b748c292d18ceeb5427760e8-0.060",
        "core-bottle-44dae93d7b7701e1eb986aac871fa4e5-0.080",
        "mujoco-BlackBlack_Nintendo_3DSXL-0.150",
        "mujoco-Pet_Dophilus_powder-0.100",
        "mujoco-Dell_Ink_Cartridge_Yellow_31-0.100",
        "core-bottle-dc005c019fbfb32c90071898148dca0e-0.150"
    ]
    dataset = FilteredObjectSet(dataset, keys=keys)
    _show_stable_poses(dataset, per_obj=1)


def print_aspect():
    from env.scene.filter_object_set import FilteredObjectSet
    cfg = DGNObjectSet.Config(
        data_path='/input/DGN/meta-v8/',
        pose_path='/tmp/dedup_poses/'
    )
    with open('/tmp/all-dgn-uniq-keys.json', 'r') as fp:
        keys = json.load(fp)
    dataset = DGNObjectSet(cfg)
    dataset = FilteredObjectSet(dataset, keys=keys)
    aspects = []
    for key in dataset.keys():
        extent = dataset.obb(key)[1]
        aspect = max(extent) / min(extent)
        aspects.append(aspect)
        print(dataset.pose(key).shape)
        break
    # from matplotlib import pyplot as plt
    # plt.hist(aspects, bins=64)
    # plt.axvline(np.median(aspects), color='r')
    # plt.grid()
    # plt.xlabel('aspect')
    # plt.ylabel('count')
    # plt.show()


def main():
    # _convert_from_default()
    show()
    # print_all_attrs()
    # _rewrite_urdf()
    # print_aspect()


if __name__ == '__main__':
    main()
