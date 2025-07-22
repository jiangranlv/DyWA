#!/usr/bin/env python3

import os
import glob
from tempfile import mkdtemp
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import shutil

import numpy as np
import trimesh
from tqdm.auto import tqdm

from icecream import ic

from util.path import ensure_directory

from util.config import ConfigBase
from env.scene.util import sample_stable_poses
from env.scene.object_set import ObjectSet
from env.scene.apply_coacd import (Config as COACDConfig,
                                       apply_coacd)

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{vis_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{col_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


class MeshObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        # Is this necessary? I guess we'll just
        # configure the mass according to this density parameter
        # and the volume ... unless we also apply mass DR
        density: float = 300.0
        table_dims: Tuple[float, float, float] = (0.4, 0.5, 0.4)
        # Might be necessary for on-line generation of
        # stable poses
        num_poses: int = 256
        # = CloudSize
        num_points: int = 512
        cache_dir: str = '/tmp/poses'

        # Remember: we also need to run CoACD
        # (probably)... to generate collision mesh
        filename: Optional[str] = None

        acd: bool = True
        coacd: COACDConfig = COACDConfig(
            mcts_max_depth=3,
            mcts_iterations=32,
            mcts_nodes=16,
            verbose=True
        )

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if cfg.filename is None:
            raise ValueError('cfg.filename should not be None!')

        files = glob.glob(cfg.filename,
                          recursive=True)

        self.__files = {str(Path(m).stem): m
                        for m in files}
        self.__mesh = {k: trimesh.load(v)
                       for k, v in self.__files.items()}
        self.__keys = sorted(list(self.__mesh.keys()))
        print(self.__keys)
        self.__metadata = {k: {} for k in self.__keys}

        self.__radius = {}
        self.__volume = {}
        self.__masses = {}
        for k, m in self.__mesh.items():
            self.__radius[k] = float(
                0.5 *
                np.linalg.norm(m.vertices,
                               axis=-1).max())
            self.__volume[k] = m.volume
            # TODO: consider randomization
            self.__masses[k] = cfg.density * m.volume

        table_dims = np.asarray(cfg.table_dims,
                                dtype=np.float32)
        self.__poses = {}
        for k, v in tqdm(self.__mesh.items(), desc='pose'):
            # FIXME: height=table_dims[2] assumes
            # table is on the ground.
            # TODO: should _probably_ use a different function;
            # mostly since trimesh implementation is super slow.
            path = ensure_directory(cfg.cache_dir)
            if (path/f'{k}.npy').is_file():
                poses = np.load(path/f'{k}.npy')
                print(f"load cached poses from {path}/{k}.npy")
            else:
                poses = sample_stable_poses(v.convex_hull,
                                            table_dims[2],
                                            cfg.num_poses)
                np.save(path/f'{k}.npy', poses)
            self.__poses[k] = poses.astype(np.float32)

        # NOTE: unnecessarily computationally costly maybe
        self.__cloud = {}
        self.__normal = {}
        self.__bbox = {}
        self.__aabb = {}
        self.__obb = {}
        for k, v in self.__mesh.items():
            samples, face_index = trimesh.sample.sample_surface(
                v, cfg.num_points)
            self.__cloud[k] = samples
            self.__normal[k] = v.face_normals[face_index]
            self.__aabb[k] = v.bounds
            self.__bbox[k] = trimesh.bounds.corners(v.bounds)
            obb = v.bounding_box_oriented
            self.__obb[k] = (
                np.asarray(obb.transform, dtype=np.float32),
                np.asarray(obb.extents, dtype=np.float32))

        # Unfortunately, no guarantee of deletion
        self.__tmpdir = mkdtemp()
        self.__write_urdf()

    def __write_urdf(self):
        cfg = self.cfg
        self.__urdf = {}
        for k in self.__keys:
            m = self.__masses[k]
            I = self.__mesh[k].moment_inertia
            vis_mesh_file = self.__files[k]

            aux = {}
            col_mesh_file = F'{self.__tmpdir}/{k}.obj'
            if True:
                shutil.copy(vis_mesh_file, col_mesh_file)
                aux['num_part'] = 1
            else:
                apply_coacd(cfg.coacd,
                            vis_mesh_file,
                            col_mesh_file, aux=aux)
            ic(vis_mesh_file, col_mesh_file)
            self.__metadata[k]['num_chulls'] = (
                aux['num_part']
            )

            params = dict(
                mass=m,
                ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2],
                ixx=m * I[0, 0], iyy=m * I[1, 1], izz=m * I[2, 2],
                vis_mesh=vis_mesh_file,
                col_mesh=col_mesh_file
            )
            filename = F'{self.__tmpdir}/{k}.urdf'
            with open(filename, 'w') as fp:
                fp.write(URDF_TEMPLATE.format(**params))
            self.__urdf[k] = filename

    def keys(self):
        return self.__keys

    def label(self, key: str) -> str:
        return key

    def urdf(self, key: str):
        return self.__urdf[key]

    def pose(self, key: str):
        return self.__poses[key]

    def code(self, key: str):
        # return self.codes[key]
        return None

    def cloud(self, key: str):
        return self.__cloud[key]

    def normal(self, key: str):
        return self.__normal[key]

    def bbox(self, key: str):
        return self.__bbox[key]

    def aabb(self, key: str):
        return self.__aabb[key]

    def obb(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.__obb[key]

    def hull(self, key: str) -> trimesh.Trimesh:
        return self.__mesh[key]

    def radius(self, key: str) -> float:
        return self.__radius[key]

    def volume(self, key: str) -> float:
        return self.__volume[key]

    def num_verts(self, key: str) -> float:
        return len(self.__mesh[key].vertices)

    def num_faces(self, key: str) -> float:
        return len(self.__mesh[key].faces)

    def num_hulls(self, key: str) -> float:
        return self.__metadata[key]['num_chulls']


def main():
    # _convert_from_previous_version()
    dataset = MeshObjectSet(
        MeshObjectSet.Config(
            num_poses=1,
            filename='/tmp/docker/bd_debug5/textured_mesh.obj'
        ))
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()
