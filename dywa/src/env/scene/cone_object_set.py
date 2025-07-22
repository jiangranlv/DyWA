#!/usr/bin/env python3

import math
import numpy as np
from env.scene.object_set import ObjectSet
from typing import Optional
from util.config import ConfigBase
from tqdm.auto import tqdm
import os
from dataclasses import dataclass
from typing import Tuple
import trimesh
from tempfile import mkdtemp
from util.torch_util import dcn
from util.math_util import quat_rotate
from env.scene.util import sample_stable_poses
import torch as th
from cho_util.math import transform as tx
from env.scene.util import rejection_sample

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
                <mesh filename="{filename}" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{filename}" scale="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


class ConeObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        num_types: int = 512

        min_dim: float = 0.05
        max_dim: float = 0.15
        max_aspect: Optional[float] = 2.5
        min_side: int = 3
        max_side: int = 6
        density: float = 300.0
        table_dims: Tuple[float, float, float] = (0.4, 0.5, 0.4)
        num_poses: int = 256
        num_points: int = 512

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.max_aspect is None:
            cone_radius = np.random.uniform(
                0.5 * cfg.min_dim, 0.5 * cfg.max_dim,
                size=(cfg.num_types,))
            cone_height = np.random.uniform(
                cfg.min_dim, cfg.max_dim,
                size=(cfg.num_types,))
        else:
            def _sample(n: int):
                return np.random.uniform(cfg.min_dim, cfg.max_dim,
                                         size=(n, 2))

            def _accept(x: np.ndarray):
                return x.max(axis=-1) < (cfg.max_aspect * x.min(axis=-1))
            x = rejection_sample(cfg.num_types, _sample, _accept,
                                 verbose=True)
            cone_radius = 0.5 * x[..., 0]
            cone_height = x[..., 1]

        cone_sides = np.random.randint(
            cfg.min_side, cfg.max_side + 1,
            size=(cfg.num_types))
        cone_config = {
            F'Cs{s:1d}:r{r:.4f}:h{h:.4f}': (r, h, s)
            for (r, h, s) in zip(
                cone_radius,
                cone_height,
                cone_sides)}
        self.__keys = sorted(list(cone_config.keys()))
        self.cone_config = cone_config
        self.__mesh = {k: trimesh.creation.cone(r, h, s)
                       for k, (r, h, s) in cone_config.items()}
        self.__radius = {}
        self.__volume = {}
        self.__masses = {}
        for k, m in self.__mesh.items():
            self.__radius[k] = float(
                0.5 *
                np.linalg.norm(
                    m.vertices,
                    axis=-
                    1).max())
            self.__volume[k] = m.volume
            # TODO: consider randomization
            self.__masses[k] = cfg.density * m.volume

        table_dims = np.asarray(cfg.table_dims, dtype=np.float32)
        self.__poses = {}
        for k, v in tqdm(cone_config.items(), desc='pose'):
            # FIXME: height=table_dims[2] assumes
            # table is on the ground.
            origin = self.__mesh[k].triangles_center
            normal = self.__mesh[k].face_normals
            poses = []
            for o, n in zip(origin, normal):
                xfm4x4 = trimesh.geometry.plane_transform(o, -n)
                quat = tx.rotation.quaternion.from_matrix(
                    tx.rotation_from_matrix(xfm4x4))
                xyz = xfm4x4[:3, 3] + (0, 0, table_dims[2])
                pose = np.concatenate([xyz, quat], axis=-1)
                poses.append(pose)
            self.__poses[k] = np.stack(poses, axis=0).astype(np.float32)

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
        self.__urdf = {}
        for k in self.__keys:
            m = self.__masses[k]
            I = self.__mesh[k].moment_inertia
            mesh_file = F'{self.__tmpdir}/{k}.obj'
            self.__mesh[k].export(mesh_file)
            params = dict(
                mass=m,
                ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2],
                ixx=m * I[0, 0], iyy=m * I[1, 1], izz=m * I[2, 2],
                filename=mesh_file
            )
            filename = F'{self.__tmpdir}/{k}.urdf'
            with open(filename, 'w') as fp:
                fp.write(URDF_TEMPLATE.format(**params))
            self.__urdf[k] = filename

    def keys(self):
        return self.__keys

    def label(self, key: str) -> str:
        """ Category of this object """
        _, _, side = self.cone_config[key]
        return F'cone{side}'

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
        return 1


def main():
    # _convert_from_previous_version()
    dataset = ConeObjectSet(
        ConeObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()
