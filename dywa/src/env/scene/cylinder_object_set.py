#!/usr/bin/env python3

from typing import Optional
import math
import numpy as np
from env.scene.object_set import ObjectSet
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


def cylinder_inertia(radius: float, height: float):
    r, h = radius, height
    ixx = iyy = (1.0 / 12) * (3 * r * r + h * h)
    izz = 0.5 * r * r
    return np.diag([ixx, iyy, izz])


class CylinderObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        num_types: int = 512

        min_dim: float = 0.05
        max_dim: float = 0.15
        max_aspect: Optional[float] = 2.5
        num_side: int = 32
        density: float = 300.0
        table_dims: Tuple[float, float, float] = (0.4, 0.5, 0.4)
        num_poses: int = 256
        num_points: int = 512
        thin_only: bool = False
        thin_min_height: float = 0.005
        thin_max_height: float = 0.01

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.max_aspect is None:
            cylinder_radius = np.random.uniform(
                0.5 * cfg.min_dim, 0.5 * cfg.max_dim,
                size=(cfg.num_types,))
            cylinder_height = np.random.uniform(
                cfg.min_dim, cfg.max_dim,
                size=(cfg.num_types,))
        elif cfg.thin_only:
            cylinder_radius = np.random.uniform(
                0.5 * cfg.min_dim, 0.5 * cfg.max_dim,
                size=(cfg.num_types,))
            cylinder_height = np.random.uniform(cfg.thin_min_height,
                                                cfg.thin_max_height,
                                                size=(cfg.num_types,))
        else:
            def _sample(n: int):
                return np.random.uniform(cfg.min_dim, cfg.max_dim,
                                         size=(n, 2))

            def _accept(x: np.ndarray):
                return x.max(axis=-1) < cfg.max_aspect * x.min(axis=-1)

            x = rejection_sample(cfg.num_types, _sample, _accept)
            cylinder_radius = 0.5 * x[..., 0]
            cylinder_height = x[..., 1]

        cylinder_config = {
            F'r{r:.4f}:h{h:.4f}': (r, h)
            for (r, h) in zip(
                cylinder_radius,
                cylinder_height)}
        self.__keys = sorted(list(cylinder_config.keys()))
        self.cylinder_config = cylinder_config
        self.__mesh = {k: trimesh.creation.cylinder(r, h, cfg.num_side)
                       for k, (r, h) in cylinder_config.items()}
        self.__radius = {}
        self.__volume = {}
        self.__masses = {}
        for k in cylinder_config.keys():
            (r, h) = cylinder_config[k]
            self.__radius[k] = np.linalg.norm([r, 0.5 * h])
            self.__volume[k] = (np.pi * r * r * h)
            # TODO: consider randomization
            self.__masses[k] = cfg.density * self.__volume[k]

        table_dims = np.asarray(cfg.table_dims, dtype=np.float32)
        self.__poses = {}
        for k, v in tqdm(cylinder_config.items(), desc='pose'):
            # FIXME: height=table_dims[2] assumes
            # table is on the ground.
            origin = self.__mesh[k].triangles_center
            normal = self.__mesh[k].face_normals
            poses = []
            for o, n in zip(origin, normal):
                xfm4x4 = trimesh.geometry.plane_transform(o, -n)
                if cfg.thin_only:
                    # WARNING: This migth confuse the policy 
                    # that all object is upward
                    if n[-1] != 1:
                        continue
                    xfm4x4[:3, :3] = np.eye(3, dtype=xfm4x4.dtype)
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
            r, h = self.cylinder_config[k]
            m = self.__masses[k]
            I = cylinder_inertia(r, h) * m
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
        _, _, side = self.cylinder_config[key]
        return F'cylinder{side}'

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
    from yourdfpy import URDF
    dataset = CylinderObjectSet(
        CylinderObjectSet.Config(num_types=16,
                                 num_side=16))

    for key in dataset.keys():
        URDF.load(dataset.urdf(key)).show()

    # for attr in dir(dataset):
    #     print(attr)
    #     if hasattr(dataset, attr):
    #         (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()
