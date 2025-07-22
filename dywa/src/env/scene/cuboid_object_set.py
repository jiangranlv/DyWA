#!/usr/bin/env python3

from typing import Optional
import math
import numpy as np
from env.scene.object_set import ObjectSet
from util.config import ConfigBase
import os
from dataclasses import dataclass
from typing import Tuple
import trimesh
from tempfile import mkdtemp
from util.torch_util import dcn
from util.math_util import quat_rotate
import torch as th
from env.scene.util import rejection_sample
from util.math_util import random_rotation_matrix


def random_directional_scale(n: int):
    R = random_rotation_matrix(n)
    s = th.randn(size=(n, 3), *args, **kwds)
    S = th.diag(S)
    return R @ S @ R.T


def shuffle_along_axis(a: np.ndarray, axis: int):
    """ from https://stackoverflow.com/a/55317373 """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


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


def box_inertia(dims: np.ndarray) -> np.ndarray:
    sq = np.square(dims)
    return (1.0 / 12.0) * np.diag(sq.sum() - sq)


def cuboid_orientations():
    IRT2 = math.sqrt(1.0 / 2)
    return np.asarray([
        # Identity
        [0.000, 0.000, 0.000, 1.000],
        # Reverse
        [1.000, 0.000, 0.000, 0.000],
        # Below two are basically identical
        # [0.000, 1.000, 0.000, 0.000],
        # [0.000, 0.000, 1.000, 0.000],
        [-IRT2, 0.000, 0.000, +IRT2],
        [+IRT2, 0.000, 0.000, +IRT2],
        [0.000, -IRT2, 0.000, +IRT2],
        [0.000, +IRT2, 0.000, +IRT2],
        # [0.000, 0.000, -IRT2, +IRT2],
        # [0.000, 0.000, +IRT2, +IRT2],
    ], dtype=np.float32)


def sample_cuboid_poses(extents: np.ndarray,
                        table_dims: np.ndarray,
                        n: int,
                        thin_only: bool=False):
    orientations = cuboid_orientations()
    size = tuple(extents.shape[:-2]) + (n,)
    if not thin_only:
        # WARNING: This migth confuse the policy 
        # that all object is upward
        idx = np.random.choice(len(orientations), size=size)
    else:
        # Identity only
        idx = np.zeros(size, dtype=np.int32)
    orns = orientations[idx]
    # Get Rotated extents
    ext2 = np.abs(dcn(quat_rotate(th.from_numpy(orns).float(),
                                  th.from_numpy(extents[None]).float())))
    # XY random
    pos = np.random.uniform(-0.5 * table_dims, +0.5 * table_dims,
                            size=size + (3,))
    # match Z with table height
    pos[..., 2] = table_dims[..., 2] + 0.5 * ext2[..., 2]
    return np.concatenate([pos, orns], axis=-1)


class CuboidObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        num_types: int = 512
        min_dim: float = 0.05
        max_dim: float = 0.15
        max_aspect: Optional[float] = 2.5
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
            extents = np.random.uniform(cfg.min_dim, cfg.max_dim,
                                        size=(cfg.num_types, 3))
        elif cfg.thin_only:
            xy = np.random.uniform(cfg.min_dim, cfg.max_dim,
                                        size=(cfg.num_types, 2))
            z = np.random.uniform(cfg.thin_min_height, cfg.thin_max_height,
                                        size=(cfg.num_types, 1))
            extents = np.concatenate([xy, z], axis = -1)
            print(extents)
        else:
            def _sample(n: int):
                return np.random.uniform(cfg.min_dim, cfg.max_dim,
                                         size=(n, 3))

            def _accept(x: np.ndarray):
                return x.max(axis=-1) < cfg.max_aspect * x.min(axis=-1)

            extents = rejection_sample(cfg.num_types, _sample, _accept)

        # extents = np.random.choice([cfg.min_dim, cfg.max_dim],
        #                             size=(cfg.num_types, 3))
        self.__keys = [F'x{i[0]:.4f}:y{i[1]:.4f}:z{i[2]:.4f}' for i in extents]
        self.__extents = {k: v for k, v in zip(self.__keys, extents)}
        self.__volume = {k: np.prod(v) for k, v in self.__extents.items()}
        # FIXME: consider randomization
        self.__masses = {k: cfg.density * v for k, v in self.__volume.items()}
        self.__radius = {
            k: 0.5 * np.linalg.norm(v) for k,
            v in self.__extents.items()}
        self.__mesh = {k: trimesh.creation.box(v)
                       for k, v in self.__extents.items()}
        table_dims = np.asarray(cfg.table_dims, dtype=np.float32)

        self.__poses = {}
        for k, v in self.__extents.items():
            self.__poses[k] = sample_cuboid_poses(
                v, table_dims, cfg.num_poses, cfg.thin_only)

        # NOTE: unnecessarily computationally costly maybe
        self.__cloud = {}
        self.__normal = {}
        self.__bbox = {}
        self.__aabb = {}
        for k, v in self.__mesh.items():
            samples, face_index = trimesh.sample.sample_surface(
                v, cfg.num_points)
            self.__cloud[k] = samples
            self.__normal[k] = v.face_normals[face_index]
            self.__aabb[k] = v.bounds
            self.__bbox[k] = trimesh.bounds.corners(v.bounds)

        # Unfortunately, no guarantee of deletion
        self.__tmpdir = mkdtemp()
        self.__write_urdf()

    def __write_urdf(self):
        self.__urdf = {}
        for k in self.__keys:
            m = self.__masses[k]
            x = self.__extents[k]
            I = box_inertia(x)
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
        return 'cuboid'

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
        return np.eye(4), self.__extents[key]

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
    dataset = CuboidObjectSet(
        CuboidObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()