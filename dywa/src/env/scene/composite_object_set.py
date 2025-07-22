#!/usr/bin/env python3

import numpy as np
from dataclasses import replace
from env.scene.object_set import ObjectSet
from typing import Optional
from util.config import ConfigBase
from tqdm.auto import tqdm
import os
from dataclasses import dataclass
from typing import Tuple
import trimesh
from tempfile import mkdtemp
from env.scene.util import sample_stable_poses
from cho_util.math import transform as tx
from env.scene.util import rejection_sample

from env.scene.cylinder_object_set import CylinderObjectSet
from env.scene.cuboid_object_set import CuboidObjectSet
from env.scene.cone_object_set import ConeObjectSet
from data.transforms.io_xfm import scene_to_mesh

from env.scene.util import (
    sample_stable_poses,
    sample_stable_poses_fast,
)
from yourdfpy import URDF, Joint

from icecream import ic

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')


class CompositeObjectSet(ObjectSet):

    @dataclass
    class Config(ConeObjectSet.Config, CylinderObjectSet.Config,
                 CuboidObjectSet.Config, ConfigBase):
        min_part: int = 2
        max_part: int = 8
        num_parts_per_prim_type: int = 32
        fallback: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert (cfg.min_part > 0)
        assert (cfg.max_part > cfg.min_part)

        part_cfg = replace(self.cfg,
                           num_types=cfg.num_parts_per_prim_type)
        self.cone = ConeObjectSet(part_cfg)
        self.cylinder = CylinderObjectSet(part_cfg)
        self.cuboid = CuboidObjectSet(part_cfg)
        object_sets = [self.cone, self.cylinder, self.cuboid]
        keyss = [s.keys() for s in object_sets]

        # combine
        merged_urdfs = []
        merged_keys = []
        max_index = [len(object_sets), cfg.num_parts_per_prim_type]
        merged_num_parts = []
        for t in range(cfg.num_types):
            num_parts = np.random.randint(cfg.min_part, cfg.max_part)
            part_indices = np.random.randint(max_index, size=(num_parts, 2))

            # == determine part urdfs ==
            part_urdfs = []
            part_keys = []
            for i, j in part_indices:
                src_set = object_sets[i]
                key = keyss[i][j]
                part_urdf_file = src_set.urdf(key)
                part_urdf = URDF.load(part_urdf_file)
                part_urdfs.append(part_urdf)
                part_keys.append(key)

            # == combine part urdfs onto base urdf ==
            base_urdf = part_urdfs[0]
            for i, part_urdf in enumerate(part_urdfs[1:]):
                part_link = part_urdf.link_map[part_urdf.base_link]
                part_link.name = F'part_{i:02d}'
                base_urdf.robot.links.append(part_link)

                joint_origin = np.eye(4)
                joint_origin[:3, :3] = tx.rotation.matrix.random()
                joint_origin[:3, 3] = np.random.normal(scale=0.03, size=3)
                part_joint = Joint(
                    name=F'base_to_part_{i:02d}',
                    type='fixed',
                    parent='base_link',
                    child=part_link.name,
                    origin=joint_origin,
                    axis=[0, 0, 1])
                base_urdf.robot.joints.append(part_joint)
            new_urdf = URDF(base_urdf.robot)

            merged_urdfs.append(new_urdf)
            merged_keys.append('-'.join(part_keys[i]) + F'{t:03d}')
            merged_num_parts.append(num_parts)

        self.__radius = {}
        self.__volume = {}
        self.__masses = {}
        self.__mesh = {k: scene_to_mesh(u.scene)
                       for (k, u) in zip(merged_keys, merged_urdfs)}
        self.__num_parts = {
            k: n for (k, n) in zip(
                merged_keys,
                merged_num_parts)}
        self.__hull = {k: m.convex_hull for (k, m) in self.__mesh.items()}
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
        self.__poses = {}  # ...???

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
        self.__keys = merged_keys
        self.__write_urdf(self.__keys, merged_urdfs)

    def __write_urdf(self, keys, urdfs):
        self.__urdf = {}
        for k, urdf in zip(keys, urdfs):
            filename = F'{self.__tmpdir}/{k}.urdf'
            urdf.write_xml_file(filename)
            self.__urdf[k] = filename

    def keys(self):
        return self.__keys

    def label(self, key: str) -> str:
        """ Category of this object """
        return key

    def urdf(self, key: str):
        return self.__urdf[key]

    def pose(self, key: str):
        # Optional fallback guard
        if (((key not in self.__poses) or (self.__poses[key] is None))
                and (self.cfg.fallback) and (key in self.__mesh)):
            pose = sample_stable_poses(self.__hull[key],
                                       # FIXME: hardcoded table height
                                       0.4,
                                       16)
            self.__poses[key] = pose
        if key not in self.__poses:
            return None
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
        return self.__hull[key]

    def radius(self, key: str) -> float:
        return self.__radius[key]

    def volume(self, key: str) -> float:
        return self.__volume[key]

    def num_verts(self, key: str) -> float:
        return len(self.__mesh[key].vertices)

    def num_faces(self, key: str) -> float:
        return len(self.__mesh[key].faces)

    def num_hulls(self, key: str) -> float:
        return self.__num_parts[key]


def print_attrs():
    # _convert_from_previous_version()
    ic(CompositeObjectSet.Config())
    dataset = CompositeObjectSet(
        CompositeObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
        # print(len(dataset.codes))


def show():
    from env.scene.util import _show_stable_poses
    cfg = CompositeObjectSet.Config(num_types=16)
    ic(cfg)
    dataset = CompositeObjectSet(cfg)
    _show_stable_poses(dataset)


def main():
    show()


if __name__ == '__main__':
    main()
