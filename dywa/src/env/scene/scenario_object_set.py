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
import glob
from xml.dom import minidom

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

class ScenarioObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        data_path: Optional[str] = None  # '/input/Scenario'
        cloud_size: int = 512
        num_repetition: int = 5
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ic(cfg)
        if cfg.data_path is not None:
            self.__metadata = {}
            print('loading metadata from {}'.format(cfg.data_path))
            jsons = glob.glob(F'{cfg.data_path}/*.json')
            for js in jsons:
                with open(js, 'r') as fp:
                    data = json.load(fp)
                    for key, val in data.items():
                        self.__metadata[key] = val
            print('loaded metadata')
        else:
            raise ValueError(f'{cfg.data_path} cannot be None')
        print('process dataset')
        self.update_requirements()
        print('process done')

    def update_requirements(self):
        self.__tmpdir = mkdtemp()

        self.hulls = {}
        self.normals = {}
        self.clouds = {}
        self.predefined_goals = {}
        self.poses = {}
        self.num_scenarios = {}

        for k in self.__metadata.keys():
            
            # 1) reconfigure urdf
            urdf_file = self.__metadata[k]['urdf']
            scales = self.__metadata[k]['scales']
            mass = self.__metadata[k]['mass']
            with open (urdf_file, 'r', encoding='utf-8') as f:
                str_urdf = f.read()
            dom = minidom.parseString(str_urdf)
            dom.getElementsByTagName("mass")[0].attributes['value'] = str(mass)
            meshes = dom.getElementsByTagName("mesh")
            if len(scales) == 1:
                scales = scales * 3
            mesh_scales = meshes[-1].attributes['scale'].value.split(' ')
            # WARNING: Scales are defined in relative manner
            new_scale = [scale * float(mesh_scale) 
                            for scale, mesh_scale in zip(scales, mesh_scales)]
            old_scale = float(meshes[-1].attributes['scale'].value.split(' ')[-1]) 
            scale_ratio = new_scale[-1] / old_scale 
            print(k)
            print(old_scale, new_scale[-1], scale_ratio)
            for mesh in meshes:
                mesh.attributes['scale'].value = \
                    ' '.join([str(scale) for scale in new_scale])
            new_urdf_path = f'{self.__tmpdir}/{k}.urdf'
            self.__metadata[k]['urdf'] = new_urdf_path
            with open(new_urdf_path, "w") as f:
                dom.writexml(f)
            print(dom.toprettyxml())
            urdf = URDF.load(new_urdf_path)
            scene = urdf.scene

            # 2) extract all required attributes
            stat = stat_from_mesh(scene)
            for prop, value in stat.items():
                self.__metadata[k][prop] = value
            self.__metadata[k]['num_hulls'] = 1

            single = scene.dump(concatenate=True)
            hull = single.convex_hull
            self.hulls[k] = hull

            cloud, face_index = trimesh.sample.sample_surface(single, 
                                                            self.cfg.cloud_size)
            self.clouds[k] = cloud 
            normal = single.face_normals[face_index]
            self.normals[k] = normal

            # 3) Get poses
            if self.__metadata[k]['pose']['mode'] == 'stable':
                poses = load_npy(self.__metadata[k]['pose']['path'])
                # NOTE: Poses are only scaled in z direction
                poses[...,2] -= 0.4
                poses[...,2] *= scale_ratio
                poses[...,2] += 0.4
                self.poses[k] = poses
                self.num_scenarios[k] = 30
                # stable_poses = load_npy(self.__metadata[k]['pose']['path'])
                # inits = []
                # goals = []
            elif self.__metadata[k]['pose']['mode'] == 'sampled':
                inits = []
                goals = []
                for data in self.__metadata[k]['pose']['value']:
                    inits.append(data['init_pose'])
                    goals.append(data['goal_pose'])
                inits = np.stack(inits, 0)
                goals = np.stack(goals, 0)
                self.poses[k] = inits
                self.predefined_goals[k] = goals
                self.num_scenarios[k] = len(goals) \
                      * self.cfg.num_repetition

    def keys(self):
        return self.__metadata.keys()

    # @cached_property
    # def poses(self):
    #     return {k: load_npy(F'{self.cfg.pose_path}/{k}.npy')
    #             for k in self.keys()}

    @cached_property
    def codes(self):
        # FIXME: temporary exception
        if Path(self.cfg.code_path).is_file():
            return load_pkl(self.cfg.code_path)
        return {k: load_npy(F'{self.cfg.code_path}/{k}.npy')
                for k in self.keys()}

    # @cached_property
    # def clouds(self):
    #     return {k: load_npy(F'{self.cfg.cloud_path}/{k}.npy')
    #             for k in self.keys()}

    # @cached_property
    # def normals(self):
    #     return {k: load_npy(F'{self.cfg.normal_path}/{k}.npy')
    #             for k in self.keys()}

    # @cached_property
    # def hulls(self) -> Dict[str, trimesh.Trimesh]:
    #     return {k: load_glb(F'{self.cfg.hull_path}/{k}.glb')
    #             for k in self.keys()}

    def label(self, key: str) -> str:
        """ Category of this object """
        return key

    def urdf(self, key: str):
        cfg = self.cfg
        return self.__metadata[key]['urdf']

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

    def predefined_goal(self, key: str):
        return self.predefined_goals[key]

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

def test_dataset():
    mock = {
        'sem-Rock-f7b2c1368b2c6f9a8b43e675a8d9c006-0.080':{
            "urdf": "/input/DGN/meta-v8/urdf/sem-Rock-f7b2c1368b2c6f9a8b43e675a8d9c006-0.080.urdf",
            "mass": 0.250,
            "scales": [0.13],
            "pose": {
                "mode": "stable",
                "path": "/input/DGN/meta-v8/pose/sem-Rock-f7b2c1368b2c6f9a8b43e675a8d9c006-0.080.npy"
            }
        },
        'core-jar-739f97be019f29f2da5969872c705114-0.060':{
            "urdf": "/input/DGN/meta-v8/urdf/core-jar-739f97be019f29f2da5969872c705114-0.060.urdf",
            "mass": 0.200,
            "scales": [0.03],
            "pose": {
                "mode": "stable",
                "path": "/input/DGN/meta-v8/pose/core-jar-739f97be019f29f2da5969872c705114-0.060.npy"
            }
        },
        'core-mug-64a9d9f6774973ebc598d38a6a69ad2-0.080':{
            "urdf": "/input/DGN/meta-v8/urdf/core-mug-64a9d9f6774973ebc598d38a6a69ad2-0.080.urdf",
            "mass": 0.250,
            "scales": [0.2],
            "pose": {
                "mode": "stable",
                "path": "/input/DGN/meta-v8/pose/core-mug-64a9d9f6774973ebc598d38a6a69ad2-0.080.npy"
            }
        },'sem-ToasterOven-3c357d1352d2d1811fddae104d1cd00e-0.060':{
            "urdf": "/input/DGN/meta-v8/urdf/sem-ToasterOven-3c357d1352d2d1811fddae104d1cd00e-0.060.urdf",
            "mass": 0.150,
            "scales": [0.16],
            "pose": {
                "mode": "stable",
                "path": "/input/DGN/meta-v8/pose/sem-ToasterOven-3c357d1352d2d1811fddae104d1cd00e-0.060.npy"
            }
        },
        'mujoco-Dog-0.060':{
            "urdf": "/input/DGN/meta-v8/urdf/mujoco-Dog-0.060.urdf",
            "mass": 0.450,
            "scales": [0.18],
            "pose": {
                "mode": "stable",
                "path": "/input/DGN/meta-v8/pose/mujoco-Dog-0.060.npy"
            }
        }
    }
    with open("/input/mock/metadata.json", "w") as fp:
        json.dump(mock, fp)

    cfg = ScenarioObjectSet.Config(
        data_path = '/input/mock'
    )

    mock_dataset = ScenarioObjectSet(cfg)
    for attr in dir(mock_dataset):
        print(attr)
        if hasattr(mock_dataset, attr):
            (getattr(mock_dataset, attr))

def main():
    test_dataset()


if __name__ == '__main__':
    main()
