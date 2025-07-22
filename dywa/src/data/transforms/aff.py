#!/usr/bin/env/python3

from typing import Optional, Tuple
import pickle
from pathlib import Path
from yourdfpy import URDF
import torch as th
from env.scene.dgn_object_set import DGNObjectSet
# import pytorch_volumetric as pv
from tempfile import TemporaryDirectory
from tqdm.auto import tqdm
from util.path import ensure_directory
from models.common import merge_shapes
from util.math_util import (
    apply_pose_tq,
    invert_pose_tq,
    compose_pose_tq,
    quat_rotate,
    matrix_from_pose
)
from util.torch_util import dcn
import copy
import trimesh
from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import einops

from data.transforms.sdf.sdf_cvx_set_th3 import (
    SignedDistanceTransform,
    IsInHulls
)
from data.transforms.df_th3 import DistanceTransform
from util.path import get_path


def get_gripper_mesh(cat: bool = False,
                     frame: str = 'panda_tool',
                     urdf_path: Optional[str] = None,
                     links: Tuple[str, ...] = ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')
                     ):
    """ as chulls though """
    if urdf_path is None:
        urdf_path = get_path('assets/franka_description/robots/franka_panda.urdf')
        print('urdf-path', urdf_path)
    urdf = URDF.load(urdf_path,
                     build_collision_scene_graph=True,
                     load_collision_meshes=True,
                     force_collision_mesh=False)

    hulls = []
    for link in ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']:
        xfm = urdf.get_transform(link, frame,
                                 collision_geometry=True)
        loc = urdf.link_map[link].collisions[0].origin
        if loc is not None:
            xfm = xfm @ loc

        hull_file = urdf.link_map[link].collisions[0].geometry.mesh.filename
        scene: trimesh.Scene = trimesh.load(
            Path(urdf_path).parent / hull_file,
            split_object=True,
            group_material=False,
            skip_texture=True,
            skip_materials=True,
            force='scene')
        scene.apply_transform(xfm)

        for node_name in scene.graph.nodes_geometry:
            (transform, geometry_name) = scene.graph[node_name]
            mesh = scene.geometry[geometry_name]
            mesh.apply_transform(transform)
            hulls.append(mesh)
    out = hulls

    # if True:
    #     axis=trimesh.creation.axis()
    #     xxx = trimesh.util.concatenate(hulls + [axis])
    #     xxx.show()

    if cat:
        out = trimesh.util.concatenate(hulls)
    return out


class CheckGripperCollisionV2():
    def __init__(self, device: Optional[str] = None,
                 frame: str = 'panda_tool'):
        self.device = device
        hulls = get_gripper_mesh(cat=False,
                                 frame=frame)
        urdf_path = get_path('assets/franka_description/robots/franka_panda.urdf')
        print('urdf-path', urdf_path)
        urdf = URDF.load(urdf_path,
                         build_collision_scene_graph=True,
                         load_collision_meshes=True,
                         force_collision_mesh=False)

        self.df = DistanceTransform(
            key_map=dict(verts='verts',
                         faces='faces',
                         query='query',
                         distance='distance',
                         grad='grad'))

        mesh = urdf.collision_scene.subscene(
            'panda_hand').dump(concatenate=True)
        xfm = urdf.get_transform('panda_hand', frame,
                                 collision_geometry=True)
        mesh.apply_transform(xfm)

        self.mesh = mesh
        self.hull_mesh = trimesh.util.concatenate(hulls)

        self.df_inputs = {'verts': th.as_tensor(mesh.vertices,
                                                dtype=th.float32,
                                                device=self.device)[None],
                          'faces': th.as_tensor(mesh.faces,
                                                dtype=th.float32,
                                                device=self.device)[None]}
        self.is_in = IsInHulls(hulls, device=device)

    def __call__(self, x: th.Tensor):
        df_inputs = {k: v.expand(x.shape[0], *v.shape[1:]) for
                     (k, v) in self.df_inputs.items()}
        df_inputs['query'] = x
        df_outputs = self.df(df_inputs)
        is_in = self.is_in(x)
        sign = -(is_in * 2.0 - 1.0)

        d = df_outputs['distance'].reshape(x.shape[:-1]) * sign
        g = df_outputs['grad'].reshape(x.shape) * sign[..., None]
        return (d, g)
