#!/usr/bin/env python3

"""
Signed distance transform on meshes which are sets of convex hulls.
"""

from typing import (
    List, Tuple, Optional, Dict,
    Iterable, Callable)

from pathlib import Path
import numpy as np
import torch as th
from scipy.spatial import ConvexHull
import trimesh

from data.transforms.df_th3 import DistanceTransform
from data.transforms.io_xfm import scene_to_mesh, load_mesh
from data.transforms.sample_points import SampleSurfacePointsFromMesh
from util.torch_util import dcn


def is_in_hull(hull: ConvexHull, points: np.ndarray,
               tol: float = 1e-8):
    return np.all((hull.equations[:, :-1] @ points.T
                   + hull.equations[:, -1:]) <= tol, axis=0)


def is_in_hulls(hulls: List[ConvexHull], points: np.ndarray):
    return all(is_in_hull(h, points) for h in hulls)


class LoadConvexHulls:
    def __init__(self):
        pass

    def __call__(self, inputs):
        # Parse inputs.
        cvx_file: str = inputs['cvx_file']
        mesh_file: str = inputs.get('mesh_file', cvx_file)
        transform: np.ndarray = inputs.get('xfm', None)

        # Load.
        scene: trimesh.Scene = trimesh.load(cvx_file,
                                            split_object=True,
                                            group_material=False,
                                            skip_texture=True,
                                            skip_materials=True)
        if transform is not None:
            scene.apply_transform(transform)

        # Convert.
        if not isinstance(scene, trimesh.Trimesh):
            # scene
            # mesh = scene_to_mesh(scene)
            mesh = load_mesh(mesh_file, True)
            hulls = [v for (k, v) in scene.geometry.items()]
        else:
            mesh = scene
            hulls = [mesh]

        # Output.
        return {**inputs,
                'mesh': mesh,
                'hulls': hulls}


class IsInHulls:
    def __init__(self, hulls: Iterable[trimesh.Trimesh],
                 device: str = 'cuda:0'):
        self.hulls = hulls
        ns, cs = zip(*[(h.face_normals, h.triangles_center)
                       for h in hulls])
        self.normals = ns
        self.offsets = [np.einsum('fd,fd->f', n, c)
                        for (n, c) in zip(ns, cs)]

        self.normals = [th.as_tensor(x,
                                     dtype=th.float32,
                                     device=device)
                        for x in self.normals]
        self.offsets = [th.as_tensor(x,
                                     dtype=th.float32,
                                     device=device)
                        for x in self.offsets]

    def __call__(self, points: np.ndarray):
        # is_in = np.zeros(points.shape[:-1], dtype=bool)
        is_in = th.zeros(points.shape[:-1], dtype=bool,
                         device=points.device)
        # TODO: consider masking and/or
        # multi-stage variants for faster queries.
        for n, d in zip(self.normals, self.offsets):
            # d_p = np.einsum('fd,qd->fq', n, points)
            # d_p = th.einsum('fd,qd->fq', n, points)
            d_p = th.einsum('fd,...qd->...fq', n, points)
            # print(d.shape) # 200
            # print(d_p.shape) # 32, 200, 2048
            # print(is_in.shape) # 32, 2048
            is_in |= (d[None, :, None] >= d_p).all(axis=1)
        return is_in


class SampleNearSurface:
    def __init__(self,
                 count: int,
                 noise_scale: float,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = 0,
                 relative: bool = True):
        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng
        self.noise_scale = noise_scale
        self.sample = SampleSurfacePointsFromMesh(count)
        self.relative = relative

    def __call__(self, inputs):
        points = self.sample(inputs)['sampled_points']
        if self.relative:
            # print(inputs.keys())
            mesh = inputs['mesh']
            obj_scale = np.linalg.norm(
                mesh.bounds[1] - mesh.bounds[0])
            # obj_scale = np.linalg.norm(
            #     np.ptp(inputs['verts'], axis=0))
            scale = self.noise_scale * obj_scale
        else:
            scale = self.noise_scale
        points = self.rng.normal(loc=points,
                                 scale=scale)
        return {**inputs,
                'query': points}


# class SampleQueryPoints:
#     """
#     Produce somewhat balanced positive-negative
#     query dataset, based on rejection-sampling.
#     """

#     def __init__(self,
#                  proposal: Callable,
#                  num_points: int,
#                  pos_fraction: float,
#                  max_iter: int):
#         assert(0 <= pos_fraction and pos_fraction <= 1)
#         self.proposal = proposal
#         self.pos_fraction = pos_fraction

#     def __call__(self, inputs):
#         scene = inputs['scene']
#         if not isinstance(scene, trimesh.Trimesh):
#             # scene
#             mesh = scene_to_mesh(scene)
#             hulls = scene.geometry
#         else:
#             mesh = scene
#             hulls = {'convex_0': mesh}

#         for _ in range(self.max_iter):
#             # q_out[replace] = queries
#             # o_out[replace] = occs
#             queries = self.proposal(inputs)
#             occs = self.is_occupied(queries)


class SignedDistanceTransform:
    """
    SDF for geometries which are unions of convex hulls.
    """

    def __init__(self,
                 key_map: Optional[Dict[str, str]] = None,
                 device: str = 'cuda'):
        self.df = DistanceTransform(key_map)
        self.device: th.device = th.device(device)

    def __call__(self, inputs):
        """
        Args:
            inputs: dictionary of
                mesh: mesh.
                hulls: hulls.
                query: Optional query coordinates.
            outputs: dictionary of
                sdf: sdf distance values.
        """

        # [0] Load.
        mesh: trimesh.Trimesh = inputs.get('mesh')
        hulls: List[trimesh.Trimesh] = inputs.get('hulls')
        query: np.ndarray = inputs.get('query')

        # [2] Compute (unsigned) distance to nearest surface.
        with th.no_grad():
            df_inputs = {'verts': th.as_tensor(mesh.vertices,
                                               dtype=th.float32,
                                               device=self.device)[None],
                         'faces': th.as_tensor(mesh.faces,
                                               dtype=th.float32,
                                               device=self.device)[None],
                         'query': th.as_tensor(query,
                                               dtype=th.float32,
                                               device=self.device)[None]}
            df_outputs = self.df(df_inputs)

        # [3] Compute the containment flag,
        # as a union of convex hulls.
        is_in = IsInHulls(hulls)(query)
        sign = is_in * 2.0 - 1.0
        out = {
            **inputs,
            'sdf': dcn(d) * sign
        }
        return out


def main():
    sdf_fn = SignedDistanceTransform()
    objs = list(
        Path('/opt/datasets/ShapeNetSem/vhacd/vhacd/').glob('*.obj'))
    np.random.shuffle(objs)
    for obj in objs:
        sdfs = sdf_fn({
            'mesh_file': obj,
        })
        print(sdfs)
        break


if __name__ == '__main__':
    main()
