#!/usr/bin/env python3

from typing import Dict, Any, Union, Optional
import logging
import numpy as np
import trimesh
import open3d as o3d

# from pytorch3d.ops import sample_points_from_meshes as sample_th3
# def sample_surface_points_from_mesh_th3():
#     pass


def sample_surface_points_from_mesh(
        mesh: trimesh.Trimesh,
        count: int,
        use_poisson: bool,
        use_even: bool,
        normal: bool = False,
        oversample_ratio: float = 4.0,
        radius:Optional[float]=None
        ) -> np.ndarray:

    if (use_poisson and
            (not isinstance(mesh, o3d.geometry.TriangleMesh))):
        logging.warn(F'poisson sampling infeasible for type {type(mesh)}')

    if isinstance(mesh, o3d.geometry.TriangleMesh):
        if use_poisson:
            points = np.asanyarray(
                mesh.sample_points_poisson_disk(count).points,
                dtype=np.float32
            )
        else:
            points = np.asanyarray(
                mesh.sample_points_uniformly(count).points,
                dtype=np.float32
            )
    else:
        if use_even:
            points, face_id = trimesh.sample.sample_surface_even(
                mesh, int(oversample_ratio * count),radius=radius)
        else:
            points, face_id = trimesh.sample.sample_surface(mesh, count)
        # NOTE: sometimes, sample_surface_even
        # does not exactly return the requested number
        # of points. So, we ensure that the
        # number of sampled points matches
        # the output specification.
        if len(points) != count:
            replace: bool = (len(points) < count)
            indices = np.random.choice(
                len(points), count, replace=replace)
            points = points[indices]
            face_id = face_id[indices]

    if normal:
        normals = mesh.face_normals[face_id]
    else:
        normals = None
    return points, normals


class SampleSurfacePointsFromMesh:
    """
    Sample a set of points from a trimesh.
    """

    def __init__(self,
                 count: int,
                 use_poisson: bool = False,
                 use_even: bool = False,
                 key: str = 'sampled_points'):
        self.count = count
        self.use_poisson = use_poisson
        self.use_even = use_even
        self.key = key

    def __call__(self, inputs: Dict[str, Any]):
        mesh: Union[
            trimesh.Trimesh,
            o3d.geometry.TriangleMesh
        ] = inputs['mesh']
        points, _ = sample_surface_points_from_mesh(mesh,
                                                 self.count,
                                                 self.use_poisson,
                                                 self.use_even)
        outputs = dict(inputs)
        outputs[self.key] = np.asarray(points, dtype=np.float32)
        return outputs
