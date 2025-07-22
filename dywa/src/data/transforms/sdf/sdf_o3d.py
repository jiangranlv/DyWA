#!/usr/bin/env python3

import numpy as np
import trimesh
from typing import Dict, Any, Optional
import open3d as o3d

from data.transforms.sample_points import SampleSurfacePointsFromMesh


class SignedDistanceTransform:
    """Compute signed distances for a set of query points.

    NOTE: Does not always work for non-watertight meshes.
    """

    def __init__(self,
                 num_query_points: int = 512,
                 rng: Optional[np.random.Generator] = None,
                 relative_noise_scale: Optional[float] = 0.1,
                 points_key: str = 'points',
                 sdf_key: str = 'sdf'
                 ):
        self.num_query_points = num_query_points
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.relative_noise_scale = relative_noise_scale

        self.points_key = points_key
        self.sdf_key = sdf_key
        self.sample_points = SampleSurfacePointsFromMesh(
            num_query_points, False, key=self.points_key)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Convert to Open3D.
        mesh_tri: trimesh.Trimesh = inputs['mesh']
        if isinstance(mesh_tri, trimesh.Trimesh):
            radius = mesh_tri.bounding_sphere.primitive.radius
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        elif isinstance(mesh_tri, o3d.t.geometry.TriangleMesh):
            mesh = mesh_tri
            bbox = mesh.get_axis_aligned_bounding_box()
            radius = np.linalg.norm(bbox.get_half_extent())
        elif isinstance(mesh_tri, o3d.geometry.TriangleMesh):
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_tri)
            bbox = mesh.get_axis_aligned_bounding_box()
            radius = np.linalg.norm(bbox.get_half_extent())

        # Spawn raycasting scene.
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        if self.points_key not in inputs:
            raise ValueError('this path disabled')
            # First, sample points on the surface.
            outputs = self.sample_points(inputs)
            # Then add noise to the surface points.
            noise_scale = self.relative_noise_scale * radius
            points = self.rng.normal(loc=outputs[self.points_key],
                                     scale=noise_scale)
            outputs[self.points_key] = points
        else:
            outputs = dict(inputs)

        points = o3d.core.Tensor(outputs[self.points_key],
                                 dtype=o3d.core.Dtype.Float32)

        sdf = scene.compute_signed_distance(points)
        sdf = np.asanyarray(-sdf.cpu().numpy(), dtype=np.float32)
        outputs[self.sdf_key] = sdf
        return outputs
