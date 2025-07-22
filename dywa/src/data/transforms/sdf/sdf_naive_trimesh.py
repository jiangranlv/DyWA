#!/usr/bin/env python3

import numpy as np
import trimesh
import open3d as o3d
from typing import Dict, Any, Optional

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
                 sdf_key: str = 'sdf'):
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
        outputs = dict(inputs)

        # mesh = trimesh.Trimesh(inputs['pos'], inputs['faces'])
        mesh: trimesh.Trimesh = inputs['mesh']

        if isinstance(mesh, trimesh.Trimesh):
            radius = mesh.bounding_sphere.primitive.radius
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            bbox = mesh.get_axis_aligned_bounding_box()
            radius = np.linalg.norm(bbox.get_half_extent())

        if self.points_key not in inputs:
            # First, sample points on the surface.
            outputs = self.sample_points(inputs)
            # Then add noise to the surface points.
            noise_scale = self.relative_noise_scale * radius
            points = self.rng.normal(loc=outputs[self.points_key],
                                     scale=noise_scale)
            outputs[self.points_key] = points
        else:
            outputs = dict(inputs)

        query = trimesh.proximity.ProximityQuery(mesh)
        sdf = query.signed_distance(outputs[self.points_key])
        outputs[self.sdf_key] = sdf
        return outputs
