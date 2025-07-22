#!/usr/bin/env python3

from typing import Dict, Any, Optional
import numpy as np
import trimesh
import open3d as o3d
from pysdf import SDF
from data.transforms.sample_points import SampleSurfacePointsFromMesh


class SignedDistanceTransform:
    """Compute signed distances for a set of query points.

    NOTE: Does not always work for non-watertight meshes.
    """

    def __init__(self,
                 num_query_points: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None,
                 relative_noise_scale: Optional[float] = 0.1,

                 points_key: str = 'points',
                 sdf_key: str = 'sdf'
                 ):
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng
        self.relative_noise_scale = relative_noise_scale
        self.points_key = points_key
        self.sdf_key = sdf_key

        self.num_query_points = num_query_points
        if num_query_points is not None:
            self.sample_points = SampleSurfacePointsFromMesh(
                num_query_points, False, key=self.points_key)
        else:
            self.sample_points = None

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Construct SDF function.
        mesh = inputs['mesh']
        if isinstance(mesh, trimesh.Trimesh):
            radius = mesh.bounding_sphere.primitive.radius
            v = np.ascontiguousarray(mesh.vertices).astype(np.float32)
            f = np.ascontiguousarray(mesh.faces).astype(np.uint32)
            sdf_fn = SDF(v, f, copy=False, robust=True)
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            v = np.ascontiguousarray(mesh.vertices).astype(np.float32)
            f = np.ascontiguousarray(mesh.triangles).astype(np.uint32)
            sdf_fn = SDF(v, f, copy=False, robust=True)
            bbox = mesh.get_axis_aligned_bounding_box()
            radius = np.linalg.norm(bbox.get_half_extent())

        if self.points_key not in inputs:
            raise ValueError('Disable this route.')
            # First, sample points on the surface.
            outputs = self.sample_points(inputs)
            # Then add noise to the surface points.
            noise_scale = self.relative_noise_scale * radius
            points = self.rng.normal(loc=outputs[self.points_key],
                                     scale=noise_scale)
            outputs[self.points_key] = points
        else:
            outputs = dict(inputs)

        # Compute SDF values.
        p = np.ascontiguousarray(outputs[self.points_key]).astype(np.float32)
        sdf = sdf_fn(p)

        # TODO: filter degenerate SDF predictions.
        # Sometimes, we get like really large SDF values,
        # which are clearly artifacts of bullshit SDF predictions.
        max_sdf = (4 * self.relative_noise_scale * radius)
        # sdf[np.abs(sdf)>max_sdf]
        # sdf[np.abs(sdf) > 10] = np.inf
        outputs[self.sdf_key] = sdf
        return outputs
