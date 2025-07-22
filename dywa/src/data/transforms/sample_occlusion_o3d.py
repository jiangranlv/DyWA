#!/usr/bin/env python3

from typing import Dict, Any, Optional
import numpy as np
import open3d as o3d
import logging


class SampleOcclusion:
    """Sample non-occluded (visible) points, with Open3D backend.

    NOTE: hidden_point_removal seems to be only approximately correct!
    """
    DEFAULT_KEY_MAP = {k: k for k in (
        ('points', 'viewpoint', 'visible_points'))}

    def __init__(self,
                 key_map: Optional[Dict[str, str]] = None,
                 rng: Optional[np.random.Generator] = None):
        logging.warn(
            'SampleOcclusion() module is only approximate.'
            'Try to use more principled alternatives instead.')
        self.key_map = dict(SampleOcclusion.DEFAULT_KEY_MAP)
        if key_map is not None:
            self.key_map.update(key_map)

        self.key_points = self.key_map['points']
        self.key_viewpoint = self.key_map['viewpoint']
        self.key_visible_points = self.key_map['visible_points']
        if rng is None:
            rng = np.random.default_rng(0)
        self.rng = rng

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        points = inputs[self.key_points]
        viewpoint = inputs.get(self.key_viewpoint, None)
        points = np.asanyarray(points, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if viewpoint is None:
            # FIXME: somewhat arbitrary sampling scheme...
            bmin = pcd.get_min_bound()
            bmax = pcd.get_max_bound()
            center = 0.5 * (bmin.ravel() + bmax.ravel())
            rad = 0.5 * np.linalg.norm(bmax.ravel() - bmin.ravel())
            viewpoint = self.rng.normal(size=3)
            viewpoint *= 2.0 * rad / np.linalg.norm(viewpoint)
            viewpoint += center

        true_rad = np.linalg.norm(points - viewpoint, axis=-1).max()
        pcd, _ = pcd.hidden_point_removal(viewpoint, true_rad * 1.5)
        out_points = np.asarray(pcd.vertices, dtype=np.float64)

        outputs = dict(inputs)
        outputs[self.key_viewpoint] = viewpoint
        outputs[self.key_visible_points] = out_points
        return outputs
