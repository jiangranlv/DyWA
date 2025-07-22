#!/usr/bin/env python3

from typing import Optional, Dict, List, Any

import numpy as np

import trimesh

from data.transforms.sdf.sdf_cvx_set_th3 import (
    SampleNearSurface
)
from data.transforms.sample_interior_points_from_chulls import (
    SampleInteriorPointsFromChulls)


class SamplePointsFromChulls:
    """
    Sample a set of points from a trimesh.
    """

    def __init__(self,
                 count: int,
                 min_pos_fraction: float,
                 use_poission: bool = False,
                 use_even: bool = False,

                 noise_scale: float = 0.1,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = 0,
                 relative: bool = True,
                 weight: bool = True,

                 key: str = 'query'):
        self.count = count
        self.use_poission = use_poission
        self.use_even = use_even
        self.key = key

        self.n_pos = int(count * min_pos_fraction)
        self.n_near = count - self.n_pos

        self.sample_hull = SampleInteriorPointsFromChulls(
            self.n_pos, is_hull=True, weight=weight)
        self.sample_near = SampleNearSurface(
            self.n_near, noise_scale, rng, seed, relative)

    def __call__(self, inputs: Dict[str, Any]):
        # mesh: trimesh.Trimesh = inputs.get('mesh')
        hulls: List[trimesh.Trimesh] = inputs.get('hulls')

        hvs = [h.vertices for h in hulls]
        points_hull = self.sample_hull(hvs)
        points_near = self.sample_near(inputs)['query']
        points = np.concatenate([points_hull, points_near],
                                axis=0)
        outputs = dict(inputs)
        outputs[self.key] = np.asarray(points,
                                       dtype=np.float64)
        return outputs
