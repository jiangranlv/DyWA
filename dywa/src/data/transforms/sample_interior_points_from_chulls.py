#!/usr/bin/env python3

from typing import Iterable

import numpy as np

from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet


def _sample_simplices(deln: np.ndarray, n: int, dims: int,
                      weight: bool = True):
    """
    Args:
        deln: Simplices; (N,4,3)
        n: Number of points to sample.
        dims: Data dimensionality (usually 3).

    Return:
        points sampled from simplices, by volume.
    """
    if weight:
        weights = np.abs(np.linalg.det(deln[:, :dims, :] - deln[:, dims:, :]))
        sample = np.random.choice(len(weights), size=n,
                                  p=weights / weights.sum())
    else:
        sample = np.random.choice(len(deln), size=n)
    return np.einsum('ijk, ij -> ik', deln[sample],
                     dirichlet.rvs([1] * (dims + 1), size=n))


def sample_points_from_hulls(
        hulls: Iterable[np.ndarray],
        n: int, is_hull: bool = False,
        weight: bool = True):
    """
    Sample interior points from convex hulls.

    Args:
        hulls: convex hulls, or point clouds.
        n: number of points to sample.
        is_hull: Whether `hulls` is already convex hulls.

    Returns:
        sampled points.

    NOTE:
        Adapted from https://stackoverflow.com/a/59086818.
    """
    # return dist_in_hulls(hulls, n)
    if not is_hull:
        hulls = [p[ConvexHull(p).vertices] for p in hulls]
    dims = hulls[0].shape[-1]
    deln = np.concatenate(
        [h[Delaunay(h).simplices] for h in hulls],
        axis=0)
    return _sample_simplices(deln, n, dims, weight=weight)


def sample_points_from_hull(points: np.ndarray, n: int,
                            is_hull: bool = False,
                            weight: bool = True):
    """  """
    return sample_points_from_hulls([points], n, is_hull,
                                    weight=weight)


class SampleInteriorPointsFromChulls:
    """
    Sample interior points from convex-hulls.
    Functor wrapper around `sample_points_from_hulls`.
    """

    def __init__(self, n: int,
                 is_hull: bool = True,
                 weight: bool = True):
        self.n = n
        self.is_hull = is_hull
        self.weight = weight

    def __call__(self, hulls: Iterable[np.ndarray]) -> np.ndarray:
        return sample_points_from_hulls(
            hulls, self.n, self.is_hull, weight=self.weight)
