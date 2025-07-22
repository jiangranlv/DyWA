#!/usr/bin/env python3

from typing import Dict, Tuple
import trimesh
import numpy as np


def normalize_mesh(mesh: trimesh.Trimesh,
                   isotropic: bool = True,
                   radius: float = 1.0
                   ) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """ Normalize mesh (in-place) to fit within bounding-box.

    Args:
        mesh: Input geometry.
        isotropic: Whether to scale all axes equally.

    Returns:
        mesh: Input geometry.
        xfm:  The transform that's been applied to the mesh.
    """
    p0, p1 = mesh.bounds
    center = 0.5 * (p0 + p1)
    scale = (0.5 / radius) * (p1 - p0)
    if isotropic:
        scale = scale.max()

    # scale transform
    iscale = 1.0 / scale
    S = np.diag([*np.multiply(np.ones(3), iscale), 1]
                ).astype(np.float32)
    # centering transform
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -center
    xfm = S @ T
    mesh.apply_transform(xfm)
    return mesh, xfm


class NormalizeMesh:
    """
    Normalize mesh.
    """

    def __init__(self, isotropic: bool = True):
        self.isotropic = isotropic

    def __call__(
            self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray]:
        return normalize_mesh(mesh,
                              self.isotropic)
