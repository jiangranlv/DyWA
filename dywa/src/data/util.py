#!/usr/bin/env python3

import numpy as np
from os import PathLike
from typing import Iterable, Optional, Union, Tuple
import itertools
import numpy as np
import torch as th
from pathlib import Path
from zipfile import ZipFile
import trimesh
import io
import open3d as o3d


def split_files(files: Iterable[Union[str, PathLike]],
                rng: np.random.Generator,
                fraction: float):
    files = list(files)
    n = int(len(files) * fraction)
    rng.shuffle(files)
    return (files[:n], files[n:])


def bbox_nd(img: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """ From https://stackoverflow.com/a/31402351 """
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.append(np.where(nonzero)[0][[0, -1]])
    return tuple(out)


def sample_occlusion(points: np.ndarray, pos: Optional[np.ndarray] = None,
                     rng: Optional[np.random.Generator] = None):
    points = np.asanyarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Camera pos (as offset)
    if pos is None:
        if rng is None:
            rng = np.random.default_rng(0)
        bmin = pcd.get_min_bound()
        bmax = pcd.get_max_bound()
        center = 0.5 * (bmin.ravel() + bmax.ravel())
        rad = 0.5 * np.linalg.norm(bmax.ravel() - bmin.ravel())
        pos = rng.normal(size=3)
        pos *= 2.0 * rad / np.linalg.norm(pos)
        pos += center

    true_rad = np.linalg.norm(points - pos, axis=-1).max()
    pcd, _ = pcd.hidden_point_removal(pos, true_rad)
    return np.asarray(pcd.vertices, dtype=np.float64)
