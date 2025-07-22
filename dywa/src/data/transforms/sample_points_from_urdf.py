#!/usr/bin/env python3

from typing import Dict, Any, Union, Optional
import logging
import numpy as np
import trimesh
import open3d as o3d
from pathlib import Path
from yourdfpy import URDF
import multiprocessing as mp
from functools import partial
import pickle

import tqdm

from util.path import ensure_directory
from data.transforms.io_xfm import scene_to_mesh
from data.transforms.sample_points import sample_surface_points_from_mesh

_OVERSAMPLE_RATIO: float = 4


def sample_surface_points_from_urdf(
        filename: str,
        count: int,
        use_poisson: bool,
        use_even: bool,
        cloud_dir: str,

        normal: bool = False,
        normal_dir: Optional[str] = None,
        **kwds):
    export: bool = kwds.pop('export', False)
    cat: bool = kwds.pop('cat', True)

    robot = URDF.load(filename)
    mesh = scene_to_mesh(robot.scene)  # .as_open3d

    if not isinstance(mesh, (trimesh.Trimesh, o3d.geometry.TriangleMesh)):
        if isinstance(mesh, list) and len(mesh) <= 0:
            logging.warn(F'mesh={mesh} not found for urdf={filename}')
            return
        raise ValueError(
            F'invalid mesh = {type(mesh)}, {mesh} for urdf={filename}')
    points, normals = sample_surface_points_from_mesh(
        mesh, count, use_poisson, use_even,
        normal=normal,
        **kwds)

    points = np.asarray(points, dtype=np.float32)

    if normal:
        normals = np.asarray(normals, dtype=np.float32)

    if normal and cat:
        points = np.concatenate([points, normals], -1)

    if export:
        # if `split`, separately save points and normals.
        split: bool = (normal and (not cat))

        basename = filename.name.replace('.urdf', '')
        if split:
            assert (normal_dir is not None)
            np.save(cloud_dir + '/' + basename, points)
            np.save(normal_dir + '/' + basename, normals)
        else:
            np.save(cloud_dir + '/' + basename, points)

    return points


class SampleSurfacePointsFromUrdf:
    """
    Sample a set of points from a trimesh.
    """

    def __init__(self,
                 count: int,
                 cloud_dir: str,
                 use_poisson: bool = False,
                 use_even: bool = False,
                 key: str = 'sampled_points',):
        self.count = count
        self.use_poisson = use_poisson
        self.use_even = use_even
        self.key = key

    def __call__(self, inputs: str):
        # mesh: Union[
        #     trimesh.Trimesh,
        #     o3d.geometry.TriangleMesh
        # ] = inputs['mesh']
        robot = URDF.load(inputs)
        mesh = robot.scene.convex_hull
        points = sample_surface_points_from_urdf(mesh,
                                                 self.count,
                                                 self.use_poisson,
                                                 self.use_even)
        outputs = dict(inputs)
        outputs[self.key] = np.asarray(points, dtype=np.float32)
        return outputs


def sample_cloud():
    filenames = sorted(
        list(Path('/input/ACRONYM/urdf/').glob('*.urdf')))
    cloud_dir = ensure_directory('/input/ACRONYM/cloudnormal/')

    with mp.Pool(64) as pool:
        export = partial(
            sample_surface_points_from_urdf,
            count=512,
            use_poisson=False,
            use_even=False,
            normal=True,
            radius=1e-3,
            cloud_dir=str(cloud_dir))
        for _ in tqdm.tqdm(pool.imap_unordered(export, filenames),
                           total=len(filenames)):
            pass


def merge_cloud():
    filenames = list(Path('/input/ACRONYM/cloudnormal/').glob('*.npy'))
    clouds = {f.stem: np.load(str(f)) for f in filenames}
    with open('/input/ACRONYM/cloudnormal.pkl', 'wb') as fp:
        pickle.dump(clouds, fp)


def show_cloud():
    with open('/input/ACRONYM/cloudnormal_4096.pkl', 'rb') as fp:
        clouds = pickle.load(fp)

    for name, cloud in clouds.items():
        trimesh.PointCloud(cloud).show()
        break


def main():
    sample_cloud()
    merge_cloud()
    # show_cloud()


if __name__ == '__main__':
    main()
