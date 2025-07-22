#!/usr/bin/env python3

from typing import Dict, Any, Union
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


def get_volume_from_mesh(
        filename: str,
        out_dir: str
        ):
    robot = URDF.load(filename)
    mesh = scene_to_mesh(robot.scene)
    try:
        hull = mesh.convex_hull
    except:
        print(f'{filename} has no hull')
        return

    if isinstance(mesh,  trimesh.Trimesh):
        volume = mesh.volume
        hull_volume = hull.volume
        name = filename.name.replace('.urdf', '')
        volumes = np.stack([volume, hull_volume], 0)
        np.save(out_dir + '/' + name, volumes)
    else:
       print(f'{filename} has ',type(mesh))
    

def get_volumes():
    filenames = sorted(
            list(Path('/input/ACRONYM/urdf/').glob('*.urdf')))
    out_dir = ensure_directory('/input/ACRONYM/volume/')

    with mp.Pool(64) as pool:
        export = partial(
            get_volume_from_mesh,
            out_dir=str(out_dir))
        for _ in tqdm.tqdm(pool.imap_unordered(export, filenames),
                           total=len(filenames)):
            pass


def merge_volume():
    filenames = list(Path('/input/ACRONYM/volume/').glob('*.npy'))
    volumes = {f.stem: np.load(str(f)) for f in filenames}
    with open('/input/ACRONYM/volume.pkl', 'wb') as fp:
        pickle.dump(volumes, fp)


def main():
    # get_volumes()
    merge_volume()
    # show_cloud()


if __name__ == '__main__':
    main()
