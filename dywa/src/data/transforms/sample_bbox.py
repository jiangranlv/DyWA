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

import tqdm
import pickle

from util.path import ensure_directory
from data.transforms.io_xfm import scene_to_mesh

def compute_bbox_from_mesh(
        filenames: str,
        out_dir: Optional[str] = None):
    robot = URDF.load(filenames)

    mesh = scene_to_mesh(robot.scene)
    try:
        bbox= mesh.bounding_box 
    except:
        print(filenames)
        return None
    name= filenames.name.replace('.urdf','')
    # np.save(out_dir+'/'+name, np.asarray(bbox.vertices))
    return np.asarray(bbox.vertices)

if __name__ == '__main__':
    filenames = sorted(
        list(Path('/input/ACRONYM/urdf/').glob('*.urdf')))
    out_dir = ensure_directory('/input/ACRONYM/bbox')

    # with mp.Pool(64) as pool:
    #     export = partial(sample_bbox_from_mesh, out_dir=str(out_dir))
    #     for _ in tqdm.tqdm(pool.imap_unordered(export, filenames),
    #                   total=len(filenames)):
    #         pass
    bboxes={}
    for filename in tqdm.tqdm(filenames):
        vertices = sample_bbox_from_mesh(filename)
        if vertices is not None:
            bboxes[str(filename)] = vertices

    with open("/input/ACRONYM/urdf/bbox.pkl","wb") as fw:
        pickle.dump(bboxes, fw)