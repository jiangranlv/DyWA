#!/usr/bin/env python3

from typing import List, Tuple
from pathlib import Path
import json
import numpy as np
import torch as th
import re
import gdown
import pickle


def get_gdrive_file_id(url: str) -> str:
    """
    Get unique file identifier from Gdrive share URL.
    FIXME: super fragile
    """
    regex = r"https://drive\.google\.com/file/d/([-\w]+)"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def load_meta_file(filename: str, url: str, binary: bool = True,
                   load_fn=pickle.load):
    if not Path(filename).is_file():
        # Try to fix URL if applicable.
        file_id = get_gdrive_file_id(url)
        if file_id is not None:
            url = F'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    flag = 'rb' if binary else 'r'
    with open(filename, flag) as fp:
        out = load_fn(fp)
    return out


def load_objects(
        data_root: str,
        hull_root: str,
        urdf_stats: str,

        single_object: bool,
        max_vertex_count: int,
        max_chull_count: int,
        num_object_types: int
) -> Tuple[List[str], List[str]]:
    if single_object:
        hull_files = list(Path(hull_root).rglob(
            'Speaker_64058330533509d1d747b49524a1246e_0.003949258269301651.glb'
        ))
        object_files = [
            Path(data_root) / f.with_suffix('.urdf').name
            for f in hull_files]
    else:
        with open(urdf_stats, 'r') as fp:
            urdf_stats = json.load(fp)

        # FILTER OBJECTS BY STATS
        object_files = []
        hull_files = []
        for urdf_file, stat in urdf_stats.items():
            if stat['num_verts'] >= max_vertex_count:
                continue
            if stat['num_chulls'] >= max_chull_count:
                continue
            hull_file = (
                Path(hull_root)
                / Path(urdf_file).with_suffix('.glb').name
            )
            if not hull_file.is_file():
                continue
            object_files.append(urdf_file)
            hull_files.append(hull_file)
        indices = np.random.choice(
            len(hull_files),
            size=num_object_types,
            replace=num_object_types > len(hull_files)
        )
        object_files = [object_files[i] for i in indices]
        hull_files = [hull_files[i] for i in indices]
    return (object_files, hull_files)


@th.jit.script
def zs_from_hulls_and_Rs(
        vertices: List[th.Tensor],
        Rs: th.Tensor,
        n: int):
    # zs = Rs @ (0, 0, 1) # Nx3x3
    zs = Rs[..., :, 2]
    out = th.zeros((n,),
                   dtype=Rs.dtype,
                   device=Rs.device)
    for i in range(n):
        out[i] = (vertices[i] @ zs[i]).min()
    return out
