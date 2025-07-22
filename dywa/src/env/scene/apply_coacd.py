#!/usr/bin/env python3

import sys
import os
import trimesh
import coacd
import numpy as np
import pymeshlab
from pathlib import Path
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass

from data.transforms.io_xfm import load_mesh
from util.path import ensure_directory


@dataclass
class Config:
    # COCAD
    max_concavity: float = 0.05
    max_convex_hull: int = 16
    preprocess: bool = False
    resolution: int = 2048
    mcts_max_depth: int = 4
    mcts_iterations: int = 512
    mcts_nodes: int = 32

    # MeshLab
    simplify: bool = True
    max_face_count: int = 512
    min_face_count: int = 32
    num_proc: int = 16
    verbose: bool = False


def apply_coacd(cfg: Config, f_in: str,
                f_out: Optional[str] = None,
                d_out: Optional[str] = None,
                aux: Optional[Dict[str, Any]] = None):
    # if not cfg.verbose:
    #    sys.stdout = open(os.devnull, 'w')
    #    sys.stderr = open(os.devnull, 'w')

    # f_out = str(ensure_directory(d_out) / Path(f_in).name)
    p_in = Path(f_in)

    if f_out is None:
        assert (d_out is not None)
        f_out = str(ensure_directory(d_out) /
                    p_in.parent.parent.with_suffix(p_in.suffix).name)

    if Path(f_out).is_file():
        return

    # mesh = trimesh.load(f_in, file_type='obj')
    mesh = load_mesh(f_in, as_mesh=True,
                     file_type='obj')

    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces
    parts = coacd.run_coacd(
        imesh,
        threshold=cfg.max_concavity,  # max concavity
        max_convex_hull=cfg.max_convex_hull,
        resolution=cfg.resolution,
        mcts_max_depth=cfg.mcts_max_depth,
        mcts_iterations=cfg.mcts_iterations,
        mcts_nodes=cfg.mcts_nodes,
        preprocess=cfg.preprocess
    )  # a list of convex hulls.k

    mesh_parts = [
        trimesh.Trimesh(
            np.array(
                p.vertices), np.array(
                p.indices).reshape(
                    (-1, 3))) for p in parts]

    if aux is not None:
        aux['num_part'] = len(mesh_parts)

    if cfg.simplify:
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = []
            for i, p in enumerate(mesh_parts):
                # Export each part.
                dst_obj = F'{tmpdir}/p{i:03d}-pre.obj'
                p.export(dst_obj)

                # simplify part with meshlab
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(dst_obj)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=max(cfg.min_face_count,
                                      cfg.max_face_count // len(mesh_parts)),
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True)
                out_file = F'{tmpdir}/p{i:03d}-post.obj'
                ms.save_current_mesh(out_file)
                sources.append(out_file)

            scene = trimesh.Scene()
            for f in sources:
                p = trimesh.load(f, file_type='obj')
                scene.add_geometry(p)
            scene.export(f_out)

    else:
        scene = trimesh.Scene()
        for p in mesh_parts:
            scene.add_geometry(p)
        scene.export(f_out)
