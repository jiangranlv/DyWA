#!/usr/bin/env python3

from typing import Optional, Iterable, Tuple, Union
import os
import sys
import ctypes
import subprocess
import numpy as np
from pathlib import Path
import tempfile

import trimesh
import open3d as o3d
import pymeshlab
from data.transforms.io_xfm import load_mesh
import coacd as coacd_lib


def to_ext(filename: str, out_dir: str, ext: str,
           force: str = 'mesh',
           cat: bool = True) -> str:
    """ Convert input to target extension via trimesh. """
    out_path = Path(out_dir)
    filepath = Path(filename)

    # Load mesh, and potentially concatenate.
    mesh = trimesh.load(filename,
                        force='mesh',
                        skip_texture=True,
                        skip_materials=True)
    if cat and isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if isinstance(mesh, list) and len(mesh) == 0:
        return None

    # Dump as output format.
    out_file = str(out_path / filepath.with_suffix(ext).name)
    mesh.export(out_file)
    return out_file


class RedirectStream(object):
    """ from:
    https://github.com/bulletphysics/bullet3/issues/3131#issuecomment-719124592
    """
    @staticmethod
    def _flush_c_stream(stream):
        streamname = stream.name[1:-1]
        libc = ctypes.CDLL(None)
        libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        # ensures python stream is unaffected.
        self.stream.flush()
        self.fd = open(self.file, "w+")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, type, value, traceback):
        # ensures C stream buffer empty
        RedirectStream._flush_c_stream(self.stream)
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()


def vhacd_pb(in_file: str, out_file: str):
    """ V-HACD. """
    import pybullet as pb
    log_file = str(Path(out_file).with_suffix('.log'))
    with RedirectStream(sys.stdout):
        pb.vhacd(str(in_file), str(out_file), log_file)
    return log_file


vhacd = vhacd_pb


def coacd(in_file: str, out_file: str,
          extra_args: Optional[Iterable[str]] = None
          ) -> Tuple[bool, Union[str, bytes]]:
    """ CoACD """
    out_path = Path(out_file)
    log_file = out_path.parent / (out_path.stem + '_log.txt')
    wrl_file = out_path.with_suffix('.wrl')
    try:
        args = ['./main',
                '-i', in_file,
                '-o', out_file,
                # '-np'
                ]
        if extra_args is not None:
            args.extend(extra_args)
        # FIXME: hardcoded directory
        result = subprocess.run(
            args, check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd='/home/user/CoACD/build').stderr
    except subprocess.CalledProcessError as e:
        return log_file, wrl_file, False, F'{e}'
    return log_file, wrl_file, True, result


def watertight_manifold(
        filename: str,
        root: Union[str, os.PathLike, Path]
) -> Tuple[str, bool, Union[str, bytes]]:
    """ convert to watertight mesh with `manifold`. """
    out_file = str(Path(root) / Path(filename).with_suffix('.obj').name)
    try:
        result = subprocess.run(
            ['./manifold',
             filename,
             out_file,
             '-s'], check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd='/home/user/Manifold/build').stderr
    except subprocess.CalledProcessError as e:
        return out_file, False, F'{e}'
    return out_file, True, result


def watertight_manifold_plus(
        filename: str,
        root: Union[str, os.PathLike, Path]
) -> Tuple[str, bool, Union[str, bytes]]:
    """ convert to watertight mesh with `manifoldplus`. """
    out_file = str(Path(root) / Path(filename).with_suffix('.obj').name)
    try:
        result = subprocess.run(
            ['./manifold',
                '--input', filename,
                '--output', out_file,
                '--depth', '9'], check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd='/home/user/ManifoldPlus/build/').stderr
    except subprocess.CalledProcessError as e:
        return out_file, False, F'{e}'
    return out_file, True, result


# By default, we use simplify_mlx.
watertight = watertight_manifold_plus


def simplify_manifold(filename: str,
                      # root: Union[str, Path, os.PathLike],
                      out_file: str,
                      ) -> Tuple[str, bytes]:
    """ convert to simplified mesh. """
    result = subprocess.run(
        ['/home/user/Manifold/build/simplify',
         '-i',
         filename,
         '-o',
         out_file,
         '-m',
         '-r',
         '0.02'], check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd='/home/user/Manifold/build'
    ).stderr

    return out_file, result


def simplify_o3d(filename: str,
                 root: Union[str, Path, os.PathLike],
                 num_tri: int = 16384,
                 t: bool = False) -> str:
    """ Simplify (quadric decimation) with Open3D backend. """
    out_file = str(Path(root) / Path(filename).with_suffix('.obj').name)
    mesh = o3d.io.read_triangle_mesh(str(filename))
    if t:
        if not isinstance(mesh, o3d.t.geometry.TriangleMesh):
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).cpu()
        target_reduction: float = 1.0 - min(
            num_tri / len(mesh.triangle['indices']), 1.0)
        dst_mesh = mesh.simplify_quadric_decimation(
            target_reduction=target_reduction)
        dst_mesh = dst_mesh.to_legacy()
    else:
        dst_mesh = mesh.simplify_quadric_decimation(num_tri)
    o3d.io.write_triangle_mesh(out_file, dst_mesh)
    return out_file


def simplify_mlx(
        filename: str,
        root: Union[str, Path, os.PathLike],
        num_tri: int = 4096) -> str:
    """ Simplify (quadric decimation) with (Py)MeshLab backend. """
    out_file = str(Path(root) / Path(filename).with_suffix('.obj').name)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(filename))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=num_tri,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=False,
        autoclean=True)
    ms.save_current_mesh(out_file)
    return out_file


def coacd_v2(f_in: str, f_out: str, simplify: bool = True,
             file_type: str = 'obj'):
    mesh = load_mesh(f_in, as_mesh=True, file_type=file_type)

    imesh = coacd_lib.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces
    parts = coacd_lib.run_coacd(
        imesh,
        threshold=0.05,  # max concavity
        max_convex_hull=16,
        resolution=2048,
        mcts_max_depth=5,
        mcts_iterations=1024,
        mcts_nodes=32
    )  # a list of convex hulls.k

    mesh_parts = [
        trimesh.Trimesh(
            np.array(
                p.vertices), np.array(
                p.indices).reshape(
                    (-1, 3))) for p in parts]
    if simplify:
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = []
            for i, p in enumerate(mesh_parts):
                # export part
                dst_obj = F'{tmpdir}/p{i:03d}-pre.obj'
                p.export(dst_obj)

                # simplify part with meshlab
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(dst_obj)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=max(32, 512 // len(mesh_parts)),
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


# By default, we use simplify_mlx.
simplify = simplify_mlx
