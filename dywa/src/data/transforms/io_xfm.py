#!/usr/bin/env python3

from typing import Tuple, Union
import numpy as np
import trimesh


def yield_file(in_file: str):
    """from pytorch_geometric.io."""
    with open(in_file) as fp:
        buf = fp.read()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split()[1:]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
        else:
            yield ['', ""]


def read_obj(in_file):
    """Read text - based .obj file.

    NOTE: taken from pytorch_geometric.io.
    """
    vertices = []
    faces = []
    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)
    if not len(faces) or not len(vertices):
        return None
    return dict(pos=vertices, face=faces)


def scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    if len(scene.graph.nodes_geometry) == 1:
        # Take cheaper option if possible.
        node_name = scene.graph.nodes_geometry[0]
        (transform, geometry_name) = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not (transform == np.eye(4)).all():
            mesh.apply_transform(transform)
    else:
        # Default = dump
        mesh = scene.dump(concatenate=True)
    return mesh


def load_mesh(mesh_file: str,
              as_mesh: bool = False,
              **kwds) -> Union[trimesh.Trimesh,
                               Tuple[np.ndarray, np.ndarray]]:
    # [1] Load Mesh.
    mesh = trimesh.load(mesh_file,
                        # force='mesh',
                        skip_texture=True,
                        skip_materials=True,
                        **kwds)

    # [2] Ensure single geometry.
    if isinstance(mesh, trimesh.Scene):
        mesh = scene_to_mesh(mesh)
    if as_mesh:
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return (verts, faces)


class LoadMesh:
    def __init__(self, *args, **kwds):
        self.__args = args
        self.__kwds = kwds

    def __call__(self, mesh_file: str) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: consider also returning normals, uv maps and texture.
        return load_mesh(mesh_file, *self.__args, **self.__kwds)
